# Build Order — Writing the Vision System From Scratch to Understand It

A study plan for re-implementing this codebase yourself, in an order that builds
understanding. The guiding idea: **write bottom-up so each file only depends on
things you've already written, and reach a runnable program as early as
possible** — then enrich it layer by layer.

You're rebuilding the package `vision` (in `src/vision/`) plus the root scripts
(`app.py`, `gui_bridge.py`, `hardware_acceptance.py`) and `tests/`.

> **Scope note (SRS v0.2).** The vision system *serves frames*; it does not
> extract centroids or run image processing — that moved to the downstream
> cueing system. So there is a single "spine" here (camera → frames →
> consumers), not two. The `CueingSystem` you build is a thin frame consumer.

> Tip: after each phase, *run something*. Seeing a frame flow, a sink receive
> it, or a test go green is what turns "I copied code" into "I get it."

---

## The dependency map (what needs what)

```
camera_types ──┬─> camera_driver ──┬─> generic_driver
               │                   ├─> opencv_driver
               │                   └─> spinnaker_driver
               ├─> config_loader
               ├─> frame_buffers ──┐
               │                   ├─> camera_service  (also uses config_loader)
               │                   └─> cueing_system
               └─> gui_bridge (CameraViewer, also uses frame_buffers)
app.py  (single runner: picks a driver backend, wires service + cueing + gui_bridge)
lens  (standalone — depends on nothing)
```

One spine: the camera service *produces* frames into buffers; the cueing system
and the GUI *consume* them. Build the driver + service first, get it running with
the simulator, then add the consumers.

---

## Phase 0 — Project skeleton

**Goal:** a place to put code that's importable.

1. Create the layout:
   ```
   src/vision/__init__.py        (empty for now)
   tests/
   config/
   ```
2. Minimal `pyproject.toml` with `name`, `requires-python`, deps
   (`numpy`, `opencv-python-headless`) and a `src` package find. Flesh out the
   optional extras (`gui`, `dev`, `all`) later.
3. `python -m venv .venv && pip install -e .` so `import vision` works.

**Concept:** the `src/` layout — why the package lives under `src/` and why you
install it editable instead of running files from the repo root.

---

## Phase 1 — The data types (the vocabulary of the whole system)

### 1. `camera_types.py`
**Why first:** almost everything imports it. It defines the *nouns* the system
passes around.

**Write:**
- `CameraStatus` (enum: DISCONNECTED/CONNECTED/STREAMING/ERROR) — the driver
  state machine.
- `CameraFeature` (Flag enum) — which settings a camera supports.
- `PixelFormat` (enum).
- `CameraConfig` (dataclass) — identity + capabilities + requested settings.
  Add `__post_init__` (derive `max_pixel_count`, default resolution/fps) and a
  `validate()` that returns warnings for the NFR targets (≥60 fps, ≥512², FOV
  ≥30°).
- `CameraFrame` (dataclass) — the image container (`data`, `timestamp`,
  `frame_id`, `camera_name`, `hw_timestamp_ns`, `pixel_format`, `metadata`).

**Concepts:** dataclasses, enums/flags, separating *capabilities* from
*requested settings*. `validate()` returning warnings (not raising) is a
deliberate design choice — note why.

**Checkpoint:** in a REPL, build a `CameraConfig`, call `.validate()`, see the
warnings for an undersized config.

---

## Phase 2 — The driver abstraction + a fake camera (first runnable slice!)

### 2. `camera_driver.py`
**Why now:** defines the *contract* every camera backend must satisfy, plus the
exception hierarchy that drives the robustness behavior later.

**Write:**
- Exceptions: `CameraError` → `CameraTimeoutError`, `FeatureNotSupportedError`,
  `MalformedFrameError`. The split matters: malformed = skip a frame; other
  errors = backend fault.
- `CameraDriver(ABC)` with abstract `connect/disconnect/start_stream/
  stop_stream/read_frame/get_config/set_config`, shared `get_status`/
  `_set_status`/`_next_frame_id`, and `__enter__/__exit__`.

**Concepts:** abstract base classes, why an interface decouples the rest of the
system from any specific SDK (NFR-009), the lifecycle state machine.

### 3. `generic_driver.py`
**Why now:** a *simulated* camera lets you run and test the entire stack with no
hardware. This is the single most important file for learning — it makes
everything else observable.

**Write:** `GenericCameraDriver(CameraDriver)` that, on `start_stream`, spins up
a thread synthesizing frames with moving Gaussian "spots" on a dark background.
Implement the lifecycle, `read_frame` (blocking with timeout), and the
fault-injection hooks `inject_malformed()` / `inject_backend_crash()`.

**Concepts:** threads + events (`threading.Event`), producing frames, how
fault injection later proves NFR-005/006.

**Checkpoint (first real milestone):**
```python
drv = GenericCameraDriver(CameraConfig(name="sim", resolution=(256,256), fps=30))
drv.connect(); drv.start_stream()
f = drv.read_frame(timeout=1.0)
print(f.frame_id, f.data.shape)   # you just captured a (simulated) frame!
```

---

## Phase 3 — Config ingestion + buffers

### 4. `config_loader.py`
**Why now:** turns a JSON/YAML file into `CameraConfig` objects (FR-001).
Depends only on `camera_types`.

**Write:** `load_camera_configs(path)` (single object or `{"cameras": [...]}`),
feature-name parsing, a `ConfigError`, and tuple/enum conversion.

**Checkpoint:** write `config/camera.json`, load it, get a `CameraConfig`.

### 5. `frame_buffers.py`
**Why now:** the hand-off containers between the service and its consumers.

**Write:**
- `FrameSink` (a `Protocol` with `.push`).
- `CircularFrameBuffer` — fixed capacity, overwrites oldest (the cueing system
  wants the *freshest* frame), blocking `pop`.
- `FIFOFrameBuffer` — bounded queue, drop-*newest* on full (the GUI wants
  ordered playback without unbounded memory).

**Concepts:** the two opposite drop policies and *why each fits its consumer*;
`Condition`/`Queue` for thread-safe hand-off; duck-typed sinks via `Protocol`.
Note `pop(timeout=0)` must be non-blocking (the GUI drains with it).

**Checkpoint:** push more than capacity into each, confirm the drop counts and
ordering match your expectation.

---

## Phase 4 — The service layer (concurrency lives here)

### 6. `camera_service.py`
**Why now:** ties driver + config + buffers together. One worker thread per
camera reads frames and *fans them out* to every registered sink; it implements
the reconnect (NFR-005) and skip-malformed (NFR-006) behavior.

**Write:** `CameraService` with `add_camera`, `add_cameras_from_config`,
`attach_sink`, `connect`, `start_streaming` (launches the worker), the
`_stream_worker` loop (the SRS 5.3 state machine: read → fan out; timeout →
retry; malformed → skip+warn; fault → `_attempt_reconnect`), and `shutdown`.

**Concepts:** this is the hardest file — read the worker loop carefully. Note
how each exception type maps to a different recovery action, and how sinks are
copied under a lock before pushing.

**Checkpoint:** service + `GenericCameraDriver` + a `CircularFrameBuffer` sink →
stream for half a second → confirm frames arrive. Then call
`inject_backend_crash()` and watch it reconnect in the logs.

---

## Phase 5 — The downstream consumer + first full pipeline

### 7. `cueing_system.py`
**Why now:** the consumer side of the handoff. It drains the
`CircularFrameBuffer` on its own thread (concurrent with the camera worker,
NFR-007) and hands each frame to a pluggable `frame_processor` callback.

**Write:** `CueingSystem` worker thread + `_run`. Keep it thin: the actual cueing
pipeline (centroid extraction, tracking, state estimation, pixel→angular cue) is
**out of scope / undefined in the SRS** — leave it as the `frame_processor` hook
and don't invent a data schema for it.

**Concepts:** producer/consumer decoupling via a buffer; skip-and-continue on
processor errors (NFR-006); clean thread shutdown.

### 8. `app.py` (root script — the single runner)
**Why now:** wire everything together and **see the whole SRS v0.2 dataflow
run**. The service fans frames out to a `CircularFrameBuffer` (→ cueing) and a
`FIFOFrameBuffer` (→ GUI). `app.py` picks the backend at runtime
(`--backend {sim,opencv,spinnaker}`), runs GUI or `--headless`, and (sim only)
`--inject-faults` injects malformed frames + a backend crash to demo NFR-006/005.
PySpin is imported only when `--backend spinnaker` is chosen, so it runs with no
SDK installed.

**Checkpoint (the big one):** `python app.py --headless --inject-faults` → watch
the run summary: frames delivered, cueing consumed, malformed skipped,
reconnects. You've now built the entire core pipeline.

---

## Phase 6 — Visualization

### 9. `gui_bridge.py` (viewer module)
**Why now:** a PyQt6 `CameraViewer` that polls a `FIFOFrameBuffer` every 33 ms
and repaints — the demand-driven, never-blocks GUI (FR-004, NFR-008, SRS 5.4).
It's a driver-agnostic **module** (not run directly); `app.py` wires a backend
into it. Adding a new camera backend needs no change here.

**Checkpoint:** `python app.py` → a live window showing the simulated camera feed.

---

## Phase 7 — Real hardware backends

### 10. `opencv_driver.py`
**Why now:** your first *real* camera (any webcam) — a gentle hardware step. A
capture thread feeds a latest-frame slot; `read_frame` waits on a condition with
a timeout (OpenCV has no native blocking-with-timeout). Run it with
`python app.py --backend opencv`.

### 11. `spinnaker_driver.py`
**Why now:** the BlackFly S backend (PySpin). Lazy-imports PySpin so the rest of
the package runs without the SDK. Note the color path (BayerRG8 → BGR8 via
`ImageProcessor`), `AcquisitionMode=Continuous`, and `NewestOnly` buffering. Run
it with `python app.py --backend spinnaker --serial <S>`. Each new backend is
just another branch in `app.py`'s driver factory — no new entry-point script.

---

## Phase 8 — Optics helper (anytime — it's standalone)

### 13. `lens.py`
Independent of everything (just `math`). Implements the FLIR lens formulas:
angular FOV, linear FOV at a working distance, and focal-length selection. You
can write this whenever; it informs the `lens_fov_deg` you put in a config but
has **no runtime dependency** on the rest.

---

## Phase 9 — Tests + packaging (ideally write tests *as you go*)

### 14. `tests/`
The best way to learn is to write each test right after its module: config load,
driver lifecycle + feature gating (`test_suite.py`), the real-hardware driver
logic via fake PySpin/cv2 (`test_drivers_mocked.py`), frame-buffer invariants
(`test_properties.py`), the service malformed-skip/reconnect, the vision→cueing
handoff, the lens math, and the hardware smoke test that skips without a camera.

### 15. Finish `pyproject.toml` / `requirements.txt`
Optional-dependency extras (`gui`, `yaml`, `dev`, `all`), and confirm
`pip install -e ".[all]"` + `python -m unittest discover -s tests` is green.

---

## The mental model to keep in your head while building

- **One producer, several consumers, decoupled by buffers:** the camera worker
  *produces* frames; the cueing worker and the GUI *consume* them. Buffers absorb
  speed differences and pick a drop policy that fits each consumer.
- **Threads** (per camera): the camera worker and the cueing worker — plus the
  GUI thread. Everything shared between them goes through a thread-safe buffer.
- **Errors are data, not crashes:** the exception *type* chosen in the driver
  decides whether the service skips one frame (malformed) or tries to reconnect
  (backend fault).
- **Interfaces decouple:** the `CameraDriver` ABC means the pipeline never knows
  if it's talking to a simulator, a webcam, or a BlackFly; the cueing
  `frame_processor` hook means the (future) processing pipeline is swappable
  without touching the vision system.

Build the simulator early, keep `app.py` runnable, and add a test per module —
that's the fastest path from "typed it out" to "I understand why it's shaped
this way."
