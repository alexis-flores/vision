# Build Order — Writing the Vision System From Scratch to Understand It

A study plan for re-implementing this codebase yourself, in an order that builds
understanding. The guiding idea: **write bottom-up so each file only depends on
things you've already written, and reach a runnable program as early as
possible** — then enrich it layer by layer.

You're rebuilding the package `vision` (in `src/vision/`) plus the root scripts
(`main.py`, `gui_bridge.py`, `run_hardware.py`) and `tests/`.

> Tip: after each phase, *run something*. Seeing a frame flow, a centroid get
> extracted, or a test go green is what turns "I copied code" into "I get it."

---

## The dependency map (what needs what)

```
camera_types ──┬─> camera_driver ──┬─> generic_driver
               │                   ├─> opencv_driver
               │                   └─> spinnaker_driver
               ├─> config_loader
               ├─> frame_buffers ──┐
               └──────────────┐    │
                              camera_service  (also uses config_loader)
centroid_types ─┬─> centroid_extraction
                └─> centroid_buffer
camera_types + centroid_* + frame_buffers ─> vision_system
centroid_buffer + centroid_types ─> queuing_subsystem
camera_types + frame_buffers ─> gui_bridge (CameraViewer)
lens  (standalone — depends on nothing)
```

Two independent "spines" (camera/frame side, and centroid side) meet at
`vision_system`. Build the camera spine first, get it running with the
simulator, then build the centroid spine, then join them.

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
   (`numpy`, `opencv-python-headless`) and a `src` package find. You can flesh
   out the optional extras (`gui`, `dev`, `all`) later.
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
- `CircularFrameBuffer` — fixed capacity, overwrites oldest (vision wants the
  *freshest* frame), blocking `pop`.
- `FIFOFrameBuffer` — bounded queue, drop-*newest* on full (GUI wants ordered
  playback without unbounded memory).

**Concepts:** the two opposite drop policies and *why each fits its consumer*;
`Condition`/`Queue` for thread-safe hand-off; duck-typed sinks via `Protocol`.

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

## Phase 5 — The centroid spine (the actual vision work)

### 7. `centroid_types.py`
**Why now:** the output vocabulary. Independent of everything else.

**Write:** `Centroid` (frozen: `x, y, intensity, peak, area, bbox`) and
`CentroidProfile` (per-frame list + `seq_id`, `frame_id`, timestamps,
`proc_latency_us`). This is the **handoff unit** to downstream systems.

### 8. `centroid_extraction.py`
**Why now:** the core algorithm (FR-003). Depends only on `centroid_types`.

**Write:** `ExtractorParams` and `CentroidExtractor.extract(image) ->
list[Centroid]`: grayscale → threshold → contour/connected-components →
intensity-weighted sub-pixel centroid + area filter. Add the optional GPU path
with CPU fallback (FR-005) last.

**Concepts:** sub-pixel centroiding (center of mass), why contour tracing is
faster than full connected-components for sparse spots, the ~5 ms latency
budget (NFR-002).

**Checkpoint:** make a test image with two white circles, extract, confirm the
centroids land on the circle centers. Time 100 runs — you should be well under
5 ms.

### 9. `centroid_buffer.py`
**Why now:** the thread-safe transport (SRS 5.2, NFR-007) carrying
`CentroidProfile`s from the vision worker to the queuing worker.

**Write:** `RxTimebase` and `CentroidRingBuffer` with independent read/write
pointers, `push` (overwrites oldest unread, counts drops), blocking `pop`, and
`wake_all` for shutdown.

**Concepts:** lock + `Condition`, independent producer/consumer pointers,
overwrite-on-lap semantics.

**Checkpoint:** the concurrency test — one thread pushing N items, another
popping, assert no duplicates/deadlock.

---

## Phase 6 — Join the spines + first full pipeline

### 10. `vision_system.py`
**Why now:** this is where the two spines meet. Pulls a `CameraFrame` from a
`CircularFrameBuffer`, runs the extractor, wraps results in a `CentroidProfile`,
pushes to the `CentroidRingBuffer`, and (optionally) calls a non-blocking GUI
callback (FR-004/NFR-008). Skips bad frames instead of crashing (NFR-006);
measures and logs latency (NFR-002).

**Write:** `VisionSystem` worker thread + `_process`. Note the
`getattr(extractor, "backend", "custom")` — the extractor contract is just
`extract()`.

### 11. `queuing_subsystem.py`
**Why now:** the downstream consumer — drains the centroid ring on its own
thread and hands each profile to a sink callback.

### 12. `main.py` (root script)
**Why now:** wire everything with the simulator and **see the whole SRS
dataflow run headless**, including the injected malformed frames and backend
crash.

**Checkpoint (the big one):** `python main.py` → watch the run summary:
frames processed, centroids, reconnects, drops. You've now built the entire core
pipeline.

---

## Phase 7 — Visualization

### 13. `gui_bridge.py` (root script)
**Why now:** a PyQt6 `CameraViewer` that polls a `FIFOFrameBuffer` every 33 ms
and repaints — the demand-driven, never-blocks GUI (FR-004, SRS 5.4). Two
wirings: raw frames straight from the service, or annotated frames via the
vision `gui_callback`.

**Checkpoint:** `python gui_bridge.py` → a live window with centroid markers on
the simulated spots.

---

## Phase 8 — Real hardware backends

### 14. `opencv_driver.py`
**Why now:** your first *real* camera (any webcam) — a gentle hardware step. A
capture thread feeds a latest-frame slot; `read_frame` waits on a condition with
a timeout (OpenCV has no native blocking-with-timeout).

### 15. `spinnaker_driver.py`
**Why now:** the BlackFly S backend (PySpin). Lazy-imports PySpin so the rest of
the package runs without the SDK. Note the color path (BayerRG8 → BGR8 via
`ImageProcessor`), `AcquisitionMode=Continuous`, and `NewestOnly` buffering.

### 16. `run_hardware.py` (root script)
**Why now:** the production entry point — same pipeline as `main.py` but with
`SpinnakerCameraDriver` and the live viewer, no fault injection.

---

## Phase 9 — Optics helper (anytime — it's standalone)

### 17. `lens.py`
Independent of everything (just `math`). Implements the FLIR lens formulas:
angular FOV, linear FOV at a working distance, and focal-length selection. You
can write this whenever; it informs the `lens_fov_deg` you put in a config but
has **no runtime dependency** on the rest.

---

## Phase 10 — Tests + packaging (ideally write tests *as you go*)

### 18. `tests/test_suite.py`
The best way to learn is to write each test right after its module (config load,
driver lifecycle, extractor accuracy + latency, ring-buffer concurrency, frame
buffers, service malformed-skip/reconnect, the end-to-end vision→queuing flow,
the lens math, and the hardware smoke test that skips without a camera).

### 19. Finish `pyproject.toml` / `requirements.txt`
Optional-dependency extras (`gui`, `yaml`, `gpu`, `dev`, `all`), and confirm
`pip install -e ".[all]"` + `python -m unittest discover -s tests` is green.

---

## The mental model to keep in your head while building

- **Two producers/consumers chained by buffers:** the camera worker *produces*
  frames; the vision worker *consumes* frames and *produces* centroid profiles;
  the queuing worker *consumes* profiles. Buffers decouple their speeds.
- **Three threads** (per camera): camera worker, vision worker, queuing worker —
  plus the GUI thread. Everything shared between them goes through a thread-safe
  buffer.
- **Errors are data, not crashes:** the exception *type* chosen in the driver
  decides whether the service skips one frame or tries to reconnect.
- **Interfaces decouple:** the `CameraDriver` ABC means the pipeline never knows
  if it's talking to a simulator, a webcam, or a BlackFly; the extractor
  contract (`extract() -> list[Centroid]`) means the algorithm is swappable.

Build the simulator early, keep `main.py` runnable, and add a test per module —
that's the fastest path from "typed it out" to "I understand why it's shaped
this way."
