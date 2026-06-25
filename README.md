# Vision System — Camera Backend & Frame-Serving Layer

Implementation of the Software Requirements Specification (**SRS v0.2**) for the
real-time vision system. The vision system interfaces the camera driver SDK
behind a modular abstraction, configures the camera, establishes a real-time
stream, and **serves frames** to two consumers: the **cueing system** (via a
circular frame buffer) and the **GUI** (via a FIFO buffer + PyQt signal).

> **Scope change in SRS v0.2.** Image processing — centroid extraction, object
> labelling/tracking, kinematic state estimation, and acquisition — has moved
> **out of the vision system and into the downstream cueing system**. The vision
> system no longer extracts centroids; it serves frames. `CueingSystem` in this
> repo is a thin, dependency-free **frame consumer** that stands in for that
> downstream subsystem (its processing pipeline is out of scope and undefined —
> see `cueing_system.py`).

```
 config file (FR-001)
      │
      ▼
 CameraDriver (ABC) ──┬── SpinnakerCameraDriver (BlackFly S, PySpin)
   connect/disconnect ├── OpenCVCameraDriver (UVC/RTSP/V4L2)
   start/stop_stream  └── GenericCameraDriver (vendor template + simulator)
   read_frame(timeout)
   get/set_config · get_status
      │ driver handle
      ▼
 CameraService  ── worker thread/camera ─ FR-002, NFR-005 reconnect, NFR-006 skip
      │ frame fan-out
      ├─────────────────────────────┐
      ▼                             ▼
 CircularFrameBuffer          FIFOFrameBuffer
      │ (fresh wins)               │ (drop-newest, bounded)
      ▼                             ▼
 CueingSystem                 gui_bridge: QTimer @33ms (SRS Fig 1)
   downstream consumer            │ FR-004 / NFR-008 non-blocking
   (frames → processing           ▼
    pipeline, out of scope)    PyQt viewer (live frames)
```

## Layout

`src/` package layout. The importable library lives in `src/vision/`; the
entry-point scripts, config, and tests sit at the repo root.

```
src/vision/        # the `vision` package (importable library)
    __init__.py    # public API re-exports
    camera_*.py  *_driver.py  cueing_system.py  frame_buffers.py  lens.py  ...
app.py  gui_bridge.py  hardware_acceptance.py   # entry-point scripts (not packaged)
config/            # camera JSON configs
tests/             # test suite
pyproject.toml  README.md
```

After `pip install -e .` the library is importable as, e.g.,
`from vision import CameraService, CircularFrameBuffer, CueingSystem` (or
`from vision.spinnaker_driver import SpinnakerCameraDriver`).

## Files

(Library modules below live under `src/vision/`; scripts/config/tests at root.)

| File | Role | SRS |
|---|---|---|
| `camera_types.py` | `CameraConfig` (+ NFR validate), `CameraFrame`, status/feature enums | 5.6, NFR-001/003/004 |
| `camera_driver.py` | Abstract `CameraDriver` + exception hierarchy (incl. `MalformedFrameError`) | NFR-009 |
| `spinnaker_driver.py` | BlackFly adapter (PySpin, GenICam nodes, HW timestamp) | upstream dep |
| `opencv_driver.py` | UVC/OpenCV adapter (capture thread → blocking read w/ timeout) | NFR-009 |
| `generic_driver.py` | Vendor template + spot simulator + fault injection | NFR-009 |
| `config_loader.py` | JSON/YAML config ingestion | FR-001 |
| `frame_buffers.py` | `CircularFrameBuffer` (→ cueing), `FIFOFrameBuffer` (→ GUI) | 5.6, NFR-007/008 |
| `camera_service.py` | Multi-camera mgmt, worker threads, reconnect, frame fan-out | FR-002, NFR-005/006 |
| `cueing_system.py` | Downstream **frame consumer** stand-in (pluggable processor hook) | 5.1 dataflow |
| `acceptance.py` | Automated acceptance battery (objective PASS/FAIL checks + report) | 7 (qualification) |
| `lens.py` | Lens/FOV calculators (FLIR app-note: angular + linear FOV) | NFR-004 |
| `gui_bridge.py` | PyQt viewer **module** (`CameraViewer`), driver-agnostic; reused by `app.py` | FR-004, NFR-011, 5.4 |
| `app.py` | Single runner — any backend (sim/opencv/spinnaker), GUI or headless | — |
| `hardware_acceptance.py` | Acceptance/qualification CLI for a real camera (exit 0/1) | 7 (qualification) |
| `tests/test_suite.py` | Unit/integration/failure tests | NFR-010, 7 |
| `tests/test_drivers_mocked.py` | Spinnaker/OpenCV driver logic via fake SDKs | NFR-009, NFR-005 |
| `tests/test_properties.py` | Property-based frame-buffer invariants (Hypothesis) | NFR-010 |
| `tests/test_acceptance.py` | Acceptance check-matrix tests (pure + simulator) | NFR-010 |
| `config/camera.json` | Example BlackFly config | FR-001 |
| `config/bfs_u3_16s2c.json` | BFS-U3-16S2C-CS color config (BayerRG8 → BGR8) | FR-001 |
| `pyproject.toml` | Package metadata, dependencies + optional-dependency extras | — |

## Quick start

Because of the `src/` layout, install the package (editable) so the `vision`
package is importable by the scripts:

```bash
python -m venv .venv && source .venv/bin/activate   # recommended

pip install -e ".[all]"                             # everything (recommended)

# or pick what you need:
pip install -e .                                    # core (numpy, opencv) + the package
pip install -e ".[gui]"                             # + PyQt6 live viewer
pip install -e ".[dev]"                             # + pytest/hypothesis/ruff/mypy
```

`".[all]"` installs everything pip-installable (core + GUI + YAML + test
tooling). The only piece it can't include is **PySpin** (the Spinnaker SDK is a
vendor download, not on PyPI) — install that separately, and only if you're
driving real BlackFly hardware.

Run it:

```bash
python app.py                                # simulator + live PyQt viewer (needs [gui])
python app.py --headless                     # simulator, no GUI
python -m unittest discover -s tests         # run the test suite
```

| Extra | Installs | Enables |
|---|---|---|
| `all`   | gui + yaml + dev | everything pip-installable, one command |
| `gui`   | PyQt6  | `app.py` live viewer / `CameraViewer` (FR-004) |
| `yaml`  | PyYAML | YAML camera configs in addition to JSON |
| `dev`   | pytest, hypothesis, coverage, ruff, mypy | running the test suite + QA |

**Real BlackFly hardware:** the FLIR/Teledyne Spinnaker SDK (PySpin) used by
`spinnaker_driver.py` is not on PyPI — install it from the vendor SDK package.
The rest of the stack runs without it via the simulated `GenericCameraDriver`.

`python app.py --headless --inject-faults` runs the simulated camera through the
entire SRS dataflow and deliberately injects malformed frames (NFR-006) and a
backend crash (NFR-005) so you can watch the skip-and-continue and auto-reconnect
behavior in the logs.

## Commands reference — runnable files & tests

All commands assume the package is installed (`pip install -e ".[all]"`) and, for
a new shell, the venv is active. There are two runnable scripts — `app.py` (the
runner) and `hardware_acceptance.py` (the qualifier); `gui_bridge.py` is the
viewer **module** they reuse, not run directly.

### `app.py` — the single runner (any backend, GUI or headless)

Runs the **simulator**, an **OpenCV/UVC** camera, or a **BlackFly S** (Spinnaker)
through the full pipeline (service → cueing + GUI). PySpin is imported only when
`--backend spinnaker` is chosen, so the script works with no SDK installed.
```bash
python app.py                                   # simulator + live viewer (default)
python app.py --backend spinnaker --serial 215  # real BlackFly S + viewer
python app.py --backend opencv --device 0       # webcam + viewer
python app.py --headless --seconds 10           # any backend, no GUI + stats
python app.py --headless --inject-faults        # sim NFR-005/006 demo
python app.py --no-cueing                       # display-only: frames -> GUI, no cueing
python app.py --backend spinnaker --exposure 12000 --gain 0   # tune brightness live
python app.py --backend spinnaker --reset       # factory-reset the camera, then exit
```

Exposure/gain/fps overrides apply at connect; the live HUD shows the resulting
exposure and gain so you can dial in "normal" brightness while watching the feed.
| Flag | Default | Description |
|---|---|---|
| `--backend {sim,opencv,spinnaker}` | `sim` | which camera driver to run |
| `--config PATH` | built-in (BFS config for spinnaker) | camera config JSON |
| `--serial S` | from config | spinnaker: bind to a specific camera serial (recommended for NFR-005) |
| `--device N` | `0` | opencv: capture device index |
| `--exposure US` | from config | exposure time in microseconds (overrides config; must be ≤ 1/fps) |
| `--gain DB` | from config | gain in dB (overrides config) |
| `--fps F` | from config | frame rate (overrides config) |
| `--headless` | off | no GUI; stream and log stats for `--seconds` |
| `--seconds N` | `10.0` | headless run duration (seconds) |
| `--inject-faults` | off | sim only: inject malformed frames + a crash to demo NFR-006/NFR-005 |
| `--no-cueing` | off | don't start the cueing consumer; serve frames to the GUI only (display-only acquisition) |
| `--reset` | off | spinnaker only: load the factory Default user set, set it as the power-on default, then exit |

The cueing consumer and the GUI are **independent fan-out branches**, so
`--no-cueing` gives you a pure live-view (`service → FIFO → CameraViewer`) with no
cueing thread. (`--no-cueing --headless` attaches no consumer at all — frames are
just acquired and dropped, useful only as a bare stream test.)

**Resetting the camera.** GenICam cameras keep their settings in the device across
app disconnects (and the vision system re-applies its config on every connect), so
to get back to factory defaults run `python app.py --backend spinnaker --reset`. It
loads the factory `Default` user set into the live registers *and* sets it as the
power-on default, so the camera also resets on a power-cycle afterward. (Note: a
normal run still re-applies your config — use `--reset` when you want the camera
clean for spinview or another tool.)

> The PyQt viewer needs the `gui` extra (`pip install -e ".[gui]"`); a real
> camera needs PySpin + the device. `gui_bridge.py` is imported by `app.py` for
> the viewer and is not meant to be run on its own.

**`hardware_acceptance.py`** — automated **acceptance/qualification** on a real
camera; prints a PASS/FAIL report and **exits 0 (pass) / 1 (fail)**. Needs PySpin
+ a camera.
```bash
python hardware_acceptance.py --serial 21512345
python hardware_acceptance.py --serial 21512345 --seconds 30 --cycles 10
python hardware_acceptance.py --mono --no-hw-timestamp        # mono / no device clock
```
| Flag | Default | Description |
|---|---|---|
| `--config PATH` | `config/bfs_u3_16s2c.json` | camera config JSON |
| `--serial S` | from config | bind to a specific camera serial |
| `--seconds N` | `20.0` | acquisition window |
| `--min-fps F` | `60.0` | min sustained FPS, NFR-001 (small finite-window tolerance applied) |
| `--min-resolution N` | `512` | min width/height in px, NFR-003 |
| `--mono` | off | expect a single-channel (mono) frame instead of color |
| `--no-hw-timestamp` | off | do not require a device hardware timestamp |
| `--max-incomplete-rate R` | `0.01` | max malformed/incomplete frame fraction, NFR-006 |
| `--max-dropped-rate R` | `0.005` | max dropped-frame fraction (device counter / timestamp gaps) |
| `--max-jitter-ms M` | `2.0` | max inter-frame interval stddev (ms) |
| `--min-mean V` | `2.0` | min mean pixel level (not black) |
| `--max-saturated R` | `0.10` | max saturated-pixel fraction (not blown out) |
| `--max-temperature C` | `75.0` | device temperature ceiling (°C) |
| `--cycles N` | `0` | also run N connect/stream/teardown cycles (0 = skip) |
| `--frames-per-cycle N` | `5` | frames read per cycle (with `--cycles`) |

### Tests

Pure stdlib `unittest` (no dependency); `pytest` also works after `pip install -e
".[dev]"`. `tests/test_properties.py` needs Hypothesis (`[dev]`) and skips
cleanly without it; `TestSpinnakerHardware` skips unless PySpin + a camera are
present.

```bash
python -m unittest discover -s tests             # run everything
python -m unittest discover -s tests -v          # verbose

# one file:
python -m unittest tests.test_acceptance
# one class / one method:
python -m unittest tests.test_suite.TestCameraService
python -m unittest tests.test_suite.TestCameraService.test_malformed_skip

# run the real-hardware smoke test on the rig (must say "Ran 1 test", not skipped):
python tests/test_suite.py TestSpinnakerHardware -v

# pytest equivalents (needs [dev]):
pytest                                            # all
pytest tests/test_suite.py::TestCameraService::test_malformed_skip
```

| Test file | What it covers | Needs |
|---|---|---|
| `tests/test_suite.py` | config, driver lifecycle, frame buffers, service (reconnect/skip), cueing handoff, lens math, script imports, HW-001 smoke | core |
| `tests/test_drivers_mocked.py` | Spinnaker/OpenCV driver logic via fake SDKs (chunk data, temperature, serial selection, teardown) | core |
| `tests/test_properties.py` | property-based frame-buffer invariants | `[dev]` (Hypothesis) |
| `tests/test_acceptance.py` | acceptance check matrix (pure) + simulator + connect cycles | core |

### Coverage & static checks (with `[dev]`)
```bash
coverage run --source=src/vision -m unittest discover -s tests && coverage report
ruff check .          # lint
mypy                  # type-check (config in pyproject.toml)
bandit -r src         # security scan   (pip install bandit)
```

## The cueing handoff (vision → cueing)

The vision system writes each frame into a `CircularFrameBuffer`; the cueing
system reads from it on its own worker thread (concurrent with the camera
worker, NFR-007). The buffer keeps the **freshest** frames (drop-oldest on lap),
which is what a real-time consumer wants.

`CueingSystem` is intentionally a thin consumer. The real cueing pipeline
(centroid extraction, tracking, state estimation, and the pixel→angular
pointing cue) is **undefined in the current SRS** and must not be invented; plug
it in via the `frame_processor` hook when a dedicated cueing spec exists:

```python
from vision import CameraService, CircularFrameBuffer, CueingSystem

svc = CameraService()
svc.add_cameras_from_config("config/camera.json", make_driver)
ring = CircularFrameBuffer(capacity=64)
svc.attach_sink("bfly0", ring)

def process(frame):          # <-- the future cueing pipeline goes here
    ...                      #     (out of scope; see cueing_system.py)

cueing = CueingSystem(ring, frame_processor=process)
svc.connect("bfly0"); cueing.start(); svc.start_streaming("bfly0")
```

## Real hardware

```python
from vision import CameraService
from vision.spinnaker_driver import SpinnakerCameraDriver

svc = CameraService()
svc.add_cameras_from_config("config/camera.json",
                            lambda cfg: SpinnakerCameraDriver(cfg))
```

### BlackFly S support (incl. color)

The Spinnaker driver works with any BlackFly S over PySpin. On `start_stream`
it sets `AcquisitionMode=Continuous` and `StreamBufferHandlingMode=NewestOnly`
(low-latency, no stale backlog), and every frame is converted on the host to
the config's `pixel_format` via `ImageProcessor` (legacy `img.Convert` on older
SDKs).

For **color** cameras this debayers automatically. Example config
`config/bfs_u3_16s2c.json` targets the **BFS-U3-16S2C-CS** (1.6 MP, color,
CS-mount, Sony IMX273, 1440×1080): the camera transmits its native `BayerRG8`
(`extra.device_pixel_format`) and the host produces 3-channel `BGR8`
(`pixel_format`) ready for the cueing system and GUI.

Optics: a **Kowa LM5JCM 5 mm f/2.8–16** C-mount lens (2/3″ image circle) gives
≈ 52.8° horizontal FOV on this sensor — well within NFR-004 (≥30°). Note the
C-mount lens needs a **5 mm C-to-CS adapter ring** on the CS-mount body.

```python
svc.add_cameras_from_config("config/bfs_u3_16s2c.json",
                            lambda cfg: SpinnakerCameraDriver(cfg))
```

`app.py --backend spinnaker` wires that driver through the whole pipeline (vision
serves frames → cueing consumer + PyQt viewer):

```bash
python app.py --backend spinnaker --serial 21512345                   # live viewer
python app.py --backend spinnaker --serial 21512345 --headless --seconds 10
```

> **Reconnect (NFR-005) needs a serial.** Bind the camera by **serial** — set
> `"serial"` in the config or pass `--serial`. Index-based selection
> (`device_index`) is **not stable across a USB re-enumeration** (unplug/replug
> or backend crash), so reconnect can bind to a stale handle and fail. Selecting
> by serial deterministically re-binds to the same physical unit. The driver
> logs a WARN at connect time when no serial is configured.

The real-device path is exercised by `TestSpinnakerHardware` (HW-001), which
captures a frame and asserts a full-resolution BGR8 image. It **skips
automatically** when PySpin or a camera is absent, so CI stays green while the
rig validates the hardware path.

### Configuring the camera (`config/bfs_u3_16s2c.json`)

The config selects the camera and holds the acquisition settings the driver
applies on `connect()`. Not every field is sent to hardware — some are advisory
(they only drive `validate()` warnings or documentation).

**Applied to the camera:** `serial` / `device_index` (selection), `resolution`
(→ Width/Height), `fps`, `exposure_us`, `gain_db`, `extra.device_pixel_format`
(device format), and `pixel_format` (host debayer **output**).

**Advisory only (not sent):** `model`, `max_resolution`, `max_fps`, `bit_depth`,
`dynamic_range_db`, `sensor_format`, `lens_fov_deg`, `focal_length_mm`, and the
`extra` notes.

What to set / tune for your rig:

- **`serial`** — set it to the value spinview reports; it binds to that physical
  unit reliably across reboots **and across reconnects (NFR-005)**. Leaving it
  `null` falls back to `device_index`, whose ordering can shuffle during a USB
  re-enumeration — so reconnect becomes unreliable (the driver warns about this
  at connect time). `device_index` only matters with multiple cameras; with one
  camera, still prefer setting `serial` if you care about auto-reconnect.
- **`exposure_us`** — the main thing to tune. `5000` is a placeholder; raise/
  lower it for your lighting. Must be ≤ 1/`fps` (≤ 16,600 µs at 60 fps).
- **`gain_db`** — keep at `0`; raise only if still too dark after exposure.
- **`fps`** — `60` meets NFR-001; the camera can do more but 60 is comfortable
  over USB3.
- **Leave** `device_pixel_format: BayerRG8` + `pixel_format: BGR8` as-is — that
  pair is the color debayer setup.

The **C-to-CS adapter** is a physical part (the config only notes it).

### First-time hardware bring-up (Ubuntu 22.04)

End-to-end checklist for first light with the BFS-U3-16S2C-CS (USB3 color).

**Phase 0 — USB3 / OS prep.** USB3 cameras need a larger USB-FS buffer or you
get incomplete/dropped frames:

```bash
# temporary (until reboot):
sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'
# permanent: add usbcore.usbfs_memory_mb=1000 to GRUB_CMDLINE_LINUX_DEFAULT
sudo nano /etc/default/grub      # ...="quiet splash usbcore.usbfs_memory_mb=1000"
sudo update-grub && sudo reboot
```

**Phase 1 — Spinnaker SDK + PySpin.** Download both for **Ubuntu 22.04 / amd64**
from the Teledyne FLIR site (account required):

```bash
sudo sh install_spinnaker.sh        # SAY YES to udev rules + flirimaging group
# then log out / back in so the group membership takes effect

python3.10 -m venv .venv && source .venv/bin/activate   # match the wheel's Python
pip install -e .
pip install spinnaker_python-<ver>-cp310-cp310-linux_x86_64.whl
```

Version traps: the **PySpin wheel must match your Python version** (Ubuntu 22.04
default is 3.10 → `cp310`); if `import PySpin` or `GetNDArray()` errors, your
SDK was built against NumPy 1.x — `pip install "numpy<2"`.

**Phase 2 — confirm the OS sees the camera.**

```bash
spinview        # GUI: you should see the camera and a live image
# or headless:
python -c "import PySpin; s=PySpin.System.GetInstance(); c=s.GetCameras(); print('cameras:', c.GetSize()); c.Clear(); s.ReleaseInstance()"
```

If `spinview` shows nothing, it's a Phase 0/1 issue (usbfs / group / udev), not
the code.

**Phase 3 — project sanity (no camera needed).**

```bash
pip install -e ".[dev]"
python -m unittest discover -s tests      # HW-001 skips -> "OK (skipped=1)"
```

**Phase 4 — hardware smoke test (now runs instead of skipping).**

```bash
python tests/test_suite.py TestSpinnakerHardware -v
```

Proves the BayerRG8→BGR8 capture path: full-res 1440×1080, 3-channel BGR8,
hardware timestamp. If this passes, the driver works with your camera.

**Phase 5 — live pipeline.**

```bash
pip install -e ".[gui]"
python app.py --backend spinnaker --serial <SERIAL>
```

A window titled `bfs16s2c (spinnaker)` opens with the live ~30 FPS feed:
natural-color (debayered BGR) video, with a status bar reading
`bfs16s2c  frame=…  age=… ms  shown=…`. The terminal logs `Streaming 'bfs16s2c'`
and `Close the window to stop.`

This confirms two things at once: **focus** sharpens (→ the 5 mm C-to-CS adapter
is fitted) and **color** looks natural (→ debayer is correct). Image dark or
saturated → adjust `exposure_us` in the config. Common bad states: black window
→ exposure too low or not streaming; pink/green tint → debayer/pixel-format
mismatch; won't focus → missing C-to-CS adapter.

**Phase 6 — robustness on real hardware.** Unplug the USB cable while streaming:
you should see `Backend fault … Reconnect attempt N`, then `Reconnected` on
replug — NFR-005 validated for real. **This requires the camera to be bound by
serial** (config `serial` or `--serial`); without it, selection falls back to
`device_index`, which is not stable across the re-enumeration, and reconnect may
fail to rebind. The driver logs a WARN at connect when no serial is set.

| Symptom | Cause | Fix |
|---|---|---|
| `spinview` sees no camera | udev / group / permissions | re-run installer, join `flirimaging`, re-login |
| Incomplete / dropped frames | USB-FS buffer too small | Phase 0 `usbfs_memory_mb=1000` |
| `import PySpin` fails | Python / NumPy mismatch | `cp310` wheel + `numpy<2` |
| Reconnect doesn't recover (NFR-005) | selecting by `device_index` | set `serial` in config or pass `--serial` |
| Image never focuses | C lens on CS body | add 5 mm C-to-CS adapter |
| Pink / green tint | wrong Bayer handling | confirm `device_pixel_format: BayerRG8` in config |

## Automated acceptance testing (rig qualification)

`hardware_acceptance.py` is the machine-vision **bring-up qualification** tool:
it streams from a real camera for a fixed window, evaluates a matrix of
objective PASS/FAIL checks, prints a report, and **exits non-zero on any
failure** (so it drops straight into CI / a go-no-go gate).

```bash
python hardware_acceptance.py --serial 21512345                 # 20s, NFR defaults
python hardware_acceptance.py --serial 21512345 --seconds 30 --min-fps 60
python hardware_acceptance.py --serial 21512345 --cycles 10     # + teardown-churn test
python hardware_acceptance.py --mono --no-hw-timestamp          # mono / no device clock
```

**Checks (PASS/FAIL):** streaming reached · frames received · resolution
(NFR-003) · pixel format & dtype · sustained throughput (NFR-001, with a small
finite-window tolerance) · frame integrity / malformed rate (NFR-006) ·
hardware-timestamp presence + monotonicity · inter-frame jitter · **dropped-frame
rate** · **device temperature** (health) · image sanity (not black / not
saturated) · stream stability (no reconnects during the run). With `--cycles N`
it also runs N connect→stream→teardown cycles to validate clean release (the
`gc.collect`/`atexit` teardown path) — the automated counterpart to a manual
unplug/replug.

**Authoritative drop detection (GenICam chunk data).** The Spinnaker driver
enables chunk data so each frame carries the camera's own **device frame
counter**; a gap in that counter is a *real* dropped frame. The acceptance tool
uses it when present (falling back to inferring drops from hardware-timestamp
gaps, then skipping if neither is available). **Device health telemetry**
(`DeviceTemperature` plus any available transport counters such as
`StreamLostFrameCount`) is polled ~1 Hz during the run, surfaced in
`CameraService.get_health()`, gated by `--max-temperature`, and reported.

**Informational only (not pass/fail):** sharpness (a focus proxy: variance of
the Laplacian) and per-channel means (a colour-tint proxy). These are
scene-dependent and can't be graded without a reference target — eyeball them
with `app.py --backend spinnaker`.

All thresholds are flags (`--min-fps`, `--max-jitter-ms`, `--max-dropped-rate`,
`--min-mean`, `--max-saturated`, …) so you can tune the gate to your rig.

The check logic itself is unit-tested without hardware (`tests/test_acceptance.py`):
`evaluate()` is a pure function of collected metrics, so every PASS/FAIL/SKIP
path is verified deterministically, and the full collect→evaluate path plus the
cycle test run against the simulator.

### Recommended rig validation sequence
```bash
# 1. plumbing go/no-go (must say "Ran 1 test", not skipped):
python tests/test_suite.py TestSpinnakerHardware -v
# 2. objective acceptance gate (exit 0 = pass):
python hardware_acceptance.py --serial <SERIAL> --seconds 30 --cycles 10
# 3. image quality (focus / exposure / true colour) — eyeball:
python app.py --backend spinnaker --serial <SERIAL>
# 4. reconnect (NFR-005) — unplug/replug the USB during step 2 or 3.
```

## Requirements traceability (SRS v0.2 §7)

| Req | Where | Test |
|---|---|---|
| FR-001 ingest config + init | `config_loader.py`, `CameraService.add_cameras_from_config` | `TestConfigLoader`, `test_config_file_registration` |
| FR-002 real-time streaming | `CameraService.start_streaming`, drivers `start_stream`/`read_frame` | `test_fanout_to_two_sinks`, `TestGenericDriver.test_lifecycle` |
| FR-003 image processing | **moved to cueing system** (out of scope); `CueingSystem` consumes frames | `TestCueingEndToEnd` (handoff) |
| FR-004 GUI callback / visualization | `frame_buffers` FIFO + `gui_bridge.py` | `test_frames_served_to_cueing_and_gui` |
| FR-005 GPU acceleration (P2) | belongs to cueing image processing (out of scope) | — |
| NFR-001 ≥60 FPS | `CameraConfig.validate` | `test_low_fps_warns` |
| NFR-002 ≤5 ms latency | **no longer required** (image processing out of scope) | — |
| NFR-003 ≥512² | `CameraConfig.validate` | `test_undersized_resolution_warns` |
| NFR-004 FOV ≥30° | `CameraConfig.validate` (lens_fov_deg), `lens.py` | `test_narrow_fov_warns`, `TestLensCalculator` |
| NFR-005 backend recovery | `CameraService._attempt_reconnect` | `test_backend_crash_reconnects`, mocked fault classification |
| NFR-006 skip bad frames | `MalformedFrameError`; service + cueing skip | `test_malformed_skip`, `test_processor_error_skipped` |
| NFR-007 thread-safe / no deadlock | frame-buffer lock+condition; concurrent service/cueing threads | `test_circular_concurrent_no_deadlock` |
| NFR-008 GUI never blocks | drop-newest FIFO + QTimer poll | `test_frames_served_to_cueing_and_gui` |
| NFR-009 driver abstraction | `CameraDriver` ABC + 3 adapters | `TestGenericDriver`, `tests/test_drivers_mocked.py` |
| NFR-010 test coverage | `tests/` (unit + mocked + property) | this suite |
| NFR-011 GUI ≥10 FPS refresh | `gui_bridge` QTimer @ 33 ms (~30 FPS) | (Milestone IV, manual) |

### Minimum test coverage (resolves SRS §7 TODO)
Critical submodules reached by the testbench: config ingestion, driver lifecycle
+ feature gating, the real-hardware driver logic (via fake PySpin/cv2), the
frame buffers (drop policies, concurrency, property-based invariants), the
service-level failure paths (malformed-skip, reconnect), the cueing handoff, and
the lens/FOV math. Regression-critical features: the NFR-005/006/007 robustness
behaviors and the vision→cueing frame handoff — all asserted on every run.

## Milestones (SRS v0.2 §7)

| Milestone | Objective | Covered by |
|---|---|---|
| I — Driver base-class types | Abstract `CameraDriver` API + shared data types | `camera_types.py`, `camera_driver.py` |
| II — Driver requirements | NFR-005/006/001/003/009, FR-001 on the driver | `TestGenericDriver`, `tests/test_drivers_mocked.py` |
| III — Service-layer requirements | NFR-005/006/001/003/007, FR-001/004 on the service | `TestCameraService`, `TestCueingEndToEnd` |
| IV — GUI testbench + handoff | NFR-011, FR-004 real-time visualization + frame handoff | `gui_bridge.py` + `app.py` (manual), `TestCueingEndToEnd` |

## Logging (SRS 5.5)
Standard `logging` with INFO (init/connect/frame receipts), WARN (dropped or
malformed frames, timeouts, reconnect attempts), and ERROR (unreachable
backends, failed connects, missing config). Verbosity is set by the host
application via `logging.basicConfig`.
