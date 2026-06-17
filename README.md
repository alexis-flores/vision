# Vision System — Camera Backend & Centroid Pipeline

Implementation of the Software Requirements Specification (SRS) for the
real-time vision system: it interfaces the camera driver SDK behind a modular
abstraction, streams frames, extracts a centroid-profile list, shares it with
the queuing subsystem through a ring buffer, and provides a non-blocking GUI
callback for visualization.

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
 VisionSystem                 gui_bridge: QTimer @33ms (SRS Fig 1)
   FR-003 centroid extract        │ FR-004 / NFR-008 non-blocking
   FR-005 optional GPU            ▼
      │ CentroidProfile        PyQt viewer (annotated frames)
      ▼
 CentroidRingBuffer  ── SRS 5.2, independent R/W ptrs, RxTimebase, NFR-007
      │
      ▼
 QueuingSubsystem  ── downstream consumer ([Vision] → [Queuing])
```

## Layout

`src/` package layout. The importable library lives in `src/vision/`; the
entry-point scripts, config, and tests sit at the repo root.

```
src/vision/        # the `vision` package (importable library)
    __init__.py    # public API re-exports
    camera_*.py  centroid_*.py  *_driver.py  ...
main.py  gui_bridge.py  run_hardware.py   # entry-point scripts (not packaged)
config/            # camera JSON configs
tests/             # test suite
pyproject.toml  requirements.txt  README.md
```

After `pip install -e .` the library is importable as, e.g.,
`from vision import CameraService, VisionSystem` (or
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
| `frame_buffers.py` | `CircularFrameBuffer` (vision), `FIFOFrameBuffer` (GUI) | 5.6 |
| `camera_service.py` | Multi-camera mgmt, worker threads, reconnect, fan-out | FR-002, NFR-005/006 |
| `centroid_types.py` | `Centroid`, `CentroidProfile` — resolves the 5.2 TODO | 5.2 |
| `centroid_buffer.py` | `CentroidRingBuffer` + `RxTimebase` (shared w/ queuing) | 5.2, NFR-007 |
| `centroid_extraction.py` | `CentroidExtractor` (contour/CC, CPU + optional GPU) | FR-003, FR-005, NFR-002 |
| `lens.py` | Lens/FOV calculators (FLIR app-note: angular + linear FOV) | NFR-004 |
| `vision_system.py` | Vision worker: frame → centroids → ring → GUI callback | FR-003/004, NFR-006/008 |
| `queuing_subsystem.py` | Downstream consumer of the centroid ring | dataflow |
| `gui_bridge.py` | PyQt viewer, 33 ms demand-driven poll | FR-004, 5.4 |
| `main.py` | Headless end-to-end demo on the simulator (incl. NFR-005/006 fault injection) | — |
| `run_hardware.py` | Full pipeline on a real BlackFly S (Spinnaker) + PyQt viewer | — |
| `tests/test_suite.py` | Unit/integration/failure tests | NFR-010, 6.1 |
| `config/camera.json` | Example BlackFly config | FR-001 |
| `config/bfs_u3_16s2c.json` | BFS-U3-16S2C-CS color config (BayerRG8 → BGR8) | FR-001 |
| `pyproject.toml` | Package metadata + optional-dependency extras | — |
| `requirements.txt` | Core runtime dependencies | — |

## Quick start

Because of the `src/` layout, install the package (editable) so the `vision`
package is importable by the scripts:

```bash
python -m venv .venv && source .venv/bin/activate   # recommended

pip install -e ".[all]"                             # everything (recommended)

# or pick what you need:
pip install -e .                                    # core (numpy, opencv) + the package
pip install -e ".[gui]"                             # + PyQt6 live viewer
pip install -e ".[dev]"                             # + pytest
```

`".[all]"` is the single command that installs everything pip-installable
(core + GUI + YAML + scipy + pytest). The only piece it can't include is
**PySpin** (the Spinnaker SDK is a vendor download, not on PyPI) — install that
separately, and only if you're driving real BlackFly hardware. GPU/CuPy is also
opt-in (`".[gpu]"`) since it needs a CUDA-matched wheel.

Run it:

```bash
python main.py                               # headless full-pipeline demo
python -m unittest discover -s tests         # run the test suite (42 cases)
python gui_bridge.py                          # live PyQt viewer w/ centroid overlay (needs [gui])
```

| Extra | Installs | Enables |
|---|---|---|
| `all`   | gui + yaml + dev | everything pip-installable, one command |
| `gui`   | PyQt6  | `gui_bridge.py` live viewer (FR-004) |
| `yaml`  | PyYAML | YAML camera configs in addition to JSON |
| `gpu`   | CuPy   | GPU extraction path (FR-005); auto-falls back to CPU |
| `scipy` | SciPy  | connected-components fallback when OpenCV is absent |
| `dev`   | pytest, SciPy | running the test suite + fallbacks |

**Real BlackFly hardware:** the FLIR/Teledyne Spinnaker SDK (PySpin) used by
`spinnaker_driver.py` is not on PyPI — install it from the vendor SDK package.
The rest of the stack runs without it via the simulated `GenericCameraDriver`.

`main.py` runs the simulated camera through the entire SRS dataflow and
deliberately injects malformed frames (NFR-006) and a backend crash (NFR-005)
so you can watch the skip-and-continue and auto-reconnect behavior in the logs.

## Real hardware

```python
from camera_service import CameraService
from spinnaker_driver import SpinnakerCameraDriver

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
(`pixel_format`) ready for the centroid pipeline and GUI.

Optics: a **Kowa LM5JCM 5 mm f/2.8–16** C-mount lens (2/3″ image circle) gives
≈ 52.8° horizontal FOV on this sensor — well within NFR-004 (≥30°). Note the
C-mount lens needs a **5 mm C-to-CS adapter ring** on the CS-mount body.

```python
svc.add_cameras_from_config("config/bfs_u3_16s2c.json",
                            lambda cfg: SpinnakerCameraDriver(cfg))
```

`run_hardware.py` wires that driver through the whole pipeline (vision →
queuing → annotated PyQt viewer) — the hardware counterpart of `main.py`:

```bash
python run_hardware.py                            # live viewer + centroid markers
python run_hardware.py --headless --seconds 10    # no GUI, logs stats
python run_hardware.py --threshold 100 --min-area 8   # tune for your scene
```

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
  unit reliably across reboots. `device_index` only matters with multiple
  cameras (it picks the Nth in Spinnaker's enumeration order, which can shuffle);
  with one camera leave it `0`.
- **`exposure_us`** — the main thing to tune. `5000` is a placeholder; raise/
  lower it for your lighting. Must be ≤ 1/`fps` (≤ 16,600 µs at 60 fps). Too
  high → motion blur / saturated blobs; too low → dark, no detections.
- **`gain_db`** — keep at `0`; raise only if still too dark after exposure
  (gain amplifies noise, which hurts centroid quality).
- **`fps`** — `60` meets NFR-001; the camera can do more but 60 is comfortable
  over USB3.
- **Leave** `device_pixel_format: BayerRG8` + `pixel_format: BGR8` as-is — that
  pair is the color debayer setup.

Not in the config: **centroid detection** (`--threshold`, `--min-area`) is tuned
at runtime on `run_hardware.py`, not here; and the **C-to-CS adapter** is a
physical part (the config only notes it).

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
pip install -r requirements.txt
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
python run_hardware.py
```

A window titled `bfs16s2c - centroids` opens with the live ~30 FPS feed:
natural-color (debayered BGR) video, a **red crosshair** on each detected
centroid, a **green box** around each blob, and a status bar reading
`bfs16s2c  frame=…  age=… ms  centroids=N  shown=…`. The terminal logs
`Streaming 'bfs16s2c' (extractor backend=cpu)` and `Close the window to stop.`

This confirms three things at once: **focus** sharpens (→ the 5 mm C-to-CS
adapter is fitted), **color** looks natural (→ debayer is correct), and the
crosshairs track your bright targets (→ FR-003 works on real data). Tuning:
no / too many markers → adjust `--threshold` / `--min-area`; image dark or
saturated → adjust `exposure_us` in the config. Common bad states: black window
→ exposure too low or not streaming; pink/green tint → debayer/pixel-format
mismatch; won't focus → missing C-to-CS adapter.

**Phase 6 — robustness on real hardware.** Unplug the USB cable while streaming:
you should see `Backend fault … Reconnect attempt N`, then `Reconnected` on
replug — NFR-005 validated for real.

| Symptom | Cause | Fix |
|---|---|---|
| `spinview` sees no camera | udev / group / permissions | re-run installer, join `flirimaging`, re-login |
| Incomplete / dropped frames | USB-FS buffer too small | Phase 0 `usbfs_memory_mb=1000` |
| `import PySpin` fails | Python / NumPy mismatch | `cp310` wheel + `numpy<2` |
| Image never focuses | C lens on CS body | add 5 mm C-to-CS adapter |
| Pink / green tint | wrong Bayer handling | confirm `device_pixel_format: BayerRG8` in config |

## Performance (NFR-001/002)

`CentroidExtractor` defaults to contour tracing, which scales with blob
boundary rather than pixel count — the right choice for sparse optical
targets. Measured single-thread CPU latency (6 spots):

| Resolution | contour p99 | connected-components p99 | NFR-002 (≤5 ms) |
|---|---|---|---|
| 512×512   | ~1.1 ms | ~2.4 ms | ✅ |
| 1024×1024 | ~1.8 ms | ~9.4 ms | contour ✅ / CC ✗ |

For very high resolutions or dense fields, enable the GPU path
(`ExtractorParams(use_gpu=True)`, FR-005), which offloads grayscale/threshold
to the device and falls back to CPU automatically if no GPU is present.

## Requirements traceability (SRS 6.1)

| Req | Where | Test |
|---|---|---|
| FR-001 ingest config + init | `config_loader.py`, `CameraService.add_cameras_from_config` | `TestConfigLoader`, `test_config_file_registration` |
| FR-002 real-time streaming | `CameraService.start_streaming`, drivers `start_stream`/`read_frame` | `test_fanout_to_two_sinks`, `TestGenericDriver.test_lifecycle` |
| FR-003 centroid extraction | `centroid_extraction.py`, `vision_system.py` | `TestCentroidExtraction`, `test_frames_to_centroids_to_queue` |
| FR-004 GUI callback | `vision_system` gui_callback, `gui_bridge.py` | `test_frames_to_centroids_to_queue` (gui_hits), gui offscreen |
| FR-005 GPU acceleration (P2) | `CentroidExtractor` GPU path + CPU fallback | exercised via CPU fallback in `TestCentroidExtraction` |
| NFR-001 ≥60 FPS | `CameraConfig.validate`, contour extractor headroom | `test_low_fps_warns`, perf table |
| NFR-002 ≤5 ms latency | contour extractor; `VisionSystem` measures & logs | `test_latency_budget` |
| NFR-003 ≥512² | `CameraConfig.validate` | `test_undersized_resolution_warns` |
| NFR-004 FOV ≥30° | `CameraConfig.validate` (lens_fov_deg) | `test_narrow_fov_warns` |
| NFR-005 backend recovery | `CameraService._attempt_reconnect` | `test_backend_crash_reconnects` |
| NFR-006 skip bad frames | `MalformedFrameError`, service + vision skip | `test_malformed_skip`, `test_invalid_frame_skipped` |
| NFR-007 thread-safe / no deadlock | `CentroidRingBuffer` lock + condition | `test_concurrent_no_deadlock` |
| NFR-008 GUI never blocks | drop-newest FIFO + QTimer poll | `test_frames_to_centroids_to_queue` |
| NFR-009 driver abstraction | `CameraDriver` ABC + 3 adapters | `TestGenericDriver` |
| NFR-010 test coverage | `tests/test_suite.py` (34 cases) | this suite |

### Minimum test coverage (resolves SRS 6.1 TODO)
Critical submodules reached by the testbench: config ingestion, driver
lifecycle + feature gating, centroid extraction (accuracy + latency), the
shared centroid ring buffer (FIFO order, overwrite, concurrency), frame
buffers, and the service-level failure paths (malformed-skip, reconnect).
Regression-critical features: centroid accuracy, NFR-002 latency budget, and
the NFR-005/006/007 robustness behaviors — all asserted on every run.

## Logging (SRS 5.5)
Standard `logging` with INFO (init/connect/frame receipts), WARN (dropped or
malformed frames, timeouts, reconnect attempts, latency-budget overruns), and
ERROR (unreachable backends, failed connects, missing config). Verbosity is
set by the host application via `logging.basicConfig`.
