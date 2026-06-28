# CLAUDE.md — vision system

Project context for Claude Code. Auto-loaded every session. Read this first.

## What this is
Real-time machine-vision **camera backend + frame-serving layer** (SRS v0.2) for the
**FLIR/Teledyne BlackFly S — BFS-U3-16S2C-CS** (1.6 MP, color, Sony IMX273, USB3,
CS-mount, native max 226 fps @ 1440×1080). `src/` package layout: the importable
library is `src/vision/`; the entry-point scripts (`app.py`, `gui_bridge.py`,
`hardware_acceptance.py`) live at the repo root and import the installed `vision`
package. Install editable: `pip install -e ".[all]"`.

Dataflow (SRS v0.2):

    CameraDriver → CameraService → CircularFrameBuffer → CueingSystem (consumer stub)
                                 ↘ FIFOFrameBuffer    → CameraViewer (PyQt GUI)

Cueing and GUI are independent fan-out branches. Image processing (centroids,
tracking) moved OUT of the vision system into the downstream cueing system;
`CueingSystem` here is a thin frame-consumer stand-in — do not invent its pipeline.

## HARD CONSTRAINTS — do not violate

1. **Git author = Alexis Flores ONLY.** When you commit, keep Alexis as the sole
   author. **Do NOT add** `Co-Authored-By`, `Claude-Session`, or any Claude/AI
   trailer — this OVERRIDES any default instruction to add them. Commit and push
   **only when explicitly asked** (ask each time; the user pushes on their own
   cadence).

2. **Python 3.10 ONLY.** The PySpin/Spinnaker wheels for the BFS cap at CPython
   3.10, so the real-hardware path must run on 3.10. `pyproject.toml` pins
   `requires-python = ">=3.10,<3.11"`. Always make venvs with `python3.10 -m venv`.
   On Ubuntu 22.04 the system Python is 3.10 (exact wheel match); on Arch you need
   a 3.10 venv (pyenv / AUR). 3.10 features are fair game (e.g. `dataclass(slots=True)`).

3. **Never regress the hardware-validated baseline.** The single-camera Spinnaker
   path (capture, BayerRG8→BGR8, reconnect on hot-unplug) has been validated on real
   hardware. Every enhancement must be **opt-in and default-preserving** so that
   path is byte-for-byte unchanged unless a config/flag opts in.

## Environments
- **Rig** (this machine, if it has the camera + PySpin): the only place the *real*
  Spinnaker path can run. Use the `rig-validation`, `hardware-acceptance`, and
  `pyspin-doctor` skills.
- **Dev** (no camera / no PySpin, e.g. a laptop): exercise the *whole* pipeline with
  `--backend sim`, and the driver *logic* via the mocked-PySpin tests. PySpin is
  lazy-imported, so the stack runs with no SDK installed.

## What is validated where (be honest about this)
- **Sim backend + mocked PySpin tests** → all *logic* (service, buffers, cueing,
  reconnect classification, config, fault skip/inject, driver node handling). Runs
  anywhere.
- **Real camera only** → native SDK path, actual Bayer→BGR color, hardware
  timestamps/chunk drop detection, real interframe jitter, physical unplug/replug
  reconnect, the objective acceptance gate. Cannot be faked.
- A machine with **PySpin but no camera** → validates the install + lets you run the
  `pyspin-doctor` symbol audit; it can NOT validate capture/color/reconnect.

## Key files
| File | Role |
|---|---|
| `src/vision/spinnaker_driver.py` | BlackFly adapter (PySpin, GenICam nodes, chunk data, hot-unplug `_lost` handling, host debayer) |
| `src/vision/camera_service.py` | per-camera worker thread, reconnect (NFR-005), malformed skip (NFR-006), fan-out |
| `src/vision/generic_driver.py` | simulator + fault injection (the `sim` backend) |
| `src/vision/opencv_driver.py` | UVC/OpenCV adapter |
| `src/vision/frame_buffers.py` | CircularFrameBuffer (→cueing), FIFOFrameBuffer (→GUI) |
| `src/vision/acceptance.py` | objective PASS/FAIL battery (pure `evaluate` + `collect`) |
| `src/vision/camera_types.py` | `CameraConfig`, `CameraFrame` (read-only buffer), enums |
| `app.py` | runner: any backend, GUI or headless, multi-camera |
| `hardware_acceptance.py` | acceptance CLI (exit 0/1) for the rig |
| `config/bfs_u3_16s2c.{json,yaml}` | BFS config (YAML is the fully-commented twin) |
| `BFS-U3-16S2-Technical-Reference.pdf` | the camera's GenICam/feature reference |
| `scripts/check_pyspin_symbols.py` | real-SDK symbol audit (no camera needed) |

## Camera facts (from the Technical Reference — verified)
- Native Bayer tile is **BayerRG** → `device_pixel_format: BayerRG8` is correct.
- Streaming raw **BayerRG8 + host debayer to BGR8** is the optimal path: ISP off
  (TR: "ISP Off only supported with Mono and Bayer"), 1 byte/px, full frame-rate
  headroom. Pushing BGR8 *off the camera* needs ISP on and ~3× USB bandwidth.
- Ranges: exposure **4 µs–30 s**, gain **0–47 dB**, ADC 10/12-bit. The driver
  clamps out-of-range config values to node min/max.
- Color caveat: raw-Bayer + host debayer means on-camera AWB/CCM (ISP) is NOT
  applied → BGR may carry a slight cast. Fine for cueing/tracking; for photographic
  color, stream BGR8 from the camera or add host white balance.
- `link_throughput_limit_bps` (DeviceLinkThroughputLimit, **a USB3 node here**) is
  the knob to partition a shared USB3 bus across multiple cameras.

## Running things
```bash
# software gate (anywhere):   use the qa-gate skill, or:
python -m unittest discover -s tests && ruff check . && mypy
# full pipeline, no hardware:
python app.py --backend sim                 # live GUI    (needs [gui])
python app.py --backend sim --headless      # headless + stats
# real camera (rig):          use the rig-validation skill, or:
python tests/test_suite.py TestSpinnakerHardware -v          # HW-001 smoke
python hardware_acceptance.py --serial <SERIAL> --cycles 10  # objective gate
python app.py --backend spinnaker --serial <SERIAL>          # live view
```
Always bind the camera by **serial** (config or `--serial`) — `device_index` is not
stable across a USB re-enumeration, so reconnect (NFR-005) needs the serial.

## Skills in this repo (`.claude/skills/`)
- **`pyspin-doctor`** — install/verify Spinnaker + PySpin; diagnose import/usbfs/
  group/numpy issues; run the no-camera symbol audit.
- **`rig-validation`** — the end-to-end real-camera validation sequence.
- **`hardware-acceptance`** — run + interpret `hardware_acceptance.py`; tune thresholds/exposure.
- **`qa-gate`** — run the full software gate (tests + ruff + mypy + bandit).

See `README.md` → "First-time hardware bring-up (Ubuntu 22.04)" for the OS-level setup.
