---
name: pyspin-doctor
description: Install, verify, or troubleshoot the FLIR Spinnaker SDK + PySpin on the rig. Use when import PySpin fails, the camera isn't enumerating, after a fresh install, or to run the no-camera symbol audit. Covers Ubuntu 22.04 and Arch.
---

# pyspin-doctor

Get the Spinnaker SDK + PySpin toolchain healthy. Much of this works **without a
camera** — it validates the install and the API surface the driver depends on.

## 0. Ground rules
- PySpin must match the venv Python: BFS wheels are **cp310** → use a `python3.10`
  venv (`python3.10 -m venv .venv && . .venv/bin/activate`). Ubuntu 22.04 system
  Python is already 3.10; on Arch make a 3.10 venv (pyenv or AUR `python310`).
- PySpin is NOT on PyPI and NOT a declared dependency — it's a vendor wheel installed
  by hand and lazy-imported. The rest of the stack runs without it (`--backend sim`).

## 1. Is it even installed? (no camera needed)
```bash
. .venv/bin/activate 2>/dev/null
python -c "import PySpin; print('PySpin', PySpin.__version__ if hasattr(PySpin,'__version__') else 'ok')"
```
- `ModuleNotFoundError` → not installed in this venv. Install: extract the FLIR
  *python* tarball and `pip install spinnaker_python-*-cp310-*.whl`. The SDK proper
  (`.deb`s / libs) must also be installed (see README bring-up).
- `ImportError: libSpinnaker.so... cannot open shared object` → the SDK runtime libs
  aren't on the loader path. Ubuntu: run FLIR's `install_spinnaker.sh`. Arch: install
  the SDK (AUR `spinnaker-sdk` or extract the `.deb`s) and ensure `/usr/lib` /
  `ldconfig` sees `libSpinnaker.so`.
- `ImportError` mentioning NumPy / `GetNDArray` → SDK built against NumPy 1.x:
  `pip install "numpy<2"`.

## 2. Does the SDK see the transport layer? (no camera needed)
```bash
python -c "import PySpin; s=PySpin.System.GetInstance(); v=s.GetLibraryVersion(); print('SDK', v.major, v.minor, v.type, v.build); c=s.GetCameras(); print('cameras:', c.GetSize()); c.Clear(); s.ReleaseInstance()"
```
`cameras: 0` with no errors = SDK + GenTL load fine, just no device. That's the
expected healthy state on a camera-less machine.

## 3. Symbol audit — does THIS SDK expose every API the driver uses? (no camera needed)
The mocked tests fake every `PySpin.*` symbol, so they can't catch SDK-version drift.
This checks the real install:
```bash
python scripts/check_pyspin_symbols.py
```
Exit 0 = all required symbols present. It reports REQUIRED (driver breaks without) vs
OPTIONAL (has a runtime fallback) so you know severity. If a REQUIRED symbol is
missing, the installed SDK is incompatible with `spinnaker_driver.py` — report which
one and we adapt the driver (most have FLIR rename history).

## 4. With a camera attached — enumeration / permissions
```bash
python -c "import PySpin; s=PySpin.System.GetInstance(); c=s.GetCameras(); print('size', c.GetSize()); [print(cam.TLDevice.DeviceSerialNumber.GetValue()) for cam in c]; c.Clear(); s.ReleaseInstance()"
```
If `spinview` / SpinView sees the camera but Python doesn't, or vice versa, it's
almost always one of:
| Symptom | Cause | Fix |
|---|---|---|
| size 0 but device plugged in | not in `flirimaging` group / udev rules | re-run installer; `sudo usermod -aG flirimaging $USER`; **log out/in** |
| "image incomplete" / dropped frames once streaming | USB-FS buffer too small | `sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'` (permanent: `usbcore.usbfs_memory_mb=1000` in GRUB, reboot) |
| Permission denied opening device | udev rules absent (Arch) | install FLIR udev rules to `/etc/udev/rules.d/`, `sudo udevadm control --reload && sudo udevadm trigger` |

## 5. Confirm the project imports cleanly
```bash
pip install -e ".[dev]"
python -c "import app, hardware_acceptance; from vision.spinnaker_driver import SpinnakerCameraDriver; print('imports ok')"
```

After the toolchain is green, hand off to **rig-validation** (with a camera) or report
the symbol audit result (without one).
