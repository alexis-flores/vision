---
name: rig-validation
description: Run the full real-camera validation sequence for the BFS-U3-16S2C-CS on the rig (HW-001 smoke, acceptance gate, live-view eyeball, hot-unplug reconnect). Use when a real camera is connected and you want to qualify the hardware path end to end.
---

# rig-validation

The end-to-end qualification of the **real** Spinnaker path. Requires a camera +
PySpin (run `pyspin-doctor` first if unsure). Bind by **serial** throughout.

Get the serial first:
```bash
python -c "import PySpin; s=PySpin.System.GetInstance(); c=s.GetCameras(); print([cam.TLDevice.DeviceSerialNumber.GetValue() for cam in c]); c.Clear(); s.ReleaseInstance()"
```

## Step 1 — plumbing go/no-go (HW-001)
```bash
python tests/test_suite.py TestSpinnakerHardware -v
```
MUST say **"Ran 1 test"**, not "skipped". Proves BayerRG8→BGR8 capture: full-res
1440×1080, 3-channel BGR8 uint8, a hardware timestamp. If it skips, PySpin/camera
isn't visible → `pyspin-doctor`.

## Step 2 — objective acceptance gate (exit 0 = pass)
```bash
python hardware_acceptance.py --serial <SERIAL> --seconds 30 --cycles 10
```
Read the printed report. This is the authoritative qualification — see the
**hardware-acceptance** skill for interpreting each PASS/FAIL/INFO line and tuning
thresholds/exposure. The `--cycles 10` adds a connect/stream/teardown churn test (the
automated counterpart to unplug/replug; validates the gc/atexit release path).

## Step 3 — image quality (human eyeball; Claude can't see the window)
```bash
python app.py --backend spinnaker --serial <SERIAL>
```
A window `bfs16s2c (spinnaker)` opens with live ~30 fps. Ask the user to confirm:
- **Focus** sharpens (→ the 5 mm C-to-CS adapter is fitted).
- **Color** looks natural (→ debayer correct). A slight cast is expected with
  raw-Bayer + host debayer (on-camera AWB is off); a *strong* pink/green tint means a
  Bayer/pixel-format mismatch.
- Dark/blown-out → tune `exposure_us` in `config/bfs_u3_16s2c.json` (must be ≤ 1/fps;
  ≤ 16600 µs at 60 fps) and rerun.
The HUD shows GUI/CAM fps, temperature, exposure/gain; the status bar shows frame id,
age, dropped, malformed, reconnects.

## Step 4 — reconnect under fault (NFR-005; physical, user does the unplug)
While step 2 or 3 is running, have the user **unplug the USB cable**, wait a few
seconds, then **replug**. Expected in the logs:
`Backend fault … → Reconnect attempt N → Reconnected`. The driver survives the
hot-unplug (skips native EndAcquisition/DeInit on the dead handle to avoid a SIGSEGV)
and rebinds by serial after the USB re-enumeration. If it does NOT recover, check the
camera is bound by serial (not device_index).

## Interpreting / iterating
- Capture the full console output of steps 1–2 and summarize PASS/FAIL for the user.
- On a FAIL, diagnose against the acceptance check meaning (hardware-acceptance skill),
  propose a config/threshold change, and rerun — but never weaken a threshold just to
  pass; flag it.
- Any code change to fix a real-hardware issue must stay **opt-in / default-preserving**
  (see CLAUDE.md). Re-run the software gate (qa-gate skill) after any code change.
- Commit/push only when the user asks; sole author Alexis, no Claude trailers.
