r"""
app.py
Single entry point for the vision system. Runs any camera backend — the
simulator, an OpenCV/UVC device, or a BlackFly S via Spinnaker — with a live
PyQt viewer or headless, through the full v0.2 dataflow:

    CameraDriver -> CameraService -> CircularFrameBuffer -> CueingSystem
                                  \-> FIFOFrameBuffer    -> CameraViewer (GUI)

The backend is chosen at runtime; PySpin is imported lazily only when
--backend spinnaker is selected, so this script runs on machines without the
Spinnaker SDK. The PyQt viewer (CameraViewer) lives in gui_bridge.py and is
reused as-is for every backend.

    python app.py                                   # simulator + live viewer
    python app.py --backend spinnaker --serial 215  # real BlackFly S + viewer
    python app.py --backend opencv --device 0       # webcam + viewer
    python app.py --headless --seconds 10           # any backend, no GUI
    python app.py --headless --inject-faults        # sim NFR-005/006 demo
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional

from vision.camera_service import CameraService
from vision.camera_types import CameraConfig, CameraFeature
from vision.cueing_system import CueingSystem
from vision.frame_buffers import CircularFrameBuffer, FIFOFrameBuffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-18s %(levelname)-5s %(message)s")
log = logging.getLogger("app")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPINNAKER_CONFIG = os.path.join(HERE, "config", "bfs_u3_16s2c.json")
BACKENDS = ("sim", "opencv", "spinnaker")


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Vision system runner — any camera backend, GUI or headless.")
    ap.add_argument("--backend", choices=BACKENDS, default="sim",
                    help="camera backend (default: sim)")
    ap.add_argument("--config", default=None,
                    help="camera config JSON (default: built-in per backend; "
                         "the BFS config for spinnaker)")
    ap.add_argument("--serial", default=None,
                    help="spinnaker: bind to this camera serial (recommended)")
    ap.add_argument("--device", type=int, default=0,
                    help="opencv: capture device index (default: 0)")
    ap.add_argument("--exposure", type=float, default=None,
                    help="exposure time in microseconds (overrides config)")
    ap.add_argument("--gain", type=float, default=None,
                    help="gain in dB (overrides config)")
    ap.add_argument("--fps", type=float, default=None,
                    help="frame rate (overrides config)")
    ap.add_argument("--headless", action="store_true",
                    help="no GUI; stream and log stats for --seconds")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="headless run duration (s)")
    ap.add_argument("--inject-faults", action="store_true",
                    help="sim only: inject malformed frames + a backend crash "
                         "to demo NFR-006 skip and NFR-005 reconnect")
    ap.add_argument("--no-cueing", action="store_true",
                    help="don't start the cueing consumer; only serve frames to "
                         "the GUI (FIFO) — display-only acquisition")
    ap.add_argument("--reset", action="store_true",
                    help="spinnaker only: load the factory Default user set and "
                         "make it the power-on default, then exit (resets the "
                         "camera, including across power-cycles)")
    return ap.parse_args(argv)


def _make_driver(args: argparse.Namespace, cfg: CameraConfig):
    """Construct the driver for the chosen backend, applying CLI overrides
    (None = keep the config value). PySpin is imported only here, only when
    needed, so the SDK stays optional."""
    if args.serial:
        cfg.serial = args.serial
    if args.exposure is not None:
        cfg.exposure_us = args.exposure
    if args.gain is not None:
        cfg.gain_db = args.gain
    if args.fps is not None:
        cfg.fps = args.fps
    if args.backend == "sim":
        from vision.generic_driver import GenericCameraDriver
        return GenericCameraDriver(cfg, n_spots=6)
    if args.backend == "opencv":
        from vision.opencv_driver import OpenCVCameraDriver
        return OpenCVCameraDriver(cfg)
    from vision.spinnaker_driver import SpinnakerCameraDriver
    return SpinnakerCameraDriver(cfg)


def _default_config(backend: str, device: int) -> CameraConfig:
    """A sensible built-in config when --config is not given (sim / opencv)."""
    if backend == "opencv":
        return CameraConfig(
            name="webcam", device_index=device, max_resolution=(640, 480),
            resolution=(640, 480), fps=30.0,
            features=CameraFeature.RESOLUTION | CameraFeature.FRAME_RATE)
    return CameraConfig(
        name="sim0", model="SimCam", max_resolution=(640, 480),
        max_fps=60.0, fps=60.0,
        features=CameraFeature.FRAME_RATE | CameraFeature.RESOLUTION)


def _build_service(args: argparse.Namespace):
    """Register one camera on a fresh service; returns (service, name)."""
    svc = CameraService()
    config = args.config
    if config is None and args.backend == "spinnaker":
        config = DEFAULT_SPINNAKER_CONFIG  # spinnaker needs a real device config
    if config:
        names = svc.add_cameras_from_config(
            config, lambda c: _make_driver(args, c))
        return svc, names[0]
    cfg = _default_config(args.backend, args.device)
    svc.add_camera(cfg.name, _make_driver(args, cfg))
    return svc, cfg.name


def _run_gui(gui_fifo: FIFOFrameBuffer, title: str, svc: CameraService,
             cam: str) -> None:
    from PyQt6.QtWidgets import QApplication  # imported only when GUI is used
    from gui_bridge import CameraViewer
    app = QApplication(sys.argv)
    win = CameraViewer(gui_fifo, title=title,
                       health_fn=lambda: svc.get_health(cam),
                       stats_fn=lambda: svc.stats(cam))
    win.show()
    log.info("Close the window to stop.")
    app.exec()


def _run_headless(args: argparse.Namespace, svc: CameraService, cam: str,
                  cueing: Optional[CueingSystem]) -> None:
    log.info("Headless run for %.1fs (Ctrl-C to stop early).", args.seconds)
    inject = args.inject_faults and args.backend == "sim"
    t_end = time.monotonic() + args.seconds
    injected = False
    try:
        while time.monotonic() < t_end:
            time.sleep(1.0)
            st = svc.stats(cam)
            cue = (f"{cueing.frames_consumed} consumed, {cueing.frames_errored} "
                   f"errored" if cueing is not None else "off")
            log.info("delivered=%d malformed=%d reconnects=%d | cueing %s",
                     st["frames_delivered"], st["malformed_frames"],
                     st["reconnects"], cue)
            if inject and not injected:
                injected = True
                log.info(">>> Injecting 3 malformed frames (NFR-006)")
                svc._entry(cam).driver.inject_malformed(3)
                time.sleep(0.5)  # let them be skipped before the crash clears faults
                log.info(">>> Injecting backend crash (NFR-005)")
                svc._entry(cam).driver.inject_backend_crash()
    except KeyboardInterrupt:
        log.info("Interrupted; shutting down.")


def _run_reset(args: argparse.Namespace) -> int:
    """Connect, load the factory Default user set, and exit (no streaming)."""
    if args.backend != "spinnaker":
        log.error("--reset is only supported for --backend spinnaker")
        return 1
    svc, cam = _build_service(args)
    try:
        svc.connect(cam)            # config is applied then overwritten by reset
        svc.reset_to_defaults(cam)
        log.info("Camera %r reset to factory defaults (and set as power-on "
                 "default — power-cycles will reset too).", cam)
        return 0
    except Exception as e:
        log.error("Reset failed: %s", e)
        return 1
    finally:
        svc.shutdown()


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.reset:
        return _run_reset(args)
    if args.inject_faults and args.backend != "sim":
        log.warning("--inject-faults is sim-only; ignoring for backend %r",
                    args.backend)

    svc, cam = _build_service(args)
    gui_fifo = FIFOFrameBuffer(capacity=4)           # service -> GUI

    # Cueing and the GUI are independent fan-out branches; either can be omitted.
    cueing = None
    if not args.no_cueing:
        cueing_ring = CircularFrameBuffer(capacity=64)   # service -> cueing
        svc.attach_sink(cam, cueing_ring)
        cueing = CueingSystem(cueing_ring)               # ingests frames; pipeline TBD
    if not args.headless:
        svc.attach_sink(cam, gui_fifo)                   # service -> GUI
    if args.no_cueing and args.headless:
        log.warning("--no-cueing with --headless: no consumer attached; "
                    "frames will just be acquired and dropped.")

    svc.connect(cam)
    if cueing is not None:
        cueing.start()
    svc.start_streaming(cam)
    log.info("Streaming %r on backend=%s (cueing=%s)", cam, args.backend,
             "off" if cueing is None else "on")

    try:
        if args.headless:
            _run_headless(args, svc, cam, cueing)
        else:
            _run_gui(gui_fifo, f"{cam} ({args.backend})", svc, cam)
    finally:
        svc.stop_streaming(cam)
        if cueing is not None:
            cueing.stop()
        svc.shutdown()
        st = svc.stats(cam)
        consumed = "off" if cueing is None else cueing.frames_consumed
        log.info("Final: status=%s delivered=%d malformed=%d reconnects=%d | "
                 "cueing consumed=%s", svc.get_status(cam).name,
                 st["frames_delivered"], st["malformed_frames"],
                 st["reconnects"], consumed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
