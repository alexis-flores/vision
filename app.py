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
    python app.py --config multi.json               # several cameras at once

A config with a top-level "cameras" list runs every camera concurrently — each
on its own service worker thread with its own cueing + GUI fan-out branch. A
single-camera setup is just the N=1 case (unchanged behavior).
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, List, Optional, cast

from vision.camera_driver import CameraError
from vision.camera_service import CameraService
from vision.camera_types import CameraConfig, CameraFeature
from vision.config_loader import ConfigError, load_camera_configs
from vision.cueing_system import CueingSystem
from vision.frame_buffers import CircularFrameBuffer, FIFOFrameBuffer

if TYPE_CHECKING:  # type-only: the sim fault hooks live on GenericCameraDriver
    from vision.generic_driver import GenericCameraDriver

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
    ap.add_argument("--no-display-wb", action="store_true",
                    help="disable the GUI's display-only white balance (preview "
                         "shows raw pixels; the cueing path is unaffected either "
                         "way, since correction is applied only for the display)")
    ap.add_argument("--reset", action="store_true",
                    help="spinnaker only: load the factory Default user set and "
                         "make it the power-on default, then exit (resets the "
                         "camera, including across power-cycles)")
    ap.add_argument("--rt", action="store_true",
                    help="real-time mode: freeze + disable GC during the run and "
                         "raise the camera worker(s) to SCHED_FIFO (Linux, needs "
                         "root) for low-jitter frame timing")
    return ap.parse_args(argv)


def _has_overrides(args: argparse.Namespace) -> bool:
    return (bool(args.serial) or args.exposure is not None
            or args.gain is not None or args.fps is not None)


def _apply_overrides(args: argparse.Namespace, cfg: CameraConfig) -> None:
    """Apply CLI overrides onto a config in place (None = keep config value)."""
    if args.serial:
        cfg.serial = args.serial
    if args.exposure is not None:
        cfg.exposure_us = args.exposure
    if args.gain is not None:
        cfg.gain_db = args.gain
    if args.fps is not None:
        cfg.fps = args.fps


def _make_driver_for_backend(args: argparse.Namespace, cfg: CameraConfig):
    """Construct the driver for the chosen backend (backend selection only;
    CLI overrides are applied separately by _apply_overrides). PySpin is
    imported only here, only for spinnaker, so the SDK stays optional."""
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
    """Register all configured cameras on a fresh service; returns (svc, names).

    A config with a top-level "cameras" list yields multiple cameras, each on
    its own worker thread. CLI overrides (--serial/--exposure/--gain/--fps) are
    applied only to a single-camera setup; for a multi-camera config each camera
    keeps its own config values (one --serial cannot bind N physical units)."""
    svc = CameraService(rt=args.rt)
    config = args.config
    if config is None and args.backend == "spinnaker":
        config = DEFAULT_SPINNAKER_CONFIG  # spinnaker needs a real device config
    if config:
        cfgs = load_camera_configs(config)
        single = len(cfgs) == 1
        if not single and _has_overrides(args):
            log.warning("--serial/--exposure/--gain/--fps ignored for a "
                        "multi-camera config (%d cameras); each camera uses "
                        "its own config values.", len(cfgs))
        names = []
        for c in cfgs:
            if single:
                _apply_overrides(args, c)
            svc.add_camera(c.name, _make_driver_for_backend(args, c))
            names.append(c.name)
        return svc, names
    cfg = _default_config(args.backend, args.device)
    _apply_overrides(args, cfg)
    svc.add_camera(cfg.name, _make_driver_for_backend(args, cfg))
    return svc, [cfg.name]


@dataclass
class _CamRun:
    """Per-camera consumer-side wiring for one runner session."""
    name: str
    gui_fifo: FIFOFrameBuffer
    cueing: Optional[CueingSystem]


def _run_gui(runs: List[_CamRun], svc: CameraService, backend: str,
             display_wb: bool = True) -> None:
    import signal

    from PyQt6.QtCore import QTimer            # imported only when GUI is used
    from PyQt6.QtWidgets import QApplication
    from gui_bridge import CameraViewer
    app = QApplication(sys.argv)
    viewers = []  # hold refs so the windows aren't garbage-collected
    for r in runs:
        name = r.name
        # partial binds this camera's name immediately (no late-binding closure)
        # and is properly typed, unlike a default-arg lambda.
        win = CameraViewer(r.gui_fifo, title=f"{name} ({backend})",
                           health_fn=partial(svc.get_health, name),
                           stats_fn=partial(svc.stats, name),
                           display_white_balance=display_wb)
        win.show()
        viewers.append(win)

    # Ctrl-C must quit the Qt loop *cleanly* so the caller's teardown
    # (stop_streaming -> EndAcquisition -> DeInit -> ReleaseInstance) runs in
    # order. Otherwise the interrupt lands inside app.exec()'s C++ loop, the SDK
    # aborts (core dump), and the camera is left mid-acquisition with its config
    # nodes locked (GenICam -2006) on the next connect. A periodic no-op timer
    # hands control back to the interpreter so the pending SIGINT is delivered
    # (Python signal handlers don't run while execution sits in app.exec()).
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    sigint_wake = QTimer()
    sigint_wake.timeout.connect(lambda: None)
    sigint_wake.start(200)

    log.info("Close the window(s) or press Ctrl-C to stop.")
    app.exec()  # one event loop; Qt quits when the last window closes (or Ctrl-C)


def _run_headless(args: argparse.Namespace, svc: CameraService,
                  runs: List[_CamRun]) -> None:
    log.info("Headless run for %.1fs (Ctrl-C to stop early).", args.seconds)
    inject = args.inject_faults and args.backend == "sim"
    t_end = time.monotonic() + args.seconds
    injected = False
    try:
        while time.monotonic() < t_end:
            time.sleep(1.0)
            for r in runs:
                st = svc.stats(r.name)
                cue = (f"{r.cueing.frames_consumed} consumed, "
                       f"{r.cueing.frames_errored} errored"
                       if r.cueing is not None else "off")
                log.info("%s: delivered=%d malformed=%d reconnects=%d | "
                         "cueing %s", r.name, st["frames_delivered"],
                         st["malformed_frames"], st["reconnects"], cue)
            if inject and not injected:
                injected = True
                for r in runs:
                    log.info(">>> [%s] Injecting 3 malformed frames (NFR-006)",
                             r.name)
                    cast("GenericCameraDriver",
                         svc._entry(r.name).driver).inject_malformed(3)
                time.sleep(0.5)  # let them be skipped before the crash
                for r in runs:
                    log.info(">>> [%s] Injecting backend crash (NFR-005)",
                             r.name)
                    cast("GenericCameraDriver",
                         svc._entry(r.name).driver).inject_backend_crash()
    except KeyboardInterrupt:
        log.info("Interrupted; shutting down.")


def _run_reset(args: argparse.Namespace) -> int:
    """Connect, load the factory Default user set, and exit (no streaming)."""
    if args.backend != "spinnaker":
        log.error("--reset is only supported for --backend spinnaker")
        return 1
    svc, names = _build_service(args)
    try:
        for name in names:
            svc.connect(name)       # config is applied then overwritten by reset
            svc.reset_to_defaults(name)
            log.info("Camera %r reset to factory defaults (and set as power-on "
                     "default — power-cycles will reset too).", name)
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

    try:
        svc, names = _build_service(args)
    except (CameraError, ConfigError) as e:
        log.error("Startup failed: %s", e)   # bad/missing config, backend setup
        return 1

    # Per-camera consumer wiring. Cueing and the GUI are independent fan-out
    # branches (either can be omitted); each camera gets its own pair.
    runs: List[_CamRun] = []
    for name in names:
        gui_fifo = FIFOFrameBuffer(capacity=4)               # service -> GUI
        cueing = None
        if not args.no_cueing:
            cueing_ring = CircularFrameBuffer(capacity=64)   # service -> cueing
            svc.attach_sink(name, cueing_ring)
            cueing = CueingSystem(cueing_ring)               # pipeline TBD
        if not args.headless:
            svc.attach_sink(name, gui_fifo)                  # service -> GUI
        runs.append(_CamRun(name=name, gui_fifo=gui_fifo, cueing=cueing))
    if args.no_cueing and args.headless:
        log.warning("--no-cueing with --headless: no consumer attached; "
                    "frames will just be acquired and dropped.")

    if args.rt:
        gc.freeze()    # move setup objects out of GC's scan set -> cheap sweeps
        gc.disable()   # no auto collections during the run (manual at teardown)
        log.info("Real-time mode: GC frozen + disabled; worker(s) -> SCHED_FIFO")

    rc = 0
    try:
        for r in runs:
            svc.connect(r.name)
            if r.cueing is not None:
                r.cueing.start()
            svc.start_streaming(r.name)
        log.info("Streaming %d camera(s) on backend=%s (cueing=%s): %s",
                 len(runs), args.backend, "off" if args.no_cueing else "on",
                 ", ".join(r.name for r in runs))
        if args.headless:
            _run_headless(args, svc, runs)
        else:
            _run_gui(runs, svc, args.backend, display_wb=not args.no_display_wb)
    except (CameraError, ConfigError) as e:
        # Expected operational failure (SDK missing, no camera, connect/stream
        # error): one clean line, no traceback. Unexpected errors still
        # propagate so genuine bugs surface with a full trace.
        log.error("%s", e)
        rc = 1
    except KeyboardInterrupt:
        log.info("Interrupted; shutting down.")
        rc = 130
    finally:
        for r in runs:
            svc.stop_streaming(r.name)
            if r.cueing is not None:
                r.cueing.stop()
        svc.shutdown()
        for r in runs:
            st = svc.stats(r.name)
            consumed = "off" if r.cueing is None else r.cueing.frames_consumed
            log.info("Final %s: status=%s delivered=%d malformed=%d "
                     "reconnects=%d | cueing consumed=%s", r.name,
                     svc.get_status(r.name).name, st["frames_delivered"],
                     st["malformed_frames"], st["reconnects"], consumed)
        if args.rt:                 # restore + manual sweep now the run is over
            gc.enable()
            gc.collect()
    return rc


if __name__ == "__main__":
    sys.exit(main())
