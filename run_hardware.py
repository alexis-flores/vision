"""
run_hardware.py
Drive a REAL BlackFly S (Spinnaker) through the full SRS v0.2 pipeline:

    Spinnaker camera
        -> CameraService worker (FR-002, NFR-005 reconnect, NFR-006 skip)
        -> CircularFrameBuffer  --->  CueingSystem  (downstream consumer)
        -> FIFOFrameBuffer      --->  PyQt viewer   (FR-004 / NFR-008)

Scope (SRS v0.2): the vision system serves frames; centroid extraction /
processing belongs to the cueing system (out of scope here — the CueingSystem
just ingests frames). Unlike main.py (simulated camera + fault injection), this
targets actual hardware: no inject_*() calls. To exercise NFR-005 reconnect on
real gear, just unplug/replug the USB cable while it runs.

Usage:
    python run_hardware.py                              # live PyQt viewer
    python run_hardware.py --headless --seconds 10      # no GUI, log stats
    python run_hardware.py --config config/bfs_u3_16s2c.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from vision.camera_service import CameraService
from vision.cueing_system import CueingSystem
from vision.frame_buffers import CircularFrameBuffer, FIFOFrameBuffer
from vision.spinnaker_driver import SpinnakerCameraDriver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-18s %(levelname)-5s %(message)s")
log = logging.getLogger("run_hardware")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "config", "bfs_u3_16s2c.json")


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the vision pipeline on a "
                                 "real BlackFly S (Spinnaker).")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="camera config JSON (default: BFS-U3-16S2C-CS)")
    ap.add_argument("--serial", default=None,
                    help="bind to this camera serial (overrides the config); "
                         "recommended for reliable NFR-005 reconnect")
    ap.add_argument("--headless", action="store_true",
                    help="no GUI; stream and log stats for --seconds")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="headless run duration (s)")
    return ap.parse_args(argv)


def _run_gui(gui_fifo: FIFOFrameBuffer, title: str) -> None:
    from PyQt6.QtWidgets import QApplication  # imported only when needed
    from gui_bridge import CameraViewer
    app = QApplication(sys.argv)
    win = CameraViewer(gui_fifo, title=f"{title} - live")
    win.show()
    log.info("Close the window to stop.")
    app.exec()


def _run_headless(seconds: float, svc: CameraService, cam: str,
                  cueing: CueingSystem) -> None:
    log.info("Headless run for %.1fs (Ctrl-C to stop early).", seconds)
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end:
            time.sleep(1.0)
            st = svc.stats(cam)
            log.info("delivered=%d malformed=%d reconnects=%d | "
                     "cueing: consumed=%d errored=%d",
                     st["frames_delivered"], st["malformed_frames"],
                     st["reconnects"], cueing.frames_consumed,
                     cueing.frames_errored)
    except KeyboardInterrupt:
        log.info("Interrupted; shutting down.")


def main(argv=None) -> None:
    args = _parse_args(argv)

    def make_driver(cfg):
        if args.serial:                 # CLI override binds to a specific unit
            cfg.serial = args.serial
        return SpinnakerCameraDriver(cfg)

    svc = CameraService()
    names = svc.add_cameras_from_config(args.config, make_driver)
    cam = names[0]

    cueing_ring = CircularFrameBuffer(capacity=64)  # service -> cueing
    gui_fifo = FIFOFrameBuffer(capacity=4)          # service -> GUI
    svc.attach_sink(cam, cueing_ring)
    if not args.headless:
        svc.attach_sink(cam, gui_fifo)

    cueing = CueingSystem(cueing_ring)  # ingests frames; pipeline out of scope

    svc.connect(cam)
    cueing.start()
    svc.start_streaming(cam)
    log.info("Streaming %r", cam)

    try:
        if args.headless:
            _run_headless(args.seconds, svc, cam, cueing)
        else:
            _run_gui(gui_fifo, cam)
    finally:
        svc.stop_streaming(cam)
        cueing.stop()
        svc.shutdown()
        st = svc.stats(cam)
        log.info("Final status: %s | delivered=%d malformed=%d | "
                 "cueing consumed=%d errored=%d",
                 svc.get_status(cam).name, st["frames_delivered"],
                 st["malformed_frames"], cueing.frames_consumed,
                 cueing.frames_errored)


if __name__ == "__main__":
    main()
