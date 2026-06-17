"""
run_hardware.py
Drive a REAL BlackFly S (Spinnaker) through the full SRS pipeline:

    Spinnaker camera
        -> CameraService worker (FR-002, NFR-005 reconnect, NFR-006 skip)
        -> CircularFrameBuffer
        -> VisionSystem      (FR-003 centroid extraction)
        -> CentroidRingBuffer
        -> QueuingSubsystem  (downstream consumer)
        (+ annotated frames -> FIFO -> PyQt viewer, FR-004 / NFR-008)

Unlike main.py (simulated camera + fault injection), this targets actual
hardware: no inject_*() calls. To exercise NFR-005 reconnect on real gear,
just unplug/replug the USB cable while it runs.

Usage:
    python run_hardware.py                              # live PyQt viewer
    python run_hardware.py --headless --seconds 10      # no GUI, log stats
    python run_hardware.py --config config/bfs_u3_16s2c.json --threshold 100
    python run_hardware.py --min-area 8                 # tune blob filtering

Threshold/min-area usually need tuning for your scene + exposure; watch the
logged centroid counts (or the GUI markers) and adjust.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from vision.camera_service import CameraService
from vision.centroid_buffer import CentroidRingBuffer
from vision.centroid_extraction import CentroidExtractor, ExtractorParams
from vision.frame_buffers import CircularFrameBuffer, FIFOFrameBuffer
from vision.queuing_subsystem import QueuingSubsystem
from vision.spinnaker_driver import SpinnakerCameraDriver
from vision.vision_system import VisionSystem

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
    ap.add_argument("--threshold", type=int, default=128,
                    help="binarization threshold 0..255, <0 = Otsu")
    ap.add_argument("--min-area", type=int, default=10,
                    help="reject blobs smaller than this many pixels")
    ap.add_argument("--headless", action="store_true",
                    help="no GUI; stream and log stats for --seconds")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="headless run duration (s)")
    return ap.parse_args(argv)


def _gui_callback(gui_fifo: FIFOFrameBuffer):
    # Non-blocking hand-off of annotated frames to the viewer (NFR-008).
    return lambda frame, profile: gui_fifo.push(frame)


def _run_gui(gui_fifo: FIFOFrameBuffer, title: str) -> None:
    from PyQt6.QtWidgets import QApplication # imported only when needed
    from gui_bridge import CameraViewer
    app = QApplication(sys.argv)
    win = CameraViewer(gui_fifo, title=f"{title} - centroids")
    win.show()
    log.info("Close the window to stop.")
    app.exec()


def _run_headless(seconds: float, vision: VisionSystem,
                  queuing: QueuingSubsystem) -> None:
    log.info("Headless run for %.1fs (Ctrl-C to stop early).", seconds)
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end:
            time.sleep(1.0)
            log.info("processed=%d skipped=%d over_budget=%d last=%.0fus | "
                     "queue: profiles=%d centroids=%d",
                     vision.frames_processed, vision.frames_skipped,
                     vision.latency_budget_exceeded, vision.last_latency_us,
                     queuing.profiles_consumed, queuing.centroids_consumed)
    except KeyboardInterrupt:
        log.info("Interrupted; shutting down.")


def main(argv=None) -> None:
    args = _parse_args(argv)

    svc = CameraService()
    names = svc.add_cameras_from_config(
        args.config, lambda cfg: SpinnakerCameraDriver(cfg))
    cam = names[0]

    vision_ring = CircularFrameBuffer(capacity=64) # service -> vision
    gui_fifo = FIFOFrameBuffer(capacity=4) # vision -> GUI
    centroid_ring = CentroidRingBuffer(n_max_samples=256) # vision -> queuing
    svc.attach_sink(cam, vision_ring)

    extractor = CentroidExtractor(ExtractorParams(
        threshold=args.threshold, min_area=args.min_area, max_centroids=64))

    vision = VisionSystem(
        frame_buffer=vision_ring,
        centroid_ring=centroid_ring,
        extractor=extractor,
        gui_callback=None if args.headless else _gui_callback(gui_fifo),
        latency_budget_us=5000.0, # NFR-002
    )
    queuing = QueuingSubsystem(centroid_ring)

    svc.connect(cam)
    queuing.start()
    vision.start()
    svc.start_streaming(cam)
    log.info("Streaming %r (extractor backend=%s)", cam,
             getattr(extractor, "backend", "custom"))

    try:
        if args.headless:
            _run_headless(args.seconds, vision, queuing)
        else:
            _run_gui(gui_fifo, cam)
    finally:
        svc.stop_streaming(cam)
        vision.stop()
        queuing.stop()
        svc.shutdown()
        log.info("Final status: %s | processed=%d skipped=%d | "
                 "queue profiles=%d centroids=%d",
                 svc.get_status(cam).name, vision.frames_processed,
                 vision.frames_skipped, queuing.profiles_consumed,
                 queuing.centroids_consumed)


if __name__ == "__main__":
    main()
