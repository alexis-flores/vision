"""
main.py
Headless end-to-end demo of the full SRS dataflow:

    config file (FR-001)
        -> CameraDriver (Spinnaker / OpenCV / Generic)
        -> CameraService worker thread (FR-002, NFR-005/006)
        -> CircularFrameBuffer
        -> VisionSystem  (FR-003 centroid extraction, FR-004 GUI callback)
        -> CentroidRingBuffer  (SRS 5.2, NFR-007)
        -> QueuingSubsystem    (downstream consumer)

The GUI path (FIFO -> PyQt) lives in gui_bridge.py.

Run:  python main.py
"""

from __future__ import annotations

import logging
import os
import time

from vision.camera_service import CameraService
from vision.camera_types import CameraFrame
from vision.centroid_buffer import CentroidRingBuffer
from vision.centroid_extraction import CentroidExtractor, ExtractorParams
from vision.centroid_types import CentroidProfile
from vision.frame_buffers import CircularFrameBuffer, FIFOFrameBuffer
from vision.generic_driver import GenericCameraDriver
from vision.queuing_subsystem import QueuingSubsystem
from vision.vision_system import VisionSystem

# SRS 5.5: INFO / WARN / ERROR via the std logging library.
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)-18s %(levelname)-5s %(message)s")
log = logging.getLogger("main")

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config", "camera.json")


def driver_factory(cfg):
    """Choose a backend per config. Swap in Spinnaker/OpenCV on real HW."""
    # from vision.spinnaker_driver import SpinnakerCameraDriver
    # return SpinnakerCameraDriver(cfg)
    return GenericCameraDriver(cfg, n_spots=6)


def main() -> None:
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Service + camera (FR-001)
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    svc = CameraService()
    names = svc.add_cameras_from_config(CONFIG_PATH, driver_factory)
    cam = names[0]

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Buffers
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    vision_ring = CircularFrameBuffer(capacity=64) # service -> vision
    gui_fifo = FIFOFrameBuffer(capacity=4) # service -> GUI
    centroid_ring = CentroidRingBuffer(n_max_samples=256) # vision -> queuing
    svc.attach_sink(cam, vision_ring)
    svc.attach_sink(cam, gui_fifo)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Vision system (FR-003/004)
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    extractor = CentroidExtractor(ExtractorParams(
        threshold=60, min_area=6, max_centroids=64, use_gpu=False))

    gui_frames = {"count": 0}
    def gui_callback(frame: CameraFrame, profile: CentroidProfile) -> None:
        # NFR-008: must never block; here it just counts.
        gui_frames["count"] += 1

    vision = VisionSystem(
        frame_buffer=vision_ring,
        centroid_ring=centroid_ring,
        extractor=extractor,
        gui_callback=gui_callback,
        latency_budget_us=5000.0, # NFR-002
    )

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Queuing subsystem (downstream)
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    last = {"profile": None}
    queuing = QueuingSubsystem(
        centroid_ring, sink=lambda p: last.__setitem__("profile", p))

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Connect + run
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    svc.connect(cam)
    queuing.start()
    vision.start()
    svc.start_streaming(cam)

    time.sleep(1.5)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   FR-002/FR-003 evidence
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    p = last["profile"]
    if p is not None:
        log.info("Latest profile: seq=%d frame=%d centroids=%d latency=%.0fus",
                 p.seq_id, p.frame_id, p.n_centroids, p.proc_latency_us)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   NFR-006 evidence: inject malformed frames, expect skip+continue
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    log.info(">>> Injecting 3 malformed frames (NFR-006)")
    svc._entry(cam).driver.inject_malformed(3) # demo only
    time.sleep(0.5)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   NFR-005 evidence: inject backend crash, expect reconnect
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    log.info(">>> Injecting backend crash (NFR-005)")
    svc._entry(cam).driver.inject_backend_crash() # demo only
    time.sleep(1.5)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Report
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    print("\n================ RUN SUMMARY ================")
    print("service stats :", svc.stats(cam))
    print("vision        : processed=%d skipped=%d backend=%s "
          "last_latency=%.0fus over_budget=%d"
          % (vision.frames_processed, vision.frames_skipped,
             getattr(extractor, "backend", "custom"), vision.last_latency_us,
             vision.latency_budget_exceeded))
    print("queuing       : profiles=%d centroids=%d"
          % (queuing.profiles_consumed, queuing.centroids_consumed))
    print("centroid ring : written=%d read=%d overwritten=%d"
          % (centroid_ring.n_written, centroid_ring.n_read,
             centroid_ring.n_overwritten))
    print("gui callbacks :", gui_frames["count"])
    print("frame ring drp:", vision_ring.dropped,
          "| gui fifo drp:", gui_fifo.dropped)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Shutdown (5.3)
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    svc.stop_streaming(cam)
    vision.stop()
    queuing.stop()
    svc.shutdown()
    print("final status  :", svc.get_status(cam).name)


if __name__ == "__main__":
    main()
