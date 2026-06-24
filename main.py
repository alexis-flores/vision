"""
main.py
Headless end-to-end demo of the SRS v0.2 dataflow:

    config file (FR-001)
        -> CameraDriver (Spinnaker / OpenCV / Generic)
        -> CameraService worker thread (FR-002, NFR-005/006)
        -> CircularFrameBuffer  --->  CueingSystem  (downstream consumer)
        -> FIFOFrameBuffer      --->  GUI           (gui_bridge.py)

Scope change (SRS v0.2): image processing / centroid extraction moved out of the
vision system and into the cueing system. The vision system now only configures
the camera, streams, and serves frames; the cueing system consumes them. The
cueing processing pipeline itself is out of scope here (no spec yet) — this demo
plugs a trivial counting `frame_processor` in where that pipeline would live.

The GUI path (FIFO -> PyQt) lives in gui_bridge.py.

Run:  python main.py
"""

from __future__ import annotations

import logging
import os
import time

from vision.camera_service import CameraService
from vision.camera_types import CameraFrame
from vision.cueing_system import CueingSystem
from vision.frame_buffers import CircularFrameBuffer, FIFOFrameBuffer
from vision.generic_driver import GenericCameraDriver

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
    #   Buffers (service fans frames out to both)
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    cueing_ring = CircularFrameBuffer(capacity=64)  # service -> cueing
    gui_fifo = FIFOFrameBuffer(capacity=4)          # service -> GUI
    svc.attach_sink(cam, cueing_ring)
    svc.attach_sink(cam, gui_fifo)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Cueing system (downstream consumer of frames)
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    # Stand-in for the (out-of-scope) cueing pipeline: count frames + remember
    # the last one. The real pixel->angular cueing logic plugs in here.
    seen = {"last": None}

    def frame_processor(frame: CameraFrame) -> None:
        seen["last"] = frame

    cueing = CueingSystem(cueing_ring, frame_processor=frame_processor)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Connect + run
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    svc.connect(cam)
    cueing.start()
    svc.start_streaming(cam)

    time.sleep(1.5)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   FR-002 evidence (frames served to the cueing system)
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    f = seen["last"]
    if f is not None:
        log.info("Latest frame to cueing: id=%d shape=%s age=%.1fms",
                 f.frame_id, f.data.shape, f.age * 1000)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   NFR-006 evidence: inject malformed frames, expect skip+continue
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    log.info(">>> Injecting 3 malformed frames (NFR-006)")
    svc._entry(cam).driver.inject_malformed(3)  # demo only
    time.sleep(0.5)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   NFR-005 evidence: inject backend crash, expect reconnect
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    log.info(">>> Injecting backend crash (NFR-005)")
    svc._entry(cam).driver.inject_backend_crash()  # demo only
    time.sleep(1.5)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Report
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    print("\n================ RUN SUMMARY ================")
    print("service stats :", svc.stats(cam))
    print("cueing        : consumed=%d errored=%d last_frame=%s"
          % (cueing.frames_consumed, cueing.frames_errored,
             cueing.last_frame_id))
    print("cueing ring   : dropped=%d | gui fifo dropped=%d"
          % (cueing_ring.dropped, gui_fifo.dropped))

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Shutdown (5.3)
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    svc.stop_streaming(cam)
    cueing.stop()
    svc.shutdown()
    print("final status  :", svc.get_status(cam).name)


if __name__ == "__main__":
    main()
