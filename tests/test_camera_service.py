"""
tests/test_camera_service.py
Service layer (FR-002 streaming, NFR-005 reconnect, NFR-006 skip, multi-camera,
opt-in RT) and the end-to-end vision->cueing->GUI handoff (IT-002, NFR-007/008).
"""

from __future__ import annotations

import os
import time
import unittest

import numpy as np

import _helpers  # noqa: F401  (path bootstrap)
from _helpers import CONFIG_DIR, basic_config, wait_until

from vision.camera_service import CameraService
from vision.camera_types import CameraFrame, CameraStatus
from vision.cueing_system import CueingSystem
from vision.frame_buffers import CircularFrameBuffer, FIFOFrameBuffer
from vision.generic_driver import GenericCameraDriver


class TestCameraService(unittest.TestCase):
    def setUp(self):
        self.svc = CameraService()
        self.svc.MAX_RECONNECT_ATTEMPTS = 5
        self.svc.RECONNECT_DELAY_S = 0.05
        self.drv = GenericCameraDriver(basic_config("svccam"), n_spots=4)
        self.svc.add_camera("svccam", self.drv)

    def tearDown(self):
        self.svc.shutdown()

    def test_fanout_to_two_sinks(self):  # FR-002 / fan-out (cueing + GUI)
        ring = CircularFrameBuffer(32)
        fifo = FIFOFrameBuffer(8)
        self.svc.attach_sink("svccam", ring)
        self.svc.attach_sink("svccam", fifo)
        self.svc.connect("svccam")
        self.svc.start_streaming("svccam")
        wait_until(lambda: self.svc.stats("svccam")["frames_delivered"] > 0)
        self.svc.stop_streaming("svccam")
        self.assertGreater(self.svc.stats("svccam")["frames_delivered"], 0)
        self.assertIsNotNone(ring.pop(timeout=0.2))

    def test_malformed_skip(self):  # NFR-006 / FT-002
        ring = CircularFrameBuffer(32)
        self.svc.attach_sink("svccam", ring)
        self.svc.connect("svccam")
        self.svc.start_streaming("svccam")
        wait_until(lambda: self.svc.stats("svccam")["frames_delivered"] > 0)
        self.drv.inject_malformed(3)
        # Poll until the worker has skipped all 3 (no fixed sleep -> no flake).
        wait_until(lambda: self.svc.stats("svccam")["malformed_frames"] >= 3)
        self.svc.stop_streaming("svccam")
        st = self.svc.stats("svccam")
        self.assertGreaterEqual(st["malformed_frames"], 3)
        # Still delivering frames after the malformed ones
        self.assertGreater(st["frames_delivered"], 0)

    def test_backend_crash_reconnects(self):  # NFR-005 / FT-001
        ring = CircularFrameBuffer(32)
        self.svc.attach_sink("svccam", ring)
        self.svc.connect("svccam")
        self.svc.start_streaming("svccam")
        time.sleep(0.1)
        self.drv.inject_backend_crash()
        # Poll for recovery instead of racing a fixed sleep, so the test stays
        # deterministic under parallel load (NFR-005 has no hard deadline here).
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if (self.svc.get_status("svccam") == CameraStatus.STREAMING
                    and self.svc.stats("svccam")["reconnects"] >= 1):
                break
            time.sleep(0.02)
        self.assertEqual(self.svc.get_status("svccam"), CameraStatus.STREAMING)
        self.assertGreaterEqual(self.svc.stats("svccam")["reconnects"], 1)

    def test_config_file_registration(self):  # FR-001 integration
        path = os.path.join(CONFIG_DIR, "camera.json")
        svc = CameraService()
        names = svc.add_cameras_from_config(
            path, lambda c: GenericCameraDriver(c))
        self.assertIn("bfly0", names)

    def test_two_cameras_stream_concurrently(self):  # multi-camera service
        svc = CameraService()
        for n in ("camA", "camB"):
            svc.add_camera(n, GenericCameraDriver(
                basic_config(n, resolution=(160, 120),
                             max_resolution=(160, 120)), n_spots=3))
        sink_a, sink_b = FIFOFrameBuffer(8), FIFOFrameBuffer(8)
        svc.attach_sink("camA", sink_a)
        svc.attach_sink("camB", sink_b)
        svc.connect_all()
        svc.start_streaming("camA")
        svc.start_streaming("camB")
        try:
            wait_until(lambda: svc.stats("camA")["frames_delivered"] > 0
                       and svc.stats("camB")["frames_delivered"] > 0,
                       timeout=3.0)
            self.assertGreater(svc.stats("camA")["frames_delivered"], 0)
            self.assertGreater(svc.stats("camB")["frames_delivered"], 0)
            self.assertEqual(set(svc.camera_names()), {"camA", "camB"})
        finally:
            svc.shutdown()
        self.assertEqual(svc.get_status("camA"), CameraStatus.DISCONNECTED)
        self.assertEqual(svc.get_status("camB"), CameraStatus.DISCONNECTED)

    def test_rt_mode_streams_or_degrades(self):  # opt-in SCHED_FIFO, best-effort
        # RT mode requests real-time priority on the worker; where it isn't
        # supported / privileged it logs and still streams normally.
        svc = CameraService(rt=True)
        svc.add_camera("rt", GenericCameraDriver(basic_config("rt"), n_spots=3))
        svc.attach_sink("rt", CircularFrameBuffer(16))
        svc.connect("rt")
        svc.start_streaming("rt")
        try:
            wait_until(lambda: svc.stats("rt")["frames_delivered"] > 0,
                       timeout=3.0)
            self.assertGreater(svc.stats("rt")["frames_delivered"], 0)
        finally:
            svc.shutdown()

    def test_stop_streaming_idempotent_quiet(self):  # cosmetic double-log fix
        self.svc.connect("svccam")
        self.svc.start_streaming("svccam")
        wait_until(lambda: self.svc.stats("svccam")["frames_delivered"] > 0)
        with self.assertLogs("vision.camera_service", level="INFO") as cm:
            self.svc.stop_streaming("svccam")   # real stop -> logs once
            self.svc.stop_streaming("svccam")   # no-op -> must NOT log again
        stops = [m for m in cm.output if "Streaming stopped" in m]
        self.assertEqual(len(stops), 1)


class TestCueingEndToEnd(unittest.TestCase):  # IT-002
    def test_frames_served_to_cueing_and_gui(self):
        svc = CameraService()
        drv = GenericCameraDriver(basic_config("e2e"), n_spots=5)
        svc.add_camera("e2e", drv)
        cueing_ring = CircularFrameBuffer(32)
        gui_fifo = FIFOFrameBuffer(8)
        svc.attach_sink("e2e", cueing_ring)   # -> cueing
        svc.attach_sink("e2e", gui_fifo)      # -> GUI (FR-004 / NFR-008 path)

        seen = []
        cueing = CueingSystem(
            cueing_ring, frame_processor=lambda f: seen.append(f.frame_id))

        svc.connect("e2e")
        cueing.start(); svc.start_streaming("e2e")
        wait_until(lambda: cueing.frames_consumed > 0 and len(seen) > 0)
        svc.stop_streaming("e2e"); cueing.stop(); svc.shutdown()

        self.assertGreater(svc.stats("e2e")["frames_delivered"], 0)
        # Cueing consumed the frames the vision system served.
        self.assertGreater(cueing.frames_consumed, 0)
        self.assertEqual(cueing.frames_errored, 0)
        self.assertGreater(len(seen), 0)
        self.assertEqual(cueing.last_frame_id, seen[-1])
        # GUI path: frames reached the FIFO too (FR-004 / NFR-008).
        self.assertIsNotNone(gui_fifo.pop(timeout=0.2))

    def test_processor_error_skipped(self):  # NFR-006 inside cueing
        ring = CircularFrameBuffer(8)

        def boom(_frame):
            raise RuntimeError("processor blew up")

        cueing = CueingSystem(ring, frame_processor=boom)
        ring.push(CameraFrame(data=np.zeros((4, 4), np.uint8),
                              timestamp=time.monotonic(), frame_id=1))
        cueing.start()
        wait_until(lambda: cueing.frames_consumed >= 1
                   and cueing.frames_errored >= 1)
        cueing.stop()
        self.assertGreaterEqual(cueing.frames_consumed, 1)
        self.assertGreaterEqual(cueing.frames_errored, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
