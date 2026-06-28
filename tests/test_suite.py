"""
tests/test_suite.py
NFR-010: unit tests covering normal, boundary, and error cases, mapped to the
SRS 7 Requirements Test Map. Pure stdlib unittest (no pytest needed).

Scope (SRS v0.2): the vision system configures the camera, streams, and serves
frames to the cueing system (CircularFrameBuffer) and GUI (FIFOFrameBuffer).
Centroid extraction / image processing moved to the cueing system and is out of
scope here; the CueingSystem is exercised as a frame consumer.

Run:  python -m unittest discover -s tests
  or: python tests/test_suite.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)  # root scripts (app.py)
sys.path.insert(0, os.path.join(_ROOT, "src"))  # the `vision` package (src layout)

from vision.camera_driver import (CameraError, FeatureNotSupportedError,
                           MalformedFrameError)
from vision.camera_service import CameraService
from vision.camera_types import (CameraConfig, CameraFeature, CameraFrame,
                          CameraStatus, PixelFormat)
from vision.config_loader import ConfigError, load_camera_config, load_camera_configs
from vision.cueing_system import CueingSystem
from vision.frame_buffers import CircularFrameBuffer, FIFOFrameBuffer
from vision.generic_driver import GenericCameraDriver
from vision.lens import (angular_fov_deg, focal_length_mm, linear_fov_mm,
                  sensor_dims_mm, working_distance_mm)


def _wait_until(predicate, timeout=5.0, interval=0.01):
    """Poll until predicate() is true (or timeout), returning its final value.
    Replaces fixed sleeps so threaded tests don't false-fail under CPU load:
    the generous timeout absorbs slow scheduling, while a genuine failure still
    surfaces (the test's own assert runs afterward)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def basic_config(name="cam", **kw):
    defaults = dict(
        name=name, model="sim", max_resolution=(512, 512), max_fps=60.0,
        resolution=(512, 512), fps=60.0,
        features=(CameraFeature.GAIN | CameraFeature.EXPOSURE |
                  CameraFeature.FRAME_RATE | CameraFeature.RESOLUTION))
    defaults.update(kw)
    return CameraConfig(**defaults)


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   FR-001: configuration ingestion
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

class TestConfigLoader(unittest.TestCase):  # UT-001
    def _write(self, obj):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        self.addCleanup(os.remove, path)
        return path

    def test_normal_load(self):
        path = self._write({
            "name": "c0", "model": "m", "max_resolution": [1024, 1024],
            "max_fps": 60.0, "features": ["GAIN", "EXPOSURE"],
            "output_pixel_format": "Mono8", "resolution": [800, 600],
            "fps": 60.0})
        cfg = load_camera_config(path)
        self.assertEqual(cfg.name, "c0")
        self.assertEqual(cfg.resolution, (800, 600))
        self.assertTrue(cfg.supports(CameraFeature.GAIN))
        self.assertEqual(cfg.output_pixel_format, PixelFormat.MONO8)

    def test_multi_camera(self):
        path = self._write({"cameras": [
            {"name": "a", "max_resolution": [512, 512]},
            {"name": "b", "max_resolution": [512, 512]}]})
        cfgs = load_camera_configs(path)
        self.assertEqual([c.name for c in cfgs], ["a", "b"])

    def test_missing_file_errors(self):  # error case
        with self.assertRaises(ConfigError):
            load_camera_config("/no/such/file.json")

    def test_malformed_file_errors(self):  # error case
        fd, path = tempfile.mkstemp(suffix=".json")
        os.write(fd, b"{ not valid json ]")
        os.close(fd)
        self.addCleanup(os.remove, path)
        with self.assertRaises(ConfigError):
            load_camera_config(path)

    def test_unknown_feature_errors(self):  # error case
        path = self._write({"name": "x", "features": ["NOT_A_FEATURE"]})
        with self.assertRaises(ConfigError):
            load_camera_config(path)

    def test_duplicate_camera_name_errors(self):  # error case
        # Two cameras sharing a name would crash registration with a raw
        # ValueError mid-loop; load-time it's a clean ConfigError instead.
        path = self._write({"cameras": [
            {"name": "dup", "max_resolution": [512, 512]},
            {"name": "dup", "max_resolution": [512, 512]}]})
        with self.assertRaises(ConfigError):
            load_camera_configs(path)


class TestConfigValidation(unittest.TestCase):  # NFR-001/003/004
    def test_compliant_has_no_warnings(self):
        cfg = basic_config(max_resolution=(1024, 1024),
                           resolution=(1024, 1024), fps=60.0,
                           lens_fov_deg=60.0)
        self.assertEqual(cfg.validate(), [])

    def test_undersized_resolution_warns(self):  # NFR-003 boundary
        cfg = basic_config(max_resolution=(256, 256), resolution=(256, 256))
        self.assertTrue(any("NFR-003" in w for w in cfg.validate()))

    def test_low_fps_warns(self):  # NFR-001 boundary
        cfg = basic_config(max_fps=30.0, fps=30.0)
        self.assertTrue(any("NFR-001" in w for w in cfg.validate()))

    def test_narrow_fov_warns(self):  # NFR-004 boundary
        cfg = basic_config(lens_fov_deg=20.0)
        self.assertTrue(any("NFR-004" in w for w in cfg.validate()))


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Driver lifecycle / FR-002 / feature gating
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

class TestGenericDriver(unittest.TestCase):  # UT-002
    def setUp(self):
        self.drv = GenericCameraDriver(basic_config(), n_spots=4)

    def tearDown(self):
        self.drv.disconnect()

    def test_lifecycle(self):  # normal
        self.assertEqual(self.drv.get_status(), CameraStatus.DISCONNECTED)
        self.drv.connect()
        self.assertEqual(self.drv.get_status(), CameraStatus.CONNECTED)
        self.drv.start_stream()
        self.assertEqual(self.drv.get_status(), CameraStatus.STREAMING)
        frame = self.drv.read_frame(timeout=1.0)
        self.assertIsInstance(frame, CameraFrame)
        self.assertEqual(frame.data.shape, (512, 512, 3))
        self.drv.stop_stream()
        self.assertEqual(self.drv.get_status(), CameraStatus.CONNECTED)

    def test_read_before_stream_errors(self):  # error
        self.drv.connect()
        with self.assertRaises(CameraError):
            self.drv.read_frame(timeout=0.1)

    def test_frame_buffer_is_read_only(self):  # immutability contract
        # Frames fan out to multiple sinks sharing one ndarray, so the buffer
        # must be read-only: an in-place write fails loudly instead of silently
        # corrupting other consumers.
        self.drv.connect(); self.drv.start_stream()
        frame = self.drv.read_frame(timeout=1.0)
        self.assertFalse(frame.data.flags.writeable)
        with self.assertRaises(ValueError):
            frame.data[0, 0] = 0

    def test_unsupported_feature_errors(self):  # error
        self.drv.connect()
        cfg = basic_config(features=CameraFeature.NONE)
        drv = GenericCameraDriver(cfg)
        drv.connect()
        with self.assertRaises(FeatureNotSupportedError):
            drv.set_config("gain_db", 1.0)
        drv.disconnect()

    def test_malformed_injection(self):  # NFR-006 source
        self.drv.connect(); self.drv.start_stream()
        self.drv.inject_malformed(2)
        with self.assertRaises(MalformedFrameError):
            self.drv.read_frame(timeout=1.0)
        with self.assertRaises(MalformedFrameError):
            self.drv.read_frame(timeout=1.0)
        # Recovers afterwards
        self.assertIsInstance(self.drv.read_frame(timeout=1.0), CameraFrame)

    def test_context_manager(self):
        with GenericCameraDriver(basic_config()) as drv:
            self.assertEqual(drv.get_status(), CameraStatus.CONNECTED)


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Frame buffers
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

class TestFrameBuffers(unittest.TestCase):
    def _frame(self, fid):
        return CameraFrame(data=np.zeros((4, 4), np.uint8),
                           timestamp=time.monotonic(), frame_id=fid)

    def test_circular_overwrite(self):  # boundary
        rb = CircularFrameBuffer(capacity=3)
        for i in range(5):
            rb.push(self._frame(i))
        self.assertEqual(rb.dropped, 2)
        self.assertEqual(rb.pop(timeout=0.1).frame_id, 2)

    def test_circular_latest(self):
        rb = CircularFrameBuffer(capacity=4)
        for i in range(3):
            rb.push(self._frame(i))
        self.assertEqual(rb.latest().frame_id, 2)

    def test_circular_pop_empty_timeout(self):  # error/empty
        rb = CircularFrameBuffer(capacity=2)
        self.assertIsNone(rb.pop(timeout=0.05))

    def test_circular_concurrent_no_deadlock(self):  # NFR-007
        rb = CircularFrameBuffer(capacity=16)
        N = 2000
        consumed = []
        stop = threading.Event()

        def producer():
            for i in range(N):
                rb.push(self._frame(i))

        def consumer():
            while not stop.is_set() or len(rb) > 0:
                f = rb.pop(timeout=0.05)
                if f is not None:
                    consumed.append(f.frame_id)

        c = threading.Thread(target=consumer)
        c.start()
        producer()
        time.sleep(0.2)
        stop.set()
        c.join(timeout=3.0)
        self.assertFalse(c.is_alive(), "consumer deadlocked")
        # Monotonic, no duplicates among consumed (drop-oldest preserves order).
        self.assertEqual(consumed, sorted(consumed))
        self.assertEqual(len(consumed), len(set(consumed)))

    def test_fifo_drop_newest(self):  # boundary
        q = FIFOFrameBuffer(capacity=2)
        for i in range(5):
            q.push(self._frame(i))
        self.assertEqual(q.dropped, 3)
        self.assertEqual(q.pop(timeout=0.1).frame_id, 0)  # order preserved


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Service-level: FR-002 streaming, NFR-005 reconnect, NFR-006 skip
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

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
        _wait_until(lambda: self.svc.stats("svccam")["frames_delivered"] > 0)
        self.svc.stop_streaming("svccam")
        self.assertGreater(self.svc.stats("svccam")["frames_delivered"], 0)
        self.assertIsNotNone(ring.pop(timeout=0.2))

    def test_malformed_skip(self):  # NFR-006 / FT-002
        ring = CircularFrameBuffer(32)
        self.svc.attach_sink("svccam", ring)
        self.svc.connect("svccam")
        self.svc.start_streaming("svccam")
        _wait_until(lambda: self.svc.stats("svccam")["frames_delivered"] > 0)
        self.drv.inject_malformed(3)
        # Poll until the worker has skipped all 3 (no fixed sleep -> no flake).
        _wait_until(lambda: self.svc.stats("svccam")["malformed_frames"] >= 3)
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
        path = os.path.join(os.path.dirname(__file__), "..",
                            "config", "camera.json")
        svc = CameraService()
        names = svc.add_cameras_from_config(
            os.path.abspath(path), lambda c: GenericCameraDriver(c))
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
            _wait_until(lambda: svc.stats("camA")["frames_delivered"] > 0
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
            _wait_until(lambda: svc.stats("rt")["frames_delivered"] > 0,
                        timeout=3.0)
            self.assertGreater(svc.stats("rt")["frames_delivered"], 0)
        finally:
            svc.shutdown()

    def test_stop_streaming_idempotent_quiet(self):  # cosmetic double-log fix
        self.svc.connect("svccam")
        self.svc.start_streaming("svccam")
        _wait_until(lambda: self.svc.stats("svccam")["frames_delivered"] > 0)
        with self.assertLogs("vision.camera_service", level="INFO") as cm:
            self.svc.stop_streaming("svccam")   # real stop -> logs once
            self.svc.stop_streaming("svccam")   # no-op -> must NOT log again
        stops = [m for m in cm.output if "Streaming stopped" in m]
        self.assertEqual(len(stops), 1)


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   End-to-end: vision serves frames -> cueing consumes (FR-002/FR-004,
#   NFR-007/NFR-008, IT-002)
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

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
        _wait_until(lambda: cueing.frames_consumed > 0 and len(seen) > 0)
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
        _wait_until(lambda: cueing.frames_consumed >= 1
                    and cueing.frames_errored >= 1)
        cueing.stop()
        self.assertGreaterEqual(cueing.frames_consumed, 1)
        self.assertGreaterEqual(cueing.frames_errored, 1)


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Lens / FOV calculators (NFR-004), validated against the FLIR app note.
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

class TestLensCalculator(unittest.TestCase):
    def test_flir_worked_example(self):
        # FLIR note: 1/2" sensor (W=6.4mm), WD=100mm, FOV=50mm ->
        # 11.3mm exact, 12.8mm approximate.
        self.assertAlmostEqual(
            focal_length_mm(6.4, 100.0, 50.0, exact=True), 11.3, delta=0.1)
        self.assertAlmostEqual(
            focal_length_mm(6.4, 100.0, 50.0, exact=False), 12.8, delta=0.1)

    def test_imx273_sensor_dims(self):
        w, h, d = sensor_dims_mm(1440, 1080, 3.45)  # BFS-U3-16S2C-CS
        self.assertAlmostEqual(w, 4.968, places=3)
        self.assertAlmostEqual(h, 3.726, places=3)
        self.assertAlmostEqual(d, 6.210, places=2)

    def test_kowa_5mm_angular_fov(self):
        w, _, _ = sensor_dims_mm(1440, 1080, 3.45)
        # Matches CameraConfig.lens_fov_deg in config/bfs_u3_16s2c.json.
        self.assertAlmostEqual(angular_fov_deg(w, 5.0), 52.8, delta=0.2)

    def test_linear_fov_at_3m(self):
        w, h, _ = sensor_dims_mm(1440, 1080, 3.45)
        self.assertAlmostEqual(
            linear_fov_mm(w, 5.0, 3000.0, exact=True), 2975.8, delta=1.0)
        self.assertAlmostEqual(
            linear_fov_mm(h, 5.0, 3000.0, exact=True), 2231.9, delta=1.0)

    def test_round_trip(self):
        w, _, _ = sensor_dims_mm(1440, 1080, 3.45)
        fov = linear_fov_mm(w, 5.0, 3000.0, exact=True)
        self.assertAlmostEqual(
            focal_length_mm(w, 3000.0, fov, exact=True), 5.0, places=6)
        self.assertAlmostEqual(
            working_distance_mm(w, 5.0, fov, exact=True), 3000.0, places=3)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            angular_fov_deg(4.968, 0.0)
        with self.assertRaises(ValueError):
            focal_length_mm(6.4, 100.0, 0.0)
        with self.assertRaises(ValueError):
            linear_fov_mm(4.968, 0.0, 1000.0)
        with self.assertRaises(ValueError):
            working_distance_mm(4.968, 0.0, 100.0)


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Regression tests (bugs found in review)
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

class TestRegressions(unittest.TestCase):
    def test_fifo_pop_zero_timeout_is_nonblocking(self):
        # gui_bridge._poll drains with pop(timeout=0); it must return None on an
        # empty queue instead of blocking forever (which froze the GUI).
        q = FIFOFrameBuffer(capacity=2)
        done = threading.Event()
        result = {}

        def drain():
            result["v"] = q.pop(timeout=0)
            done.set()

        threading.Thread(target=drain, daemon=True).start()
        self.assertTrue(done.wait(timeout=1.0), "pop(timeout=0) blocked on empty")
        self.assertIsNone(result["v"])

    def test_bad_pixel_format_raises_config_error(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.write(fd, json.dumps(
            {"name": "x", "output_pixel_format": "BadFmt"}).encode())
        os.close(fd)
        self.addCleanup(os.remove, path)
        with self.assertRaises(ConfigError):
            load_camera_config(path)


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Entry-point scripts import cleanly (no camera / PyQt needed at import time).
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

class TestScriptsImport(unittest.TestCase):
    def test_app_imports(self):
        import app
        # arg parsing works without a camera or any backend SDK.
        ns = app._parse_args(["--headless", "--seconds", "1"])
        self.assertTrue(ns.headless)
        self.assertEqual(ns.backend, "sim")          # default backend
        ns2 = app._parse_args(["--backend", "spinnaker", "--serial", "21512345"])
        self.assertEqual(ns2.backend, "spinnaker")
        self.assertEqual(ns2.serial, "21512345")
        self.assertFalse(ns.no_cueing)                         # default: cueing on
        self.assertTrue(app._parse_args(["--no-cueing"]).no_cueing)
        ns3 = app._parse_args(["--exposure", "12000", "--gain", "3", "--fps", "30"])
        self.assertEqual(ns3.exposure, 12000.0)
        self.assertEqual(ns3.gain, 3.0)
        self.assertEqual(ns3.fps, 30.0)
        self.assertFalse(ns.reset)                             # default: no reset
        self.assertTrue(app._parse_args(["--reset"]).reset)
        self.assertFalse(ns.rt)                                # default: no rt mode
        self.assertTrue(app._parse_args(["--rt"]).rt)

    def test_main_clean_exit_on_operational_error(self):
        # An expected backend failure (e.g. SDK missing / no camera) must exit
        # with code 1 and a clean message, NOT raise a traceback.
        import app
        with mock.patch.object(app, "_build_service",
                               side_effect=CameraError("no device")):
            rc = app.main(["--backend", "spinnaker", "--headless"])
        self.assertEqual(rc, 1)

    def test_runner_drives_multiple_cameras(self):  # multi-camera runner
        import app
        fd, path = tempfile.mkstemp(suffix=".json")
        os.write(fd, json.dumps({"cameras": [
            {"name": "m0", "max_resolution": [160, 120],
             "resolution": [160, 120], "fps": 60.0,
             "features": ["RESOLUTION", "FRAME_RATE"]},
            {"name": "m1", "max_resolution": [160, 120],
             "resolution": [160, 120], "fps": 60.0,
             "features": ["RESOLUTION", "FRAME_RATE"]}]}).encode())
        os.close(fd)
        try:
            with self.assertLogs("app", level="INFO") as cm:
                rc = app.main(["--backend", "sim", "--config", path,
                               "--headless", "--seconds", "1"])
            self.assertEqual(rc, 0)
            joined = "\n".join(cm.output)
            self.assertIn("Final m0", joined)     # both cameras ran + reported
            self.assertIn("Final m1", joined)
        finally:
            os.unlink(path)

    def test_multi_camera_config_ignores_cli_overrides(self):
        # One --serial can't bind N cameras: the runner warns and each camera
        # keeps its own config serial.
        import app
        fd, path = tempfile.mkstemp(suffix=".json")
        os.write(fd, json.dumps({"cameras": [
            {"name": "s0", "serial": "AAA", "max_resolution": [160, 120]},
            {"name": "s1", "serial": "BBB", "max_resolution": [160, 120]}]
        }).encode())
        os.close(fd)
        try:
            ns = app._parse_args(["--backend", "sim", "--config", path,
                                  "--serial", "ZZZ", "--headless"])
            with self.assertLogs("app", level="WARNING") as cm:
                svc, names = app._build_service(ns)
            try:
                self.assertEqual(set(names), {"s0", "s1"})
                self.assertTrue(any("ignored" in m for m in cm.output))
                self.assertEqual(svc._entry("s0").driver.config.serial, "AAA")
                self.assertEqual(svc._entry("s1").driver.config.serial, "BBB")
            finally:
                svc.shutdown()
        finally:
            os.unlink(path)

    def test_hardware_acceptance_imports(self):
        import hardware_acceptance
        ns = hardware_acceptance._parse_args(
            ["--serial", "21512345", "--seconds", "5", "--min-fps", "60",
             "--mono", "--no-hw-timestamp"])
        crit = hardware_acceptance._criteria(ns)
        self.assertEqual(ns.serial, "21512345")
        self.assertFalse(crit.require_color)        # --mono
        self.assertFalse(crit.require_hw_timestamp)  # --no-hw-timestamp
        self.assertEqual(crit.min_fps, 60.0)


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
# Hardware smoke test (real BlackFly S over Spinnaker). Skipped automatically
# when PySpin or a physical camera is not present, so CI stays green; runs on
# the rig to prove the BFS-U3-16S2C-CS color path (BayerRG8 -> BGR8).
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

def _spinnaker_camera_present() -> bool:
    try:
        import PySpin
    except Exception:
        return False
    system = None
    try:
        system = PySpin.System.GetInstance()
        cams = system.GetCameras()
        n = cams.GetSize()
        cams.Clear()
        return n > 0
    except Exception:
        return False
    finally:
        if system is not None:
            try:
                system.ReleaseInstance()
            except Exception:
                pass


@unittest.skipUnless(_spinnaker_camera_present(),
                     "PySpin + a physical BlackFly S camera not present")
class TestSpinnakerHardware(unittest.TestCase):  # HW-001 (real device)
    def test_color_bgr_capture(self):
        from vision.spinnaker_driver import SpinnakerCameraDriver
        cfg_path = os.path.join(os.path.dirname(__file__), "..",
                                "config", "bfs_u3_16s2c.json")
        cfg = load_camera_config(os.path.abspath(cfg_path))
        drv = SpinnakerCameraDriver(cfg)
        # connect() failing here -> SDK/permissions/camera not found:
        #   check spinview, the flirimaging group/udev, and Phase 0/1 of the
        #   bring-up section in the README.
        drv.connect()
        try:
            self.assertEqual(
                drv.get_status(), CameraStatus.CONNECTED,
                "connect() did not reach CONNECTED — camera opened but init "
                "failed; check the serial/device_index in the config.")
            drv.start_stream()
            frame = drv.read_frame(timeout=2.0)

            # Color camera must yield a 3-channel BGR8 frame at full res.
            self.assertEqual(
                frame.data.ndim, 3,
                f"Got a {frame.data.ndim}-D frame; expected 3-D color. The "
                "camera isn't debayering to BGR — check 'output_pixel_format' "
                "(BGR8) and 'device_pixel_format' (BayerRG8) in the config, and "
                "that ImageProcessor/Convert ran (see spinnaker_driver._convert).")
            self.assertEqual(
                frame.data.shape[2], 3,
                f"Got {frame.data.shape[2]} channels; expected 3 (BGR). "
                "Pixel-format conversion produced the wrong layout — verify "
                "'output_pixel_format': 'BGR8' in the config.")
            self.assertEqual(
                (frame.data.shape[1], frame.data.shape[0]), tuple(cfg.resolution),
                f"Frame {frame.data.shape[1]}x{frame.data.shape[0]} != configured "
                f"{tuple(cfg.resolution)}. Width/Height weren't applied (a saved "
                "user set, binning, or ROI offset may differ) — check "
                "'resolution' and that RESOLUTION is in 'features'.")
            self.assertEqual(
                frame.data.dtype, np.uint8,
                f"Frame dtype {frame.data.dtype} != uint8. Conversion target is "
                "wrong — use an 8-bit 'output_pixel_format' (BGR8/Mono8), not a "
                "16-bit format, for the 8-bit pipeline.")
            self.assertIsNotNone(
                frame.hw_timestamp_ns,
                "No hardware timestamp — GetTimeStamp() returned nothing; the "
                "driver couldn't read the device clock (check Spinnaker/PySpin "
                "version compatibility).")
            drv.stop_stream()
        finally:
            drv.disconnect()


if __name__ == "__main__":
    unittest.main(verbosity=2)
