"""
tests/test_acceptance.py
Tests for the camera acceptance battery (acceptance.py).

Two layers:
  * `evaluate()` is pure, so each PASS/FAIL/SKIP path is checked deterministically
    with hand-built Metrics (no timing flakiness, no hardware).
  * an integration check runs the real collect+evaluate against the simulator,
    plus the connect/teardown cycle test.
"""

from __future__ import annotations

import unittest

import numpy as np

import _helpers  # noqa: F401  (path bootstrap)

from vision.acceptance import (AcceptanceCriteria, FrameStat, Metrics,
                               evaluate, run_acceptance, run_bandwidth_stress,
                               run_connect_cycles, run_recovery_cycles)
from vision.camera_driver import CameraError
from vision.camera_service import CameraService
from vision.camera_types import CameraConfig, CameraFeature
from vision.generic_driver import GenericCameraDriver


def healthy_metrics(n=120, fps=60.0, w=640, h=640, color=True, hw=True,
                    level=100, device_ids=False, temps=None):
    period = 1.0 / fps
    base = 1000.0
    hw0 = 1_000_000_000
    frames = [
        FrameStat(
            host_ts=base + i * period,
            hw_ts=(hw0 + int(i * period * 1e9)) if hw else None,
            frame_id=i, ndim=3 if color else 2, channels=3 if color else 1,
            dtype="uint8", width=w, height=h,
            device_frame_id=(i if device_ids else None))
        for i in range(n)]
    shape = (h, w, 3) if color else (h, w)
    samples = [np.full(shape, level, np.uint8) for _ in range(4)]
    return Metrics(frames=frames, samples=samples, delivered=n, malformed=0,
                   reconnects=0, streaming_reached=True, cfg_resolution=(w, h),
                   cfg_fps=fps, duration_s=n * period,
                   temperatures=list(temps or []))


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


class TestEvaluatePure(unittest.TestCase):
    def setUp(self):
        self.crit = AcceptanceCriteria(min_fps=60.0, min_resolution=512)

    def test_healthy_passes(self):
        rep = evaluate(healthy_metrics(), self.crit)
        self.assertTrue(rep.passed, rep.format())

    def test_no_frames_fails(self):
        m = Metrics(streaming_reached=True)
        rep = evaluate(m, self.crit)
        self.assertFalse(rep.passed)
        self.assertFalse(_check(rep, "frames_received").passed)

    def test_low_fps_fails(self):
        rep = evaluate(healthy_metrics(fps=30.0), self.crit)  # 30 < min 60
        self.assertFalse(rep.passed)
        self.assertFalse(_check(rep, "throughput_fps").passed)

    def test_malformed_rate_fails(self):
        m = healthy_metrics()
        m.delivered, m.malformed = 100, 50   # 33% malformed
        rep = evaluate(m, self.crit)
        self.assertFalse(_check(rep, "frame_integrity").passed)

    def test_missing_hw_timestamp_fails_when_required(self):
        rep = evaluate(healthy_metrics(hw=False), self.crit)
        self.assertTrue(self.crit.require_hw_timestamp)
        self.assertFalse(_check(rep, "hw_timestamp").passed)

    def test_no_hw_timestamp_skips_when_not_required(self):
        crit = AcceptanceCriteria(require_hw_timestamp=False)
        rep = evaluate(healthy_metrics(hw=False), crit)
        self.assertTrue(_check(rep, "hw_timestamp").skipped)
        self.assertTrue(_check(rep, "interframe_jitter").skipped)
        self.assertTrue(_check(rep, "dropped_frames").skipped)
        self.assertTrue(rep.passed, rep.format())

    def test_non_monotonic_hw_timestamp_fails(self):
        m = healthy_metrics()
        m.frames[10].hw_ts = m.frames[9].hw_ts - 5  # goes backwards
        rep = evaluate(m, self.crit)
        self.assertFalse(_check(rep, "hw_timestamp").passed)

    def test_black_image_fails(self):
        m = healthy_metrics(level=0)
        rep = evaluate(m, self.crit)
        self.assertFalse(_check(rep, "image_sanity").passed)

    def test_saturated_image_fails(self):
        m = healthy_metrics(level=255)
        rep = evaluate(m, self.crit)
        self.assertFalse(_check(rep, "image_sanity").passed)

    def test_resolution_fail(self):
        m = healthy_metrics(w=256, h=256)   # below min 512
        rep = evaluate(m, self.crit)
        self.assertFalse(_check(rep, "resolution").passed)

    def test_jitter_fails(self):
        m = healthy_metrics()
        # inject big timing wobble into the device timestamps
        for i in range(1, len(m.frames), 2):
            m.frames[i].hw_ts += 5_000_000   # +5 ms
        rep = evaluate(m, AcceptanceCriteria(max_jitter_ms=0.5))
        self.assertFalse(_check(rep, "interframe_jitter").passed)

    def test_dropped_frames_fail(self):
        m = healthy_metrics(fps=60.0)
        # create a ~3x period gap after frame 50 -> ~2 dropped
        for i in range(51, len(m.frames)):
            m.frames[i].hw_ts += 2 * int(1e9 / 60)
        rep = evaluate(m, AcceptanceCriteria(max_dropped_rate=0.0))
        self.assertFalse(_check(rep, "dropped_frames").passed)

    def test_stability_fails_on_reconnect(self):
        m = healthy_metrics()
        m.reconnects = 2
        rep = evaluate(m, self.crit)
        self.assertFalse(_check(rep, "stream_stability").passed)

    def test_device_counter_no_gap_passes(self):  # chunk-based drop detection
        rep = evaluate(healthy_metrics(device_ids=True), self.crit)
        drop = _check(rep, "dropped_frames")
        self.assertTrue(drop.passed)
        self.assertIn("device frame counter", drop.detail)  # authoritative path

    def test_device_counter_gap_fails(self):
        m = healthy_metrics(device_ids=True)
        for i in range(50, len(m.frames)):
            m.frames[i].device_frame_id += 5   # 5 missing device ids -> drops
        rep = evaluate(m, self.crit)
        self.assertFalse(_check(rep, "dropped_frames").passed)

    def test_temperature_over_limit_fails(self):
        m = healthy_metrics(temps=[40.0, 78.0, 50.0])
        rep = evaluate(m, AcceptanceCriteria(max_temperature_c=75.0))
        self.assertFalse(_check(rep, "temperature").passed)

    def test_temperature_within_limit_passes(self):
        m = healthy_metrics(temps=[40.0, 55.0])
        rep = evaluate(m, AcceptanceCriteria(max_temperature_c=75.0))
        self.assertTrue(_check(rep, "temperature").passed)

    def test_temperature_absent_skips(self):
        rep = evaluate(healthy_metrics(), self.crit)  # no temps
        self.assertTrue(_check(rep, "temperature").skipped)


class TestAcceptanceIntegration(unittest.TestCase):
    def _config(self):
        return CameraConfig(
            name="acc", model="sim", max_resolution=(640, 640),
            resolution=(640, 640), max_fps=60.0, fps=30.0,
            features=(CameraFeature.FRAME_RATE | CameraFeature.RESOLUTION))

    def test_simulator_passes(self):
        svc = CameraService()
        svc.add_camera("acc", GenericCameraDriver(self._config(), n_spots=5))
        svc.connect("acc")
        # Generous margins so the check passes even on a heavily loaded CI box
        # (the sim targets 30 fps; we only require >= 5 over a 1.5 s window).
        crit = AcceptanceCriteria(
            seconds=1.5, min_fps=5.0, require_color=True,
            require_hw_timestamp=False, min_mean_level=1.0, sample_every=3)
        try:
            rep = run_acceptance(svc, "acc", crit)
        finally:
            svc.shutdown()
        self.assertTrue(rep.passed, rep.format())

    def test_connect_cycles_pass(self):
        cfg = self._config()
        res = run_connect_cycles(
            lambda: GenericCameraDriver(cfg, n_spots=3),
            cycles=3, frames_per_cycle=3)
        self.assertTrue(res.passed, res.detail)

    def test_connect_cycles_detect_failure(self):
        cfg = self._config()

        class _BadDriver(GenericCameraDriver):
            def connect(self):
                raise CameraError("simulated connect failure")

        res = run_connect_cycles(lambda: _BadDriver(cfg), cycles=2)
        self.assertFalse(res.passed)

    def test_recovery_cycles_pass(self):
        # Mid-stream interrupt/recover loop resumes frames every cycle.
        drv = GenericCameraDriver(self._config(), n_spots=3)
        res = run_recovery_cycles(drv, cycles=3, frames_per_cycle=2,
                                  read_timeout=2.0)
        self.assertTrue(res.passed, res.detail)

    def test_recovery_cycles_detect_failure(self):
        # A backend that cannot re-acquire after the first connect must FAIL.
        class _NoRecoverDriver(GenericCameraDriver):
            def __init__(self, cfg, **kw):
                super().__init__(cfg, **kw)
                self._connects = 0

            def connect(self):
                self._connects += 1
                if self._connects > 1:
                    raise CameraError("cannot re-acquire")
                super().connect()

        drv = _NoRecoverDriver(self._config(), n_spots=3)
        res = run_recovery_cycles(drv, cycles=2, frames_per_cycle=2)
        self.assertFalse(res.passed)

    def test_bandwidth_stress_pass_and_restores_config(self):
        # Under a throughput squeeze the sim still streams stably; the original
        # link_throughput_limit_bps is restored afterwards (non-persisted).
        svc = CameraService()
        drv = GenericCameraDriver(self._config(), n_spots=5)
        svc.add_camera("acc", drv)
        svc.connect("acc")
        crit = AcceptanceCriteria(
            seconds=1.0, min_fps=5.0, require_color=True,
            require_hw_timestamp=False, min_mean_level=1.0, sample_every=3)
        self.assertIsNone(drv.config.link_throughput_limit_bps)
        try:
            res = run_bandwidth_stress(svc, "acc", crit, limit_bps=1_000_000)
        finally:
            svc.shutdown()
        self.assertTrue(res.passed, res.detail)
        self.assertIsNone(drv.config.link_throughput_limit_bps)  # restored


if __name__ == "__main__":
    unittest.main(verbosity=2)
