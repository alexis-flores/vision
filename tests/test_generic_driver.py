"""
tests/test_generic_driver.py
UT-002: GenericCameraDriver (the simulator backend) lifecycle, feature gating,
fault injection (NFR-006 source), the frame-immutability contract, and the
context-manager sugar.
"""

from __future__ import annotations

import unittest

import _helpers  # noqa: F401  (path bootstrap)
from _helpers import basic_config

from vision.camera_driver import (CameraError, FeatureNotSupportedError,
                                  MalformedFrameError)
from vision.camera_types import CameraFeature, CameraFrame, CameraStatus
from vision.generic_driver import GenericCameraDriver


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
