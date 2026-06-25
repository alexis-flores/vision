"""
tests/test_drivers_mocked.py
Exercises the real-hardware drivers WITHOUT a camera by injecting a fake PySpin
module and a fake cv2.VideoCapture. This verifies the driver *logic* that can't
otherwise run on a dev machine: connect/stream/convert sequence, value clamping,
ROI-offset reset, and the timeout-vs-fault error classification (NFR-005).
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from vision.camera_driver import (CameraError, CameraTimeoutError,
                                  MalformedFrameError)
from vision.camera_types import (CameraConfig, CameraFeature, CameraStatus,
                                 PixelFormat)


# --------------------------------------------------------------------------- #
#   Fake PySpin
# --------------------------------------------------------------------------- #

def _make_fake_pyspin(getnext_errorcode=None, incomplete=False,
                      frame_hw=(1080, 1440, 3)):
    PySpin = types.ModuleType("PySpin")
    PySpin._select = {"mode": None, "arg": None}  # records GetBySerial vs GetByIndex

    class SpinnakerException(Exception):
        def __init__(self, msg="err", errorcode=-1):
            super().__init__(msg)
            self.message = msg
            self.errorcode = errorcode

    PySpin.SpinnakerException = SpinnakerException
    PySpin.EVENT_TIMEOUT_INFINITE = -1
    PySpin.SPINNAKER_ERR_TIMEOUT = -1011
    PySpin.HQ_LINEAR = 0
    PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR = 1
    PySpin.AcquisitionMode_Continuous = 2
    PySpin.ExposureAuto_Off = 0
    PySpin.GainAuto_Off = 0
    for nm in ("PixelFormat_BGR8", "PixelFormat_RGB8", "PixelFormat_Mono8",
               "PixelFormat_Mono16", "PixelFormat_BayerRG8"):
        setattr(PySpin, nm, nm)
    PySpin.ChunkSelector_FrameID = "ChunkSelector_FrameID"
    PySpin.ChunkSelector_Timestamp = "ChunkSelector_Timestamp"

    class Num:
        def __init__(self, v, lo, hi):
            self.v, self.lo, self.hi = v, lo, hi
        def GetMin(self): return self.lo
        def GetMax(self): return self.hi
        def GetValue(self): return self.v
        def SetValue(self, x): self.v = x

    class Enum:
        def __init__(self): self.v = None
        def SetValue(self, x): self.v = x

    class Image:
        def __init__(self, arr, inc=False, frame_id=42):
            self.arr, self.inc, self.released = arr, inc, False
            self._frame_id = frame_id
        def IsIncomplete(self): return self.inc
        def GetImageStatus(self): return 9
        def GetNDArray(self): return self.arr
        def GetTimeStamp(self): return 123456789
        def GetChunkData(self):
            fid = self._frame_id
            class _CD:
                def GetFrameID(self): return fid
            return _CD()
        def Release(self): self.released = True

    class ImageProcessor:
        def SetColorProcessing(self, algo): self.algo = algo
        def Convert(self, img, target):
            return Image(np.zeros(frame_hw, np.uint8))

    PySpin.ImageProcessor = ImageProcessor

    class EnumEntry:
        def GetValue(self): return 5

    class CEnum:
        def GetEntryByName(self, name): return EnumEntry()
        def SetIntValue(self, v): self.v = v
        def GetValue(self): return 0           # int-node read (stream stats)

    PySpin.CEnumerationPtr = lambda node: node
    PySpin.CIntegerPtr = lambda node: node
    PySpin.IsAvailable = lambda n: True
    PySpin.IsWritable = lambda n: True
    PySpin.IsReadable = lambda n: True

    class StreamNodeMap:
        def GetNode(self, name): return CEnum()

    class Camera:
        def __init__(self):
            self.Width = Num(640, 0, 1440)       # max 1440 (full frame)
            self.Height = Num(480, 0, 1080)
            self.Gain = Num(0.0, 0.0, 47.0)
            self.ExposureTime = Num(5000.0, 13.0, 30000000.0)
            self.AcquisitionFrameRate = Num(30.0, 1.0, 226.0)
            self.OffsetX = Num(8, 0, 0)          # persisted offset; min 0
            self.OffsetY = Num(8, 0, 0)
            self.BlackLevel = Num(0.0, 0.0, 10.0)
            self.Gamma = Num(1.0, 0.1, 4.0)
            self.PixelFormat = Enum()
            self.ExposureAuto = Enum()
            self.GainAuto = Enum()
            self.AcquisitionFrameRateEnable = Enum()
            self.AcquisitionMode = Enum()
            self.DeviceTemperature = Num(45.5, 0.0, 100.0)
            self.ChunkModeActive = Enum()
            self.ChunkSelector = Enum()
            self.ChunkEnable = Enum()
            self.inited = self.acquiring = False
        def Init(self): self.inited = True
        def DeInit(self): self.inited = False
        def BeginAcquisition(self): self.acquiring = True
        def EndAcquisition(self): self.acquiring = False
        def GetTLStreamNodeMap(self): return StreamNodeMap()
        def GetNextImage(self, timeout):
            if getnext_errorcode is not None:
                raise SpinnakerException("injected GetNextImage error",
                                         errorcode=getnext_errorcode)
            return Image(np.zeros(frame_hw, np.uint8), inc=incomplete)

    cam = Camera()

    class CamList:
        def GetSize(self): return 1
        def GetBySerial(self, s):
            PySpin._select.update(mode="serial", arg=s); return cam
        def GetByIndex(self, i):
            PySpin._select.update(mode="index", arg=i); return cam
        def Clear(self): pass

    PySpin._released = {"count": 0}

    class System:
        @staticmethod
        def GetInstance(): return System()
        def GetCameras(self): return CamList()
        def ReleaseInstance(self): PySpin._released["count"] += 1

    PySpin.System = System
    PySpin._cam = cam
    return PySpin


def _bfs_config(serial=None):
    return CameraConfig(
        name="bfs", model="BFS-U3-16S2C-CS", device_index=0, serial=serial,
        max_resolution=(1440, 1080), max_fps=226.0,
        resolution=(1440, 1080), fps=60.0, exposure_us=5000.0, gain_db=0.0,
        pixel_format=PixelFormat.BGR8,
        features=(CameraFeature.GAIN | CameraFeature.EXPOSURE
                  | CameraFeature.FRAME_RATE | CameraFeature.RESOLUTION
                  | CameraFeature.PIXEL_FORMAT),
        extra={"device_pixel_format": "BayerRG8"})


class TestSpinnakerDriverMocked(unittest.TestCase):
    def _install(self, serial=None, **kw):
        fake = _make_fake_pyspin(**kw)
        sys.modules["PySpin"] = fake
        self.addCleanup(lambda: sys.modules.pop("PySpin", None))
        from vision.spinnaker_driver import SpinnakerCameraDriver
        return fake, SpinnakerCameraDriver(_bfs_config(serial=serial))

    def test_connect_stream_read_color(self):
        fake, drv = self._install()
        drv.connect()
        self.assertEqual(drv.get_status(), CameraStatus.CONNECTED)
        drv.start_stream()
        frame = drv.read_frame(timeout=1.0)
        self.assertEqual(frame.data.ndim, 3)
        self.assertEqual(frame.data.shape[2], 3)
        self.assertEqual((frame.data.shape[1], frame.data.shape[0]), (1440, 1080))
        self.assertEqual(frame.data.dtype, np.uint8)
        self.assertIsNotNone(frame.hw_timestamp_ns)
        drv.stop_stream()
        drv.disconnect()
        self.assertEqual(drv.get_status(), CameraStatus.DISCONNECTED)

    def test_roi_offset_reset_and_resolution_applied(self):
        fake, drv = self._install()
        drv.connect()
        self.assertEqual(fake._cam.OffsetX.GetValue(), 0)   # reset to min
        self.assertEqual(fake._cam.OffsetY.GetValue(), 0)
        self.assertEqual(fake._cam.Width.GetValue(), 1440)
        self.assertEqual(fake._cam.Height.GetValue(), 1080)

    def test_value_clamping(self):
        fake, drv = self._install()
        drv.connect()
        drv.set_config("fps", 9999.0)               # above max 226
        self.assertEqual(fake._cam.AcquisitionFrameRate.GetValue(), 226.0)
        drv.set_config("gain_db", 3.0)              # in range -> unchanged
        self.assertEqual(fake._cam.Gain.GetValue(), 3.0)

    def test_timeout_is_classified_as_timeout(self):
        fake, drv = self._install(getnext_errorcode=-1011)   # SPINNAKER_ERR_TIMEOUT
        drv.connect(); drv.start_stream()
        with self.assertRaises(CameraTimeoutError):
            drv.read_frame(timeout=0.5)

    def test_fault_is_classified_as_error_not_timeout(self):
        # NFR-005: a non-timeout SpinnakerException must surface as CameraError
        # (-> service reconnect), NOT a timeout (-> infinite retry).
        fake, drv = self._install(getnext_errorcode=-1004)   # non-timeout fault
        drv.connect(); drv.start_stream()
        with self.assertRaises(CameraError) as ctx:
            drv.read_frame(timeout=0.5)
        self.assertNotIsInstance(ctx.exception, CameraTimeoutError)

    def test_incomplete_frame_is_malformed(self):
        fake, drv = self._install(incomplete=True)
        drv.connect(); drv.start_stream()
        with self.assertRaises(MalformedFrameError):
            drv.read_frame(timeout=0.5)

    def test_no_serial_warns_and_selects_by_index(self):  # NFR-005 footgun
        # Without a serial, selection falls back to device_index, which is not
        # stable across USB re-enumeration; the driver must warn about it.
        fake, drv = self._install()  # _bfs_config(serial=None)
        with self.assertLogs("vision.spinnaker_driver", level="WARNING") as cm:
            drv.connect()
        self.assertTrue(any("serial" in m.lower() for m in cm.output))
        self.assertEqual(fake._select["mode"], "index")
        drv.disconnect()

    def test_serial_selects_by_serial(self):  # NFR-005 stable rebind
        fake, drv = self._install(serial="BFS123456")
        drv.connect()
        self.assertEqual(fake._select["mode"], "serial")
        self.assertEqual(fake._select["arg"], "BFS123456")
        drv.disconnect()

    def test_disconnect_releases_system(self):  # ordered teardown + gc.collect
        fake, drv = self._install()
        drv.connect()
        drv.disconnect()
        self.assertEqual(drv.get_status(), CameraStatus.DISCONNECTED)
        self.assertGreaterEqual(fake._released["count"], 1)

    def test_release_on_exit_is_idempotent(self):  # atexit safety net
        fake, drv = self._install()
        drv.connect()
        drv._release_on_exit()           # simulate atexit firing on unclean exit
        self.assertEqual(drv.get_status(), CameraStatus.DISCONNECTED)
        drv._release_on_exit()           # firing again must never raise

    def test_device_frame_id_from_chunk(self):  # chunk data -> drop detection
        fake, drv = self._install()
        drv.connect(); drv.start_stream()
        frame = drv.read_frame(timeout=1.0)
        self.assertEqual(frame.device_frame_id, 42)
        drv.stop_stream(); drv.disconnect()

    def test_get_health_reports_temperature(self):  # device health telemetry
        fake, drv = self._install()
        drv.connect()
        health = drv.get_health()
        self.assertAlmostEqual(health["temperature_c"], 45.5)
        drv.disconnect()


# --------------------------------------------------------------------------- #
#   Fake OpenCV VideoCapture
# --------------------------------------------------------------------------- #

class _FakeCapture:
    opened = True
    read_ok = True

    def __init__(self, *a, **k):
        self._opened = _FakeCapture.opened
    def isOpened(self): return self._opened
    def set(self, prop, val): return True
    def get(self, prop): return 0.0
    def read(self):
        if not _FakeCapture.read_ok:
            return False, None
        return True, np.zeros((480, 640, 3), np.uint8)
    def release(self): pass


def _ocv_config():
    return CameraConfig(
        name="webcam", device_index=0, max_resolution=(640, 480),
        resolution=(640, 480), fps=30.0,
        features=CameraFeature.RESOLUTION | CameraFeature.FRAME_RATE)


class TestOpenCVDriverMocked(unittest.TestCase):
    def setUp(self):
        _FakeCapture.opened = True
        _FakeCapture.read_ok = True

    def test_connect_stream_read(self):
        from vision.opencv_driver import OpenCVCameraDriver
        with mock.patch("cv2.VideoCapture", _FakeCapture):
            drv = OpenCVCameraDriver(_ocv_config())
            drv.connect()
            self.assertEqual(drv.get_status(), CameraStatus.CONNECTED)
            drv.start_stream()
            frame = drv.read_frame(timeout=1.0)
            self.assertEqual(frame.data.shape, (480, 640, 3))
            drv.stop_stream()
            drv.disconnect()
            self.assertEqual(drv.get_status(), CameraStatus.DISCONNECTED)

    def test_open_failure_raises(self):
        from vision.opencv_driver import OpenCVCameraDriver
        _FakeCapture.opened = False
        with mock.patch("cv2.VideoCapture", _FakeCapture):
            drv = OpenCVCameraDriver(_ocv_config())
            with self.assertRaises(CameraError):
                drv.connect()

    def test_read_failure_becomes_backend_fault(self):
        from vision.opencv_driver import OpenCVCameraDriver
        _FakeCapture.read_ok = False
        with mock.patch("cv2.VideoCapture", _FakeCapture):
            drv = OpenCVCameraDriver(_ocv_config())
            drv.connect(); drv.start_stream()
            with self.assertRaises(CameraError):
                drv.read_frame(timeout=1.0)
            drv.disconnect()


if __name__ == "__main__":
    unittest.main(verbosity=2)
