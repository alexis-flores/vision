"""
tests/test_drivers_mocked.py
Exercises the real-hardware drivers WITHOUT a camera by injecting a fake PySpin
module and a fake cv2.VideoCapture. This verifies the driver *logic* that can't
otherwise run on a dev machine: connect/stream/convert sequence, value clamping,
ROI-offset reset, and the timeout-vs-fault error classification (NFR-005).
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import numpy as np

import _helpers  # noqa: F401  (path bootstrap)

from vision.camera_driver import (CameraError, CameraTimeoutError,
                                  MalformedFrameError)
from vision.camera_types import (CameraConfig, CameraFeature, CameraStatus,
                                 PixelFormat)


# --------------------------------------------------------------------------- #
#   Fake PySpin
# --------------------------------------------------------------------------- #

def _make_fake_pyspin(getnext_errorcode=None, incomplete=False, crc_fail=False,
                      select_raises=False, frame_hw=(1080, 1440, 3)):
    PySpin = types.ModuleType("PySpin")
    PySpin.SPINNAKER_IMAGE_STATUS_CRC_CHECK_FAILED = 1
    PySpin._select = {"mode": None, "arg": None}  # records GetBySerial vs GetByIndex

    class SpinnakerException(Exception):
        def __init__(self, msg="err", errorcode=-1):
            super().__init__(msg)
            self.message = msg
            self.errorcode = errorcode

    PySpin.SpinnakerException = SpinnakerException
    PySpin.EVENT_TIMEOUT_INFINITE = -1
    PySpin.SPINNAKER_ERR_TIMEOUT = -1011
    PySpin.SPINNAKER_ERR_ABORT = -1012          # device-fatal codes (never retried)
    PySpin.SPINNAKER_ERR_INVALID_HANDLE = -1006
    PySpin.SPINNAKER_ERR_ACCESS_DENIED = -1005
    PySpin.SPINNAKER_ERR_IO = -1010
    PySpin.SPINNAKER_ERR_NOT_AVAILABLE = -1014
    PySpin.HQ_LINEAR = 0
    PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR = 1
    PySpin.AcquisitionMode_Continuous = 2
    PySpin.ExposureAuto_Off = 0
    PySpin.GainAuto_Off = 0
    for nm in ("PixelFormat_BGR8", "PixelFormat_RGB8", "PixelFormat_Mono8",
               "PixelFormat_Mono16", "PixelFormat_BayerRG8"):
        setattr(PySpin, nm, nm)
    # Static Image.GetImageStatusDescription(status) -> human-readable text,
    # mirroring the real SDK (used by the driver's NFR-006 diagnostics).
    PySpin.Image = type("Image", (), {
        "GetImageStatusDescription": staticmethod(lambda s: f"status-desc-{s}")})
    PySpin.ChunkSelector_FrameID = "ChunkSelector_FrameID"
    PySpin.ChunkSelector_Timestamp = "ChunkSelector_Timestamp"
    PySpin.UserSetSelector_Default = "UserSetSelector_Default"
    PySpin.UserSetDefault_Default = "UserSetDefault_Default"

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

    class Command:
        def __init__(self): self.executed = False
        def Execute(self): self.executed = True

    class Image:
        def __init__(self, arr, inc=False, frame_id=42, status=9):
            self.arr, self.inc, self.released = arr, inc, False
            self._frame_id, self._status = frame_id, status
        def IsIncomplete(self): return self.inc
        def GetImageStatus(self): return self._status
        def GetNDArray(self): return self.arr
        def GetTimeStamp(self): return 123456789
        def GetChunkData(self):
            fid = self._frame_id
            class _CD:
                def GetFrameID(self): return fid
                def GetExposureTime(self): return 5000.0
                def GetGain(self): return 1.5
                def GetBlackLevel(self): return 2.0
            return _CD()
        def Release(self): self.released = True

    class ImageProcessor:
        def SetColorProcessing(self, algo): self.algo = algo
        def Convert(self, img, target):
            return Image(np.zeros(frame_hw, np.uint8))
        def ApplyGamma(self, img, gamma): return img

    PySpin.ImageProcessor = ImageProcessor

    # Host CCM API (mirror the modern SDK so white_balance="ccm" runs off-rig).
    for nm in ("SPINNAKER_CCM_SENSOR_IMX273", "SPINNAKER_CCM_TYPE_LINEAR",
               "SPINNAKER_CCM_COLOR_SPACE_SRGB",
               "SPINNAKER_CCM_APPLICATION_MICROSCOPY",
               "SPINNAKER_CCM_COLOR_TEMP_LED_4649K",
               "SPINNAKER_COLOR_PROCESSING_ALGORITHM_RIGOROUS"):
        setattr(PySpin, nm, nm)
    PySpin.CCMSettings = type("CCMSettings", (), {})  # attrs set dynamically
    PySpin.Image.Create = staticmethod(
        lambda w, h, ox, oy, fmt, buf: Image(np.asarray(buf)))

    class ImageUtilityCCM:
        calls = {"count": 0}
        @staticmethod
        def CreateColorCorrected(img, settings):
            ImageUtilityCCM.calls["count"] += 1
            return Image(np.asarray(img.GetNDArray()))
    PySpin.ImageUtilityCCM = ImageUtilityCCM

    class EnumEntry:
        def GetValue(self): return 5

    class CEnum:
        def __init__(self): self.v = 0
        def GetEntryByName(self, name): return EnumEntry()
        def SetIntValue(self, v): self.v = v
        def SetValue(self, v): self.v = v      # CIntegerPtr int-node writes
        def GetValue(self): return self.v      # 0 by default (stream stats)

    PySpin.CEnumerationPtr = lambda node: node
    PySpin.CIntegerPtr = lambda node: node
    PySpin.CBooleanPtr = lambda node: node
    PySpin.IsAvailable = lambda n: True
    PySpin.IsWritable = lambda n: True
    PySpin.IsReadable = lambda n: True

    class StreamNodeMap:                        # cache nodes so writes persist
        def __init__(self): self._nodes = {}
        def GetNode(self, name): return self._nodes.setdefault(name, CEnum())

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
            self.DeviceLinkThroughputLimit = Num(380000000, 0, 480000000)
            self.DeviceLinkCurrentThroughput = Num(93312000, 0, 480000000)
            self.AcquisitionResultingFrameRate = Num(60.0, 1.0, 226.0)
            self.ChunkModeActive = Enum()
            self.ChunkSelector = Enum()
            self.ChunkEnable = Enum()
            self.UserSetSelector = Enum()
            self.UserSetDefault = Enum()
            self.UserSetLoad = Command()
            self.DeviceReset = Command()
            self.TimestampLatch = Command()
            self.TimestampLatchValue = Num(123456789, 0, 2**63)  # matches img ts
            self.inited = self.acquiring = False
            self.deinit_calls = self.endacq_calls = 0
            self.images_in_use = 0
        def GetNumImagesInUse(self): return self.images_in_use
        def Init(self): self.inited = True
        def DeInit(self): self.inited = False; self.deinit_calls += 1
        def BeginAcquisition(self): self.acquiring = True
        def EndAcquisition(self):
            self.acquiring = False; self.endacq_calls += 1
        def GetTLStreamNodeMap(self):
            if not hasattr(self, "_snm"):
                self._snm = StreamNodeMap()
            return self._snm
        def GetNextImage(self, timeout):
            if getnext_errorcode is not None:
                raise SpinnakerException("injected GetNextImage error",
                                         errorcode=getnext_errorcode)
            return Image(np.zeros(frame_hw, np.uint8), inc=incomplete,
                         status=(1 if crc_fail else 9))

    cam = Camera()

    PySpin._cam_list = {"cleared": 0}

    class CamList:
        def GetSize(self): return 1
        def GetBySerial(self, s):
            if select_raises:
                raise SpinnakerException("injected selection failure")
            PySpin._select.update(mode="serial", arg=s); return cam
        def GetByIndex(self, i):
            if select_raises:
                raise SpinnakerException("injected selection failure")
            PySpin._select.update(mode="index", arg=i); return cam
        def Clear(self): PySpin._cam_list["cleared"] += 1

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
        output_pixel_format=PixelFormat.BGR8,
        features=(CameraFeature.GAIN | CameraFeature.EXPOSURE
                  | CameraFeature.FRAME_RATE | CameraFeature.RESOLUTION
                  | CameraFeature.PIXEL_FORMAT),
        device_pixel_format="BayerRG8")


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

    def test_stream_buffer_count_opt_in(self):
        # Default (None) leaves the SDK default; a set value is applied at
        # start_stream via StreamBufferCountManual.
        fake, drv = self._install()
        snm = fake._cam.GetTLStreamNodeMap()
        drv.connect(); drv.start_stream()
        self.assertEqual(snm.GetNode("StreamBufferCountManual").GetValue(), 0)
        drv.stop_stream(); drv.disconnect()

        fake2, drv2 = self._install()
        drv2.config.stream_buffer_count = 20
        snm2 = fake2._cam.GetTLStreamNodeMap()
        drv2.connect(); drv2.start_stream()
        self.assertEqual(snm2.GetNode("StreamBufferCountManual").GetValue(), 20)
        drv2.stop_stream(); drv2.disconnect()

    def test_link_throughput_limit_opt_in(self):
        # Default (None) leaves the device default; a set value is applied at
        # start_stream via DeviceLinkThroughputLimit (USB3 multi-cam bandwidth).
        fake, drv = self._install()
        drv.connect(); drv.start_stream()
        self.assertEqual(fake._cam.DeviceLinkThroughputLimit.GetValue(),
                         380000000)  # untouched default
        drv.stop_stream(); drv.disconnect()

        fake2, drv2 = self._install()
        drv2.config.link_throughput_limit_bps = 120000000
        drv2.connect(); drv2.start_stream()
        self.assertEqual(fake2._cam.DeviceLinkThroughputLimit.GetValue(),
                         120000000)
        drv2.stop_stream(); drv2.disconnect()

    def test_reset_to_defaults(self):
        fake, drv = self._install()
        drv.connect()
        drv.reset_to_defaults()
        self.assertTrue(fake._cam.UserSetLoad.executed)                  # loaded
        self.assertEqual(fake._cam.UserSetSelector.v, "UserSetSelector_Default")
        self.assertEqual(fake._cam.UserSetDefault.v, "UserSetDefault_Default")
        drv.disconnect()

    def test_reset_to_defaults_requires_connected(self):
        fake, drv = self._install()
        with self.assertRaises(CameraError):           # not connected -> raises
            drv.reset_to_defaults()

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
        self.assertTrue(drv._lost)               # flagged as device-lost

    def test_lost_device_teardown_skips_native_cleanup(self):
        # Hot-unplug: a backend fault marks the device lost; disconnect() must
        # NOT call EndAcquisition/DeInit on the dead handle (those SIGSEGV in the
        # real SDK). It should still release the System and not raise.
        fake, drv = self._install(getnext_errorcode=-1012)   # stream aborted
        drv.connect(); drv.start_stream()
        with self.assertRaises(CameraError):
            drv.read_frame(timeout=0.5)
        before = fake._released["count"]
        drv.disconnect()                          # must not raise / segfault
        self.assertEqual(fake._cam.endacq_calls, 0)   # skipped EndAcquisition
        self.assertEqual(fake._cam.deinit_calls, 0)   # skipped DeInit
        self.assertEqual(fake._released["count"], before + 1)  # System released
        self.assertEqual(drv.get_status(), CameraStatus.DISCONNECTED)
        self.assertFalse(drv._lost)               # flag cleared for next connect

    def test_connect_waits_for_writable_nodes(self):
        # Post-USB-re-enumeration the camera is briefly not-writable (-2006);
        # connect() must wait for writability, then apply config without raising.
        fake, drv = self._install()
        calls = {"n": 0}
        def flaky_writable(node):
            calls["n"] += 1
            return calls["n"] > 3            # not writable for the first polls
        fake.IsWritable = flaky_writable
        drv.connect()                        # must not raise; waits then applies
        self.assertEqual(drv.get_status(), CameraStatus.CONNECTED)
        self.assertGreater(calls["n"], 3)    # actually polled until writable
        self.assertEqual(fake._cam.Width.GetValue(), 1440)  # config applied
        drv.disconnect()

    def test_clean_teardown_does_call_native_cleanup(self):
        # Normal (not lost) disconnect must still do the full ordered teardown.
        fake, drv = self._install()
        drv.connect(); drv.start_stream()
        drv.disconnect()
        self.assertEqual(fake._cam.endacq_calls, 1)
        self.assertEqual(fake._cam.deinit_calls, 1)

    def test_timestamp_sync_off_by_default(self):
        # Default path is unchanged: no latch executed, no host time on frames.
        fake, drv = self._install()
        drv.connect(); drv.start_stream()
        frame = drv.read_frame(timeout=1.0)
        self.assertFalse(fake._cam.TimestampLatch.executed)
        self.assertNotIn("host_capture_time_s", frame.metadata)
        self.assertIsNone(drv.device_to_host_time(123456789))
        drv.stop_stream(); drv.disconnect()

    def test_timestamp_sync_opt_in_tags_host_time(self):
        # Opt-in: connect latches a device->host offset; each frame carries
        # host_capture_time_s on the host monotonic timebase.
        fake, drv = self._install()
        drv.config.timestamp_sync = True
        drv.connect()
        self.assertTrue(fake._cam.TimestampLatch.executed)   # latched at connect
        self.assertIsNotNone(drv._ts_offset_s)
        drv.start_stream()
        frame = drv.read_frame(timeout=1.0)
        self.assertIn("host_capture_time_s", frame.metadata)
        # host time == hw_timestamp_ns/1e9 + offset (the converter is the source)
        expected = drv.device_to_host_time(frame.hw_timestamp_ns)
        self.assertAlmostEqual(frame.metadata["host_capture_time_s"], expected)
        drv.stop_stream(); drv.disconnect()
        self.assertIsNone(drv._ts_offset_s)   # cleared on disconnect

    def test_convert_without_processor_degrades_not_crashes(self):
        # Spinnaker >= 4.x removed ImagePtr.Convert, so when self._processor is
        # None the legacy img.Convert() raises AttributeError (NOT a
        # SpinnakerException). read_frame must DEGRADE to the raw frame, never
        # let that escape and kill the worker thread (NFR-006). The fake Image
        # has no Convert method, reproducing the real-SDK AttributeError.
        fake, drv = self._install()
        drv.connect()
        drv._processor = None                 # force the legacy fallback branch
        drv.start_stream()
        frame = drv.read_frame(timeout=1.0)   # must not raise
        self.assertIsNotNone(frame.data)
        drv.stop_stream(); drv.disconnect()

    def test_gray_world_balance_equalizes_channels(self):
        # Pure-numpy white balance: a green-cast frame -> equal channel means.
        from vision.image_ops import gray_world_balance
        cast = np.dstack([np.full((4, 4), 40, np.uint8),
                          np.full((4, 4), 200, np.uint8),
                          np.full((4, 4), 60, np.uint8)])
        out = gray_world_balance(cast)
        means = out.reshape(-1, 3).mean(0)
        self.assertTrue(np.allclose(means, means[0], atol=1.0), means)

    def test_white_balance_gray_world_through_convert(self):
        # gray_world actually balances end-to-end through read_frame->_convert
        # (not just wiring): a green-cast conversion output -> equalized channel
        # means on the delivered frame.
        fake, drv = self._install()
        drv.config.white_balance = "gray_world"
        drv.connect()
        cast = np.dstack([np.full((8, 8), 40, np.uint8),
                          np.full((8, 8), 200, np.uint8),
                          np.full((8, 8), 60, np.uint8)])
        _Image = type(fake._cam.GetNextImage(0))   # the fake Image class

        class _CastProcessor:
            def Convert(self, img, target): return _Image(cast)
        drv._processor = _CastProcessor()
        drv.start_stream()
        means = drv.read_frame(timeout=1.0).data.reshape(-1, 3).mean(0)
        self.assertTrue(np.allclose(means, means[0], atol=1.0), means)
        drv.stop_stream(); drv.disconnect()

    def test_white_balance_ccm_applies_when_available(self):
        # white_balance=ccm builds CCM settings at connect (trial-validated) and
        # CreateColorCorrected is invoked per frame.
        fake, drv = self._install()
        drv.config.white_balance = "ccm"
        drv.config.ccm_color_temp = "LED_4649K"
        before = fake.ImageUtilityCCM.calls["count"]
        drv.connect()
        self.assertIsNotNone(drv._ccm_settings)      # built + trial-validated
        drv.start_stream(); drv.read_frame(timeout=1.0)
        self.assertGreater(fake.ImageUtilityCCM.calls["count"], before)
        drv.stop_stream(); drv.disconnect()

    def test_white_balance_ccm_unavailable_degrades(self):
        # ccm requested but the SDK lacks the CCM API -> disabled, plain debayer.
        fake, drv = self._install()
        delattr(fake, "ImageUtilityCCM")
        drv.config.white_balance = "ccm"
        drv.connect()
        self.assertIsNone(drv._ccm_settings)
        drv.start_stream()
        self.assertEqual(drv.read_frame(timeout=1.0).data.ndim, 3)  # no crash
        drv.stop_stream(); drv.disconnect()

    def test_color_algorithm_is_configurable(self):
        # color_algorithm selects the debayer algorithm on the processor.
        fake, drv = self._install()
        drv.config.color_algorithm = "RIGOROUS"
        drv.connect()
        self.assertEqual(drv._processor.algo,
                         "SPINNAKER_COLOR_PROCESSING_ALGORITHM_RIGOROUS")
        drv.disconnect()

    def test_chunk_telemetry_on_by_default_populates_metadata(self):
        # chunk_telemetry is ON by default (additive) -> the device's actual
        # per-frame exposure/gain/black land in frame.metadata. Relies on the
        # default, so it also guards the default value.
        fake, drv = self._install()
        drv.connect(); drv.start_stream()
        m = drv.read_frame(timeout=1.0).metadata
        self.assertEqual(m["chunk_exposure_us"], 5000.0)
        self.assertEqual(m["chunk_gain_db"], 1.5)
        self.assertEqual(m["chunk_black_level"], 2.0)
        drv.stop_stream(); drv.disconnect()

    def test_chunk_telemetry_opt_out(self):
        fake, drv = self._install()
        drv.config.chunk_telemetry = False
        drv.connect(); drv.start_stream()
        m = drv.read_frame(timeout=1.0).metadata
        self.assertNotIn("chunk_exposure_us", m)
        drv.stop_stream(); drv.disconnect()

    def test_chunk_crc_failure_flags_and_delivers(self):
        # Opt-in chunk_crc: a CRC-failed (but 'complete') frame is DELIVERED
        # (never dropped) with metadata["crc_ok"]=False and counted, so a
        # false-positive CRC can't starve the pipeline.
        fake, drv = self._install(crc_fail=True)
        drv.config.chunk_crc = True
        drv.connect(); drv.start_stream()
        frame = drv.read_frame(timeout=1.0)               # must NOT raise
        self.assertIs(frame.metadata["crc_ok"], False)
        self.assertEqual(drv.get_health().get("crc_failed_count"), 1)
        drv.stop_stream(); drv.disconnect()

    def test_chunk_crc_ok_frame_flagged_true(self):
        fake, drv = self._install()  # crc_fail=False
        drv.config.chunk_crc = True
        drv.connect(); drv.start_stream()
        self.assertIs(drv.read_frame(timeout=1.0).metadata["crc_ok"], True)
        drv.stop_stream(); drv.disconnect()

    def test_crc_not_flagged_when_chunk_crc_off(self):
        # chunk_crc explicitly off -> frame delivered, no crc_ok key.
        fake, drv = self._install(crc_fail=True)
        drv.config.chunk_crc = False
        drv.connect(); drv.start_stream()
        frame = drv.read_frame(timeout=1.0)
        self.assertNotIn("crc_ok", frame.metadata)
        drv.stop_stream(); drv.disconnect()

    def test_soft_reset_off_by_default(self):
        # Default: soft_reset is a no-op (no DeviceReset issued).
        fake, drv = self._install()
        drv.connect()
        self.assertFalse(drv.soft_reset())
        self.assertFalse(fake._cam.DeviceReset.executed)
        drv.disconnect()

    def test_soft_reset_opt_in_issues_device_reset(self):
        # Opt-in: soft_reset issues DeviceReset and flags the handle lost (the
        # device re-enumerates, so teardown must skip native cleanup).
        fake, drv = self._install()
        drv.config.soft_reset_on_fault = True
        drv.connect()
        self.assertTrue(drv.soft_reset())
        self.assertTrue(fake._cam.DeviceReset.executed)
        self.assertTrue(drv._lost)
        drv.disconnect()

    def test_read_retry_on_fault_retries_then_faults(self):
        # Opt-in read_retry_on_fault: a NON-fatal, non-timeout error is retried
        # before the backend fault is declared.
        fake, drv = self._install(getnext_errorcode=-9999)  # non-fatal transient
        drv.config.read_retry_on_fault = 2
        drv.connect(); drv.start_stream()
        with self.assertLogs("vision.spinnaker_driver", level="WARNING") as cm:
            with self.assertRaises(CameraError):
                drv.read_frame(timeout=0.5)
        self.assertEqual(sum("retrying" in m for m in cm.output), 2)
        self.assertTrue(drv._lost)   # exhausted -> conservative lost

    def test_device_fatal_code_is_not_retried(self):
        # A device-fatal code (ABORT) faults immediately even with retry budget.
        fake, drv = self._install(getnext_errorcode=-1012)  # SPINNAKER_ERR_ABORT
        drv.config.read_retry_on_fault = 5
        drv.connect(); drv.start_stream()
        with self.assertRaises(CameraError):
            drv.read_frame(timeout=0.5)
        self.assertTrue(drv._lost)

    def test_packet_resend_opt_in_runs(self):
        # Opt-in packet_resend applies without crashing (node present in fake).
        fake, drv = self._install()
        drv.config.packet_resend = True
        drv.connect(); drv.start_stream()
        self.assertEqual(drv.get_status(), CameraStatus.STREAMING)
        drv.stop_stream(); drv.disconnect()

    def test_disconnect_warns_on_images_in_use(self):
        # Lifecycle guard: a non-zero GetNumImagesInUse at teardown logs a WARN
        # (a consumer held a frame past release) but never blocks disconnect.
        fake, drv = self._install()
        drv.connect()
        fake._cam.images_in_use = 2
        with self.assertLogs("vision.spinnaker_driver", level="WARNING") as cm:
            drv.disconnect()
        self.assertTrue(any("in use" in m for m in cm.output))
        self.assertEqual(drv.get_status(), CameraStatus.DISCONNECTED)

    def test_connect_clears_camera_list_on_selection_failure(self):
        # Hardening: if GetBySerial/GetByIndex raises, the camera list is still
        # Clear()ed (try/finally) so it can't outlive its camera at teardown.
        fake, drv = self._install(serial="BFS1", select_raises=True)
        with self.assertRaises(CameraError):
            drv.connect()
        self.assertGreaterEqual(fake._cam_list["cleared"], 1)

    def test_incomplete_frame_is_malformed(self):
        fake, drv = self._install(incomplete=True)
        drv.connect(); drv.start_stream()
        with self.assertRaises(MalformedFrameError) as ctx:
            drv.read_frame(timeout=0.5)
        # NFR-006 diagnostics: the message carries the human-readable status
        # description (Image.GetImageStatusDescription), not just the enum int.
        self.assertIn("status-desc-9", str(ctx.exception))

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

    def test_get_health_reports_throughput_and_resulting_fps(self):
        # Bandwidth / actual-rate telemetry for the BFS (saturation diagnostic).
        fake, drv = self._install()
        drv.connect()
        health = drv.get_health()
        self.assertAlmostEqual(health["resulting_fps"], 60.0)
        self.assertAlmostEqual(health["link_throughput_bps"], 93312000.0)
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
