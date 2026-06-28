"""
tests/test_hardware.py
The real-device tier for the BFS-U3-16S2C-CS over Spinnaker. Two layers, each
skipping cleanly so the software gate stays green on a dev machine:

  * TestSpinnakerRealSDK  — drives the ACTUALLY INSTALLED PySpin with NO camera
    (real host debayer through _convert, the real System lifecycle, real pixel
    -format/timeout constants, the SDK-4.x ImagePtr.Convert removal). Skips when
    PySpin is absent. Complements the symbol audit (scripts/check_pyspin_symbols.py)
    and the fully-mocked tests/test_drivers_mocked.py.
  * TestSpinnakerHardware — HW-001 smoke against a PHYSICAL camera (BayerRG8 ->
    BGR8 full-res color capture). Skips when no camera is present.

NOTE: PySpin's Image.Create(w,h,ox,oy,fmt,pData) REFERENCES the buffer; it does
not copy. The backing ndarray must outlive the Image or the SDK double-frees at
teardown. We therefore keep every backing array alive on the instance.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

import _helpers  # noqa: F401  (path bootstrap)
from _helpers import CONFIG_DIR

try:
    import PySpin  # type: ignore
    _HAVE_PYSPIN = True
except Exception:  # ImportError, or a numpy-ABI mismatch on import
    _HAVE_PYSPIN = False

from vision.camera_driver import CameraError
from vision.camera_types import (CameraConfig, CameraFeature, CameraStatus,
                                 PixelFormat)
from vision.config_loader import load_camera_config
from vision.spinnaker_driver import SpinnakerCameraDriver


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Real installed SDK, NO camera
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

@unittest.skipUnless(_HAVE_PYSPIN, "PySpin/Spinnaker SDK not installed")
class TestSpinnakerRealSDK(unittest.TestCase):
    """Exercise spinnaker_driver.py against the real SDK without a camera."""

    def setUp(self):
        # Keep backing buffers alive for the whole test (see module docstring).
        self._keepalive = []

    def _bayer_image(self, w=1440, h=1080):
        arr = np.random.default_rng(0).integers(0, 256, (h, w), np.uint8)
        self._keepalive.append(arr)
        return PySpin.Image.Create(w, h, 0, 0, PySpin.PixelFormat_BayerRG8, arr)

    def _mono_image(self, w=64, h=48):
        arr = np.zeros((h, w), np.uint8)
        self._keepalive.append(arr)
        return PySpin.Image.Create(w, h, 0, 0, PySpin.PixelFormat_Mono8, arr)

    def _driver(self, output=PixelFormat.BGR8, processor=None):
        drv = SpinnakerCameraDriver(CameraConfig(
            name="bfs", output_pixel_format=output,
            device_pixel_format="BayerRG8",
            features=CameraFeature.RESOLUTION))
        drv._pyspin = PySpin
        drv._processor = processor
        return drv

    # --- constants the driver's logic depends on -------------------------- #

    def test_required_constants_present_and_typed(self):
        self.assertEqual(PySpin.SPINNAKER_ERR_TIMEOUT, -1011)
        self.assertEqual(PySpin.EVENT_TIMEOUT_INFINITE, 2**64 - 1)
        for name in ("PixelFormat_BayerRG8", "PixelFormat_BGR8",
                     "PixelFormat_Mono8", "PixelFormat_RGB8",
                     "PixelFormat_Mono16", "AcquisitionMode_Continuous",
                     "ExposureAuto_Off", "GainAuto_Off",
                     "UserSetSelector_Default"):
            self.assertTrue(hasattr(PySpin, name), f"missing {name}")

    def test_spin_pixel_format_maps_every_enum(self):
        drv = self._driver()
        for pf in PixelFormat:
            self.assertIsNotNone(drv._spin_pixel_format(pf),
                                 f"{pf} did not map to a real PySpin constant")
        self.assertIsNone(drv._spin_pixel_format(None))

    # --- the real host debayer through the driver ------------------------- #

    def test_init_processor_builds_real_processor(self):
        drv = self._driver()
        drv._init_processor()
        self.assertIsNotNone(drv._processor)
        self.assertTrue(hasattr(drv._processor, "Convert"))

    def test_real_bayer_to_bgr_via_driver_convert(self):
        drv = self._driver(output=PixelFormat.BGR8)
        drv._init_processor()
        out = drv._convert(self._bayer_image())
        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape, (1080, 1440, 3))
        self.assertEqual(out.dtype, np.uint8)
        # _convert must return an OWNED copy (use-after-free guard).
        self.assertTrue(out.flags.owndata)

    def test_real_mono_passthrough_via_driver_convert(self):
        drv = self._driver(output=PixelFormat.MONO8)
        drv._init_processor()
        out = drv._convert(self._mono_image())
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(out.flags.owndata)

    def test_getndarray_copy_is_independent(self):
        # Validates the premise of the _convert use-after-free guard: a copy is
        # independent of the SDK-owned source buffer.
        drv = self._driver()
        drv._init_processor()
        bgr = drv._processor.Convert(self._bayer_image(), PySpin.PixelFormat_BGR8)
        copy = np.array(bgr.GetNDArray(), copy=True)
        self.assertTrue(copy.flags.owndata)

    # --- the SDK-4.x ImagePtr.Convert removal (the bug this suite found) --- #

    def test_imageptr_has_no_convert_on_modern_sdk(self):
        # On Spinnaker >= 4.x ImagePtr.Convert is gone; conversion is
        # ImageProcessor-only. This is WHY _convert must catch AttributeError.
        v = PySpin.System.GetInstance().GetLibraryVersion()
        if v.major >= 4:
            self.assertFalse(hasattr(PySpin.ImagePtr, "Convert"))
        self.assertFalse(issubclass(AttributeError, PySpin.SpinnakerException))

    def test_convert_without_processor_degrades_to_raw(self):
        # Regression: processor=None on a modern SDK -> legacy img.Convert()
        # raises AttributeError. _convert must degrade to the raw frame, never
        # raise (would otherwise kill the worker thread, NFR-006).
        drv = self._driver(output=PixelFormat.BGR8, processor=None)
        out = drv._convert(self._bayer_image())   # must not raise
        self.assertIsNotNone(out)
        self.assertEqual(out.dtype, np.uint8)

    # --- NFR-006 diagnostics: real image-status descriptions -------------- #

    def test_image_status_description_is_readable(self):
        # The driver turns an incomplete-frame status into actionable text via
        # the real Image.GetImageStatusDescription. Status 9 (DATA_INCOMPLETE)
        # is the classic USB3-bandwidth symptom on the BFS.
        status = PySpin.SPINNAKER_IMAGE_STATUS_DATA_INCOMPLETE
        desc = PySpin.Image.GetImageStatusDescription(status)
        self.assertIsInstance(desc, str)
        self.assertIn("incomplete", desc.lower())

    # --- the real System lifecycle on the no-camera path ------------------ #

    def test_no_camera_connect_raises_and_releases_cleanly(self):
        # Drives the real System.GetInstance -> GetCameras (size 0) ->
        # ReleaseInstance path. connect() must raise CameraError; disconnect()
        # must release the System and return to DISCONNECTED with no segfault.
        drv = SpinnakerCameraDriver(CameraConfig(
            name="bfs", serial="NOSUCHSERIAL",
            output_pixel_format=PixelFormat.BGR8))
        with self.assertRaises(CameraError):
            drv.connect()
        drv.disconnect()
        self.assertEqual(drv.get_status(), CameraStatus.DISCONNECTED)


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Real PHYSICAL camera (HW-001)
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
        cfg_path = os.path.join(CONFIG_DIR, "bfs_u3_16s2c.json")
        cfg = load_camera_config(cfg_path)
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
