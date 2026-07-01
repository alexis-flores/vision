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
import time
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

    # --- opt-in host color correction (WB / CCM / gamma) ------------------ #

    def _color_driver(self, **cfg_kw):
        drv = SpinnakerCameraDriver(CameraConfig(
            name="bfs", output_pixel_format=PixelFormat.BGR8,
            device_pixel_format="BayerRG8", features=CameraFeature.RESOLUTION,
            **cfg_kw))
        drv._pyspin = PySpin
        drv._init_processor()
        return drv

    def test_real_ccm_supported_temp_enables(self):
        # A sensor-supported color temp -> CCM is built AND trial-validated on
        # the real SDK, and _convert yields a full-res 3-channel frame.
        drv = self._color_driver(white_balance="ccm", ccm_color_temp="LED_4649K")
        self.assertIsNotNone(drv._ccm_settings)
        out = drv._convert(self._bayer_image())
        self.assertEqual(out.shape, (1080, 1440, 3))

    def test_real_ccm_unsupported_temp_disables_gracefully(self):
        # The IMX273 rejects the GENERAL combo (verified -1003). The connect-time
        # trial must disable CCM so frames are a plain 3-channel debayer, NOT a
        # broken raw 2-D frame. This is the regression the real SDK revealed.
        drv = self._color_driver(white_balance="ccm", ccm_color_temp="GENERAL")
        self.assertIsNone(drv._ccm_settings)
        out = drv._convert(self._bayer_image())
        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape, (1080, 1440, 3))

    def test_real_gray_world_and_gamma_run(self):
        drv = self._color_driver(white_balance="gray_world", host_gamma=0.5)
        out = drv._convert(self._bayer_image())
        self.assertEqual(out.shape, (1080, 1440, 3))
        self.assertEqual(out.dtype, np.uint8)

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

    # --- reliability: error-code classification (named constants) --------- #

    def test_fatal_error_codes_resolve_on_real_sdk(self):
        # The device-fatal set is resolved by NAME against the installed SDK
        # (values differ across versions — never hardcode). TIMEOUT is NOT fatal.
        codes = self._driver()._fatal_error_codes()
        self.assertIn(PySpin.SPINNAKER_ERR_ABORT, codes)
        self.assertIn(PySpin.SPINNAKER_ERR_IO, codes)
        self.assertNotIn(PySpin.SPINNAKER_ERR_TIMEOUT, codes)

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


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Opt-in FEATURES on a real camera (HW-002)
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

@unittest.skipUnless(_spinnaker_camera_present(),
                     "PySpin + a physical BlackFly S camera not present")
class TestSpinnakerFeaturesHW(unittest.TestCase):
    """Exercises the opt-in features against a REAL camera, so `run the tests`
    confirms each actually works on the sensor (not just in mocks). Each test
    builds the BFS config with one feature enabled, connects, streams a few
    frames, and asserts the behaviour. Skips cleanly with no camera.

    Env: `VISION_TEST_SERIAL` binds a specific unit (else index 0);
    `VISION_TEST_SOFT_RESET=1` includes the disruptive DeviceReset check.
    """

    def _driver(self, **overrides) -> SpinnakerCameraDriver:
        """Fresh driver from the BFS config with `overrides` applied. Registers
        disconnect() cleanup so a failing assert still releases the camera."""
        cfg = load_camera_config(os.path.join(CONFIG_DIR, "bfs_u3_16s2c.json"))
        serial = os.environ.get("VISION_TEST_SERIAL")
        if serial:
            cfg.serial = serial
        for key, val in overrides.items():
            setattr(cfg, key, val)
        drv = SpinnakerCameraDriver(cfg)
        self.addCleanup(drv.disconnect)
        return drv

    def _frames(self, drv, n=3, timeout=2.0):
        drv.start_stream()
        try:
            return [drv.read_frame(timeout=timeout) for _ in range(n)]
        finally:
            if drv.get_status() == CameraStatus.STREAMING:
                drv.stop_stream()

    # --- timestamps ------------------------------------------------------- #

    def test_hw_timestamp_sync_tags_host_time(self):
        drv = self._driver(timestamp_sync=True)
        drv.connect()
        f = self._frames(drv, n=2)[-1]
        self.assertIn("host_capture_time_s", f.metadata)
        # on the host monotonic timebase -> close to 'now'
        self.assertLess(abs(f.metadata["host_capture_time_s"] - time.monotonic()),
                        5.0)

    # --- chunk data ------------------------------------------------------- #

    def test_hw_chunk_telemetry_reports_exposure(self):
        drv = self._driver()  # chunk_telemetry is default-on
        drv.connect()
        f = self._frames(drv)[-1]
        self.assertIn("chunk_exposure_us", f.metadata)
        self.assertGreater(f.metadata["chunk_exposure_us"], 0.0)

    def test_hw_chunk_crc_flags_and_never_starves(self):
        # CRC on: every frame is delivered and FLAGGED (crc_ok in metadata); a
        # healthy BFS reports crc_ok=True. Frames never drop, so CRC can't starve.
        drv = self._driver(chunk_crc=True)
        drv.connect()
        frames = self._frames(drv, n=5)
        self.assertEqual(len(frames), 5)
        self.assertIn("crc_ok", frames[-1].metadata)

    # --- host colour ------------------------------------------------------ #

    def test_hw_ccm_enables_and_delivers(self):
        drv = self._driver(white_balance="ccm", ccm_color_temp="LED_4649K")
        drv.connect()
        self.assertIsNotNone(drv._ccm_settings)  # built + trial-validated on SDK
        self.assertEqual(self._frames(drv)[-1].data.ndim, 3)

    def test_hw_gray_world_delivers(self):
        drv = self._driver(white_balance="gray_world")
        drv.connect()
        self.assertEqual(self._frames(drv)[-1].data.shape[2], 3)

    def test_hw_rgb8_output_labelled_and_shaped(self):
        drv = self._driver(output_pixel_format=PixelFormat.RGB8)
        drv.connect()
        f = self._frames(drv)[-1]
        self.assertEqual(f.data.shape[2], 3)
        self.assertEqual(f.pixel_format, PixelFormat.RGB8)

    # --- reliability knobs (start cleanly) -------------------------------- #

    def test_hw_packet_resend_and_read_retry_start_cleanly(self):
        drv = self._driver(packet_resend=True, read_retry_on_fault=2)
        drv.connect()
        f = self._frames(drv, n=2)[-1]  # must stream without error
        self.assertIsNotNone(f.data)

    # --- health ----------------------------------------------------------- #

    def test_hw_get_health_reports_temperature(self):
        drv = self._driver()
        drv.connect()
        health = drv.get_health()
        self.assertIn("temperature_c", health)
        self.assertGreater(health["temperature_c"], 0.0)

    # --- device reset (disruptive; opt-in via env var) -------------------- #

    @unittest.skipUnless(os.environ.get("VISION_TEST_SOFT_RESET") == "1",
                         "set VISION_TEST_SOFT_RESET=1 to run the DeviceReset check")
    def test_hw_soft_reset_reboots_and_recovers(self):
        drv = self._driver(soft_reset_on_fault=True)
        drv.connect()
        self.assertTrue(drv.soft_reset())        # issues DeviceReset, flags lost
        self.assertTrue(drv._lost)
        drv.disconnect()                         # skips native cleanup (lost)
        # The camera re-enumerates (DeviceReset takes seconds) — poll-reconnect.
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            try:
                drv.connect()
                break
            except CameraError:
                time.sleep(0.5)
        self.assertEqual(drv.get_status(), CameraStatus.CONNECTED,
                         "camera did not re-enumerate within 20s of DeviceReset")


if __name__ == "__main__":
    unittest.main(verbosity=2)
