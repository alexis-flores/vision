"""
spinnaker_driver.py
CameraDriver implementation for FLIR/Teledyne BlackFly cameras via the
Spinnaker SDK (PySpin). PySpin is imported lazily so the rest of the
package works on machines without the SDK installed.
"""

from __future__ import annotations

import atexit
import gc
import logging
import platform
import time
from typing import Any, Optional

import numpy as np

from .camera_driver import (CameraDriver, CameraError, CameraTimeoutError,
                           FeatureNotSupportedError, MalformedFrameError)
from .camera_types import (CameraConfig, CameraFeature, CameraFrame,
                          CameraStatus, PixelFormat)

log = logging.getLogger(__name__)

# Map our generic attribute names -> (GenICam node name, required feature)
_ATTR_MAP = {
    "gain_db":      ("Gain", CameraFeature.GAIN),
    "exposure_us":  ("ExposureTime", CameraFeature.EXPOSURE),
    "fps":          ("AcquisitionFrameRate", CameraFeature.FRAME_RATE),
    "width":        ("Width", CameraFeature.RESOLUTION),
    "height":       ("Height", CameraFeature.RESOLUTION),
    "black_level":  ("BlackLevel", CameraFeature.BLACK_LEVEL),
    "gamma":        ("Gamma", CameraFeature.GAMMA),
}


class SpinnakerCameraDriver(CameraDriver):
    """BlackFly / Spinnaker SDK adapter."""

    def __init__(self, config: CameraConfig) -> None:
        super().__init__(config)
        self._system: Any = None # PySpin.System singleton ref
        self._cam: Any = None # PySpin.CameraPtr
        self._pyspin: Any = None # module handle
        self._processor: Any = None # PySpin.ImageProcessor (host debayer/convert)
        self._lost = False # device removed/aborted mid-stream (hot-unplug)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Lifecycle
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def connect(self) -> None:
        if self.get_status() != CameraStatus.DISCONNECTED:
            return
        try:
            import PySpin # deferred import
        except ImportError as e:
            raise CameraError(
                "PySpin (Spinnaker SDK) is not installed") from e
        self._pyspin = PySpin

        try:
            self._system = PySpin.System.GetInstance()
            cam_list = self._system.GetCameras()
            if cam_list.GetSize() == 0:
                cam_list.Clear()
                raise CameraError("No Spinnaker cameras found")

            if self.config.serial:
                self._cam = cam_list.GetBySerial(self.config.serial)
            else:
                # Index-based selection is NOT stable across a USB
                # re-enumeration (unplug/replug or backend crash), so reconnect
                # (NFR-005) can bind to a stale/not-ready handle. Selecting by
                # serial deterministically re-binds to the same physical unit.
                log.warning(
                    "Camera %r has no 'serial' configured; selecting by "
                    "device_index=%d. Reconnect across USB re-enumeration "
                    "(NFR-005) may be unreliable — set 'serial' in the config "
                    "(or pass --serial) to bind to a specific unit.",
                    self.config.name, self.config.device_index)
                self._cam = cam_list.GetByIndex(self.config.device_index)
            cam_list.Clear()

            self._cam.Init()
            self._init_processor()
            self._lost = False  # fresh handle from re-enumeration; clear the flag
            # Mark CONNECTED before applying config: set_config()/_require_connected
            # require it, and the camera handle is already valid after Init().
            self._set_status(CameraStatus.CONNECTED)
            # Wait out the brief post-Init window where nodes are not yet writable
            # (matters on reconnect after a USB re-enumeration); no-op when ready.
            self._await_ready()
            self._apply_initial_config()
            # Safety net: if the process exits without a clean disconnect()
            # (unhandled exception, Ctrl-C), still DeInit the camera and release
            # the System singleton so the SDK doesn't crash at interpreter exit.
            atexit.register(self._release_on_exit)
        except PySpin.SpinnakerException as e:
            self._set_status(CameraStatus.ERROR)
            raise CameraError(f"Spinnaker connect failed: {e}") from e
        except CameraError:
            self._set_status(CameraStatus.ERROR)
            raise

    def disconnect(self) -> None:
        # If the device was lost mid-stream (hot-unplug), its transport is dead:
        # EndAcquisition()/DeInit() on the removed handle crash the SDK natively
        # (uncatchable SIGSEGV). Skip them and just drop our references; releasing
        # the System (after gc) reclaims everything safely. Otherwise do the full
        # ordered teardown.
        lost = self._lost
        if self.get_status() == CameraStatus.STREAMING:
            if lost:
                self._set_status(CameraStatus.CONNECTED)  # abandon the dead stream
            else:
                try:
                    self.stop_stream()
                except CameraError as e:  # cleanup must never raise
                    log.debug("stop_stream during disconnect: %s", e)
        if self._cam is not None:
            if not lost:
                try:
                    self._cam.DeInit()
                except Exception as e:  # cleanup must never raise
                    log.debug("DeInit during disconnect: %s", e)
            del self._cam
            self._cam = None
        self._processor = None
        # Force the native CameraPtr destructor to run BEFORE releasing the
        # System singleton. CPython refcounting frees it immediately when `del`
        # drops the last reference, but a lingering/cyclic reference (SWIG
        # wrappers, cached nodemaps) would otherwise outlive `del` and crash the
        # SDK when ReleaseInstance() tears down the C++ layer out of order.
        # disconnect() is a cold path, so the collect cost is irrelevant.
        gc.collect()
        if self._system is not None:
            try:
                self._system.ReleaseInstance()
            except Exception as e:  # cleanup must never raise
                log.debug("ReleaseInstance during disconnect: %s", e)
            self._system = None
        # Clean teardown done; the atexit safety net is no longer needed.
        atexit.unregister(self._release_on_exit)
        self._lost = False
        self._set_status(CameraStatus.DISCONNECTED)

    def _release_on_exit(self) -> None:
        """atexit safety net for an unclean exit (unhandled exception, Ctrl-C).
        Runs during interpreter shutdown, so it must never raise — it just
        attempts the normal ordered teardown."""
        try:
            self.disconnect()
        except Exception as e:  # shutdown cleanup must never raise
            log.debug("atexit cleanup error: %s", e)

    def start_stream(self) -> None:
        if self.get_status() == CameraStatus.STREAMING:
            return  # idempotent, matching generic_driver / opencv_driver
        self._require(CameraStatus.CONNECTED, "start_stream")
        try:
            self._configure_acquisition()
            self._cam.BeginAcquisition()
            self._set_status(CameraStatus.STREAMING)
        except self._pyspin.SpinnakerException as e:
            self._set_status(CameraStatus.ERROR)
            raise CameraError(f"BeginAcquisition failed: {e}") from e

    def stop_stream(self) -> None:
        if self.get_status() != CameraStatus.STREAMING:
            return
        try:
            self._cam.EndAcquisition()
        except self._pyspin.SpinnakerException as e:
            raise CameraError(f"EndAcquisition failed: {e}") from e
        finally:
            self._set_status(CameraStatus.CONNECTED)

    def reset_to_defaults(self) -> None:
        """Load the factory Default user set into the live registers AND make it
        the power-on default, so the camera reverts to factory settings and
        stays reset across power cycles. Requires CONNECTED (not streaming)."""
        self._require(CameraStatus.CONNECTED, "reset_to_defaults")
        PySpin = self._pyspin
        try:
            self._cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
            self._cam.UserSetLoad.Execute()
            log.info("Loaded factory Default user set")
            # Make Default the power-on set so future power-cycles also reset.
            const = getattr(PySpin, "UserSetDefault_Default", None)
            node = getattr(self._cam, "UserSetDefault", None)
            if node is not None and const is not None:
                try:
                    node.SetValue(const)
                    log.info("Set power-on default user set = Default")
                except PySpin.SpinnakerException as e:
                    log.debug("Could not set UserSetDefault: %s", e)
        except PySpin.SpinnakerException as e:
            raise CameraError(f"Reset to defaults failed: {e}") from e

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Frames
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def read_frame(self, timeout: Optional[float] = None) -> CameraFrame:
        self._require(CameraStatus.STREAMING, "read_frame")
        PySpin = self._pyspin
        timeout_ms = (PySpin.EVENT_TIMEOUT_INFINITE if timeout is None
                      else max(1, int(timeout * 1000)))
        try:
            img = self._cam.GetNextImage(timeout_ms)
        except PySpin.SpinnakerException as e:
            # Only a genuine timeout is a CameraTimeoutError (service retries).
            # Any other GetNextImage failure (e.g. device disconnected) is a
            # backend fault, which must trigger NFR-005 reconnect.
            timeout_code = getattr(PySpin, "SPINNAKER_ERR_TIMEOUT", -1011)
            if getattr(e, "errorcode", None) == timeout_code:
                raise CameraTimeoutError(
                    f"No frame within {timeout}s: {e}") from e
            # Non-timeout GetNextImage failure means the stream was aborted /
            # the device was removed (e.g. hot-unplug, ERR -1012). The handle's
            # transport is now dead: flag it so teardown SKIPS EndAcquisition /
            # DeInit, which would otherwise dereference the removed device and
            # SIGSEGV inside the SDK (a native crash Python cannot catch).
            self._lost = True
            raise CameraError(f"GetNextImage failed: {e}") from e

        try:
            if img.IsIncomplete():
                raise MalformedFrameError(
                    f"Incomplete image: {img.GetImageStatus()}")
            # Convert to the requested output format on the host. For color
            # cameras (e.g. BFS-U3-16S2C-CS, native BayerRG8) this debayers to
            # BGR8; for mono it is an inexpensive passthrough/copy.
            data = self._convert(img) # returns an owned copy
            hw_ts = img.GetTimeStamp() # ns, device clock
            device_fid = self._chunk_frame_id(img) # device counter (if chunk on)
        finally:
            img.Release()

        return CameraFrame(
            data=data,
            timestamp=time.monotonic(),
            frame_id=self._next_frame_id(),
            camera_name=self.config.name,
            hw_timestamp_ns=hw_ts,
            device_frame_id=device_fid,
            pixel_format=self.config.output_pixel_format,
        )

    def _chunk_frame_id(self, img) -> Optional[int]:
        """Device frame counter from chunk data, or None if chunk data is off /
        unsupported. Best-effort: must never break frame delivery."""
        try:
            fid = img.GetChunkData().GetFrameID()
            return int(fid) if fid is not None else None
        except Exception:  # chunk disabled/unsupported -> no device id
            return None

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Configuration
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def get_config(self, attribute: str) -> Any:
        self._require_connected("get_config")
        node_name, _ = self._resolve(attribute)
        node = getattr(self._cam, node_name, None)
        if node is None:
            raise CameraError(f"Node {node_name} not found")
        return node.GetValue()

    def set_config(self, attribute: str, value: Any) -> None:
        self._require_connected("set_config")
        node_name, feature = self._resolve(attribute)
        if not self.config.supports(feature):
            raise FeatureNotSupportedError(
                f"{self.config.name} does not support {attribute}")
        try:
            # Disable auto modes where relevant before manual writes.
            if attribute == "exposure_us":
                self._cam.ExposureAuto.SetValue(
                    self._pyspin.ExposureAuto_Off)
            elif attribute == "gain_db":
                self._cam.GainAuto.SetValue(self._pyspin.GainAuto_Off)
            elif attribute == "fps":
                self._cam.AcquisitionFrameRateEnable.SetValue(True)
            node = getattr(self._cam, node_name)
            node.SetValue(self._clamp(attribute, node, value))
        except self._pyspin.SpinnakerException as e:
            raise CameraError(
                f"Failed to set {attribute}={value}: {e}") from e

    def _clamp(self, attribute: str, node, value):
        """Clamp a numeric value to the node's [min, max] (per SDK examples).

        Avoids connect failures when a config value is slightly out of range
        (e.g. fps above the exposure-limited max). Non-numeric nodes pass through.
        """
        try:
            lo, hi = node.GetMin(), node.GetMax()
        except (AttributeError, self._pyspin.SpinnakerException):
            return value
        clamped = max(lo, min(hi, value))
        if clamped != value:
            log.warning("%s=%s out of range [%s, %s]; clamped to %s",
                        attribute, value, lo, hi, clamped)
        return clamped

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Health telemetry
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def get_health(self) -> dict:
        """Device temperature (°C) plus best-effort transport counters. Safe to
        call while streaming AND concurrently with a reconnect: it runs on the
        GUI / acceptance thread while the worker thread may clear self._cam in
        disconnect(), so we snapshot the handle once and use the local — a
        torn read can't NPE. Every node read is guarded too, so a missing node
        or a handle being torn down is a no-op, not a raise."""
        cam, pyspin = self._cam, self._pyspin
        if cam is None or pyspin is None:
            return {}
        health: dict = {}
        temp_node = getattr(cam, "DeviceTemperature", None)
        if temp_node is not None:
            try:
                health["temperature_c"] = float(temp_node.GetValue())
            except pyspin.SpinnakerException as e:
                log.debug("DeviceTemperature read failed: %s", e)
        # Transport/stream statistics live on the TL stream nodemap; node names
        # vary by camera/SDK, so read whichever are present.
        try:
            snm = cam.GetTLStreamNodeMap()
            for name in ("StreamLostFrameCount", "StreamDroppedFrameCount",
                         "StreamIncompleteFrameCount", "StreamFailedBufferCount"):
                val = self._read_int_node(snm, name)
                if val is not None:
                    health[name] = val
        except pyspin.SpinnakerException as e:
            log.debug("Stream statistics read failed: %s", e)
        # Live acquisition settings + bandwidth/rate telemetry (tuning feedback
        # for the GUI and a bandwidth-saturation diagnostic on a shared bus).
        # AcquisitionResultingFrameRate is the rate the device can actually
        # sustain given exposure + throughput limits — if it sits below the
        # requested fps you're exposure- or bandwidth-bound.
        for key, node_name in (("exposure_us", "ExposureTime"),
                               ("gain_db", "Gain"),
                               ("fps", "AcquisitionFrameRate"),
                               ("resulting_fps", "AcquisitionResultingFrameRate"),
                               ("link_throughput_bps",
                                "DeviceLinkCurrentThroughput")):
            node = getattr(cam, node_name, None)
            if node is None:
                continue
            try:
                health[key] = float(node.GetValue())
            except pyspin.SpinnakerException as e:
                log.debug("%s read failed: %s", node_name, e)
        return health

    def _read_int_node(self, nodemap, name: str) -> Optional[int]:
        PySpin = self._pyspin
        try:
            node = PySpin.CIntegerPtr(nodemap.GetNode(name))
            if PySpin.IsAvailable(node) and PySpin.IsReadable(node):
                return int(node.GetValue())
        except PySpin.SpinnakerException:
            return None
        return None

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Internals
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def _init_processor(self) -> None:
        """Create a host-side ImageProcessor for debayer/format conversion.

        ImageProcessor is the modern Spinnaker API (>= 2.x). On older SDKs the
        attribute is absent and read_frame() falls back to img.Convert().
        """
        PySpin = self._pyspin
        self._processor = None
        if hasattr(PySpin, "ImageProcessor"):
            try:
                proc = PySpin.ImageProcessor()
                algo = getattr(
                    PySpin,
                    "SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR", None)
                if algo is not None:
                    proc.SetColorProcessing(algo)
                self._processor = proc
            except PySpin.SpinnakerException as e:
                log.warning("ImageProcessor unavailable (%s); using legacy "
                            "img.Convert()", e)

    def _configure_acquisition(self) -> None:
        """Set acquisition mode and stream buffering for low-latency capture.

        Both are best-effort: we don't trust persisted user sets to be in
        continuous mode, and NewestOnly buffering avoids serving a stale
        backlog if a consumer falls behind.
        """
        PySpin = self._pyspin
        # On Linux/macOS, GEV cameras need StreamMode=Socket (no native filter
        # driver). The node is absent on USB3, so this is a guarded no-op there.
        if platform.system() in ("Linux", "Darwin"):
            try:
                snm = self._cam.GetTLStreamNodeMap()
                sm = PySpin.CEnumerationPtr(snm.GetNode("StreamMode"))
                if PySpin.IsAvailable(sm) and PySpin.IsWritable(sm):
                    entry = sm.GetEntryByName("Socket")
                    if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                        sm.SetIntValue(entry.GetValue())
            except PySpin.SpinnakerException as e:
                log.warning("Could not set StreamMode=Socket: %s", e)
        try:
            self._cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        except PySpin.SpinnakerException as e:
            log.warning("Could not set AcquisitionMode=Continuous: %s", e)
        try:
            snm = self._cam.GetTLStreamNodeMap()
            mode = PySpin.CEnumerationPtr(
                snm.GetNode("StreamBufferHandlingMode"))
            if PySpin.IsAvailable(mode) and PySpin.IsWritable(mode):
                entry = mode.GetEntryByName("NewestOnly")
                if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                    mode.SetIntValue(entry.GetValue())
        except PySpin.SpinnakerException as e:
            log.warning("Could not set StreamBufferHandlingMode=NewestOnly: %s",
                        e)
        # Optional manual stream-buffer pool size (default: leave the SDK default,
        # so the validated baseline is unchanged unless a config opts in).
        if self.config.stream_buffer_count:
            try:
                snm = self._cam.GetTLStreamNodeMap()
                cmode = PySpin.CEnumerationPtr(
                    snm.GetNode("StreamBufferCountMode"))
                if PySpin.IsAvailable(cmode) and PySpin.IsWritable(cmode):
                    entry = cmode.GetEntryByName("Manual")
                    if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                        cmode.SetIntValue(entry.GetValue())
                count = PySpin.CIntegerPtr(snm.GetNode("StreamBufferCountManual"))
                if PySpin.IsAvailable(count) and PySpin.IsWritable(count):
                    count.SetValue(int(self.config.stream_buffer_count))
                    log.info("Stream buffer count set to %d",
                             self.config.stream_buffer_count)
            except PySpin.SpinnakerException as e:
                log.warning("Could not set stream buffer count: %s", e)
        # Optional bandwidth cap (default: leave the device default untouched, so
        # a single-camera rig is unchanged). On a shared USB3 bus this partitions
        # bandwidth across cameras. Must be set before BeginAcquisition.
        if self.config.link_throughput_limit_bps:
            node = getattr(self._cam, "DeviceLinkThroughputLimit", None)
            if node is not None:
                try:
                    node.SetValue(self._clamp(
                        "link_throughput_limit_bps", node,
                        int(self.config.link_throughput_limit_bps)))
                    log.info("DeviceLinkThroughputLimit set to %d Bps",
                             self.config.link_throughput_limit_bps)
                except PySpin.SpinnakerException as e:
                    log.warning("Could not set DeviceLinkThroughputLimit: %s", e)
        self._enable_chunk_data()

    def _enable_chunk_data(self) -> None:
        """Enable GenICam chunk data so each frame carries the device frame
        counter (and timestamp). A gap in the device frame id is an
        authoritative dropped-frame signal (vs. inferring drops from timestamp
        gaps). Best-effort: cameras without chunk support stream normally and
        device_frame_id stays None."""
        PySpin = self._pyspin
        if not hasattr(self._cam, "ChunkModeActive"):
            return
        try:
            self._cam.ChunkModeActive.SetValue(True)
            for chunk in ("FrameID", "Timestamp"):
                sel = getattr(PySpin, f"ChunkSelector_{chunk}", None)
                if sel is None:
                    continue
                try:
                    self._cam.ChunkSelector.SetValue(sel)
                    self._cam.ChunkEnable.SetValue(True)
                except PySpin.SpinnakerException as e:
                    log.debug("Chunk %s not enabled: %s", chunk, e)
        except PySpin.SpinnakerException as e:
            log.warning("Could not enable chunk data: %s", e)

    def _spin_pixel_format(self, pf: Optional[PixelFormat]):
        """Map our PixelFormat enum to a PySpin PixelFormat_* constant."""
        if pf is None:
            return None
        PySpin = self._pyspin
        mapping = {
            PixelFormat.BGR8: "PixelFormat_BGR8",
            PixelFormat.RGB8: "PixelFormat_RGB8",
            PixelFormat.MONO8: "PixelFormat_Mono8",
            PixelFormat.MONO16: "PixelFormat_Mono16",
            PixelFormat.BAYER_RG8: "PixelFormat_BayerRG8",
        }
        return getattr(PySpin, mapping.get(pf, ""), None)

    def _convert(self, img):
        """Return an OWNED ndarray (copied) in the configured output format.

        GetNDArray() shares memory with its source image; we copy while that
        image is still referenced so the buffer cannot be freed underneath the
        array (a classic PySpin use-after-free). Falls back to the raw device
        array if no mapping or conversion path is available.
        """
        PySpin = self._pyspin
        target = self._spin_pixel_format(self.config.output_pixel_format)
        if target is None:
            return np.array(img.GetNDArray(), copy=True)
        try:
            if self._processor is not None:
                converted = self._processor.Convert(img, target)
            else:
                converted = img.Convert(target, getattr(PySpin, "HQ_LINEAR", 0))
            # copy before `converted` goes out of scope and frees its buffer
            return np.array(converted.GetNDArray(), copy=True)
        except PySpin.SpinnakerException as e:
            log.warning("Pixel-format conversion failed (%s); using raw frame",
                        e)
            return np.array(img.GetNDArray(), copy=True)

    def _await_ready(self, timeout_s: float = 2.0) -> None:
        """After Init() — especially on reconnect following a USB re-enumeration
        — config nodes like Width can be transiently NOT writable (GenICam
        AccessException, ERR -2006) while the camera firmware settles. Briefly
        wait for writability so _apply_initial_config() succeeds on the first
        try instead of failing and forcing extra reconnect attempts. No-op on a
        healthy camera, where the node is already writable."""
        PySpin = self._pyspin
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                if PySpin.IsWritable(self._cam.Width):
                    return
            except PySpin.SpinnakerException:
                pass
            time.sleep(0.05)
        log.debug("Config nodes still not writable after %.1fs; applying anyway",
                  timeout_s)

    def _apply_initial_config(self) -> None:
        cfg = self.config
        # Optional device-side source format (e.g. "BayerRG8" for color), so
        # the camera transmits its native format and the host debayers.
        dev_fmt = cfg.device_pixel_format
        if dev_fmt:
            try:
                const = getattr(self._pyspin, f"PixelFormat_{dev_fmt}")
                self._cam.PixelFormat.SetValue(const)
            except (AttributeError, self._pyspin.SpinnakerException) as e:
                log.warning("Could not set device PixelFormat=%s: %s",
                            dev_fmt, e)
        # Reset any persisted ROI offset before resizing: a full-frame
        # Width/Height can otherwise exceed (sensor - offset) and be rejected.
        for off in ("OffsetX", "OffsetY"):
            try:
                node = getattr(self._cam, off)
                node.SetValue(node.GetMin())   # min is the canonical reset (== 0)
            except (AttributeError, self._pyspin.SpinnakerException):
                pass
        if cfg.supports(CameraFeature.RESOLUTION) and cfg.resolution:
            self.set_config("width", cfg.resolution[0])
            self.set_config("height", cfg.resolution[1])
        # Exposure/gain BEFORE frame rate: the valid AcquisitionFrameRate range
        # depends on the exposure time, so the rate must be set last.
        if cfg.supports(CameraFeature.EXPOSURE) and cfg.exposure_us:
            self.set_config("exposure_us", cfg.exposure_us)
        if cfg.supports(CameraFeature.GAIN) and cfg.gain_db is not None:
            self.set_config("gain_db", cfg.gain_db)
        if cfg.supports(CameraFeature.FRAME_RATE) and cfg.fps:
            self.set_config("fps", cfg.fps)

    @staticmethod
    def _resolve(attribute: str):
        try:
            return _ATTR_MAP[attribute]
        except KeyError:
            raise CameraError(f"Unknown attribute {attribute!r}") from None

    def _require(self, status: CameraStatus, op: str) -> None:
        if self.get_status() != status:
            raise CameraError(
                f"{op} requires status {status.name}, "
                f"current: {self.get_status().name}")

    def _require_connected(self, op: str) -> None:
        if self.get_status() not in (CameraStatus.CONNECTED,
                                     CameraStatus.STREAMING):
            raise CameraError(f"{op} requires a connected camera")
