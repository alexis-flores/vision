"""
camera_types.py
Shared data types for the camera abstraction layer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Enums
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

class CameraStatus(Enum):
    """Lifecycle state of a camera driver."""
    DISCONNECTED = auto() # No backend handle
    CONNECTED = auto() # Handle acquired, not streaming
    STREAMING = auto() # Acquisition active
    ERROR = auto() # Fault state; reconnect required


class CameraFeature(Flag):
    """
    Bitmask of configurable features a camera supports.
    Used by drivers to validate set_config() calls.
    """
    NONE = 0
    GAIN = auto()
    EXPOSURE = auto()
    FRAME_RATE = auto()
    RESOLUTION = auto()
    PIXEL_FORMAT = auto()
    TRIGGER = auto()
    WHITE_BALANCE = auto()
    GAMMA = auto()
    BLACK_LEVEL = auto()
    BINNING = auto()
    ROI = auto()


class PixelFormat(Enum):
    MONO8 = "Mono8"
    MONO16 = "Mono16"
    BGR8 = "BGR8"
    RGB8 = "RGB8"
    BAYER_RG8 = "BayerRG8"


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Camera configuration
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

@dataclass
class CameraConfig:
    """
    Static characteristics + desired acquisition settings for a camera.

    Capability fields describe what the hardware *can* do; the `features`
    flag declares which attributes are settable through set_config().
    Acquisition fields are the requested operating point applied on connect.
    """
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Identity / metadata
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    name: str = "camera"
    model: str = "unknown"
    serial: Optional[str] = None # Spinnaker serial, /dev path, etc.
    device_index: int = 0 # OpenCV index or enumeration index

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Hardware characteristics
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    max_resolution: Tuple[int, int] = (1920, 1080) # (width, height)
    max_fps: float = 30.0
    bit_depth: int = 8 # ADC bit depth
    dynamic_range_db: Optional[float] = None
    max_pixel_count: int = field(init=False)
    sensor_format: Optional[str] = None # e.g. "1/1.8\""

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Lens / optics (NFR-004)
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    lens_fov_deg: Optional[float] = None # horizontal FOV of mounted lens
    focal_length_mm: Optional[float] = None

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Supported features (validation mask for set_config)
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    features: CameraFeature = CameraFeature.NONE

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Requested acquisition settings (applied at connect/start)
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    resolution: Optional[Tuple[int, int]] = None
    fps: Optional[float] = None
    exposure_us: Optional[float] = None
    gain_db: Optional[float] = None
    # Two distinct pixel formats:
    #   device_pixel_format — what the CAMERA transmits over the wire (its real
    #     on-device PixelFormat register, e.g. "BayerRG8" so the host debayers;
    #     1 byte/px = lower USB bandwidth). None = leave the device's current
    #     format. Backend-specific string (GenICam name).
    #   output_pixel_format — what the host converts each frame TO for downstream
    #     consumers (e.g. BGR8). Drives host-side conversion only; never the
    #     device.
    device_pixel_format: Optional[str] = None
    output_pixel_format: PixelFormat = PixelFormat.BGR8
    # Optional stream-buffer pool size (Spinnaker StreamBufferCountManual).
    # None = SDK default (unchanged). Raise it when multiple cameras sharing a
    # USB3/PCIe link drop frames under bandwidth contention.
    stream_buffer_count: Optional[int] = None
    # Optional per-camera bandwidth cap in Bytes/s (Spinnaker
    # DeviceLinkThroughputLimit). None = device default (unchanged). On a
    # multi-camera USB3 rig this is THE knob to partition the shared bus: give
    # each camera a slice that sums to under the controller's capacity so none
    # starves and drops frames. Lower than what the requested fps needs will
    # reduce the achievable rate. A single camera needs no limit.
    link_throughput_limit_bps: Optional[int] = None
    # Opt-in device->host clock sync. When True the driver latches the device
    # timestamp against the host clock once at connect (BFS: TimestampLatch +
    # TimestampLatchValue) and tags each frame's metadata with
    # "host_capture_time_s" — the device hw_timestamp_ns expressed on the host
    # monotonic timebase, so downstream can correlate frames to host events /
    # other cameras. None/False = unchanged validated path (no latch, no tag).
    timestamp_sync: bool = False
    # Opt-in HOST color correction for the raw-Bayer→BGR debayer path (which
    # bypasses the on-camera ISP, so no white balance/CCM is applied → a slight
    # green/yellow cast). Applied host-side in the driver's _convert(), color
    # output only. Defaults leave the validated path unchanged.
    #   white_balance: "none"  = unchanged (device hw_timestamp path intact)
    #                  "gray_world" = cheap numpy per-channel gain (illuminant-
    #                     agnostic; kills the dominant green cast; ~µs/frame)
    #                  "ccm"  = Spinnaker ImageUtilityCCM color-correction matrix
    #                     for the sensor (colorimetric; per-frame SDK cost — watch
    #                     resulting_fps). Color is cosmetic for cueing; this mainly
    #                     improves the GUI preview.
    white_balance: str = "none"
    # CCM illuminant preset (used only when white_balance == "ccm"); maps to
    # PySpin SPINNAKER_CCM_COLOR_TEMP_<name>. NOTE: the IMX273 (BFS-16S2C) only
    # supports a specific set — verified on SDK 4.3: INCANDESCENT_2765K,
    # HALOGEN_3188K, FLUORESCENT_4665K, LED_4649K, LED_H_AND_E_4649K,
    # DAYLIGHT_5034K, DAYLIGHT_H_AND_E_5034K. Pick the one nearest your lighting;
    # LED_4649K suits typical lab LED. An unsupported combo is detected at connect
    # and CCM is disabled (falls back to plain debayer) rather than breaking frames.
    ccm_color_temp: str = "LED_4649K"
    # Host debayer algorithm; maps to SPINNAKER_COLOR_PROCESSING_ALGORITHM_<name>.
    # HQ_LINEAR is the balanced default; RIGOROUS / WEIGHTED_DIRECTIONAL_FILTER
    # are higher quality but slower (measure resulting_fps before adopting).
    color_algorithm: str = "HQ_LINEAR"
    # Optional HOST gamma (ImageProcessor.ApplyGamma) on color frames. None = off.
    # Named host_gamma to distinguish it from the DEVICE Gamma node, which is a
    # separate mechanism set via set_config("gamma") / the GAMMA feature.
    host_gamma: Optional[float] = None
    # Opt-in extended GenICam chunk data (default off = only FrameID+Timestamp).
    #   chunk_crc: enable the per-frame CRC chunk and reject frames that fail the
    #     CRC check as MalformedFrameError (NFR-006) — catches corruption that
    #     IsIncomplete() misses (a "complete" but corrupt frame).
    #   chunk_telemetry: enable ExposureTime/Gain/BlackLevel chunks and surface
    #     the device's ACTUAL per-frame values in frame.metadata (chunk_* keys).
    #     DEFAULT ON — purely additive metadata (delivered frames are unchanged),
    #     negligible cost, and useful ground-truth for downstream. Guarded, so a
    #     camera without those chunks is a no-op.
    chunk_crc: bool = False
    chunk_telemetry: bool = True
    # Opt-in reliability knobs (default off = validated path unchanged;
    # HARDWARE-VALIDATION-PENDING — real behavior needs the camera).
    #   soft_reset_on_fault: on a backend fault, issue a device reset (reboot the
    #     camera firmware) before reconnecting — recovers a wedged-but-present
    #     device without a physical unplug.
    #   read_retry_on_fault: retry GetNextImage this many times on a NON-fatal,
    #     non-timeout error before declaring a backend fault (0 = fault
    #     immediately, as before). Device-fatal codes never retry.
    #   packet_resend: enable USB3/GEV stream packet retransmission (guarded;
    #     largely a GEV feature and may be absent on USB3).
    soft_reset_on_fault: bool = False
    read_retry_on_fault: int = 0
    packet_resend: bool = False

    def __post_init__(self) -> None:
        self.max_pixel_count = self.max_resolution[0] * self.max_resolution[1]
        if self.resolution is None:
            self.resolution = self.max_resolution
        if self.fps is None:
            self.fps = self.max_fps

    def supports(self, feature: CameraFeature) -> bool:
        return bool(self.features & feature)

    def validate(self, *,
                 min_resolution: int = 512,
                 min_fps: float = 60.0,
                 min_fov_deg: float = 30.0) -> List[str]:
        """
        Check the requested operating point against SRS NFR targets.

        Returns a list of human-readable warnings (empty == fully compliant).
        Callers decide whether to treat warnings as fatal; the service logs
        them at WARN level so an undersized rig still runs for development.
        """
        warnings: List[str] = []
        w, h = self.resolution or self.max_resolution
        if min(w, h) < min_resolution: # NFR-003
            warnings.append(
                f"resolution {w}x{h} below NFR-003 minimum "
                f"{min_resolution}x{min_resolution}")
        if (self.fps or 0) < min_fps: # NFR-001
            warnings.append(
                f"fps {self.fps} below NFR-001 minimum {min_fps}")
        if self.lens_fov_deg is not None and \
                self.lens_fov_deg < min_fov_deg: # NFR-004
            warnings.append(
                f"lens FOV {self.lens_fov_deg}deg below NFR-004 "
                f"minimum {min_fov_deg}deg")
        if (self.resolution and
                (self.resolution[0] > self.max_resolution[0] or
                 self.resolution[1] > self.max_resolution[1])):
            warnings.append(
                f"requested resolution {self.resolution} exceeds sensor "
                f"max {self.max_resolution}")
        # Catch fat-fingered enum-like string options: an unrecognised value
        # otherwise SILENTLY disables the feature (no error), which is easy to
        # miss. These are advisory — the driver already fails safe.
        if self.white_balance not in ("none", "gray_world", "ccm"):
            warnings.append(
                f"white_balance {self.white_balance!r} unrecognised "
                "(expected none|gray_world|ccm); no correction will be applied")
        return warnings


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Camera frame
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

@dataclass(slots=True)
class CameraFrame:
    """
    Standard frame container shared across all drivers and consumers.

    slots=True (Python 3.10+): this is the per-frame hot object, so dropping the
    per-instance __dict__ trims allocation + memory at 60 fps. (Frames are
    treated as immutable; no code sets dynamic attributes on them.)
    """
    data: np.ndarray # Image data (H, W) or (H, W, C)
    timestamp: float # Host time, time.monotonic()
    frame_id: int # Monotonic per-stream counter (host-assigned)
    camera_name: str = ""
    hw_timestamp_ns: Optional[int] = None # Device timestamp if available
    device_frame_id: Optional[int] = None # Device frame counter (GenICam chunk);
                                          # gaps == authoritative dropped frames
    pixel_format: Optional[PixelFormat] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Frames fan out to multiple sinks (cueing ring + GUI FIFO) as the SAME
        # object sharing the SAME ndarray. Mark the pixel buffer read-only so an
        # in-place write by any consumer (e.g. a future cueing processor doing
        # `img -= bg`) fails LOUDLY here instead of silently corrupting what the
        # other sinks see. Consumers that need to modify pixels must copy first.
        # Best-effort: a buffer that's already read-only / can't toggle is fine.
        try:
            self.data.flags.writeable = False
        except (ValueError, AttributeError):
            pass

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def age(self) -> float:
        """Seconds since this frame was captured (host clock)."""
        return time.monotonic() - self.timestamp
