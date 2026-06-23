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
    pixel_format: PixelFormat = PixelFormat.BGR8

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Extra backend-specific options (passed through verbatim)
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    extra: Dict[str, Any] = field(default_factory=dict)

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
        return warnings


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Camera frame
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

@dataclass
class CameraFrame:
    """
    Standard frame container shared across all drivers and consumers.
    """
    data: np.ndarray # Image data (H, W) or (H, W, C)
    timestamp: float # Host time, time.monotonic()
    frame_id: int # Monotonic per-stream counter
    camera_name: str = ""
    hw_timestamp_ns: Optional[int] = None # Device timestamp if available
    pixel_format: Optional[PixelFormat] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def age(self) -> float:
        """Seconds since this frame was captured (host clock)."""
        return time.monotonic() - self.timestamp
