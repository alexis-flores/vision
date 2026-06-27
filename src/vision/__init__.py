"""
vision — camera backend & frame-serving layer (SRS v0.2 implementation).

Scope (SRS v0.2): this package is the *vision system* — it configures the
camera, establishes a real-time stream, and serves frames to two consumers:
the cueing system (via a CircularFrameBuffer) and the GUI (via a FIFOFrameBuffer
that a QTimer polls on demand). Image processing / centroid extraction / tracking moved to the
downstream cueing system; `CueingSystem` here is a thin frame-consumer stand-in
for that subsystem (see cueing_system.py).

Public API is re-exported here so callers can do, e.g.:

    from vision import CameraService, CircularFrameBuffer, CueingSystem

Submodules remain importable directly (e.g. `from vision.spinnaker_driver
import SpinnakerCameraDriver`) for the less common pieces.
"""

from __future__ import annotations

from .camera_driver import (CameraDriver, CameraError, CameraTimeoutError,
                            FeatureNotSupportedError, MalformedFrameError)
from .camera_service import CameraService
from .camera_types import (CameraConfig, CameraFeature, CameraFrame,
                           CameraStatus, PixelFormat)
from .config_loader import (ConfigError, load_camera_config,
                            load_camera_configs)
from .cueing_system import CueingSystem, FrameProcessor
from .frame_buffers import CircularFrameBuffer, FIFOFrameBuffer, FrameSink
from .generic_driver import GenericCameraDriver

__version__ = "2.0.0"

__all__ = [
    "CameraDriver", "CameraError", "CameraTimeoutError",
    "FeatureNotSupportedError", "MalformedFrameError",
    "CameraService",
    "CameraConfig", "CameraFeature", "CameraFrame", "CameraStatus",
    "PixelFormat",
    "ConfigError", "load_camera_config", "load_camera_configs",
    "CueingSystem", "FrameProcessor",
    "CircularFrameBuffer", "FIFOFrameBuffer", "FrameSink",
    "GenericCameraDriver",
]
