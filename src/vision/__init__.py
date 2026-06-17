"""
vision — camera backend & centroid-extraction pipeline (SRS implementation).

Public API is re-exported here so callers can do, e.g.:

    from vision import CameraService, VisionSystem, CentroidExtractor

Submodules remain importable directly (e.g. `from vision.spinnaker_driver
import SpinnakerCameraDriver`) for the less common pieces.
"""

from __future__ import annotations

from .camera_driver import (CameraDriver, CameraError, CameraTimeoutError,
                            FeatureNotSupportedError, MalformedFrameError)
from .camera_service import CameraService
from .camera_types import (CameraConfig, CameraFeature, CameraFrame,
                           CameraStatus, PixelFormat)
from .centroid_buffer import CentroidRingBuffer, RxTimebase
from .centroid_extraction import CentroidExtractor, ExtractorParams
from .centroid_types import Centroid, CentroidProfile
from .config_loader import (ConfigError, load_camera_config,
                            load_camera_configs)
from .frame_buffers import CircularFrameBuffer, FIFOFrameBuffer, FrameSink
from .generic_driver import GenericCameraDriver
from .queuing_subsystem import QueuingSubsystem
from .vision_system import VisionSystem

__version__ = "1.0.0"

__all__ = [
    "CameraDriver", "CameraError", "CameraTimeoutError",
    "FeatureNotSupportedError", "MalformedFrameError",
    "CameraService",
    "CameraConfig", "CameraFeature", "CameraFrame", "CameraStatus",
    "PixelFormat",
    "CentroidRingBuffer", "RxTimebase",
    "CentroidExtractor", "ExtractorParams",
    "Centroid", "CentroidProfile",
    "ConfigError", "load_camera_config", "load_camera_configs",
    "CircularFrameBuffer", "FIFOFrameBuffer", "FrameSink",
    "GenericCameraDriver",
    "QueuingSubsystem",
    "VisionSystem",
]
