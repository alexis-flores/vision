"""
camera_driver.py
Abstract camera driver interface. All concrete backends (Spinnaker, OpenCV,
custom) inherit from CameraDriver and override its methods.
"""

from __future__ import annotations

import abc
import threading
from typing import Any, Optional

from .camera_types import CameraConfig, CameraFrame, CameraStatus


class CameraError(Exception):
    """Base exception for camera driver failures."""


class CameraTimeoutError(CameraError):
    """read_frame() exceeded its timeout."""


class FeatureNotSupportedError(CameraError):
    """set_config()/get_config() called for a feature the camera lacks."""


class MalformedFrameError(CameraError):
    """
    A frame arrived but was incomplete/corrupt (NFR-006).

    This is intentionally distinct from CameraError so the service worker can
    *skip and continue* rather than treating it as a backend fault that
    triggers reconnection.
    """


class CameraDriver(abc.ABC):
    """
    Abstract base class for camera backends.

    Lifecycle:
        connect() -> start_stream() -> read_frame()* -> stop_stream() -> disconnect()

    Concrete drivers MUST override the abstract methods. Shared state
    (_status, _frame_counter, _lock) is provided here so subclasses behave
    consistently.
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._status = CameraStatus.DISCONNECTED
        self._frame_counter = 0
        self._lock = threading.Lock()

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Abstract interface - override in subclasses
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    @abc.abstractmethod
    def connect(self) -> None:
        """Acquire a backend handle to the physical camera."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Release the backend handle. Safe to call repeatedly."""

    @abc.abstractmethod
    def start_stream(self) -> None:
        """Begin acquisition / open the video stream."""

    @abc.abstractmethod
    def stop_stream(self) -> None:
        """End acquisition / close the video stream."""

    @abc.abstractmethod
    def read_frame(self, timeout: Optional[float] = None) -> CameraFrame:
        """
        Blocking read of the next frame.

        Args:
            timeout: Max seconds to wait. None blocks indefinitely.

        Raises:
            CameraTimeoutError: no frame arrived within `timeout`.
            CameraError: stream not active or backend failure.
        """

    @abc.abstractmethod
    def get_config(self, attribute: str) -> Any:
        """Read a live acquisition attribute from the device (e.g. 'gain_db')."""

    @abc.abstractmethod
    def set_config(self, attribute: str, value: Any) -> None:
        """
        Write an acquisition attribute to the device.

        Implementations should validate against self.config.features and
        raise FeatureNotSupportedError for unsupported attributes.
        """

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #           Shared concrete helpers
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def get_health(self) -> dict:
        """Best-effort device health telemetry (e.g. temperature, transport
        counters) as a flat dict of name -> value. Default: no telemetry.
        Backends that can read device registers override this."""
        return {}

    def reset_to_defaults(self) -> None:
        """Reset the camera to factory defaults (e.g. load the GenICam Default
        user set). Backends that support it override this; the default raises."""
        raise CameraError("reset_to_defaults is not supported by this backend")

    def get_status(self) -> CameraStatus:
        with self._lock:
            return self._status

    def _set_status(self, status: CameraStatus) -> None:
        with self._lock:
            self._status = status

    def _next_frame_id(self) -> int:
        with self._lock:
            self._frame_counter += 1
            return self._frame_counter

    # Context-manager sugar: `with driver:` connects/disconnects.
    def __enter__(self) -> "CameraDriver":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self.stop_stream()
        except CameraError:
            pass
        self.disconnect()

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name={self.config.name!r}, "
                f"status={self.get_status().name})")
