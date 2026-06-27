"""
opencv_driver.py
CameraDriver implementation for any camera reachable through OpenCV's
VideoCapture (UVC webcams, GStreamer pipelines, RTSP, V4L2, etc.).

OpenCV's grab model is pull-based with no native blocking-with-timeout,
so a lightweight capture thread feeds a latest-frame slot guarded by a
Condition; read_frame() waits on that condition with the caller's timeout.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import numpy as np

from .camera_driver import (CameraDriver, CameraError, CameraTimeoutError,
                           FeatureNotSupportedError)
from .camera_types import (CameraConfig, CameraFeature, CameraFrame,
                          CameraStatus)

log = logging.getLogger(__name__)

# attribute name -> (cv2 property id name, required feature)
_ATTR_MAP = {
    "fps":         ("CAP_PROP_FPS", CameraFeature.FRAME_RATE),
    "width":       ("CAP_PROP_FRAME_WIDTH", CameraFeature.RESOLUTION),
    "height":      ("CAP_PROP_FRAME_HEIGHT", CameraFeature.RESOLUTION),
    "exposure_us": ("CAP_PROP_EXPOSURE", CameraFeature.EXPOSURE),
    "gain_db":     ("CAP_PROP_GAIN", CameraFeature.GAIN),
    "brightness":  ("CAP_PROP_BRIGHTNESS", CameraFeature.NONE),
}


class OpenCVCameraDriver(CameraDriver):
    """Generic UVC/OpenCV adapter."""

    def __init__(self, config: CameraConfig, api_preference: int = 0) -> None:
        super().__init__(config)
        self._cap: Any = None
        self._cv2: Any = None
        self._api_preference = api_preference # e.g. cv2.CAP_V4L2

        self._capture_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._cond = threading.Condition()
        self._latest: Optional[CameraFrame] = None
        self._last_delivered_id = 0

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Lifecycle
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def connect(self) -> None:
        if self.get_status() != CameraStatus.DISCONNECTED:
            return
        try:
            import cv2
        except ImportError as e:
            raise CameraError("opencv-python is not installed") from e
        self._cv2 = cv2

        source: Any = (self.config.serial
                       if self.config.serial else self.config.device_index)
        self._cap = cv2.VideoCapture(source, self._api_preference) \
            if self._api_preference else cv2.VideoCapture(source)

        if not self._cap.isOpened():
            self._cap = None
            self._set_status(CameraStatus.ERROR)
            raise CameraError(f"Could not open OpenCV source {source!r}")

        self._apply_initial_config()
        self._set_status(CameraStatus.CONNECTED)

    def disconnect(self) -> None:
        if self.get_status() == CameraStatus.STREAMING:
            self.stop_stream()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._set_status(CameraStatus.DISCONNECTED)

    def start_stream(self) -> None:
        if self.get_status() == CameraStatus.STREAMING:
            return
        if self.get_status() != CameraStatus.CONNECTED:
            raise CameraError("start_stream requires CONNECTED status")
        self._stop_evt.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"opencv-capture-{self.config.name}",
            daemon=True,
        )
        self._capture_thread.start()
        self._set_status(CameraStatus.STREAMING)

    def stop_stream(self) -> None:
        if self.get_status() != CameraStatus.STREAMING:
            return
        self._stop_evt.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            if self._capture_thread.is_alive():
                log.warning("Capture thread for %r did not exit within 2s",
                            self.config.name)
            self._capture_thread = None
        with self._cond:
            self._latest = None
            self._cond.notify_all()
        self._set_status(CameraStatus.CONNECTED)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Frames
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def read_frame(self, timeout: Optional[float] = None) -> CameraFrame:
        if self.get_status() != CameraStatus.STREAMING:
            raise CameraError("read_frame requires STREAMING status")

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                if (self._latest is not None
                        and self._latest.frame_id > self._last_delivered_id):
                    frame = self._latest
                    self._last_delivered_id = frame.frame_id
                    return frame
                remaining = (None if deadline is None
                             else deadline - time.monotonic())
                if remaining is not None and remaining <= 0:
                    raise CameraTimeoutError(
                        f"No frame within {timeout}s")
                if not self._cond.wait(timeout=remaining):
                    raise CameraTimeoutError(
                        f"No frame within {timeout}s")
                if self.get_status() != CameraStatus.STREAMING:
                    raise CameraError("Stream stopped while waiting")

    def _capture_loop(self) -> None:
        while not self._stop_evt.is_set():
            ok, data = self._cap.read()
            if not ok:
                self._set_status(CameraStatus.ERROR)
                with self._cond:
                    self._cond.notify_all()
                return
            frame = CameraFrame(
                data=np.asarray(data),
                timestamp=time.monotonic(),
                frame_id=self._next_frame_id(),
                camera_name=self.config.name,
                pixel_format=self.config.output_pixel_format,
            )
            with self._cond:
                self._latest = frame
                self._cond.notify_all()

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Configuration
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def get_config(self, attribute: str) -> Any:
        self._require_open("get_config")
        prop_name, _ = self._resolve(attribute)
        return self._cap.get(getattr(self._cv2, prop_name))

    def set_config(self, attribute: str, value: Any) -> None:
        self._require_open("set_config")
        prop_name, feature = self._resolve(attribute)
        if feature is not CameraFeature.NONE and \
                not self.config.supports(feature):
            raise FeatureNotSupportedError(
                f"{self.config.name} does not support {attribute}")
        if not self._cap.set(getattr(self._cv2, prop_name), float(value)):
            raise CameraError(f"Backend rejected {attribute}={value}")

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Internals
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def _apply_initial_config(self) -> None:
        cfg = self.config
        if cfg.supports(CameraFeature.RESOLUTION) and cfg.resolution:
            self.set_config("width", cfg.resolution[0])
            self.set_config("height", cfg.resolution[1])
        if cfg.supports(CameraFeature.FRAME_RATE) and cfg.fps:
            self.set_config("fps", cfg.fps)

    @staticmethod
    def _resolve(attribute: str):
        try:
            return _ATTR_MAP[attribute]
        except KeyError:
            raise CameraError(f"Unknown attribute {attribute!r}") from None

    def _require_open(self, op: str) -> None:
        if self._cap is None:
            raise CameraError(f"{op} requires a connected camera")
