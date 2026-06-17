"""
generic_driver.py
Template CameraDriver for "other" cameras (vendor SDKs, custom hardware,
network sources). Ships as a working *simulated* camera so the full stack —
service layer, buffers, GUI — can run with no hardware attached.

To adapt for real hardware: subclass GenericCameraDriver (or copy it) and
replace the _sim_* hooks with vendor SDK calls.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

import numpy as np

from .camera_driver import (CameraDriver, CameraError, CameraTimeoutError,
                           FeatureNotSupportedError, MalformedFrameError)
from .camera_types import (CameraConfig, CameraFeature, CameraFrame,
                          CameraStatus)


class GenericCameraDriver(CameraDriver):
    """
    Skeleton for arbitrary backends. By default it synthesizes frames with
    a few moving bright spots on a dark background, which gives the vision
    system real targets to extract and makes it useful for development, CI,
    and GUI testing.

    Fault-injection hooks (for NFR-005/NFR-006 failure tests):
        inject_backend_crash():  next read raises a fatal CameraError -> ERROR
        inject_malformed(n=1):   next n reads raise MalformedFrameError
    """

    def __init__(self, config: CameraConfig, n_spots: int = 5) -> None:
        super().__init__(config)
        self._attrs: dict[str, Any] = {} # simulated device registers
        self._frame_evt = threading.Event()
        self._stop_evt = threading.Event()
        self._gen_thread: Optional[threading.Thread] = None
        self._latest: Optional[CameraFrame] = None
        self._latest_lock = threading.Lock()
        self._n_spots = n_spots
        # Fault injection state
        self._crash_pending = False
        self._malformed_pending = 0

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Lifecycle — replace bodies with vendor SDK calls for real hardware
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def connect(self) -> None:
        if self.get_status() != CameraStatus.DISCONNECTED:
            return
        # --- vendor: open handle, e.g. sdk.open(self.config.serial) ---
        self._crash_pending = False # clear faults on (re)connect
        self._malformed_pending = 0
        self._attrs = {
            "fps": self.config.fps,
            "exposure_us": self.config.exposure_us or 10_000.0,
            "gain_db": self.config.gain_db or 0.0,
            "width": self.config.resolution[0],
            "height": self.config.resolution[1],
        }
        self._set_status(CameraStatus.CONNECTED)

    def disconnect(self) -> None:
        # --- vendor: close handle ---
        self._halt_generation() # stop thread from any state
        self._set_status(CameraStatus.DISCONNECTED)

    def start_stream(self) -> None:
        if self.get_status() == CameraStatus.STREAMING:
            return
        if self.get_status() != CameraStatus.CONNECTED:
            raise CameraError("start_stream requires CONNECTED status")
        # --- vendor: begin acquisition ---
        self._stop_evt.clear()
        self._gen_thread = threading.Thread(
            target=self._sim_generate, daemon=True,
            name=f"sim-{self.config.name}")
        self._gen_thread.start()
        self._set_status(CameraStatus.STREAMING)

    def stop_stream(self) -> None:
        if self.get_status() != CameraStatus.STREAMING:
            return
        # --- vendor: end acquisition ---
        self._halt_generation()
        self._set_status(CameraStatus.CONNECTED)

    def _halt_generation(self) -> None:
        self._stop_evt.set()
        if self._gen_thread is not None:
            self._gen_thread.join(timeout=2.0)
            self._gen_thread = None
        self._frame_evt.clear()

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Frames
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def read_frame(self, timeout: Optional[float] = None) -> CameraFrame:
        if self.get_status() != CameraStatus.STREAMING:
            raise CameraError("read_frame requires STREAMING status")
        # --- fault injection (failure-test hooks) ---
        if self._crash_pending:
            self._crash_pending = False
            self._set_status(CameraStatus.ERROR)
            raise CameraError("Injected backend crash")
        if self._malformed_pending > 0:
            self._malformed_pending -= 1
            raise MalformedFrameError("Injected malformed frame")
        if not self._frame_evt.wait(timeout=timeout):
            raise CameraTimeoutError(f"No frame within {timeout}s")
        self._frame_evt.clear()
        with self._latest_lock:
            if self._latest is None:
                raise CameraError("Frame event set but no frame available")
            return self._latest

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Fault-injection hooks (testing only)
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def inject_backend_crash(self) -> None:
        """Force the next read_frame() to fault the backend (NFR-005 test)."""
        self._crash_pending = True

    def inject_malformed(self, n: int = 1) -> None:
        """Force the next n read_frame() calls to return malformed (NFR-006)."""
        self._malformed_pending += n

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Configuration
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    _FEATURE_FOR_ATTR = {
        "fps": CameraFeature.FRAME_RATE,
        "exposure_us": CameraFeature.EXPOSURE,
        "gain_db": CameraFeature.GAIN,
        "width": CameraFeature.RESOLUTION,
        "height": CameraFeature.RESOLUTION,
    }

    def get_config(self, attribute: str) -> Any:
        if attribute not in self._attrs:
            raise CameraError(f"Unknown attribute {attribute!r}")
        # --- vendor: read register/node ---
        return self._attrs[attribute]

    def set_config(self, attribute: str, value: Any) -> None:
        feature = self._FEATURE_FOR_ATTR.get(attribute)
        if feature is not None and not self.config.supports(feature):
            raise FeatureNotSupportedError(
                f"{self.config.name} does not support {attribute}")
        if attribute not in self._attrs:
            raise CameraError(f"Unknown attribute {attribute!r}")
        # --- vendor: write register/node ---
        self._attrs[attribute] = value

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Simulation internals (delete when adapting to real hardware)
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def _sim_generate(self) -> None:
        period = 1.0 / float(self._attrs.get("fps") or 30.0)
        w = int(self._attrs["width"])
        h = int(self._attrs["height"])
        rng = np.random.default_rng(0)
        # Random per-spot orbit parameters: center, radius, angular speed.
        cx0 = rng.uniform(0.2, 0.8, self._n_spots) * w
        cy0 = rng.uniform(0.2, 0.8, self._n_spots) * h
        radius = rng.uniform(0.05, 0.2, self._n_spots) * min(w, h)
        omega = rng.uniform(0.5, 2.0, self._n_spots)
        phase = rng.uniform(0, 2 * np.pi, self._n_spots)
        sigma = rng.uniform(3.0, 6.0, self._n_spots)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        t0 = time.monotonic()
        while not self._stop_evt.is_set():
            t = time.monotonic() - t0
            # Dark background + small sensor noise.
            gray = rng.normal(8, 3, (h, w)).clip(0, 255).astype(np.float32)
            for i in range(self._n_spots):
                sx = cx0[i] + radius[i] * np.cos(omega[i] * t + phase[i])
                sy = cy0[i] + radius[i] * np.sin(omega[i] * t + phase[i])
                blob = 230.0 * np.exp(
                    -(((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * sigma[i] ** 2)))
                gray += blob
            gray = gray.clip(0, 255).astype(np.uint8)
            data = np.repeat(gray[:, :, None], 3, axis=2) # BGR8

            frame = CameraFrame(
                data=data,
                timestamp=time.monotonic(),
                frame_id=self._next_frame_id(),
                camera_name=self.config.name,
                pixel_format=self.config.pixel_format,
                metadata={"simulated": True, "n_spots": self._n_spots},
            )
            with self._latest_lock:
                self._latest = frame
            self._frame_evt.set()
            self._stop_evt.wait(timeout=period)
