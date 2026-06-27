"""
camera_service.py
Management layer above the drivers. Owns one or more CameraDriver handles,
runs a streaming worker thread per camera, and fans frames out to any
number of registered sinks (circular buffer -> vision, FIFO -> GUI, ...).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .camera_driver import (CameraDriver, CameraError, CameraTimeoutError,
                           MalformedFrameError)
from .camera_types import CameraFrame, CameraStatus
from .frame_buffers import FrameSink

log = logging.getLogger(__name__)


@dataclass
class _CameraEntry:
    driver: CameraDriver
    sinks: List[FrameSink] = field(default_factory=list)
    worker: Optional[threading.Thread] = None
    stop_evt: threading.Event = field(default_factory=threading.Event)
    # Stats counters: written ONLY by this camera's single worker thread
    # (_stream_worker / _attempt_reconnect) and read by stats() from other
    # threads. Single-writer + atomic plain-int reads ⇒ GIL-safe, so no lock is
    # taken on the per-frame hot path. (Do not add a second writer.)
    frames_delivered: int = 0
    read_timeouts: int = 0
    malformed_frames: int = 0
    reconnects: int = 0
    sink_lock: threading.Lock = field(default_factory=threading.Lock)


class CameraService:
    """
    Facade for higher-level applications.

    Typical use:
        svc = CameraService()
        svc.add_camera("nav", driver)
        svc.attach_sink("nav", vision_ring)     # CircularFrameBuffer
        svc.attach_sink("nav", gui_fifo)        # FIFOFrameBuffer
        svc.connect("nav")
        svc.start_streaming("nav")
        ...
        svc.shutdown()
    """

    READ_TIMEOUT_S = 1.0 # per-iteration blocking read budget
    RECONNECT_DELAY_S = 0.5 # wait between reconnect attempts (NFR-005)
    # ~60 x 0.5s = a ~30s window, enough time for a human to replug a hot-removed
    # camera (and to ride out a USB re-enumeration); reconnects within ~0.5s once
    # the device is back. 0 == retry forever; >0 bounds the attempts.
    MAX_RECONNECT_ATTEMPTS = 60

    def __init__(self) -> None:
        self._cams: Dict[str, _CameraEntry] = {}
        self._lock = threading.Lock()

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Registration
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def add_camera(self, name: str, driver: CameraDriver) -> None:
        with self._lock:
            if name in self._cams:
                raise ValueError(f"Camera {name!r} already registered")
            self._cams[name] = _CameraEntry(driver=driver)
        log.info("Registered camera %r (%s)", name,
                 driver.__class__.__name__)

    def add_cameras_from_config(self, path: str, driver_factory) -> List[str]:
        """
        FR-001: ingest one or more camera configs from a file and register a
        driver per config. `driver_factory(config) -> CameraDriver` lets the
        caller choose the backend (Spinnaker / OpenCV / generic).

        NFR-001/003/004 compliance warnings from CameraConfig.validate() are
        logged at WARN but are non-fatal so undersized dev rigs still run.
        """
        from .config_loader import load_camera_configs # local import
        configs = load_camera_configs(path)
        names: List[str] = []
        for cfg in configs:
            for warning in cfg.validate():
                log.warning("Camera %r: %s", cfg.name, warning)
            self.add_camera(cfg.name, driver_factory(cfg))
            names.append(cfg.name)
        return names

    def remove_camera(self, name: str) -> None:
        self.stop_streaming(name)
        self.disconnect(name)
        with self._lock:
            self._cams.pop(name, None)

    def attach_sink(self, name: str, sink: FrameSink) -> None:
        """Register a frame consumer (any object with .push(frame))."""
        entry = self._entry(name)
        with entry.sink_lock:
            entry.sinks.append(sink)

    def detach_sink(self, name: str, sink: FrameSink) -> None:
        entry = self._entry(name)
        with entry.sink_lock:
            if sink in entry.sinks:
                entry.sinks.remove(sink)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Connection / configuration
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def connect(self, name: str) -> None:
        self._entry(name).driver.connect()

    def disconnect(self, name: str) -> None:
        self._entry(name).driver.disconnect()

    def connect_all(self) -> None:
        for name in self.camera_names():
            self.connect(name)

    def set_config(self, name: str, attribute: str, value) -> None:
        self._entry(name).driver.set_config(attribute, value)

    def get_config(self, name: str, attribute: str):
        return self._entry(name).driver.get_config(attribute)

    def get_status(self, name: str) -> CameraStatus:
        return self._entry(name).driver.get_status()

    def get_health(self, name: str) -> dict:
        """Device health telemetry (temperature, transport counters) for a
        camera. Empty dict if the backend exposes none."""
        return self._entry(name).driver.get_health()

    def reset_to_defaults(self, name: str) -> None:
        """Reset a camera to factory defaults (loads the Default user set).
        Must be CONNECTED and not streaming. Raises if the backend can't."""
        self._entry(name).driver.reset_to_defaults()

    def camera_names(self) -> List[str]:
        with self._lock:
            return list(self._cams.keys())

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Streaming
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def start_streaming(self, name: str) -> None:
        entry = self._entry(name)
        if entry.worker is not None and entry.worker.is_alive():
            return
        entry.driver.start_stream()
        entry.stop_evt.clear()
        entry.worker = threading.Thread(
            target=self._stream_worker, args=(name, entry),
            name=f"cam-worker-{name}", daemon=True)
        entry.worker.start()
        log.info("Streaming started for %r", name)

    def stop_streaming(self, name: str) -> None:
        entry = self._entry(name)
        # Idempotent: if there's no worker AND the driver isn't streaming, there
        # is nothing to stop. Return quietly so a redundant call (e.g. an
        # explicit stop_streaming() followed by shutdown(), which also stops)
        # doesn't emit a second "Streaming stopped" log. This never skips a real
        # stop_stream() — the guard requires status != STREAMING.
        if (entry.worker is None
                and entry.driver.get_status() != CameraStatus.STREAMING):
            return
        entry.stop_evt.set()
        if entry.worker is not None:
            entry.worker.join(timeout=3.0)
            if entry.worker.is_alive():
                # Worker didn't exit (e.g. stuck in a slow driver connect during
                # a reconnect). Surface it rather than silently abandoning the
                # thread; the driver stop/disconnect below still cleans up state.
                log.warning("Worker for %r did not exit within 3s; abandoning",
                            name)
            entry.worker = None
        if entry.driver.get_status() == CameraStatus.STREAMING:
            entry.driver.stop_stream()
        log.info("Streaming stopped for %r", name)

    def start_all(self) -> None:
        for name in self.camera_names():
            self.start_streaming(name)

    def shutdown(self) -> None:
        """Stop all streams and disconnect everything."""
        for name in self.camera_names():
            try:
                self.stop_streaming(name)
                self.disconnect(name)
            except CameraError as e:
                log.warning("Shutdown issue on %r: %s", name, e)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Stats
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def stats(self, name: str) -> dict:
        entry = self._entry(name)
        return {
            "status": entry.driver.get_status().name,
            "frames_delivered": entry.frames_delivered,
            "read_timeouts": entry.read_timeouts,
            "malformed_frames": entry.malformed_frames,
            "reconnects": entry.reconnects,
            "sinks": len(entry.sinks),
        }

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Internals
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def _entry(self, name: str) -> _CameraEntry:
        with self._lock:
            try:
                return self._cams[name]
            except KeyError:
                raise KeyError(f"Unknown camera {name!r}") from None

    def _stream_worker(self, name: str, entry: _CameraEntry) -> None:
        """
        Per-camera loop implementing the SRS 5.3 state machine:

          Normal operation : blocking read -> fan out to sinks
          Invalid input    : MalformedFrameError -> skip + WARN (NFR-006)
          Timeout/failure  : backend fault -> attempt reconnect, else log
                             ERROR and wait; never raise out (NFR-005)
          Shutdown         : stop_evt set -> exit
        """
        driver = entry.driver
        while not entry.stop_evt.is_set():
            try:
                frame: CameraFrame = driver.read_frame(
                    timeout=self.READ_TIMEOUT_S)
            except CameraTimeoutError:
                entry.read_timeouts += 1
                continue
            except MalformedFrameError as e: # NFR-006: skip + continue
                entry.malformed_frames += 1
                log.warning("Malformed frame on %r, skipping: %s", name, e)
                continue
            except CameraError as e: # backend fault
                log.error("Backend fault on %r: %s", name, e)
                if not self._attempt_reconnect(name, entry):
                    log.error("Reconnect abandoned for %r; worker idling",
                              name)
                    # Do not raise; wait for shutdown (5.3 "log error and wait")
                    entry.stop_evt.wait(self.RECONNECT_DELAY_S)
                    if entry.stop_evt.is_set():
                        break
                continue

            with entry.sink_lock:
                sinks = list(entry.sinks)
            for sink in sinks:
                try:
                    sink.push(frame)
                except Exception:
                    log.exception("Sink push failed on %r", name)
            entry.frames_delivered += 1
        log.debug("Worker for %r exited", name)

    def _attempt_reconnect(self, name: str, entry: _CameraEntry) -> bool:
        """
        NFR-005: try to re-establish the backend. Returns True if streaming
        was restored, False if attempts were exhausted. Never raises.
        """
        driver = entry.driver
        attempt = 0
        while not entry.stop_evt.is_set():
            attempt += 1
            if (self.MAX_RECONNECT_ATTEMPTS and
                    attempt > self.MAX_RECONNECT_ATTEMPTS):
                return False
            log.warning("Reconnect attempt %d for %r", attempt, name)
            try:
                driver.disconnect()
            except CameraError:
                pass
            if entry.stop_evt.wait(self.RECONNECT_DELAY_S):
                return False
            try:
                driver.connect()
                driver.start_stream()
                entry.reconnects += 1
                log.info("Reconnected %r after %d attempt(s)", name, attempt)
                return True
            except CameraError as e:
                log.error("Reconnect %d failed for %r: %s", attempt, name, e)
        return False
