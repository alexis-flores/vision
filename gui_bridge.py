"""
gui_bridge.py
GUI path for the vision system (FR-004, NFR-008, SRS 5.4 Figure 1).

The SRS specifies the GUI runs at a *lower* rate than acquisition and signals
when it is ready for the next frame ("33ms polling interval"). We implement
that with a QTimer in the GUI thread: every ~33 ms (30 FPS) it drains the
FIFO, keeps only the newest frame (discarding any backlog so the view never
lags), and repaints. The producer side only ever does a non-blocking push,
so visualization can never block the camera or cueing workers (NFR-008).

Under SRS v0.2 the vision system serves raw frames; the GUI is fed straight
from a FIFOFrameBuffer the camera service pushes into:

    service -> FIFOFrameBuffer -> CameraViewer

This module is the reusable **viewer component** (driver-agnostic): it displays
whatever frames the service fans into its FIFO, for any backend. To actually run
a camera through it, use `app.py` (it wires a backend + this viewer). Adding a
new camera backend needs no change here.

Requires: pip install PyQt6   (swap imports to PyQt5 if needed)
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QLabel, QMainWindow, QSizePolicy, QStatusBar,
                             QVBoxLayout, QWidget)

from vision.camera_types import CameraFrame
from vision.frame_buffers import FIFOFrameBuffer

GUI_POLL_MS = 33        # ~30 FPS visualization (SRS 5.4)
STATS_REFRESH_S = 0.5   # recompute FPS / read health+stats at ~2 Hz (cheap path)

# Optional callbacks the viewer polls for live telemetry, kept decoupled from
# CameraService (e.g. lambda: svc.get_health(cam) / lambda: svc.stats(cam)).
HealthFn = Callable[[], dict]
StatsFn = Callable[[], dict]


def frame_to_qimage(frame: CameraFrame) -> QImage:
    """Convert a CameraFrame ndarray to QImage (mono or BGR)."""
    data = np.ascontiguousarray(frame.data)
    if data.ndim == 2:  # mono
        if data.dtype != np.uint8:  # e.g. Mono16
            data = (data >> (data.itemsize * 8 - 8)).astype(np.uint8)
            data = np.ascontiguousarray(data)
        h, w = data.shape
        return QImage(data.data, w, h, w,
                      QImage.Format.Format_Grayscale8).copy()
    h, w, ch = data.shape  # color, assume BGR
    return QImage(data.data, w, h, ch * w,
                  QImage.Format.Format_BGR888).copy()


class CameraViewer(QMainWindow):
    """
    Demand-driven live-view window. A QTimer polls the FIFO at GUI_POLL_MS and
    repaints the freshest frame with its **aspect ratio preserved**; the
    producer never blocks on the GUI.

    If `health_fn` / `stats_fn` are supplied, the status bar also shows live
    telemetry — display + camera FPS, device temperature, dropped/malformed
    frames, reconnects — refreshed at ~2 Hz so the per-frame path stays cheap.
    """

    def __init__(self, fifo: FIFOFrameBuffer, title: str = "Camera",
                 poll_ms: int = GUI_POLL_MS,
                 health_fn: Optional[HealthFn] = None,
                 stats_fn: Optional[StatsFn] = None) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self._fifo = fifo
        self._health_fn = health_fn
        self._stats_fn = stats_fn

        # FPS / telemetry state
        self._frames_shown = 0
        self._anchor_t = time.monotonic()
        self._anchor_shown = 0
        self._disp_fps = 0.0
        self._cam_fps = 0.0
        self._prev_delivered: Optional[int] = None
        self._telemetry = ""
        self._last_refresh = 0.0

        self._label = QLabel("Waiting for frames...")
        self._label.setMinimumSize(320, 240)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self._label)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.resize(960, 720)            # sensible 4:3 default

        # The timer IS the "ready for next frame" signal (SRS 5.4).
        self._timer = QTimer(self)
        self._timer.setInterval(poll_ms)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    def _poll(self) -> None:
        # Drain backlog, keep only the freshest frame to avoid lag.
        frame = None
        while True:
            nxt = self._fifo.pop(timeout=0)   # non-blocking; None when drained
            if nxt is None:
                break
            frame = nxt
        if frame is None:
            return

        # Scale to the current label size, preserving aspect ratio (no stretch).
        pixmap = QPixmap.fromImage(frame_to_qimage(frame))
        self._label.setPixmap(pixmap.scaled(
            self._label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        self._frames_shown += 1

        now = time.monotonic()
        if now - self._last_refresh >= STATS_REFRESH_S:
            self._refresh_telemetry(now)
        self._show_status(frame)

    def _refresh_telemetry(self, now: float) -> None:
        st = self._safe_call(self._stats_fn)
        health = self._safe_call(self._health_fn)
        dt = now - self._anchor_t
        if dt >= 0.25:   # skip the tiny first window to avoid a noisy FPS spike
            self._disp_fps = (self._frames_shown - self._anchor_shown) / dt
            delivered = st.get("frames_delivered")
            if delivered is not None:
                if self._prev_delivered is not None:
                    self._cam_fps = (delivered - self._prev_delivered) / dt
                self._prev_delivered = delivered
            self._anchor_t = now
            self._anchor_shown = self._frames_shown
        self._telemetry = self._build_telemetry(st, health)
        self._last_refresh = now

    def _build_telemetry(self, st: dict, health: dict) -> str:
        parts = []
        if self._stats_fn is not None:
            parts.append(f"cam {self._cam_fps:.1f} fps")
            if st.get("malformed_frames"):
                parts.append(f"malformed={st['malformed_frames']}")
            if st.get("reconnects"):
                parts.append(f"reconnects={st['reconnects']}")
        if self._health_fn is not None:
            temp = health.get("temperature_c")
            if temp is not None:
                parts.append(f"{temp:.1f}°C")
            lost = health.get("StreamLostFrameCount")
            if lost:
                parts.append(f"lost={lost}")
        return "  ".join(parts)

    @staticmethod
    def _safe_call(fn: Optional[Callable[[], dict]]) -> dict:
        if fn is None:
            return {}
        try:
            return fn() or {}
        except Exception:   # telemetry must never break the live view
            return {}

    def _show_status(self, frame: CameraFrame) -> None:
        h, w = frame.data.shape[0], frame.data.shape[1]
        parts = [frame.camera_name or "camera", f"{w}x{h}",
                 f"frame={frame.frame_id}", f"{self._disp_fps:.1f} fps",
                 f"age={frame.age * 1000:.1f} ms", f"shown={self._frames_shown}"]
        if self._telemetry:
            parts.append(self._telemetry)
        self.statusBar().showMessage("   ".join(parts))

    def closeEvent(self, event) -> None:
        self._timer.stop()
        super().closeEvent(event)
