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
    Demand-driven live-view window styled like an industry live feed: a dark
    letterboxed video area, a translucent corner **HUD** (camera, resolution,
    FPS, temperature), and a bottom detail strip. A QTimer polls the FIFO at
    GUI_POLL_MS and repaints the freshest frame with its aspect ratio preserved;
    the producer never blocks on the GUI.

    With `health_fn` / `stats_fn` it shows live telemetry (display + camera FPS,
    device temperature, dropped/malformed/lost frames, reconnects), refreshed at
    ~2 Hz so the per-frame path stays cheap.
    """

    HUD_MARGIN = 12

    def __init__(self, fifo: FIFOFrameBuffer, title: str = "Camera",
                 poll_ms: int = GUI_POLL_MS,
                 health_fn: Optional[HealthFn] = None,
                 stats_fn: Optional[StatsFn] = None) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self._fifo = fifo
        self._health_fn = health_fn
        self._stats_fn = stats_fn

        # FPS / telemetry state (cached; refreshed ~2 Hz)
        self._frames_shown = 0
        self._anchor_t = time.monotonic()
        self._anchor_shown = 0
        self._disp_fps = 0.0
        self._cam_fps = 0.0
        self._prev_delivered: Optional[int] = None
        self._temp: Optional[float] = None
        self._exposure: Optional[float] = None   # us
        self._gain: Optional[float] = None        # dB
        self._malformed = 0
        self._reconnects = 0
        self._lost: Optional[int] = None
        self._last_refresh = 0.0

        # Video area — centered, aspect-preserved, on a dark (#101010) backdrop
        # so the letterbox bars look intentional (an ID selector avoids styling
        # the child HUD).
        self._label = QLabel("Waiting for frames…")
        self._label.setMinimumSize(320, 240)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        central = QWidget()
        central.setObjectName("videoArea")
        central.setStyleSheet("#videoArea { background-color: #101010; }")
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        self.setCentralWidget(central)

        # On-image HUD overlay (top-left OSD) — the headline live metrics.
        self._hud = QLabel(self._label)
        self._hud.setTextFormat(Qt.TextFormat.RichText)
        self._hud.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._hud.setStyleSheet(
            "QLabel { background-color: rgba(0, 0, 0, 160); color: #EAEAEA;"
            " padding: 6px 10px; border-radius: 6px; font-size: 12px;"
            " font-family: 'Menlo','Consolas','DejaVu Sans Mono',monospace; }")
        self._hud.move(self.HUD_MARGIN, self.HUD_MARGIN)
        self._hud.hide()

        self.setStatusBar(QStatusBar())   # bottom: per-frame + bookkeeping detail
        self.resize(960, 720)             # sensible 4:3 default

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
        self._update_hud(frame)
        self._update_status(frame)

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
        self._malformed = st.get("malformed_frames", self._malformed)
        self._reconnects = st.get("reconnects", self._reconnects)
        self._temp = health.get("temperature_c", self._temp)
        self._exposure = health.get("exposure_us", self._exposure)
        self._gain = health.get("gain_db", self._gain)
        if "StreamLostFrameCount" in health:
            self._lost = health["StreamLostFrameCount"]
        self._last_refresh = now

    def _update_hud(self, frame: CameraFrame) -> None:
        h, w = frame.data.shape[0], frame.data.shape[1]
        cam = f"&nbsp;&middot;&nbsp; CAM {self._cam_fps:.0f}" if self._stats_fn else ""
        temp = ""
        if self._temp is not None:
            color = ("#7CCB7C" if self._temp < 60 else
                     "#E6B53C" if self._temp < 75 else "#E05A5A")
            temp = (f'<br><span style="color:{color}">&#9679; '
                    f'{self._temp:.1f} &deg;C</span>')
        settings = ""
        if self._exposure is not None:
            settings = (f'<br><span style="color:#9AA0A6">exp '
                        f'{self._exposure / 1000:.1f} ms')
            if self._gain is not None:
                settings += f' &middot; gain {self._gain:.0f} dB'
            settings += "</span>"
        self._hud.setText(
            f'<b>{frame.camera_name or "camera"}</b>'
            f'<span style="color:#9AA0A6">&nbsp;&nbsp;{w}&times;{h}</span>'
            f'<br>GUI {self._disp_fps:.0f}{cam} fps{temp}{settings}')
        self._hud.adjustSize()
        self._hud.show()
        self._hud.raise_()

    def _update_status(self, frame: CameraFrame) -> None:
        detail = [f"frame {frame.frame_id}", f"age {frame.age * 1000:.0f} ms",
                  f"shown {self._frames_shown}",
                  f"gui-dropped {self._fifo.dropped}"]
        if self._stats_fn is not None:
            detail += [f"malformed {self._malformed}",
                       f"reconnects {self._reconnects}"]
        if self._lost is not None:
            detail.append(f"lost {self._lost}")
        self.statusBar().showMessage("    ·    ".join(detail))

    @staticmethod
    def _safe_call(fn: Optional[Callable[[], dict]]) -> dict:
        if fn is None:
            return {}
        try:
            return fn() or {}
        except Exception:   # telemetry must never break the live view
            return {}

    def closeEvent(self, event) -> None:
        self._timer.stop()
        super().closeEvent(event)
