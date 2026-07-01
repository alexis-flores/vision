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

import os
import time
from typing import Callable, Optional

import numpy as np
from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QLabel, QMainWindow, QSizePolicy, QStatusBar,
                             QVBoxLayout, QWidget)

from vision.camera_types import CameraFrame, PixelFormat
from vision.frame_buffers import FIFOFrameBuffer
from vision.image_ops import gray_world_balance

GUI_POLL_MS = 33        # ~30 FPS visualization (SRS 5.4)
STATS_REFRESH_S = 0.5   # recompute FPS / read health+stats at ~2 Hz (cheap path)

# Optional brand logo overlay. Drop a PNG (transparent background recommended)
# at assets/logo.png and it appears in the top-right of the live view
# automatically; if the file is absent, nothing is shown.
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "assets", "logo.png")

# Optional callbacks the viewer polls for live telemetry, kept decoupled from
# CameraService (e.g. lambda: svc.get_health(cam) / lambda: svc.stats(cam)).
HealthFn = Callable[[], dict]
StatsFn = Callable[[], dict]


def frame_to_qimage(frame: CameraFrame) -> QImage:
    """Convert a CameraFrame to QImage, honouring its pixel_format so RGB8 and
    BGR8 (and mono) render with the right channel order."""
    return ndarray_to_qimage(frame.data,
                             rgb=frame.pixel_format == PixelFormat.RGB8)


def ndarray_to_qimage(data: np.ndarray, rgb: bool = False) -> QImage:
    """Convert an (H,W) mono or (H,W,3) colour ndarray to a QImage (deep-copied).
    `rgb` picks RGB888 vs BGR888 for 3-channel data (default BGR — the pipeline's
    usual output_pixel_format; the driver emits BGR8 for the BFS)."""
    data = np.ascontiguousarray(data)
    if data.ndim == 2:  # mono
        if data.dtype != np.uint8:  # e.g. Mono16
            data = (data >> (data.itemsize * 8 - 8)).astype(np.uint8)
            data = np.ascontiguousarray(data)
        h, w = data.shape
        # numpy's contiguous buffer (a memoryview) is a valid QImage data arg at
        # runtime; the PyQt6 stub only types `bytes`. QImage.copy() below deep-
        # copies, so the buffer can be freed — and we avoid a per-frame bytes()
        # copy. Hence the targeted call-overload ignore.
        return QImage(data.data, w, h, w,  # type: ignore[call-overload]
                      QImage.Format.Format_Grayscale8).copy()
    h, w, ch = data.shape  # colour
    fmt = QImage.Format.Format_RGB888 if rgb else QImage.Format.Format_BGR888
    return QImage(data.data, w, h, ch * w, fmt).copy()  # type: ignore[call-overload]


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
    LOGO_HEIGHT = 40        # logo overlay scaled to this px height (aspect kept)

    def __init__(self, fifo: FIFOFrameBuffer, title: str = "Camera",
                 poll_ms: int = GUI_POLL_MS,
                 health_fn: Optional[HealthFn] = None,
                 stats_fn: Optional[StatsFn] = None,
                 logo_path: Optional[str] = None,
                 display_white_balance: bool = True) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self._fifo = fifo
        self._health_fn = health_fn
        self._stats_fn = stats_fn
        # Display-only white balance: correct the raw-Bayer green cast for the
        # PREVIEW without touching the shared frame or the cueing path (this
        # viewer receives its own frame; the correction is applied to a copy on
        # the way to the screen). On already-balanced sources (webcam/sim) it is
        # ~a no-op. Off => show pixels exactly as delivered.
        self._display_wb = display_white_balance

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

        # Brand logo overlay (top-right), balancing the top-left HUD. The image
        # is drawn directly over the video with a transparent background (no
        # panel), so only the logo's own pixels show. Stays hidden unless a
        # readable image exists at `logo_path` (default: assets/logo.png) — a
        # missing file is a silent no-op.
        self._logo = QLabel(self._label)
        self._logo.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._logo.setStyleSheet("QLabel { background: transparent; }")
        self._logo.hide()
        self._load_logo(logo_path or LOGO_PATH)
        # Track the VIDEO LABEL's own resizes (not the window's) so the logo
        # re-pins on the initial layout settle too — at construction/first show
        # the label has no final width yet, so positioning off it here would land
        # the logo mid-top.
        self._label.installEventFilter(self)

        self._statusbar = QStatusBar()    # kept as a typed ref (statusBar()->Optional)
        self.setStatusBar(self._statusbar)  # bottom: per-frame + bookkeeping detail
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

        # Display-only white balance (a copy; never mutates the shared frame).
        # Gray-world is channel-order-agnostic, so it's applied before the
        # RGB/BGR format is chosen.
        data = frame.data
        if self._display_wb and data.ndim == 3:
            data = gray_world_balance(data)
        # Scale to the current label size, preserving aspect ratio (no stretch).
        rgb = frame.pixel_format == PixelFormat.RGB8
        pixmap = QPixmap.fromImage(ndarray_to_qimage(data, rgb=rgb))
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
        self._statusbar.showMessage("    ·    ".join(detail))

    def _load_logo(self, path: str) -> None:
        """Show the brand logo overlay if `path` is a readable image; otherwise
        leave it hidden, so a missing/unreadable file is a silent no-op rather
        than an error (the live view never depends on the logo)."""
        pix = QPixmap(path)
        if pix.isNull():
            return
        self._logo.setPixmap(pix.scaledToHeight(
            self.LOGO_HEIGHT, Qt.TransformationMode.SmoothTransformation))
        self._logo.adjustSize()
        self._logo.show()
        self._logo.raise_()
        self._reposition_logo()

    def _reposition_logo(self) -> None:
        """Pin the logo to the top-right of the video area, tracking resizes."""
        if self._logo.isHidden():
            return
        x = self._label.width() - self._logo.width() - self.HUD_MARGIN
        self._logo.move(max(self.HUD_MARGIN, x), self.HUD_MARGIN)
        self._logo.raise_()

    def eventFilter(self, obj, event) -> bool:
        # The video label's resize (incl. the first-show layout settle) is the
        # reliable signal that its width is final — re-pin the logo top-right
        # then. The window's own resizeEvent fires too early (label not yet sized).
        if obj is self._label and event.type() == QEvent.Type.Resize:
            self._reposition_logo()
        return super().eventFilter(obj, event)

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
