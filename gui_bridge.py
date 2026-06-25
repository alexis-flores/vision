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

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QLabel, QMainWindow, QStatusBar, QVBoxLayout,
                             QWidget)

from vision.camera_types import CameraFrame
from vision.frame_buffers import FIFOFrameBuffer

GUI_POLL_MS = 33  # ~30 FPS visualization (SRS 5.4)


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
    Demand-driven live-view window. A QTimer polls the FIFO at GUI_POLL_MS;
    the producer never blocks on the GUI.
    """

    def __init__(self, fifo: FIFOFrameBuffer, title: str = "Camera",
                 poll_ms: int = GUI_POLL_MS) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self._fifo = fifo
        self._frames_shown = 0

        self._label = QLabel("Waiting for frames...")
        self._label.setMinimumSize(320, 240)
        self._label.setScaledContents(True)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self._label)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

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
        self._label.setPixmap(QPixmap.fromImage(frame_to_qimage(frame)))
        self._frames_shown += 1
        self.statusBar().showMessage(
            f"{frame.camera_name}  frame={frame.frame_id}  "
            f"age={frame.age * 1000:.1f} ms  shown={self._frames_shown}")

    def closeEvent(self, event) -> None:
        self._timer.stop()
        super().closeEvent(event)
