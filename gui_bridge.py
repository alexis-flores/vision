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

Requires: pip install PyQt6   (swap imports to PyQt5 if needed)
"""

from __future__ import annotations

import sys

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QLabel, QMainWindow, QStatusBar,
                             QVBoxLayout, QWidget)

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


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Standalone demo: simulated camera -> service -> live viewer (+ cueing)
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

if __name__ == "__main__":
    import logging

    from vision.camera_service import CameraService
    from vision.camera_types import CameraConfig, CameraFeature
    from vision.cueing_system import CueingSystem
    from vision.frame_buffers import CircularFrameBuffer
    from vision.generic_driver import GenericCameraDriver

    logging.basicConfig(level=logging.INFO)

    cfg = CameraConfig(
        name="sim0", model="SimCam", max_resolution=(640, 480),
        max_fps=60.0, fps=60.0,
        features=CameraFeature.FRAME_RATE | CameraFeature.RESOLUTION)

    svc = CameraService()
    svc.add_camera("sim0", GenericCameraDriver(cfg, n_spots=6))

    cueing_ring = CircularFrameBuffer(capacity=32)  # service -> cueing
    gui_fifo = FIFOFrameBuffer(capacity=2)          # service -> GUI
    svc.attach_sink("sim0", cueing_ring)
    svc.attach_sink("sim0", gui_fifo)

    cueing = CueingSystem(cueing_ring)  # consumes frames; pipeline out of scope

    svc.connect("sim0")
    cueing.start()
    svc.start_streaming("sim0")

    app = QApplication(sys.argv)
    win = CameraViewer(gui_fifo, title="sim0 - live")
    win.show()
    rc = app.exec()

    svc.stop_streaming("sim0")
    cueing.stop()
    svc.shutdown()
    sys.exit(rc)
