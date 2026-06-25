"""
tests/test_gui.py
Offscreen tests for the PyQt viewer (gui_bridge.CameraViewer). Render with the
Qt 'offscreen' platform so they run headless in CI; skip cleanly when PyQt6 is
not installed (it's the optional `gui` extra).
"""

from __future__ import annotations

import os
import sys
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # render with no display

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from vision.camera_types import CameraFrame
from vision.frame_buffers import FIFOFrameBuffer

try:
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication
    import gui_bridge
    _HAVE_QT = True
except Exception:                                     # pragma: no cover
    _HAVE_QT = False


def _frame(fid, color=True, w=160, h=120):
    data = (np.random.randint(0, 255, (h, w, 3), np.uint8) if color
            else np.random.randint(0, 255, (h, w), np.uint8))
    return CameraFrame(data=data, timestamp=time.monotonic(), frame_id=fid,
                       camera_name="cam")


@unittest.skipUnless(_HAVE_QT, "PyQt6 not installed")
class TestCameraViewer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One QApplication for the whole class (must outlive every QWidget).
        cls.app = QApplication.instance() or QApplication([])

    def _run(self, win, ms=200):
        win.show()
        QTimer.singleShot(ms, self.app.quit)
        self.app.exec()

    def test_aspect_ratio_preserved_not_stretched(self):
        # Regression: must NOT use setScaledContents (that stretches/distorts);
        # the viewer scales with KeepAspectRatio instead.
        win = gui_bridge.CameraViewer(FIFOFrameBuffer(2))
        self.assertFalse(win._label.hasScaledContents())
        win.close()

    def test_renders_with_telemetry(self):
        fifo = FIFOFrameBuffer(4)
        fifo.push(_frame(1, color=True))   # exercises the BGR path
        health = {"temperature_c": 41.5}
        stats = {"frames_delivered": 30, "malformed_frames": 0, "reconnects": 0}
        win = gui_bridge.CameraViewer(
            fifo, title="cam", poll_ms=10,
            health_fn=lambda: health, stats_fn=lambda: stats)
        self._run(win, ms=200)
        self.assertGreaterEqual(win._frames_shown, 1)
        msg = win.statusBar().currentMessage()
        self.assertIn("cam", msg)          # camera name
        self.assertIn("160x120", msg)      # resolution
        self.assertIn("fps", msg)          # display FPS
        self.assertIn("°C", msg)           # device temperature
        win.close()

    def test_works_without_callbacks_and_mono(self):
        fifo = FIFOFrameBuffer(4)
        fifo.push(_frame(7, color=False))  # exercises the mono path
        win = gui_bridge.CameraViewer(fifo, title="m", poll_ms=10)
        self._run(win, ms=150)
        self.assertGreaterEqual(win._frames_shown, 1)
        msg = win.statusBar().currentMessage()
        self.assertIn("frame=7", msg)
        self.assertNotIn("°C", msg)        # no health_fn -> no temperature
        win.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
