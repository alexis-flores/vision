"""
tests/_helpers.py
Shared test scaffolding. Importing this module FIRST does the src-layout path
bootstrap (so `from vision import ...` and `import app` work whether or not the
package is pip-installed), then exposes the fixtures used across several test
files. Per-file fakes (the PySpin/cv2 doubles, metric builders) stay local to
the file that needs them.

Every test module starts with `import _helpers  # noqa: F401` so the bootstrap
runs before any `vision`/`app` import.
"""

from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)                       # root scripts (app.py, ...)
sys.path.insert(0, os.path.join(_ROOT, "src"))  # the `vision` package (src layout)

from vision.camera_types import CameraConfig, CameraFeature  # noqa: E402

CONFIG_DIR = os.path.join(_ROOT, "config")


def wait_until(predicate, timeout=5.0, interval=0.01):
    """Poll until predicate() is true (or timeout), returning its final value.
    Replaces fixed sleeps so threaded tests don't false-fail under CPU load:
    the generous timeout absorbs slow scheduling, while a genuine failure still
    surfaces (the test's own assert runs afterward)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def basic_config(name="cam", **kw):
    """A minimal CameraConfig for the simulator/service tests."""
    defaults = dict(
        name=name, model="sim", max_resolution=(512, 512), max_fps=60.0,
        resolution=(512, 512), fps=60.0,
        features=(CameraFeature.GAIN | CameraFeature.EXPOSURE |
                  CameraFeature.FRAME_RATE | CameraFeature.RESOLUTION))
    defaults.update(kw)
    return CameraConfig(**defaults)
