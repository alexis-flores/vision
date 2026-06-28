"""
tests/test_app.py
The entry-point scripts (app.py runner, hardware_acceptance.py) import cleanly and
parse args without a camera/PyQt/SDK, the runner drives multiple cameras, and the
clean error-exit + multi-camera override guard behave.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import _helpers  # noqa: F401  (path bootstrap; also puts repo root on sys.path)

from vision.camera_driver import CameraError


class TestScriptsImport(unittest.TestCase):
    def test_app_imports(self):
        import app
        # arg parsing works without a camera or any backend SDK.
        ns = app._parse_args(["--headless", "--seconds", "1"])
        self.assertTrue(ns.headless)
        self.assertEqual(ns.backend, "sim")          # default backend
        ns2 = app._parse_args(["--backend", "spinnaker", "--serial", "21512345"])
        self.assertEqual(ns2.backend, "spinnaker")
        self.assertEqual(ns2.serial, "21512345")
        self.assertFalse(ns.no_cueing)                         # default: cueing on
        self.assertTrue(app._parse_args(["--no-cueing"]).no_cueing)
        ns3 = app._parse_args(["--exposure", "12000", "--gain", "3", "--fps", "30"])
        self.assertEqual(ns3.exposure, 12000.0)
        self.assertEqual(ns3.gain, 3.0)
        self.assertEqual(ns3.fps, 30.0)
        self.assertFalse(ns.reset)                             # default: no reset
        self.assertTrue(app._parse_args(["--reset"]).reset)
        self.assertFalse(ns.rt)                                # default: no rt mode
        self.assertTrue(app._parse_args(["--rt"]).rt)

    def test_main_clean_exit_on_operational_error(self):
        # An expected backend failure (e.g. SDK missing / no camera) must exit
        # with code 1 and a clean message, NOT raise a traceback.
        import app
        with mock.patch.object(app, "_build_service",
                               side_effect=CameraError("no device")):
            rc = app.main(["--backend", "spinnaker", "--headless"])
        self.assertEqual(rc, 1)

    def test_runner_drives_multiple_cameras(self):  # multi-camera runner
        import app
        fd, path = tempfile.mkstemp(suffix=".json")
        os.write(fd, json.dumps({"cameras": [
            {"name": "m0", "max_resolution": [160, 120],
             "resolution": [160, 120], "fps": 60.0,
             "features": ["RESOLUTION", "FRAME_RATE"]},
            {"name": "m1", "max_resolution": [160, 120],
             "resolution": [160, 120], "fps": 60.0,
             "features": ["RESOLUTION", "FRAME_RATE"]}]}).encode())
        os.close(fd)
        try:
            with self.assertLogs("app", level="INFO") as cm:
                rc = app.main(["--backend", "sim", "--config", path,
                               "--headless", "--seconds", "1"])
            self.assertEqual(rc, 0)
            joined = "\n".join(cm.output)
            self.assertIn("Final m0", joined)     # both cameras ran + reported
            self.assertIn("Final m1", joined)
        finally:
            os.unlink(path)

    def test_multi_camera_config_ignores_cli_overrides(self):
        # One --serial can't bind N cameras: the runner warns and each camera
        # keeps its own config serial.
        import app
        fd, path = tempfile.mkstemp(suffix=".json")
        os.write(fd, json.dumps({"cameras": [
            {"name": "s0", "serial": "AAA", "max_resolution": [160, 120]},
            {"name": "s1", "serial": "BBB", "max_resolution": [160, 120]}]
        }).encode())
        os.close(fd)
        try:
            ns = app._parse_args(["--backend", "sim", "--config", path,
                                  "--serial", "ZZZ", "--headless"])
            with self.assertLogs("app", level="WARNING") as cm:
                svc, names = app._build_service(ns)
            try:
                self.assertEqual(set(names), {"s0", "s1"})
                self.assertTrue(any("ignored" in m for m in cm.output))
                self.assertEqual(svc._entry("s0").driver.config.serial, "AAA")
                self.assertEqual(svc._entry("s1").driver.config.serial, "BBB")
            finally:
                svc.shutdown()
        finally:
            os.unlink(path)

    def test_hardware_acceptance_imports(self):
        import hardware_acceptance
        ns = hardware_acceptance._parse_args(
            ["--serial", "21512345", "--seconds", "5", "--min-fps", "60",
             "--mono", "--no-hw-timestamp"])
        crit = hardware_acceptance._criteria(ns)
        self.assertEqual(ns.serial, "21512345")
        self.assertFalse(crit.require_color)        # --mono
        self.assertFalse(crit.require_hw_timestamp)  # --no-hw-timestamp
        self.assertEqual(crit.min_fps, 60.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
