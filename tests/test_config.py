"""
tests/test_config.py
FR-001 configuration ingestion (config_loader) and NFR-001/003/004 config
validation (CameraConfig.validate). Pure stdlib unittest.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import _helpers  # noqa: F401  (path bootstrap)
from _helpers import basic_config

from vision.camera_types import CameraFeature, PixelFormat
from vision.config_loader import (ConfigError, load_camera_config,
                                  load_camera_configs)


class TestConfigLoader(unittest.TestCase):  # UT-001
    def _write(self, obj):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        self.addCleanup(os.remove, path)
        return path

    def test_normal_load(self):
        path = self._write({
            "name": "c0", "model": "m", "max_resolution": [1024, 1024],
            "max_fps": 60.0, "features": ["GAIN", "EXPOSURE"],
            "output_pixel_format": "Mono8", "resolution": [800, 600],
            "fps": 60.0})
        cfg = load_camera_config(path)
        self.assertEqual(cfg.name, "c0")
        self.assertEqual(cfg.resolution, (800, 600))
        self.assertTrue(cfg.supports(CameraFeature.GAIN))
        self.assertEqual(cfg.output_pixel_format, PixelFormat.MONO8)

    def test_multi_camera(self):
        path = self._write({"cameras": [
            {"name": "a", "max_resolution": [512, 512]},
            {"name": "b", "max_resolution": [512, 512]}]})
        cfgs = load_camera_configs(path)
        self.assertEqual([c.name for c in cfgs], ["a", "b"])

    def test_missing_file_errors(self):  # error case
        with self.assertRaises(ConfigError):
            load_camera_config("/no/such/file.json")

    def test_malformed_file_errors(self):  # error case
        fd, path = tempfile.mkstemp(suffix=".json")
        os.write(fd, b"{ not valid json ]")
        os.close(fd)
        self.addCleanup(os.remove, path)
        with self.assertRaises(ConfigError):
            load_camera_config(path)

    def test_malformed_yaml_errors(self):  # error case: YAMLError -> ConfigError
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.write(fd, b"name: cam\n  bad: : indent")   # invalid YAML syntax
        os.close(fd)
        self.addCleanup(os.remove, path)
        with self.assertRaises(ConfigError):          # not a raw yaml.YAMLError
            load_camera_config(path)

    def test_unknown_feature_errors(self):  # error case
        path = self._write({"name": "x", "features": ["NOT_A_FEATURE"]})
        with self.assertRaises(ConfigError):
            load_camera_config(path)

    def test_duplicate_camera_name_errors(self):  # error case
        # Two cameras sharing a name would crash registration with a raw
        # ValueError mid-loop; load-time it's a clean ConfigError instead.
        path = self._write({"cameras": [
            {"name": "dup", "max_resolution": [512, 512]},
            {"name": "dup", "max_resolution": [512, 512]}]})
        with self.assertRaises(ConfigError):
            load_camera_configs(path)

    def test_bad_pixel_format_raises_config_error(self):  # regression
        path = self._write({"name": "x", "output_pixel_format": "BadFmt"})
        with self.assertRaises(ConfigError):
            load_camera_config(path)


class TestConfigValidation(unittest.TestCase):  # NFR-001/003/004
    def test_compliant_has_no_warnings(self):
        cfg = basic_config(max_resolution=(1024, 1024),
                           resolution=(1024, 1024), fps=60.0,
                           lens_fov_deg=60.0)
        self.assertEqual(cfg.validate(), [])

    def test_undersized_resolution_warns(self):  # NFR-003 boundary
        cfg = basic_config(max_resolution=(256, 256), resolution=(256, 256))
        self.assertTrue(any("NFR-003" in w for w in cfg.validate()))

    def test_low_fps_warns(self):  # NFR-001 boundary
        cfg = basic_config(max_fps=30.0, fps=30.0)
        self.assertTrue(any("NFR-001" in w for w in cfg.validate()))

    def test_narrow_fov_warns(self):  # NFR-004 boundary
        cfg = basic_config(lens_fov_deg=20.0)
        self.assertTrue(any("NFR-004" in w for w in cfg.validate()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
