"""
tests/test_lens.py
Lens / FOV calculators (NFR-004), validated against the FLIR "Selecting a Lens"
application note and the BFS-U3-16S2C-CS + Kowa 5 mm geometry.
"""

from __future__ import annotations

import unittest

import _helpers  # noqa: F401  (path bootstrap)

from vision.lens import (angular_fov_deg, focal_length_mm, linear_fov_mm,
                        sensor_dims_mm, working_distance_mm)


class TestLensCalculator(unittest.TestCase):
    def test_flir_worked_example(self):
        # FLIR note: 1/2" sensor (W=6.4mm), WD=100mm, FOV=50mm ->
        # 11.3mm exact, 12.8mm approximate.
        self.assertAlmostEqual(
            focal_length_mm(6.4, 100.0, 50.0, exact=True), 11.3, delta=0.1)
        self.assertAlmostEqual(
            focal_length_mm(6.4, 100.0, 50.0, exact=False), 12.8, delta=0.1)

    def test_imx273_sensor_dims(self):
        w, h, d = sensor_dims_mm(1440, 1080, 3.45)  # BFS-U3-16S2C-CS
        self.assertAlmostEqual(w, 4.968, places=3)
        self.assertAlmostEqual(h, 3.726, places=3)
        self.assertAlmostEqual(d, 6.210, places=2)

    def test_kowa_5mm_angular_fov(self):
        w, _, _ = sensor_dims_mm(1440, 1080, 3.45)
        # Matches CameraConfig.lens_fov_deg in config/bfs_u3_16s2c.json.
        self.assertAlmostEqual(angular_fov_deg(w, 5.0), 52.8, delta=0.2)

    def test_linear_fov_at_3m(self):
        w, h, _ = sensor_dims_mm(1440, 1080, 3.45)
        self.assertAlmostEqual(
            linear_fov_mm(w, 5.0, 3000.0, exact=True), 2975.8, delta=1.0)
        self.assertAlmostEqual(
            linear_fov_mm(h, 5.0, 3000.0, exact=True), 2231.9, delta=1.0)

    def test_round_trip(self):
        w, _, _ = sensor_dims_mm(1440, 1080, 3.45)
        fov = linear_fov_mm(w, 5.0, 3000.0, exact=True)
        self.assertAlmostEqual(
            focal_length_mm(w, 3000.0, fov, exact=True), 5.0, places=6)
        self.assertAlmostEqual(
            working_distance_mm(w, 5.0, fov, exact=True), 3000.0, places=3)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            angular_fov_deg(4.968, 0.0)
        with self.assertRaises(ValueError):
            focal_length_mm(6.4, 100.0, 0.0)
        with self.assertRaises(ValueError):
            linear_fov_mm(4.968, 0.0, 1000.0)
        with self.assertRaises(ValueError):
            working_distance_mm(4.968, 0.0, 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
