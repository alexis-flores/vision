"""
lens.py
Optics helpers for selecting/characterizing a lens against a sensor, following
Teledyne FLIR's "Selecting a Lens for your Camera" application note.

Two notions of "field of view" are kept distinct on purpose:

  * Angular FOV (degrees) — a fixed property of lens + sensor (far-field):
        angular_fov = 2 * atan(sensor / (2 * f))
    This is what the SRS NFR-004 (>=30 deg) and CameraConfig.lens_fov_deg use.

  * Linear FOV (mm at a working distance) — the scene extent actually covered,
    which depends on the working distance (FLIR app-note formula):
        exact:  f = sensor * WD / (FOV + sensor)   <=>  FOV = sensor*(WD - f)/f
        approx: f = sensor * WD / FOV              <=>  FOV = sensor * WD / f
    The approximation drops the sensor term in the denominator and is accurate
    when WD >> f (typical machine-vision standoff).

All distances are millimetres unless noted. `sensor` is one linear dimension of
the active sensor area (width for horizontal FOV, height for vertical).
"""

from __future__ import annotations

import math
from typing import Tuple


def sensor_dims_mm(width_px: int, height_px: int,
                   pixel_um: float) -> Tuple[float, float, float]:
    """Active sensor (width, height, diagonal) in mm from pixel geometry.

    Preferred over nominal fractional-inch sizes, which (per the FLIR note)
    don't scale directly to the real imaging-area dimensions.
    """
    w = width_px * pixel_um * 1e-3
    h = height_px * pixel_um * 1e-3
    return w, h, math.hypot(w, h)


def angular_fov_deg(sensor_mm: float, focal_mm: float) -> float:
    """Far-field angular field of view (degrees) for one sensor dimension."""
    if focal_mm <= 0:
        raise ValueError("focal_mm must be > 0")
    return math.degrees(2.0 * math.atan(sensor_mm / (2.0 * focal_mm)))


def linear_fov_mm(sensor_mm: float, focal_mm: float, working_dist_mm: float,
                  *, exact: bool = True) -> float:
    """Linear field of view (mm) covered at a given working distance."""
    if focal_mm <= 0:
        raise ValueError("focal_mm must be > 0")
    if exact:
        return sensor_mm * (working_dist_mm - focal_mm) / focal_mm
    return sensor_mm * working_dist_mm / focal_mm


def focal_length_mm(sensor_mm: float, working_dist_mm: float, fov_mm: float,
                    *, exact: bool = True) -> float:
    """Focal length (mm) to capture `fov_mm` at `working_dist_mm` (FLIR)."""
    if fov_mm <= 0:
        raise ValueError("fov_mm must be > 0")
    if exact:
        return sensor_mm * working_dist_mm / (fov_mm + sensor_mm)
    return sensor_mm * working_dist_mm / fov_mm


def working_distance_mm(sensor_mm: float, focal_mm: float, fov_mm: float,
                        *, exact: bool = True) -> float:
    """Working distance (mm) at which a lens yields `fov_mm` coverage."""
    if focal_mm <= 0:
        raise ValueError("focal_mm must be > 0")
    if exact:
        return focal_mm * (fov_mm + sensor_mm) / sensor_mm
    return focal_mm * fov_mm / sensor_mm


if __name__ == "__main__":
    # BFS-U3-16S2C-CS (Sony IMX273) + Kowa LM5JCM 5 mm C-mount.
    w_mm, h_mm, d_mm = sensor_dims_mm(1440, 1080, 3.45)
    f = 5.0
    print(f"Sensor: {w_mm:.3f} x {h_mm:.3f} mm  (diag {d_mm:.3f} mm)")
    print(f"Angular FOV @ f={f} mm: "
          f"H={angular_fov_deg(w_mm, f):.1f} deg  "
          f"V={angular_fov_deg(h_mm, f):.1f} deg  "
          f"D={angular_fov_deg(d_mm, f):.1f} deg")
    for wd in (2000.0, 3000.0, 4000.0):
        print(f"Linear FOV @ WD={wd/1000:.0f} m: "
              f"H={linear_fov_mm(w_mm, f, wd)/1000:.2f} m  "
              f"V={linear_fov_mm(h_mm, f, wd)/1000:.2f} m")
