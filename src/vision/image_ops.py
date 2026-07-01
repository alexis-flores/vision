"""
image_ops.py
Small, dependency-light host image operations shared across the vision layers —
the driver's optional bake-in color path and the GUI's display-only correction.
Kept separate so gui_bridge does not import a camera backend.
"""

from __future__ import annotations

import numpy as np


def gray_world_balance(data: np.ndarray) -> np.ndarray:
    """Gray-world white balance: scale each channel so its mean matches the
    overall mean, neutralising the raw-Bayer green/yellow cast. Illuminant-
    agnostic and cheap (a few vectorised passes).

    Returns a NEW owned uint8 array, so it is safe on a read-only input (e.g. a
    shared CameraFrame buffer) — the source is never mutated. A near-black frame
    passes through (gain≈1) so sensor noise isn't amplified, and a mono (2-D)
    frame is returned unchanged.
    """
    if data.ndim != 3:
        return data
    f = data.astype(np.float32)
    means = f.reshape(-1, f.shape[2]).mean(axis=0)
    gray = float(means.mean())
    # np.maximum in the denominator avoids a 0/0 warning on a near-black frame.
    gains = np.where(means > 1e-3, gray / np.maximum(means, 1e-3),
                     1.0).astype(np.float32)
    return np.clip(f * gains, 0, 255).astype(np.uint8)
