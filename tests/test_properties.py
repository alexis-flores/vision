"""
tests/test_properties.py
Property-based tests (Hypothesis): assert invariants hold over a wide space of
generated inputs, catching edge cases hand-written cases miss. Skipped cleanly
if Hypothesis is not installed.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

try:
    from hypothesis import given, settings, HealthCheck
    from hypothesis import strategies as st
    from hypothesis.extra import numpy as hnp
    _HAVE_HYP = True
except ImportError:                                   # pragma: no cover
    _HAVE_HYP = False

from vision.centroid_extraction import CentroidExtractor, ExtractorParams
from vision.centroid_buffer import CentroidRingBuffer
from vision.centroid_types import CentroidProfile


@unittest.skipUnless(_HAVE_HYP, "hypothesis not installed")
class TestExtractorProperties(unittest.TestCase):
    @settings(max_examples=150, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    @given(hnp.arrays(
        dtype=np.uint8,
        shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=8, max_side=160)))
    def test_mono_never_crashes_and_invariants(self, img):
        p = ExtractorParams(threshold=60, min_area=4, max_centroids=64)
        cents = CentroidExtractor(p).extract(img)
        self.assertIsInstance(cents, list)
        self.assertLessEqual(len(cents), p.max_centroids)
        h, w = img.shape
        for c in cents:
            self.assertTrue(0.0 <= c.x <= w)
            self.assertTrue(0.0 <= c.y <= h)
            self.assertGreaterEqual(c.area, p.min_area)
            self.assertGreaterEqual(c.intensity, 0.0)

    @settings(max_examples=80, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    @given(hnp.arrays(
        dtype=np.uint8,
        shape=st.tuples(st.integers(8, 96), st.integers(8, 96), st.just(3))))
    def test_color_never_crashes(self, img):
        cents = CentroidExtractor(ExtractorParams(threshold=60)).extract(img)
        self.assertIsInstance(cents, list)

    @settings(max_examples=60, deadline=None)
    @given(st.integers(min_value=0, max_value=255))
    def test_threshold_monotonic_uniform_image(self, level):
        # A uniform image yields either one big blob (>= threshold) or none.
        img = np.full((64, 64), level, np.uint8)
        cents = CentroidExtractor(ExtractorParams(threshold=128, min_area=4)).extract(img)
        # cv2 THRESH_BINARY is strict '>': pixel must exceed the threshold.
        self.assertEqual(len(cents), 1 if level > 128 else 0)


@unittest.skipUnless(_HAVE_HYP, "hypothesis not installed")
class TestRingBufferProperties(unittest.TestCase):
    @settings(max_examples=200, deadline=None)
    @given(st.integers(min_value=1, max_value=64),
           st.integers(min_value=0, max_value=300))
    def test_overwrite_keeps_newest_contiguous_tail(self, cap, n):
        rb = CentroidRingBuffer(n_max_samples=cap)
        for i in range(n):
            rb.push(CentroidProfile(seq_id=i, frame_id=i, t_rx_monotonic_us=i))
        out = []
        while True:
            p = rb.pop(timeout=0)
            if p is None:
                break
            out.append(p.seq_id)
        # Invariants: strictly increasing, no dups, and exactly the newest
        # min(n, cap) items survive (drop-oldest-on-lap).
        self.assertEqual(out, sorted(out))
        self.assertEqual(len(out), len(set(out)))
        expected = list(range(max(0, n - cap), n))
        self.assertEqual(out, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
