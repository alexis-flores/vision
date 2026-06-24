"""
tests/test_properties.py
Property-based tests (Hypothesis): assert invariants hold over a wide space of
generated inputs, catching edge cases hand-written cases miss. Skipped cleanly
if Hypothesis is not installed.

Under SRS v0.2 the vision system's job is serving frames, so the invariants
worth fuzzing live in the two frame buffers and their opposite drop policies:
CircularFrameBuffer (drop-oldest, freshest wins) and FIFOFrameBuffer
(drop-newest, ordered playback).
"""

from __future__ import annotations

import os
import sys
import time
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
    _HAVE_HYP = True
except ImportError:                                   # pragma: no cover
    _HAVE_HYP = False

from vision.camera_types import CameraFrame
from vision.frame_buffers import CircularFrameBuffer, FIFOFrameBuffer


def _frame(fid: int) -> CameraFrame:
    return CameraFrame(data=np.zeros((2, 2), np.uint8),
                       timestamp=time.monotonic(), frame_id=fid)


@unittest.skipUnless(_HAVE_HYP, "hypothesis not installed")
class TestFrameBufferProperties(unittest.TestCase):
    @settings(max_examples=200, deadline=None)
    @given(st.integers(min_value=1, max_value=64),
           st.integers(min_value=0, max_value=300))
    def test_circular_keeps_newest_contiguous_tail(self, cap, n):
        # CircularFrameBuffer drops the OLDEST on lap: the newest min(n, cap)
        # survive, strictly increasing, no dups.
        rb = CircularFrameBuffer(capacity=cap)
        for i in range(n):
            rb.push(_frame(i))
        out = []
        while True:
            f = rb.pop(timeout=0)
            if f is None:
                break
            out.append(f.frame_id)
        self.assertEqual(out, sorted(out))
        self.assertEqual(len(out), len(set(out)))
        self.assertEqual(out, list(range(max(0, n - cap), n)))
        self.assertEqual(rb.dropped, max(0, n - cap))

    @settings(max_examples=200, deadline=None)
    @given(st.integers(min_value=1, max_value=64),
           st.integers(min_value=0, max_value=300))
    def test_fifo_keeps_oldest_in_order(self, cap, n):
        # FIFOFrameBuffer drops the NEWEST on full: the oldest min(n, cap)
        # survive in arrival order.
        q = FIFOFrameBuffer(capacity=cap)
        for i in range(n):
            q.push(_frame(i))
        out = []
        while True:
            f = q.pop(timeout=0)
            if f is None:
                break
            out.append(f.frame_id)
        kept = min(n, cap)
        self.assertEqual(out, list(range(kept)))
        self.assertEqual(q.dropped, max(0, n - cap))


if __name__ == "__main__":
    unittest.main(verbosity=2)
