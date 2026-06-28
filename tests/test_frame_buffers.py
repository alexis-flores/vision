"""
tests/test_frame_buffers.py
The two consumer buffers and their opposite drop policies:
CircularFrameBuffer (drop-oldest, freshest wins -> cueing) and FIFOFrameBuffer
(drop-newest, ordered playback -> GUI). Example-based + concurrency (NFR-007) +
property-based (Hypothesis, skipped cleanly if absent) + regressions.
"""

from __future__ import annotations

import threading
import time
import unittest

import numpy as np

import _helpers  # noqa: F401  (path bootstrap)

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
    _HAVE_HYP = True
except ImportError:                                   # pragma: no cover
    _HAVE_HYP = False

from vision.camera_types import CameraFrame
from vision.frame_buffers import CircularFrameBuffer, FIFOFrameBuffer


def _frame(fid: int) -> CameraFrame:
    return CameraFrame(data=np.zeros((4, 4), np.uint8),
                       timestamp=time.monotonic(), frame_id=fid)


class TestFrameBuffers(unittest.TestCase):
    def test_circular_overwrite(self):  # boundary
        rb = CircularFrameBuffer(capacity=3)
        for i in range(5):
            rb.push(_frame(i))
        self.assertEqual(rb.dropped, 2)
        self.assertEqual(rb.pop(timeout=0.1).frame_id, 2)

    def test_circular_latest(self):
        rb = CircularFrameBuffer(capacity=4)
        for i in range(3):
            rb.push(_frame(i))
        self.assertEqual(rb.latest().frame_id, 2)

    def test_circular_pop_empty_timeout(self):  # error/empty
        rb = CircularFrameBuffer(capacity=2)
        self.assertIsNone(rb.pop(timeout=0.05))

    def test_circular_concurrent_no_deadlock(self):  # NFR-007
        rb = CircularFrameBuffer(capacity=16)
        N = 2000
        consumed = []
        stop = threading.Event()

        def producer():
            for i in range(N):
                rb.push(_frame(i))

        def consumer():
            while not stop.is_set() or len(rb) > 0:
                f = rb.pop(timeout=0.05)
                if f is not None:
                    consumed.append(f.frame_id)

        c = threading.Thread(target=consumer)
        c.start()
        producer()
        time.sleep(0.2)
        stop.set()
        c.join(timeout=3.0)
        self.assertFalse(c.is_alive(), "consumer deadlocked")
        # Monotonic, no duplicates among consumed (drop-oldest preserves order).
        self.assertEqual(consumed, sorted(consumed))
        self.assertEqual(len(consumed), len(set(consumed)))

    def test_fifo_drop_newest(self):  # boundary
        q = FIFOFrameBuffer(capacity=2)
        for i in range(5):
            q.push(_frame(i))
        self.assertEqual(q.dropped, 3)
        self.assertEqual(q.pop(timeout=0.1).frame_id, 0)  # order preserved

    def test_fifo_pop_zero_timeout_is_nonblocking(self):  # regression
        # gui_bridge._poll drains with pop(timeout=0); it must return None on an
        # empty queue instead of blocking forever (which froze the GUI).
        q = FIFOFrameBuffer(capacity=2)
        done = threading.Event()
        result = {}

        def drain():
            result["v"] = q.pop(timeout=0)
            done.set()

        threading.Thread(target=drain, daemon=True).start()
        self.assertTrue(done.wait(timeout=1.0), "pop(timeout=0) blocked on empty")
        self.assertIsNone(result["v"])


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
