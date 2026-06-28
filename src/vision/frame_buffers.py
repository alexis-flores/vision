"""
frame_buffers.py
Consumer-side buffers the camera service pushes into.

- CircularFrameBuffer: fixed capacity, overwrites oldest. Vision pipelines
  want the freshest data; dropping stale frames under load is correct.
- FIFOFrameBuffer: bounded queue.Queue with drop-newest-on-full policy.
  GUIs want ordered playback without unbounded memory growth.

Both expose the same push()/pop() surface so the service layer treats
them uniformly (FrameSink protocol).
"""

from __future__ import annotations

import collections
import queue
import threading
from typing import List, Optional, Protocol, runtime_checkable

from .camera_types import CameraFrame


@runtime_checkable
class FrameSink(Protocol):
    """Anything the camera service can push frames into.

    The pushed CameraFrame is shared across all sinks and its pixel buffer is
    read-only (see CameraFrame.__post_init__); a sink that needs to modify
    pixels must copy first. Implementations must not block the producer.
    """
    def push(self, frame: CameraFrame) -> None: ...


class CircularFrameBuffer:
    """
    Thread-safe ring buffer. When full, the oldest frame is overwritten.
    Feeds the vision processor.
    """

    def __init__(self, capacity: int = 64) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._buf: collections.deque[CameraFrame] = \
            collections.deque(maxlen=capacity)
        self._cond = threading.Condition()
        self._dropped = 0

    def push(self, frame: CameraFrame) -> None:
        with self._cond:
            if len(self._buf) == self._buf.maxlen:
                self._dropped += 1
            self._buf.append(frame)
            self._cond.notify_all()

    def pop(self, timeout: Optional[float] = None) -> Optional[CameraFrame]:
        """Blocking pop of the oldest buffered frame. None on timeout."""
        with self._cond:
            if not self._buf and not self._cond.wait(timeout=timeout):
                return None
            return self._buf.popleft() if self._buf else None

    def latest(self) -> Optional[CameraFrame]:
        """Peek the newest frame without consuming (vision often wants this)."""
        with self._cond:
            return self._buf[-1] if self._buf else None

    def drain(self) -> List[CameraFrame]:
        """Take everything currently buffered (batch processing)."""
        with self._cond:
            frames = list(self._buf)
            self._buf.clear()
            return frames

    def __len__(self) -> int:
        with self._cond:
            return len(self._buf)

    @property
    def dropped(self) -> int:
        return self._dropped


class FIFOFrameBuffer:
    """
    Bounded FIFO. When full, the incoming frame is dropped (the GUI is
    behind; preserving order matters more than completeness).
    Feeds the GUI bridge.
    """

    def __init__(self, capacity: int = 8) -> None:
        self._q: "queue.Queue[CameraFrame]" = queue.Queue(maxsize=capacity)
        self._dropped = 0

    def push(self, frame: CameraFrame) -> None:
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            self._dropped += 1

    def pop(self, timeout: Optional[float] = None) -> Optional[CameraFrame]:
        try:
            if timeout == 0:                 # non-blocking drain (GUI poll)
                return self._q.get_nowait()
            if timeout is None:
                return self._q.get()
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def __len__(self) -> int:
        return self._q.qsize()

    @property
    def dropped(self) -> int:
        return self._dropped
