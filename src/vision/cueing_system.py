"""
cueing_system.py
The downstream consumer in the SRS v0.2 dataflow:

    [Spinnaker SDK] -> [Vision System] -> [Cueing System] -> [downstream consumer]

Scope change (SRS v0.2): image processing / centroid extraction / object
labelling / tracking / kinematic state estimation / acquisition moved OUT of the
vision system and INTO the cueing system. The vision system now only configures
the camera, streams, and serves frames; this subsystem ingests those frames.

This module is a thin, dependency-free frame consumer: a worker thread that
drains the shared CircularFrameBuffer concurrently with the camera-service
worker (NFR-007) and hands each CameraFrame to a pluggable `frame_processor`
callback.

What is deliberately NOT implemented here, and why:
    The actual cueing pipeline (centroid extraction, tracking, state estimation,
    acquisition, and the pixel -> angular pointing cue) is undefined in the
    current spec — there is no dedicated cueing SRS yet, the centroid-profile
    type is a §5.2 TODO, and the cue output format / reference frame / units are
    unspecified. Per project guidance these must not be invented. Plug the real
    pipeline in through `frame_processor` once that spec lands.

Reliability mirrors the vision/service side (shared NFRs):
    * processor exceptions are caught, counted, and logged at WARN, then the loop
      continues (skip + continue, NFR-006);
    * the worker never raises out to the caller and exits cleanly on shutdown
      (deadlock-free, NFR-007).
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from .camera_types import CameraFrame
from .frame_buffers import CircularFrameBuffer

log = logging.getLogger(__name__)

# Hook for the (future, out-of-scope) cueing pipeline. Receives each frame on
# the cueing worker thread, so it should return reasonably quickly; anything
# slow should hand off to its own worker/buffer. The frame's pixel buffer is
# read-only and shared with the GUI sink — copy before any in-place pixel op.
FrameProcessor = Callable[[CameraFrame], None]


class CueingSystem:
    """
    Frame consumer that reads from the shared CircularFrameBuffer the camera
    service writes into. See the module docstring for scope and rationale.
    """

    def __init__(self, frame_buffer: CircularFrameBuffer,
                 frame_processor: Optional[FrameProcessor] = None,
                 poll_timeout: float = 0.5) -> None:
        self._frames = frame_buffer
        self._processor = frame_processor
        self._poll_timeout = poll_timeout

        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        # Telemetry: written ONLY by the single cueing worker thread (_run),
        # read by other threads (stats/logging). Single-writer + atomic int
        # reads ⇒ GIL-safe, no lock needed.
        self.frames_consumed = 0
        self.frames_errored = 0
        self.last_frame_id: Optional[int] = None

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Lifecycle
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, name="cueing-system", daemon=True)
        self._thread.start()
        log.info("Cueing system started")

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                log.warning("Cueing worker did not exit within 3s; abandoning")
            self._thread = None
            log.info("Cueing system stopped (consumed=%d errored=%d)",
                     self.frames_consumed, self.frames_errored)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Worker loop
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            frame = self._frames.pop(timeout=self._poll_timeout)
            if frame is None:
                continue
            self.frames_consumed += 1
            self.last_frame_id = frame.frame_id
            if self._processor is not None:
                try:
                    self._processor(frame)
                except Exception:  # NFR-006: skip + continue, never crash
                    self.frames_errored += 1
                    log.warning("Cueing processor raised on frame %s; "
                                "continuing", frame.frame_id, exc_info=True)
