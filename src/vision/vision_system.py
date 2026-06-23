"""
vision_system.py
The SRS "Vision System" (Figure 2 "Vision Processor"). Runs in its own
worker thread, concurrent with the queuing worker (NFR-007).

Per frame it:
  1. pulls a CameraFrame from the circular frame buffer,
  2. extracts centroids (FR-003, optional GPU via FR-005),
  3. wraps them in a CentroidProfile and pushes to the centroid ring
     buffer shared with the queuing subsystem,
  4. optionally emits an annotated frame to a GUI callback (FR-004),
     never blocking on it (NFR-008).

Robustness (state machine 5.3 / NFR-006): malformed frames are skipped and
logged at WARN; the loop never raises out to the caller.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

from .camera_types import CameraFrame
from .centroid_buffer import CentroidRingBuffer
from .centroid_extraction import CentroidExtractor
from .centroid_types import CentroidProfile
from .frame_buffers import CircularFrameBuffer

log = logging.getLogger(__name__)

# Called with an annotated BGR frame + its profile; must be non-blocking.
GuiCallback = Callable[[CameraFrame, CentroidProfile], None]


class VisionSystem:
    def __init__(self,
                 frame_buffer: CircularFrameBuffer,
                 centroid_ring: CentroidRingBuffer,
                 extractor: Optional[CentroidExtractor] = None,
                 gui_callback: Optional[GuiCallback] = None,
                 latency_budget_us: float = 5000.0, # NFR-002
                 poll_timeout: float = 0.5,
                 annotate: bool = True) -> None:
        self._frames = frame_buffer
        self._ring = centroid_ring
        self._extractor = extractor or CentroidExtractor()
        self._gui_callback = gui_callback
        self._latency_budget_us = latency_budget_us
        self._poll_timeout = poll_timeout
        self._annotate = annotate

        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._seq = 0

        # Telemetry
        self.frames_processed = 0
        self.frames_skipped = 0
        self.latency_budget_exceeded = 0
        self.last_latency_us = 0.0

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Lifecycle
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, name="vision-system", daemon=True)
        self._thread.start()
        log.info("Vision system started (extractor backend=%s)",
                 getattr(self._extractor, "backend", "custom"))

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._ring.wake_all()
        log.info("Vision system stopped (processed=%d skipped=%d)",
                 self.frames_processed, self.frames_skipped)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Worker loop
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            frame = self._frames.pop(timeout=self._poll_timeout)
            if frame is None:
                continue
            try:
                self._process(frame)
            except Exception: # NFR-006: never crash
                self.frames_skipped += 1
                log.warning("Skipping frame %s due to processing error",
                            getattr(frame, "frame_id", "?"), exc_info=True)

    def _process(self, frame: CameraFrame) -> None:
        if not self._valid(frame):
            self.frames_skipped += 1
            log.warning("Malformed frame %s ignored", frame.frame_id)
            return

        t0 = time.perf_counter()
        centroids = self._extractor.extract(frame.data)
        latency_us = (time.perf_counter() - t0) * 1e6
        self.last_latency_us = latency_us
        if latency_us > self._latency_budget_us: # NFR-002
            self.latency_budget_exceeded += 1
            log.warning("Frame %d extraction %.0f us exceeds %.0f us budget",
                        frame.frame_id, latency_us, self._latency_budget_us)

        self._seq += 1
        profile = CentroidProfile(
            seq_id=self._seq,
            frame_id=frame.frame_id,
            t_rx_monotonic_us=int(frame.timestamp * 1e6),
            centroids=centroids,
            hw_timestamp_ns=frame.hw_timestamp_ns,
            proc_latency_us=latency_us,
            camera_name=frame.camera_name,
        )
        self._ring.push(profile) # -> queuing
        self.frames_processed += 1

        if self._gui_callback is not None: # FR-004 / NFR-008
            out = self._annotated(frame, centroids) if self._annotate else frame
            try:
                self._gui_callback(out, profile)
            except Exception:
                log.warning("GUI callback raised; continuing", exc_info=True)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Helpers
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    @staticmethod
    def _valid(frame: CameraFrame) -> bool:
        return (isinstance(frame.data, np.ndarray)
                and frame.data.size > 0
                and frame.data.ndim in (2, 3))

    def _annotated(self, frame: CameraFrame, centroids) -> CameraFrame:
        """Draw centroid markers for GUI overlay (best-effort)."""
        try:
            import cv2
        except ImportError:
            return frame
        img = frame.data
        canvas = (cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                  if img.ndim == 2 else img.copy())
        for c in centroids:
            cx, cy = int(round(c.x)), int(round(c.y))
            cv2.drawMarker(canvas, (cx, cy), (0, 0, 255),
                           cv2.MARKER_CROSS, 12, 1)
            x, y, w, h = c.bbox
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 1)
        return CameraFrame(
            data=canvas, timestamp=frame.timestamp, frame_id=frame.frame_id,
            camera_name=frame.camera_name, hw_timestamp_ns=frame.hw_timestamp_ns,
            pixel_format=frame.pixel_format,
            metadata={**frame.metadata, "annotated": True,
                      "n_centroids": len(centroids)})
