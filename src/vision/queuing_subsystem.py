"""
queuing_subsystem.py
The downstream consumer in the SRS dataflow:
    [Spinnaker SDK] -> [Vision System] -> [Queuing System]

It is a minimal, dependency-free stand-in for the real queuing subsystem:
a worker thread that drains the shared centroid ring buffer concurrently
with the vision worker (NFR-007) and hands each CentroidProfile to a sink
callback. Object labelling, tracking, and state estimation are explicitly
out of scope (SRS 3.3) and left to whatever consumes from here.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from .centroid_buffer import CentroidRingBuffer
from .centroid_types import CentroidProfile

log = logging.getLogger(__name__)

ProfileSink = Callable[[CentroidProfile], None]


class QueuingSubsystem:
    def __init__(self, centroid_ring: CentroidRingBuffer,
                 sink: Optional[ProfileSink] = None,
                 poll_timeout: float = 0.5) -> None:
        self._ring = centroid_ring
        self._sink = sink
        self._poll_timeout = poll_timeout
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self.profiles_consumed = 0
        self.centroids_consumed = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, name="queuing-subsystem", daemon=True)
        self._thread.start()
        log.info("Queuing subsystem started")

    def stop(self) -> None:
        self._stop_evt.set()
        self._ring.wake_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        log.info("Queuing subsystem stopped (consumed=%d profiles, "
                 "%d centroids)", self.profiles_consumed,
                 self.centroids_consumed)

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            profile = self._ring.pop(timeout=self._poll_timeout)
            if profile is None:
                continue
            self.profiles_consumed += 1
            self.centroids_consumed += profile.n_centroids
            if self._sink is not None:
                try:
                    self._sink(profile)
                except Exception:
                    log.warning("Queuing sink raised; continuing",
                                exc_info=True)
