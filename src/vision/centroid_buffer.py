"""
centroid_buffer.py
SRS 5.2 data/message format. The shared ring buffer transporting
CentroidProfile records from the vision worker to the queuing worker, with
independent read/write pointers and a thread-safe lock (NFR-007), plus an
Rx timebase that aligns the global sample index to host time.

This follows the RingBuffer sketch in the SRS but stores CentroidProfile
objects (the resolved "centroid profile type") rather than raw uint16
samples, and keeps the same pointer/timebase bookkeeping fields.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .centroid_types import CentroidProfile


@dataclass
class RxTimebase:
    """
    Maps a global sample/sequence index to host monotonic time.

    Anchored on the first record (seq0, host time us). With a known sample
    rate, host_time(seq) = t0_us + (seq - seq0) / f_sample_hz * 1e6.
    If f_sample_hz is unknown, each record's own t_rx_monotonic_us is used.
    """
    seq0: int = 0
    t0_us: np.int64 = np.int64(0)
    f_sample_hz: float = 0.0
    is_set: bool = False

    def set_anchor(self, seq_id: int, t_rx_us: int,
                   f_sample_hz: float = 0.0) -> None:
        self.seq0 = seq_id
        self.t0_us = np.int64(t_rx_us)
        self.f_sample_hz = f_sample_hz
        self.is_set = True

    def host_time_us(self, seq_id: int) -> np.int64:
        if not self.is_set or self.f_sample_hz <= 0:
            return self.t0_us
        dt = (seq_id - self.seq0) / self.f_sample_hz * 1e6
        return self.t0_us + np.int64(dt)


class CentroidRingBuffer:
    """
    Thread-safe ring buffer of CentroidProfile records with independent
    read and write pointers. Overwrites oldest unread data when the writer
    laps the reader (the queuing system should keep up; drops are counted).
    """

    def __init__(self, n_max_samples: int = 256,
                 rx_timebase: Optional[RxTimebase] = None) -> None:
        if n_max_samples < 1:
            raise ValueError("n_max_samples must be >= 1")
        self.n_max = n_max_samples
        self.buffer: List[Optional[CentroidProfile]] = \
            [None] * n_max_samples
        self.lock = threading.Lock()
        self._not_empty = threading.Condition(self.lock)

        # Read/write counters (independent pointers, per SRS)
        self.pos_write: int = 0
        self.n_written: int = 0
        self.pos_read: int = 0
        self.n_read: int = 0
        self.n_overwritten: int = 0

        # Timebase state
        self.rx_timebase = rx_timebase or RxTimebase()
        self.timebase_set: bool = False

        # Last reported record metadata (mirrors SRS fields)
        self.f_sample_hz: float = 0.0
        self.t_rx_monotonic_us: np.int64 = np.int64(0)
        self.seq_id: np.int32 = np.int32(0)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Producer side (vision worker)
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def push(self, profile: CentroidProfile) -> None:
        with self._not_empty:
            if not self.timebase_set:
                self.rx_timebase.set_anchor(
                    profile.seq_id, profile.t_rx_monotonic_us,
                    self.f_sample_hz)
                self.timebase_set = True

            unread = self.n_written - self.n_read
            if unread >= self.n_max: # writer laps reader
                self.n_overwritten += 1
                self.pos_read = (self.pos_read + 1) % self.n_max
                self.n_read += 1

            self.buffer[self.pos_write] = profile
            self.pos_write = (self.pos_write + 1) % self.n_max
            self.n_written += 1

            self.t_rx_monotonic_us = np.int64(profile.t_rx_monotonic_us)
            self.seq_id = np.int32(profile.seq_id)
            self._not_empty.notify()

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Consumer side (queuing worker)
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def pop(self, timeout: Optional[float] = None) -> Optional[CentroidProfile]:
        """Blocking read of the next unread profile. None on timeout."""
        with self._not_empty:
            if self.n_read >= self.n_written:
                if not self._not_empty.wait(timeout=timeout):
                    return None
                if self.n_read >= self.n_written:
                    return None
            profile = self.buffer[self.pos_read]
            self.pos_read = (self.pos_read + 1) % self.n_max
            self.n_read += 1
            return profile

    def available(self) -> int:
        with self.lock:
            return self.n_written - self.n_read

    def wake_all(self) -> None:
        """Unblock any waiting consumers (used on shutdown)."""
        with self._not_empty:
            self._not_empty.notify_all()
