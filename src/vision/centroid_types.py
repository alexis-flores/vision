"""
centroid_types.py
Resolves the SRS section 5.2 "Centroid profile type: TODO: Define".

A CentroidProfile is the per-frame output of the vision system: the list of
detected blob centroids plus the metadata the queuing subsystem needs to
align detections to host time. These records are what the shared ring buffer
transports to the queuing subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Centroid:
    """A single detected blob centroid (sub-pixel)."""
    x: float # sub-pixel column (intensity-weighted)
    y: float # sub-pixel row
    intensity: float # integrated intensity of the blob
    peak: float # peak pixel value
    area: int # pixel count
    bbox: Tuple[int, int, int, int] # (x, y, w, h)


@dataclass
class CentroidProfile:
    """
    Vision-system output for one frame. This is the unit stored in the
    centroid ring buffer shared with the queuing subsystem.
    """
    seq_id: int # monotonic profile sequence id
    frame_id: int # source CameraFrame id
    t_rx_monotonic_us: int # host receive time (monotonic, us)
    centroids: List[Centroid] = field(default_factory=list)
    hw_timestamp_ns: Optional[int] = None # device capture time if available
    proc_latency_us: float = 0.0 # extraction latency (NFR-002 check)
    camera_name: str = ""

    @property
    def n_centroids(self) -> int:
        return len(self.centroids)
