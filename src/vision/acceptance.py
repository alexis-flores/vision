"""
acceptance.py
Automated camera acceptance battery — machine-vision bring-up qualification.

Streams from a CameraService for a fixed window, collects per-frame metrics, and
evaluates a set of objective PASS/FAIL checks against tunable criteria:

  * streaming reached and frames received,
  * resolution (NFR-003) and pixel format / dtype,
  * sustained throughput / frame rate (NFR-001),
  * frame integrity — incomplete/malformed rate (NFR-006),
  * hardware-timestamp presence + monotonicity,
  * inter-frame jitter and dropped-frame rate (from device timestamps),
  * image sanity — not black / not saturated (exposure plausibility),
  * stream stability — no backend faults / reconnects during the run (NFR-005).

Focus (sharpness) and colour balance are reported as INFORMATIONAL metrics only:
they depend on the scene and cannot be pass/fail without a reference target.

Design: collection (`collect`) and evaluation (`evaluate`) are separate, and
`evaluate` is a pure function of `Metrics`. That lets the whole check matrix be
unit-tested on the simulator and with synthetic metrics — no hardware required.
A separate `run_connect_cycles` exercises repeated connect/stream/teardown to
validate clean release (the gc/atexit teardown path) automatically.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .camera_driver import CameraDriver, CameraError
from .camera_service import CameraService
from .camera_types import CameraFrame, CameraStatus

log = logging.getLogger(__name__)


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Criteria / results
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

@dataclass
class AcceptanceCriteria:
    """Tunable thresholds. Defaults target the SRS NFRs for the BFS-U3-16S2C-CS."""
    seconds: float = 20.0                 # acquisition window
    min_fps: float = 60.0                 # NFR-001
    fps_tolerance: float = 0.02           # allow 2% for finite-window measurement
    min_resolution: int = 512             # NFR-003 (min of width/height)
    require_color: bool = True            # 3-channel BGR expected (color camera)
    require_uint8: bool = True
    max_incomplete_rate: float = 0.01     # <=1% malformed/incomplete frames (NFR-006)
    require_hw_timestamp: bool = True
    max_dropped_rate: float = 0.005       # <=0.5% device-side dropped frames
    max_jitter_ms: float = 2.0            # inter-frame interval stddev (device clock)
    min_mean_level: float = 2.0           # not-black floor (0..255)
    max_saturated_frac: float = 0.10      # <=10% pixels at/near full scale
    expect_resolution: Optional[Tuple[int, int]] = None  # (w, h); default: config
    sample_every: int = 10                # keep every Nth frame for image stats
    max_samples: int = 32


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    skipped: bool = False                 # not applicable (e.g. no hw timestamps)

    @property
    def status(self) -> str:
        return "SKIP" if self.skipped else ("PASS" if self.passed else "FAIL")


@dataclass
class AcceptanceReport:
    checks: List[CheckResult] = field(default_factory=list)
    info: List[Tuple[str, str]] = field(default_factory=list)  # (label, value)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if not c.skipped)

    def format(self) -> str:
        width = max((len(c.name) for c in self.checks), default=0)
        lines = ["", "==================== CAMERA ACCEPTANCE ===================="]
        for c in self.checks:
            lines.append(f"  [{c.status:4}] {c.name:<{width}}  {c.detail}")
        if self.info:
            lines.append("  ---- informational (scene-dependent) ----")
            ilabel = max(len(k) for k, _ in self.info)
            for k, v in self.info:
                lines.append(f"  [INFO] {k:<{ilabel}}  {v}")
        verdict = "PASS" if self.passed else "FAIL"
        lines.append("-----------------------------------------------------------")
        lines.append(f"  OVERALL: {verdict}")
        lines.append("===========================================================")
        return "\n".join(lines)


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Metrics collection
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

@dataclass
class FrameStat:
    host_ts: float
    hw_ts: Optional[int]
    frame_id: int
    ndim: int
    channels: int
    dtype: str


@dataclass
class Metrics:
    """Everything `evaluate` needs — collected from a real run or built by hand."""
    frames: List[FrameStat] = field(default_factory=list)
    samples: List[np.ndarray] = field(default_factory=list)  # sampled full frames
    delivered: int = 0
    malformed: int = 0
    reconnects: int = 0
    streaming_reached: bool = False
    cfg_resolution: Tuple[int, int] = (0, 0)
    cfg_fps: float = 0.0
    duration_s: float = 0.0


class _Collector:
    """FrameSink that records lightweight per-frame stats on the camera worker
    thread, and stashes a capped sample of full frames for image analysis."""

    def __init__(self, sample_every: int, max_samples: int) -> None:
        self._sample_every = max(1, sample_every)
        self._max_samples = max_samples
        self._n = 0
        self.frames: List[FrameStat] = []
        self.samples: List[np.ndarray] = []

    def push(self, frame: CameraFrame) -> None:
        d = frame.data
        ch = d.shape[2] if d.ndim == 3 else 1
        self.frames.append(FrameStat(
            host_ts=frame.timestamp, hw_ts=frame.hw_timestamp_ns,
            frame_id=frame.frame_id, ndim=d.ndim, channels=ch, dtype=str(d.dtype)))
        if (self._n % self._sample_every == 0
                and len(self.samples) < self._max_samples):
            self.samples.append(np.array(d, copy=True))
        self._n += 1


def collect(service: CameraService, name: str,
            criteria: AcceptanceCriteria) -> Metrics:
    """Stream `name` for `criteria.seconds` and gather metrics. Assumes the
    camera is already connected; starts and stops streaming itself."""
    cfg = service._entry(name).driver.config  # snapshot resolution/fps
    collector = _Collector(criteria.sample_every, criteria.max_samples)
    service.attach_sink(name, collector)
    t0 = time.monotonic()
    service.start_streaming(name)
    streaming = service.get_status(name) == CameraStatus.STREAMING
    try:
        time.sleep(criteria.seconds)  # interruptible by Ctrl-C; finally stops cleanly
    finally:
        service.stop_streaming(name)
        service.detach_sink(name, collector)
    duration = time.monotonic() - t0
    stats = service.stats(name)
    res = cfg.resolution or cfg.max_resolution
    return Metrics(
        frames=collector.frames, samples=collector.samples,
        delivered=stats["frames_delivered"], malformed=stats["malformed_frames"],
        reconnects=stats["reconnects"], streaming_reached=streaming,
        cfg_resolution=(res[0], res[1]), cfg_fps=float(cfg.fps or 0.0),
        duration_s=duration)


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Image statistics
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

@dataclass
class ImageStats:
    mean_level: float
    saturated_frac: float
    sharpness: float                       # variance of Laplacian (focus proxy)
    channel_means: Tuple[float, ...]       # per-channel mean (tint proxy)


def _image_stats(samples: List[np.ndarray]) -> Optional[ImageStats]:
    if not samples:
        return None
    means, sats, sharps, ch_means = [], [], [], []
    for img in samples:
        a = img.astype(np.float32)
        means.append(float(a.mean()))
        sats.append(float((img >= 254).mean()))
        gray = a.mean(axis=2) if a.ndim == 3 else a
        sharps.append(_sharpness(gray))
        if img.ndim == 3:
            ch_means.append(tuple(float(img[..., c].mean())
                                  for c in range(img.shape[2])))
    n_ch = len(ch_means[0]) if ch_means else 0
    channel_means = tuple(
        float(np.mean([cm[c] for cm in ch_means])) for c in range(n_ch))
    return ImageStats(
        mean_level=float(np.mean(means)),
        saturated_frac=float(np.mean(sats)),
        sharpness=float(np.mean(sharps)),
        channel_means=channel_means)


def _sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian — higher = sharper/more in focus."""
    try:
        import cv2
        return float(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var())
    except Exception:
        # Pure-numpy fallback: variance of a discrete Laplacian.
        lap = (-4 * gray
               + np.roll(gray, 1, 0) + np.roll(gray, -1, 0)
               + np.roll(gray, 1, 1) + np.roll(gray, -1, 1))
        return float(lap.var())


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Evaluation (pure: Metrics -> report)
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

def evaluate(m: Metrics, criteria: AcceptanceCriteria) -> AcceptanceReport:
    rep = AcceptanceReport()
    add = rep.checks.append
    n = len(m.frames)

    # 1. Streaming reached + frames received
    add(CheckResult("streaming_reached", m.streaming_reached,
                    "stream started" if m.streaming_reached else "never STREAMING"))
    add(CheckResult("frames_received", n > 0, f"{n} frames in {m.duration_s:.1f}s"))
    if n == 0:
        return rep  # nothing else is meaningful

    last = m.frames[-1]

    # 2. Resolution (NFR-003) + matches configured
    want = criteria.expect_resolution or m.cfg_resolution
    # frame is (h, w[, c]); compare (w, h)
    fw, fh = _frame_wh(m.samples, m.frames)
    res_ok = (min(fw, fh) >= criteria.min_resolution
              and (fw, fh) == want)
    add(CheckResult("resolution", res_ok,
                    f"{fw}x{fh} (want {want[0]}x{want[1]}, "
                    f"min {criteria.min_resolution})"))

    # 3. Pixel format / dtype
    fmt_ok = True
    fmt_detail = f"ndim={last.ndim} ch={last.channels} dtype={last.dtype}"
    if criteria.require_color:
        fmt_ok = fmt_ok and last.ndim == 3 and last.channels == 3
    if criteria.require_uint8:
        fmt_ok = fmt_ok and last.dtype == "uint8"
    add(CheckResult("pixel_format", fmt_ok, fmt_detail))

    # 4. Throughput (NFR-001). Allow a small tolerance: a compliant camera
    # measured over a finite window reads slightly under nominal (the last
    # frame lands just before the window closes).
    span = m.frames[-1].host_ts - m.frames[0].host_ts
    fps = (n - 1) / span if span > 0 else 0.0
    floor = criteria.min_fps * (1 - criteria.fps_tolerance)
    add(CheckResult("throughput_fps", fps >= floor,
                    f"{fps:.1f} fps (min {criteria.min_fps}, "
                    f"floor {floor:.1f})"))

    # 5. Frame integrity (NFR-006)
    total = m.delivered + m.malformed
    inc_rate = (m.malformed / total) if total else 0.0
    add(CheckResult("frame_integrity", inc_rate <= criteria.max_incomplete_rate,
                    f"{m.malformed}/{total} malformed ({inc_rate * 100:.2f}%, "
                    f"max {criteria.max_incomplete_rate * 100:.2f}%)"))

    # 6. Hardware timestamp present + monotonic
    hw = [f.hw_ts for f in m.frames if f.hw_ts is not None]
    if not criteria.require_hw_timestamp and not hw:
        add(CheckResult("hw_timestamp", True, "not required / not present",
                        skipped=True))
    else:
        present = len(hw) == n
        monotonic = all(b > a for a, b in zip(hw, hw[1:])) if len(hw) > 1 else present
        add(CheckResult("hw_timestamp", present and monotonic,
                        f"present={len(hw)}/{n} monotonic={monotonic}"))

    # 7. Jitter + dropped frames (device clock if available)
    _eval_timing(rep, m, criteria, hw, fps)

    # 8. Image sanity (not black / not saturated)
    stats = _image_stats(m.samples)
    if stats is None:
        add(CheckResult("image_sanity", True, "no samples", skipped=True))
    else:
        sane = (stats.mean_level >= criteria.min_mean_level
                and stats.saturated_frac <= criteria.max_saturated_frac)
        add(CheckResult("image_sanity", sane,
                        f"mean={stats.mean_level:.1f} "
                        f"(min {criteria.min_mean_level}), "
                        f"saturated={stats.saturated_frac * 100:.2f}% "
                        f"(max {criteria.max_saturated_frac * 100:.1f}%)"))
        rep.info.append(("sharpness (focus proxy)", f"{stats.sharpness:.1f}"))
        if stats.channel_means:
            rep.info.append(("channel means (B,G,R tint)",
                             ", ".join(f"{v:.1f}" for v in stats.channel_means)))

    # 9. Stream stability (no backend faults during the run)
    add(CheckResult("stream_stability", m.reconnects == 0,
                    f"{m.reconnects} reconnect(s) during run"))

    rep.info.append(("effective fps", f"{fps:.2f}"))
    rep.info.append(("frames / duration", f"{n} / {m.duration_s:.1f}s"))
    return rep


def _eval_timing(rep: AcceptanceReport, m: Metrics, criteria: AcceptanceCriteria,
                 hw: List[int], fps: float) -> None:
    if len(hw) >= 2:
        deltas = np.diff(np.asarray(hw, dtype=np.float64)) / 1e6  # ns -> ms
        period_ms = 1000.0 / m.cfg_fps if m.cfg_fps > 0 else float(np.median(deltas))
        jitter_ms = float(deltas.std())
        dropped = int(sum(max(0, round(d / period_ms) - 1) for d in deltas))
        drop_rate = dropped / len(deltas)
        rep.checks.append(CheckResult(
            "interframe_jitter", jitter_ms <= criteria.max_jitter_ms,
            f"{jitter_ms:.3f} ms stddev (max {criteria.max_jitter_ms})"))
        rep.checks.append(CheckResult(
            "dropped_frames", drop_rate <= criteria.max_dropped_rate,
            f"{dropped} dropped ({drop_rate * 100:.2f}%, "
            f"max {criteria.max_dropped_rate * 100:.2f}%)"))
        rep.info.append(("max interframe gap", f"{float(deltas.max()):.3f} ms"))
    else:
        rep.checks.append(CheckResult(
            "interframe_jitter", True,
            "no hardware timestamps — host-clock jitter is informational only",
            skipped=True))
        rep.checks.append(CheckResult(
            "dropped_frames", True, "n/a without device timestamps", skipped=True))


def _frame_wh(samples: List[np.ndarray],
              frames: List[FrameStat]) -> Tuple[int, int]:
    if samples:
        s = samples[-1]
        return int(s.shape[1]), int(s.shape[0])  # (w, h)
    return (0, 0)


# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
#   Top-level helpers
# ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

def run_acceptance(service: CameraService, name: str,
                   criteria: AcceptanceCriteria) -> AcceptanceReport:
    """Connect (if needed), stream, and evaluate the full acceptance battery."""
    if service.get_status(name) == CameraStatus.DISCONNECTED:
        service.connect(name)
    return evaluate(collect(service, name, criteria), criteria)


def run_connect_cycles(driver_factory: Callable[[], CameraDriver],
                       cycles: int = 5,
                       frames_per_cycle: int = 5,
                       read_timeout: float = 2.0) -> CheckResult:
    """Repeatedly connect -> stream -> read -> stop -> disconnect to validate
    clean teardown (the gc.collect/atexit release path) under churn — the
    automated counterpart to a manual unplug/replug. Returns a single check."""
    for i in range(cycles):
        drv = driver_factory()
        try:
            drv.connect()
            drv.start_stream()
            for _ in range(frames_per_cycle):
                drv.read_frame(timeout=read_timeout)
            drv.stop_stream()
        except CameraError as e:
            return CheckResult("connect_cycles", False,
                               f"cycle {i + 1}/{cycles} failed: {e}")
        finally:
            drv.disconnect()
    return CheckResult("connect_cycles", True,
                       f"{cycles} clean connect/stream/teardown cycles")
