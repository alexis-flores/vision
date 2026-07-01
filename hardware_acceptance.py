"""
hardware_acceptance.py
Run the automated camera acceptance battery on a real BlackFly S (Spinnaker)
and exit non-zero on any failure — the machine-vision bring-up qualification
counterpart to app.py.

It streams for a fixed window and evaluates objective PASS/FAIL checks
(resolution, throughput, frame integrity, hardware-timestamp monotonicity,
inter-frame jitter, dropped frames, image sanity, stream stability) plus an
optional connect/stream/teardown cycle test. Focus and colour balance are
reported as informational metrics (scene-dependent, not pass/fail).

Usage:
    python hardware_acceptance.py --serial 21512345
    python hardware_acceptance.py --serial 21512345 --seconds 30 --min-fps 60
    python hardware_acceptance.py --mono --no-hw-timestamp        # mono / no device clock
    python hardware_acceptance.py --cycles 10                     # teardown churn test

Exit code: 0 = all checks PASS, 1 = one or more FAIL (or setup error).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from vision.acceptance import (AcceptanceCriteria, CheckResult, run_acceptance,
                               run_bandwidth_stress, run_connect_cycles,
                               run_recovery_cycles)
from vision.camera_service import CameraService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s")
log = logging.getLogger("hardware_acceptance")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "config", "bfs_u3_16s2c.json")


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Camera acceptance battery (Spinnaker).")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--serial", default=None,
                    help="bind to this camera serial (recommended)")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--min-fps", type=float, default=60.0)
    ap.add_argument("--min-resolution", type=int, default=512)
    ap.add_argument("--mono", action="store_true",
                    help="expect a single-channel (mono) frame instead of color")
    ap.add_argument("--no-hw-timestamp", action="store_true",
                    help="do not require a device hardware timestamp")
    ap.add_argument("--max-incomplete-rate", type=float, default=0.01)
    ap.add_argument("--max-dropped-rate", type=float, default=0.005)
    ap.add_argument("--max-jitter-ms", type=float, default=2.0)
    ap.add_argument("--min-mean", type=float, default=2.0)
    ap.add_argument("--max-saturated", type=float, default=0.10)
    ap.add_argument("--max-temperature", type=float, default=75.0,
                    help="device temperature ceiling in C (health check)")
    ap.add_argument("--cycles", type=int, default=0,
                    help="also run N connect/stream/teardown cycles (0 = skip)")
    ap.add_argument("--frames-per-cycle", type=int, default=5)
    # --- stress mode (opt-in; verify reliability under adverse conditions) ----
    # Note: --seconds doubles as the soak knob (e.g. --seconds 3600 for a 1h run).
    ap.add_argument("--stress-reconnect", type=int, default=0, metavar="N",
                    help="STRESS: N mid-stream interrupt/recover cycles, "
                         "asserting frames resume each time (automated NFR-005; "
                         "0 = skip)")
    ap.add_argument("--stress-bandwidth", type=int, default=None, metavar="BPS",
                    help="STRESS: re-run the window with DeviceLinkThroughputLimit "
                         "squeezed to BPS Bytes/s to provoke drops, asserting the "
                         "pipeline stays stable and drops are detected & bounded "
                         "(live register, non-persisted; reverts on power-cycle)")
    return ap.parse_args(argv)


def _criteria(args: argparse.Namespace) -> AcceptanceCriteria:
    return AcceptanceCriteria(
        seconds=args.seconds, min_fps=args.min_fps,
        min_resolution=args.min_resolution, require_color=not args.mono,
        max_incomplete_rate=args.max_incomplete_rate,
        require_hw_timestamp=not args.no_hw_timestamp,
        max_dropped_rate=args.max_dropped_rate, max_jitter_ms=args.max_jitter_ms,
        min_mean_level=args.min_mean, max_saturated_frac=args.max_saturated,
        max_temperature_c=args.max_temperature)


def main(argv=None) -> int:
    args = _parse_args(argv)
    from vision.config_loader import load_camera_config
    from vision.spinnaker_driver import SpinnakerCameraDriver

    def make_driver(cfg):
        if args.serial:
            cfg.serial = args.serial
        return SpinnakerCameraDriver(cfg)

    def factory():  # a fresh driver for the cycle / recovery stress tests
        cfg = load_camera_config(args.config)
        if args.serial:
            cfg.serial = args.serial
        return SpinnakerCameraDriver(cfg)

    svc = CameraService()
    try:
        names = svc.add_cameras_from_config(args.config, make_driver)
        cam = names[0]
        svc.connect(cam)
        report = run_acceptance(svc, cam, _criteria(args))
    except Exception as e:  # setup/connect/evaluation failure = failed acceptance
        log.error("Acceptance run failed before evaluation: %s", e)
        svc.shutdown()
        return 1

    # Optional STRESS: bandwidth squeeze. Re-uses the still-connected service
    # (run_bandwidth_stress reconnects as needed). Guarded so a stress-path
    # failure surfaces as a FAILED check rather than discarding the acceptance
    # report we already have.
    try:
        if args.stress_bandwidth is not None:
            report.checks.append(run_bandwidth_stress(
                svc, cam, _criteria(args), args.stress_bandwidth))
    except Exception as e:
        log.error("Bandwidth stress failed: %s", e)
        report.checks.append(
            CheckResult("bandwidth_stress", False, f"stress raised: {e}"))
    finally:
        svc.shutdown()

    # Optional teardown-churn cycle test (fresh driver each cycle).
    if args.cycles > 0:
        report.checks.append(run_connect_cycles(
            factory, cycles=args.cycles,
            frames_per_cycle=args.frames_per_cycle))

    # Optional STRESS: automated mid-stream interrupt/recover (fresh driver).
    if args.stress_reconnect > 0:
        report.checks.append(run_recovery_cycles(
            factory(), cycles=args.stress_reconnect,
            frames_per_cycle=args.frames_per_cycle))

    print(report.format())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
