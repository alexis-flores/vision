---
name: hardware-acceptance
description: Run and interpret the objective camera acceptance battery (hardware_acceptance.py) for the BFS-U3-16S2C-CS. Use to qualify a real camera against PASS/FAIL thresholds, debug a failing check, or tune exposure/thresholds for the rig.
---

# hardware-acceptance

`hardware_acceptance.py` streams the real camera for a fixed window, evaluates a matrix
of objective checks, prints a report, and **exits 0 (all pass) / 1 (any fail)** — a
go/no-go gate. Needs PySpin + a camera. The check logic is pure (`acceptance.evaluate`)
and unit-tested without hardware.

## Run
```bash
python hardware_acceptance.py --serial <SERIAL>                       # 20s, NFR defaults
python hardware_acceptance.py --serial <SERIAL> --seconds 30 --cycles 10
python hardware_acceptance.py --serial <SERIAL> --mono --no-hw-timestamp
```
Key flags (all thresholds are tunable so you can match the rig):
`--seconds --min-fps --min-resolution --max-incomplete-rate --max-dropped-rate
--max-jitter-ms --min-mean --max-saturated --max-temperature --cycles
--frames-per-cycle --mono --no-hw-timestamp`.

## What each check means (and what a FAIL implies)
| Check | Pass criterion | If it FAILS |
|---|---|---|
| `streaming_reached` | BeginAcquisition succeeded | SDK/config/connect issue — check logs above the report |
| `frames_received` | >0 frames | not streaming; usbfs buffer; bandwidth |
| `resolution` | matches config & ≥ min (NFR-003) | Width/Height not applied; ROI offset; wrong config |
| `output_pixel_format` | 3-ch BGR8 uint8 (color) | debayer/convert path; `--mono` if mono camera |
| `throughput_fps` | ≥ min_fps (×(1-tol)) (NFR-001) | exposure too long (>1/fps), bandwidth limit, low fps config |
| `frame_integrity` | malformed ≤ max_incomplete_rate (NFR-006) | USB errors; usbfs buffer; cabling |
| `hw_timestamp` | present on all + monotonic | chunk/timestamp off; pass `--no-hw-timestamp` if N/A |
| `interframe_jitter` | stddev ≤ max_jitter_ms | system load; enable `--rt`; OS tuning (isolcpus, governor) |
| `dropped_frames` | ≤ max_dropped_rate (device frame counter) | bandwidth contention; raise `stream_buffer_count` / set `link_throughput_limit_bps` |
| `image_sanity` | mean ≥ min, saturated ≤ max | exposure: too dark → raise `exposure_us`; blown → lower it |
| `stream_stability` | 0 reconnects during run | flaky cable/port; device faulting mid-run |
| `temperature` | max ≤ max_temperature | airflow; ambient; long run |
| `connect_cycles` (`--cycles`) | N clean teardowns | release/atexit path; resource leak |

INFO lines (not pass/fail, scene-dependent): `sharpness` (focus proxy — variance of
Laplacian), `channel means` (tint proxy), effective fps, and any device transport
counters (`StreamLostFrameCount`, etc.) + `resulting_fps` / `link_throughput_bps`.

## Tuning workflow
1. Run with defaults. Summarize PASS/FAIL for the user.
2. **Exposure** is the most common knob: `image_sanity` dark → raise `exposure_us` in
   `config/bfs_u3_16s2c.json`; saturated → lower it. Keep `exposure_us ≤ 1/fps`
   (≤ 16600 µs at 60 fps) or `throughput_fps` drops.
3. **Drops/bandwidth** (multi-cam or high fps): set `stream_buffer_count` (absorbs
   bursts) and/or `link_throughput_limit_bps` (partitions sustained USB3 bandwidth).
   Watch `resulting_fps` in the INFO lines — below requested fps = exposure- or
   bandwidth-bound.
4. **Jitter**: try `python app.py ... --rt` and the OS tuning in README; validate with
   the `interframe_jitter` check.
5. Re-run until exit 0. **Do not weaken a threshold just to pass** — if a target is
   genuinely wrong for the rig, change it explicitly and tell the user why.

## Notes
- Bind by serial; reconnect needs it.
- Any code fix stays opt-in / default-preserving (CLAUDE.md). Re-run qa-gate after.
