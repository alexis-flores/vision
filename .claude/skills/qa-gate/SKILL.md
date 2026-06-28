---
name: qa-gate
description: Run the full software quality gate for the vision repo (unit tests, ruff, mypy, optionally bandit/vulture/build). Use after any code change, before committing, or to confirm the baseline is green. Runs anywhere — no camera or PySpin required.
---

# qa-gate

The software gate. Runs on **any** machine (no camera, no PySpin — the hardware test
skips cleanly). Use Python 3.10 (see CLAUDE.md). Establish/refresh the venv if needed:
```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"      # add ",gui,yaml" extras if testing those paths
```

## Core gate (run all; report each)
```bash
python -m unittest discover -s tests        # expect "OK" (a few skips are normal:
                                            #   HW-001 + Hypothesis if not installed)
ruff check .                                 # lint — expect "All checks passed!"
mypy                                         # types — expect "Success: no issues"
```
The baseline is **clean**: all tests pass, ruff/mypy report nothing. Any new failure
is a regression you introduced — fix it before moving on, don't accept a red gate.

## Extended (optional, when hardening or before a release)
```bash
pip install bandit vulture build twine
bandit -r src app.py gui_bridge.py hardware_acceptance.py   # security; expect 0 issues
vulture src app.py gui_bridge.py hardware_acceptance.py --min-confidence 80  # dead code
python -m build && twine check dist/*        # packaging sanity (ships py.typed)
```

## Conventions when changing code
- Match surrounding style (the codebase uses lazy `%`-logging, guarded cleanup that
  never raises, single-writer GIL-safe counters, daemon threads with bounded joins).
- Keep changes **opt-in / default-preserving** — never regress the hardware-validated
  single-camera path (CLAUDE.md).
- Add/extend tests for any behavior change: pure-logic in `tests/test_suite.py` or
  `tests/test_acceptance.py`; driver logic via the fake SDKs in
  `tests/test_drivers_mocked.py` (extend the fake `Camera`/`cv2` rather than mocking ad hoc).
- mypy is configured over `src/vision` + the three root scripts (`pyproject.toml`).
- Re-run this gate after every change. Commit/push only when asked; sole author
  Alexis, no Claude/Co-Authored-By/Claude-Session trailers.
