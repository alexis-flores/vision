#!/usr/bin/env python3
"""
check_pyspin_symbols.py
Audit the INSTALLED PySpin/Spinnaker SDK for every module-level symbol that
spinnaker_driver.py depends on — WITHOUT a camera.

Why this exists: the mocked tests (tests/test_drivers_mocked.py) fake every
PySpin.* constant, so they can NOT catch an SDK that renamed/removed a symbol
across versions. This imports the real SDK and checks the actual API surface.

Run on the rig (or any machine with PySpin installed):

    python scripts/check_pyspin_symbols.py

Exit code: 0 = all REQUIRED + COLOR symbols present; 1 = a needed symbol is
missing (the installed SDK is incompatible with the driver as written) or PySpin
can't be imported. OPTIONAL symbols only warn — the driver has runtime fallbacks
for them.

NOTE: this checks module-level PySpin symbols only. GenICam *node* names (Gain,
ExposureTime, AcquisitionFrameRate, BlackLevel, Gamma, DeviceTemperature,
DeviceLinkThroughputLimit, ...) live on the camera nodemap and can only be
verified with a camera attached — that's what HW-001 + hardware_acceptance.py do.
"""

from __future__ import annotations

import sys

# Symbols spinnaker_driver.py uses with NO graceful fallback: absent => the
# driver raises (AttributeError escapes the SpinnakerException handlers, or a
# core call can't run).
REQUIRED = [
    "System",                    # connect(): System.GetInstance()
    "SpinnakerException",        # caught throughout; missing => except clauses break
    "IsAvailable", "IsReadable", "IsWritable",  # guarded node access (in try/except SpinnakerException)
    "CEnumerationPtr", "CIntegerPtr",           # TL stream-node typing
    "AcquisitionMode_Continuous",               # _configure_acquisition / start_stream
    "ExposureAuto_Off",          # set_config("exposure_us")
    "GainAuto_Off",              # set_config("gain_db")
    "UserSetSelector_Default",   # reset_to_defaults()
    "EVENT_TIMEOUT_INFINITE",    # read_frame(timeout=None)
]

# Needed for CORRECT COLOR on the BFS-U3-16S2C-CS specifically. Accessed via
# getattr-with-fallback, but if absent the device format / host conversion is
# wrong, so for this camera treat them as must-have.
COLOR = [
    "PixelFormat_BayerRG8",      # device on-wire format (native Bayer tile)
    "PixelFormat_BGR8",          # host conversion target (output_pixel_format)
]

# Symbols the driver resolves with getattr(..., default)/hasattr — absent is fine,
# the driver degrades gracefully (legacy path, skipped feature, etc.).
OPTIONAL = [
    "ImageProcessor",                                    # else legacy img.Convert()
    "SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR",   # debayer quality hint
    "HQ_LINEAR",                                         # legacy Convert() quality arg
    "SPINNAKER_ERR_TIMEOUT",                            # timeout classification (default -1011)
    "UserSetDefault_Default",                          # power-on default user set
    "ChunkSelector_FrameID", "ChunkSelector_Timestamp", # chunk drop detection
    "PixelFormat_RGB8", "PixelFormat_Mono8", "PixelFormat_Mono16",  # other output formats
]


def _check(mod, names):
    return [(n, hasattr(mod, n)) for n in names]


def _print_group(title, results):
    print(f"\n{title}")
    for name, ok in results:
        print(f"  [{'OK ' if ok else 'MISS'}] {name}")
    return [n for n, ok in results if not ok]


def main() -> int:
    try:
        import PySpin
    except Exception as e:  # noqa: BLE001 - report any import failure verbatim
        print(f"FAILED to import PySpin: {e}")
        print("=> Install the SDK + the cp310 wheel; see the pyspin-doctor skill.")
        return 1

    try:
        s = PySpin.System.GetInstance()
        v = s.GetLibraryVersion()
        print(f"Spinnaker library: {v.major}.{v.minor}.{v.type}.{v.build}")
        cams = s.GetCameras()
        print(f"Cameras detected : {cams.GetSize()} (a symbol audit needs none)")
        cams.Clear()
        s.ReleaseInstance()
    except Exception as e:  # noqa: BLE001
        print(f"(could not query library version / enumerate: {e})")

    missing_required = _print_group("REQUIRED (driver breaks without these):",
                                    _check(PySpin, REQUIRED))
    missing_color = _print_group("COLOR (needed for correct BFS-16S2C color):",
                                 _check(PySpin, COLOR))
    missing_optional = _print_group("OPTIONAL (driver has a fallback):",
                                    _check(PySpin, OPTIONAL))

    print("\n" + "=" * 60)
    hard = missing_required + missing_color
    if hard:
        print(f"RESULT: FAIL — {len(hard)} needed symbol(s) missing: "
              f"{', '.join(hard)}")
        print("The installed SDK is incompatible with spinnaker_driver.py as "
              "written. Report which symbols are missing so the driver can be "
              "adapted (most have FLIR rename history across versions).")
        return 1
    if missing_optional:
        print(f"RESULT: PASS (with {len(missing_optional)} optional symbol(s) "
              f"absent: {', '.join(missing_optional)} — fallbacks will be used)")
    else:
        print("RESULT: PASS — every symbol the driver references is present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
