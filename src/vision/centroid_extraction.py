"""
centroid_extraction.py
FR-003: image processing and centroid extraction.
FR-005 (P2): optional GPU acceleration of the pipeline, with automatic CPU
fallback so the same interface runs on any machine.

Pipeline (bright-blob model, suited to optical target queuing):
    grayscale -> optional blur -> threshold -> connected components
    -> per-blob intensity-weighted sub-pixel centroid + area/peak filter

The extractor is designed to stay within the NFR-002 <=5ms budget on a
512x512 frame; the vision system measures and logs actual latency.

==============================================================================
HANDOFF CONTRACT  (read this if you are taking ownership of this module)
==============================================================================
This module is self-contained: you may rewrite everything below however you
like -- different algorithm, libraries, tuning -- as long as you preserve the
small boundary the rest of the system depends on. There is intentionally NO
abstract base class; it is just one concrete extractor honoring this contract.

Provide a class with:

  extract(image: np.ndarray) -> list[Centroid]
      * `image` is the raw CameraFrame.data: 2-D (H, W) mono OR 3-D (H, W, 3)
        BGR, dtype uint8 or uint16. Treat it as read-only.
      * Return a (possibly empty) list of centroid_types.Centroid records.
        Downstream only consumes x, y and bbox (GUI overlay); still fill in
        intensity / peak / area for whoever reads the centroid profile.
      * Called once per frame from the single vision worker thread, so it need
        not be thread-safe. Target <=5 ms @ 512x512 (NFR-002); VisionSystem
        logs -- but tolerates -- overruns.
      * Raising is safe: VisionSystem catches it, skips that frame, and keeps
        going (NFR-006). It simply counts as a dropped frame.

  backend: str   (OPTIONAL)
      Short label shown in logs / run summaries (e.g. "cpu" / "gpu"). If you
      omit it, callers fall back to "custom".

Wiring: VisionSystem(frame_buffer, centroid_ring, extractor=YourExtractor()).
If no extractor is passed, VisionSystem defaults to CentroidExtractor().
Construction sites to update if you swap the class:
    main.py, run_hardware.py, gui_bridge.py   (one line each)

Behavioural tests that pin the current contract live in
tests/test_suite.py::TestCentroidExtraction (accuracy, blank/min-area
boundaries, color input, NFR-002 latency). Keep them green, or change them
deliberately as part of the handoff.
==============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .centroid_types import Centroid

log = logging.getLogger(__name__)

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:                 # pragma: no cover
    _HAVE_CV2 = False

# cupy is only used when use_gpu=True; absence triggers CPU fallback.
try:
    import cupy as _cp             # type: ignore
    _HAVE_CUPY = True
except Exception:                  # pragma: no cover - optional dep
    _cp = None
    _HAVE_CUPY = False


@dataclass
class ExtractorParams:
    threshold: int = 128 # 0..255; <0 selects Otsu
    min_area: int = 4 # reject specks
    max_centroids: int = 256 # cap profile size for bounded latency
    blur_ksize: int = 0 # 0 disables; odd >0 enables Gaussian blur
    use_gpu: bool = False # FR-005; falls back to CPU if unavailable
    invert: bool = False # set True for dark targets on bright bg
    method: str = "contour" # "contour" (fast, sparse spots) | "cc"


class CentroidExtractor:
    """Stateless, thread-safe centroid extractor (FR-003 / FR-005)."""

    def __init__(self, params: Optional[ExtractorParams] = None) -> None:
        self.params = params or ExtractorParams()
        self._gpu_active = False
        if self.params.use_gpu:
            self._gpu_active = self._init_gpu()
            if not self._gpu_active:
                log.warning("GPU requested but unavailable; using CPU path")

    @property
    def backend(self) -> str:
        return "gpu" if self._gpu_active else "cpu"

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   Public API
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def extract(self, image: np.ndarray) -> List[Centroid]:
        """Return detected centroids for one frame."""
        if self._gpu_active:
            try:
                return self._extract_gpu(image)
            except Exception:                      # pragma: no cover
                log.exception("GPU path failed; falling back to CPU")
                self._gpu_active = False
        return self._extract_cpu(image)

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   CPU path
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def _to_gray_u8(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            if _HAVE_CV2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image[..., :3].mean(axis=2)
        else:
            gray = image
        if gray.dtype != np.uint8:
            if np.issubdtype(gray.dtype, np.integer): # scale Mono16 etc. to 8-bit
                shift = max(0, gray.itemsize * 8 - 8)
                gray = (gray.astype(np.uint32) >> shift).astype(np.uint8)
            else: # float (e.g. no-cv2 channel mean): clip + cast, don't bit-shift
                gray = np.clip(gray, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(gray)

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        p = self.params
        if p.blur_ksize and p.blur_ksize >= 3 and _HAVE_CV2:
            k = p.blur_ksize | 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        thr_type = cv2.THRESH_BINARY_INV if p.invert else cv2.THRESH_BINARY
        if _HAVE_CV2:
            if p.threshold < 0: # Otsu
                _, mask = cv2.threshold(gray, 0, 255,
                                        thr_type | cv2.THRESH_OTSU)
            else:
                _, mask = cv2.threshold(gray, p.threshold, 255, thr_type)
            return mask
        thr = p.threshold if p.threshold >= 0 else int(gray.mean() + gray.std())
        # Match cv2 THRESH_BINARY semantics: strict '>' (and inclusive '<=' for
        # the inverted case), so the no-OpenCV fallback agrees at the boundary.
        binmask = (gray <= thr) if p.invert else (gray > thr)
        return binmask.astype(np.uint8) * 255

    def _extract_cpu(self, image: np.ndarray) -> List[Centroid]:
        gray = self._to_gray_u8(image)
        mask = self._binarize(gray)
        if _HAVE_CV2 and self.params.method == "contour":
            return self._components_contour(gray, mask)
        if _HAVE_CV2:
            return self._components_cv2(gray, mask)
        return self._components_scipy(gray, mask)

    def _components_contour(self, gray, mask) -> List[Centroid]:
        """
        Contour-traced detection for sparse bright targets. ~15x faster than
        full-frame connected components at 1024x1024 because cost scales with
        boundary length, not pixel count. Centroids are intensity-weighted
        over each contour's filled bounding box.
        """
        p = self.params
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        out: List[Centroid] = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            blob = np.zeros((h, w), np.uint8)
            cv2.drawContours(blob, [c], -1, 1, thickness=cv2.FILLED,
                             offset=(-x, -y))
            bm = blob.astype(bool)
            npix = int(bm.sum())
            if npix < p.min_area:
                continue
            sub = gray[y:y + h, x:x + w].astype(np.float32)
            weights = sub * bm
            total = float(weights.sum())
            if total <= 0:
                m = cv2.moments(c)
                if m["m00"] <= 0:
                    continue
                cx, cy, peak = m["m10"] / m["m00"], m["m01"] / m["m00"], 0.0
            else:
                ys, xs = np.nonzero(bm)
                wv = weights[ys, xs]
                cx = x + float((wv * xs).sum() / total)
                cy = y + float((wv * ys).sum() / total)
                peak = float(sub[bm].max())
            out.append(Centroid(x=cx, y=cy, intensity=total, peak=peak,
                                area=npix, bbox=(x, y, w, h)))
            if len(out) >= p.max_centroids:
                break
        return out

    def _components_cv2(self, gray, mask) -> List[Centroid]:
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        return self._refine(gray, labels, stats, n, centroids)

    def _components_scipy(self, gray, mask) -> List[Centroid]: # fallback
        from scipy import ndimage
        labels, n = ndimage.label(mask > 0)
        stats = np.zeros((n + 1, 5), dtype=np.int64)
        for lbl in range(1, n + 1):
            ys, xs = np.where(labels == lbl)
            stats[lbl] = [xs.min(), ys.min(),
                          xs.max() - xs.min() + 1,
                          ys.max() - ys.min() + 1, len(xs)]
        return self._refine(gray, labels, stats, n + 1, None)

    def _refine(self, gray, labels, stats, n_labels, cc_centroids) -> List[Centroid]:
        """
        Intensity-weighted sub-pixel centroid per component (skip bg=0).

        Only each blob's bounding box is touched (cast to float32 locally),
        so cost scales with detected blob area, not full-frame area — this is
        what keeps extraction inside the NFR-002 budget.
        """
        p = self.params
        out: List[Centroid] = []
        # Order components by area desc so max_centroids keeps the biggest.
        order = sorted(range(1, n_labels),
                       key=lambda i: stats[i, 4], reverse=True)
        for lbl in order:
            area = int(stats[lbl, 4])
            if area < p.min_area:
                continue
            x0, y0, w, h = (int(stats[lbl, 0]), int(stats[lbl, 1]),
                            int(stats[lbl, 2]), int(stats[lbl, 3]))
            sub_lbl = labels[y0:y0 + h, x0:x0 + w]
            sub_int = gray[y0:y0 + h, x0:x0 + w].astype(np.float32)
            blob = (sub_lbl == lbl)
            weights = sub_int * blob
            total = float(weights.sum())
            if total <= 0:
                # Degenerate (all-zero intensity): fall back to geometric.
                if cc_centroids is not None:
                    cx, cy = float(cc_centroids[lbl, 0]), float(cc_centroids[lbl, 1])
                else:
                    continue
                peak = 0.0
            else:
                ys, xs = np.nonzero(blob)
                wv = weights[ys, xs]
                cx = x0 + float((wv * xs).sum() / total)
                cy = y0 + float((wv * ys).sum() / total)
                peak = float(sub_int[blob].max())
            out.append(Centroid(
                x=cx, y=cy, intensity=total, peak=peak,
                area=area, bbox=(x0, y0, w, h)))
            if len(out) >= p.max_centroids:
                break
        return out

    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵
    #   GPU path (FR-005)
    # ✵✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✧✵

    def _init_gpu(self) -> bool:
        if _HAVE_CUPY:
            try:
                _ = _cp.zeros((1,)) # probe device
                return True
            except Exception:            # pragma: no cover
                return False
        # cv2.cuda is the secondary GPU option.
        if _HAVE_CV2 and hasattr(cv2, "cuda") and \
                cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return True
        return False

    def _extract_gpu(self, image: np.ndarray) -> List[Centroid]:
        """
        GPU-accelerated preprocessing (grayscale/blur/threshold on device),
        then CPU connected-components on the reduced mask. Connected-component
        labeling lacks a portable GPU primitive, so we keep that on CPU; the
        pixel-wise stages are the heavy, parallelizable part.
        """
        if _HAVE_CUPY:
            g = _cp.asarray(image)
            if g.ndim == 3:
                g = (0.114 * g[..., 0] + 0.587 * g[..., 1]
                     + 0.299 * g[..., 2])
            g = g.astype(_cp.float32)
            thr = (self.params.threshold if self.params.threshold >= 0
                   else float(g.mean() + g.std()))
            mask = (g <= thr) if self.params.invert else (g >= thr)
            gray = _cp.asnumpy(g).astype(np.uint8)
            mask = _cp.asnumpy(mask).astype(np.uint8) * 255
            if _HAVE_CV2:
                return self._components_cv2(gray, mask)
            return self._components_scipy(gray, mask)
        # cv2.cuda preprocessing
        gpu = cv2.cuda_GpuMat()  # type: ignore[attr-defined]
        gpu.upload(image)
        if image.ndim == 3:
            gpu = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2GRAY)  # type: ignore[attr-defined]
        gray = gpu.download()
        mask = self._binarize(gray)
        return self._components_cv2(gray, mask)
