"""Torso color descriptor for the within-camera color guard.

A cheap, model-free appearance signal orthogonal to the ReID embedding: an
HS histogram over the torso (shirt) region of a person crop. Used to VETO
attaching/merging two people who clearly differ in clothing color, which the
CLIP-ReID embedding cannot separate on this footage (near-zero margin).

Calibrated on held-out crops (scripts/eval_color_guard.py): at the aggregate
(tracklet-mean) level, same-person within-camera CORREL p05=0.84 vs
different-person within-camera p50=0.56 — clean separation. Veto threshold 0.83
gives ~2% false-veto on same-person while blocking ~92% of wrong-color attaches.
Applied WITHIN-CAMERA ONLY: cross-camera color shifts too much (lighting) to
discriminate, so the guard never vetoes cross-camera links.

Crops are BGR (cv2.imdecode IMREAD_COLOR), matching the calibration.
"""
from __future__ import annotations

import numpy as np
import cv2

# Torso region as ratios of the crop (y0, y1, x0, x1) — shirt area, excluding
# head / legs / background. Must match scripts/eval_color_guard.py.
TORSO = (0.15, 0.55, 0.15, 0.85)
H_BINS, S_BINS = 16, 4
V_MIN = 40            # drop near-black pixels (unreliable hue)
MIN_MASK_PIXELS = 20  # below this the torso ROI is too dark/small to trust


def torso_hist(crop_bgr: np.ndarray | None) -> np.ndarray | None:
    """HS histogram of the torso region of a BGR crop, or None if unusable."""
    if crop_bgr is None or getattr(crop_bgr, "size", 0) <= 0 or crop_bgr.ndim != 3:
        return None
    h, w = crop_bgr.shape[:2]
    if h < 4 or w < 4:
        return None
    y0, y1, x0, x1 = TORSO
    roi = crop_bgr[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 2] > V_MIN).astype(np.uint8)
    if int(mask.sum()) < MIN_MASK_PIXELS:
        return None
    hist = cv2.calcHist([hsv], [0, 1], mask, [H_BINS, S_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.astype(np.float32)


def aggregate(hists: list[np.ndarray]) -> np.ndarray | None:
    """Mean of per-crop histograms (re-normalized) — the tracklet/person descriptor."""
    hists = [h for h in hists if h is not None]
    if not hists:
        return None
    m = np.mean(np.stack(hists, axis=0), axis=0).astype(np.float32)
    cv2.normalize(m, m, 0, 1, cv2.NORM_MINMAX)
    return m


def descriptor_from_entries(entries, max_frames: int = 12) -> np.ndarray | None:
    """Build a torso color descriptor from tracklet/cluster entries.

    Prefers the highest-visibility, least-overlapped entries (cleanest torso)."""
    if not entries:
        return None
    ranked = sorted(
        entries,
        key=lambda e: (float(getattr(e, "v_score", 0.0)),
                       -float(getattr(e, "overlap_ratio", 0.0))),
        reverse=True,
    )
    hists = []
    for e in ranked:
        if len(hists) >= max_frames:
            break
        hh = torso_hist(getattr(e, "crop", None))
        if hh is not None:
            hists.append(hh)
    return aggregate(hists)


def color_sim(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """CORREL similarity in [-1, 1] (higher = more similar color), or None."""
    if a is None or b is None:
        return None
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))
