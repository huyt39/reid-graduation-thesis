"""Tracklet consistency features.

Distinguishes genuinely stable tracklets of one person from tracklets that
are long but mix multiple persons (e.g., due to ID switches in the tracker).
"""

from dataclasses import dataclass

from src.tracklet.models import TrackletEntry


@dataclass
class TrackletConsistency:
    bbox_size_stability: float  # 1.0 = perfectly stable, 0.0 = highly unstable
    position_stability: float   # 1.0 = smooth movement, 0.0 = erratic jumps
    good_frame_streak: int      # max consecutive frames with v_score >= threshold
    good_frame_ratio: float     # fraction of frames that are "good"
    overall: float              # composite consistency score in [0, 1]


def compute_bbox_size_stability(entries: list[TrackletEntry]) -> float:
    """Measure how stable the bbox area is across consecutive frames.

    For each pair of consecutive frames, compute:
        delta_area = abs(area_t - area_{t-1}) / area_{t-1}
    Then return 1 - mean(delta_area), clamped to [0, 1].

    High stability (close to 1.0) means the person's apparent size doesn't
    jump erratically, which suggests consistent tracking of one person.
    """
    if len(entries) < 2:
        return 1.0

    deltas = []
    for i in range(1, len(entries)):
        prev = entries[i - 1].bbox_xyxy
        curr = entries[i].bbox_xyxy
        area_prev = max((prev[2] - prev[0]) * (prev[3] - prev[1]), 1)
        area_curr = max((curr[2] - curr[0]) * (curr[3] - curr[1]), 1)
        delta = abs(area_curr - area_prev) / area_prev
        deltas.append(delta)

    mean_delta = sum(deltas) / len(deltas)
    return max(0.0, min(1.0, 1.0 - mean_delta))


def compute_position_stability(entries: list[TrackletEntry]) -> float:
    """Measure how smooth the bbox center movement is across the tracklet.

    For each pair of consecutive frames, compute normalized displacement
    (displacement / bbox_size). Returns 1 - mean(normalized_disp), clamped.

    Reuses the same logic as vel_smooth but aggregated at tracklet level.
    """
    if len(entries) < 2:
        return 1.0

    displacements = []
    for i in range(1, len(entries)):
        prev = entries[i - 1].bbox_xyxy
        curr = entries[i].bbox_xyxy
        cx_prev = (prev[0] + prev[2]) / 2
        cy_prev = (prev[1] + prev[3]) / 2
        cx_curr = (curr[0] + curr[2]) / 2
        cy_curr = (curr[1] + curr[3]) / 2

        displacement = ((cx_curr - cx_prev) ** 2 + (cy_curr - cy_prev) ** 2) ** 0.5
        bbox_size = max(curr[2] - curr[0], curr[3] - curr[1], 1)
        normalized = displacement / bbox_size
        displacements.append(normalized)

    mean_disp = sum(displacements) / len(displacements)
    # Normalize: displacement < 0.3 bbox_size per frame is normal
    return max(0.0, min(1.0, 1.0 - mean_disp / 0.5))


def compute_good_frame_streak(entries: list[TrackletEntry], good_threshold: float = 0.6) -> int:
    """Find the max consecutive streak of frames with v_score >= threshold."""
    max_streak = 0
    current_streak = 0
    for entry in entries:
        if entry.v_score >= good_threshold:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak


def compute_good_frame_ratio(entries: list[TrackletEntry], good_threshold: float = 0.6) -> float:
    """Fraction of frames that are 'good' (v_score >= threshold)."""
    if not entries:
        return 0.0
    good_count = sum(1 for e in entries if e.v_score >= good_threshold)
    return good_count / len(entries)


def compute_tracklet_consistency(
    entries: list[TrackletEntry],
    good_threshold: float = 0.6,
) -> TrackletConsistency:
    """Compute all consistency features for a tracklet.

    Returns a TrackletConsistency dataclass with individual features
    and an overall composite score.
    """
    size_stab = compute_bbox_size_stability(entries)
    pos_stab = compute_position_stability(entries)
    streak = compute_good_frame_streak(entries, good_threshold)
    ratio = compute_good_frame_ratio(entries, good_threshold)

    # Overall consistency: weighted combination
    overall = 0.35 * size_stab + 0.35 * pos_stab + 0.30 * ratio

    return TrackletConsistency(
        bbox_size_stability=round(size_stab, 4),
        position_stability=round(pos_stab, 4),
        good_frame_streak=streak,
        good_frame_ratio=round(ratio, 4),
        overall=round(overall, 4),
    )
