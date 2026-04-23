from dataclasses import dataclass

from src.tracklet.models import TrackletEntry


@dataclass
class TrackletConsistency:
    bbox_size_stability: float
    position_stability: float
    good_frame_streak: int
    good_frame_ratio: float
    overall: float


def compute_bbox_size_stability(entries: list[TrackletEntry]) -> float:
    if len(entries) < 2:
        return 1.0
    deltas = []
    for i in range(1, len(entries)):
        prev, curr = entries[i - 1].bbox_xyxy, entries[i].bbox_xyxy
        area_prev = max((prev[2] - prev[0]) * (prev[3] - prev[1]), 1)
        area_curr = max((curr[2] - curr[0]) * (curr[3] - curr[1]), 1)
        deltas.append(abs(area_curr - area_prev) / area_prev)
    mean_delta = sum(deltas) / len(deltas)
    return max(0.0, min(1.0, 1.0 - mean_delta))


def compute_position_stability(entries: list[TrackletEntry]) -> float:
    if len(entries) < 2:
        return 1.0
    displacements = []
    for i in range(1, len(entries)):
        prev, curr = entries[i - 1].bbox_xyxy, entries[i].bbox_xyxy
        cx_prev, cy_prev = (prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2
        cx_curr, cy_curr = (curr[0] + curr[2]) / 2, (curr[1] + curr[3]) / 2
        displacement = ((cx_curr - cx_prev) ** 2 + (cy_curr - cy_prev) ** 2) ** 0.5
        bbox_size = max(curr[2] - curr[0], curr[3] - curr[1], 1)
        displacements.append(displacement / bbox_size)
    mean_disp = sum(displacements) / len(displacements)
    return max(0.0, min(1.0, 1.0 - mean_disp / 0.5))


def compute_good_frame_streak(entries: list[TrackletEntry], good_threshold: float = 0.6) -> int:
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
    if not entries:
        return 0.0
    good_count = sum(1 for e in entries if e.v_score >= good_threshold)
    return good_count / len(entries)


def compute_tracklet_consistency(
    entries: list[TrackletEntry],
    good_threshold: float = 0.6,
) -> TrackletConsistency:
    size_stab = compute_bbox_size_stability(entries)
    pos_stab = compute_position_stability(entries)
    streak = compute_good_frame_streak(entries, good_threshold)
    ratio = compute_good_frame_ratio(entries, good_threshold)
    overall = 0.35 * size_stab + 0.35 * pos_stab + 0.30 * ratio
    return TrackletConsistency(
        bbox_size_stability=round(size_stab, 4),
        position_stability=round(pos_stab, 4),
        good_frame_streak=streak,
        good_frame_ratio=round(ratio, 4),
        overall=round(overall, 4),
    )
