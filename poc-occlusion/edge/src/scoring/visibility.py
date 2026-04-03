DEFAULT_WEIGHTS = {
    "cut_off": 0.25,
    "area_ratio": 0.20,
    "aspect_ratio": 0.15,
    "det_conf": 0.10,
    "person_overlap": 0.30,
}


def compute_cutoff(bbox_xyxy: list[float], frame_w: int, frame_h: int, margin: int = 5) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    touches = 0
    if x1 <= margin:
        touches += 1
    if y1 <= margin:
        touches += 1
    if x2 >= frame_w - margin:
        touches += 1
    if y2 >= frame_h - margin:
        touches += 1
    penalties = {0: 1.0, 1: 0.6, 2: 0.3, 3: 0.1, 4: 0.05}
    return penalties.get(touches, 0.05)


def compute_area_ratio(bbox_xyxy: list[float], frame_w: int, frame_h: int) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    bbox_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h
    ratio = bbox_area / frame_area if frame_area > 0 else 0

    if ratio < 0.002:
        return 0.1
    elif ratio < 0.005:
        return 0.4
    elif ratio < 0.01:
        return 0.7
    elif ratio < 0.25:
        return 1.0
    elif ratio < 0.40:
        return 0.7
    else:
        return 0.4


def compute_aspect_ratio(bbox_xyxy: list[float]) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    ratio = h / w

    if 1.5 <= ratio <= 4.0:
        return 1.0
    elif 1.0 <= ratio < 1.5:
        return 0.7
    elif 4.0 < ratio <= 6.0:
        return 0.7
    elif 0.5 <= ratio < 1.0:
        return 0.4
    else:
        return 0.2


def compute_det_conf_score(confidence: float) -> float:
    return max(0.3, min(1.0, (confidence - 0.25) / 0.75 * 0.7 + 0.3))


def compute_person_overlap(bbox_xyxy: list[float], all_bboxes: list[list[float]]) -> float:

    if len(all_bboxes) <= 1:
        return 1.0 

    x1, y1, x2, y2 = bbox_xyxy
    bbox_area = max((x2 - x1) * (y2 - y1), 1)
    max_overlap_ratio = 0.0

    for other in all_bboxes:
        ox1, oy1, ox2, oy2 = other
   
        if abs(ox1 - x1) < 1 and abs(oy1 - y1) < 1 and abs(ox2 - x2) < 1 and abs(oy2 - y2) < 1:
            continue

        ix1 = max(x1, ox1)
        iy1 = max(y1, oy1)
        ix2 = min(x2, ox2)
        iy2 = min(y2, oy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

        overlap_ratio = inter / bbox_area
        max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)

    if max_overlap_ratio < 0.05:
        return 1.0
    elif max_overlap_ratio < 0.15:
        return 0.8
    elif max_overlap_ratio < 0.30:
        return 0.6
    elif max_overlap_ratio < 0.50:
        return 0.4
    else:
        return 0.2


def compute_overlap_ratio(bbox_xyxy: list[float], all_bboxes: list[list[float]]) -> float:
    if len(all_bboxes) <= 1:
        return 0.0

    x1, y1, x2, y2 = bbox_xyxy
    bbox_area = max((x2 - x1) * (y2 - y1), 1)
    max_overlap_ratio = 0.0

    for other in all_bboxes:
        ox1, oy1, ox2, oy2 = other
        if abs(ox1 - x1) < 1 and abs(oy1 - y1) < 1 and abs(ox2 - x2) < 1 and abs(oy2 - y2) < 1:
            continue
        ix1 = max(x1, ox1)
        iy1 = max(y1, oy1)
        ix2 = min(x2, ox2)
        iy2 = min(y2, oy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        max_overlap_ratio = max(max_overlap_ratio, inter / bbox_area)

    return min(max_overlap_ratio, 1.0)


def compute_subscores(
    bbox_xyxy: list[float],
    confidence: float,
    frame_w: int,
    frame_h: int,
    all_bboxes: list[list[float]] | None = None,
) -> dict[str, float]:
    return {
        "cut_off": compute_cutoff(bbox_xyxy, frame_w, frame_h),
        "area_ratio": compute_area_ratio(bbox_xyxy, frame_w, frame_h),
        "aspect_ratio": compute_aspect_ratio(bbox_xyxy),
        "det_conf": compute_det_conf_score(confidence),
        "person_overlap": compute_person_overlap(bbox_xyxy, all_bboxes or []),
    }


def compute_visibility_score(
    subscores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:

    w = weights or DEFAULT_WEIGHTS
    v = sum(w[k] * subscores[k] for k in w)
    return round(v, 4)
