DEFAULT_WEIGHTS = {
    "cut_off": 0.25,
    "area_ratio": 0.20,
    "aspect_ratio": 0.15,
    "det_conf": 0.10,
    "person_overlap": 0.30,
}

DEFAULT_CUTOFF_MARGIN_RATIO = 5.0 / 1080.0
_PENALTIES = {0: 1.0, 1: 0.6, 2: 0.3, 3: 0.1, 4: 0.05}

# compute cut-off score
def compute_cutoff(
    bbox_xyxy: list[float],
    frame_w: int,
    frame_h: int,
    *,
    margin_ratio: float = DEFAULT_CUTOFF_MARGIN_RATIO,
    margin_px: int | None = None,
) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    if frame_w <= 0 or frame_h <= 0:
        return _PENALTIES[4]
    if x2 <= x1 or y2 <= y1:
        return _PENALTIES[4]

    if margin_px is not None:
        m = max(1, margin_px)
    else:
        short = min(frame_w, frame_h)
        m = max(1, int(round(short * margin_ratio)))

    mx = min(m, max(0, frame_w // 2))
    my = min(m, max(0, frame_h // 2))

    left = x1 <= mx
    top = y1 <= my
    right = x2 >= frame_w - mx
    bottom = y2 >= frame_h - my

    touches = int(left) + int(right) + int(top) + int(bottom)
    if touches == 2 and ((left and right) or (top and bottom)):
        touches = 3

    return _PENALTIES.get(touches, _PENALTIES[4])


# compute area ratio score (bbox/frame size)
def compute_area_ratio(bbox_xyxy: list[float], frame_w: int, frame_h: int) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    if frame_w <= 0 or frame_h <= 0:
        return 0.1
    if x2 <= x1 or y2 <= y1:
        return 0.1

    bbox_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h
    ratio = bbox_area / frame_area

    if ratio < 0.001:
        return 0.05
    elif ratio < 0.003:
        return 0.2
    elif ratio < 0.0075:
        return 0.5
    elif ratio < 0.015:
        return 0.8
    elif ratio < 0.12:
        return 1.0
    elif ratio < 0.20:
        return 0.9
    elif ratio < 0.30:
        return 0.75
    elif ratio < 0.45:
        return 0.55
    else:
        return 0.35


# compute aspect ratio score (bbox's height/weight)
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

# compute detection's confidence score (from previous detected by yolo)
def compute_det_conf_score(confidence: float) -> float:
    return max(0.3, min(1.0, (confidence - 0.25) / 0.75 * 0.7 + 0.3))


# compute overlap ratio score

def _is_same_bbox(bbox_xyxy: list[float], other_xyxy: list[float], eps: float = 1.0) -> bool:
    return all(abs(a - b) < eps for a, b in zip(bbox_xyxy, other_xyxy))


def _intersection_rect(
    bbox_xyxy: list[float],
    other_xyxy: list[float],
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = bbox_xyxy
    ox1, oy1, ox2, oy2 = other_xyxy
    ix1 = max(x1, ox1)
    iy1 = max(y1, oy1)
    ix2 = min(x2, ox2)
    iy2 = min(y2, oy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return (ix1, iy1, ix2, iy2)


def _union_area(rects: list[tuple[float, float, float, float]]) -> float:
    if not rects:
        return 0.0

    xs = sorted({x1 for x1, _, x2, _ in rects} | {x2 for _, _, x2, _ in rects})
    total = 0.0

    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue

        intervals: list[tuple[float, float]] = []
        for x1, y1, x2, y2 in rects:
            if x1 < right and x2 > left:
                intervals.append((y1, y2))

        if not intervals:
            continue

        intervals.sort()
        covered = 0.0
        cur_start, cur_end = intervals[0]
        for start, end in intervals[1:]:
            if start > cur_end:
                covered += cur_end - cur_start
                cur_start, cur_end = start, end
            else:
                cur_end = max(cur_end, end)
        covered += cur_end - cur_start

        total += covered * (right - left)

    return total


def compute_overlap_ratio(bbox_xyxy: list[float], all_bboxes: list[list[float]]) -> float:
    if len(all_bboxes) <= 1:
        return 0.0

    x1, y1, x2, y2 = bbox_xyxy
    if x2 <= x1 or y2 <= y1:
        return 0.0

    bbox_area = (x2 - x1) * (y2 - y1)
    overlap_rects: list[tuple[float, float, float, float]] = []

    for other in all_bboxes:
        if _is_same_bbox(bbox_xyxy, other):
            continue

        rect = _intersection_rect(bbox_xyxy, other)
        if rect is not None:
            overlap_rects.append(rect)

    union_overlap = _union_area(overlap_rects)
    return min(union_overlap / bbox_area, 1.0)


def compute_person_overlap(bbox_xyxy: list[float], all_bboxes: list[list[float]]) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    if x2 <= x1 or y2 <= y1:
        return 0.2
    if len(all_bboxes) <= 1:
        return 1.0

    max_overlap_ratio = compute_overlap_ratio(bbox_xyxy, all_bboxes)

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
