import numpy as np


def compute_iou_prev(bbox_curr: np.ndarray, bbox_prev: np.ndarray | None) -> float:
    if bbox_prev is None:
        return 0.5

    x1 = max(bbox_curr[0], bbox_prev[0])
    y1 = max(bbox_curr[1], bbox_prev[1])
    x2 = min(bbox_curr[2], bbox_prev[2])
    y2 = min(bbox_curr[3], bbox_prev[3])
    inter = max(0, x2 - x1)*max(0, y2 - y1)
    area_a = (bbox_curr[2] - bbox_curr[0])*(bbox_curr[3] - bbox_curr[1])
    area_b = (bbox_prev[2] - bbox_prev[0])*(bbox_prev[3] - bbox_prev[1])
    iou = inter / (area_a + area_b - inter + 1e-7)

    if iou >= 0.5:
        return 1.0
    elif iou >= 0.3:
        return 0.7
    elif iou >= 0.1:
        return 0.4
    else:
        return 0.2

def compute_vel_smooth(
    center_curr: np.ndarray,
    center_prev: np.ndarray | None,
    center_prev2: np.ndarray | None,
    bbox_size: float,
) -> float:
    if center_prev is None:
        return 0.5

    displacement = np.linalg.norm(center_curr - center_prev)
    normalized_disp = displacement / (bbox_size + 1e-7)

    if center_prev2 is not None:
        vel_curr = center_curr - center_prev
        vel_prev = center_prev - center_prev2
        accel = np.linalg.norm(vel_curr - vel_prev) / (bbox_size + 1e-7)
    else:
        accel = 0.0

    if normalized_disp < 0.3:
        disp_score = 1.0
    elif normalized_disp < 0.5:
        disp_score = 0.6
    else:
        disp_score = 0.3

    if accel < 0.2:
        accel_score = 1.0
    elif accel < 0.4:
        accel_score = 0.6
    else:
        accel_score = 0.3

    return 0.6*disp_score + 0.4*accel_score

def compute_v_worker(
    v_edge: float,
    iou_prev_score: float,
    vel_smooth_score: float,
    weights: dict[str, float] | None = None,
) -> float:
    w = weights or {"v_edge": 0.60, "iou_prev": 0.25, "vel_smooth": 0.15}
    return (
        w["v_edge"] * v_edge + w["iou_prev"] * iou_prev_score + w["vel_smooth"] * vel_smooth_score
    )
