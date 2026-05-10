from typing import Union

import cv2
import numpy as np


def xywh2ltwh(x: list):
    y = np.array(x, copy=True)
    y[0] = x[0] - x[2] / 2
    y[1] = x[1] - x[3] / 2
    return y


def xyxy2xywh(x: list):
    assert len(x) == 4
    y = np.array(x, copy=True)
    y[0] = (x[0] + x[2]) / 2
    y[1] = (x[1] + x[3]) / 2
    y[2] = x[2] - x[0]
    y[3] = x[3] - x[1]
    return y


def xywh2xyxy(x: list):
    assert len(x) == 4
    y = np.array(x, copy=True)
    xy = x[:2]
    wh = x[2:] / 2
    y[:2] = xy - wh
    y[2:] = xy + wh
    return y


def _get_covariance_matrix(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gbbs = np.concatenate((np.square(boxes[:, 2:4]) / 12.0, boxes[:, 4:]), axis=-1)
    a, b, c = np.split(gbbs, 3, axis=-1)
    cos = np.cos(c)
    sin = np.sin(c)
    cos2 = np.square(cos)
    sin2 = np.square(sin)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def batch_probiou(
    obb1: Union[np.ndarray, list],
    obb2: Union[np.ndarray, list],
    eps: float = 1e-7,
) -> np.ndarray:
    obb1 = np.asarray(obb1, dtype=np.float32)
    obb2 = np.asarray(obb2, dtype=np.float32)

    x1 = obb1[:, 0:1]
    y1 = obb1[:, 1:2]
    x2 = obb2[:, 0][None, :]
    y2 = obb2[:, 1][None, :]
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x[:, 0][None, :] for x in _get_covariance_matrix(obb2))

    denominator = (a1 + a2) * (b1 + b2) - np.square(c1 + c2) + eps
    t1 = (((a1 + a2) * np.square(y1 - y2) + (b1 + b2) * np.square(x1 - x2)) / denominator) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / denominator) * 0.5

    det1 = np.clip(a1 * b1 - np.square(c1), 0.0, None)
    det2 = np.clip(a2 * b2 - np.square(c2), 0.0, None)
    t3 = (
        np.log((((a1 + a2) * (b1 + b2) - np.square(c1 + c2)) / (4 * np.sqrt(det1 * det2) + eps)) + eps)
        * 0.5
    )

    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd


def bbox_ioa(box1: np.ndarray, box2: np.ndarray, iou: bool = False, eps: float = 1e-7) -> np.ndarray:
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area
    return inter_area / (area + eps)


def crop_image(image: np.ndarray, bboxes: list[list[float]]) -> list[np.ndarray]:
    cropped_images = []
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cropped_images.append(image[y1:y2, x1:x2])
    return cropped_images
