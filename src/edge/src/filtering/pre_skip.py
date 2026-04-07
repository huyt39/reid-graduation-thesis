import cv2
import numpy as np

class PreFrameSkipper:
    def __init__(
        self,
        *,
        max_skip_with_boxes: int = 5,
        max_skip_without_boxes: int = 30,
        box_count_weight: float = 0.1,
        criterion_scale: float = 0.1,
        gray_size: tuple[int, int] = (160, 90),
    ):
        self.max_skip_with_boxes = max(1, max_skip_with_boxes)
        self.max_skip_without_boxes = max(1, max_skip_without_boxes)
        self.box_count_weight = max(0.0, box_count_weight)
        self.criterion_scale = max(0.0, criterion_scale)
        self.gray_size = gray_size

        self.prev_gray: np.ndarray | None = None
        self.skipped_frames = 0
        self.latest_box_count = 0
        self.latest_mean_box_area = 0.0
        self.skip_limit = self.max_skip_without_boxes

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.gray_size[0] > 0 and self.gray_size[1] > 0:
            gray = cv2.resize(gray, self.gray_size, interpolation=cv2.INTER_AREA)
        return gray

    def _criterion(self, frame_area: float) -> float:
        if frame_area <= 0:
            return 0.0
        return (
            (self.latest_mean_box_area / frame_area)
            + self.box_count_weight * self.latest_box_count
        ) * self.criterion_scale

    @staticmethod
    def _frame_change_ratio(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        if prev_gray.shape != curr_gray.shape:
            raise ValueError("Grayscale frames must have the same shape")
        diff = np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32))
        return float(diff.mean() / 255.0)

    def should_process(self, frame: np.ndarray) -> bool:
        curr_gray = self._to_gray(frame)
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return True

        change_ratio = self._frame_change_ratio(self.prev_gray, curr_gray)
        frame_h, frame_w = frame.shape[:2]
        criterion = self._criterion(frame_w * frame_h)

        self.prev_gray = curr_gray

        if change_ratio > criterion:
            return True
        if self.skipped_frames >= self.skip_limit:
            return True

        self.skipped_frames += 1
        return False

    def update_after_detection(self, detections: list[dict]) -> None:
        areas = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            if x2 > x1 and y2 > y1:
                areas.append((x2 - x1) * (y2 - y1))

        self.latest_box_count = len(areas)
        self.latest_mean_box_area = float(sum(areas) / len(areas)) if areas else 0.0
        self.skip_limit = (
            self.max_skip_with_boxes if self.latest_box_count > 0 else self.max_skip_without_boxes
        )
        self.skipped_frames = 0
