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

        self.latest_box_count = 0
        self.frames_until_next_process = 0
        self._last_gray = None

    def _downsample_gray(self, frame):
        resized = cv2.resize(frame, self.gray_size, interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:
            return resized
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    def _remember_frame(self, frame) -> None:
        self._last_gray = self._downsample_gray(frame)

    def _has_large_change(self, frame) -> bool:
        if self._last_gray is None:
            return True
        gray = self._downsample_gray(frame)
        diff = np.mean(cv2.absdiff(gray, self._last_gray))
        return bool(diff >= (255.0 * self.criterion_scale))

    def should_process(self, frame) -> bool:
        if self.frames_until_next_process <= 0:
            self._remember_frame(frame)
            return True

        if self._has_large_change(frame):
            self._remember_frame(frame)
            self.frames_until_next_process = 0
            return True

        self.frames_until_next_process -= 1
        return False

    def update_after_detection(self, detections: list[dict]) -> None:
        self.latest_box_count = len(detections)
        self.frames_until_next_process = (
            self.max_skip_with_boxes if self.latest_box_count > 0 else self.max_skip_without_boxes
        )
