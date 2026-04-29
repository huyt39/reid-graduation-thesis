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

    def should_process(self, frame) -> bool:
        if self.frames_until_next_process <= 0:
            return True

        self.frames_until_next_process -= 1
        return False

    def update_after_detection(self, detections: list[dict]) -> None:
        self.latest_box_count = len(detections)
        self.frames_until_next_process = (
            self.max_skip_with_boxes if self.latest_box_count > 0 else self.max_skip_without_boxes
        )
