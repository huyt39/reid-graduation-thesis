from collections import OrderedDict
from typing import Any

import numpy as np


class TrackState:
    # các trạng thái cơ bản của một track
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    # bộ đếm id dùng chung cho các track
    _count = 0

    # khởi tạo thông tin nền tảng của một track
    def __init__(self):
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.history = OrderedDict()
        self.features = []
        self.curr_feature = None
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.location = (np.inf, np.inf)

    @property
    # frame cuối cùng mà track được cập nhật
    def end_frame(self) -> int:
        return self.frame_id

    @staticmethod
    # cấp track id tăng dần
    def next_id() -> int:
        BaseTrack._count += 1
        return BaseTrack._count

    # hàm con phải tự định nghĩa cách kích hoạt track
    def activate(self, *args: Any) -> None:
        raise NotImplementedError

    # hàm con phải tự định nghĩa cách dự đoán vị trí track
    def predict(self) -> None:
        raise NotImplementedError

    # hàm con phải tự định nghĩa cách cập nhật track
    def update(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    # đánh dấu track bị mất tạm thời
    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    # đánh dấu track đã bị loại bỏ
    def mark_removed(self) -> None:
        self.state = TrackState.Removed

    @staticmethod
    # reset bộ đếm track id
    def reset_id() -> None:
        BaseTrack._count = 0
