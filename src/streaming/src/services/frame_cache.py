from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class FrameData:
    device_id: str
    frame_number: int
    tracked_persons: list[dict]
    created_at: int
    image_base64: str
    image_width: int | None = None
    image_height: int | None = None
    schema_version: int = 2
    source: str = "processed"


class FrameCache:
    """Thread-safe cache storing the latest frame per device."""

    def __init__(self) -> None:
        self._frames: dict[str, FrameData] = {}
        self._lock = threading.Lock()

    def update(self, frame: FrameData) -> None:
        with self._lock:
            self._frames[frame.device_id] = frame

    def get(self, device_id: str) -> FrameData | None:
        with self._lock:
            return self._frames.get(device_id)

    def device_ids(self) -> list[str]:
        with self._lock:
            return list(self._frames.keys())
