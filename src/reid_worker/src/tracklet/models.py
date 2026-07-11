from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class TrackletState(str, Enum):
    ACTIVE = "active" # collect frames
    READY = "ready" # ready to be processes
    EMBEDDED = "embedded"
    MATCHED = "matched"
    TENTATIVE = "tentative"


@dataclass
class TrackletEntry: # once see person in 1 frame
    frame_idx: int
    crop: np.ndarray
    v_score: float
    bbox_xyxy: list[float]
    timestamp_ns: int
    attribute_crop: np.ndarray | None = None
    overlap_ratio: float = 0.0
    frame_w: int = 0
    frame_h: int = 0


@dataclass
class Tracklet:
    track_id: int
    entries: list[TrackletEntry] = field(default_factory=list)
    state: TrackletState = TrackletState.ACTIVE
    created_at_ns: int = 0
    person_id: int | None = None
