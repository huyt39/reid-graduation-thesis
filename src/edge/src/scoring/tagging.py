from enum import Enum

class VisibilityTag(str, Enum):
    GOOD = "good"
    MID = "mid"
    BAD = "bad"

def tag_detection(v: float, good_thresh: float = 0.7, mid_thresh: float = 0.4) -> VisibilityTag:
    if v >= good_thresh:
        return VisibilityTag.GOOD
    elif v >= mid_thresh:
        return VisibilityTag.MID
    else:
        return VisibilityTag.BAD
        