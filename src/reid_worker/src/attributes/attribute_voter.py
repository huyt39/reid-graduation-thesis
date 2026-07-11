"""Multi-attribute voting with per-tracklet majority + per-person hysteresis.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# person's current status
@dataclass
class _PersonAttrHistory:
    current_label: str = "unknown"
    current_confidence: float = 0.0
    consecutive_agree: int = 0
    last_tracklet_label: str = "unknown"
    stable_support: int = 0

# lưu phiếu trong một tracklet cho từng loại thuộc tính - đếm số lần mỗi label xuất hiện và tổng confidence của label đó
@dataclass
class _TrackletTaskVote:
    counts: dict[str, int] = field(default_factory=dict)
    label_conf: dict[str, float] = field(default_factory=dict)  # per-label confidence sum
    n: int = 0

# khởi tạo các ngưỡng quyết định và 2 bộ nhớ chính:_tracklet_votes: gom dự đoán theo từng track_id; _person_history: lưu lịch sử thuộc tính theo từng person_id
class AttributeVoter:
    """Two-level voting over arbitrary attribute tasks.
    Level 1 (per tracklet): accumulate frame-level predictions, majority wins.
    Level 2 (per person): only flip a person-level label when 2 consecutive
    """

    def __init__(
        self,
        person_threshold: float = 0.7,
        flip_threshold: float = 0.85,
        task_flip_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.person_threshold = person_threshold
        self.flip_threshold = flip_threshold
        self.task_flip_thresholds = task_flip_thresholds or {}
        # track_id -> task -> running vote tally
        self._tracklet_votes: dict[int, dict[str, _TrackletTaskVote]] = {}
        # person_id -> task -> hysteresis state
        self._person_history: dict[int, dict[str, _PersonAttrHistory]] = {}

    # tracklet level

    def vote_frame(self, track_id: int, attrs: dict[str, dict]) -> None:
        votes = self._tracklet_votes.setdefault(track_id, {})
        for task, info in attrs.items():
            label = info.get("label") if isinstance(info, dict) else None
            if not isinstance(label, str):
                continue
            confidence = float(info.get("confidence", 0.0))
            v = votes.setdefault(task, _TrackletTaskVote())
            v.counts[label] = v.counts.get(label, 0) + 1
            v.label_conf[label] = v.label_conf.get(label, 0.0) + confidence
            v.n += 1

    def peek_tracklet_gender(self, track_id: int) -> tuple[str, float]:
        votes = self._tracklet_votes.get(track_id, {})
        v = votes.get("gender")
        if v is None or v.n == 0:
            return ("unknown", 0.0)
        best_label = max(v.label_conf.items(), key=lambda kv: kv[1])[0]
        best_label_avg_conf = v.label_conf[best_label] / v.counts[best_label]
        return (best_label, round(best_label_avg_conf, 4))

    def resolve_tracklet(self, track_id: int) -> dict[str, tuple[str, float]]:
        votes = self._tracklet_votes.pop(track_id, {})
        out: dict[str, tuple[str, float]] = {}
        for task, v in votes.items():
            if v.n == 0:
                out[task] = ("unknown", 0.0)
                continue

            best_label = max(v.label_conf.items(), key=lambda kv: kv[1])[0]
            best_label_avg_conf = v.label_conf[best_label] / v.counts[best_label]
            out[task] = (best_label, round(best_label_avg_conf, 4))
        return out

    # person level

    def resolve_person(
        self,
        person_id: int,
        tracklet_attrs: dict[str, tuple[str, float]],
    ) -> dict[str, tuple[str, float]]:
        per_person = self._person_history.setdefault(person_id, {})
        out: dict[str, tuple[str, float]] = {}
        for task, (t_label, t_conf) in tracklet_attrs.items():
            h = per_person.setdefault(task, _PersonAttrHistory())

            if h.current_label == "unknown":
                if t_conf >= self.person_threshold:
                    h.current_label = t_label
                    h.current_confidence = t_conf
                    h.stable_support = 1
                out[task] = (h.current_label, h.current_confidence)
                continue

            if t_label == h.current_label:
                h.consecutive_agree = 0
                h.last_tracklet_label = t_label
                h.stable_support += 1
                h.current_confidence = 0.8 * h.current_confidence + 0.2 * t_conf
                out[task] = (h.current_label, round(h.current_confidence, 4))
                continue

            if t_label == h.last_tracklet_label:
                h.consecutive_agree += 1
            else:
                h.consecutive_agree = 1
            h.last_tracklet_label = t_label

            flip_threshold = float(self.task_flip_thresholds.get(task, self.flip_threshold))
            if h.consecutive_agree >= 2 and t_conf >= flip_threshold:
                h.current_label = t_label
                h.current_confidence = t_conf
                h.consecutive_agree = 0
                h.stable_support = 1

            out[task] = (h.current_label, round(h.current_confidence, 4))
        return out


    def person_snapshot(self, person_id: int) -> dict[str, tuple[str, float]]:
        per_person = self._person_history.get(person_id, {})
        return {task: (h.current_label, h.current_confidence) for task, h in per_person.items()}

    def known_person_ids(self) -> set[int]:
        return set(self._person_history.keys())

    def person_task_stable_support(self, person_id: int, task: str) -> int:
        per_person = self._person_history.get(person_id, {})
        history = per_person.get(task)
        if history is None or history.current_label == "unknown":
            return 0
        return history.stable_support
