"""Multi-attribute voting with per-tracklet majority + per-person hysteresis.

Generalizes the previous gender-only voter to handle the 8 PA-100K attributes (and
any other tasks the inference engine returns). Each task is voted independently;
per-person history is tracked per-task so a flip on `gender` doesn't reset state
on `lower` or `hat`.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _PersonAttrHistory:
    current_label: str = "unknown"
    current_confidence: float = 0.0
    consecutive_agree: int = 0
    last_tracklet_label: str = "unknown"
    stable_support: int = 0


@dataclass
class _TrackletTaskVote:
    counts: dict[str, int] = field(default_factory=dict)
    label_conf: dict[str, float] = field(default_factory=dict)  # per-label confidence sum
    n: int = 0


class AttributeVoter:
    """Two-level voting over arbitrary attribute tasks.

    Level 1 (per tracklet): accumulate frame-level predictions, majority wins.
    Level 2 (per person): only flip a person-level label when 2 consecutive
    tracklets agree on a *different* label with confidence >= ``person_threshold``.

    Frame inputs follow the inference engine's ``/attributes/classify`` shape:
    ``{task: {"label": str, "confidence": float, "probabilities": ...}}``.
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

    # ── Tracklet-level ────────────────────────────────────────────────

    def vote_frame(self, track_id: int, attrs: dict[str, dict]) -> None:
        """Accumulate one frame's predictions for the given tracklet."""
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
        """Return the current majority-vote gender without consuming tracklet state."""
        votes = self._tracklet_votes.get(track_id, {})
        v = votes.get("gender")
        if v is None or v.n == 0:
            return ("unknown", 0.0)
        best_label = max(v.label_conf.items(), key=lambda kv: kv[1])[0]
        best_label_avg_conf = v.label_conf[best_label] / v.counts[best_label]
        return (best_label, round(best_label_avg_conf, 4))

    def resolve_tracklet(self, track_id: int) -> dict[str, tuple[str, float]]:
        """Majority-vote each task. Returns ``{task: (label, avg_confidence)}``. Drops state."""
        votes = self._tracklet_votes.pop(track_id, {})
        out: dict[str, tuple[str, float]] = {}
        for task, v in votes.items():
            if v.n == 0:
                out[task] = ("unknown", 0.0)
                continue
            # Confidence-weighted majority: the label with the highest total confidence wins.
            # Per-label avg confidence is used for the threshold check so a label that wins
            # by count but with low per-frame confidence doesn't inflate the reported value.
            best_label = max(v.label_conf.items(), key=lambda kv: kv[1])[0]
            best_label_avg_conf = v.label_conf[best_label] / v.counts[best_label]
            out[task] = (best_label, round(best_label_avg_conf, 4))
        return out

    # ── Person-level (hysteresis, per task) ───────────────────────────

    def resolve_person(
        self,
        person_id: int,
        tracklet_attrs: dict[str, tuple[str, float]],
    ) -> dict[str, tuple[str, float]]:
        """Update per-person, per-task hysteresis. Returns the (possibly unchanged) snapshot."""
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

            # Same label as current — reinforce, slow exponential update.
            if t_label == h.current_label:
                h.consecutive_agree = 0
                h.last_tracklet_label = t_label
                h.stable_support += 1
                h.current_confidence = 0.8 * h.current_confidence + 0.2 * t_conf
                out[task] = (h.current_label, round(h.current_confidence, 4))
                continue

            # Different label — count consecutive agreements on the new label.
            if t_label == h.last_tracklet_label:
                h.consecutive_agree += 1
            else:
                h.consecutive_agree = 1
            h.last_tracklet_label = t_label

            # Flip only on 2 consecutive high-confidence agreements.
            flip_threshold = float(self.task_flip_thresholds.get(task, self.flip_threshold))
            if h.consecutive_agree >= 2 and t_conf >= flip_threshold:
                h.current_label = t_label
                h.current_confidence = t_conf
                h.consecutive_agree = 0
                h.stable_support = 1

            out[task] = (h.current_label, round(h.current_confidence, 4))
        return out

    # ── Read-only accessors (used by emission code) ───────────────────

    def person_snapshot(self, person_id: int) -> dict[str, tuple[str, float]]:
        """Return ``{task: (current_label, current_confidence)}`` for tasks seen so far."""
        per_person = self._person_history.get(person_id, {})
        return {task: (h.current_label, h.current_confidence) for task, h in per_person.items()}

    def known_person_ids(self) -> set[int]:
        """Return the set of person_ids that already have person-level history."""
        return set(self._person_history.keys())

    def person_task_stable_support(self, person_id: int, task: str) -> int:
        """Return how many resolved tracklets have reinforced the current task label."""
        per_person = self._person_history.get(person_id, {})
        history = per_person.get(task)
        if history is None or history.current_label == "unknown":
            return 0
        return history.stable_support
