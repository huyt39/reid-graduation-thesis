"""Gender voting with per-tracklet majority vote and per-person hysteresis."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _TrackletVote:
    male: int = 0
    female: int = 0
    total_conf: float = 0.0


@dataclass
class _PersonHistory:
    current_gender: str = "unknown"
    current_confidence: float = 0.0
    consecutive_agree: int = 0
    last_tracklet_gender: str = "unknown"


class GenderVoter:
    """Two-level gender voting.

    Level 1 — per tracklet: accumulate frame-level predictions, majority wins.
    Level 2 — per person: only change the person-level label when 2 consecutive
    tracklets agree on a *different* label with confidence above ``person_threshold``.
    """

    def __init__(self, person_threshold: float = 0.7) -> None:
        self.person_threshold = person_threshold
        self._tracklet_votes: dict[int, _TrackletVote] = {}  # track_id -> vote
        self._person_history: dict[int, _PersonHistory] = {}  # person_id -> history

    # ── Tracklet-level ────────────────────────────────────────────────

    def vote_frame(self, track_id: int, gender: str, confidence: float) -> None:
        v = self._tracklet_votes.setdefault(track_id, _TrackletVote())
        if gender == "male":
            v.male += 1
        elif gender == "female":
            v.female += 1
        v.total_conf += confidence

    def resolve_tracklet(self, track_id: int) -> tuple[str, float]:
        """Return (gender, avg_confidence) for a tracklet. Removes state."""
        v = self._tracklet_votes.pop(track_id, _TrackletVote())
        total = v.male + v.female
        if total == 0:
            return "unknown", 0.0
        gender = "male" if v.male >= v.female else "female"
        avg_conf = v.total_conf / total
        return gender, round(avg_conf, 4)

    # ── Person-level (hysteresis) ─────────────────────────────────────

    def resolve_person(
        self, person_id: int, tracklet_gender: str, tracklet_confidence: float,
    ) -> tuple[str, float]:
        """Update person-level label with hysteresis.

        Returns the (possibly unchanged) person-level (gender, confidence).
        """
        h = self._person_history.setdefault(person_id, _PersonHistory())

        # First assignment
        if h.current_gender == "unknown":
            if tracklet_confidence >= self.person_threshold:
                h.current_gender = tracklet_gender
                h.current_confidence = tracklet_confidence
            return h.current_gender, h.current_confidence

        # Same as current — reinforce
        if tracklet_gender == h.current_gender:
            h.consecutive_agree = 0
            h.last_tracklet_gender = tracklet_gender
            # Slowly update confidence
            h.current_confidence = 0.8 * h.current_confidence + 0.2 * tracklet_confidence
            return h.current_gender, round(h.current_confidence, 4)

        # Different label — count consecutive agreements on the *new* label
        if tracklet_gender == h.last_tracklet_gender:
            h.consecutive_agree += 1
        else:
            h.consecutive_agree = 1
        h.last_tracklet_gender = tracklet_gender

        # Flip only if 2 consecutive tracklets agree AND confidence is high
        if h.consecutive_agree >= 2 and tracklet_confidence >= self.person_threshold:
            h.current_gender = tracklet_gender
            h.current_confidence = tracklet_confidence
            h.consecutive_agree = 0

        return h.current_gender, round(h.current_confidence, 4)
