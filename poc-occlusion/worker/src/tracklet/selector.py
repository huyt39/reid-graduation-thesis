from src.tracklet.models import TrackletEntry


class TopKSelector:
    """Select top-K frames by overlap-penalized visibility score with temporal diversity.

    selection_score = v_worker - lambda * overlap_ratio

    This penalizes frames where the person is heavily overlapped by another person,
    preferring frames with clear, unoccluded views even if v_worker is slightly lower.
    """

    def __init__(
        self,
        k: int = 5,
        min_temporal_gap: int = 3,
        overlap_lambda: float = 0.3,
        min_tracklet_len: int = 8,
        min_high_quality_frames: int = 3,
        high_quality_threshold: float = 0.6,
    ):
        self.k = k
        self.min_temporal_gap = min_temporal_gap
        self.overlap_lambda = overlap_lambda
        self.min_tracklet_len = min_tracklet_len
        self.min_high_quality_frames = min_high_quality_frames
        self.high_quality_threshold = high_quality_threshold

    def _selection_score(self, entry: TrackletEntry) -> float:
        return entry.v_score - self.overlap_lambda * entry.overlap_ratio

    def is_tracklet_ready(self, entries: list[TrackletEntry]) -> bool:
        """Check if a tracklet meets minimum quality requirements for embedding.

        A tracklet must be long enough AND have enough high-quality frames.
        If not, it should stay tentative rather than being used for ID creation/update.
        """
        if len(entries) < self.min_tracklet_len:
            return False
        high_quality = sum(
            1 for e in entries if e.v_score >= self.high_quality_threshold
        )
        return high_quality >= self.min_high_quality_frames

    def select(self, entries: list[TrackletEntry]) -> list[TrackletEntry]:
        """Select top-K frames by overlap-penalized score with temporal diversity."""
        sorted_entries = sorted(entries, key=self._selection_score, reverse=True)
        selected: list[TrackletEntry] = []
        selected_frame_idxs: list[int] = []

        for entry in sorted_entries:
            if len(selected) >= self.k:
                break
            too_close = any(
                abs(entry.frame_idx - sel_idx) < self.min_temporal_gap
                for sel_idx in selected_frame_idxs
            )
            if not too_close:
                selected.append(entry)
                selected_frame_idxs.append(entry.frame_idx)

        # Fill remaining slots relaxing constraint
        if len(selected) < self.k:
            remaining = [e for e in sorted_entries if e not in selected]
            for entry in remaining:
                if len(selected) >= self.k:
                    break
                selected.append(entry)

        return selected
