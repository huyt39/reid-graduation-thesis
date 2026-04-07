import numpy as np

from src.filtering.pre_skip import PreFrameSkipper
from src.filtering.post_skip import PostFrameSkipper
from src.scoring.tagging import VisibilityTag


class TestPreFrameSkipper:
    @staticmethod
    def _frame(value: int, shape: tuple[int, int, int] = (32, 32, 3)) -> np.ndarray:
        return np.full(shape, value, dtype=np.uint8)

    def test_first_frame_is_processed(self):
        skipper = PreFrameSkipper()
        assert skipper.should_process(self._frame(0))

    def test_small_change_skips_until_limit(self):
        skipper = PreFrameSkipper(max_skip_with_boxes=2, max_skip_without_boxes=4)
        assert skipper.should_process(self._frame(0))
        skipper.update_after_detection([{"bbox": [0, 0, 10, 10]}])

        assert not skipper.should_process(self._frame(0))
        assert not skipper.should_process(self._frame(0))
        assert skipper.should_process(self._frame(0))

    def test_large_change_triggers_processing(self):
        skipper = PreFrameSkipper(max_skip_with_boxes=5, max_skip_without_boxes=30)
        assert skipper.should_process(self._frame(0))
        skipper.update_after_detection([{"bbox": [0, 0, 10, 10]}])

        assert skipper.should_process(self._frame(255))

    def test_no_boxes_uses_longer_skip_limit(self):
        skipper = PreFrameSkipper(max_skip_with_boxes=2, max_skip_without_boxes=3)
        assert skipper.should_process(self._frame(0))
        skipper.update_after_detection([])

        assert not skipper.should_process(self._frame(0))
        assert not skipper.should_process(self._frame(0))
        assert not skipper.should_process(self._frame(0))
        assert skipper.should_process(self._frame(0))


class TestPostFrameSkipper:
    def test_good_tag_is_sent_every_second_hit(self):
        skipper = PostFrameSkipper(rates={"good": 2, "mid": 3, "bad": 5})
        results = [skipper.should_send(VisibilityTag.GOOD, 0.8, "key1") for _ in range(6)]
        assert results == [False, True, False, True, False, True]

    def test_bad_tag_is_sent_every_fifth_hit(self):
        skipper = PostFrameSkipper(rates={"good": 2, "mid": 3, "bad": 5})
        results = [skipper.should_send(VisibilityTag.BAD, 0.3, "key2") for _ in range(10)]
        assert results == [False, False, False, False, True, False, False, False, False, True]

    def test_drop_floor_blocks_low_visibility(self):
        skipper = PostFrameSkipper(drop_floor=0.15)
        assert not skipper.should_send(VisibilityTag.BAD, 0.10, "key3")

    def test_stale_keys_are_pruned(self):
        skipper = PostFrameSkipper(rates={"good": 2, "mid": 3, "bad": 5}, stale_after=2)
        assert not skipper.should_send(VisibilityTag.GOOD, 0.8, "key1", frame_idx=1)
        assert skipper.should_send(VisibilityTag.GOOD, 0.8, "key1", frame_idx=2)
        assert not skipper.should_send(VisibilityTag.GOOD, 0.8, "other", frame_idx=5)
        assert not skipper.should_send(VisibilityTag.GOOD, 0.8, "key1", frame_idx=6)
