from src.filtering.pre_skip import PreFrameSkipper
from src.filtering.post_skip import PostFrameSkipper
from src.scoring.tagging import VisibilityTag


class TestPreFrameSkipper:
    def test_skip_rate_2(self):
        skipper = PreFrameSkipper(skip_rate=2)
        results = [skipper.should_process(i) for i in range(1, 11)]
        assert results == [False, True, False, True, False, True, False, True, False, True]

    def test_skip_rate_1_processes_all(self):
        skipper = PreFrameSkipper(skip_rate=1)
        assert all(skipper.should_process(i) for i in range(1, 11))

    def test_skip_rate_3(self):
        skipper = PreFrameSkipper(skip_rate=3)
        results = [skipper.should_process(i) for i in range(1, 10)]
        assert results == [False, False, True, False, False, True, False, False, True]


class TestPostFrameSkipper:
    def test_good_sends_every_2nd(self):
        skipper = PostFrameSkipper(rates={"good": 2, "mid": 3, "bad": 5})
        results = [skipper.should_send(VisibilityTag.GOOD, 0.8, "key1") for _ in range(6)]
        assert results == [False, True, False, True, False, True]

    def test_bad_sends_every_5th(self):
        skipper = PostFrameSkipper(rates={"good": 2, "mid": 3, "bad": 5})
        results = [skipper.should_send(VisibilityTag.BAD, 0.3, "key2") for _ in range(10)]
        assert results.count(True) == 2

    def test_drop_floor(self):
        skipper = PostFrameSkipper(drop_floor=0.15)
        assert not skipper.should_send(VisibilityTag.BAD, 0.10, "key3")

    def test_above_drop_floor(self):
        skipper = PostFrameSkipper(drop_floor=0.15)
        # Should eventually send
        sent = any(skipper.should_send(VisibilityTag.BAD, 0.20, "key4") for _ in range(10))
        assert sent
