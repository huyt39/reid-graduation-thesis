import numpy as np

from src.tracklet.buffer import TrackletBuffer
from src.tracklet.models import TrackletEntry, TrackletState


def _make_entry(frame_idx: int, ts_offset_s: float = 0.0) -> TrackletEntry:
    return TrackletEntry(
        frame_idx=frame_idx,
        crop=np.zeros((64, 32, 3), dtype=np.uint8),
        v_score=0.8,
        bbox_xyxy=[10.0, 20.0, 50.0, 120.0],
        timestamp_ns=int(ts_offset_s * 1e9),
    )


def test_not_ready_until_window_has_elapsed():
    buf = TrackletBuffer(min_entries=3, window_seconds=1.0)
    for i in range(5):
        buf.append(1, _make_entry(i, ts_offset_s=0.0))
    ready = buf.get_ready_tracklets(current_time_ns=int(0.5e9))
    assert ready == []
    assert buf.tracklets[1].state == TrackletState.ACTIVE


def test_ready_after_enough_entries_and_window_elapsed():
    buf = TrackletBuffer(min_entries=3, window_seconds=1.0)
    for i in range(5):
        buf.append(1, _make_entry(i, ts_offset_s=i * 0.25))
    ready = buf.get_ready_tracklets(current_time_ns=int(1.0e9))
    assert len(ready) == 1
    assert ready[0].state == TrackletState.READY


def test_ready_when_max_entries_reached_even_before_window():
    buf = TrackletBuffer(min_entries=3, max_entries=4, window_seconds=10.0)
    for i in range(4):
        buf.append(1, _make_entry(i, ts_offset_s=i * 0.1))

    ready = buf.pop_ready_tracklets(current_time_ns=int(0.4e9))

    assert len(ready) == 1
    assert [entry.frame_idx for entry in ready[0].entries] == [0, 1, 2, 3]


def test_pop_ready_tracklets_closes_elapsed_window_before_more_appends():
    buf = TrackletBuffer(min_entries=3, window_seconds=1.0)
    for i in range(3):
        buf.append(1, _make_entry(i, ts_offset_s=i * 0.5))

    ready = buf.pop_ready_tracklets(current_time_ns=int(1.0e9))
    assert len(ready) == 1
    assert [entry.frame_idx for entry in ready[0].entries] == [0, 1, 2]

    buf.append(1, _make_entry(3, ts_offset_s=0.1))
    assert [entry.frame_idx for entry in ready[0].entries] == [0, 1, 2]
    assert [entry.frame_idx for entry in buf.tracklets[1].entries] == [3]


def test_pop_ready_tracklets_skips_tracks_already_processing():
    buf = TrackletBuffer(min_entries=3, window_seconds=1.0)
    for i in range(3):
        buf.append(1, _make_entry(i, ts_offset_s=i * 0.5))
        buf.append(2, _make_entry(i, ts_offset_s=i * 0.5))

    ready = buf.pop_ready_tracklets(current_time_ns=int(1.0e9), skip_track_ids={1})

    assert [tracklet.track_id for tracklet in ready] == [2]
    assert 1 in buf.tracklets
    assert [entry.frame_idx for entry in buf.tracklets[1].entries] == [0, 1, 2]
    assert 2 not in buf.tracklets


def test_pop_stale_tracklets_skips_tracks_already_processing():
    buf = TrackletBuffer(min_entries=3, stale_seconds=1.0)
    buf.append(1, _make_entry(1, ts_offset_s=0.0))
    buf.append(2, _make_entry(1, ts_offset_s=0.0))

    stale = buf.pop_stale_tracklets(current_time_ns=int(2.0e9), skip_track_ids={1})

    assert [tracklet.track_id for tracklet in stale] == [2]
    assert 1 in buf.tracklets
    assert 2 not in buf.tracklets
