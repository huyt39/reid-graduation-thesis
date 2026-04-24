from src.services.frame_cache import FrameCache, FrameData


def _make_frame(device_id: str = "cam-1", frame_number: int = 1) -> FrameData:
    return FrameData(
        device_id=device_id,
        frame_number=frame_number,
        tracked_persons=[],
        created_at=0,
        image_base64="abc",
    )


def test_update_and_get():
    cache = FrameCache()
    frame = _make_frame()
    cache.update(frame)
    assert cache.get("cam-1") is frame
    assert cache.get("cam-unknown") is None


def test_device_ids_dynamic():
    cache = FrameCache()
    assert cache.device_ids() == []
    cache.update(_make_frame("cam-1"))
    cache.update(_make_frame("cam-2"))
    assert sorted(cache.device_ids()) == ["cam-1", "cam-2"]


def test_update_overwrites_previous():
    cache = FrameCache()
    cache.update(_make_frame("cam-1", frame_number=1))
    cache.update(_make_frame("cam-1", frame_number=5))
    assert cache.get("cam-1").frame_number == 5
