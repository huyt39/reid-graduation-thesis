import asyncio

from src.services.frame_cache import FrameData
from src.workers import kafka_loop
from src.workers.kafka_loop import _with_snapshot_urls, run_kafka_loop


class StubMinIO:
    def presigned_url(self, object_key: str | None) -> str | None:
        if not object_key:
            return None
        return f"https://example.com/{object_key}"


def test_with_snapshot_urls_enriches_without_mutating_input():
    tracked_persons = [
        {"person_id": 1, "snapshot_key": "persons/1/best.jpg"},
        {"person_id": 2, "snapshot_key": None},
    ]

    enriched = _with_snapshot_urls(tracked_persons, StubMinIO())

    assert tracked_persons[0].get("snapshot_url") is None
    assert enriched[0]["snapshot_url"] == "https://example.com/persons/1/best.jpg"
    assert enriched[1]["snapshot_url"] is None


class StubConsumer:
    def __init__(self, messages):
        self._messages = list(messages)

    def poll(self, timeout_ms: int = 1000, max_records: int = 50):
        if self._messages:
            messages, self._messages = self._messages[:max_records], self._messages[max_records:]
            return messages
        return []


class StubCache:
    def __init__(self):
        self.frames = []

    def update(self, frame):
        self.frames.append(frame)


class StubBroadcaster:
    def __init__(self):
        self.frames = []

    async def broadcast(self, frame):
        self.frames.append(frame)


def _message(frame_number: int) -> dict:
    return {"device_id": "cam-1", "frame_number": frame_number}


def test_run_kafka_loop_throttles_broadcasts_per_device(monkeypatch):
    consumer = StubConsumer([_message(1), _message(2), _message(3)])
    cache = StubCache()
    broadcaster = StubBroadcaster()

    monkeypatch.setattr(
        kafka_loop,
        "_decode_frame",
        lambda msg, jpeg_quality, *, minio_urls, source: FrameData(
            device_id=msg["device_id"],
            frame_number=msg["frame_number"],
            tracked_persons=[],
            created_at=msg["frame_number"],
            image_base64="frame",
            source=source,
        ),
    )

    async def runner():
        task = asyncio.create_task(
            run_kafka_loop(
                consumer,
                cache,
                broadcaster,
                max_poll_records=10,
                jpeg_quality=75,
                broadcast_max_fps=1.0,
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(runner())

    assert len(cache.frames) == 3
    assert len(broadcaster.frames) == 1
    assert broadcaster.frames[0].frame_number == 1
