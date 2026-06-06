import asyncio
import json

import pytest
from starlette.websockets import WebSocketState

from src.services.broadcaster import WebSocketBroadcaster
from src.services.frame_cache import FrameCache, FrameData


# ── Fake WebSocket ────────────────────────────────────────────────────

class FakeWebSocket:
    """Minimal stand-in for fastapi.WebSocket for unit testing."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.client_state = WebSocketState.CONNECTED
        self.client = ("127.0.0.1", 9999)

    async def send_text(self, data: str) -> None:
        self.sent.append(data)

    async def send_json(self, data: dict) -> None:
        self.sent.append(json.dumps(data))


class BrokenWebSocket(FakeWebSocket):
    async def send_text(self, data: str) -> None:
        raise ConnectionError("client gone")


# ── Tests ─────────────────────────────────────────────────────────────

@pytest.fixture
def frame_cache() -> FrameCache:
    cache = FrameCache()
    cache.update(
        FrameData(
            device_id="cam-1",
            frame_number=10,
            tracked_persons=[{"person_id": 1, "bbox": [0, 0, 50, 100]}],
            created_at=1234567890,
            image_base64="base64img",
        )
    )
    return cache


@pytest.fixture
def broadcaster(frame_cache: FrameCache) -> WebSocketBroadcaster:
    return WebSocketBroadcaster(frame_cache, semaphore_limit=5)


@pytest.mark.asyncio
async def test_send_device_list(broadcaster: WebSocketBroadcaster):
    ws = FakeWebSocket()
    await broadcaster.send_device_list(ws)
    msg = json.loads(ws.sent[0])
    assert msg["type"] == "device_list"
    assert "cam-1" in msg["devices"]


@pytest.mark.asyncio
async def test_send_latest_frame(broadcaster: WebSocketBroadcaster):
    ws = FakeWebSocket()
    await broadcaster.send_latest_frame(ws, "cam-1")
    msg = json.loads(ws.sent[0])
    assert msg["type"] == "frame_update"
    assert msg["device_id"] == "cam-1"
    assert msg["frame_number"] == 10


@pytest.mark.asyncio
async def test_send_latest_frame_unknown_device(broadcaster: WebSocketBroadcaster):
    ws = FakeWebSocket()
    await broadcaster.send_latest_frame(ws, "nonexistent")
    msg = json.loads(ws.sent[0])
    assert msg["type"] == "error"


@pytest.mark.asyncio
async def test_broadcast_reaches_all_clients(broadcaster: WebSocketBroadcaster):
    ws1, ws2 = FakeWebSocket(), FakeWebSocket()
    broadcaster.add(ws1)
    broadcaster.add(ws2)
    broadcaster.subscribe(ws1, ["cam-2"])
    broadcaster.subscribe(ws2, ["cam-2"])

    frame = FrameData(
        device_id="cam-2",
        frame_number=42,
        tracked_persons=[],
        created_at=0,
        image_base64="x",
    )
    await broadcaster.broadcast(frame)

    for ws in (ws1, ws2):
        msg = json.loads(ws.sent[0])
        assert msg["device_id"] == "cam-2"
        assert msg["frame_number"] == 42


@pytest.mark.asyncio
async def test_broadcast_skips_unsubscribed_clients(broadcaster: WebSocketBroadcaster):
    ws = FakeWebSocket()
    broadcaster.add(ws)

    frame = FrameData(
        device_id="cam-1",
        frame_number=99,
        tracked_persons=[],
        created_at=0,
        image_base64="x",
    )
    await broadcaster.broadcast(frame)

    assert ws.sent == []


@pytest.mark.asyncio
async def test_broadcast_removes_broken_client(broadcaster: WebSocketBroadcaster):
    good_ws = FakeWebSocket()
    bad_ws = BrokenWebSocket()
    broadcaster.add(good_ws)
    broadcaster.add(bad_ws)
    broadcaster.subscribe(good_ws, ["cam-1"])
    broadcaster.subscribe(bad_ws, ["cam-1"])

    frame = FrameData(
        device_id="cam-1",
        frame_number=1,
        tracked_persons=[],
        created_at=0,
        image_base64="x",
    )
    await broadcaster.broadcast(frame)

    assert broadcaster.connection_count == 1
    assert len(good_ws.sent) == 1
