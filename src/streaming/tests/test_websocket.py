import pytest
from fastapi.testclient import TestClient

from src.api.app import app, frame_cache
from src.services.frame_cache import FrameData


def test_healthz():
    """Healthz endpoint still works after rewrite."""
    with TestClient(app) as c:
        r = c.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_readyz_reports_devices():
    """Readyz endpoint returns device list and connection count."""
    with TestClient(app) as c:
        r = c.get("/readyz")
        assert r.status_code in {200, 503}
        payload = r.json()
        body = payload if r.status_code == 200 else payload["detail"]
        assert "devices" in body
        assert "connections" in body


def test_websocket_connect_receives_device_list():
    """Client receives device_list message on connect."""
    with TestClient(app) as c:
        with c.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert data["type"] == "device_list"
            assert isinstance(data["devices"], list)


def test_websocket_subscribe_unknown_device():
    """Subscribing to unknown device returns error."""
    with TestClient(app) as c:
        with c.websocket_connect("/ws") as ws:
            # Drain initial device_list
            ws.receive_json()
            ws.send_json({"type": "subscribe_device", "device_id": "nonexistent"})
            data = ws.receive_json()
            assert data["type"] == "error"


def test_websocket_subscribe_devices_receives_only_requested_frames():
    frame_cache.update(
        FrameData(
            device_id="cam-1",
            frame_number=1,
            tracked_persons=[],
            created_at=1,
            image_base64="a",
        )
    )
    frame_cache.update(
        FrameData(
            device_id="cam-2",
            frame_number=2,
            tracked_persons=[],
            created_at=2,
            image_base64="b",
        )
    )

    with TestClient(app) as c:
        with c.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "subscribe_devices", "device_ids": ["cam-2"]})
            data = ws.receive_json()
            assert data["type"] == "frame_update"
            assert data["device_id"] == "cam-2"
