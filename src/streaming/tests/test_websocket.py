import json

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


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
        assert r.status_code == 200
        body = r.json()
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
