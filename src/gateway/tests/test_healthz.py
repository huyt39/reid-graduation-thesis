from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from src.api import app as app_module
from src.api.app import app


def test_healthz():
    c = TestClient(app)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readyz_returns_ready_when_upstreams_are_ready():
    class DummyResponse:
        def __init__(self, status_code: int):
            self.status_code = status_code

    client = AsyncMock()
    client.get = AsyncMock(side_effect=[
        DummyResponse(200),
        DummyResponse(200),
    ])

    old_client = app_module._http_client
    app_module._http_client = client
    try:
        c = TestClient(app)
        r = c.get("/readyz")
    finally:
        app_module._http_client = old_client

    assert r.status_code == 200
    assert r.json()["status"] == "ready"
    assert r.json()["checks"] == {
        "query_service": True,
        "streaming": True,
    }


def test_readyz_returns_503_when_any_upstream_is_not_ready():
    class DummyResponse:
        def __init__(self, status_code: int):
            self.status_code = status_code

    client = AsyncMock()
    client.get = AsyncMock(side_effect=[
        DummyResponse(200),
        DummyResponse(503),
    ])

    old_client = app_module._http_client
    app_module._http_client = client
    try:
        c = TestClient(app)
        r = c.get("/readyz")
    finally:
        app_module._http_client = old_client

    assert r.status_code == 503
    assert r.json()["detail"]["status"] == "not_ready"
    assert r.json()["detail"]["checks"] == {
        "query_service": True,
        "streaming": False,
    }
