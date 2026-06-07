from unittest.mock import AsyncMock

import httpx
from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api.app import app
from src.proxy import http_proxy


def test_api_proxy_returns_504_when_query_service_times_out():
    client = AsyncMock()
    client.request = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))

    old_client = app_module._http_client
    app_module._http_client = client
    try:
        c = TestClient(app)
        r = c.get("/api/v1/persons")
    finally:
        app_module._http_client = old_client

    assert r.status_code == 504
    assert r.json()["detail"] == {
        "error": "upstream_timeout",
        "target": "http://query_service:8090",
        "path": "persons",
    }


def test_api_proxy_returns_502_when_query_service_is_unavailable():
    client = AsyncMock()
    client.request = AsyncMock(side_effect=httpx.ConnectError("boom"))

    old_client = app_module._http_client
    app_module._http_client = client
    try:
        c = TestClient(app)
        r = c.get("/api/v1/persons")
    finally:
        app_module._http_client = old_client

    assert r.status_code == 502
    assert r.json()["detail"] == {
        "error": "upstream_unavailable",
        "target": "http://query_service:8090",
        "path": "persons",
    }


def test_api_proxy_preserves_existing_request_id_header():
    upstream = httpx.Response(
        status_code=200,
        content=b'{"ok": true}',
        headers={"content-type": "application/json"},
    )
    client = AsyncMock()
    client.request = AsyncMock(return_value=upstream)

    old_client = app_module._http_client
    app_module._http_client = client
    try:
        c = TestClient(app)
        r = c.get("/api/v1/persons", headers={"X-Request-ID": "req-123"})
    finally:
        app_module._http_client = old_client

    assert r.status_code == 200
    forwarded_headers = client.request.await_args.kwargs["headers"]
    assert forwarded_headers["x-request-id"] == "req-123"


def test_api_proxy_generates_request_id_when_missing():
    upstream = httpx.Response(
        status_code=200,
        content=b'{"ok": true}',
        headers={"content-type": "application/json"},
    )
    client = AsyncMock()
    client.request = AsyncMock(return_value=upstream)

    old_client = app_module._http_client
    app_module._http_client = client
    try:
        c = TestClient(app)
        r = c.get("/api/v1/persons")
    finally:
        app_module._http_client = old_client

    assert r.status_code == 200
    forwarded_headers = client.request.await_args.kwargs["headers"]
    assert "x-request-id" in forwarded_headers
    assert forwarded_headers["x-request-id"]


def test_api_proxy_returns_request_id_header_to_client():
    upstream = httpx.Response(
        status_code=200,
        content=b'{"ok": true}',
        headers={"content-type": "application/json"},
    )
    client = AsyncMock()
    client.request = AsyncMock(return_value=upstream)

    old_client = app_module._http_client
    app_module._http_client = client
    try:
        c = TestClient(app)
        r = c.get("/api/v1/persons", headers={"X-Request-ID": "req-456"})
    finally:
        app_module._http_client = old_client

    assert r.status_code == 200
    assert r.headers["x-request-id"] == "req-456"


def test_api_proxy_preserves_upstream_response_request_id_header():
    upstream = httpx.Response(
        status_code=200,
        content=b'{"ok": true}',
        headers={
            "content-type": "application/json",
            "x-request-id": "upstream-req-999",
        },
    )
    client = AsyncMock()
    client.request = AsyncMock(return_value=upstream)

    old_client = app_module._http_client
    app_module._http_client = client
    try:
        c = TestClient(app)
        r = c.get("/api/v1/persons", headers={"X-Request-ID": "client-req-123"})
    finally:
        app_module._http_client = old_client

    assert r.status_code == 200
    assert r.headers["x-request-id"] == "upstream-req-999"


def test_api_proxy_logs_success_with_request_context(monkeypatch):
    upstream = httpx.Response(
        status_code=200,
        content=b'{"ok": true}',
        headers={"content-type": "application/json"},
    )
    client = AsyncMock()
    client.request = AsyncMock(return_value=upstream)

    events: list[tuple[str, dict]] = []

    class DummyLogger:
        def info(self, event, **kwargs):
            events.append((event, kwargs))

        def warning(self, event, **kwargs):
            events.append((event, kwargs))

    monkeypatch.setattr(http_proxy, "logger", DummyLogger())

    old_client = app_module._http_client
    app_module._http_client = client
    try:
        c = TestClient(app)
        r = c.get("/api/v1/persons", headers={"X-Request-ID": "req-log-1"})
    finally:
        app_module._http_client = old_client

    assert r.status_code == 200
    assert len(events) == 1
    event, payload = events[0]
    assert event == "gateway.proxy_response"
    assert payload["request_id"] == "req-log-1"
    assert payload["method"] == "GET"
    assert payload["target"] == "http://query_service:8090"
    assert payload["path"] == "persons"
    assert payload["status_code"] == 200
    assert isinstance(payload["duration_ms"], float)


def test_api_proxy_logs_timeout_with_request_context(monkeypatch):
    client = AsyncMock()
    client.request = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))

    events: list[tuple[str, dict]] = []

    class DummyLogger:
        def info(self, event, **kwargs):
            events.append((event, kwargs))

        def warning(self, event, **kwargs):
            events.append((event, kwargs))

    monkeypatch.setattr(http_proxy, "logger", DummyLogger())

    old_client = app_module._http_client
    app_module._http_client = client
    try:
        c = TestClient(app)
        r = c.get("/api/v1/persons", headers={"X-Request-ID": "req-timeout-1"})
    finally:
        app_module._http_client = old_client

    assert r.status_code == 504
    assert len(events) == 1
    event, payload = events[0]
    assert event == "gateway.proxy_timeout"
    assert payload["request_id"] == "req-timeout-1"
    assert payload["method"] == "GET"
    assert payload["target"] == "http://query_service:8090"
    assert payload["path"] == "persons"
    assert isinstance(payload["duration_ms"], float)


def test_api_proxy_logs_http_error_with_request_context(monkeypatch):
    client = AsyncMock()
    client.request = AsyncMock(side_effect=httpx.ConnectError("boom"))

    events: list[tuple[str, dict]] = []

    class DummyLogger:
        def info(self, event, **kwargs):
            events.append((event, kwargs))

        def warning(self, event, **kwargs):
            events.append((event, kwargs))

    monkeypatch.setattr(http_proxy, "logger", DummyLogger())

    old_client = app_module._http_client
    app_module._http_client = client
    try:
        c = TestClient(app)
        r = c.get("/api/v1/persons", headers={"X-Request-ID": "req-error-1"})
    finally:
        app_module._http_client = old_client

    assert r.status_code == 502
    assert len(events) == 1
    event, payload = events[0]
    assert event == "gateway.proxy_error"
    assert payload["request_id"] == "req-error-1"
    assert payload["method"] == "GET"
    assert payload["target"] == "http://query_service:8090"
    assert payload["path"] == "persons"
    assert isinstance(payload["duration_ms"], float)
