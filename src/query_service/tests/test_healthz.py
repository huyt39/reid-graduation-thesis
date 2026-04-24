from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from src.api import deps
from src.api import app as app_module
from src.api.app import app


def test_healthz():
    c = TestClient(app)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_healthz_preserves_existing_request_id_header():
    c = TestClient(app)
    r = c.get("/healthz", headers={"X-Request-ID": "req-qs-123"})

    assert r.status_code == 200
    assert r.headers["x-request-id"] == "req-qs-123"


def test_healthz_generates_request_id_header_when_missing():
    c = TestClient(app)
    r = c.get("/healthz")

    assert r.status_code == 200
    assert "x-request-id" in r.headers
    assert r.headers["x-request-id"]


def test_healthz_logs_request_context(monkeypatch):
    events: list[tuple[str, dict]] = []

    class DummyLogger:
        def info(self, event, **kwargs):
            events.append((event, kwargs))

    monkeypatch.setattr(app_module, "log", DummyLogger())

    c = TestClient(app)
    r = c.get("/healthz", headers={"X-Request-ID": "req-qs-log-1"})

    assert r.status_code == 200
    assert len(events) == 1
    event, payload = events[0]
    assert event == "query_service.http_response"
    assert payload["request_id"] == "req-qs-log-1"
    assert payload["method"] == "GET"
    assert payload["path"] == "/healthz"
    assert payload["status_code"] == 200
    assert isinstance(payload["duration_ms"], float)


def test_readyz_returns_ready_when_all_dependencies_pass(monkeypatch):
    mongo = AsyncMock()
    mongo.ping.return_value = True
    qdrant = MagicMock()
    qdrant.ping.return_value = True
    redis_cache = AsyncMock()
    redis_cache.ping.return_value = True
    minio_urls = MagicMock()
    minio_urls.ping.return_value = True

    monkeypatch.setattr(deps, "get_mongo", lambda: mongo)
    monkeypatch.setattr(deps, "get_qdrant", lambda: qdrant)
    monkeypatch.setattr(deps, "get_redis", lambda: redis_cache)
    monkeypatch.setattr(deps, "get_minio_urls", lambda: minio_urls)

    c = TestClient(app)
    r = c.get("/readyz")

    assert r.status_code == 200
    assert r.json()["status"] == "ready"
    assert r.json()["checks"] == {
        "mongo": True,
        "qdrant": True,
        "redis": True,
        "minio": True,
    }


def test_readyz_returns_503_when_any_dependency_fails(monkeypatch):
    mongo = AsyncMock()
    mongo.ping.return_value = True
    qdrant = MagicMock()
    qdrant.ping.return_value = False
    redis_cache = AsyncMock()
    redis_cache.ping.return_value = True
    minio_urls = MagicMock()
    minio_urls.ping.return_value = True

    monkeypatch.setattr(deps, "get_mongo", lambda: mongo)
    monkeypatch.setattr(deps, "get_qdrant", lambda: qdrant)
    monkeypatch.setattr(deps, "get_redis", lambda: redis_cache)
    monkeypatch.setattr(deps, "get_minio_urls", lambda: minio_urls)

    c = TestClient(app)
    r = c.get("/readyz")

    assert r.status_code == 503
    assert r.json()["detail"]["status"] == "not_ready"
    assert r.json()["detail"]["checks"] == {
        "mongo": True,
        "qdrant": False,
        "redis": True,
        "minio": True,
    }
