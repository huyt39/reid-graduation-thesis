from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from src.api import deps
from src.api.app import app


def test_healthz():
    c = TestClient(app)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


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
