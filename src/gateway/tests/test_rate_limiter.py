from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.middleware.rate_limiter import RateLimiterMiddleware

# Tiny app with a very low rate limit for testing
_app = FastAPI()
_app.add_middleware(RateLimiterMiddleware, rpm=3)


@_app.get("/ping")
def ping():
    return {"ok": True}


@_app.get("/healthz")
def healthz():
    return {"ok": True}


def test_requests_within_limit_succeed():
    c = TestClient(_app)
    for _ in range(3):
        r = c.get("/ping")
        assert r.status_code == 200


def test_exceeding_limit_returns_429():
    c = TestClient(_app)
    # Exhaust the bucket
    for _ in range(3):
        c.get("/ping")
    r = c.get("/ping")
    assert r.status_code == 429
    assert "Retry-After" in r.headers


def test_healthz_is_exempt_from_rate_limit():
    c = TestClient(_app)
    # Exhaust the bucket with /ping
    for _ in range(3):
        c.get("/ping")
    # healthz should still work
    r = c.get("/healthz")
    assert r.status_code == 200
