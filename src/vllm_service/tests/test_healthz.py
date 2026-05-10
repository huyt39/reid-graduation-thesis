from fastapi.testclient import TestClient

from src.api.app import app


def test_healthz():
    c = TestClient(app)
    r = c.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "vllm_service"


def test_healthz_preserves_request_id_header():
    c = TestClient(app)
    r = c.get("/healthz", headers={"X-Request-ID": "req-vllm-1"})
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "req-vllm-1"


def test_readyz_returns_ok_when_llm_not_required():
    """Default config sets require_llm_for_ready=False, so readiness should pass
    regardless of whether the upstream LLM is reachable."""
    c = TestClient(app)
    with c:  # triggers lifespan so _llm_client is initialized
        r = c.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert "llm_reachable" in body["checks"]
