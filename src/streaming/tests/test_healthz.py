from src.api import app as app_module
from fastapi.testclient import TestClient
from src.api.app import app


def test_healthz():
    c = TestClient(app)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_healthz_preserves_existing_request_id_header():
    c = TestClient(app)
    r = c.get("/healthz", headers={"X-Request-ID": "req-stream-123"})

    assert r.status_code == 200
    assert r.headers["x-request-id"] == "req-stream-123"


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

        def warning(self, event, **kwargs):
            events.append((event, kwargs))

        def error(self, event, **kwargs):
            events.append((event, kwargs))

    monkeypatch.setattr(app_module, "logger", DummyLogger())

    c = TestClient(app)
    r = c.get("/healthz", headers={"X-Request-ID": "req-stream-log-1"})

    assert r.status_code == 200
    assert len(events) == 1
    event, payload = events[0]
    assert event == "streaming.http_response"
    assert payload["request_id"] == "req-stream-log-1"
    assert payload["method"] == "GET"
    assert payload["path"] == "/healthz"
    assert payload["status_code"] == 200
    assert isinstance(payload["duration_ms"], float)


def test_readyz_returns_ready_when_kafka_loop_is_running():
    old_state = dict(app_module.streaming_state)
    try:
        app_module.streaming_state["kafka_loop_running"] = True
        app_module.streaming_state["kafka_loop_failed"] = False

        c = TestClient(app)
        r = c.get("/readyz")
    finally:
        app_module.streaming_state.update(old_state)

    assert r.status_code == 200
    assert r.json()["status"] == "ready"
    assert r.json()["checks"] == {
        "kafka_loop_running": True,
        "kafka_loop_failed": False,
    }


def test_readyz_returns_503_when_kafka_loop_failed():
    old_state = dict(app_module.streaming_state)
    try:
        app_module.streaming_state["kafka_loop_running"] = False
        app_module.streaming_state["kafka_loop_failed"] = True

        c = TestClient(app)
        r = c.get("/readyz")
    finally:
        app_module.streaming_state.update(old_state)

    assert r.status_code == 503
    assert r.json()["detail"]["status"] == "not_ready"
    assert r.json()["detail"]["checks"] == {
        "kafka_loop_running": False,
        "kafka_loop_failed": True,
    }
