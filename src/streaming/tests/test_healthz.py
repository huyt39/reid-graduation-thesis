from src.api import app as app_module
from fastapi.testclient import TestClient
from src.api.app import app


def test_healthz():
    c = TestClient(app)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


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
