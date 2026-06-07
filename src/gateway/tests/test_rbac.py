from fastapi.testclient import TestClient

from src.api.app import app


def test_auth_login_route_removed():
    c = TestClient(app)
    r = c.post("/auth/login", json={"username": "admin", "password": "admin"})
    assert r.status_code == 404


def test_auth_refresh_route_removed():
    c = TestClient(app)
    r = c.post("/auth/refresh")
    assert r.status_code == 404


def test_api_proxy_without_token_attempts_proxy():
    """Proxy routes are now public.
    Since no upstream is running, we expect a connection error (502-ish)
    or a successful proxy depending on env. We just verify there is no auth gate."""
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/api/v1/persons")
    assert r.status_code not in (401, 403)
