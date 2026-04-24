from fastapi.testclient import TestClient

from src.api.app import app
from src.auth.jwt_handler import create_token
from src.auth.models import Role


def _auth_header(role: Role = Role.ADMIN) -> dict:
    token, _ = create_token("testuser", role)
    return {"Authorization": f"Bearer {token}"}


def test_login_success():
    c = TestClient(app)
    r = c.post("/auth/login", json={"username": "admin", "password": "admin"})
    assert r.status_code == 200
    body = r.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"


def test_login_bad_credentials():
    c = TestClient(app)
    r = c.post("/auth/login", json={"username": "admin", "password": "wrong"})
    assert r.status_code == 401


def test_refresh_with_valid_token():
    c = TestClient(app)
    r = c.post("/auth/refresh", headers=_auth_header())
    assert r.status_code == 200
    assert "access_token" in r.json()


def test_refresh_without_token():
    c = TestClient(app)
    r = c.post("/auth/refresh")
    assert r.status_code == 401


def test_api_proxy_without_token_is_401():
    c = TestClient(app)
    r = c.get("/api/v1/persons")
    assert r.status_code == 401


def test_api_proxy_with_token_attempts_proxy():
    """With a valid token the gateway tries to proxy.
    Since no upstream is running, we expect a connection error (502-ish)
    or a successful proxy depending on env. We just verify auth passes."""
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/api/v1/persons", headers=_auth_header())
    # Auth passed — any status other than 401/403 means auth worked
    assert r.status_code not in (401, 403)
