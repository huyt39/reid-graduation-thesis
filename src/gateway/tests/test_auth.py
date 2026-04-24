import time

from jose import jwt

from src.auth.jwt_handler import create_token, decode_token
from src.auth.models import Role
from src.core.config import settings


def test_create_and_decode_token():
    token, expires_in = create_token("alice", Role.ADMIN)
    assert isinstance(token, str)
    assert expires_in == settings.jwt_expiry_minutes * 60

    payload = decode_token(token)
    assert payload.sub == "alice"
    assert payload.role == Role.ADMIN


def test_decode_bad_token_raises():
    import pytest
    from jose import JWTError

    with pytest.raises(JWTError):
        decode_token("not.a.valid.token")


def test_expired_token_raises():
    import pytest
    from jose import JWTError

    payload = {"sub": "bob", "role": "viewer", "exp": int(time.time()) - 10}
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

    with pytest.raises(JWTError):
        decode_token(token)


def test_roles_roundtrip():
    for role in Role:
        token, _ = create_token("user", role)
        p = decode_token(token)
        assert p.role == role
