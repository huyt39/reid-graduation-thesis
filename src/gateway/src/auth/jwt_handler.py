from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from src.auth.models import Role, TokenPayload
from src.core.config import settings


def create_token(sub: str, role: Role) -> tuple[str, int]:
    """Return (encoded_jwt, expires_in_seconds)."""
    expires_delta = timedelta(minutes=settings.jwt_expiry_minutes)
    expire = datetime.now(timezone.utc) + expires_delta
    payload = {"sub": sub, "role": role.value, "exp": expire}
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, int(expires_delta.total_seconds())


def decode_token(token: str) -> TokenPayload:
    """Decode and validate a JWT. Raises JWTError on failure."""
    data = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    return TokenPayload(sub=data["sub"], role=Role(data["role"]), exp=data["exp"])
