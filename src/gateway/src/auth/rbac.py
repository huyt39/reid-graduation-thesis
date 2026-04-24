from __future__ import annotations

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from src.auth.jwt_handler import decode_token
from src.auth.models import Role, TokenPayload

_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> TokenPayload:
    """FastAPI dependency that extracts and validates the JWT from the Authorization header."""
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing token")
    try:
        return decode_token(credentials.credentials)
    except JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")


async def get_ws_user(token: str = Query(...)) -> TokenPayload:
    """Validate JWT passed as a query parameter for WebSocket connections."""
    try:
        return decode_token(token)
    except JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")


def require_role(*allowed: Role):
    """Return a dependency that checks the user's role against the allowed set."""

    async def _check(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        if user.role not in allowed:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Insufficient permissions")
        return user

    return _check
