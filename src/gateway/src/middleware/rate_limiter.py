from __future__ import annotations

import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """In-memory token-bucket rate limiter keyed by client IP."""

    def __init__(self, app, rpm: int = 120) -> None:
        super().__init__(app)
        self.rpm = rpm
        self.refill_rate = rpm / 60.0  # tokens per second
        self._buckets: dict[str, tuple[float, float]] = {}  # ip -> (tokens, last_refill)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ("/healthz", "/readyz"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        tokens, last_refill = self._buckets.get(client_ip, (float(self.rpm), now))

        # Refill tokens based on elapsed time
        elapsed = now - last_refill
        tokens = min(self.rpm, tokens + elapsed * self.refill_rate)
        last_refill = now

        if tokens < 1.0:
            self._buckets[client_ip] = (tokens, last_refill)
            return Response(
                content='{"detail":"Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(int(1.0 / self.refill_rate))},
            )

        tokens -= 1.0
        self._buckets[client_ip] = (tokens, last_refill)
        return await call_next(request)
