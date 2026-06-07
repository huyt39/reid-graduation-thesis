from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI, Request, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.middleware.rate_limiter import RateLimiterMiddleware
from src.proxy.http_proxy import proxy_request
from src.proxy.ws_proxy import proxy_websocket

logger = structlog.get_logger()

# Shared httpx client for connection pooling
_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient()
    yield
    await _http_client.aclose()


app = FastAPI(title=settings.service_name, lifespan=lifespan)
app.add_middleware(RateLimiterMiddleware, rpm=settings.rate_limit_rpm)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health endpoints

@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": settings.service_name}


@app.get("/readyz")
async def readyz():
    assert _http_client is not None

    checks = {
        "query_service": False,
        "streaming": False,
    }

    try:
        query_resp = await _http_client.get(f"{settings.query_service_url}/readyz", timeout=5.0)
        checks["query_service"] = query_resp.status_code == 200
    except Exception:
        checks["query_service"] = False

    try:
        streaming_http_base = settings.streaming_url.replace("ws://", "http://").replace("wss://", "https://")
        streaming_resp = await _http_client.get(f"{streaming_http_base}/readyz", timeout=5.0)
        checks["streaming"] = streaming_resp.status_code == 200
    except Exception:
        checks["streaming"] = False

    ready = all(checks.values())
    if not ready:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "service": settings.service_name,
                "checks": checks,
            },
        )

    return {
        "status": "ready",
        "service": settings.service_name,
        "checks": checks,
    }


# WebSocket proxy to streaming service

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    upstream = f"{settings.streaming_url}/ws"
    await proxy_websocket(ws, upstream)


# API proxy to query service

@app.api_route(
    "/api/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
)
async def api_proxy(path: str, request: Request):
    return await proxy_request(
        request,
        target_base=settings.query_service_url,
        path=path,
        client=_http_client,
    )
