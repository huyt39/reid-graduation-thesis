from __future__ import annotations
import time
import structlog
from uuid import uuid4
import httpx
from fastapi import Request, Response, HTTPException

# Hop-by-hop headers that must not be forwarded
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)

logger = structlog.get_logger()


async def proxy_request(
    request: Request,
    target_base: str,
    path: str,
    client: httpx.AsyncClient,
) -> Response:
    """Forward an incoming HTTP request to an upstream service."""
    url = f"{target_base}/{path}"

    # Forward headers, stripping hop-by-hop and host
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() != "host"
    }
    request_id = request.headers.get("x-request-id", str(uuid4()))
    headers.setdefault("x-request-id", request_id)

    started_at = time.perf_counter()

    try:
        upstream = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            params=dict(request.query_params),
            content=await request.body(),
            timeout=30.0,
        )
    except httpx.TimeoutException:
        logger.warning(
            "gateway.proxy_timeout",
            request_id=request_id,
            method=request.method,
            target=target_base,
            path=path,
            duration_ms=round((time.perf_counter() - started_at)*1000,2),
        )
        raise HTTPException(
            status_code=504,
            detail={
                "error": "upstream_timeout",
                "target": target_base,
                "path": path,
            },
        )
    except httpx.HTTPError:
        logger.warning(
            "gateway.proxy_error",
            request_id=request_id,
            method=request.method,
            target=target_base,
            path=path,
            duration_ms=round((time.perf_counter() - started_at)*1000,2),
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error": "upstream_unavailable",
                "target": target_base,
                "path": path,
            },
        )

    # Forward response, stripping hop-by-hop
    resp_headers = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }
    resp_headers.setdefault("x-request-id", request_id)

    # request successfully
    logger.info(
        "gateway.proxy_response",
        request_id=request_id,
        method=request.method,
        target=target_base,
        path=path,
        status_code=upstream.status_code,
        duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
    )


    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("content-type"),
    )
