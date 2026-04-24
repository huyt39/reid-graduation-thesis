from __future__ import annotations

import httpx
from fastapi import Request, Response

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

    upstream = await client.request(
        method=request.method,
        url=url,
        headers=headers,
        params=dict(request.query_params),
        content=await request.body(),
        timeout=30.0,
    )

    # Forward response, stripping hop-by-hop
    resp_headers = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("content-type"),
    )
