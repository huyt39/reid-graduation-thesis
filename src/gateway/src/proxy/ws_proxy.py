from __future__ import annotations

import asyncio

import structlog
import websockets
from fastapi import WebSocket

logger = structlog.get_logger()


async def proxy_websocket(client_ws: WebSocket, upstream_url: str) -> None:
    """Bidirectional WebSocket bridge between a FastAPI client and an upstream service."""
    await client_ws.accept()

    try:
        async with websockets.connect(
            upstream_url,
            ping_interval=20,
            ping_timeout=10,
            max_size=10 * 1024 * 1024,
        ) as upstream_ws:
            client_to_upstream = asyncio.create_task(
                _forward(client_ws, upstream_ws, direction="client->upstream"),
            )
            upstream_to_client = asyncio.create_task(
                _forward_reverse(upstream_ws, client_ws, direction="upstream->client"),
            )

            # Wait for either direction to finish (disconnect)
            done, pending = await asyncio.wait(
                [client_to_upstream, upstream_to_client],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
    except websockets.exceptions.InvalidURI:
        logger.error("ws_proxy.invalid_upstream_url", url=upstream_url)
        await client_ws.close(code=1011, reason="Bad upstream URL")
    except Exception:
        logger.error("ws_proxy.error", exc_info=True)
        try:
            await client_ws.close(code=1011)
        except Exception:
            pass


async def _forward(client_ws: WebSocket, upstream_ws, *, direction: str) -> None:
    """Client -> Upstream: read from FastAPI WebSocket, send to websockets client."""
    try:
        while True:
            data = await client_ws.receive_text()
            await upstream_ws.send(data)
    except Exception:
        logger.debug("ws_proxy.closed", direction=direction)


async def _forward_reverse(upstream_ws, client_ws: WebSocket, *, direction: str) -> None:
    """Upstream -> Client: read from websockets client, send to FastAPI WebSocket."""
    try:
        async for message in upstream_ws:
            await client_ws.send_text(message)
    except Exception:
        logger.debug("ws_proxy.closed", direction=direction)
