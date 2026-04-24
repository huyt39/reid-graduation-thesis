from __future__ import annotations

import asyncio
import json

import structlog
from fastapi import WebSocket
from starlette.websockets import WebSocketState

from src.services.frame_cache import FrameCache, FrameData

logger = structlog.get_logger()


class WebSocketBroadcaster:
    """Manages WebSocket connections and broadcasts frame updates."""

    def __init__(self, frame_cache: FrameCache, semaphore_limit: int = 20) -> None:
        self.frame_cache = frame_cache
        self._connections: list[WebSocket] = []
        self._semaphore = asyncio.Semaphore(semaphore_limit)

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    def add(self, ws: WebSocket) -> None:
        self._connections.append(ws)

    def remove(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)

    async def send_device_list(self, ws: WebSocket) -> None:
        devices = self.frame_cache.device_ids()
        await ws.send_json({"type": "device_list", "devices": devices})

    async def send_latest_frame(self, ws: WebSocket, device_id: str) -> None:
        frame = self.frame_cache.get(device_id)
        if frame is None:
            await ws.send_json(
                {"type": "error", "message": f"Device {device_id} not found"}
            )
            return
        await ws.send_text(self._frame_to_json(frame))

    async def broadcast(self, frame: FrameData) -> None:
        if not self._connections:
            return

        message = self._frame_to_json(frame)
        disconnected: list[WebSocket] = []

        tasks = []
        for ws in self._connections:
            tasks.append(self._safe_send(ws, message, disconnected))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        for ws in disconnected:
            self.remove(ws)

    async def _safe_send(
        self, ws: WebSocket, message: str, disconnected: list[WebSocket],
    ) -> None:
        async with self._semaphore:
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(message)
            except Exception:
                disconnected.append(ws)

    @staticmethod
    def _frame_to_json(frame: FrameData) -> str:
        return json.dumps(
            {
                "type": "frame_update",
                "schema_version": frame.schema_version,
                "device_id": frame.device_id,
                "frame_number": frame.frame_number,
                "tracked_persons": frame.tracked_persons,
                "created_at": frame.created_at,
                "image_base64": frame.image_base64,
            }
        )
