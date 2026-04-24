from __future__ import annotations

import asyncio
import base64

import cv2
import numpy as np
import structlog

from src.kafka.consumer import StreamingKafkaConsumer
from src.services.broadcaster import WebSocketBroadcaster
from src.services.frame_cache import FrameCache, FrameData

logger = structlog.get_logger()


async def run_kafka_loop(
    consumer: StreamingKafkaConsumer,
    frame_cache: FrameCache,
    broadcaster: WebSocketBroadcaster,
    *,
    max_poll_records: int = 50,
    jpeg_quality: int = 75,
) -> None:
    """Poll Kafka, decode frames, update cache, and broadcast to WebSocket clients.

    Back-pressure strategy: the cache always receives the latest frame. Broadcast
    is fired as a background task. If a previous broadcast is still running when the
    next frame arrives, the old task is cancelled so clients always see the newest
    frame rather than accumulating a queue of stale ones.
    """
    logger.info("kafka_loop.started")
    _broadcast_task: asyncio.Task | None = None

    while True:
        try:
            await asyncio.sleep(0)  # yield to WebSocket handlers

            messages = await asyncio.to_thread(
                consumer.poll, timeout_ms=1000, max_records=max_poll_records,
            )

            if not messages:
                await asyncio.sleep(0.05)
                continue

            for msg in messages:
                frame = _decode_frame(msg, jpeg_quality)
                if frame is None:
                    continue

                # Cache always gets the freshest frame regardless of broadcast state
                frame_cache.update(frame)

                # Cancel pending broadcast — newest frame wins
                if _broadcast_task is not None and not _broadcast_task.done():
                    _broadcast_task.cancel()

                _broadcast_task = asyncio.create_task(
                    broadcaster.broadcast(frame), name="ws-broadcast",
                )

        except asyncio.CancelledError:
            logger.info("kafka_loop.cancelled")
            if _broadcast_task is not None:
                _broadcast_task.cancel()
            raise
        except Exception:
            logger.error("kafka_loop.error", exc_info=True)
            await asyncio.sleep(1)


def _decode_frame(msg: dict, jpeg_quality: int) -> FrameData | None:
    try:
        image_bytes: bytes = msg["image_data"]
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("decode_frame.invalid_image", device_id=msg.get("device_id"))
            return None

        _, buf = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )
        image_base64 = base64.b64encode(buf.tobytes()).decode("ascii")

        return FrameData(
            device_id=msg["device_id"],
            frame_number=msg["frame_number"],
            tracked_persons=msg["tracked_persons"],
            created_at=msg["created_at"],
            image_base64=image_base64,
            schema_version=int(msg.get("schema_version", 2)),
        )
    except Exception:
        logger.error("decode_frame.error", exc_info=True)
        return None
