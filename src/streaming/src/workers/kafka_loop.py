from __future__ import annotations

import asyncio
import base64
import time

import cv2
import numpy as np
import structlog

from src.kafka.consumer import StreamingKafkaConsumer
from src.services.broadcaster import WebSocketBroadcaster
from src.services.frame_cache import FrameCache, FrameData
from src.services.minio_urls import MinIOURLBuilder

logger = structlog.get_logger()


async def run_kafka_loop(
    consumer: StreamingKafkaConsumer,
    frame_cache: FrameCache,
    broadcaster: WebSocketBroadcaster,
    minio_urls: MinIOURLBuilder | None = None,
    *,
    max_poll_records: int = 50,
    jpeg_quality: int = 75,
    broadcast_max_fps: float = 12.0,
    source: str = "processed",
) -> None:
    """Poll Kafka, decode frames, update cache, and broadcast to WebSocket clients.

    Back-pressure strategy: the cache always receives the latest frame. Broadcast
    is fired as a background task. If a previous broadcast is still running when the
    next frame arrives, the old task is cancelled so clients always see the newest
    frame rather than accumulating a queue of stale ones.
    """
    logger.info("kafka_loop.started")
    _broadcast_task: asyncio.Task | None = None
    min_broadcast_interval = 0.0 if broadcast_max_fps <= 0 else 1.0 / broadcast_max_fps
    last_broadcast_at_by_device: dict[str, float] = {}

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
                frame = _decode_frame(msg, jpeg_quality, minio_urls=minio_urls, source=source)
                if frame is None:
                    continue

                # Cache always gets the freshest frame regardless of broadcast state
                frame_cache.update(frame)

                now = time.monotonic()
                last_broadcast_at = last_broadcast_at_by_device.get(frame.device_id, 0.0)
                if min_broadcast_interval > 0 and (now - last_broadcast_at) < min_broadcast_interval:
                    continue
                last_broadcast_at_by_device[frame.device_id] = now

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


def _with_snapshot_urls(
    tracked_persons: list[dict], minio_urls: MinIOURLBuilder | None,
) -> list[dict]:
    if minio_urls is None:
        return tracked_persons

    enriched: list[dict] = []
    for person in tracked_persons:
        item = dict(person)
        item["snapshot_url"] = minio_urls.presigned_url(item.get("snapshot_key"))
        enriched.append(item)
    return enriched


def _decode_frame(
    msg: dict, jpeg_quality: int, *, minio_urls: MinIOURLBuilder | None, source: str,
) -> FrameData | None:
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

        tracked_persons = msg.get("tracked_persons")
        if tracked_persons is None:
            tracked_persons = [
                {
                    "person_id": None,
                    "bbox": det["bbox"],
                    "confidence": det.get("confidence", 0.0),
                    "gender": "raw",
                    "gender_confidence": 0.0,
                    "tracklet_id": None,
                    "tracklet_state": "raw_edge",
                    "snapshot_key": None,
                    "visibility_score": det.get("visibility_score", 0.0),
                    "live_visibility_score": det.get("visibility_score", 0.0),
                    "overlap_ratio": det.get("overlap_ratio", 0.0),
                    "quality": None,
                    "matching": None,
                    "attributes": {
                        "source": "raw_edge",
                        "debug_label": f"raw conf={float(det.get('confidence', 0.0)):.2f}",
                        "class_id": str(det.get("class_id", 0)),
                        "overlap_ratio": f"{float(det.get('overlap_ratio', 0.0)):.4f}",
                    },
                    "status": "recovering",
                }
                for det in msg.get("detections", [])
            ]
        else:
            tracked_persons = _with_snapshot_urls(tracked_persons, minio_urls)

        return FrameData(
            device_id=msg["device_id"],
            frame_number=msg["frame_number"],
            tracked_persons=tracked_persons,
            created_at=msg["created_at"],
            image_base64=image_base64,
            schema_version=int(msg.get("schema_version", 2)),
            source=source,
        )
    except Exception:
        logger.error("decode_frame.error", exc_info=True)
        return None
