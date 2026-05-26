from __future__ import annotations

import asyncio
import base64
import time

import structlog

from src.kafka.consumer import StreamingKafkaConsumer
from src.services.broadcaster import WebSocketBroadcaster
from src.services.frame_cache import FrameCache, FrameData
from src.services.minio_urls import MinIOURLBuilder

logger = structlog.get_logger()


def _jpeg_dimensions(image_bytes: bytes) -> tuple[int | None, int | None]:
    if len(image_bytes) < 4 or image_bytes[:2] != b"\xff\xd8":
        return None, None
    idx = 2
    size = len(image_bytes)
    while idx + 9 < size:
        if image_bytes[idx] != 0xFF:
            idx += 1
            continue
        while idx < size and image_bytes[idx] == 0xFF:
            idx += 1
        if idx >= size:
            break
        marker = image_bytes[idx]
        idx += 1
        if marker in {0xD8, 0xD9}:
            continue
        if idx + 2 > size:
            break
        segment_len = int.from_bytes(image_bytes[idx:idx + 2], "big")
        if segment_len < 2 or idx + segment_len > size:
            break
        if marker in {
            0xC0, 0xC1, 0xC2, 0xC3,
            0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB,
            0xCD, 0xCE, 0xCF,
        }:
            if segment_len >= 7:
                height = int.from_bytes(image_bytes[idx + 3:idx + 5], "big")
                width = int.from_bytes(image_bytes[idx + 5:idx + 7], "big")
                return width, height
            break
        idx += segment_len
    return None, None


async def run_kafka_loop(
    consumer: StreamingKafkaConsumer,
    frame_cache: FrameCache,
    broadcaster: WebSocketBroadcaster,
    minio_urls: MinIOURLBuilder | None = None,
    *,
    max_poll_records: int = 50,
    broadcast_max_fps: float = 30.0,
    source: str = "processed",
) -> None:
    """Poll Kafka, decode frames, update cache, and broadcast to WebSocket clients.

    Back-pressure strategy: the cache always receives the latest frame. Broadcast
    is fired as a background task. If a previous broadcast is still running when the
    next frame arrives, the old task is cancelled so clients always see the newest
    frame rather than accumulating a queue of stale ones.
    """
    logger.info("kafka_loop.started", source=source)
    _broadcast_task: asyncio.Task | None = None
    min_broadcast_interval = 0.0 if broadcast_max_fps <= 0 else 1.0 / broadcast_max_fps
    last_broadcast_at_by_device: dict[str, float] = {}

    # Lightweight per-device counters for FPS observability. Reset each window.
    summary_window_s = 5.0
    consumed_by_device: dict[str, int] = {}
    broadcast_by_device: dict[str, int] = {}
    next_summary_at = time.monotonic() + summary_window_s

    while True:
        try:
            await asyncio.sleep(0)  # yield to WebSocket handlers

            messages = await asyncio.to_thread(
                consumer.poll, timeout_ms=1000, max_records=max_poll_records,
            )

            now_wall = time.monotonic()
            if now_wall >= next_summary_at:
                window = max(now_wall - (next_summary_at - summary_window_s), 1e-6)
                if consumed_by_device or broadcast_by_device:
                    logger.info(
                        "streaming.fps_summary",
                        source=source,
                        window_s=round(window, 2),
                        consumed_fps={
                            d: round(n / window, 2) for d, n in consumed_by_device.items()
                        },
                        broadcast_fps={
                            d: round(n / window, 2) for d, n in broadcast_by_device.items()
                        },
                    )
                consumed_by_device.clear()
                broadcast_by_device.clear()
                next_summary_at = now_wall + summary_window_s

            if not messages:
                await asyncio.sleep(0.05)
                continue

            for msg in messages:
                frame = _decode_frame(msg, minio_urls=minio_urls, source=source)
                if frame is None:
                    continue

                consumed_by_device[frame.device_id] = (
                    consumed_by_device.get(frame.device_id, 0) + 1
                )

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
                broadcast_by_device[frame.device_id] = (
                    broadcast_by_device.get(frame.device_id, 0) + 1
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
    msg: dict, *, minio_urls: MinIOURLBuilder | None, source: str,
) -> FrameData | None:
    try:
        image_bytes: bytes = msg["image_data"]
        if not image_bytes:
            logger.warning("decode_frame.empty_image", device_id=msg.get("device_id"))
            return None
        # Pass-through: edge already encoded JPEG. Re-decoding + re-encoding
        # here would waste CPU and add no quality (edge controls quality).
        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        image_width, image_height = _jpeg_dimensions(image_bytes)

        tracked_persons = msg.get("tracked_persons")
        if tracked_persons is None:
            tracked_persons = [
                {
                    "person_id": None,
                    "bbox": det["bbox"],
                    "confidence": det.get("confidence", 0.0),
                    "gender": "raw",
                    "gender_confidence": 0.0,
                    "track_id": None,
                    "live_track_key": None,
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
            image_width=image_width,
            image_height=image_height,
            schema_version=int(msg.get("schema_version", 2)),
            source=source,
        )
    except Exception:
        logger.error("decode_frame.error", exc_info=True)
        return None
