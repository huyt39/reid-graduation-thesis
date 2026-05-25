from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request

from src.core.config import settings
from src.kafka.consumer import StreamingKafkaConsumer
from src.services.broadcaster import WebSocketBroadcaster
from src.services.frame_cache import FrameCache
from src.services.minio_urls import MinIOURLBuilder
from src.workers.kafka_loop import run_kafka_loop

logger = structlog.get_logger()

# Shared state initialised during lifespan
frame_cache = FrameCache()
raw_frame_cache = FrameCache()
broadcaster = WebSocketBroadcaster(
    frame_cache, semaphore_limit=settings.broadcast_semaphore,
)
raw_broadcaster = WebSocketBroadcaster(
    raw_frame_cache, semaphore_limit=settings.broadcast_semaphore,
)
streaming_state = {
    "kafka_loop_running": False,
    "kafka_loop_failed": False,
    "raw_kafka_loop_running": False,
    "raw_kafka_loop_failed": False,
}


def _build_minio_urls() -> MinIOURLBuilder | None:
    try:
        return MinIOURLBuilder(
            settings.minio_internal_endpoint,
            settings.minio_public_endpoint,
            settings.minio_access_key,
            settings.minio_secret_key,
            secure=settings.minio_secure,
        )
    except Exception:
        logger.warning("streaming.minio_urls_unavailable", exc_info=True)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start Kafka consumer in a background task
    minio_urls = _build_minio_urls()
    consumer = StreamingKafkaConsumer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        topic=settings.output_topic,
        group_id=settings.consumer_group,
        schema_path=settings.schema_path,
    )
    raw_consumer = StreamingKafkaConsumer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        topic=settings.input_topic,
        group_id=settings.raw_consumer_group,
        schema_path=settings.input_schema_path,
    )

    streaming_state["kafka_loop_running"] = True
    streaming_state["kafka_loop_failed"] = False
    streaming_state["raw_kafka_loop_running"] = True
    streaming_state["raw_kafka_loop_failed"] = False

    kafka_task = asyncio.create_task(
        run_kafka_loop(
            consumer,
            frame_cache,
            broadcaster,
            minio_urls,
            max_poll_records=settings.max_poll_records,
            jpeg_quality=settings.jpeg_quality,
            broadcast_max_fps=settings.broadcast_max_fps,
            source="processed",
        ),
        name="kafka-consumer-loop",
    )
    raw_kafka_task = asyncio.create_task(
        run_kafka_loop(
            raw_consumer,
            raw_frame_cache,
            raw_broadcaster,
            None,
            max_poll_records=settings.max_poll_records,
            jpeg_quality=settings.jpeg_quality,
            broadcast_max_fps=settings.broadcast_max_fps,
            source="raw",
        ),
        name="raw-kafka-consumer-loop",
    )
    logger.info("streaming.started")

    def _on_kafka_task_done(task: asyncio.Task) -> None:
        streaming_state["kafka_loop_running"] = False
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            streaming_state["kafka_loop_failed"] = True
            logger.error("streaming.kafka_loop_failed", exc_info=exc)

    kafka_task.add_done_callback(_on_kafka_task_done)

    def _on_raw_kafka_task_done(task: asyncio.Task) -> None:
        streaming_state["raw_kafka_loop_running"] = False
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            streaming_state["raw_kafka_loop_failed"] = True
            logger.error("streaming.raw_kafka_loop_failed", exc_info=exc)

    raw_kafka_task.add_done_callback(_on_raw_kafka_task_done)


    yield

    # Shutdown
    kafka_task.cancel()
    raw_kafka_task.cancel()
    try:
        await kafka_task
    except asyncio.CancelledError:
        pass
    try:
        await raw_kafka_task
    except asyncio.CancelledError:
        pass
    streaming_state["kafka_loop_running"] = False
    streaming_state["raw_kafka_loop_running"] = False
    consumer.close()
    raw_consumer.close()
    logger.info("streaming.stopped")


app = FastAPI(title=settings.service_name, lifespan=lifespan)

@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid4()))
    request.state.request_id = request_id

    started_at = time.perf_counter()
    response = await call_next(request)
    response.headers.setdefault("x-request-id", request_id)

    logger.info(
        "streaming.http_response",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
    )
    return response



# Health endpoints

@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": settings.service_name}


@app.get("/readyz")
def readyz():
    checks = {
        "kafka_loop_running": bool(streaming_state["kafka_loop_running"]),
        "kafka_loop_failed": bool(streaming_state["kafka_loop_failed"]),
        "raw_kafka_loop_running": bool(streaming_state["raw_kafka_loop_running"]),
        "raw_kafka_loop_failed": bool(streaming_state["raw_kafka_loop_failed"]),
    }
    ready = (
        checks["kafka_loop_running"]
        and not checks["kafka_loop_failed"]
        and checks["raw_kafka_loop_running"]
        and not checks["raw_kafka_loop_failed"]
    )

    if not ready:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "service": settings.service_name,
                "checks": checks,
                "connections": broadcaster.connection_count,
                "devices": frame_cache.device_ids(),
                "raw_connections": raw_broadcaster.connection_count,
                "raw_devices": raw_frame_cache.device_ids(),
            },
        )

    return {
        "status": "ready",
        "service": settings.service_name,
        "checks": checks,
        "connections": broadcaster.connection_count,
        "devices": frame_cache.device_ids(),
        "raw_connections": raw_broadcaster.connection_count,
        "raw_devices": raw_frame_cache.device_ids(),
    }


# WebSocket endpoint

async def _serve_websocket(ws: WebSocket, ws_broadcaster: WebSocketBroadcaster):
    if ws_broadcaster.connection_count >= settings.websocket_max_connections:
        await ws.close(code=1013, reason="Too many connections")
        return

    await ws.accept()
    ws_broadcaster.add(ws)
    logger.info("ws.connected", client=ws.client)

    try:
        # Send current device list on connect
        await ws_broadcaster.send_device_list(ws)

        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
                if data.get("type") == "subscribe_device":
                    device_id = data.get("device_id", "")
                    await ws_broadcaster.send_latest_frame(ws, device_id)
            except json.JSONDecodeError:
                logger.warning("ws.invalid_json")
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.error("ws.error", exc_info=True)
    finally:
        ws_broadcaster.remove(ws)
        logger.info("ws.disconnected", client=ws.client)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await _serve_websocket(ws, broadcaster)


@app.websocket("/ws/raw")
async def raw_websocket_endpoint(ws: WebSocket):
    await _serve_websocket(ws, raw_broadcaster)
