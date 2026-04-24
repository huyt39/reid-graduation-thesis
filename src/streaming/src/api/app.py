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
from src.workers.kafka_loop import run_kafka_loop

logger = structlog.get_logger()

# Shared state initialised during lifespan
frame_cache = FrameCache()
broadcaster = WebSocketBroadcaster(
    frame_cache, semaphore_limit=settings.broadcast_semaphore,
)
streaming_state = {
    "kafka_loop_running": False,
    "kafka_loop_failed": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start Kafka consumer in a background task
    consumer = StreamingKafkaConsumer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        topic=settings.output_topic,
        group_id=settings.consumer_group,
        schema_path=settings.schema_path,
    )

    streaming_state["kafka_loop_running"] = True
    streaming_state["kafka_loop_failed"] = False

    kafka_task = asyncio.create_task(
        run_kafka_loop(
            consumer,
            frame_cache,
            broadcaster,
            max_poll_records=settings.max_poll_records,
            jpeg_quality=settings.jpeg_quality,
        ),
        name="kafka-consumer-loop",
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


    yield

    # Shutdown
    kafka_task.cancel()
    try:
        await kafka_task
    except asyncio.CancelledError:
        pass
    streaming_state["kafka_loop_running"] = False
    consumer.close()
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
    }
    ready = checks["kafka_loop_running"] and not checks["kafka_loop_failed"]

    if not ready:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "service": settings.service_name,
                "checks": checks,
                "connections": broadcaster.connection_count,
                "devices": frame_cache.device_ids(),
            },
        )

    return {
        "status": "ready",
        "service": settings.service_name,
        "checks": checks,
        "connections": broadcaster.connection_count,
        "devices": frame_cache.device_ids(),
    }


# WebSocket endpoint

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    if broadcaster.connection_count >= settings.websocket_max_connections:
        await ws.close(code=1013, reason="Too many connections")
        return

    await ws.accept()
    broadcaster.add(ws)
    logger.info("ws.connected", client=ws.client)

    try:
        # Send current device list on connect
        await broadcaster.send_device_list(ws)

        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
                if data.get("type") == "subscribe_device":
                    device_id = data.get("device_id", "")
                    await broadcaster.send_latest_frame(ws, device_id)
            except json.JSONDecodeError:
                logger.warning("ws.invalid_json")
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.error("ws.error", exc_info=True)
    finally:
        broadcaster.remove(ws)
        logger.info("ws.disconnected", client=ws.client)
