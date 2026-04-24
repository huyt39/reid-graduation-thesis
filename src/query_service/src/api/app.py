from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4
import structlog
import time
from fastapi import FastAPI, HTTPException, Request

from src.api import deps
from src.api.routes import devices, persons, search, stats
from src.core.config import settings
from src.db.mongo_client import MongoQueryClient
from src.db.qdrant_client import QdrantQueryClient
from src.db.redis_client import RedisQueryCache
from src.services.nl_parser import NLQueryParser
from src.services.query_executor import QueryExecutor
from src.services.minio_urls import MinIOURLBuilder

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    mongo = MongoQueryClient(settings.mongo_uri, settings.mongo_db)
    qdrant = QdrantQueryClient(settings.qdrant_host, settings.qdrant_port)
    redis_cache = RedisQueryCache(settings.redis_url)
    executor = QueryExecutor(mongo, qdrant, redis_cache)
    nl_parser = NLQueryParser(vllm_url=settings.vllm_service_url)
    minio_urls = MinIOURLBuilder(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
    )

    deps.init(mongo, qdrant, redis_cache, executor, nl_parser, minio_urls)
    log.info("query_service.started")
    yield

    mongo.close()
    await redis_cache.close()
    log.info("query_service.stopped")


app = FastAPI(title=settings.service_name, lifespan=lifespan)

@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid4()))
    request.state.request_id = request_id

    started_at = time.perf_counter()
    response = await call_next(request)
    response.headers.setdefault("x-request-id", request_id)

    log.info(
        "query_service.http_response",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
    )
    return response


app.include_router(persons.router)
app.include_router(devices.router)
app.include_router(search.router)
app.include_router(stats.router)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": settings.service_name}


@app.get("/readyz")
async def readyz():
    mongo = deps.get_mongo()
    qdrant = deps.get_qdrant()
    redis_cache = deps.get_redis()
    minio_urls = deps.get_minio_urls()

    checks = {
        "mongo": await mongo.ping(),
        "qdrant": qdrant.ping(),
        "redis": await redis_cache.ping(),
        "minio": minio_urls.ping(),
    }
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