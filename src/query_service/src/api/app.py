from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from src.api import deps
from src.api.routes import devices, persons, search, stats
from src.core.config import settings
from src.db.mongo_client import MongoQueryClient
from src.db.qdrant_client import QdrantQueryClient
from src.db.redis_client import RedisQueryCache
from src.services.nl_parser import NLQueryParser
from src.services.query_executor import QueryExecutor

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    mongo = MongoQueryClient(settings.mongo_uri, settings.mongo_db)
    qdrant = QdrantQueryClient(settings.qdrant_host, settings.qdrant_port)
    redis_cache = RedisQueryCache(settings.redis_url)
    executor = QueryExecutor(mongo, qdrant, redis_cache)
    nl_parser = NLQueryParser(vllm_url=settings.vllm_service_url)

    deps.init(mongo, qdrant, redis_cache, executor, nl_parser)
    log.info("query_service.started")
    yield

    mongo.close()
    await redis_cache.close()
    log.info("query_service.stopped")


app = FastAPI(title=settings.service_name, lifespan=lifespan)

app.include_router(persons.router)
app.include_router(devices.router)
app.include_router(search.router)
app.include_router(stats.router)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": settings.service_name}


@app.get("/readyz")
def readyz():
    return {"status": "ready", "service": settings.service_name}
