"""Redis cache layer — check before hitting MongoDB."""
from __future__ import annotations

import json

import redis.asyncio as aioredis
import structlog

log = structlog.get_logger()

_TTL = 300  # 5 minutes


class RedisQueryCache:
    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        self._redis = aioredis.from_url(url, decode_responses=True)

    async def get_person(self, person_id: int) -> dict | None:
        try:
            raw = await self._redis.get(f"reid:person:{person_id}:meta")
            return json.loads(raw) if raw else None
        except Exception:
            return None

    async def set_person(self, person_id: int, data: dict) -> None:
        try:
            await self._redis.setex(f"reid:person:{person_id}:meta", _TTL, json.dumps(data, default=str))
        except Exception:
            log.error("redis_cache.set_failed", person_id=person_id, exc_info=True)

    async def get_cached(self, key: str) -> dict | None:
        try:
            raw = await self._redis.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    async def set_cached(self, key: str, data, ttl: int = 30) -> None:
        try:
            await self._redis.setex(key, ttl, json.dumps(data, default=str))
        except Exception:
            pass

    async def close(self) -> None:
        await self._redis.aclose()
