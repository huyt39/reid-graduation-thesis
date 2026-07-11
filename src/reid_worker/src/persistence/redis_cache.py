"""Redis hot-cache for person metadata and embeddings."""
from __future__ import annotations

import json

import numpy as np
import redis
import redis.asyncio as aioredis
import structlog

log = structlog.get_logger()

_META_TTL = 300   # 5 minutes
_EMB_TTL = 60     # 1 minute


class RedisPersonCache: # cache metadata và embedding người trong thời gian ngắn
    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        self._redis = aioredis.from_url(url, decode_responses=False)

    def _meta_key(self, person_id: int) -> str:
        return f"reid:person:{person_id}:meta"

    def _emb_key(self, person_id: int) -> str:
        return f"reid:person:{person_id}:emb"

    # ── Person metadata ───────────────────────────────────────────────

    async def set_person_meta(self, person_id: int, data: dict, ttl: int = _META_TTL) -> None: # lưu và đọc metadata người dưới dạng json, có ttl để tự hết hạn
        try:
            await self._redis.setex(self._meta_key(person_id), ttl, json.dumps(data))
        except Exception:
            log.error("redis.set_meta_failed", person_id=person_id, exc_info=True)

    async def get_person_meta(self, person_id: int) -> dict | None:
        try:
            raw = await self._redis.get(self._meta_key(person_id))
            return json.loads(raw) if raw else None
        except Exception:
            log.error("redis.get_meta_failed", person_id=person_id, exc_info=True)
            return None

    # ── Person embedding ──────────────────────────────────────────────

    async def set_person_embedding(self, person_id: int, embedding: np.ndarray, ttl: int = _EMB_TTL) -> None: # Lưu và đọc embedding dạng bytes float32. Dùng khi cần lấy embedding nhanh mà không đi qua storage nặng hơn
        try:
            await self._redis.setex(self._emb_key(person_id), ttl, embedding.astype(np.float32).tobytes())
        except Exception:
            log.error("redis.set_emb_failed", person_id=person_id, exc_info=True)

    async def get_person_embedding(self, person_id: int) -> np.ndarray | None:
        try:
            raw = await self._redis.get(self._emb_key(person_id))
            if raw:
                return np.frombuffer(raw, dtype=np.float32)
            return None
        except Exception:
            log.error("redis.get_emb_failed", person_id=person_id, exc_info=True)
            return None

    # ── Invalidation ──────────────────────────────────────────────────

    async def invalidate(self, person_id: int) -> None: # Xóa cache metadata và embedding của một person khi dữ liệu không còn đáng tin hoặc đã cập nhật
        try:
            await self._redis.delete(self._meta_key(person_id), self._emb_key(person_id))
        except Exception:
            log.error("redis.invalidate_failed", person_id=person_id, exc_info=True)

    async def close(self) -> None:
        await self._redis.aclose()


class RedisPersonIdAllocator: # Cấp person_id tăng dần bằng Redis INCR. Đây là cách đảm bảo mỗi person mới có ID duy nhất
    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        key: str = "reid:seq:person_id",
    ) -> None:
        self._redis = redis.Redis.from_url(url, decode_responses=True)
        self._key = key

    def allocate(self) -> int:
        return int(self._redis.incr(self._key))

    def close(self) -> None:
        self._redis.close()
