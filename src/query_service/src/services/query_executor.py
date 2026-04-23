"""Dispatches structured queries to the appropriate backend."""
from __future__ import annotations

import structlog

from src.db.mongo_client import MongoQueryClient
from src.db.qdrant_client import QdrantQueryClient
from src.db.redis_client import RedisQueryCache
from src.schemas.query import (
    DeviceLookupParams,
    PersonLookupParams,
    PersonSearchParams,
    SightingAggregationParams,
    SimilaritySearchParams,
    StructuredQueryRequest,
    TimelineParams,
)


log = structlog.get_logger()


class QueryExecutor:
    def __init__(
        self,
        mongo: MongoQueryClient,
        qdrant: QdrantQueryClient,
        redis_cache: RedisQueryCache,
    ) -> None:
        self.mongo = mongo
        self.qdrant = qdrant
        self.redis = redis_cache

    async def execute(self, query: dict | StructuredQueryRequest) -> dict:
        if isinstance(query, StructuredQueryRequest):
            payload = query.model_dump()
        else:
            payload = query

        qt = payload.get("query_type", "error")
        params = payload.get("params", {})

        handler = {
            "person_lookup": self._person_lookup,
            "person_search": self._person_search,
            "timeline": self._timeline,
            "similarity_search": self._similarity_search,
            "sighting_aggregation": self._sighting_aggregation,
            "device_lookup": self._device_lookup,
        }.get(qt)

        if handler is None:
            return {"error": payload.get("message", f"Unknown query type: {qt}")}

        return await handler(params)

    async def _person_lookup(self, params: dict) -> dict:
        parsed = PersonLookupParams(**params)
        pid = parsed.person_id
        # Check cache first
        cached = await self.redis.get_person(pid)
        if cached:
            return {"person": cached}
        person = await self.mongo.get_person(pid)
        if person is None:
            return {"error": f"Person {pid} not found"}
        await self.redis.set_person(pid, person)
        return {"person": person}

    async def _person_search(self, params: dict) -> dict:
        parsed = PersonSearchParams(**params)
        filters = parsed.filters
        mongo_query: dict = {}
        if filters.gender:
            mongo_query["attributes.gender"] = filters.gender
        if filters.gender_confidence_min is not None:
            mongo_query["attributes.gender_confidence"] = {"$gte": filters.gender_confidence_min}
        if filters.last_seen_device:
            mongo_query["stats.last_seen_device"] = filters.last_seen_device
        if filters.last_seen_after:
            mongo_query.setdefault("stats.last_seen_at", {})["$gte"] = filters.last_seen_after
        if filters.last_seen_before:
            mongo_query.setdefault("stats.last_seen_at", {})["$lte"] = filters.last_seen_before
        if filters.is_active is not None:
            mongo_query["is_active"] = filters.is_active

        skip = (parsed.page - 1) * parsed.page_size

        items, total = await self.mongo.search_persons(
            filters=mongo_query, skip=skip, limit=parsed.page_size,
        )
        return {"items": items, "total": total, "page": parsed.page, "page_size": parsed.page_size}

    async def _timeline(self, params: dict) -> dict:
        parsed = TimelineParams(**params)

        items, total = await self.mongo.get_timeline(
            person_id=parsed.person_id,
            start_time=parsed.start_time,
            end_time=parsed.end_time,
            event_types=parsed.event_types,
        )
        return {"items": items, "total": total}

    async def _similarity_search(self, params: dict) -> dict:
        parsed = SimilaritySearchParams(**params)

        results = self.qdrant.search_similar(
            person_id=parsed.person_id,
            top_k=parsed.top_k,
            min_score=parsed.min_score,
        )
        return {"similar_persons": results}

    async def _sighting_aggregation(self, params: dict) -> dict:
        parsed = SightingAggregationParams(**params)
        results = await self.mongo.aggregate_sightings(
            person_id=parsed.person_id,
            device_id=parsed.device_id,
            start_time=parsed.start_time,
            end_time=parsed.end_time,
            group_by=parsed.group_by,
        )
        return {"aggregation": results}

    async def _device_lookup(self, params: dict) -> dict:
        parsed = DeviceLookupParams(**params)
        device_id = parsed.device_id
        if device_id:
            device = await self.mongo.get_device(device_id)
            return {"device": device} if device else {"error": f"Device {device_id} not found"}
        devices = await self.mongo.list_devices()
        return {"devices": devices}
