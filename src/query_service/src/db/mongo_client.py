"""Async MongoDB read client for the query service."""
from __future__ import annotations

from datetime import datetime

import pymongo
from motor.motor_asyncio import AsyncIOMotorClient


class MongoQueryClient:
    def __init__(self, uri: str, db_name: str) -> None:
        self._client = AsyncIOMotorClient(uri)
        self._db = self._client[db_name]

    # Persons

    async def get_person(self, person_id: int) -> dict | None:
        return await self._db.persons.find_one({"person_id": person_id}, {"_id": 0})

    async def search_persons(
        self,
        filters: dict | None = None,
        sort_field: str = "stats.last_seen_at",
        sort_order: int = pymongo.DESCENDING,
        skip: int = 0,
        limit: int = 20,
    ) -> tuple[list[dict], int]:
        query = filters or {}
        total = await self._db.persons.count_documents(query)
        cursor = (
            self._db.persons.find(query, {"_id": 0})
            .sort(sort_field, sort_order)
            .skip(skip)
            .limit(limit)
        )
        items = await cursor.to_list(length=limit)
        return items, total

    # Sightings

    async def get_sightings(
        self,
        person_id: int,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        skip: int = 0,
        limit: int = 20,
    ) -> tuple[list[dict], int]:
        query: dict = {"person_id": person_id}
        if start_time or end_time:
            time_filter: dict = {}
            if start_time:
                time_filter["$gte"] = start_time
            if end_time:
                time_filter["$lte"] = end_time
            query["started_at"] = time_filter

        total = await self._db.sightings.count_documents(query)
        cursor = (
            self._db.sightings.find(query, {"_id": 0})
            .sort("started_at", pymongo.DESCENDING)
            .skip(skip)
            .limit(limit)
        )
        items = await cursor.to_list(length=limit)
        return items, total

    async def get_tracklets(
        self,
        person_id: int,
        skip: int = 0,
        limit: int = 20,
    ) -> tuple[list[dict], int]:
        query: dict = {"person_id": person_id}
        total = await self._db.tracklets.count_documents(query)
        cursor = (
            self._db.tracklets.find(query, {"_id": 0})
            .sort("created_at", pymongo.DESCENDING)
            .skip(skip)
            .limit(limit)
        )
        items = await cursor.to_list(length=limit)
        return items, total

    async def get_occlusion_candidates(
        self,
        status: str | None = "unconfirmed",
        device_id: str | None = None,
        skip: int = 0,
        limit: int = 20,
    ) -> tuple[list[dict], int]:
        query: dict = {}
        if status:
            query["status"] = status
        if device_id:
            query["device_id"] = device_id
        total = await self._db.occlusion_candidates.count_documents(query)
        cursor = (
            self._db.occlusion_candidates.find(query, {"_id": 0})
            .sort("created_at", pymongo.DESCENDING)
            .skip(skip)
            .limit(limit)
        )
        items = await cursor.to_list(length=limit)
        return items, total

    # Timeline

    async def get_timeline(
        self,
        person_id: int,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[str] | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[list[dict], int]:
        query: dict = {"person_id": person_id}
        if start_time or end_time:
            tf: dict = {}
            if start_time:
                tf["$gte"] = start_time
            if end_time:
                tf["$lte"] = end_time
            query["timestamp"] = tf
        if event_types:
            query["event_type"] = {"$in": event_types}

        total = await self._db.timeline.count_documents(query)
        cursor = (
            self._db.timeline.find(query, {"_id": 0})
            .sort("timestamp", pymongo.DESCENDING)
            .skip(skip)
            .limit(limit)
        )
        items = await cursor.to_list(length=limit)
        return items, total

    # Devices

    async def get_device(self, device_id: str) -> dict | None:
        doc = await self._db.devices.find_one({"device_id": device_id}, {"_id": 0})
        if doc:
            return doc

        pipeline = [
            {"$match": {"device_id": device_id}},
            {"$group": {
                "_id": "$device_id",
                "sighting_count": {"$sum": 1},
                "last_seen_at": {"$max": "$started_at"},
                "first_seen_at": {"$min": "$started_at"},
                "unique_persons": {"$addToSet": "$person_id"},
            }}
        ]
        cursor = self._db.sightings.aggregate(pipeline)
        rows = await cursor.to_list(length=1)
        if not rows:
            return None

        r = rows[0]
        return {
            "device_id": device_id,
            "sighting_count": r["sighting_count"],
            "unique_person_count": len(r["unique_persons"]),
            "first_seen_at": r["first_seen_at"],
            "last_seen_at": r["last_seen_at"],
        }

    async def list_devices(self) -> list[dict]:
        cursor = self._db.devices.find({}, {"_id": 0})
        registered = await cursor.to_list(length=100)
        if registered:
            return registered

        pipeline = [
            {
                "$group": {
                    "_id": "$device_id",
                    "sighting_count": {"$sum": 1},
                    "last_seen_at": {"$max": "$started_at"},
                    "unique_persons": {"$addToSet": "$person_id"},
                }
            },
            {"$sort": {"last_seen_at": -1}},
        ]
        cursor = self._db.sightings.aggregate(pipeline)
        rows = await cursor.to_list(length=100)

        return [
            {
                "device_id": row["_id"],
                "sighting_count": row["sighting_count"],
                "unique_person_count": len(row["unique_persons"]),
                "last_seen_at": row["last_seen_at"],
            }
            for row in rows
        ]

    # Stats

    async def get_stats(self) -> dict:
        total_persons = await self._db.persons.count_documents({})
        active_persons = await self._db.persons.count_documents({"is_active": True})
        total_sightings = await self._db.sightings.count_documents({})

        total_devices = await self._db.devices.count_documents({})
        if total_devices == 0:
            device_ids = await self._db.sightings.distinct("device_id")
            total_devices = len(device_ids)

        return {
            "total_persons": total_persons,
            "active_persons": active_persons,
            "total_sightings": total_sightings,
            "total_devices": total_devices,
        }

    # Aggregation

    async def aggregate_sightings(
        self,
        person_id: int | None = None,
        device_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        group_by: str = "hour",
    ) -> list[dict]:
        match: dict = {}
        if person_id is not None:
            match["person_id"] = person_id
        if device_id:
            match["device_id"] = device_id
        if start_time or end_time:
            tf: dict = {}
            if start_time:
                tf["$gte"] = start_time
            if end_time:
                tf["$lte"] = end_time
            match["started_at"] = tf

        group_expr: dict
        if group_by == "device":
            group_expr = "$device_id"
        elif group_by == "day":
            group_expr = {"$dateToString": {"format": "%Y-%m-%d", "date": "$started_at"}}
        else:  # hour
            group_expr = {"$dateToString": {"format": "%Y-%m-%dT%H:00", "date": "$started_at"}}

        pipeline = [
            {"$match": match},
            {"$group": {
                "_id": group_expr,
                "count": {"$sum": 1},
                "total_duration": {"$sum": "$duration_seconds"},
                "avg_quality": {"$avg": "$quality_score"},
            }},
            {"$sort": {"_id": 1}},
        ]
        cursor = self._db.sightings.aggregate(pipeline)
        return await cursor.to_list(length=200)

    async def ping(self) -> bool:
        try:
            await self._client.admin.command("ping")
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()
