"""Async MongoDB persistence for persons, tracklets, sightings, and timeline."""
from __future__ import annotations

from datetime import datetime, timezone

import structlog
from motor.motor_asyncio import AsyncIOMotorClient

log = structlog.get_logger()


class MongoPersonStore:
    """Fire-and-forget async writes — never blocks the main pipeline."""

    PERSONS = "persons"
    TRACKLETS = "tracklets"
    SIGHTINGS = "sightings"
    TIMELINE = "timeline"

    def __init__(self, uri: str = "mongodb://localhost:27017", db_name: str = "reid_production") -> None:
        self._client = AsyncIOMotorClient(uri)
        self._db = self._client[db_name]

    async def ensure_indexes(self) -> None:
        db = self._db
        await db[self.PERSONS].create_index("person_id", unique=True)
        await db[self.PERSONS].create_index("stats.last_seen_at")
        await db[self.TRACKLETS].create_index("tracklet_id", unique=True)
        await db[self.TRACKLETS].create_index([("person_id", 1), ("created_at", -1)])
        await db[self.TRACKLETS].create_index([("device_id", 1), ("created_at", -1)])
        await db[self.SIGHTINGS].create_index([("person_id", 1), ("started_at", -1)])
        await db[self.SIGHTINGS].create_index([("device_id", 1), ("started_at", -1)])
        await db[self.TIMELINE].create_index([("person_id", 1), ("timestamp", -1)])
        log.info("mongo.indexes_ensured")

    # ── Writes ────────────────────────────────────────────────────────

    async def upsert_person(
        self,
        person_id: int,
        *,
        attributes: dict[str, tuple[str, float]] | None = None,
        device_id: str = "",
        snapshot_key: str | None = None,
        source: str = "new_detection",
    ) -> None:
        """Upsert a person doc.

        ``attributes`` is the per-task person-level snapshot from the AttributeVoter:
        ``{task: (label, confidence)}``. Each task lands as ``attributes.<task>`` and
        ``attributes.<task>_confidence`` in the document.
        """
        now = datetime.now(timezone.utc)
        attr_set: dict = {}
        for task, (label, conf) in (attributes or {}).items():
            attr_set[f"attributes.{task}"] = label
            attr_set[f"attributes.{task}_confidence"] = float(conf)
        update: dict = {
            "$set": {
                **attr_set,
                "stats.last_seen_at": now,
                "stats.last_seen_device": device_id,
                "updated_at": now,
            },
            "$inc": {"stats.sighting_count": 1},
            "$setOnInsert": {
                "person_id": person_id,
                "stats.first_seen_at": now,
                "stats.first_seen_device": device_id,
                "source": source,
                "is_active": True,
                "created_at": now,
            },
        }
        if snapshot_key:
            update["$set"]["snapshot_key"] = snapshot_key
        try:
            await self._db[self.PERSONS].update_one(
                {"person_id": person_id}, update, upsert=True,
            )
        except Exception:
            log.error("mongo.upsert_person_failed", person_id=person_id, exc_info=True)

    async def add_tracklet_record(
        self,
        *,
        tracklet_id: str,
        track_id: int,
        person_id: int | None,
        device_id: str,
        state: str,
        frame_start: int,
        frame_end: int,
        entry_count: int,
        quality: dict,
        matching: dict,
        best_crop_key: str | None = None,
    ) -> None:
        doc = {
            "tracklet_id": tracklet_id,
            "track_id": track_id,
            "person_id": person_id,
            "device_id": device_id,
            "state": state,
            "frame_range": {"start": frame_start, "end": frame_end},
            "entry_count": entry_count,
            "quality": quality,
            "matching": matching,
            "best_crop_key": best_crop_key,
            "created_at": datetime.now(timezone.utc),
        }
        try:
            await self._db[self.TRACKLETS].insert_one(doc)
        except Exception:
            log.error("mongo.add_tracklet_failed", tracklet_id=tracklet_id, exc_info=True)

    async def add_sighting(
        self,
        *,
        person_id: int,
        device_id: str,
        tracklet_id: str,
        started_at: datetime,
        ended_at: datetime,
        entry_count: int,
        quality_score: float,
        snapshot_key: str | None = None,
        attributes: dict[str, tuple[str, float]] | None = None,
    ) -> None:
        """Insert a sighting row.

        ``attributes`` is the same per-task ``(label, confidence)`` snapshot used by
        ``upsert_person``. Stored as ``{<task>: label, <task>_confidence: conf, ...}``.
        """
        attr_doc: dict = {}
        for task, (label, conf) in (attributes or {}).items():
            attr_doc[task] = label
            attr_doc[f"{task}_confidence"] = float(conf)
        doc = {
            "person_id": person_id,
            "device_id": device_id,
            "tracklet_id": tracklet_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": (ended_at - started_at).total_seconds(),
            "entry_count": entry_count,
            "quality_score": quality_score,
            "snapshot_key": snapshot_key,
            "attributes": attr_doc,
        }
        try:
            await self._db[self.SIGHTINGS].insert_one(doc)
        except Exception:
            log.error("mongo.add_sighting_failed", person_id=person_id, exc_info=True)

    async def add_timeline_event(
        self,
        *,
        person_id: int,
        event_type: str,
        device_id: str = "",
        details: dict | None = None,
    ) -> None:
        doc = {
            "person_id": person_id,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc),
            "device_id": device_id,
            "details": details or {},
        }
        try:
            await self._db[self.TIMELINE].insert_one(doc)
        except Exception:
            log.error("mongo.add_timeline_failed", person_id=person_id, exc_info=True)

    def close(self) -> None:
        self._client.close()
