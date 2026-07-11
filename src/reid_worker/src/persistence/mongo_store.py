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
    OCCLUSION_CANDIDATES = "occlusion_candidates"

    # kết nối mongodb và chọn database lưu dữ liệu reid
    def __init__(self, uri: str = "mongodb://localhost:27017", db_name: str = "reid_production") -> None:
        self._client = AsyncIOMotorClient(uri)
        self._db = self._client[db_name]

    # kiểm tra xung đột gender mạnh để chặn merge nhầm
    @staticmethod
    def attributes_have_strong_gender_conflict(attrs_a: dict, attrs_b: dict) -> bool:
        """Gender-only confident disagreement. Unconditional block — never
        overridden by embedding similarity. Rationale: at confidence ≥ 0.85,
        the gender attribute model is reliable, and gender doesn't change
        across cameras / poses / time the way clothing can. If both sides
        confidently say different genders, they are different people, even
        when the embedding scores high (the embedding model can be confused
        on similar clothing colors / silhouettes).
        """
        confidence_threshold = 0.85
        label_a = attrs_a.get("gender")
        label_b = attrs_b.get("gender")
        conf_a = float(attrs_a.get("gender_confidence", 0.0) or 0.0)
        conf_b = float(attrs_b.get("gender_confidence", 0.0) or 0.0)
        return (
            bool(label_a)
            and bool(label_b)
            and label_a != "unknown"
            and label_b != "unknown"
            and label_a != label_b
            and conf_a >= confidence_threshold
            and conf_b >= confidence_threshold
        )

    # kiểm tra xung đột thuộc tính mức vừa để làm tín hiệu nghi ngờ split id
    @staticmethod
    def attributes_have_moderate_conflict(attrs_a: dict, attrs_b: dict) -> bool:
        """Looser version of attributes_have_strong_conflict.

        Used as a *signal* (not a blocker) to detect identity-split situations
        even when one side's voted attribute confidence is muted by prior
        contamination. A person whose tracklets vote 3 male / 2 female has a
        voted gender_confidence around 0.6 — below the strong-conflict floor of
        0.85 — so the strong check misses the disagreement. The moderate check
        runs at 0.60 to catch this pattern.
        """
        confidence_threshold = 0.60
        label_a = attrs_a.get("gender")
        label_b = attrs_b.get("gender")
        conf_a = float(attrs_a.get("gender_confidence", 0.0) or 0.0)
        conf_b = float(attrs_b.get("gender_confidence", 0.0) or 0.0)
        if (
            label_a
            and label_b
            and label_a != "unknown"
            and label_b != "unknown"
            and label_a != label_b
            and conf_a >= confidence_threshold
            and conf_b >= confidence_threshold
        ):
            return True
        return False

    # kiểm tra xung đột thuộc tính mạnh đủ để chặn merge identity
    @staticmethod
    def attributes_have_strong_conflict(attrs_a: dict, attrs_b: dict) -> bool:
        """Return True only for semantic conflicts strong enough to block merging.

        Under occlusion, single accessory/clothing attributes can flip because the
        crop may only show a partial body or another object. Gender remains a hard
        conflict when both sides are high-confidence; other attributes need at
        least two independent conflicts before they can veto an appearance merge.
        """
        confidence_threshold = 0.85
        label_a = attrs_a.get("gender")
        label_b = attrs_b.get("gender")
        conf_a = float(attrs_a.get("gender_confidence", 0.0) or 0.0)
        conf_b = float(attrs_b.get("gender_confidence", 0.0) or 0.0)
        if (
            label_a
            and label_b
            and label_a != "unknown"
            and label_b != "unknown"
            and label_a != label_b
            and conf_a >= confidence_threshold
            and conf_b >= confidence_threshold
        ):
            return True

        conflict_count = 0
        stable_tasks = ("backpack", "hat", "lower", "sleeve")
        for task in stable_tasks:
            label_a = attrs_a.get(task)
            label_b = attrs_b.get(task)
            conf_a = float(attrs_a.get(f"{task}_confidence", 0.0) or 0.0)
            conf_b = float(attrs_b.get(f"{task}_confidence", 0.0) or 0.0)
            if (
                label_a
                and label_b
                and label_a != "unknown"
                and label_b != "unknown"
                and label_a != label_b
                and conf_a >= confidence_threshold
                and conf_b >= confidence_threshold
            ):
                conflict_count += 1
        return conflict_count >= 2

    # tạo index cho các collection để query nhanh và tránh trùng khóa chính
    async def ensure_indexes(self) -> None:
        db = self._db
        await db[self.PERSONS].create_index("person_id", unique=True)
        await db[self.PERSONS].create_index("stats.last_seen_at")
        await db[self.TRACKLETS].create_index("tracklet_id", unique=True)
        await db[self.TRACKLETS].create_index([("person_id", 1), ("created_at", -1)])
        await db[self.TRACKLETS].create_index([("device_id", 1), ("created_at", -1)])
        await db[self.TRACKLETS].create_index([("person_id", 1), ("frame_range.start", 1)])
        await db[self.SIGHTINGS].create_index([("person_id", 1), ("started_at", -1)])
        await db[self.SIGHTINGS].create_index([("device_id", 1), ("started_at", -1)])
        await db[self.TIMELINE].create_index([("person_id", 1), ("timestamp", -1)])
        await db[self.OCCLUSION_CANDIDATES].create_index("candidate_id", unique=True)
        await db[self.OCCLUSION_CANDIDATES].create_index([("status", 1), ("created_at", -1)])
        await db[self.OCCLUSION_CANDIDATES].create_index([("device_id", 1), ("created_at", -1)])
        log.info("mongo.indexes_ensured")

    # ── Writes ────────────────────────────────────────────────────────

    # tạo hoặc cập nhật document person cùng attributes và snapshot tốt nhất
    async def upsert_person(
        self,
        person_id: int,
        *,
        attributes: dict[str, tuple[str, float]] | None = None,
        device_id: str = "",
        snapshot_key: str | None = None,
        snapshot_score: float | None = None,
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
        try:
            await self._db[self.PERSONS].update_one(
                {"person_id": person_id}, update, upsert=True,
            )
            if snapshot_key:
                score = float(snapshot_score or 0.0)
                await self._db[self.PERSONS].update_one(
                    {
                        "person_id": person_id,
                        "$or": [
                            {"stats.best_snapshot_score": {"$exists": False}},
                            {"stats.best_snapshot_score": {"$lt": score}},
                        ],
                    },
                    {
                        "$set": {
                            "snapshot_key": snapshot_key,
                            "stats.best_snapshot_score": score,
                        }
                    },
                )
        except Exception:
            log.error("mongo.upsert_person_failed", person_id=person_id, exc_info=True)

    # lưu record tracklet đã xử lý cùng quality, matching và evidence
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
        frame_indices: list[int] | None = None,
        entry_count: int,
        quality: dict,
        matching: dict,
        evidence: dict,
        first_bbox_xyxy: list[float] | None = None,
        last_bbox_xyxy: list[float] | None = None,
        best_crop_key: str | None = None,
    ) -> None:
        # frame_indices: the actual frames where this tracklet had detections.
        # Used by persons_cooccur for precise per-frame cooccurrence checks
        # (vs. crude frame_range overlap which gives false positives on
        # identity splits). Optional for backward compat.
        doc = {
            "tracklet_id": tracklet_id,
            "track_id": track_id,
            "person_id": person_id,
            "device_id": device_id,
            "state": state,
            "frame_range": {"start": frame_start, "end": frame_end},
            "frame_indices": [int(f) for f in (frame_indices or [])],
            "entry_count": entry_count,
            "quality": quality,
            "matching": matching,
            "evidence": evidence,
            "first_bbox_xyxy": [float(v) for v in (first_bbox_xyxy or [])],
            "last_bbox_xyxy": [float(v) for v in (last_bbox_xyxy or [])],
            "best_crop_key": best_crop_key,
            "created_at": datetime.now(timezone.utc),
        }
        try:
            await self._db[self.TRACKLETS].insert_one(doc)
        except Exception:
            log.error("mongo.add_tracklet_failed", tracklet_id=tracklet_id, exc_info=True)

    # lưu một lần xuất hiện của person theo tracklet
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

    # cập nhật snapshot đại diện nếu snapshot mới có điểm tốt hơn
    async def update_person_snapshot(
        self,
        person_id: int,
        *,
        snapshot_key: str,
        snapshot_score: float,
    ) -> None:
        try:
            await self._db[self.PERSONS].update_one(
                {
                    "person_id": person_id,
                    "$or": [
                        {"stats.best_snapshot_score": {"$exists": False}},
                        {"stats.best_snapshot_score": {"$lt": float(snapshot_score)}},
                    ],
                },
                {
                    "$set": {
                        "snapshot_key": snapshot_key,
                        "stats.best_snapshot_score": float(snapshot_score),
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )
        except Exception:
            log.error(
                "mongo.update_person_snapshot_failed",
                person_id=person_id,
                snapshot_key=snapshot_key,
                exc_info=True,
            )

    # cập nhật asset/evidence bổ sung cho tracklet sau khi upload ảnh
    async def update_tracklet_assets(
        self,
        tracklet_id: str,
        *,
        best_crop_key: str | None = None,
        evidence: dict | None = None,
    ) -> None:
        update_set: dict = {}
        if best_crop_key is not None:
            update_set["best_crop_key"] = best_crop_key
        if evidence is not None:
            update_set["evidence"] = evidence
        if not update_set:
            return
        try:
            await self._db[self.TRACKLETS].update_one(
                {"tracklet_id": tracklet_id},
                {"$set": update_set},
            )
        except Exception:
            log.error(
                "mongo.update_tracklet_assets_failed",
                tracklet_id=tracklet_id,
                exc_info=True,
            )

    # cập nhật snapshot key cho sighting theo tracklet
    async def update_sighting_snapshot(
        self,
        tracklet_id: str,
        *,
        snapshot_key: str,
    ) -> None:
        try:
            await self._db[self.SIGHTINGS].update_one(
                {"tracklet_id": tracklet_id},
                {"$set": {"snapshot_key": snapshot_key}},
            )
        except Exception:
            log.error(
                "mongo.update_sighting_snapshot_failed",
                tracklet_id=tracklet_id,
                snapshot_key=snapshot_key,
                exc_info=True,
            )

    # ghi event vào timeline của person
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

    # lưu candidate bị occlusion hoặc chưa đủ chắc để gán identity
    async def add_occlusion_candidate(
        self,
        *,
        candidate_id: str,
        track_id: int,
        device_id: str,
        reason: str,
        status: str,
        frame_start: int,
        frame_end: int,
        entry_count: int,
        quality: dict,
        evidence: dict,
        best_crop_key: str | None = None,
        matching: dict | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        doc = {
            "candidate_id": candidate_id,
            "track_id": track_id,
            "device_id": device_id,
            "reason": reason,
            "status": status,
            "frame_range": {"start": frame_start, "end": frame_end},
            "entry_count": entry_count,
            "quality": quality,
            "matching": matching or {},
            "evidence": evidence,
            "best_crop_key": best_crop_key,
            "created_at": now,
            "updated_at": now,
        }
        try:
            await self._db[self.OCCLUSION_CANDIDATES].update_one(
                {"candidate_id": candidate_id},
                {"$set": doc},
                upsert=True,
            )
        except Exception:
            log.error(
                "mongo.add_occlusion_candidate_failed",
                candidate_id=candidate_id,
                track_id=track_id,
                exc_info=True,
            )

    # đánh dấu occlusion candidate đã được attach vào một person
    async def mark_occlusion_candidate_attached(self, candidate_id: str, person_id: int) -> None:
        """Resolve an orphan occlusion candidate after it was attached to a person
        as evidence — flips its status so it no longer reads as unresolved."""
        try:
            await self._db[self.OCCLUSION_CANDIDATES].update_one(
                {"candidate_id": candidate_id},
                {"$set": {
                    "status": "attached",
                    "attached_person_id": int(person_id),
                    "updated_at": datetime.now(timezone.utc),
                }},
            )
        except Exception:
            log.error(
                "mongo.mark_occlusion_candidate_attached_failed",
                candidate_id=candidate_id,
                exc_info=True,
            )

    # đếm toàn bộ tracklet đang gắn với person
    async def count_tracklets(self, person_id: int) -> int:
        return await self._db[self.TRACKLETS].count_documents({"person_id": person_id})

    # đếm tracklet canonical đủ tin cậy để làm bằng chứng merge
    async def count_canonical_tracklets(self, person_id: int) -> int:
        """Count person evidence that is allowed to anchor identity merges.

        Occlusion-attached/provisional rows are useful ReID evidence, but they
        are intentionally not canonical anchors. Counting them as support makes
        weak occlusion bridges overconfident and can collapse two established
        people into one ID.
        """
        return await self._db[self.TRACKLETS].count_documents(
            {
                "person_id": person_id,
                "state": {"$ne": "occlusion_attached"},
                "matching.provisional": {"$ne": True},
            }
        )

    # tính độ di chuyển tổng thể để phát hiện vật thể tĩnh bị nhận nhầm là người
    async def person_motion_extent(self, person_id: int) -> dict | None:
        """Aggregate a person's bbox centroid spread + mean size across ALL its
        tracklets (first+last bbox of each). Used by the person-level static-
        artifact filter: a static object (e.g. fire extinguisher) has near-zero
        centroid spread relative to its bbox size, robustly across the whole run
        (immune to single-tracklet jitter). Returns None if no bbox data."""
        cur = self._db[self.TRACKLETS].find(
            {"person_id": person_id},
            {"_id": 0, "first_bbox_xyxy": 1, "last_bbox_xyxy": 1},
        )
        docs = await cur.to_list(length=10000)
        cxs: list[float] = []
        cys: list[float] = []
        ws: list[float] = []
        hs: list[float] = []
        for d in docs:
            for key in ("first_bbox_xyxy", "last_bbox_xyxy"):
                b = d.get(key)
                if not b or len(b) < 4:
                    continue
                cxs.append((float(b[0]) + float(b[2])) / 2.0)
                cys.append((float(b[1]) + float(b[3])) / 2.0)
                ws.append(abs(float(b[2]) - float(b[0])))
                hs.append(abs(float(b[3]) - float(b[1])))
        if not cxs:
            return None
        mean_w = sum(ws) / len(ws)
        mean_h = sum(hs) / len(hs)
        return {
            "spread_x": max(cxs) - min(cxs),
            "spread_y": max(cys) - min(cys),
            "mean_width": mean_w,
            "mean_height": mean_h,
            "tracklet_count": len(docs),
            "point_count": len(cxs),
        }

    # xóa cứng person và các record liên quan trong mongo
    async def remove_person(self, person_id: int) -> None:
        """Hard-delete a person and all its records (PERSONS/SIGHTINGS/TRACKLETS/
        TIMELINE). Used by the static-artifact filter to drop a false-positive
        (static object) identity at stream finalization."""
        try:
            await self._db[self.TRACKLETS].delete_many({"person_id": person_id})
            await self._db[self.SIGHTINGS].delete_many({"person_id": person_id})
            await self._db[self.TIMELINE].delete_many({"person_id": person_id})
            await self._db[self.PERSONS].delete_one({"person_id": person_id})
        except Exception:
            log.error("mongo.remove_person_failed", person_id=person_id, exc_info=True)

    # lấy danh sách person gần đây còn active
    async def list_recent_person_ids(self, limit: int = 50) -> list[int]:
        cursor = self._db[self.PERSONS].find(
            {"is_active": {"$ne": False}},
            {"_id": 0, "person_id": 1},
        ).sort("stats.last_seen_at", -1).limit(int(limit))
        docs = await cursor.to_list(length=int(limit))
        return [int(doc["person_id"]) for doc in docs if doc.get("person_id") is not None]

    # kiểm tra hai person có xuất hiện chồng frame trên cùng camera không
    async def persons_cooccur(self, person_a: int, person_b: int) -> bool:
        """Return True when two persons appear in overlapping frames on the same device."""
        cursor = self._db[self.TRACKLETS].find(
            {"person_id": {"$in": [person_a, person_b]}},
            {"_id": 0, "person_id": 1, "device_id": 1, "frame_range": 1},
        )
        docs = await cursor.to_list(length=500)
        ranges: dict[int, list[tuple[str, int, int]]] = {person_a: [], person_b: []}
        for doc in docs:
            pid = int(doc.get("person_id"))
            frame_range = doc.get("frame_range") or {}
            ranges.setdefault(pid, []).append(
                (
                    str(doc.get("device_id", "")),
                    int(frame_range.get("start", -1)),
                    int(frame_range.get("end", -1)),
                )
            )
        for device_a, start_a, end_a in ranges.get(person_a, []):
            for device_b, start_b, end_b in ranges.get(person_b, []):
                if device_a != device_b:
                    continue
                if start_a <= end_b and start_b <= end_a:
                    return True
        return False

    # tính khoảng cách frame nhỏ nhất giữa hai person trên cùng camera
    async def persons_min_frame_gap(self, person_a: int, person_b: int) -> int | None:
        """Return minimum non-overlap frame gap between two persons on one device.

        A gap of 0 means their frame ranges touch or overlap. ``None`` means no
        comparable tracklets were found on the same device.
        """
        cursor = self._db[self.TRACKLETS].find(
            {"person_id": {"$in": [person_a, person_b]}},
            {"_id": 0, "person_id": 1, "device_id": 1, "frame_range": 1},
        )
        docs = await cursor.to_list(length=500)
        ranges: dict[int, list[tuple[str, int, int]]] = {person_a: [], person_b: []}
        for doc in docs:
            pid = int(doc.get("person_id"))
            frame_range = doc.get("frame_range") or {}
            ranges.setdefault(pid, []).append(
                (
                    str(doc.get("device_id", "")),
                    int(frame_range.get("start", -1)),
                    int(frame_range.get("end", -1)),
                )
            )

        best_gap: int | None = None
        for device_a, start_a, end_a in ranges.get(person_a, []):
            for device_b, start_b, end_b in ranges.get(person_b, []):
                if device_a != device_b:
                    continue
                if start_a <= end_b and start_b <= end_a:
                    gap = 0
                elif end_a < start_b:
                    gap = start_b - end_a
                else:
                    gap = start_a - end_b
                best_gap = gap if best_gap is None else min(best_gap, gap)
        return best_gap

    # đếm số camera khác nhau mà hai person cùng xuất hiện
    async def persons_distinct_device_count(self, person_a: int, person_b: int) -> int:
        """Count distinct cameras across both persons' tracklets.

        Used to scope the cross-device re-link path: it must only fire when the
        two identities together span >= 2 cameras. In a single-camera (single
        stream) run this is always 1, so that path provably never fires and the
        tuned single-stream behaviour is unaffected.
        """
        cursor = self._db[self.TRACKLETS].find(
            {"person_id": {"$in": [person_a, person_b]}},
            {"_id": 0, "device_id": 1},
        )
        docs = await cursor.to_list(length=500)
        devices = {str(doc.get("device_id", "")) for doc in docs if doc.get("device_id")}
        return len(devices)

    # lấy gap gần nhất kèm bbox chuyển tiếp để hỗ trợ so sánh duplicate
    async def persons_min_frame_gap_with_bboxes(
        self,
        person_a: int,
        person_b: int,
    ) -> dict | None:
        """Return closest temporal gap plus transition bboxes for two persons.

        Uses persisted tracklet endpoints, so duplicate reconciliation can still
        reason about split IDs after in-memory tracker state has been pruned.
        """
        cursor = self._db[self.TRACKLETS].find(
            {"person_id": {"$in": [person_a, person_b]}},
            {
                "_id": 0,
                "person_id": 1,
                "device_id": 1,
                "frame_range": 1,
                "first_bbox_xyxy": 1,
                "last_bbox_xyxy": 1,
            },
        )
        docs = await cursor.to_list(length=500)
        ranges: dict[int, list[dict]] = {person_a: [], person_b: []}
        for doc in docs:
            pid = int(doc.get("person_id"))
            frame_range = doc.get("frame_range") or {}
            ranges.setdefault(pid, []).append({
                "device_id": str(doc.get("device_id", "")),
                "start": int(frame_range.get("start", -1)),
                "end": int(frame_range.get("end", -1)),
                "first_bbox": [float(v) for v in (doc.get("first_bbox_xyxy") or [])],
                "last_bbox": [float(v) for v in (doc.get("last_bbox_xyxy") or [])],
            })

        best: dict | None = None
        for a in ranges.get(person_a, []):
            for b in ranges.get(person_b, []):
                if a["device_id"] != b["device_id"]:
                    continue
                if a["start"] <= b["end"] and b["start"] <= a["end"]:
                    gap = 0
                    transition_a = a["last_bbox"]
                    transition_b = b["first_bbox"]
                elif a["end"] < b["start"]:
                    gap = b["start"] - a["end"]
                    transition_a = a["last_bbox"]
                    transition_b = b["first_bbox"]
                else:
                    gap = a["start"] - b["end"]
                    transition_a = a["first_bbox"]
                    transition_b = b["last_bbox"]
                closeness = None
                if len(transition_a) >= 4 and len(transition_b) >= 4:
                    ax = (transition_a[0] + transition_a[2]) / 2.0
                    ay = (transition_a[1] + transition_a[3]) / 2.0
                    bx = (transition_b[0] + transition_b[2]) / 2.0
                    by = (transition_b[1] + transition_b[3]) / 2.0
                    size_a = max(transition_a[2] - transition_a[0], transition_a[3] - transition_a[1], 1.0)
                    size_b = max(transition_b[2] - transition_b[0], transition_b[3] - transition_b[1], 1.0)
                    closeness = (((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5) / max(size_a, size_b, 1.0)
                best_closeness = None if best is None else best.get("center_distance_ratio")
                if (
                    best is None
                    or gap < best["gap"]
                    or (
                        gap == best["gap"]
                        and closeness is not None
                        and (best_closeness is None or closeness < best_closeness)
                    )
                ):
                    best = {
                        "gap": int(gap),
                        "bbox_a": transition_a,
                        "bbox_b": transition_b,
                        "center_distance_ratio": closeness,
                    }
        return best

    # tìm chuyển tiếp bbox gần nhất trong một cửa sổ thời gian
    async def persons_closest_spatial_transition_with_bboxes(
        self,
        person_a: int,
        person_b: int,
        *,
        max_gap_frames: int,
    ) -> dict | None:
        """Return the closest bbox transition within a temporal window.

        Unlike persons_min_frame_gap_with_bboxes, this prioritizes spatial
        continuity over the smallest frame gap. It is used only for short
        occlusion fragments where the detector emits a partial duplicate in
        overlapping frames, but the physical trajectory becomes obvious a few
        frames later.
        """
        cursor = self._db[self.TRACKLETS].find(
            {"person_id": {"$in": [person_a, person_b]}},
            {
                "_id": 0,
                "person_id": 1,
                "device_id": 1,
                "frame_range": 1,
                "first_bbox_xyxy": 1,
                "last_bbox_xyxy": 1,
            },
        )
        docs = await cursor.to_list(length=500)
        ranges: dict[int, list[dict]] = {person_a: [], person_b: []}
        for doc in docs:
            pid = int(doc.get("person_id"))
            frame_range = doc.get("frame_range") or {}
            ranges.setdefault(pid, []).append({
                "device_id": str(doc.get("device_id", "")),
                "start": int(frame_range.get("start", -1)),
                "end": int(frame_range.get("end", -1)),
                "first_bbox": [float(v) for v in (doc.get("first_bbox_xyxy") or [])],
                "last_bbox": [float(v) for v in (doc.get("last_bbox_xyxy") or [])],
            })

        def _center_distance_ratio(box_a: list[float], box_b: list[float]) -> float | None:
            if len(box_a) < 4 or len(box_b) < 4:
                return None
            ax = (box_a[0] + box_a[2]) / 2.0
            ay = (box_a[1] + box_a[3]) / 2.0
            bx = (box_b[0] + box_b[2]) / 2.0
            by = (box_b[1] + box_b[3]) / 2.0
            size_a = max(box_a[2] - box_a[0], box_a[3] - box_a[1], 1.0)
            size_b = max(box_b[2] - box_b[0], box_b[3] - box_b[1], 1.0)
            return (((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5) / max(size_a, size_b, 1.0)

        best: dict | None = None
        max_gap = int(max_gap_frames)
        for a in ranges.get(person_a, []):
            for b in ranges.get(person_b, []):
                if a["device_id"] != b["device_id"]:
                    continue
                if a["start"] <= b["end"] and b["start"] <= a["end"]:
                    # Compare both endpoint directions for overlapping split boxes.
                    candidates = [
                        (0, a["last_bbox"], b["first_bbox"]),
                        (0, a["first_bbox"], b["last_bbox"]),
                    ]
                elif a["end"] < b["start"]:
                    candidates = [(b["start"] - a["end"], a["last_bbox"], b["first_bbox"])]
                else:
                    candidates = [(a["start"] - b["end"], a["first_bbox"], b["last_bbox"])]
                for gap, transition_a, transition_b in candidates:
                    if gap > max_gap:
                        continue
                    closeness = _center_distance_ratio(transition_a, transition_b)
                    if closeness is None:
                        continue
                    if (
                        best is None
                        or closeness < best["center_distance_ratio"]
                        or (
                            closeness == best["center_distance_ratio"]
                            and gap < best["gap"]
                        )
                    ):
                        best = {
                            "gap": int(gap),
                            "bbox_a": transition_a,
                            "bbox_b": transition_b,
                            "center_distance_ratio": closeness,
                        }
        return best

    # kiểm tra hai person có xung đột thuộc tính mạnh không
    async def persons_have_strong_attribute_conflict(self, person_a: int, person_b: int) -> bool:
        """Return True for high-confidence semantic conflicts between two identities."""
        docs = await self._db[self.PERSONS].find(
            {"person_id": {"$in": [person_a, person_b]}},
            {"_id": 0, "person_id": 1, "attributes": 1},
        ).to_list(length=2)
        attrs_by_pid = {int(doc.get("person_id")): doc.get("attributes") or {} for doc in docs}
        attrs_a = attrs_by_pid.get(person_a, {})
        attrs_b = attrs_by_pid.get(person_b, {})
        return self.attributes_have_strong_conflict(attrs_a, attrs_b)

    # kiểm tra riêng xung đột gender mạnh giữa hai person
    async def persons_have_strong_gender_conflict(self, person_a: int, person_b: int) -> bool:
        """Unconditional gender-conflict block — see attributes_have_strong_gender_conflict."""
        docs = await self._db[self.PERSONS].find(
            {"person_id": {"$in": [person_a, person_b]}},
            {"_id": 0, "person_id": 1, "attributes": 1},
        ).to_list(length=2)
        attrs_by_pid = {int(doc.get("person_id")): doc.get("attributes") or {} for doc in docs}
        attrs_a = attrs_by_pid.get(person_a, {})
        attrs_b = attrs_by_pid.get(person_b, {})
        return self.attributes_have_strong_gender_conflict(attrs_a, attrs_b)

    # kiểm tra disagreement gender theo nhiều sighting liên tiếp
    async def persons_have_clear_gender_disagreement(
        self,
        person_a: int,
        person_b: int,
        sighting_confidence_threshold: float = 0.90,
        min_consecutive: int = 2,
    ) -> bool:
        """Tracklet-level gender disagreement with hysteresis (PDF Bước 6).

        A person's gender is "committed" only when the most recent
        ``min_consecutive`` high-confidence tracklet sightings (sorted by
        ``started_at``) all agree on the same label. A single bad tracklet
        cannot poison the canonical label — it requires two consecutive
        confident tracklets to "flip" or "commit" a gender, per the PDF.

        Two persons disagree only if BOTH sides are committed to opposite
        labels. If either side is not yet committed (mixed history or only
        one supporting tracklet), the embedding-similarity vote decides.

        Note: each ``add_sighting`` writes one doc per tracklet, so sightings
        ARE tracklet-level; no extra collection or schema change needed.
        """
        sightings = await self._db[self.SIGHTINGS].find(
            {"person_id": {"$in": [person_a, person_b]}},
            {
                "_id": 0,
                "person_id": 1,
                "started_at": 1,
                "attributes.gender": 1,
                "attributes.gender_confidence": 1,
            },
        ).to_list(length=None)

        def _committed_label(pid: int) -> str | None:
            confident = []
            for s in sightings:
                if int(s.get("person_id", -1)) != pid:
                    continue
                attrs = s.get("attributes") or {}
                label = attrs.get("gender")
                conf = float(attrs.get("gender_confidence", 0.0) or 0.0)
                if not label or label == "unknown":
                    continue
                if conf < sighting_confidence_threshold:
                    continue
                confident.append((s.get("started_at"), label))
            if len(confident) < min_consecutive:
                return None
            # Sort by start time; walk from most recent back, counting the
            # current streak. If the streak reaches min_consecutive, that's
            # the committed label.
            confident.sort(key=lambda x: x[0] or 0)
            recent_label = confident[-1][1]
            streak = 1
            for _, label in reversed(confident[:-1]):
                if label == recent_label:
                    streak += 1
                    if streak >= min_consecutive:
                        return recent_label
                else:
                    break
            return None

        label_a = _committed_label(person_a)
        label_b = _committed_label(person_b)
        if label_a is None or label_b is None:
            return False
        return label_a != label_b

    # lấy attributes của hai person trong một lần query
    async def fetch_two_persons_attributes(
        self, person_a: int, person_b: int
    ) -> tuple[dict, dict]:
        """Single round-trip fetch — caller runs the (cheap) conflict checks
        against the returned attrs without re-querying.
        """
        docs = await self._db[self.PERSONS].find(
            {"person_id": {"$in": [person_a, person_b]}},
            {"_id": 0, "person_id": 1, "attributes": 1},
        ).to_list(length=2)
        attrs_by_pid = {int(doc.get("person_id")): doc.get("attributes") or {} for doc in docs}
        return attrs_by_pid.get(person_a, {}), attrs_by_pid.get(person_b, {})

    # kiểm tra xung đột thuộc tính mức vừa để dùng như tín hiệu mềm
    async def persons_have_moderate_attribute_conflict(self, person_a: int, person_b: int) -> bool:
        """Moderate version — see attributes_have_moderate_conflict.

        Used as a soft-split detection signal, not as a hard blocker.
        """
        docs = await self._db[self.PERSONS].find(
            {"person_id": {"$in": [person_a, person_b]}},
            {"_id": 0, "person_id": 1, "attributes": 1},
        ).to_list(length=2)
        attrs_by_pid = {int(doc.get("person_id")): doc.get("attributes") or {} for doc in docs}
        attrs_a = attrs_by_pid.get(person_a, {})
        attrs_b = attrs_by_pid.get(person_b, {})
        return self.attributes_have_moderate_conflict(attrs_a, attrs_b)

    # gộp source person vào target person trên các collection mongo
    async def merge_person(self, *, source_person_id: int, target_person_id: int, reason: dict) -> None:
        """Merge source identity into target across Mongo collections."""
        now = datetime.now(timezone.utc)
        source = await self._db[self.PERSONS].find_one({"person_id": source_person_id}, {"_id": 0})
        source_sightings = await self._db[self.SIGHTINGS].count_documents({"person_id": source_person_id})
        source_tracklets = await self._db[self.TRACKLETS].count_documents({"person_id": source_person_id})

        await self._db[self.TRACKLETS].update_many(
            {"person_id": source_person_id},
            {"$set": {"person_id": target_person_id, "merged_from_person_id": source_person_id}},
        )
        await self._db[self.SIGHTINGS].update_many(
            {"person_id": source_person_id},
            {"$set": {"person_id": target_person_id, "merged_from_person_id": source_person_id}},
        )
        await self._db[self.TIMELINE].update_many(
            {"person_id": source_person_id},
            {"$set": {"person_id": target_person_id, "merged_from_person_id": source_person_id}},
        )

        update: dict = {
            "$set": {"updated_at": now},
            "$inc": {"stats.sighting_count": source_sightings or source_tracklets},
            "$push": {
                "merged_person_ids": source_person_id,
                "merge_events": {"source_person_id": source_person_id, "merged_at": now, **reason},
            },
        }
        if source and source.get("snapshot_key"):
            target = await self._db[self.PERSONS].find_one({"person_id": target_person_id}, {"_id": 0})
            source_score = float(((source.get("stats") or {}).get("best_snapshot_score")) or 0.0)
            target_score = float((((target or {}).get("stats") or {}).get("best_snapshot_score")) or 0.0)
            if source_score > target_score:
                update["$set"]["snapshot_key"] = source.get("snapshot_key")
                update["$set"]["stats.best_snapshot_score"] = source_score

        await self._db[self.PERSONS].update_one({"person_id": target_person_id}, update)
        await self._db[self.PERSONS].delete_one({"person_id": source_person_id})

        # Recompute the target person's voted attributes from ALL sightings now
        # under that person_id, including the freshly-merged ones. Without
        # this, a merged identity keeps the attributes it had pre-merge, which
        # is wrong when the absorbed person's tracklets disagree — e.g., a
        # mostly-male identity that absorbed several confident-female sightings
        # would still display "male" until the next regular tracklet for that
        # person triggers an AttributeVoter update.
        await self._recompute_voted_attributes(target_person_id)

        await self.add_timeline_event(
            person_id=target_person_id,
            event_type="identity_merged",
            details={"source_person_id": source_person_id, **reason},
        )

    # tính lại attributes của person từ toàn bộ sightings bằng voting confidence
    async def _recompute_voted_attributes(self, person_id: int) -> None:
        """Confidence-weighted majority vote across all sightings of a person.

        For each task, sum sighting confidences per label and pick the label with
        the highest total support. Sets `attributes.<task>` and
        `attributes.<task>_confidence` on the person doc to reflect the new
        aggregate.
        """
        sightings = await self._db[self.SIGHTINGS].find(
            {"person_id": person_id}, {"_id": 0, "attributes": 1}
        ).to_list(length=None)
        if not sightings:
            return

        tasks = ("gender", "age_child", "backpack", "sidebag", "hat", "glasses", "sleeve", "lower")
        new_attrs: dict[str, tuple[str, float]] = {}
        for task in tasks:
            # support[label] = (sum_conf, count) — confidence-weighted vote
            support: dict[str, tuple[float, int]] = {}
            for s in sightings:
                attrs = s.get("attributes") or {}
                label = attrs.get(task)
                conf = float(attrs.get(f"{task}_confidence", 0.0) or 0.0)
                if not label or label == "unknown" or conf <= 0:
                    continue
                prev_sum, prev_count = support.get(label, (0.0, 0))
                support[label] = (prev_sum + conf, prev_count + 1)
            if not support:
                continue
            best_label, (best_sum, best_count) = max(support.items(), key=lambda kv: kv[1][0])
            new_attrs[task] = (best_label, best_sum / best_count)

        if not new_attrs:
            return
        attr_set: dict = {}
        for task, (label, conf) in new_attrs.items():
            attr_set[f"attributes.{task}"] = label
            attr_set[f"attributes.{task}_confidence"] = float(round(conf, 4))
        await self._db[self.PERSONS].update_one(
            {"person_id": person_id}, {"$set": attr_set}
        )

    # refresh attributes person từ evidence sighting đã xác nhận
    async def recompute_person_attributes(self, person_id: int) -> None:
        """Refresh person-level attributes from confirmed sighting evidence."""
        await self._recompute_voted_attributes(person_id)

    # đóng kết nối mongodb
    def close(self) -> None:
        self._client.close()
