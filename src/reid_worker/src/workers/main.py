from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import cv2
import numpy as np
import structlog

from src.attributes.attribute_voter import AttributeVoter
from src.core.config import settings
from src.embedding.aggregator import WeightedEmbeddingAggregator
from src.embedding.client import ModelServiceClient
from src.kafka.consumer import WorkerKafkaConsumer
from src.kafka.producer import WorkerKafkaProducer
from src.matching.qdrant_store import QdrantPersonStore
from src.matching.reid_matcher import ReIDMatcher, PersonIdAllocationError
from src.persistence.minio_store import MinIOSnapshotStore
from src.persistence.mongo_store import MongoPersonStore
from src.persistence.redis_cache import RedisPersonCache, RedisPersonIdAllocator
from src.scoring.enhanced_visibility import (
    compute_iou_prev,
    compute_vel_smooth,
    compute_v_worker,
)
from src.tracking.byte_tracker import BYTETracker
from src.tracklet.buffer import TrackletBuffer
from src.tracklet.consistency import compute_tracklet_consistency
from src.tracklet.models import TrackletEntry, TrackletState
from src.tracklet.selector import TopKSelector
from src.utils.ops import xyxy2xywh

log = structlog.get_logger()

# Tasks emitted on every TrackedPerson — must match Avro schema field names.
_ATTRIBUTE_TASKS = ("gender", "age_child", "backpack", "sidebag",
                    "hat", "glasses", "sleeve", "lower")


class WorkerPipeline:
    def __init__(self) -> None:
        self.settings = settings
        tracker_args = SimpleNamespace(
            track_high_thresh=self.settings.track_high_thresh,
            track_low_thresh=self.settings.track_low_thresh,
            match_thresh=self.settings.match_thresh,
            new_track_thresh=self.settings.new_track_thresh,
            track_buffer=self.settings.track_buffer,
            fuse_score=self.settings.fuse_score,
        )
        self.tracker = BYTETracker(tracker_args, frame_rate=30)
        self.tracklet_buffer = TrackletBuffer(
            min_entries=self.settings.tracklet_min_entries,
            max_entries=self.settings.tracklet_max_entries,
            window_seconds=self.settings.tracklet_window_seconds,
            stale_seconds=self.settings.tracklet_stale_seconds,
        )
        self.topk_selector = TopKSelector(
            k=self.settings.topk_k,
            min_temporal_gap=self.settings.topk_min_temporal_gap,
            overlap_lambda=self.settings.overlap_lambda,
            min_tracklet_len=self.settings.tracklet_min_entries,
            min_high_quality_frames=self.settings.min_high_quality_frames,
            high_quality_threshold=self.settings.high_quality_threshold,
        )
        self.aggregator = WeightedEmbeddingAggregator(gamma=self.settings.gamma)
        self.qdrant_store = QdrantPersonStore(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
            embedding_dim=self.settings.embedding_dim,
            similarity_threshold=self.settings.similarity_threshold,
            momentum=self.settings.momentum,
            max_gallery_size=self.settings.max_gallery_size,
        )
        self.model_client = ModelServiceClient(base_url=self.settings.model_service_url)

        # Persistence
        self.mongo = MongoPersonStore(uri=self.settings.mongo_uri, db_name=self.settings.mongo_db)
        self.redis_cache = RedisPersonCache(url=self.settings.redis_url)
        self.person_id_allocator = RedisPersonIdAllocator(
            url=self.settings.redis_url,
            key=self.settings.person_id_seq_key,
        )
        self.minio = MinIOSnapshotStore(
            endpoint=self.settings.minio_endpoint,
            access_key=self.settings.minio_access_key,
            secret_key=self.settings.minio_secret_key,
        )
        self.attribute_voter = AttributeVoter(
            person_threshold=self.settings.gender_person_threshold,
            flip_threshold=self.settings.gender_flip_threshold,
        )
        self.matcher = ReIDMatcher(
            self.qdrant_store,
            id_allocator=self.person_id_allocator.allocate,
            promote_v_threshold=self.settings.promote_v_threshold,
            promote_consistency_threshold=self.settings.promote_consistency_threshold,
            tentative_max_attempts=self.settings.tentative_max_attempts,
            tentative_fallback_enabled=self.settings.tentative_fallback_enabled,
            update_v_threshold=self.settings.update_v_threshold,
            update_consistency_threshold=self.settings.update_consistency_threshold,
            update_min_tracklet_len=self.settings.update_min_tracklet_len,
            update_sim_threshold=self.settings.update_sim_threshold,
            match_margin=self.settings.match_margin,
            spatial_reuse_threshold=self.settings.spatial_reuse_threshold,
            soft_match_threshold=self.settings.soft_match_threshold,
        )

        self.consumer = WorkerKafkaConsumer(
            bootstrap_servers=self.settings.kafka_bootstrap_servers,
            topic=self.settings.input_topic,
            group_id=self.settings.consumer_group,
            schema_path=self.settings.schema_path,
        )
        self.producer = WorkerKafkaProducer(
            bootstrap_servers=self.settings.kafka_bootstrap_servers,
            topic=self.settings.output_topic,
            schema_path=self.settings.output_schema_path,
        )
        self.prev_bboxes: dict[int, list[np.ndarray]] = {}
        self.track_id_to_person_id: dict[int, int] = {}
        self.track_metadata: dict[int, dict] = {}
        self.track_last_seen_ns: dict[int, int] = {}
        self.person_last_observation: dict[int, dict] = {}
        self._current_device_id: str = ""
        self.processed_messages = 0
        self.ready_tracklets = 0
        self.embedded_tracklets = 0
        self.matched_tracklets = 0
        self.worker_started_at = time.time()

    def _cleanup_inactive_tracks(self, current_time_ns: int) -> None:
        stale_after_ns = int(self.settings.tracklet_stale_seconds * 1e9)
        stale_track_ids = {
            track_id
            for track_id, last_seen_ns in self.track_last_seen_ns.items()
            if current_time_ns - last_seen_ns > stale_after_ns
        }

        for track_id in stale_track_ids:
            self.prev_bboxes.pop(track_id, None)
            self.track_id_to_person_id.pop(track_id, None)
            self.track_metadata.pop(track_id, None)
            self.track_last_seen_ns.pop(track_id, None)
        stale_person_ids = {
            person_id
            for person_id, obs in self.person_last_observation.items()
            if current_time_ns - int(obs["timestamp_ns"]) > stale_after_ns
        }
        for person_id in stale_person_ids:
            self.person_last_observation.pop(person_id, None)

    @staticmethod
    def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union

    @staticmethod
    def _center_distance_ratio(box_a: list[float], box_b: list[float]) -> float:
        center_a = np.array([(box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2], dtype=np.float32)
        center_b = np.array([(box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2], dtype=np.float32)
        distance = float(np.linalg.norm(center_a - center_b))
        size_a = max(box_a[2] - box_a[0], box_a[3] - box_a[1], 1.0)
        size_b = max(box_b[2] - box_b[0], box_b[3] - box_b[1], 1.0)
        return distance / max(size_a, size_b, 1.0)

    def _find_recent_person_hint(
        self,
        bbox_xyxy: list[float],
        current_time_ns: int,
        blocked_person_ids: set[int],
    ) -> int | None:
        if not self.settings.recent_person_reuse_enabled:
            return None

        best_person_id = None
        best_score = -1.0
        max_gap_ns = int(self.settings.recent_person_reuse_seconds * 1e9)
        for person_id, obs in self.person_last_observation.items():
            if person_id in blocked_person_ids:
                continue
            gap_ns = current_time_ns - int(obs["timestamp_ns"])
            if gap_ns < 0 or gap_ns > max_gap_ns:
                continue

            obs_bbox = obs["bbox_xyxy"]
            iou = self._bbox_iou(bbox_xyxy, obs_bbox)
            center_ratio = self._center_distance_ratio(bbox_xyxy, obs_bbox)
            if (
                iou < self.settings.recent_person_reuse_min_iou
                and center_ratio > self.settings.recent_person_reuse_max_center_distance_ratio
            ):
                continue

            score = iou - 0.1 * center_ratio
            if score > best_score:
                best_person_id = person_id
                best_score = score

        return best_person_id

    def run(self) -> None:
        log.info("worker_started", service=self.settings.service_name)
        try:
            asyncio.run(self._run_loop())
        finally:
            self.consumer.close()
            self.producer.close()
            self.person_id_allocator.close()

    async def _run_loop(self) -> None:
        await self.mongo.ensure_indexes()
        async with self.model_client:
            while True:
                messages = self.consumer.poll(timeout_ms=1000)
                if not messages:
                    continue
                for msg in messages:
                    await self._process_message(msg)
                await asyncio.sleep(self.settings.poll_interval_s)

    async def _process_message(self, msg: dict) -> None:
        self.processed_messages += 1
        device_id = msg["device_id"]
        self._current_device_id = device_id
        frame_number = msg["frame_number"]
        detections = msg["detections"]
        image_data = msg["image_data"]
        timestamp_ns = msg["created_at"]
        img_array = np.frombuffer(image_data, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None or not detections:
            return

        frame_h, frame_w = frame.shape[:2]
        bboxes, scores, classes, det_v_edges, det_overlap_ratios = [], [], [], [], []
        for det in detections:
            bboxes.append(xyxy2xywh(np.array(det["bbox"])))
            scores.append(det["confidence"])
            classes.append(det["class_id"])
            det_v_edges.append(det["visibility_score"])
            det_overlap_ratios.append(det.get("overlap_ratio", 0.0))

        track_results = self.tracker.update(
            np.array(scores, dtype=np.float32),
            np.array(bboxes, dtype=np.float32),
            np.array(classes, dtype=np.float32),
            frame,
        )
        current_time_ns = time.time_ns()
        if len(track_results) == 0:
            self._cleanup_inactive_tracks(current_time_ns)
            return

        visible_track_ids = {int(track[4]) for track in track_results}

        for track in track_results:
            bbox_xyxy = track[:4]
            track_id = int(track[4])
            self.track_last_seen_ns[track_id] = current_time_ns
            v_edge = 0.5
            overlap_ratio = 0.0
            if det_v_edges:
                min_dist = float("inf")
                best_idx = 0
                for i, det in enumerate(detections):
                    dist = np.linalg.norm(bbox_xyxy - np.array(det["bbox"]))
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
                v_edge = det_v_edges[best_idx]
                overlap_ratio = det_overlap_ratios[best_idx]

            prev_list = self.prev_bboxes.get(track_id, [])
            bbox_prev = prev_list[-1] if len(prev_list) >= 1 else None
            center_curr = np.array([(bbox_xyxy[0] + bbox_xyxy[2]) / 2, (bbox_xyxy[1] + bbox_xyxy[3]) / 2])
            center_prev = None
            if bbox_prev is not None:
                center_prev = np.array([(bbox_prev[0] + bbox_prev[2]) / 2, (bbox_prev[1] + bbox_prev[3]) / 2])
            bbox_prev2_center = None
            if len(prev_list) >= 2:
                bp2 = prev_list[-2]
                bbox_prev2_center = np.array([(bp2[0] + bp2[2]) / 2, (bp2[1] + bp2[3]) / 2])

            bbox_size = max(bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1])
            v_worker = compute_v_worker(
                v_edge,
                compute_iou_prev(bbox_xyxy, bbox_prev),
                compute_vel_smooth(center_curr, center_prev, bbox_prev2_center, bbox_size),
            )
            self.prev_bboxes.setdefault(track_id, []).append(bbox_xyxy.copy())
            self.prev_bboxes[track_id] = self.prev_bboxes[track_id][-3:]

            x1, y1, x2, y2 = map(int, bbox_xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            self.tracklet_buffer.append(
                track_id,
                TrackletEntry(
                    frame_idx=frame_number,
                    crop=crop,
                    v_score=v_worker,
                    bbox_xyxy=bbox_xyxy.tolist(),
                    timestamp_ns=timestamp_ns,
                    overlap_ratio=overlap_ratio,
                ),
            )

        ready_tracklets = self.tracklet_buffer.get_ready_tracklets(current_time_ns)
        self.ready_tracklets += len(ready_tracklets)
        self.tracklet_buffer.evict_stale(current_time_ns)
        self._cleanup_inactive_tracks(current_time_ns)

        frame_reserved_person_ids = {
            pid
            for tid, pid in self.track_id_to_person_id.items()
            if tid in visible_track_ids
        }

        # Emit output immediately using currently-known person_ids so the frame
        # is not delayed by embedding HTTP calls. Tentative tracks (no person_id
        # yet) are included with their track_id as a temporary display id so the
        # UI shows bounding boxes from the first detected frame.
        tracked_persons = []
        emitted_person_ids: set[int] = set()
        for track in track_results:
            track_id = int(track[4])
            person_id = self.track_id_to_person_id.get(track_id)
            is_tentative = person_id is None
            display_id = track_id if is_tentative else person_id

            if display_id in emitted_person_ids:
                if not is_tentative:
                    log.warning(
                        "duplicate_person_id_suppressed",
                        frame_number=frame_number,
                        track_id=track_id,
                        person_id=display_id,
                    )
                continue
            emitted_person_ids.add(display_id)

            if is_tentative:
                t = self.tracklet_buffer.tracklets.get(track_id)
                log.warning(
                    "tentative_in_frame",
                    track_id=track_id,
                    buffer_entries=len(t.entries) if t else 0,
                    state=t.state.value if t else "not_in_buffer",
                )
                payload = {
                    "person_id": display_id,
                    "bbox": [float(v) for v in track[:4].tolist()],
                    "confidence": float(track[5]),
                    "tracklet_id": None,
                    "tracklet_state": "tentative",
                    "visibility_score": 0.0,
                    "quality": None,
                    "attributes": None,
                }
                for task in _ATTRIBUTE_TASKS:
                    payload[task] = "unknown"
                    payload[f"{task}_confidence"] = 0.0
            else:
                self.person_last_observation[person_id] = {
                    "bbox_xyxy": [float(v) for v in track[:4].tolist()],
                    "timestamp_ns": current_time_ns,
                    "device_id": device_id,
                }
                person_attrs = self.attribute_voter.person_snapshot(person_id)
                meta = self.track_metadata.get(track_id, {})
                payload = {
                    "person_id": person_id,
                    "bbox": [float(v) for v in track[:4].tolist()],
                    "confidence": float(track[5]),
                    "tracklet_id": meta.get("tracklet_id"),
                    "tracklet_state": meta.get("tracklet_state"),
                    "visibility_score": float(meta.get("visibility_score", 0.0)),
                    "quality": meta.get("quality"),
                    "attributes": meta.get("attributes"),
                }
                for task in _ATTRIBUTE_TASKS:
                    label, conf = person_attrs.get(task, ("unknown", 0.0))
                    payload[task] = label
                    payload[f"{task}_confidence"] = float(conf)

            tracked_persons.append(payload)

        self.producer.send(
            device_id=device_id,
            frame_number=frame_number,
            tracked_persons=tracked_persons,
            image_data=image_data,
            timestamp_ns=timestamp_ns,
        )
        self._maybe_log_progress(frame_number)

        # Process ready tracklets AFTER output is already sent, as background
        # tasks so the Kafka poll loop is not blocked by embedding HTTP calls.
        if ready_tracklets:
            # Shared mutable set: each background task adds its assigned person_id so
            # subsequent tasks see it as blocked (asyncio is single-threaded, so
            # match_tracklet runs atomically between await points — no lock needed).
            for tracklet in ready_tracklets:
                asyncio.ensure_future(
                    self._process_tracklet(
                        tracklet,
                        reserved_person_ids=frame_reserved_person_ids,
                    )
                )

    def _maybe_log_progress(self, frame_number: int) -> None:
        every_n = self.settings.log_every_n_messages
        if every_n <= 0 or self.processed_messages % every_n != 0:
            return

        elapsed = max(time.time() - self.worker_started_at, 1e-6)
        log.info(
            "worker_progress",
            processed_messages=self.processed_messages,
            frame_number=frame_number,
            messages_per_second=round(self.processed_messages / elapsed, 2),
            ready_tracklets=self.ready_tracklets,
            embedded_tracklets=self.embedded_tracklets,
            matched_tracklets=self.matched_tracklets,
            active_tracks=len(self.track_last_seen_ns),
            identified_tracks=len(self.track_id_to_person_id),
        )

    async def _process_tracklet(
        self,
        tracklet,
        reserved_person_ids: set[int] | None = None,
    ) -> int | None:
        if not self.topk_selector.is_tracklet_ready(tracklet.entries):
            tracklet.state = TrackletState.ACTIVE  # allow re-evaluation as new frames arrive
            recent_v = [round(e.v_score, 3) for e in tracklet.entries[-5:]]
            log.warning("tracklet_quality_gate_fail",
                        track_id=tracklet.track_id,
                        entries=len(tracklet.entries),
                        recent_v_scores=recent_v,
                        threshold=getattr(self.topk_selector, "high_quality_threshold", None))
            return None
        consistency = compute_tracklet_consistency(tracklet.entries)
        selected = self.topk_selector.select(tracklet.entries)

        embeddings, v_scores, overlap_ratios = [], [], []
        best_entry = selected[0] if selected else None

        async def _extract_one(entry):
            ok, buf = cv2.imencode(".jpg", entry.crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return None, None
            img_bytes = buf.tobytes()
            emb_vec = None
            try:
                _, result = await self.model_client.extract_features(img_bytes)
                emb_vec = np.array(result["embedding"], dtype=np.float32)
                norm = np.linalg.norm(emb_vec)
                if norm > 1e-8:
                    emb_vec /= norm
                else:
                    emb_vec = None
            except Exception as err:
                log.warning("feature_extraction_failed", error=str(err))
            try:
                attrs = await self.model_client.classify_attributes(img_bytes)
                self.attribute_voter.vote_frame(tracklet.track_id, attrs)
            except Exception:
                pass
            return emb_vec, entry

        for emb_vec, entry in await asyncio.gather(*[_extract_one(e) for e in selected]):
            if emb_vec is None:
                continue
            embeddings.append(emb_vec)
            v_scores.append(entry.v_score)
            overlap_ratios.append(entry.overlap_ratio)
            if best_entry is None or entry.v_score > best_entry.v_score:
                best_entry = entry

        if not embeddings:
            log.warning("tracklet_no_embeddings", track_id=tracklet.track_id, selected=len(selected))
            return

        self.embedded_tracklets += 1
        emb_consistency = WeightedEmbeddingAggregator.compute_embedding_consistency(embeddings)
        tracklet_embedding = self.aggregator.aggregate(embeddings, v_scores, overlap_ratios)
        v_avg = sum(v_scores) / len(v_scores)

        # Resolve tracklet-level attributes (all 8 tasks).
        t_attrs = self.attribute_voter.resolve_tracklet(tracklet.track_id)

        try:
            if reserved_person_ids is None:
                reserved_person_ids = set()
            current_person_id = self.track_id_to_person_id.get(tracklet.track_id)
            blocked_person_ids = set(reserved_person_ids)
            if current_person_id is not None:
                blocked_person_ids.discard(current_person_id)
            reuse_person_id = None
            if current_person_id is None and tracklet.entries:
                reuse_person_id = self._find_recent_person_hint(
                    tracklet.entries[-1].bbox_xyxy,
                    tracklet.entries[-1].timestamp_ns,
                    blocked_person_ids,
                )

            person_id = self.matcher.match_tracklet(
                track_id=tracklet.track_id,
                embedding=tracklet_embedding,
                v_avg=v_avg,
                embedding_consistency=emb_consistency,
                tracklet_len=len(tracklet.entries),
                blocked_person_ids=blocked_person_ids,
                current_person_id=current_person_id,
                reuse_person_id=reuse_person_id,
            )
        except PersonIdAllocationError:
            log.error("person_id_allocation_failed", track_id=tracklet.track_id, exc_info=True)
            self.tracklet_buffer.remove(tracklet.track_id)
            return None

        if (
            person_id is not None
            and current_person_id != person_id
            and person_id in reserved_person_ids
        ):
            # Score ≥ 0.90 bypass in match_tracklet returned this person_id despite it being
            # reserved — the two ByteTracker track_ids represent the same physical person.
            # Assign the person_id and let the output loop suppress the duplicate bbox.
            # Removing the tracklet here causes an infinite retry loop (ByteTracker keeps
            # the track_id alive, buffer restarts at entries=1, collision fires again every cycle).
            log.warning(
                "reserved_person_id_collision",
                track_id=tracklet.track_id,
                person_id=person_id,
            )
            self.track_id_to_person_id[tracklet.track_id] = person_id
            tracklet.state = TrackletState.MATCHED
            return person_id


        tracklet_id = str(uuid.uuid4())

        if person_id is not None:
            self.matched_tracklets += 1
            tracklet.person_id = person_id
            tracklet.state = TrackletState.MATCHED
            self.track_id_to_person_id[tracklet.track_id] = person_id
            reserved_person_ids.add(person_id)  # make visible to other concurrent tasks

            # Resolve person-level attributes with per-task hysteresis.
            p_attrs = self.attribute_voter.resolve_person(person_id, t_attrs)
            p_gender, p_gender_conf = p_attrs.get("gender", ("unknown", 0.0))
            self.track_metadata[tracklet.track_id] = {
                "tracklet_id": tracklet_id,
                "tracklet_state": tracklet.state.value,
                "visibility_score": round(v_avg, 4),
                "quality": {
                    "v_avg": float(round(v_avg, 4)),
                    "embedding_consistency": float(round(emb_consistency, 4)),
                    "overall_consistency": float(round(consistency.overall, 4)),
                    "good_frame_ratio": float(round(consistency.good_frame_ratio, 4)),
                },
                # Compact label-only summary for the optional Avro `attributes` map.
                "attributes": {task: label for task, (label, _) in p_attrs.items()},
            }

            # ── Persistence (fire-and-forget) ─────────────────────────
            try:
                await self._persist_tracklet(
                    tracklet=tracklet,
                    tracklet_id=tracklet_id,
                    person_id=person_id,
                    consistency=consistency,
                    v_avg=v_avg,
                    emb_consistency=emb_consistency,
                    best_entry=best_entry,
                    person_attrs=p_attrs,
                )
            except Exception:
                log.error("persistence_failed", tracklet_id=tracklet_id, exc_info=True)

            log.info(
                "tracklet_matched",
                track_id=tracklet.track_id,
                person_id=person_id,
                v_avg=round(v_avg, 4),
                consistency=consistency.overall,
                gender=p_gender,
                gender_conf=round(p_gender_conf, 3),
            )
        else:
            # match_tracklet returned None (still tentative) — reset to ACTIVE so
            # get_ready_tracklets() re-queues this tracklet on the next frame and
            # tentative attempt count increments until fallback fires.
            tracklet.state = TrackletState.ACTIVE
            tent = getattr(self.matcher, "tentative", {}).get(tracklet.track_id, {})
            log.warning("tracklet_tentative_pending",
                        track_id=tracklet.track_id,
                        attempts=tent.get("attempts", 0),
                        v_avg=round(v_avg, 4),
                        consistency=round(emb_consistency, 4))

        return person_id

    async def _persist_tracklet(
        self, *, tracklet, tracklet_id, person_id, consistency,
        v_avg, emb_consistency, best_entry, person_attrs,
    ) -> None:
        """Write to MongoDB, MinIO, and invalidate Redis — all async.

        ``person_attrs`` is the per-task ``{task: (label, confidence)}`` snapshot
        from the AttributeVoter, written to ``persons.attributes.*`` and
        ``sightings.attributes`` so the query service can filter on any of them.
        """
        device_id = self._current_device_id
        entries = tracklet.entries
        started_at = datetime.fromtimestamp(entries[0].timestamp_ns / 1e9, tz=timezone.utc)
        ended_at = datetime.fromtimestamp(entries[-1].timestamp_ns / 1e9, tz=timezone.utc)

        # Upload best crop to MinIO
        crop_key = ""
        if best_entry is not None and best_entry.crop.size > 0:
            ok, buf = cv2.imencode(".jpg", best_entry.crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                crop_key = await asyncio.to_thread(
                    self.minio.upload_tracklet_snapshot, tracklet_id, buf.tobytes(),
                )

        quality = {
            "v_avg": round(v_avg, 4),
            "embedding_consistency": round(emb_consistency, 4),
            "bbox_size_stability": round(consistency.bbox_size_stability, 4),
            "position_stability": round(consistency.position_stability, 4),
            "good_frame_ratio": round(consistency.good_frame_ratio, 4),
            "overall_consistency": round(consistency.overall, 4),
        }

        # All writes in parallel
        await asyncio.gather(
            self.mongo.add_tracklet_record(
                tracklet_id=tracklet_id,
                track_id=tracklet.track_id,
                person_id=person_id,
                device_id=device_id,
                state=tracklet.state.value,
                frame_start=entries[0].frame_idx,
                frame_end=entries[-1].frame_idx,
                entry_count=len(entries),
                quality=quality,
                matching={"similarity_score": None, "was_promoted": False},
                best_crop_key=crop_key,
            ),
            self.mongo.upsert_person(
                person_id,
                attributes=person_attrs,
                device_id=device_id,
                snapshot_key=crop_key or None,
            ),
            self.mongo.add_sighting(
                person_id=person_id,
                device_id=device_id,
                tracklet_id=tracklet_id,
                started_at=started_at,
                ended_at=ended_at,
                entry_count=len(entries),
                quality_score=round(consistency.overall, 4),
                snapshot_key=crop_key or None,
                attributes=person_attrs,
            ),
            self.mongo.add_timeline_event(
                person_id=person_id,
                event_type="sighting_start",
                device_id=device_id,
                details={"tracklet_id": tracklet_id, "quality_score": round(consistency.overall, 4)},
            ),
            self.redis_cache.invalidate(person_id),
            return_exceptions=True,
        )


def run() -> None:
    WorkerPipeline().run()
