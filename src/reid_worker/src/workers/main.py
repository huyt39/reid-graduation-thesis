from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import cv2
import numpy as np
import structlog

from src.attributes.gender_voter import GenderVoter
from src.core.config import settings
from src.embedding.aggregator import WeightedEmbeddingAggregator
from src.embedding.client import ModelServiceClient
from src.kafka.consumer import WorkerKafkaConsumer
from src.kafka.producer import WorkerKafkaProducer
from src.matching.qdrant_store import QdrantPersonStore
from src.matching.reid_matcher import ReIDMatcher
from src.persistence.minio_store import MinIOSnapshotStore
from src.persistence.mongo_store import MongoPersonStore
from src.persistence.redis_cache import RedisPersonCache
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
        )
        self.matcher = ReIDMatcher(
            self.qdrant_store,
            promote_v_threshold=self.settings.promote_v_threshold,
            promote_consistency_threshold=self.settings.promote_consistency_threshold,
            update_v_threshold=self.settings.update_v_threshold,
            update_consistency_threshold=self.settings.update_consistency_threshold,
            update_min_tracklet_len=self.settings.update_min_tracklet_len,
            update_sim_threshold=self.settings.update_sim_threshold,
        )
        self.model_client = ModelServiceClient(base_url=self.settings.model_service_url)

        # Persistence
        self.mongo = MongoPersonStore(uri=self.settings.mongo_uri, db_name=self.settings.mongo_db)
        self.redis_cache = RedisPersonCache(url=self.settings.redis_url)
        self.minio = MinIOSnapshotStore(
            endpoint=self.settings.minio_endpoint,
            access_key=self.settings.minio_access_key,
            secret_key=self.settings.minio_secret_key,
        )
        self.gender_voter = GenderVoter(person_threshold=self.settings.gender_person_threshold)

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
        self._current_device_id: str = ""

    def run(self) -> None:
        log.info("worker_started", service=self.settings.service_name)
        try:
            asyncio.run(self._run_loop())
        finally:
            self.consumer.close()
            self.producer.close()

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
        if len(track_results) == 0:
            return

        for track in track_results:
            bbox_xyxy = track[:4]
            track_id = int(track[4])
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

        current_time_ns = time.time_ns()
        for tracklet in self.tracklet_buffer.get_ready_tracklets(current_time_ns):
            await self._process_tracklet(tracklet)
        self.tracklet_buffer.evict_stale(current_time_ns)

        tracked_persons = []
        for track in track_results:
            track_id = int(track[4])
            person_id = self.track_id_to_person_id.get(track_id)
            if person_id is None:
                continue
            # Get person-level gender from voter
            p_hist = self.gender_voter._person_history.get(person_id)
            gender = p_hist.current_gender if p_hist else "unknown"
            gender_conf = p_hist.current_confidence if p_hist else 0.0
            meta = self.track_metadata.get(track_id, {})

            tracked_persons.append(
                {
                    "person_id": person_id,
                    "bbox": [float(v) for v in track[:4].tolist()],
                    "confidence": float(track[5]),
                    "gender": gender,
                    "gender_confidence": float(gender_conf),
                    "tracklet_id": meta.get("tracklet_id"),
                    "tracklet_state": meta.get("tracklet_state"),
                    "visibility_score": float(meta.get("visibility_score", 0.0)),
                    "quality": meta.get("quality"),
                    "attributes": meta.get("attributes"),
                }
            )

        self.producer.send(
            device_id=device_id,
            frame_number=frame_number,
            tracked_persons=tracked_persons,
            image_data=image_data,
            timestamp_ns=timestamp_ns,
        )

    async def _process_tracklet(self, tracklet) -> None:
        if not self.topk_selector.is_tracklet_ready(tracklet.entries):
            tracklet.state = TrackletState.TENTATIVE
            return
        consistency = compute_tracklet_consistency(tracklet.entries)
        selected = self.topk_selector.select(tracklet.entries)

        embeddings, v_scores, overlap_ratios = [], [], []
        gender_results: list[dict] = []
        best_entry = selected[0] if selected else None

        for entry in selected:
            ok, buf = cv2.imencode(".jpg", entry.crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            img_bytes = buf.tobytes()
            try:
                _, result = await self.model_client.extract_features(img_bytes)
                embedding = np.array(result["embedding"], dtype=np.float32)
                norm = np.linalg.norm(embedding)
                if norm > 1e-8:
                    embedding = embedding / norm
                embeddings.append(embedding)
                v_scores.append(entry.v_score)
                overlap_ratios.append(entry.overlap_ratio)
                # Track best entry by v_score
                if best_entry is None or entry.v_score > best_entry.v_score:
                    best_entry = entry
            except Exception as err:
                log.warning("feature_extraction_failed", error=str(err))

            # Gender classification (fire-and-forget per frame)
            try:
                gr = await self.model_client.classify_gender(img_bytes)
                gender_results.append(gr)
                self.gender_voter.vote_frame(tracklet.track_id, gr["gender"], gr["confidence"])
            except Exception:
                pass  # gender is optional

        if not embeddings:
            return

        emb_consistency = WeightedEmbeddingAggregator.compute_embedding_consistency(embeddings)
        tracklet_embedding = self.aggregator.aggregate(embeddings, v_scores, overlap_ratios)
        v_avg = sum(v_scores) / len(v_scores)

        # Resolve tracklet-level gender
        t_gender, t_gender_conf = self.gender_voter.resolve_tracklet(tracklet.track_id)

        person_id = self.matcher.match_tracklet(
            track_id=tracklet.track_id,
            embedding=tracklet_embedding,
            v_avg=v_avg,
            embedding_consistency=emb_consistency,
            tracklet_len=len(tracklet.entries),
        )

        tracklet_id = str(uuid.uuid4())

        if person_id is not None:
            tracklet.person_id = person_id
            tracklet.state = TrackletState.MATCHED
            self.track_id_to_person_id[tracklet.track_id] = person_id

            # Resolve person-level gender with hysteresis
            p_gender, p_gender_conf = self.gender_voter.resolve_person(
                person_id, t_gender, t_gender_conf,
            )
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
                "attributes": {
                    "gender": p_gender,
                },
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
                    gender=p_gender,
                    gender_conf=p_gender_conf,
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
            )
        self.tracklet_buffer.remove(tracklet.track_id)

    async def _persist_tracklet(
        self, *, tracklet, tracklet_id, person_id, consistency,
        v_avg, emb_consistency, best_entry, gender, gender_conf,
    ) -> None:
        """Write to MongoDB, MinIO, and invalidate Redis — all async."""
        device_id = self._current_device_id
        now = datetime.now(timezone.utc)
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
                gender=gender,
                gender_confidence=gender_conf,
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
                gender=gender,
                gender_confidence=gender_conf,
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
