import asyncio
import time
from types import SimpleNamespace

import cv2
import numpy as np
from dotenv import load_dotenv

from src.config import WorkerSettings
from src.utils.logger import Logger
from src.utils.ops import crop_image, xyxy2xywh
from src.scoring.enhanced_visibility import compute_iou_prev, compute_vel_smooth, compute_v_worker
from src.tracking.byte_tracker import BYTETracker
from src.tracklet.buffer import TrackletBuffer
from src.tracklet.consistency import compute_tracklet_consistency
from src.tracklet.models import TrackletEntry, TrackletState
from src.tracklet.selector import TopKSelector
from src.embedding.client import ModelServiceClient
from src.embedding.aggregator import WeightedEmbeddingAggregator
from src.matching.qdrant_store import QdrantPersonStore
from src.matching.reid_matcher import ReIDMatcher
from src.kafka.consumer import WorkerKafkaConsumer

load_dotenv()
logger = Logger("worker")


class WorkerPipeline:
    def __init__(self, settings: WorkerSettings | None = None):
        self.settings = settings or WorkerSettings()

        # Tracker
        tracker_args = SimpleNamespace(
            track_high_thresh=self.settings.track_high_thresh,
            track_low_thresh=self.settings.track_low_thresh,
            match_thresh=self.settings.match_thresh,
            new_track_thresh=self.settings.new_track_thresh,
            track_buffer=self.settings.track_buffer,
            fuse_score=self.settings.fuse_score,
        )
        self.tracker = BYTETracker(tracker_args, frame_rate=30)

        # Tracklet management
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

        # Matching
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

        # Model client
        self.model_client = ModelServiceClient(
            base_url=self.settings.model_service_url,
        )

        # Kafka
        self.consumer = WorkerKafkaConsumer(
            bootstrap_servers=self.settings.kafka_bootstrap_servers,
            topic=self.settings.input_topic,
            group_id=self.settings.consumer_group,
            schema_path=self.settings.schema_path,
        )

        # Per-track previous bbox cache for enhanced scoring
        self.prev_bboxes: dict[int, list[np.ndarray]] = {}

    def run(self):
        logger.info("Worker pipeline starting...")
        asyncio.run(self._run_loop())

    async def _run_loop(self):
        async with self.model_client:
            logger.info("Connected to model service")
            while True:
                messages = self.consumer.poll(timeout_ms=1000)
                if not messages:
                    continue

                for msg in messages:
                    await self._process_message(msg)

    async def _process_message(self, msg: dict):
        device_id = msg["device_id"]
        frame_number = msg["frame_number"]
        detections = msg["detections"]
        image_data = msg["image_data"]
        timestamp_ns = msg["created_at"]

        # Decode image
        img_array = np.frombuffer(image_data, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning(f"Failed to decode frame {frame_number}")
            return

        frame_h, frame_w = frame.shape[:2]

        # Prepare detections for tracker
        if not detections:
            return

        bboxes = []
        scores = []
        classes = []
        det_v_edges = []
        det_overlap_ratios = []

        for det in detections:
            bbox_xyxy = det["bbox"]
            xywh = xyxy2xywh(np.array(bbox_xyxy))
            bboxes.append(xywh)
            scores.append(det["confidence"])
            classes.append(det["class_id"])
            det_v_edges.append(det["visibility_score"])
            det_overlap_ratios.append(det.get("overlap_ratio", 0.0))

        bboxes_np = np.array(bboxes, dtype=np.float32)
        scores_np = np.array(scores, dtype=np.float32)
        classes_np = np.array(classes, dtype=np.float32)

        # Run tracker
        track_results = self.tracker.update(scores_np, bboxes_np, classes_np, frame)

        if len(track_results) == 0:
            return

        # Process each tracked person
        for track in track_results:
            bbox_xyxy = track[:4]
            track_id = int(track[4])
            det_score = float(track[5])

            # Find matching detection (nearest bbox) for v_edge and overlap_ratio
            v_edge = 0.5
            overlap_ratio = 0.0
            if det_v_edges:
                min_dist = float("inf")
                best_idx = 0
                for i, det in enumerate(detections):
                    det_bbox = np.array(det["bbox"])
                    dist = np.linalg.norm(bbox_xyxy - det_bbox)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
                v_edge = det_v_edges[best_idx]
                overlap_ratio = det_overlap_ratios[best_idx]

            # Enhanced scoring
            prev_list = self.prev_bboxes.get(track_id, [])
            bbox_prev = prev_list[-1] if len(prev_list) >= 1 else None
            bbox_prev2_center = None
            center_curr = np.array([(bbox_xyxy[0] + bbox_xyxy[2]) / 2, (bbox_xyxy[1] + bbox_xyxy[3]) / 2])
            center_prev = None
            if bbox_prev is not None:
                center_prev = np.array([(bbox_prev[0] + bbox_prev[2]) / 2, (bbox_prev[1] + bbox_prev[3]) / 2])
            if len(prev_list) >= 2:
                bp2 = prev_list[-2]
                bbox_prev2_center = np.array([(bp2[0] + bp2[2]) / 2, (bp2[1] + bp2[3]) / 2])

            bbox_size = max(bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1])
            iou_score = compute_iou_prev(bbox_xyxy, bbox_prev)
            vel_score = compute_vel_smooth(center_curr, center_prev, bbox_prev2_center, bbox_size)
            v_worker = compute_v_worker(v_edge, iou_score, vel_score)

            # Update prev bbox cache
            if track_id not in self.prev_bboxes:
                self.prev_bboxes[track_id] = []
            self.prev_bboxes[track_id].append(bbox_xyxy.copy())
            if len(self.prev_bboxes[track_id]) > 3:
                self.prev_bboxes[track_id] = self.prev_bboxes[track_id][-3:]

            # Crop person image
            x1, y1, x2, y2 = map(int, bbox_xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Add to tracklet buffer (with overlap_ratio from edge)
            entry = TrackletEntry(
                frame_idx=frame_number,
                crop=crop,
                v_score=v_worker,
                bbox_xyxy=bbox_xyxy.tolist(),
                timestamp_ns=timestamp_ns,
                overlap_ratio=overlap_ratio,
            )
            self.tracklet_buffer.append(track_id, entry)

        # Check for ready tracklets
        current_time_ns = time.time_ns()
        ready_tracklets = self.tracklet_buffer.get_ready_tracklets(current_time_ns)

        for tracklet in ready_tracklets:
            await self._process_tracklet(tracklet)

        # Evict stale tracklets
        self.tracklet_buffer.evict_stale(current_time_ns)

    async def _process_tracklet(self, tracklet):
        """Select top-K frames, extract embeddings, aggregate, and match."""
        # Check tracklet quality gate
        if not self.topk_selector.is_tracklet_ready(tracklet.entries):
            logger.info(
                f"Tracklet {tracklet.track_id} not ready "
                f"(len={len(tracklet.entries)}, keeping tentative)"
            )
            tracklet.state = TrackletState.TENTATIVE
            return

        # Compute tracklet consistency
        consistency = compute_tracklet_consistency(tracklet.entries)
        logger.info(
            f"Tracklet {tracklet.track_id} consistency: "
            f"size={consistency.bbox_size_stability:.2f} pos={consistency.position_stability:.2f} "
            f"streak={consistency.good_frame_streak} ratio={consistency.good_frame_ratio:.2f} "
            f"overall={consistency.overall:.2f}"
        )

        # Select top-K frames (overlap-penalized scoring)
        selected = self.topk_selector.select(tracklet.entries)

        # Extract embeddings for selected frames
        embeddings = []
        v_scores = []
        overlap_ratios = []
        for entry in selected:
            _, buf = cv2.imencode(".jpg", entry.crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_bytes = buf.tobytes()
            try:
                _, result = await self.model_client.extract_features(image_bytes)
                embedding = np.array(result["embedding"], dtype=np.float32)
                # L2 normalize
                norm = np.linalg.norm(embedding)
                if norm > 1e-8:
                    embedding = embedding / norm
                embeddings.append(embedding)
                v_scores.append(entry.v_score)
                overlap_ratios.append(entry.overlap_ratio)
            except Exception as e:
                logger.error(f"Failed to extract features: {e}")
                continue

        if not embeddings:
            logger.warning(f"No embeddings extracted for tracklet {tracklet.track_id}")
            return

        # Compute embedding consistency
        emb_consistency = WeightedEmbeddingAggregator.compute_embedding_consistency(embeddings)
        logger.info(f"Tracklet {tracklet.track_id} embedding consistency: {emb_consistency:.3f}")

        # Aggregate embeddings (overlap-aware weights)
        tracklet_embedding = self.aggregator.aggregate(embeddings, v_scores, overlap_ratios)
        v_avg = sum(v_scores) / len(v_scores)

        # Match (with consistency and tracklet length for promote/update policies)
        person_id = self.matcher.match_tracklet(
            track_id=tracklet.track_id,
            embedding=tracklet_embedding,
            v_avg=v_avg,
            embedding_consistency=emb_consistency,
            tracklet_len=len(tracklet.entries),
        )

        if person_id is not None:
            tracklet.person_id = person_id
            tracklet.state = TrackletState.MATCHED
            logger.info(
                f"Tracklet {tracklet.track_id} -> Person {person_id} "
                f"(v_avg={v_avg:.3f}, consistency={emb_consistency:.3f}, frames={len(selected)})"
            )
        else:
            tracklet.state = TrackletState.TENTATIVE

        # Remove processed tracklet from buffer to start fresh
        self.tracklet_buffer.remove(tracklet.track_id)


if __name__ == "__main__":
    pipeline = WorkerPipeline()
    pipeline.run()
