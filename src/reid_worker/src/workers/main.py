from __future__ import annotations

import asyncio
import contextvars
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import cv2
from kafka.errors import KafkaError, NoBrokersAvailable
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
from src.matching import color_descriptor
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

# Per-task flag (ContextVar => isolated per asyncio task) marking that the
# current task already holds the identity-serialization lock, so nested helper
# calls don't try to re-acquire it (asyncio.Lock is not reentrant). Being a
# ContextVar (not an instance attribute) it stays correct under concurrency:
# a different task has its own copy defaulting to False and will wait on the lock.
_IDENTITY_SERIAL_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "reid_identity_serial_active", default=False
)

# Tasks emitted on every TrackedPerson — must match Avro schema field names.
_ATTRIBUTE_TASKS = ("gender", "age_child", "backpack", "sidebag",
                    "hat", "glasses", "sleeve", "lower")


def _summarize_live_status(
    live_visibility_score: float,
    overlap_ratio: float,
    quality: dict | None,
) -> str:
    if quality is None:
        return "tentative"
    if live_visibility_score < 0.45 or overlap_ratio >= 0.35:
        return "recovering"
    return "confirmed"


def _build_attribute_crop(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    *,
    top_padding_ratio: float,
    side_padding_ratio: float,
    bottom_padding_ratio: float,
) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    pad_x = width * side_padding_ratio
    pad_top = height * top_padding_ratio
    pad_bottom = height * bottom_padding_ratio

    crop_x1 = max(0, int(round(x1 - pad_x)))
    crop_y1 = max(0, int(round(y1 - pad_top)))
    crop_x2 = min(frame_w, int(round(x2 + pad_x)))
    crop_y2 = min(frame_h, int(round(y2 + pad_bottom)))
    return frame[crop_y1:crop_y2, crop_x1:crop_x2]


def _compute_person_snapshot_score(
    *,
    v_avg: float,
    overall_consistency: float,
    embedding_consistency: float,
    overlap_ratio: float = 0.0,
) -> float:
    """Rank candidate person snapshots by visual quality, not recency.

    We bias toward visibility because the UI avatar should remain a clear,
    representative crop even when later tracklets are valid but partially
    occluded or less frontal.
    """
    overlap_penalty = 0.35 * max(0.0, min(1.0, float(overlap_ratio or 0.0)))
    score = (
        (0.5 * v_avg)
        + (0.3 * overall_consistency)
        + (0.2 * embedding_consistency)
        - overlap_penalty
    )
    return round(float(score), 4)


def _rank_snapshot_entry(entry: TrackletEntry) -> tuple[float, float, int]:
    """Prefer clear crops for person-facing snapshots."""
    overlap = float(getattr(entry, "overlap_ratio", 0.0) or 0.0)
    visibility = float(getattr(entry, "v_score", 0.0) or 0.0)
    return (visibility - (0.7 * overlap), -overlap, int(getattr(entry, "frame_idx", 0)))


def _choose_person_snapshot_entry(
    entries: list[TrackletEntry],
    selected: list[TrackletEntry],
    *,
    max_overlap_ratio: float,
) -> TrackletEntry | None:
    """Pick a clean representative frame, or None when every crop is ambiguous.

    Tracklets can still be persisted as occlusion evidence. This helper only
    gates the canonical/person-facing snapshot so a two-person crop cannot
    overwrite the identity avatar or become high-scoring evidence.
    """
    def _clean(candidates: list[TrackletEntry]) -> list[TrackletEntry]:
        unique: dict[int, TrackletEntry] = {}
        for entry in candidates:
            if getattr(entry, "crop", None) is None or entry.crop.size <= 0:
                continue
            unique[int(entry.frame_idx)] = entry
        return [
            entry for entry in unique.values()
            if float(getattr(entry, "overlap_ratio", 0.0) or 0.0) <= max_overlap_ratio
        ]

    clean_selected = _clean(list(selected or []))
    if clean_selected:
        return max(clean_selected, key=_rank_snapshot_entry)

    clean_all = _clean(list(entries or []))
    if not clean_all:
        return None
    return max(clean_all, key=_rank_snapshot_entry)


def _tracklet_motion_summary(entries: list[TrackletEntry]) -> dict[str, float]:
    if not entries:
        return {
            "mean_width_px": 0.0,
            "mean_height_px": 0.0,
            "path_displacement_ratio": 0.0,
            "endpoint_displacement_ratio": 0.0,
            "boundary_contact_ratio": 0.0,
        }

    widths = []
    heights = []
    centers: list[np.ndarray] = []
    boundary_hits = 0
    boundary_eligible = 0
    edge_tolerance_px = 2.0
    for entry in entries:
        x1, y1, x2, y2 = entry.bbox_xyxy
        widths.append(max(float(x2 - x1), 1.0))
        heights.append(max(float(y2 - y1), 1.0))
        centers.append(np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32))
        fw = float(getattr(entry, "frame_w", 0) or 0)
        fh = float(getattr(entry, "frame_h", 0) or 0)
        if fw > 0 and fh > 0:
            boundary_eligible += 1
            if (
                x1 <= edge_tolerance_px
                or y1 <= edge_tolerance_px
                or x2 >= fw - edge_tolerance_px
                or y2 >= fh - edge_tolerance_px
            ):
                boundary_hits += 1

    norm = max(float(np.median(heights)), 1.0)
    path_disp = 0.0
    for prev, curr in zip(centers, centers[1:]):
        path_disp += float(np.linalg.norm(curr - prev))
    endpoint_disp = 0.0
    if len(centers) >= 2:
        endpoint_disp = float(np.linalg.norm(centers[-1] - centers[0]))

    boundary_ratio = (boundary_hits / boundary_eligible) if boundary_eligible > 0 else 0.0

    return {
        "mean_width_px": round(float(sum(widths) / len(widths)), 4),
        "mean_height_px": round(float(sum(heights) / len(heights)), 4),
        "path_displacement_ratio": round(path_disp / norm, 4),
        "endpoint_displacement_ratio": round(endpoint_disp / norm, 4),
        "boundary_contact_ratio": round(float(boundary_ratio), 4),
    }


def _compute_max_good_streak(entries: list[TrackletEntry], threshold: float) -> int:
    """PDF Bước 2: longest run of consecutive entries with v_score >= threshold.

    Filters out random YOLO flicker — a real person produces sustained
    high-quality frames; a noise detection rarely produces 4-in-a-row.
    """
    if not entries:
        return 0
    best = 0
    current = 0
    for entry in entries:
        if float(getattr(entry, "v_score", 0.0) or 0.0) >= float(threshold):
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


def _bbox_center(box: list[float] | np.ndarray) -> np.ndarray:
    return np.array([(float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0], dtype=np.float32)


def _select_embedding_consensus_indices(
    embeddings: list[np.ndarray],
    v_scores: list[float],
    *,
    similarity_threshold: float,
) -> list[int]:
    """Return indices for the strongest appearance-consistent embedding cluster."""
    if len(embeddings) <= 1:
        return list(range(len(embeddings)))

    stacked = np.stack(embeddings, axis=0).astype(np.float32)
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    stacked = stacked / np.maximum(norms, 1e-8)
    sims = stacked @ stacked.T

    best_cluster: list[int] = []
    best_score = -1.0
    for idx in range(len(embeddings)):
        cluster = [
            other_idx
            for other_idx in range(len(embeddings))
            if other_idx == idx or float(sims[idx, other_idx]) >= similarity_threshold
        ]
        # Prefer more support first, then better visual quality.
        score = (len(cluster) * 10.0) + sum(v_scores[i] for i in cluster)
        if score > best_score:
            best_cluster = cluster
            best_score = score

    return sorted(best_cluster)


@dataclass
class _MergeAttemptResult:
    person_id: int
    merged: bool = False
    gender_blocked: bool = False
    retryable_blocked: bool = False


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
            window_frames=self.settings.tracklet_window_frames,
            stale_frames=self.settings.tracklet_stale_frames,
        )
        self.topk_selector = TopKSelector(
            k=self.settings.topk_k,
            min_temporal_gap=self.settings.topk_min_temporal_gap,
            overlap_lambda=self.settings.overlap_lambda,
            min_tracklet_len=self.settings.tracklet_min_entries,
            min_high_quality_frames=self.settings.min_high_quality_frames,
            high_quality_threshold=self.settings.high_quality_threshold,
        )
        self.aggregator = WeightedEmbeddingAggregator(
            gamma=self.settings.gamma,
            outlier_threshold=self.settings.agg_outlier_threshold,
            exclude_overlap_ratio=self.settings.embedding_aggregate_max_overlap_ratio,
        )
        self.qdrant_store = QdrantPersonStore(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
            embedding_dim=self.settings.embedding_dim,
            similarity_threshold=self.settings.similarity_threshold,
            momentum=self.settings.momentum,
            max_gallery_size=self.settings.max_gallery_size,
            consensus_weight=self.settings.gallery_consensus_weight,
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
            flip_threshold=self.settings.attribute_flip_threshold,
            task_flip_thresholds={"gender": self.settings.gender_flip_threshold},
        )
        # Within-camera torso color evidence per person: person_id -> device_id ->
        # {"hist": aggregated HS histogram, "count": n samples}. Updated on confirmed
        # (non-provisional) assignment; read by the color guard to veto attaching a
        # clearly different-colored person. In-memory only (demo --reset wipes it).
        self._person_color_evidence: dict[int, dict[str, dict]] = {}
        self.matcher = ReIDMatcher(
            self.qdrant_store,
            id_allocator=self.person_id_allocator.allocate,
            promote_v_threshold=self.settings.promote_v_threshold,
            promote_consistency_threshold=self.settings.promote_consistency_threshold,
            new_identity_min_tracklet_len=self.settings.new_identity_min_tracklet_len,
            min_high_quality_frames=self.settings.min_high_quality_frames,
            tentative_max_attempts=self.settings.tentative_max_attempts,
            tentative_fallback_enabled=self.settings.tentative_fallback_enabled,
            update_v_threshold=self.settings.update_v_threshold,
            update_consistency_threshold=self.settings.update_consistency_threshold,
            update_min_tracklet_len=self.settings.update_min_tracklet_len,
            update_sim_threshold=self.settings.update_sim_threshold,
            update_anchor_min_score=self.settings.gallery_update_anchor_min_score,
            match_margin=self.settings.match_margin,
            spatial_reuse_threshold=self.settings.spatial_reuse_threshold,
            soft_match_threshold=self.settings.soft_match_threshold,
            eager_soft_match_threshold=self.settings.eager_soft_match_threshold,
            match_consistency_threshold=self.settings.match_consistency_threshold,
            low_visibility_threshold=self.settings.low_visibility_threshold,
            low_visibility_match_threshold=self.settings.low_visibility_match_threshold,
            blocked_match_score_threshold=self.settings.blocked_match_score_threshold,
            current_identity_min_score=self.settings.current_identity_min_score,
            current_identity_switch_min_score=self.settings.current_identity_switch_min_score,
            current_identity_switch_min_margin=self.settings.current_identity_switch_min_margin,
            current_identity_switch_max_current_score=self.settings.current_identity_switch_max_current_score,
            capped_identity_soft_match_threshold=self.settings.capped_identity_soft_match_threshold,
            near_gallery_defer_threshold=self.settings.near_gallery_defer_threshold,
            near_gallery_deferred_mint_max_score=self.settings.near_gallery_deferred_mint_max_score,
            good_streak_min_consecutive=self.settings.good_streak_min_consecutive,
            good_streak_promotion_enabled=self.settings.good_streak_promotion_enabled,
            scale_aux_gallery_enabled=self.settings.scale_aux_gallery_enabled,
            scale_aux_match_threshold=self.settings.scale_aux_match_threshold,
            scale_aux_match_margin=self.settings.scale_aux_match_margin,
            scale_aux_full_gallery_min_score=self.settings.scale_aux_full_gallery_min_score,
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
        self.current_track_metrics: dict[int, dict[str, float]] = {}
        self.track_identity_memory: dict[int, dict] = {}
        self.track_forbidden_person_ids: dict[int, set[int]] = {}
        self.track_cooccurrence_counts: dict[int, dict[int, int]] = {}
        self.occlusion_candidate_track_ids: set[int] = set()
        self.untracked_detection_clusters: list[dict] = []
        self.untracked_detection_cluster_seq = 0
        self.processing_tracklet_ids: set[int] = set()
        self.fragment_recovery_clusters: list[dict] = []
        # Admission appearance-gate state. Per track_id, cache the
        # embeddings of frames already accepted into the buffer so we can
        # compare a new frame's embedding against the running mean cheaply.
        # _track_id_split_counts assigns deterministic virtual track_ids when
        # a frame diverges from its track's appearance (likely ByteTrack swap).
        self._tracklet_embedding_cache: dict[int, list[np.ndarray]] = {}
        self._track_id_split_counts: dict[int, int] = {}
        self._tracklet_gate_last_check_frame: dict[int, int] = {}
        # Tracks fire-and-forget background tasks so the worker can drain them
        # on shutdown. Without this, idle-flush / fragment-recovery futures
        # complete after the consumer closes, causing post-stream Mongo writes.
        self._inflight: set[asyncio.Task] = set()
        self._stream_finalizing = False
        self._current_device_id: str = ""
        self.last_message_time_ns: int = 0
        self.last_idle_flush_ns: int = 0
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
            self.current_track_metrics.pop(track_id, None)
            self.track_forbidden_person_ids.pop(track_id, None)
            self.track_cooccurrence_counts.pop(track_id, None)
            getattr(self, "occlusion_candidate_track_ids", set()).discard(track_id)
            getattr(self, "processing_tracklet_ids", set()).discard(track_id)
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

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(vec_a / norm_a, vec_b / norm_b))

    def _tracklet_identity_shift_risk(self, tracklet) -> dict | None:
        """Detect likely ByteTrack ID swaps inside one finalized tracklet.

        This is deliberately geometry-only and conservative. A normal tracklet
        may move or scale, but the high-risk pattern for contamination is a long
        already-bound tracklet whose endpoint displacement and bbox growth both
        cross large thresholds. Those tracklets should not enter a confirmed
        identity through low-threshold continuity.
        """
        if not bool(getattr(self.settings, "tracklet_identity_shift_guard_enabled", True)):
            return None
        entries = list(getattr(tracklet, "entries", []) or [])
        min_entries = int(getattr(self.settings, "tracklet_identity_shift_min_entries", 12))
        if len(entries) < min_entries:
            return None

        first = [float(v) for v in entries[0].bbox_xyxy]
        last = [float(v) for v in entries[-1].bbox_xyxy]
        memory = getattr(self, "track_identity_memory", {}).get(tracklet.track_id) or {}
        anchor = memory.get("anchor_bbox_xyxy")
        anchor_frame_idx = memory.get("anchor_frame_idx")
        if anchor and len(anchor) >= 4 and anchor_frame_idx is not None:
            frame_gap = int(entries[-1].frame_idx) - int(anchor_frame_idx)
            if frame_gap >= int(
                getattr(self.settings, "tracklet_identity_shift_anchor_min_frame_gap", 24)
            ):
                anchor_risk = self._bbox_identity_shift_metrics(
                    [float(v) for v in anchor],
                    last,
                    entry_count=len(entries),
                    endpoint_threshold=float(
                        getattr(
                            self.settings,
                            "tracklet_identity_shift_anchor_min_endpoint_displacement_ratio",
                            0.40,
                        )
                    ),
                    size_threshold=float(
                        getattr(
                            self.settings,
                            "tracklet_identity_shift_anchor_min_size_ratio",
                            1.45,
                        )
                    ),
                    area_threshold=float(
                        getattr(
                            self.settings,
                            "tracklet_identity_shift_anchor_min_area_ratio",
                            1.90,
                        )
                    ),
                )
                if anchor_risk is not None:
                    anchor_risk["reason"] = "anchor_shift"
                    anchor_risk["anchor_frame_gap"] = int(frame_gap)
                    return anchor_risk

        return self._bbox_identity_shift_metrics(
            first,
            last,
            entry_count=len(entries),
            endpoint_threshold=float(
                getattr(
                    self.settings,
                    "tracklet_identity_shift_min_endpoint_displacement_ratio",
                    0.55,
                )
            ),
            size_threshold=float(
                getattr(self.settings, "tracklet_identity_shift_min_size_ratio", 1.60)
            ),
            area_threshold=float(
                getattr(self.settings, "tracklet_identity_shift_min_area_ratio", 2.20)
            ),
        )

    def _bbox_identity_shift_metrics(
        self,
        first: list[float],
        last: list[float],
        *,
        entry_count: int,
        endpoint_threshold: float,
        size_threshold: float,
        area_threshold: float,
    ) -> dict | None:
        if len(first) < 4 or len(last) < 4:
            return None
        first_w = max(first[2] - first[0], 1.0)
        first_h = max(first[3] - first[1], 1.0)
        last_w = max(last[2] - last[0], 1.0)
        last_h = max(last[3] - last[1], 1.0)
        first_area = first_w * first_h
        last_area = last_w * last_h
        endpoint_disp_ratio = float(
            np.linalg.norm(_bbox_center(first) - _bbox_center(last))
            / max(first_w, first_h, last_w, last_h, 1.0)
        )
        first_size = max(first_w, first_h, 1.0)
        last_size = max(last_w, last_h, 1.0)
        size_ratio = max(first_size, last_size) / max(min(first_size, last_size), 1.0)
        area_ratio = max(first_area, last_area) / max(min(first_area, last_area), 1.0)

        if (
            endpoint_disp_ratio < endpoint_threshold
            or size_ratio < size_threshold
            or area_ratio < area_threshold
        ):
            return None
        return {
            "endpoint_displacement_ratio": round(endpoint_disp_ratio, 4),
            "size_ratio": round(float(size_ratio), 4),
            "area_ratio": round(float(area_ratio), 4),
            "entry_count": int(entry_count),
        }

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

    def _is_spatially_distinct(
        self,
        box_a: list[float],
        box_b: list[float],
    ) -> bool:
        iou = self._bbox_iou(box_a, box_b)
        center_ratio = self._center_distance_ratio(box_a, box_b)
        return (
            iou <= self.settings.cooccurrence_guard_max_iou
            and center_ratio >= self.settings.cooccurrence_guard_min_center_distance_ratio
        )

    def _register_temporal_exclusion(
        self,
        track_id: int,
        forbidden_person_id: int,
    ) -> None:
        counts = self.track_cooccurrence_counts.setdefault(track_id, {})
        counts[forbidden_person_id] = counts.get(forbidden_person_id, 0) + 1
        if counts[forbidden_person_id] >= self.settings.cooccurrence_guard_min_shared_frames:
            self.track_forbidden_person_ids.setdefault(track_id, set()).add(forbidden_person_id)

    def _update_temporal_exclusions(
        self,
        track_results: np.ndarray,
    ) -> None:
        if not self.settings.cooccurrence_guard_enabled:
            return

        visible = [
            (int(track[4]), [float(v) for v in track[:4].tolist()])
            for track in track_results
        ]
        for idx, (track_id_a, bbox_a) in enumerate(visible):
            person_id_a = self.track_id_to_person_id.get(track_id_a)
            for track_id_b, bbox_b in visible[idx + 1:]:
                if not self._is_spatially_distinct(bbox_a, bbox_b):
                    continue
                person_id_b = self.track_id_to_person_id.get(track_id_b)
                if person_id_a is not None:
                    self._register_temporal_exclusion(track_id_b, person_id_a)
                if person_id_b is not None:
                    self._register_temporal_exclusion(track_id_a, person_id_b)
                # Persist the "two genuinely different people" signal for offline
                # quality audits: when two KNOWN person_ids have spatially-distinct
                # boxes in the same frame they cannot be the same person. Dedupe by
                # unordered pair so the log carries one line per proven-different
                # pair, not one per frame.
                if (
                    person_id_a is not None
                    and person_id_b is not None
                    and person_id_a != person_id_b
                ):
                    pair = (min(person_id_a, person_id_b), max(person_id_a, person_id_b))
                    seen = getattr(self, "_logged_spatial_exclusion_pairs", None)
                    if seen is None:
                        seen = set()
                        self._logged_spatial_exclusion_pairs = seen
                    if pair not in seen:
                        seen.add(pair)
                        log.info(
                            "spatial_exclusion",
                            person_a=pair[0],
                            person_b=pair[1],
                            track_a=track_id_a,
                            track_b=track_id_b,
                        )

    def _find_attribute_incompatible_person_ids(
        self,
        tracklet_attrs: dict[str, tuple[str, float]],
        current_person_id: int | None,
    ) -> set[int]:
        if not self.settings.attribute_conflict_guard_enabled:
            return set()

        t_gender, t_gender_conf = tracklet_attrs.get("gender", ("unknown", 0.0))
        if t_gender not in {"male", "female"}:
            return set()

        strong_tracklet_thresh = float(self.settings.attribute_conflict_tracklet_confidence)
        # Below this floor the gender classifier is effectively noise; don't let
        # it influence identity matching.
        weak_tracklet_floor = 0.55
        if t_gender_conf < weak_tracklet_floor:
            return set()

        tracklet_is_strong = t_gender_conf >= strong_tracklet_thresh
        if not tracklet_is_strong:
            log.warning(
                "attribute_guard_skipped_low_tracklet_conf",
                tracklet_gender=t_gender,
                tracklet_gender_conf=round(float(t_gender_conf), 3),
                threshold=strong_tracklet_thresh,
                current_person_id=current_person_id,
            )

        strong_person_conf = self._attribute_conflict_ready_person_confidence()
        strong_person_support = self._attribute_conflict_ready_person_support()

        incompatible: set[int] = set()
        for person_id in self.attribute_voter.known_person_ids():
            if current_person_id is not None and person_id == current_person_id:
                continue
            snapshot = self.attribute_voter.person_snapshot(person_id)
            p_gender, p_gender_conf = snapshot.get("gender", ("unknown", 0.0))
            p_gender_support = self.attribute_voter.person_task_stable_support(person_id, "gender")
            if p_gender not in {"male", "female"} or p_gender == t_gender:
                continue
            if tracklet_is_strong:
                if (
                    p_gender_conf >= strong_person_conf
                    or p_gender_support >= strong_person_support
                ):
                    incompatible.add(person_id)
            else:
                # Weak tracklet: block only against a person gender that is
                # itself conflict-ready. A single weak singleton attribute is
                # evidence for review, not a hard identity barrier.
                if (
                    p_gender_conf >= strong_person_conf
                    or p_gender_support >= strong_person_support
                ):
                    incompatible.add(person_id)
        return incompatible

    def _attribute_conflict_ready_person_confidence(self) -> float:
        return max(
            0.80,
            float(getattr(self.settings, "attribute_conflict_person_confidence", 0.88)),
        )

    def _attribute_conflict_ready_person_support(self) -> int:
        return max(
            2,
            int(getattr(self.settings, "attribute_conflict_person_min_support", 2)),
        )

    def _has_current_identity_attribute_conflict(
        self,
        tracklet_attrs: dict[str, tuple[str, float]],
        current_person_id: int | None,
    ) -> bool:
        if (
            current_person_id is None
            or not self.settings.attribute_conflict_guard_enabled
        ):
            return False

        t_gender, t_gender_conf = tracklet_attrs.get("gender", ("unknown", 0.0))
        if t_gender not in {"male", "female"}:
            return False
        if t_gender_conf < float(self.settings.attribute_conflict_tracklet_confidence):
            return False

        snapshot = self.attribute_voter.person_snapshot(current_person_id)
        p_gender, p_gender_conf = snapshot.get("gender", ("unknown", 0.0))
        if p_gender not in {"male", "female"} or p_gender == t_gender:
            return False

        p_gender_support = self.attribute_voter.person_task_stable_support(
            current_person_id,
            "gender",
        )
        return (
            p_gender_conf >= self._attribute_conflict_ready_person_confidence()
            or p_gender_support >= self._attribute_conflict_ready_person_support()
        )

    def _is_occlusion_attribute_unreliable(
        self,
        tracklet,
        *,
        v_avg: float,
    ) -> bool:
        """Return True when tracklet attributes should not update person state.

        Synthetic negative track_ids come from untracked-detection clusters:
        exactly the partial/body-overlap path where the PDF treats attributes
        as weak supporting evidence, not canonical identity state.
        """
        entries = list(getattr(tracklet, "entries", []) or [])
        max_overlap = max(
            (float(getattr(entry, "overlap_ratio", 0.0) or 0.0) for entry in entries),
            default=0.0,
        )
        return (
            int(getattr(tracklet, "track_id", 0)) < 0
            or float(v_avg) < float(getattr(self.settings, "low_visibility_threshold", 0.65))
            or max_overlap >= 0.35
        )

    def _has_person_attribute_conflict(
        self,
        tracklet_attrs: dict[str, tuple[str, float]],
        person_id: int | None,
    ) -> bool:
        """Conservative attribute veto for provisional occlusion attachment."""
        if (
            person_id is None
            or not self.settings.attribute_conflict_guard_enabled
        ):
            return False

        t_gender, t_gender_conf = tracklet_attrs.get("gender", ("unknown", 0.0))
        if t_gender not in {"male", "female"}:
            return False
        if t_gender_conf < 0.55:
            return False

        snapshot = self.attribute_voter.person_snapshot(person_id)
        p_gender, p_gender_conf = snapshot.get("gender", ("unknown", 0.0))
        if p_gender not in {"male", "female"} or p_gender == t_gender:
            return False

        p_gender_support = self.attribute_voter.person_task_stable_support(
            person_id,
            "gender",
        )
        return (
            p_gender_conf >= self._attribute_conflict_ready_person_confidence()
            or p_gender_support >= self._attribute_conflict_ready_person_support()
        )

    def _clean_tracklet_color(self, entries):
        """Build a torso color descriptor from only CLEAN frames (high visibility,
        low overlap). Returns None if too few reliable frames — the guard then
        abstains rather than veto on noisy/occluded color (the attempt-#1 mistake)."""
        if not entries:
            return None
        v_floor = float(getattr(self.settings, "color_guard_min_frame_visibility", 0.5))
        ov_ceil = float(getattr(self.settings, "color_guard_max_frame_overlap", 0.30))
        min_frames = int(getattr(self.settings, "color_guard_min_reliable_frames", 3))
        clean = [
            e for e in entries
            if float(getattr(e, "v_score", 0.0)) >= v_floor
            and float(getattr(e, "overlap_ratio", 0.0)) <= ov_ceil
        ]
        if len(clean) < min_frames:
            return None
        desc = color_descriptor.descriptor_from_entries(
            clean, max_frames=int(getattr(self.settings, "color_guard_max_frames", 12))
        )
        return desc

    def _update_person_color_evidence(self, person_id, device_id, entries) -> None:
        """Record a confirmed tracklet's CLEAN torso color into the person's
        per-device color evidence (running mean of HS histograms)."""
        if not bool(getattr(self.settings, "color_guard_enabled", True)):
            return
        if person_id is None or not device_id:
            return
        desc = self._clean_tracklet_color(entries)
        if desc is None:
            return
        per_dev = self._person_color_evidence.setdefault(int(person_id), {})
        cur = per_dev.get(str(device_id))
        if cur is None:
            per_dev[str(device_id)] = {"hist": desc.astype(np.float32), "count": 1}
        else:
            n = int(cur["count"])
            cur["hist"] = (cur["hist"].astype(np.float32) * n + desc) / (n + 1)
            cur["count"] = n + 1

    def _find_color_incompatible_person_ids(self, tracklet, current_person_id, device_id) -> set[int]:
        """Within-camera PRIMARY-MATCH color guard: persons on THIS camera whose
        reference torso color clearly differs from this (clean) tracklet. Fed into
        the matcher's forbidden set so primary gallery match can't glue two
        different-colored people (which CLIP-ReID's ~0 margin lets through).
        Cross-camera persons are never forbidden (color shifts with lighting)."""
        if not bool(getattr(self.settings, "color_guard_enabled", True)) or not device_id:
            return set()
        desc = self._clean_tracklet_color(list(getattr(tracklet, "entries", None) or []))
        if desc is None:
            return set()  # tracklet color unreliable -> abstain (never false-veto)
        thr = float(getattr(self.settings, "color_conflict_veto_threshold", 0.83))
        min_ev = int(getattr(self.settings, "color_guard_min_person_evidence", 1))
        incompatible: set[int] = set()
        for pid, per_dev in self._person_color_evidence.items():
            if current_person_id is not None and pid == current_person_id:
                continue
            cur = per_dev.get(str(device_id))
            if cur is None or int(cur["count"]) < min_ev:
                continue
            sim = color_descriptor.color_sim(desc, cur["hist"])
            if sim is not None and sim < thr:
                incompatible.add(int(pid))
        return incompatible

    def _tracklet_color_conflicts_person(self, entries, person_id, device_id) -> bool:
        """Reliability-gated color conflict for the occlusion-recovery paths.

        Returns True ONLY when the candidate's color is RELIABLE (>=N clean frames)
        AND clearly differs from the person's same-camera color evidence. On noisy/
        occluded crops _clean_tracklet_color returns None -> abstain (no veto). This
        is the fix for attempt #1, which vetoed on noisy color and over-fragmented."""
        if not bool(getattr(self.settings, "color_guard_enabled", True)):
            return False
        if person_id is None or not device_id:
            return False
        desc = self._clean_tracklet_color(list(entries or []))
        if desc is None:
            return False  # color not trustworthy here -> let other guards decide
        per_dev = self._person_color_evidence.get(int(person_id))
        if not per_dev:
            return False
        cur = per_dev.get(str(device_id))
        if cur is None or int(cur["count"]) < int(getattr(self.settings, "color_guard_min_person_evidence", 1)):
            return False
        sim = color_descriptor.color_sim(desc, cur["hist"])
        if sim is None:
            return False
        return sim < float(getattr(self.settings, "color_conflict_veto_threshold", 0.83))

    def _persons_color_conflict(self, source_pid, target_pid) -> bool:
        """Within-camera color veto for the duplicate-MERGE decision: True if the two
        persons' reference torso colors clearly differ on a SHARED camera. Both are
        established persons here, so their color evidence is a clean aggregate — the
        most reliable place for the guard (validated 0.94 same vs 0.55 diff at the
        aggregate level). Abstains when they share no device with evidence (e.g. a
        legitimate cross-camera merge), so cross-view linking is never blocked."""
        if not bool(getattr(self.settings, "color_guard_enabled", True)):
            return False
        src = self._person_color_evidence.get(int(source_pid))
        tgt = self._person_color_evidence.get(int(target_pid))
        if not src or not tgt:
            return False
        thr = float(getattr(self.settings, "color_conflict_veto_threshold", 0.83))
        min_ev = int(getattr(self.settings, "color_guard_min_person_evidence", 1))
        for dev in set(src) & set(tgt):
            s, t = src[dev], tgt[dev]
            if int(s["count"]) < min_ev or int(t["count"]) < min_ev:
                continue
            sim = color_descriptor.color_sim(s["hist"], t["hist"])
            if sim is not None and sim < thr:
                return True
        return False

    def _maybe_accept_occlusion_provisional_match(
        self,
        *,
        tracklet,
        matching: dict,
        v_avg: float,
        tracklet_attrs: dict[str, tuple[str, float]],
        forbidden_person_ids: set[int],
        recent_incompatible_person_ids: set[int],
        blocked_person_ids: set[int],
    ) -> tuple[int | None, dict]:
        """Attach near-gallery occlusion evidence without updating identity anchors.

        This is intentionally narrower than a normal gallery match: it only
        consumes a matcher decision that already deferred a near-gallery hit,
        requires an occlusion/untracked signal, enforces margin + attribute
        guards, and marks the result so persistence skips person snapshot and
        attribute updates. The goal is realtime occlusion ReID evidence, not
        lowering the global match threshold.
        """
        if not bool(getattr(self.settings, "occlusion_provisional_match_enabled", True)):
            return None, matching
        if not isinstance(matching, dict):
            return None, matching
        if matching.get("method") not in {
            "near_gallery_deferred",
            "fragment_recovery_deferred_near_gallery",
        }:
            return None, matching

        reuse_pid = matching.get("reuse_person_id")
        if reuse_pid is None:
            return None, matching
        try:
            reuse_pid = int(reuse_pid)
        except (TypeError, ValueError):
            return None, matching

        if (
            reuse_pid in forbidden_person_ids
            or reuse_pid in recent_incompatible_person_ids
            or reuse_pid in blocked_person_ids
        ):
            return None, matching
        if self._has_person_attribute_conflict(tracklet_attrs, reuse_pid):
            log.warning(
                "occlusion_provisional_match_rejected_attribute_conflict",
                track_id=tracklet.track_id,
                reuse_person_id=reuse_pid,
                tracklet_gender=tracklet_attrs.get("gender"),
                person_snapshot=self.attribute_voter.person_snapshot(reuse_pid),
            )
            return None, matching
        if self._tracklet_color_conflicts_person(
            getattr(tracklet, "entries", None), reuse_pid, getattr(self, "_current_device_id", "")
        ):
            log.warning(
                "occlusion_provisional_match_rejected_color_conflict",
                track_id=tracklet.track_id,
                reuse_person_id=reuse_pid,
            )
            return None, matching

        score = matching.get("similarity_score")
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            return None, matching
        entries = list(getattr(tracklet, "entries", []) or [])
        short_reentry = self._is_short_reentry_to_person(
            tracklet=tracklet,
            person_id=reuse_pid,
            score=score_f,
        )
        base_min_score = float(getattr(self.settings, "occlusion_provisional_match_threshold", 0.62))
        reentry_min_score = float(
            getattr(self.settings, "occlusion_provisional_reentry_min_similarity", base_min_score)
        )
        min_score = min(base_min_score, reentry_min_score) if short_reentry else base_min_score
        if score_f < min_score:
            log.info(
                "occlusion_provisional_rejected_score",
                track_id=tracklet.track_id,
                reuse_person_id=reuse_pid,
                similarity_score=round(score_f, 4),
                threshold=round(min_score, 4),
                short_reentry=bool(short_reentry),
            )
            return None, matching

        runner_up = matching.get("runner_up_score")
        if runner_up is not None:
            try:
                margin = score_f - float(runner_up)
            except (TypeError, ValueError):
                margin = None
            min_margin = float(getattr(self.settings, "occlusion_provisional_min_margin", 0.03))
            if margin is not None and margin < min_margin:
                log.info(
                    "occlusion_provisional_rejected_margin",
                    track_id=tracklet.track_id,
                    reuse_person_id=reuse_pid,
                    similarity_score=round(score_f, 4),
                    runner_up_score=round(float(runner_up), 4),
                    margin=round(float(margin), 4),
                    min_margin=round(min_margin, 4),
                    short_reentry=bool(short_reentry),
                )
                return None, matching

        max_overlap = max((float(getattr(entry, "overlap_ratio", 0.0) or 0.0) for entry in entries), default=0.0)
        # Crop-contamination guard: a detection whose box heavily overlaps another
        # person's box yields a crop polluted by that other person, which can match
        # an existing identity at an inflated score (the crop literally contains the
        # target's pixels). Refuse to absorb such a detection so it forms its own
        # identity instead of contaminating the matched person's snapshots/evidence.
        contaminated_overlap = float(
            getattr(self.settings, "occlusion_provisional_match_max_overlap_ratio", 0.40)
        )
        if max_overlap >= contaminated_overlap:
            log.info(
                "occlusion_provisional_rejected_contaminated_overlap",
                track_id=tracklet.track_id,
                reuse_person_id=reuse_pid,
                similarity_score=round(score_f, 4),
                max_overlap=round(max_overlap, 4),
                max_overlap_threshold=round(contaminated_overlap, 4),
                short_reentry=bool(short_reentry),
            )
            return None, matching
        occlusion_like = (
            int(tracklet.track_id) < 0
            or float(v_avg) < float(getattr(self.settings, "low_visibility_threshold", 0.65))
            or max_overlap >= 0.35
            or short_reentry
        )
        if not occlusion_like:
            log.info(
                "occlusion_provisional_rejected_no_occlusion_signal",
                track_id=tracklet.track_id,
                reuse_person_id=reuse_pid,
                similarity_score=round(score_f, 4),
                v_avg=round(float(v_avg), 4),
                max_overlap=round(max_overlap, 4),
                low_visibility_threshold=round(float(getattr(self.settings, "low_visibility_threshold", 0.65)), 4),
                short_reentry=bool(short_reentry),
            )
            return None, matching

        accepted = {
            **matching,
            "method": "occlusion_provisional_match",
            "source": matching.get("source") or "near_gallery_deferred",
            "reuse_person_id": reuse_pid,
            "similarity_score": score_f,
            "canonical_update_applied": False,
            "provisional": True,
            "provisional_reason": matching.get("method"),
        }
        log.info(
            "occlusion_provisional_match_accepted",
            track_id=tracklet.track_id,
            person_id=reuse_pid,
            similarity_score=round(score_f, 4),
            v_avg=round(float(v_avg), 4),
            max_overlap=round(max_overlap, 4),
            short_reentry=bool(short_reentry),
        )
        return reuse_pid, accepted

    def _is_short_reentry_to_person(self, *, tracklet, person_id: int, score: float) -> bool:
        if not bool(getattr(self.settings, "occlusion_provisional_short_reentry_enabled", False)):
            return False
        if float(score) < float(getattr(self.settings, "occlusion_provisional_reentry_min_similarity", 0.58)):
            return False
        entries = list(getattr(tracklet, "entries", []) or [])
        if not entries:
            return False
        if len(entries) > int(getattr(self.settings, "occlusion_provisional_reentry_max_entries", 8)):
            return False
        obs = self.person_last_observation.get(int(person_id))
        if not obs:
            return False
        obs_frame = obs.get("frame_idx")
        if obs_frame is None:
            return False
        frame_gap = int(entries[0].frame_idx) - int(obs_frame)
        if frame_gap <= 0 or frame_gap > int(getattr(self.settings, "occlusion_provisional_reentry_max_gap_frames", 180)):
            return False
        obs_bbox = obs.get("bbox_xyxy")
        if not obs_bbox or len(obs_bbox) < 4:
            return False
        center_ratio = self._center_distance_ratio(
            [float(v) for v in entries[0].bbox_xyxy],
            [float(v) for v in obs_bbox],
        )
        return center_ratio <= float(
            getattr(self.settings, "occlusion_provisional_reentry_max_center_distance_ratio", 2.0)
        )

    def _update_person_last_observation_from_tracklet(self, person_id: int, tracklet) -> None:
        entries = list(getattr(tracklet, "entries", []) or [])
        if not entries:
            return
        entry = entries[-1]
        self.person_last_observation[int(person_id)] = {
            "bbox_xyxy": [float(v) for v in entry.bbox_xyxy],
            "timestamp_ns": int(entry.timestamp_ns),
            "device_id": str(getattr(self, "_current_device_id", "")),
            "frame_idx": int(entry.frame_idx),
        }

    def _find_blocked_duplicate_person_ids(
        self,
        bbox_xyxy: list[float],
        blocked_person_ids: set[int],
    ) -> set[int]:
        duplicate_person_ids: set[int] = set()
        for person_id in blocked_person_ids:
            obs = self.person_last_observation.get(person_id)
            if not obs:
                continue
            iou = self._bbox_iou(bbox_xyxy, obs["bbox_xyxy"])
            if iou >= self.settings.duplicate_track_iou_threshold:
                duplicate_person_ids.add(person_id)
        return duplicate_person_ids

    def _find_co_active_person_ids(
        self,
        own_track_id: int,
        current_person_id: int | None,
        current_time_ns: int,
        co_active_window_ns: int = int(0.6 * 1e9),
    ) -> set[int]:
        """Person IDs currently bound to other live tracks.

        If track A → person 3 was seen in the last `co_active_window_ns`,
        person 3 cannot also be this new track. This catches the case where
        a track fragments mid-life and the new track_id tries to merge into
        the still-active twin's identity.
        """
        co_active: set[int] = set()
        for other_track_id, person_id in self.track_id_to_person_id.items():
            if other_track_id == own_track_id:
                continue
            if current_person_id is not None and person_id == current_person_id:
                continue
            last_seen_ns = self.track_last_seen_ns.get(other_track_id)
            if last_seen_ns is None:
                continue
            if current_time_ns - last_seen_ns <= co_active_window_ns:
                co_active.add(person_id)
        return co_active

    def _find_recent_incompatible_person_ids(
        self,
        bbox_xyxy: list[float],
        current_time_ns: int,
        current_person_id: int | None,
    ) -> set[int]:
        if not self.settings.recent_match_guard_enabled:
            return set()

        incompatible_person_ids: set[int] = set()
        max_gap_ns = int(self.settings.recent_match_guard_seconds * 1e9)
        for person_id, obs in self.person_last_observation.items():
            if current_person_id is not None and person_id == current_person_id:
                continue
            gap_ns = current_time_ns - int(obs["timestamp_ns"])
            if gap_ns < 0 or gap_ns > max_gap_ns:
                continue

            obs_bbox = obs["bbox_xyxy"]
            iou = self._bbox_iou(bbox_xyxy, obs_bbox)
            center_ratio = self._center_distance_ratio(bbox_xyxy, obs_bbox)
            if (
                iou < self.settings.recent_match_guard_min_iou
                and center_ratio > self.settings.recent_match_guard_max_center_distance_ratio
            ):
                incompatible_person_ids.add(person_id)
        return incompatible_person_ids

    def _should_ignore_pretrack_static_artifact(
        self,
        track_id: int,
        bbox_xyxy: list[float],
        frame_w: int = 0,
        frame_h: int = 0,
    ) -> bool:
        if not self.settings.pretrack_static_filter_enabled:
            return False
        # Preserve the PDF's cut_off/occlusion signal: small stationary boxes
        # touching the frame boundary are often partial humans, not signage.
        if frame_w > 0 and frame_h > 0:
            x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
            edge_tolerance_px = 2.0
            if (
                x1 <= edge_tolerance_px
                or y1 <= edge_tolerance_px
                or x2 >= float(frame_w) - edge_tolerance_px
                or y2 >= float(frame_h) - edge_tolerance_px
            ):
                return False
        history = self.prev_bboxes.get(track_id, [])
        if len(history) < self.settings.pretrack_static_filter_min_frames:
            return False

        widths = [float(box[2] - box[0]) for box in history]
        heights = [float(box[3] - box[1]) for box in history]
        if (
            max(widths) > self.settings.pretrack_static_filter_max_width_px
            or max(heights) > self.settings.pretrack_static_filter_max_height_px
        ):
            return False

        centers = [_bbox_center(box) for box in history]
        anchor = centers[0]
        max_center_drift = max(float(np.linalg.norm(center - anchor)) for center in centers[1:]) if len(centers) > 1 else 0.0
        return max_center_drift <= self.settings.pretrack_static_filter_max_center_drift_px

    def _should_suppress_new_identity(self, tracklet: Tracklet) -> bool:
        if not self.settings.static_artifact_filter_enabled:
            return False
        if len(tracklet.entries) < self.settings.static_artifact_min_entries:
            return False

        motion = _tracklet_motion_summary(tracklet.entries)
        # PDF Bước 1: boundary contact is an OCCLUSION signal (cut_off). A
        # tracklet with substantial boundary contact is a partial-body person,
        # not a static artifact — exempt from suppression even if it has small
        # bbox and low motion.
        boundary_skip = float(
            getattr(self.settings, "static_artifact_boundary_contact_skip", 0.3)
        )
        if motion.get("boundary_contact_ratio", 0.0) >= boundary_skip:
            return False
        consistency = compute_tracklet_consistency(tracklet.entries)
        size_ok = (
            motion["mean_width_px"] <= self.settings.static_artifact_max_mean_width_px
            and motion["mean_height_px"] <= self.settings.static_artifact_max_mean_height_px
        )
        low_motion_static = (
            motion["path_displacement_ratio"] <= self.settings.static_artifact_max_path_displacement_ratio
            and motion["endpoint_displacement_ratio"] <= self.settings.static_artifact_max_endpoint_displacement_ratio
        )
        stable_bbox_static = (
            consistency.bbox_size_stability >= float(
                getattr(self.settings, "static_artifact_min_bbox_stability", 0.97)
            )
            and consistency.position_stability >= float(
                getattr(self.settings, "static_artifact_min_position_stability", 0.97)
            )
        )
        suppressed = (
            size_ok
            and (low_motion_static or stable_bbox_static)
        )
        if suppressed:
            # Diagnostic only — make data-driven threshold tuning possible without
            # guessing which boundary persons or static false positives were caught.
            log.warning(
                "new_identity_suppressed_static_artifact",
                track_id=tracklet.track_id,
                entries=len(tracklet.entries),
                mean_width_px=round(motion["mean_width_px"], 2),
                mean_height_px=round(motion["mean_height_px"], 2),
                path_displacement_ratio=round(motion["path_displacement_ratio"], 4),
                endpoint_displacement_ratio=round(motion["endpoint_displacement_ratio"], 4),
                bbox_size_stability=round(float(consistency.bbox_size_stability), 4),
                position_stability=round(float(consistency.position_stability), 4),
            )
        return suppressed

    def _track_inflight(self, task: asyncio.Task) -> asyncio.Task:
        """Register a fire-and-forget task so it can be awaited on shutdown."""
        # Tests that bypass __init__ via .__new__() won't have _inflight set;
        # lazy-init to keep this resilient.
        if not hasattr(self, "_inflight"):
            self._inflight = set()
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)
        return task

    def _remember_track_identity(self, person_id: int, tracklet: Tracklet) -> None:
        if not tracklet.entries:
            return
        if not hasattr(self, "track_identity_memory"):
            self.track_identity_memory = {}
        first_entry = tracklet.entries[0]
        last_entry = tracklet.entries[-1]
        existing = self.track_identity_memory.get(tracklet.track_id) or {}
        same_person = int(existing.get("person_id", person_id)) == int(person_id)
        anchor_bbox = (
            existing.get("anchor_bbox_xyxy")
            if same_person and existing.get("anchor_bbox_xyxy")
            else [float(v) for v in first_entry.bbox_xyxy]
        )
        anchor_frame_idx = (
            existing.get("anchor_frame_idx")
            if same_person and existing.get("anchor_frame_idx") is not None
            else int(first_entry.frame_idx)
        )
        self.track_identity_memory[tracklet.track_id] = {
            "person_id": int(person_id),
            "anchor_frame_idx": int(anchor_frame_idx),
            "anchor_bbox_xyxy": [float(v) for v in anchor_bbox],
            "last_frame_idx": int(last_entry.frame_idx),
            "last_bbox_xyxy": [float(v) for v in last_entry.bbox_xyxy],
        }

    def _find_recent_track_identity(self, tracklet: Tracklet) -> int | None:
        if not tracklet.entries:
            return None
        memory = getattr(self, "track_identity_memory", {}).get(tracklet.track_id)
        if not memory:
            return None

        first_entry = tracklet.entries[0]
        frame_gap = int(first_entry.frame_idx) - int(memory.get("last_frame_idx", -10**9))
        max_gap = int(getattr(self.settings, "track_identity_memory_max_gap_frames", 180))
        if frame_gap < 0 or frame_gap > max_gap:
            return None

        previous_box = memory.get("last_bbox_xyxy")
        if not previous_box:
            return None
        center_ratio = self._center_distance_ratio(
            previous_box,
            [float(v) for v in first_entry.bbox_xyxy],
        )
        max_center_ratio = float(
            getattr(self.settings, "track_identity_memory_max_center_distance_ratio", 1.25)
        )
        if center_ratio > max_center_ratio:
            return None
        return int(memory["person_id"])

    def _can_allocate_new_identity(self, tracklet: Tracklet | None = None) -> bool:
        """Gate new person_id allocation on worker-side wall-clock idleness.

        Reference is ``self.last_message_time_ns`` — the worker's wall-clock
        time the last Kafka batch was received. NOT the message's claimed
        ``timestamp_ns`` (that's the edge service's view of time, which can
        be hours/months in the past for replays and breaks the gate entirely).

        When ``stream_quiescence_seconds`` of wall-clock time have passed
        since the last Kafka batch, the stream is considered finished and no
        new identities should be minted. Set the threshold long enough that
        legitimate end-of-stream tracklets (which finalize ~3s after last
        frame, then process serially through the async task queue) all
        complete BEFORE the gate fires. 20s default is empirical headroom
        for typical workloads; bump via env var if your workload has more
        end-of-stream tracklets than that.

        ``max_new_identity_lag_seconds`` is a separate freshness guard keyed to
        the edge publish timestamp in the tracklet entries. It covers Kafka
        backlog: while the worker is still receiving old messages, quiescence
        alone never fires because ``last_message_time_ns`` keeps refreshing.
        """
        if not getattr(self, "_stream_finalizing", False):
            max_lag_s = float(getattr(self.settings, "max_new_identity_lag_seconds", 0.0) or 0.0)
            if not self._tracklet_is_fresh_enough(
                tracklet,
                max_lag_s=max_lag_s,
                log_event="new_identity_blocked_stale_tracklet",
            ):
                return False

        quiescence_s = float(getattr(self.settings, "stream_quiescence_seconds", 0.0) or 0.0)
        if quiescence_s <= 0:
            return True
        last_ns = getattr(self, "last_message_time_ns", None)
        if not last_ns:
            return True  # startup — no messages yet, allow
        return (time.time_ns() - int(last_ns)) <= int(quiescence_s * 1e9)

    def _tracklet_is_fresh_enough(
        self,
        tracklet: Tracklet | None,
        *,
        max_lag_s: float,
        log_event: str,
    ) -> bool:
        if max_lag_s <= 0 or tracklet is None:
            return True
        entries = list(getattr(tracklet, "entries", []) or [])
        if not entries:
            return True
        latest_entry_ns = max(int(getattr(entry, "timestamp_ns", 0) or 0) for entry in entries)
        if latest_entry_ns <= 0:
            return True
        lag_ns = time.time_ns() - latest_entry_ns
        if lag_ns <= int(max_lag_s * 1e9):
            return True
        log.info(
            log_event,
            track_id=getattr(tracklet, "track_id", None),
            lag_seconds=round(lag_ns / 1e9, 3),
            max_lag_seconds=max_lag_s,
            frame_start=getattr(entries[0], "frame_idx", None),
            frame_end=getattr(entries[-1], "frame_idx", None),
        )
        return False

    def _known_identity_count(self) -> int:
        known_person_ids: set[int] = set(self.track_id_to_person_id.values())
        known_person_ids.update(self.person_last_observation.keys())
        known_person_ids.update(self.attribute_voter.known_person_ids())
        return len(known_person_ids)

    def _identity_cap_reached(self) -> bool:
        max_person_identities = int(getattr(self.settings, "max_person_identities", 0) or 0)
        return max_person_identities > 0 and self._known_identity_count() >= max_person_identities

    async def _persist_untracked_detection_candidates(
        self,
        *,
        detections: list[dict],
        track_results,
        frame,
        frame_number: int,
        timestamp_ns: int,
    ) -> None:
        """Persist YOLO detections that ByteTrack did not keep as evidence only.

        These crops are not promoted to identities. They exist to expose recall
        failures under occlusion/small-person cases where the detector fires but
        the tracker cannot maintain a tracklet long enough for ReID confirmation.
        """
        if not getattr(self.settings, "untracked_detection_candidates_enabled", True):
            return
        if not detections:
            return

        frame_h, frame_w = frame.shape[:2]
        tracked_boxes = [np.asarray(track[:4], dtype=np.float32) for track in track_results]
        min_conf = float(getattr(self.settings, "untracked_detection_min_confidence", 0.25))
        min_visibility = float(getattr(self.settings, "untracked_detection_min_visibility", 0.35))
        max_track_iou = float(getattr(self.settings, "untracked_detection_max_track_iou", 0.20))
        cluster_enabled = bool(getattr(self.settings, "untracked_detection_cluster_enabled", True))
        raw_enabled = bool(getattr(self.settings, "untracked_detection_raw_candidates_enabled", False))

        for det_idx, det in enumerate(detections):
            bbox = np.asarray(det["bbox"], dtype=np.float32)
            confidence = float(det.get("confidence", 0.0) or 0.0)
            visibility = float(det.get("visibility_score", 0.0) or 0.0)
            if confidence < min_conf or visibility < min_visibility:
                continue
            best_iou = max((compute_iou_prev(bbox, box) for box in tracked_boxes), default=0.0)
            if best_iou > max_track_iou:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            attribute_crop = _build_attribute_crop(
                frame,
                bbox,
                top_padding_ratio=self.settings.attribute_crop_top_padding_ratio,
                side_padding_ratio=self.settings.attribute_crop_side_padding_ratio,
                bottom_padding_ratio=self.settings.attribute_crop_bottom_padding_ratio,
            )
            if attribute_crop.size == 0:
                attribute_crop = crop

            synthetic_track_id = -int(frame_number * 1000 + det_idx + 1)
            tracklet = type(
                "UntrackedDetectionTracklet",
                (),
                {
                    "track_id": synthetic_track_id,
                    "entries": [
                        TrackletEntry(
                            frame_idx=frame_number,
                            crop=crop,
                            v_score=visibility,
                            bbox_xyxy=bbox.tolist(),
                            timestamp_ns=timestamp_ns,
                            attribute_crop=attribute_crop,
                            overlap_ratio=float(det.get("overlap_ratio", 0.0) or 0.0),
                            frame_w=int(frame_w),
                            frame_h=int(frame_h),
                        )
                    ],
                },
            )()
            if cluster_enabled:
                await self._update_untracked_detection_cluster(
                    tracklet.entries[0],
                    detection_confidence=confidence,
                    nearest_track_iou=best_iou,
                )
            if raw_enabled:
                await self._persist_occlusion_candidate(
                    tracklet,
                    reason="untracked_detection",
                    matching={
                        "method": "unconfirmed",
                        "source": "untracked_detection",
                        "similarity_score": None,
                        "nearest_track_iou": round(best_iou, 4),
                        "detection_confidence": round(confidence, 4),
                    },
                    min_entries=1,
                )

        if cluster_enabled:
            await self._flush_stale_untracked_detection_clusters(frame_number)

    def _cluster_embedding_similarity(
        self,
        embedding: np.ndarray,
        cluster: dict,
    ) -> float | None:
        embeddings_by_frame = cluster.get("embeddings_by_frame") or {}
        embeddings = list(cluster.get("reference_embeddings") or [])
        embeddings.extend([
            embeddings_by_frame.get(int(entry.frame_idx))
            for entry in cluster.get("entries", []) or []
        ])
        embeddings = [emb for emb in embeddings if emb is not None]
        if not embeddings:
            return None
        ref = np.mean(np.stack(embeddings, axis=0), axis=0)
        ref_norm = np.linalg.norm(ref)
        emb_norm = np.linalg.norm(embedding)
        if ref_norm < 1e-8 or emb_norm < 1e-8:
            return 0.0
        return float(np.dot(ref / ref_norm, embedding / emb_norm))

    def _find_untracked_detection_cluster(
        self,
        entry: TrackletEntry,
        *,
        embedding: np.ndarray | None = None,
    ) -> dict | None:
        if not hasattr(self, "untracked_detection_clusters"):
            self.untracked_detection_clusters = []
        max_gap = int(getattr(self.settings, "untracked_detection_cluster_max_gap_frames", 18))
        max_distance = float(getattr(self.settings, "untracked_detection_cluster_max_center_distance_ratio", 1.25))
        min_similarity = float(
            getattr(self.settings, "untracked_detection_cluster_appearance_min_similarity", 0.62)
        )
        curr_center = _bbox_center(entry.bbox_xyxy)
        curr_height = max(float(entry.bbox_xyxy[3] - entry.bbox_xyxy[1]), 1.0)
        best_cluster = None
        best_score = float("inf")

        for cluster in self.untracked_detection_clusters:
            last_frame = int(cluster["last_frame"])
            frame_gap = int(entry.frame_idx) - last_frame
            if frame_gap <= 0 or frame_gap > max_gap:
                continue
            last_bbox = cluster["last_bbox"]
            last_center = _bbox_center(last_bbox)
            last_height = max(float(last_bbox[3] - last_bbox[1]), 1.0)
            norm = max(curr_height, last_height, 1.0)
            center_distance = float(np.linalg.norm(curr_center - last_center)) / norm
            if center_distance > max_distance:
                continue
            if embedding is not None:
                sim = self._cluster_embedding_similarity(embedding, cluster)
                if sim is not None and sim < min_similarity:
                    continue
            score = center_distance + (frame_gap / max(max_gap, 1))
            if score < best_score:
                best_cluster = cluster
                best_score = score
        return best_cluster

    async def _update_untracked_detection_cluster(
        self,
        entry: TrackletEntry,
        *,
        detection_confidence: float,
        nearest_track_iou: float,
    ) -> None:
        embedding = None
        if (
            bool(getattr(self.settings, "untracked_detection_cluster_appearance_gate_enabled", True))
            and float(entry.v_score)
            >= float(getattr(self.settings, "untracked_detection_cluster_appearance_min_visibility", 0.55))
        ):
            embedding = await self._extract_crop_embedding(entry.crop)

        cluster = self._find_untracked_detection_cluster(entry, embedding=embedding)
        if cluster is None:
            self.untracked_detection_cluster_seq = int(getattr(self, "untracked_detection_cluster_seq", 0)) + 1
            cluster = {
                "cluster_id": -9_000_000 - self.untracked_detection_cluster_seq,
                "entries": [],
                "confidences": [],
                "nearest_track_ious": [],
                "embeddings_by_frame": {},
                "reference_embeddings": [],
                "last_bbox": entry.bbox_xyxy,
                "last_frame": int(entry.frame_idx),
            }
            self.untracked_detection_clusters.append(cluster)

        cluster["entries"].append(entry)
        cluster["confidences"].append(float(detection_confidence))
        cluster["nearest_track_ious"].append(float(nearest_track_iou))
        if embedding is not None:
            cluster.setdefault("embeddings_by_frame", {})[int(entry.frame_idx)] = embedding
        cluster["last_bbox"] = entry.bbox_xyxy
        cluster["last_frame"] = int(entry.frame_idx)

        # MEMORY CAP: each TrackletEntry holds a full-resolution crop ndarray
        # (~60 KB for a typical bbox). Without this cap one persistent cluster
        # accumulated 100+ entries (≥6 MB just for crops), and several such
        # clusters together OOM-killed the worker container. After promotion,
        # the buffer copy is authoritative; the cluster only needs a sliding
        # window of recent entries to keep growing if the person sticks around.
        cluster_max_entries = int(getattr(self.settings, "untracked_detection_cluster_max_entries", 30))
        if len(cluster["entries"]) > cluster_max_entries:
            cluster["entries"] = cluster["entries"][-cluster_max_entries:]
            cluster["confidences"] = cluster["confidences"][-cluster_max_entries:]
            cluster["nearest_track_ious"] = cluster["nearest_track_ious"][-cluster_max_entries:]
            live_frames = {int(e.frame_idx) for e in cluster["entries"]}
            cluster["embeddings_by_frame"] = {
                int(frame): emb
                for frame, emb in (cluster.get("embeddings_by_frame") or {}).items()
                if int(frame) in live_frames
            }

        # Once a cluster is promoted into the normal tracklet buffer, that
        # buffer becomes the source of truth. Continuing to upsert diagnostic
        # occlusion candidates for the same promoted cluster floods the UI and
        # makes a successfully recovered occluded person look unresolved.
        if self._maybe_promote_untracked_cluster(cluster, entry):
            return

        min_entries = int(getattr(self.settings, "untracked_detection_cluster_min_entries", 2))
        if len(cluster["entries"]) < min_entries:
            return

        # Persistence is for diagnostic/UI purposes only — the in-memory cluster
        # is authoritative for promotion decisions. Throttle Mongo writes to
        # threshold crossings (min_entries, then logarithmic steps) rather than
        # every entry, which was 16+ writes for one 17-entry cluster.
        entry_count = len(cluster["entries"])
        persistence_steps = {2, 3, 5, 8, 12, 16, 22, 30, 40, 55, 75, 100}
        should_persist = (
            entry_count == min_entries
            or entry_count in persistence_steps
        )

        if should_persist:
            tracklet = type(
                "UntrackedDetectionClusterTracklet",
                (),
                {
                    "track_id": int(cluster["cluster_id"]),
                    "entries": list(cluster["entries"]),
                },
            )()
            start_frame = int(cluster["entries"][0].frame_idx)
            candidate_id = (
                f"{self._current_device_id}:untracked_cluster:"
                f"{abs(int(cluster['cluster_id']))}:{start_frame}"
            )
            # Remember the candidate id so a later evidence-attach can mark this
            # exact row resolved (instead of leaving an orphan / adding a new row).
            cluster["candidate_id"] = candidate_id
            await self._persist_occlusion_candidate(
                tracklet,
                reason="untracked_detection_cluster",
                matching={
                    "method": "unconfirmed",
                    "source": "untracked_detection_cluster",
                    "similarity_score": None,
                    "cluster_entry_count": entry_count,
                    "detection_confidence_avg": round(float(np.mean(cluster["confidences"])), 4),
                    "detection_confidence_max": round(float(np.max(cluster["confidences"])), 4),
                    "nearest_track_iou_min": round(float(np.min(cluster["nearest_track_ious"])), 4),
                    "nearest_track_iou_max": round(float(np.max(cluster["nearest_track_ious"])), 4),
                },
                min_entries=min_entries,
                candidate_id_override=candidate_id,
            )

    def _maybe_promote_untracked_cluster(self, cluster: dict, latest_entry: TrackletEntry) -> bool:
        # Open a path for an untracked detection cluster (a person YOLO sees
        # but ByteTrack can't track — small/distant or boundary-crossing) to
        # reach the normal embedding+matcher pipeline. The synthetic negative
        # track_id flows through tracklet_buffer → _process_tracklet, where
        # every existing safeguard (consensus filter, near_gallery_defer,
        # promote_consistency, fragment_recovery) still applies.
        if not bool(getattr(self.settings, "untracked_cluster_promote_enabled", False)):
            return False

        entries = cluster["entries"]
        if cluster.get("promoted_to_buffer", False):
            synthetic_track_id = int(cluster["cluster_id"])
            self.tracklet_buffer.append(synthetic_track_id, latest_entry)
            cluster["entries"] = []
            cluster["confidences"] = []
            cluster["nearest_track_ious"] = []
            cluster["embeddings_by_frame"] = {}
            return True

        max_v = max(e.v_score for e in entries)
        n_entries = len(entries)
        consistency = compute_tracklet_consistency(entries)

        # Two-tier promotion: long-evidence tier OR high-confidence brief tier.
        # The brief tier is intentionally narrow: enough clean temporal support
        # to rescue ByteTrack misses, but not enough to revive 2-4 frame static
        # false positives as confirmed identities.
        min_entries_slow = int(getattr(self.settings, "untracked_cluster_promote_min_entries", 6))
        min_visibility_slow = float(getattr(self.settings, "untracked_cluster_promote_min_visibility", 0.65))
        min_entries_fast = int(getattr(self.settings, "untracked_cluster_promote_min_entries_fast", 4))
        min_visibility_fast = float(getattr(self.settings, "untracked_cluster_promote_min_visibility_fast", 0.85))
        min_fast_consistency = float(
            getattr(self.settings, "untracked_cluster_promote_fast_min_overall_consistency", 0.88)
        )

        slow_tier_met = n_entries >= min_entries_slow and max_v >= min_visibility_slow
        fast_tier_met = (
            n_entries >= min_entries_fast
            and max_v >= min_visibility_fast
            and float(consistency.overall) >= min_fast_consistency
        )

        if not (slow_tier_met or fast_tier_met):
            # Only log clusters that have enough entries to be plausible promotion
            # candidates (>= fast tier entry count). Below that, the cluster is too
            # young to evaluate and the log would just be noise.
            if n_entries >= min_entries_fast:
                log.info(
                    "untracked_cluster_rejected",
                    cluster_id=cluster["cluster_id"],
                    n_entries=n_entries,
                    max_visibility=round(max_v, 4),
                    overall_consistency=round(float(consistency.overall), 4),
                    slow_required_entries=min_entries_slow,
                    slow_required_visibility=round(min_visibility_slow, 4),
                    fast_required_entries=min_entries_fast,
                    fast_required_visibility=round(min_visibility_fast, 4),
                    fast_required_consistency=round(min_fast_consistency, 4),
                )
            return False

        # For logging only — which tier triggered.
        tier = "slow" if slow_tier_met else "fast"
        min_entries_required = min_entries_slow if slow_tier_met else min_entries_fast
        min_visibility_required = min_visibility_slow if slow_tier_met else min_visibility_fast

        synthetic_track_id = int(cluster["cluster_id"])
        cluster["reference_embeddings"] = list((cluster.get("embeddings_by_frame") or {}).values())[-8:]

        # First promotion: push all existing entries so the synthetic tracklet
        # starts at full evidence rather than rebuilding one frame at a time.
        for cluster_entry in entries:
            self.tracklet_buffer.append(synthetic_track_id, cluster_entry)
        cluster["promoted_to_buffer"] = True
        # The buffer now owns the synthetic tracklet — release the cluster's
        # own crop references so memory doesn't double up while the cluster
        # continues to grow with new untracked detections.
        cluster["entries"] = []
        cluster["confidences"] = []
        cluster["nearest_track_ious"] = []
        cluster["embeddings_by_frame"] = {}

        log.info(
            "untracked_cluster_promoted",
            cluster_id=cluster["cluster_id"],
            entries=n_entries,
            max_visibility=round(max_v, 4),
            overall_consistency=round(float(consistency.overall), 4),
            tier=tier,
            min_entries_required=min_entries_required,
            min_visibility_required=min_visibility_required,
        )
        return True

    def _is_synthetic_fast_tracklet_ready(self, tracklet) -> bool:
        if int(getattr(tracklet, "track_id", 0)) >= 0:
            return False
        entries = list(getattr(tracklet, "entries", []) or [])
        min_entries = int(getattr(self.settings, "untracked_cluster_promote_min_entries_fast", 5))
        if len(entries) < min_entries:
            return False
        min_visibility = float(getattr(self.settings, "untracked_cluster_promote_min_visibility_fast", 0.85))
        if max((float(entry.v_score) for entry in entries), default=0.0) < min_visibility:
            return False
        high_quality_threshold = float(getattr(self.settings, "high_quality_threshold", 0.55))
        min_high_quality = int(getattr(self.settings, "min_high_quality_frames", 3))
        high_quality = sum(1 for entry in entries if float(entry.v_score) >= high_quality_threshold)
        if high_quality < min_high_quality:
            return False
        min_consistency = float(
            getattr(self.settings, "untracked_cluster_promote_fast_min_overall_consistency", 0.88)
        )
        consistency = compute_tracklet_consistency(entries)
        return float(consistency.overall) >= min_consistency

    async def _admission_gate_or_split(
        self,
        track_id: int,
        crop: np.ndarray,
        v_worker: float,
        frame_idx: int | None = None,
    ) -> int:
        """Block contaminated frames from poisoning a tracklet buffer.

        When ByteTrack swaps identities during occlusion, the original track_id
        starts receiving crops of a different person. By the time the tracklet
        is finalized and embedded, the buffer holds a mixture. This gate runs
        BEFORE append: if the new high-quality frame's embedding disagrees with
        the running mean of the buffer's accepted embeddings, the frame is
        routed to a fresh virtual track_id so the original tracklet stays clean
        and the new appearance gets its own chance at matching.

        Returns the track_id under which the new frame should be appended —
        either the original (frame is on-distribution) or a deterministic
        virtual id of the form ``-(abs(track_id) * 1_000_000 + n)``.
        """
        if not getattr(self.settings, "tracklet_appearance_gate_enabled", True):
            return track_id
        min_v = float(getattr(self.settings, "tracklet_appearance_gate_min_v", 0.6))
        if v_worker < min_v:
            return track_id
        cached = self._tracklet_embedding_cache.get(track_id, [])
        if len(cached) < 2:
            # Need at least two prior frames to form a reliable mean. First few
            # frames go in unconditionally, but we still compute + cache their
            # embeddings if v_worker is high so the gate is armed quickly.
            emb = await self._extract_crop_embedding(crop)
            if emb is not None:
                self._tracklet_embedding_cache.setdefault(track_id, []).append(emb)
            return track_id

        check_interval = max(
            1,
            int(getattr(self.settings, "tracklet_appearance_gate_check_interval_frames", 1) or 1),
        )
        if frame_idx is not None:
            if not hasattr(self, "_tracklet_gate_last_check_frame"):
                self._tracklet_gate_last_check_frame = {}
            last_checked = self._tracklet_gate_last_check_frame.get(track_id)
            if last_checked is not None and int(frame_idx) - last_checked < check_interval:
                return track_id
            self._tracklet_gate_last_check_frame[track_id] = int(frame_idx)

        new_emb = await self._extract_crop_embedding(crop)
        if new_emb is None:
            return track_id

        ref = np.mean(np.stack(cached, axis=0), axis=0)
        ref_norm = np.linalg.norm(ref)
        new_norm = np.linalg.norm(new_emb)
        if ref_norm < 1e-8 or new_norm < 1e-8:
            return track_id
        sim = float(np.dot(ref / ref_norm, new_emb / new_norm))
        split_threshold = float(getattr(self.settings, "tracklet_split_threshold", 0.55))
        if sim >= split_threshold:
            self._tracklet_embedding_cache[track_id].append(new_emb)
            # Bound cache size to avoid unbounded growth on long tracks.
            if len(self._tracklet_embedding_cache[track_id]) > 16:
                self._tracklet_embedding_cache[track_id] = (
                    self._tracklet_embedding_cache[track_id][-16:]
                )
            return track_id

        # Diverged — route to a virtual track_id so this person gets its own
        # tracklet instead of being merged into the original's buffer.
        n = self._track_id_split_counts.get(track_id, 0) + 1
        self._track_id_split_counts[track_id] = n
        virtual_id = -(abs(int(track_id)) * 1_000_000 + n)
        self._tracklet_embedding_cache[virtual_id] = [new_emb]
        log.info(
            "tracklet_appearance_split",
            original_track_id=int(track_id),
            virtual_track_id=int(virtual_id),
            similarity=round(sim, 4),
            split_threshold=split_threshold,
        )
        return virtual_id

    async def _extract_crop_embedding(self, crop: np.ndarray) -> np.ndarray | None:
        """Encode crop to JPEG and call /embedding. Returns None on failure."""
        try:
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                return None
            _, result = await self.model_client.extract_features(
                buf.tobytes(),
                model=self.settings.embedding_model,
            )
            emb = result.get("embedding") if isinstance(result, dict) else None
            if emb is None:
                return None
            return np.asarray(emb, dtype=np.float32)
        except Exception as exc:
            logger_fn = getattr(log, "debug", log.info)
            logger_fn("admission_embedding_failed", error=str(exc))
            return None

    def _build_scale_aux_upper_crop(self, crop: np.ndarray) -> np.ndarray | None:
        if crop is None or crop.size <= 0:
            return None
        h = int(crop.shape[0])
        if h <= 1:
            return None
        ratio = float(getattr(self.settings, "scale_aux_crop_top_ratio", 0.62))
        cut_y = int(round(h * min(max(ratio, 0.30), 0.90)))
        cut_y = max(1, min(h, cut_y))
        upper = crop[:cut_y, :]
        if upper.size <= 0:
            return None
        return upper

    async def _extract_scale_aux_embedding(self, entry) -> np.ndarray | None:
        if not bool(getattr(self.settings, "scale_aux_gallery_enabled", False)):
            return None
        crop = getattr(entry, "crop", None)
        if crop is None or crop.size <= 0:
            crop = getattr(entry, "attribute_crop", None)
        upper = self._build_scale_aux_upper_crop(crop)
        if upper is None:
            return None
        emb = await self._extract_crop_embedding(upper)
        if emb is None:
            return None
        norm = np.linalg.norm(emb)
        if norm <= 1e-8:
            return None
        return emb / norm

    async def _maybe_persist_scale_aux_embedding(
        self,
        *,
        person_id: int,
        tracklet,
        entry,
        v_avg: float,
        emb_consistency: float,
        overall_consistency: float,
        selected_max_overlap: float,
        matching: dict,
    ) -> None:
        if not bool(getattr(self.settings, "scale_aux_gallery_enabled", False)):
            return
        if entry is None:
            return
        if float(v_avg) < float(getattr(self.settings, "scale_aux_min_v", 0.70)):
            return
        if float(emb_consistency) < float(getattr(self.settings, "scale_aux_min_consistency", 0.80)):
            return
        if len(getattr(tracklet, "entries", []) or []) < int(getattr(self.settings, "scale_aux_min_tracklet_len", 5)):
            return
        if float(selected_max_overlap) > float(getattr(self.settings, "scale_aux_max_overlap_ratio", 0.35)):
            return
        if bool((matching or {}).get("provisional")):
            return
        aux_emb = await self._extract_scale_aux_embedding(entry)
        if aux_emb is None:
            return
        try:
            await asyncio.to_thread(
                self.qdrant_store.add_upper_body_embedding,
                person_id,
                aux_emb,
                {
                    "source": "scale_aux_upper",
                    "track_id": int(tracklet.track_id),
                    "v_avg": round(float(v_avg), 4),
                    "consistency": round(float(emb_consistency), 4),
                    "overall_consistency": round(float(overall_consistency), 4),
                    "match_method": str((matching or {}).get("method", "")),
                },
            )
        except Exception:
            log.warning(
                "scale_aux_upper_persist_failed",
                person_id=person_id,
                track_id=getattr(tracklet, "track_id", None),
                exc_info=True,
            )

    async def _try_attach_cluster_as_evidence(self, cluster: dict) -> bool:
        """Attach a CLEAR untracked cluster to an EXISTING person as occlusion
        evidence (a sighting), if it confidently matches that person's gallery.

        This rescues clear occluded-gap detections ByteTrack missed — too short to
        safely mint a new identity, but clear enough to enrich the matched person.
        It NEVER mints and NEVER updates the gallery/snapshot/attributes (see
        _persist_attached_occlusion_evidence). The gallery score + margin guard
        prevent attaching to the wrong (similar-looking) person.
        """
        if not bool(getattr(self.settings, "untracked_cluster_evidence_attach_enabled", True)):
            return False
        if cluster.get("evidence_attached") or cluster.get("promoted_to_buffer"):
            return False
        entries = list(cluster.get("entries") or [])
        embs = [np.asarray(e, dtype=np.float32) for e in (cluster.get("embeddings_by_frame") or {}).values()]
        min_entries = int(getattr(self.settings, "untracked_detection_cluster_min_entries", 2))
        if len(entries) < min_entries or not embs:
            return False
        max_v = max((float(getattr(e, "v_score", 0.0)) for e in entries), default=0.0)
        if max_v < float(getattr(self.settings, "untracked_cluster_evidence_attach_min_visibility", 0.65)):
            return False
        emb = np.mean(np.stack(embs, axis=0), axis=0)
        emb_norm = float(np.linalg.norm(emb))
        if emb_norm < 1e-8:
            return False
        emb = emb / emb_norm
        min_score = float(getattr(self.settings, "similarity_threshold", 0.73))
        hits = self.qdrant_store.search(emb, top_k=2, score_threshold=min_score)
        if not hits:
            # No existing person to attach to (best match < similarity_threshold).
            # Last-resort: mint a NEW person if this is a clear, sustained,
            # clearly-distinct orphan (the clear-but-missed-person case).
            return await self._lastresort_mint_clear_cluster(cluster, entries, embs, emb, max_v)
        pid, score = int(hits[0][0]), float(hits[0][1])
        runner_up = float(hits[1][1]) if len(hits) > 1 else None
        margin = (score - runner_up) if runner_up is not None else float("inf")
        if margin < float(getattr(self.settings, "match_margin", 0.06)):
            return False
        # Reliability-gated within-camera color guard: block a CLEAR cluster whose
        # torso color differs from the person's same-camera color (abstains on
        # noisy clusters). This path bypasses attribute guards (tracklet_attrs=None).
        _cluster_device = str(cluster.get("device_id") or getattr(self, "_current_device_id", ""))
        if self._tracklet_color_conflicts_person(entries, pid, _cluster_device):
            log.info(
                "untracked_cluster_evidence_rejected_color_conflict",
                cluster_id=cluster.get("cluster_id"),
                person_id=pid,
                similarity_score=round(score, 4),
            )
            return False
        cluster_tracklet = type(
            "UntrackedClusterEvidenceTracklet",
            (),
            {"track_id": int(cluster["cluster_id"]), "entries": entries},
        )()
        consistency = compute_tracklet_consistency(entries)
        v_avg = float(np.mean([float(getattr(e, "v_score", 0.0)) for e in entries]))
        emb_consistency = WeightedEmbeddingAggregator.compute_embedding_consistency(embs)
        tracklet_id = str(uuid.uuid4())
        await self._persist_attached_occlusion_evidence(
            tracklet=cluster_tracklet,
            tracklet_id=tracklet_id,
            person_id=pid,
            consistency=consistency,
            v_avg=v_avg,
            emb_consistency=emb_consistency,
            selected=entries,
            matching={
                "method": "untracked_cluster_evidence",
                "source": "untracked_detection_cluster",
                "reuse_person_id": pid,
                "similarity_score": score,
                "provisional": True,
                "canonical_update_applied": False,
            },
            tracklet_attrs=None,
        )
        cluster["evidence_attached"] = True
        cand_id = cluster.get("candidate_id")
        if cand_id:
            try:
                await self.mongo.mark_occlusion_candidate_attached(cand_id, pid)
            except Exception:
                log.debug("mark_occlusion_candidate_attached_failed", exc_info=True)
        log.info(
            "untracked_cluster_evidence_attached",
            cluster_id=cluster["cluster_id"],
            person_id=pid,
            similarity_score=round(score, 4),
            margin=None if margin == float("inf") else round(float(margin), 4),
            entries=len(entries),
            max_visibility=round(max_v, 4),
        )
        return True

    async def _lastresort_mint_clear_cluster(self, cluster, entries, embs, emb, max_v) -> bool:
        """Final-net mint for a CLEAR, sustained untracked orphan that matched no
        existing person (best < similarity_threshold). Recovers a clearly-detected
        person ByteTrack missed whom the matcher never minted (plan E12). Tightly
        gated so it never mints noise or duplicates a plausible re-entry."""
        if not bool(getattr(self.settings, "untracked_cluster_lastresort_mint_enabled", True)):
            return False
        if float(max_v) < float(getattr(self.settings, "untracked_cluster_lastresort_mint_min_visibility", 0.85)):
            return False
        if len(entries) < int(getattr(self.settings, "untracked_cluster_lastresort_mint_min_entries", 6)):
            return False
        consistency = compute_tracklet_consistency(entries)
        if float(consistency.overall) < float(getattr(self.settings, "promote_consistency_threshold", 0.65)):
            return False
        # Clearly-distinct guard: the cluster must NOT be a plausible re-entry of an
        # existing person. If its best gallery match (above the defer floor) is at
        # or above the clearly-distinct bar, don't mint (avoid duplicating a person
        # whose cross-view re-entry just scored low) — protects 51/52.
        distinct_below = float(getattr(self.settings, "near_gallery_deferred_mint_max_score", 0.64)) or 0.64
        defer_floor = float(getattr(self.settings, "near_gallery_defer_threshold", 0.58))
        near = self.qdrant_store.search(emb, top_k=1, score_threshold=defer_floor)
        if near and float(near[0][1]) >= distinct_below:
            return False
        cluster_tracklet = type(
            "UntrackedClusterMintTracklet",
            (),
            {"track_id": int(cluster["cluster_id"]), "entries": entries},
        )()
        if self._identity_cap_reached() or not self._can_allocate_new_identity(cluster_tracklet):
            return False
        try:
            pid = self.person_id_allocator.allocate()
        except Exception as err:
            raise PersonIdAllocationError(str(err)) from err
        self.qdrant_store.add_person(
            pid, emb,
            {"source": "untracked_cluster_lastresort", "total_entries": len(entries)},
        )
        v_avg = float(np.mean([float(getattr(e, "v_score", 0.0)) for e in entries]))
        emb_consistency = WeightedEmbeddingAggregator.compute_embedding_consistency(embs)
        best_entry = max(entries, key=lambda e: float(getattr(e, "v_score", 0.0)))
        tracklet_id = str(uuid.uuid4())
        await self._persist_tracklet(
            tracklet=cluster_tracklet,
            tracklet_id=tracklet_id,
            person_id=pid,
            consistency=consistency,
            v_avg=v_avg,
            emb_consistency=emb_consistency,
            best_entry=best_entry,
            selected=entries,
            matching={
                "method": "new_identity",
                "source": "untracked_cluster_lastresort",
                "similarity_score": None,
            },
            person_attrs={},
            tracklet_attrs=None,
        )
        cluster["evidence_attached"] = True
        cand_id = cluster.get("candidate_id")
        if cand_id:
            try:
                await self.mongo.mark_occlusion_candidate_attached(cand_id, pid)
            except Exception:
                log.debug("mark_occlusion_candidate_attached_failed", exc_info=True)
        log.info(
            "untracked_cluster_lastresort_minted",
            cluster_id=cluster["cluster_id"],
            person_id=pid,
            entries=len(entries),
            max_visibility=round(float(max_v), 4),
            overall_consistency=round(float(consistency.overall), 4),
            best_existing=None if not near else round(float(near[0][1]), 4),
        )
        return True

    async def _flush_stale_untracked_detection_clusters(self, frame_number: int) -> None:
        if not hasattr(self, "untracked_detection_clusters"):
            self.untracked_detection_clusters = []
        flush_after = int(getattr(self.settings, "untracked_detection_cluster_flush_after_frames", 36))
        kept: list[dict] = []
        for cluster in self.untracked_detection_clusters:
            if int(frame_number) - int(cluster["last_frame"]) <= flush_after:
                kept.append(cluster)
                continue
            # Cluster is leaving memory — last chance to turn a clear orphan into
            # evidence for an existing person before it's dropped.
            try:
                await self._try_attach_cluster_as_evidence(cluster)
            except Exception:
                log.debug("cluster_evidence_attach_failed", exc_info=True)
        self.untracked_detection_clusters = kept

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
        reconciler_task = None
        interval = float(getattr(self.settings, "background_reconciler_interval_s", 0.0))
        if interval > 0:
            reconciler_task = asyncio.create_task(self._background_reconciler_loop(interval))
        try:
            async with self.model_client:
                while True:
                    messages = await asyncio.to_thread(self.consumer.poll, timeout_ms=1000)
                    if not messages:
                        await self._flush_idle_tracklets_if_needed(time.time_ns())
                        continue
                    # Heartbeat messages (empty detections) come from the edge
                    # to keep the streaming preview smooth; they carry no ReID
                    # signal and must not reset the quiescence gate, otherwise
                    # `_can_allocate_new_identity` would never fire while the
                    # edge keeps pumping empty frames.
                    has_reid_signal = any(msg.get("detections") for msg in messages)
                    if has_reid_signal:
                        self.last_message_time_ns = time.time_ns()
                    for msg in messages:
                        await self._process_message(msg)
                        # Refresh per-message so the quiescence gate doesn't
                        # mistake long batch processing time for stream end.
                        # A batch that takes 30s to process must not look
                        # idle from the matcher's perspective at second 21.
                        # Heartbeats are excluded for the same reason as above.
                        if msg.get("detections"):
                            self.last_message_time_ns = time.time_ns()
                    await asyncio.sleep(self.settings.poll_interval_s)
        finally:
            if reconciler_task is not None:
                reconciler_task.cancel()
                try:
                    await reconciler_task
                except (asyncio.CancelledError, Exception):
                    pass
            # Drain fire-and-forget background work so the worker shutdown is
            # deterministic. Without this, occlusion-candidate persists / late
            # short-fragment tasks complete after the consumer closes.
            if self._inflight:
                await self._drain_inflight_tasks(timeout_s=10.0)

    async def _background_reconciler_loop(self, interval_s: float) -> None:
        """Periodically re-check recently-touched persons for duplicates.

        The inline merge in _maybe_merge_duplicate_person fires once per
        tracklet finalization, which is too rare to catch fragmentation that
        only becomes evident after both sides have accumulated evidence. This
        loop revisits persons asynchronously, leaning on the same
        _maybe_merge_duplicate_person path so all guards (cooccurrence,
        attribute, gender) apply identically.
        """
        max_persons = int(getattr(self.settings, "background_reconciler_max_persons", 50))
        while True:
            try:
                await asyncio.sleep(interval_s)
                await self._reconcile_recent_persons(
                    max_persons=max_persons,
                    passes=1,
                    reason="background",
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("background_reconciler_iter_failed", error=str(exc))

    async def _process_message(self, msg: dict) -> None:
        self.processed_messages += 1
        device_id = msg["device_id"]
        self._current_device_id = device_id
        frame_number = msg["frame_number"]
        if self._is_end_of_stream_message(msg):
            self.last_message_time_ns = time.time_ns()
            await self._finalize_stream(device_id=device_id)
            return

        detections = msg["detections"]
        if not detections:
            return
        image_data = msg["image_data"]
        timestamp_ns = msg["created_at"]
        img_array = np.frombuffer(image_data, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
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
        await self._persist_untracked_detection_candidates(
            detections=detections,
            track_results=track_results,
            frame=frame,
            frame_number=frame_number,
            timestamp_ns=timestamp_ns,
        )
        if len(track_results) == 0:
            self._cleanup_inactive_tracks(current_time_ns)
            return

        visible_track_ids = {int(track[4]) for track in track_results}
        self._update_temporal_exclusions(track_results)

        ignored_track_ids: set[int] = set()
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
                v_edge_floor_ratio=float(getattr(self.settings, "v_worker_edge_floor_ratio", 0.0)),
            )
            self.prev_bboxes.setdefault(track_id, []).append(bbox_xyxy.copy())
            keep_frames = max(3, int(self.settings.pretrack_static_filter_min_frames))
            self.prev_bboxes[track_id] = self.prev_bboxes[track_id][-keep_frames:]
            self.current_track_metrics[track_id] = {
                "live_visibility_score": float(round(v_worker, 4)),
                "overlap_ratio": float(round(overlap_ratio, 4)),
            }
            if (
                track_id not in self.track_id_to_person_id
                and self._should_ignore_pretrack_static_artifact(
                    track_id,
                    bbox_xyxy.tolist(),
                    frame_w=frame_w,
                    frame_h=frame_h,
                )
            ):
                ignored_track_ids.add(track_id)
                self.tracklet_buffer.remove(track_id)
                continue

            x1, y1, x2, y2 = map(int, bbox_xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            attribute_crop = _build_attribute_crop(
                frame,
                bbox_xyxy,
                top_padding_ratio=self.settings.attribute_crop_top_padding_ratio,
                side_padding_ratio=self.settings.attribute_crop_side_padding_ratio,
                bottom_padding_ratio=self.settings.attribute_crop_bottom_padding_ratio,
            )
            if attribute_crop.size == 0:
                attribute_crop = crop

            target_track_id = await self._admission_gate_or_split(
                track_id=track_id,
                crop=crop,
                v_worker=float(v_worker),
                frame_idx=int(frame_number),
            )
            self.tracklet_buffer.append(
                target_track_id,
                TrackletEntry(
                    frame_idx=frame_number,
                    crop=crop,
                    v_score=v_worker,
                    bbox_xyxy=bbox_xyxy.tolist(),
                    timestamp_ns=timestamp_ns,
                    attribute_crop=attribute_crop,
                    overlap_ratio=overlap_ratio,
                    frame_w=int(frame_w),
                    frame_h=int(frame_h),
                ),
            )

        # Buffer readiness/staleness is keyed on frame index (deterministic),
        # not wall-clock current_time_ns, so the same video flushes the same
        # tracklets in the same order every run.
        buffer_clock = int(frame_number)
        pop_ready_tracklets = getattr(self.tracklet_buffer, "pop_ready_tracklets", None)
        processing_tracklet_ids = set(getattr(self, "processing_tracklet_ids", set()))
        ready_tracklets = (
            pop_ready_tracklets(buffer_clock, skip_track_ids=processing_tracklet_ids)
            if callable(pop_ready_tracklets)
            else self.tracklet_buffer.get_ready_tracklets(buffer_clock)
        )
        self.ready_tracklets += len(ready_tracklets)
        pop_stale_tracklets = getattr(self.tracklet_buffer, "pop_stale_tracklets", None)
        stale_tracklets = (
            pop_stale_tracklets(buffer_clock, skip_track_ids=processing_tracklet_ids)
            if callable(pop_stale_tracklets)
            else []
        )
        for stale_tracklet in stale_tracklets:
            if stale_tracklet.person_id is not None:
                continue
            if stale_tracklet.state == TrackletState.MATCHED:
                continue
            if (
                getattr(self.settings, "recover_stale_tracklets_enabled", True)
                and len(stale_tracklet.entries) >= int(getattr(self.settings, "tracklet_min_entries", 4))
            ):
                self._schedule_tracklet_processing(stale_tracklet, reserved_person_ids=set())
                continue
            self._track_inflight(asyncio.ensure_future(
                self._process_short_fragment_tracklet(
                    stale_tracklet,
                    reason="short_stale_tracklet",
                )
            ))
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
            if track_id in ignored_track_ids:
                continue
            live_track_key = f"{device_id}:{track_id}"
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
                    "track_id": track_id,
                    "live_track_key": live_track_key,
                    "tracklet_id": None,
                    "tracklet_state": "tentative",
                    "snapshot_key": None,
                    "visibility_score": 0.0,
                    "live_visibility_score": float(self.current_track_metrics.get(track_id, {}).get("live_visibility_score", 0.0)),
                    "overlap_ratio": float(self.current_track_metrics.get(track_id, {}).get("overlap_ratio", 0.0)),
                    "quality": None,
                    "matching": None,
                    "attributes": None,
                    "status": "tentative",
                }
                for task in _ATTRIBUTE_TASKS:
                    payload[task] = "unknown"
                    payload[f"{task}_confidence"] = 0.0
            else:
                self.person_last_observation[person_id] = {
                    "bbox_xyxy": [float(v) for v in track[:4].tolist()],
                    "timestamp_ns": current_time_ns,
                    "device_id": device_id,
                    "frame_idx": int(frame_number),
                }
                person_attrs = self.attribute_voter.person_snapshot(person_id)
                meta = self.track_metadata.get(track_id, {})
                payload = {
                    "person_id": person_id,
                    "bbox": [float(v) for v in track[:4].tolist()],
                    "confidence": float(track[5]),
                    "track_id": track_id,
                    "live_track_key": live_track_key,
                    "tracklet_id": meta.get("tracklet_id"),
                    "tracklet_state": meta.get("tracklet_state"),
                    "snapshot_key": meta.get("snapshot_key"),
                    "visibility_score": float(meta.get("visibility_score", 0.0)),
                    "live_visibility_score": float(self.current_track_metrics.get(track_id, {}).get("live_visibility_score", 0.0)),
                    "overlap_ratio": float(self.current_track_metrics.get(track_id, {}).get("overlap_ratio", 0.0)),
                    "quality": meta.get("quality"),
                    "matching": meta.get("matching"),
                    "attributes": meta.get("attributes"),
                    "status": _summarize_live_status(
                        float(self.current_track_metrics.get(track_id, {}).get("live_visibility_score", 0.0)),
                        float(self.current_track_metrics.get(track_id, {}).get("overlap_ratio", 0.0)),
                        meta.get("quality"),
                    ),
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
                self._schedule_tracklet_processing(
                    tracklet,
                    reserved_person_ids=frame_reserved_person_ids,
                )

    @staticmethod
    def _is_end_of_stream_message(msg: dict) -> bool:
        return (
            int(msg.get("frame_number", 0) or 0) < 0
            and not msg.get("detections")
            and not msg.get("image_data")
        )

    async def _finalize_stream(self, *, device_id: str) -> None:
        """Flush buffered evidence immediately after a finite video reaches EOF.

        The normal realtime path uses idle timers because a live camera has no
        explicit end. File/demo inputs do have an end, so the edge service sends
        a sentinel message. At that point every queued frame for the video has
        already been consumed, and the right behavior is to finalize once,
        drain persistence/snapshot tasks, then leave no buffered tracklets that
        can keep growing UI counters after playback has stopped.
        """
        if getattr(self, "_stream_finalizing", False):
            return

        self._stream_finalizing = True
        log.info(
            "stream_finalization_started",
            device_id=device_id,
            buffered_tracklets=len(getattr(self.tracklet_buffer, "tracklets", {})),
            inflight=len(getattr(self, "_inflight", set())),
        )
        try:
            finalization_timeout_s = float(
                getattr(self.settings, "stream_finalization_timeout_seconds", 60.0)
            )
            max_finalize_passes = max(
                3,
                int(getattr(self.settings, "tentative_max_attempts", 5)) + 1,
            )
            for _ in range(max_finalize_passes):
                tracklets = list(getattr(self.tracklet_buffer, "tracklets", {}).values())
                if not tracklets:
                    if getattr(self, "_inflight", set()):
                        drained = await self._drain_inflight_tasks(timeout_s=finalization_timeout_s)
                        if not drained:
                            break
                        continue
                    break

                processing_tracklet_ids = set(getattr(self, "processing_tracklet_ids", set()))
                processable_tracklets = [
                    tracklet
                    for tracklet in tracklets
                    if tracklet.track_id not in processing_tracklet_ids
                ]
                for tracklet in processable_tracklets:
                    self.tracklet_buffer.remove(tracklet.track_id)
                    if tracklet.person_id is not None or tracklet.state == TrackletState.MATCHED:
                        continue
                    if len(tracklet.entries) >= int(getattr(self.settings, "tracklet_min_entries", 4)):
                        self._schedule_tracklet_processing(
                            tracklet,
                            reserved_person_ids=set(),
                            allow_tentative_fallback=True,
                        )
                    else:
                        self._track_inflight(asyncio.ensure_future(
                            self._process_short_fragment_tracklet(
                                tracklet,
                                reason="end_of_stream_short_tracklet",
                            )
                        ))

                if not getattr(self, "_inflight", set()):
                    break
                drained = await self._drain_inflight_tasks(timeout_s=finalization_timeout_s)
                if not drained:
                    break

            if getattr(self, "_inflight", set()):
                await self._drain_inflight_tasks(timeout_s=finalization_timeout_s)

            if getattr(self.tracklet_buffer, "tracklets", {}):
                log.warning(
                    "stream_finalization_dropped_unflushed_tracklets",
                    remaining=len(self.tracklet_buffer.tracklets),
                    processing=len(getattr(self, "processing_tracklet_ids", set())),
                )
            await self._reconcile_recent_persons(
                max_persons=int(getattr(self.settings, "background_reconciler_max_persons", 50)),
                passes=int(getattr(self.settings, "final_reconciler_passes", 3)),
                reason="end_of_stream",
            )
            await self._filter_static_artifact_persons()
            self.tracklet_buffer.tracklets.clear()
            getattr(self, "untracked_detection_clusters", []).clear()
            getattr(self, "fragment_recovery_clusters", []).clear()
            self._reset_stream_state()
            log.info("stream_finalization_completed", device_id=device_id)
        finally:
            self._stream_finalizing = False

    async def _filter_static_artifact_persons(self) -> None:
        """Drop false-positive identities that are actually STATIC objects (e.g. a
        fire extinguisher): a person whose bbox centroid barely moved across all
        its tracklets, with a small bbox. Person-level + size-gated so tall moving
        or standing people are never removed. Robust to per-tracklet jitter that
        evades the per-tracklet `_should_suppress_new_identity` guard."""
        if not bool(getattr(self.settings, "static_person_filter_enabled", True)):
            return
        ratio = float(getattr(self.settings, "static_person_max_centroid_spread_ratio", 1.5))
        max_w = float(getattr(self.settings, "static_artifact_max_mean_width_px", 130.0))
        max_h = float(getattr(self.settings, "static_artifact_max_mean_height_px", 260.0))
        min_tracklets = int(getattr(self.settings, "static_person_min_tracklets", 2))
        zm_x = float(getattr(self.settings, "static_person_zero_motion_max_spread_x_px", 150.0))
        zm_y = float(getattr(self.settings, "static_person_zero_motion_max_spread_y_px", 45.0))
        zm_min_tracklets = int(getattr(self.settings, "static_person_zero_motion_min_tracklets", 1))
        try:
            person_ids = await self.mongo.list_recent_person_ids(
                limit=int(getattr(self.settings, "background_reconciler_max_persons", 50))
            )
        except Exception:
            return
        for pid in person_ids:
            try:
                m = await self.mongo.person_motion_extent(int(pid))
            except Exception:
                continue
            if not m:
                continue
            tracklets = int(m["tracklet_count"])
            spread_x = float(m["spread_x"])
            spread_y = float(m["spread_y"])
            mean_w = max(float(m["mean_width"]), 1.0)
            mean_h = max(float(m["mean_height"]), 1.0)
            # Branch A (original): small bbox (not a tall person) AND centroid
            # barely moved relative to its size — catches small static objects.
            small_static = (
                tracklets >= min_tracklets
                and mean_w <= max_w and mean_h <= max_h
                and spread_x <= ratio * mean_w
                and spread_y <= ratio * mean_h
            )
            # Branch B (new): near-zero ABSOLUTE motion, size-INDEPENDENT — catches
            # LARGE fixed objects (e.g. a door) a person walking never matches.
            zero_motion = (
                zm_x > 0.0 and zm_y > 0.0
                and tracklets >= zm_min_tracklets
                and spread_x <= zm_x
                and spread_y <= zm_y
            )
            if small_static or zero_motion:
                log.warning(
                    "static_artifact_person_removed",
                    person_id=int(pid),
                    branch="zero_motion" if zero_motion else "small_static",
                    spread_x=round(spread_x, 1),
                    spread_y=round(spread_y, 1),
                    mean_width=round(mean_w, 1),
                    mean_height=round(mean_h, 1),
                    tracklets=tracklets,
                )
                try:
                    await self.mongo.remove_person(int(pid))
                    await self.redis_cache.invalidate(int(pid))
                except Exception:
                    log.debug("static_artifact_person_remove_failed", person_id=int(pid), exc_info=True)

    async def _reconcile_recent_persons(self, *, max_persons: int, passes: int, reason: str) -> None:
        if not getattr(self.settings, "duplicate_merge_enabled", False):
            return
        try:
            person_ids = await self.mongo.list_recent_person_ids(limit=max_persons)
        except Exception:
            person_ids = list(self.person_last_observation.keys())[-max_persons:]
        # Iterate in a deterministic (ascending pid) order. The recency order
        # from list_recent_person_ids varies run-to-run with frame timing, which
        # changes the chain-merge sequence (A->B->C vs B->A->C) and makes the
        # final identity set non-deterministic. Sorting removes that variance.
        person_ids = sorted({int(pid) for pid in person_ids})
        for pass_idx in range(max(1, int(passes))):
            if not person_ids:
                return
            merged_count = 0
            for pid in list(person_ids):
                try:
                    merged_pid = await self._maybe_merge_duplicate_person(int(pid))
                    if int(merged_pid) != int(pid):
                        merged_count += 1
                except Exception as exc:
                    log.warning(
                        "reconciler_person_failed",
                        person_id=pid,
                        reason=reason,
                        error=str(exc),
                    )
            try:
                merged_count += await self._reconcile_spatial_split_persons(person_ids)
            except Exception as exc:
                log.warning(
                    "reconciler_spatial_split_failed",
                    reason=reason,
                    error=str(exc),
                )
            log.info(
                "reconciler_pass_completed",
                reason=reason,
                pass_idx=pass_idx,
                candidates=len(person_ids),
                merged_count=merged_count,
            )
            if merged_count <= 0:
                return
            try:
                person_ids = await self.mongo.list_recent_person_ids(limit=max_persons)
            except Exception:
                person_ids = list(self.person_last_observation.keys())[-max_persons:]
            person_ids = sorted({int(pid) for pid in person_ids})

    async def _reconcile_spatial_split_persons(self, person_ids: list[int]) -> int:
        """Repair tracker/detector splits that embedding search does not rank first.

        Occluded crops can be bad enough that the correct duplicate is not the
        nearest Qdrant neighbor. For those cases, use the PDF's temporal and bbox
        continuity cues to propose narrow pairwise merge attempts, then run the
        normal merge guards through _try_merge_candidate.
        """
        unique_ids = sorted({int(pid) for pid in person_ids})
        if len(unique_ids) < 2:
            return 0

        counts = await asyncio.gather(
            *(self.mongo.count_tracklets(pid) for pid in unique_ids)
        )
        count_by_pid = dict(zip(unique_ids, counts))
        merged_count = 0
        removed: set[int] = set()
        for idx, person_a in enumerate(unique_ids):
            if person_a in removed:
                continue
            for person_b in unique_ids[idx + 1:]:
                if person_b in removed:
                    continue
                current_count, candidate_count = await asyncio.gather(
                    self.mongo.count_tracklets(person_a),
                    self.mongo.count_tracklets(person_b),
                )
                score = await asyncio.to_thread(
                    self.qdrant_store.person_pair_similarity,
                    person_a,
                    person_b,
                )
                if not await self._persons_have_soft_split_transition(
                    person_a,
                    person_b,
                    score=score,
                    current_count=int(current_count),
                    candidate_count=int(candidate_count),
                ):
                    continue
                result = await self._try_merge_candidate(
                    person_a,
                    (person_b, float(score), None),
                )
                if result.merged:
                    merged_count += 1
                    source_candidates = {person_a, person_b} - {int(result.person_id)}
                    removed.update(source_candidates)
                    break
        return merged_count

    def _reset_stream_state(self) -> None:
        """Clear short-lived tracking state while keeping persisted identity stores."""
        tracker_args = SimpleNamespace(
            track_high_thresh=self.settings.track_high_thresh,
            track_low_thresh=self.settings.track_low_thresh,
            match_thresh=self.settings.match_thresh,
            new_track_thresh=self.settings.new_track_thresh,
            track_buffer=self.settings.track_buffer,
            fuse_score=self.settings.fuse_score,
        )
        self.tracker = BYTETracker(tracker_args, frame_rate=30)
        self.prev_bboxes.clear()
        self.track_id_to_person_id.clear()
        self.track_metadata.clear()
        self.track_last_seen_ns.clear()
        self.person_last_observation.clear()
        self.current_track_metrics.clear()
        self.track_forbidden_person_ids.clear()
        self.track_cooccurrence_counts.clear()
        self.occlusion_candidate_track_ids.clear()
        self.processing_tracklet_ids.clear()
        self._tracklet_embedding_cache.clear()
        self._track_id_split_counts.clear()
        self._tracklet_gate_last_check_frame.clear()

    async def _drain_inflight_tasks(self, *, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(float(timeout_s), 0.1)
        while getattr(self, "_inflight", set()):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.warning("inflight_drain_timeout", pending=len(self._inflight))
                return False
            pending = list(self._inflight)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=min(remaining, 5.0),
                )
            except asyncio.TimeoutError:
                continue
        return True

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

    def _schedule_tracklet_processing(
        self,
        tracklet,
        *,
        reserved_person_ids: set[int],
        allow_tentative_fallback: bool = True,
    ) -> bool:
        if not hasattr(self, "processing_tracklet_ids"):
            self.processing_tracklet_ids = set()
        if tracklet.track_id in self.processing_tracklet_ids:
            return False

        self.processing_tracklet_ids.add(tracklet.track_id)
        task = asyncio.ensure_future(
            self._process_tracklet(
                tracklet,
                reserved_person_ids=reserved_person_ids,
                allow_tentative_fallback=allow_tentative_fallback,
            )
        )
        self._track_inflight(task)

        def _release_processing(done_task: asyncio.Future) -> None:
            self.processing_tracklet_ids.discard(tracklet.track_id)
            if done_task.cancelled():
                return
            try:
                done_task.result()
            except Exception:
                log.error(
                    "tracklet_processing_task_failed",
                    track_id=tracklet.track_id,
                    exc_info=True,
                )

        task.add_done_callback(_release_processing)
        return True

    async def _flush_idle_tracklets_if_needed(self, current_time_ns: int) -> None:
        if not getattr(self.settings, "tracklet_idle_flush_enabled", True):
            return
        if self.last_message_time_ns <= 0:
            return
        idle_ns = int(float(getattr(self.settings, "tracklet_idle_flush_seconds", 1.5)) * 1e9)
        if current_time_ns - self.last_message_time_ns < idle_ns:
            return
        if self.last_idle_flush_ns and current_time_ns - self.last_idle_flush_ns < idle_ns:
            return
        self.last_idle_flush_ns = current_time_ns

        tracklets = list(getattr(self.tracklet_buffer, "tracklets", {}).values())
        if not tracklets:
            return
        log.info("idle_tracklet_flush_started", tracklet_count=len(tracklets))
        for tracklet in tracklets:
            if tracklet.person_id is not None or tracklet.state == TrackletState.MATCHED:
                self.tracklet_buffer.remove(tracklet.track_id)
                continue
            if tracklet.track_id in getattr(self, "processing_tracklet_ids", set()):
                continue
            self.tracklet_buffer.remove(tracklet.track_id)
            if len(tracklet.entries) >= int(getattr(self.settings, "tracklet_min_entries", 4)):
                self._schedule_tracklet_processing(
                    tracklet,
                    reserved_person_ids=set(),
                    allow_tentative_fallback=False,
                )
            else:
                self._track_inflight(asyncio.ensure_future(
                    self._process_short_fragment_tracklet(
                        tracklet,
                        reason="idle_flush_short_tracklet",
                    )
                ))

    def _prune_fragment_recovery_clusters(self, frame_idx: int) -> None:
        if not hasattr(self, "fragment_recovery_clusters"):
            self.fragment_recovery_clusters = []
        max_gap = int(getattr(self.settings, "fragment_recovery_max_gap_frames", 180))
        self.fragment_recovery_clusters = [
            cluster for cluster in self.fragment_recovery_clusters
            if int(frame_idx) - int(cluster["frame_end"]) <= max_gap
        ]

    def _find_fragment_recovery_cluster(
        self,
        *,
        embedding: np.ndarray,
        bbox_xyxy: list[float],
        frame_start: int,
    ) -> dict | None:
        min_sim = float(getattr(self.settings, "fragment_recovery_min_similarity", 0.62))
        max_gap = int(getattr(self.settings, "fragment_recovery_max_gap_frames", 180))
        max_dist = float(getattr(self.settings, "fragment_recovery_max_center_distance_ratio", 1.8))
        best_cluster = None
        best_score = -1.0
        for cluster in getattr(self, "fragment_recovery_clusters", []):
            if int(frame_start) < int(cluster["frame_end"]):
                continue
            if int(frame_start) - int(cluster["frame_end"]) > max_gap:
                continue
            sim = self._cosine_similarity(embedding, cluster["embedding"])
            if sim < min_sim:
                continue
            center_ratio = self._center_distance_ratio(bbox_xyxy, cluster["last_bbox"])
            if center_ratio > max_dist:
                continue
            score = sim - (0.05 * center_ratio)
            if score > best_score:
                best_cluster = cluster
                best_score = score
        return best_cluster

    def _add_fragment_recovery_candidate(
        self,
        *,
        tracklet,
        embedding: np.ndarray,
        v_avg: float,
        emb_consistency: float,
    ) -> tuple[int | None, dict | None]:
        if not getattr(self.settings, "fragment_recovery_enabled", True):
            return None, None
        entries = list(tracklet.entries or [])
        if not entries:
            return None, None
        if float(v_avg) < float(getattr(self.settings, "fragment_recovery_min_visibility", 0.72)):
            return None, None
        # Fragment recovery must obey the same static-artifact filter as the
        # regular tracklet path before minting identities.
        if self._should_suppress_new_identity(tracklet):
            return None, None

        frame_start = int(entries[0].frame_idx)
        frame_end = int(entries[-1].frame_idx)
        last_bbox = [float(v) for v in entries[-1].bbox_xyxy]
        self._prune_fragment_recovery_clusters(frame_start)
        cluster = self._find_fragment_recovery_cluster(
            embedding=embedding,
            bbox_xyxy=last_bbox,
            frame_start=frame_start,
        )
        if cluster is None:
            cluster = {
                "embedding": embedding.astype(np.float32),
                "fragments": 0,
                "total_entries": 0,
                "track_ids": set(),
                "frame_start": frame_start,
                "frame_end": frame_end,
                "last_bbox": last_bbox,
                "v_sum": 0.0,
                "min_emb_consistency": float(emb_consistency),
            }
            self.fragment_recovery_clusters.append(cluster)

        if int(tracklet.track_id) in cluster["track_ids"]:
            return None, None

        cluster["track_ids"].add(int(tracklet.track_id))
        cluster["fragments"] += 1
        cluster["total_entries"] += len(entries)
        cluster["frame_start"] = min(int(cluster["frame_start"]), frame_start)
        cluster["frame_end"] = max(int(cluster["frame_end"]), frame_end)
        cluster["last_bbox"] = last_bbox
        cluster["v_sum"] += float(v_avg) * len(entries)
        cluster["min_emb_consistency"] = min(float(cluster["min_emb_consistency"]), float(emb_consistency))

        weight_old = max(int(cluster["total_entries"]) - len(entries), 0)
        combined = (cluster["embedding"] * weight_old) + (embedding.astype(np.float32) * len(entries))
        norm = float(np.linalg.norm(combined))
        cluster["embedding"] = combined / norm if norm > 1e-8 else embedding.astype(np.float32)

        min_fragments = int(getattr(self.settings, "fragment_recovery_min_fragments", 2))
        min_entries = int(getattr(self.settings, "fragment_recovery_min_total_entries", 5))
        if cluster["fragments"] < min_fragments or cluster["total_entries"] < min_entries:
            return None, None

        near_threshold = float(getattr(self.settings, "fragment_recovery_near_gallery_threshold", 0.52))
        near_hits = self.qdrant_store.search(cluster["embedding"], top_k=1, score_threshold=near_threshold)
        if near_hits:
            pid, score = near_hits[0]
            return None, {
                "method": "fragment_recovery_deferred_near_gallery",
                "source": "fragment_recovery",
                "similarity_score": float(score),
                "reuse_person_id": int(pid),
                "fragment_count": int(cluster["fragments"]),
                "total_entries": int(cluster["total_entries"]),
            }

        if not self._can_allocate_new_identity(tracklet):
            # Tracklet evidence is stale — don't mint a fresh identity from
            # a fragment-recovery cluster whose last entry was long ago. The
            # cluster stays in self.fragment_recovery_clusters and can promote
            # later if new observations arrive.
            log.info(
                "fragment_recovery_skipped_quiescence",
                fragments=int(cluster.get("fragments", 0)),
                total_entries=int(cluster.get("total_entries", 0)),
            )
            return None, None
        try:
            person_id = self.person_id_allocator.allocate()
        except Exception as err:
            raise PersonIdAllocationError(str(err)) from err

        self.qdrant_store.add_person(
            person_id,
            cluster["embedding"],
            {
                "source": "fragment_recovery",
                "fragment_count": int(cluster["fragments"]),
                "total_entries": int(cluster["total_entries"]),
            },
        )
        self.fragment_recovery_clusters = [
            existing
            for existing in self.fragment_recovery_clusters
            if existing is not cluster
        ]
        return person_id, {
            "method": "new_identity",
            "source": "fragment_recovery",
            "similarity_score": None,
            "fragment_count": int(cluster["fragments"]),
            "total_entries": int(cluster["total_entries"]),
            "frame_start": int(cluster["frame_start"]),
            "frame_end": int(cluster["frame_end"]),
        }

    async def _maybe_merge_duplicate_person(self, person_id: int) -> int:
        """Merge weak duplicate identities using appearance + non-cooccurrence.

        This is intentionally post-hoc: a short fragmented track may be too
        ambiguous to merge immediately, but once both identities have gallery
        evidence we can safely collapse the weaker one if they never co-occurred
        on the same device.

        When the top candidate is blocked by an attribute or cooccurrence guard,
        retry with the next-best candidate up to ``_DUPLICATE_MERGE_MAX_RETRIES``.
        This lets duplicate fragments merge when the closest gallery neighbor is
        not the safest merge target.
        """
        if not getattr(self.settings, "duplicate_merge_enabled", False):
            return person_id

        min_score = float(getattr(self.settings, "duplicate_merge_singleton_min_score", 0.49))
        if bool(getattr(self.settings, "duplicate_merge_temporal_continuity_enabled", False)):
            min_score = min(
                min_score,
                float(getattr(self.settings, "duplicate_merge_temporal_continuity_min_score", 0.85)),
            )
        if bool(getattr(self.settings, "duplicate_merge_adjacent_fragment_enabled", False)):
            min_score = min(
                min_score,
                float(getattr(self.settings, "duplicate_merge_adjacent_fragment_min_score", 0.70)),
            )
        if bool(getattr(self.settings, "duplicate_merge_occlusion_reentry_enabled", False)):
            min_score = min(
                min_score,
                float(getattr(self.settings, "duplicate_merge_occlusion_reentry_min_score", 0.58)),
            )
        if bool(getattr(self.settings, "duplicate_merge_scale_aware_reentry_enabled", False)):
            min_score = min(
                min_score,
                float(getattr(self.settings, "duplicate_merge_scale_aware_reentry_min_score", 0.55)),
            )
        if bool(getattr(self.settings, "duplicate_merge_same_gender_singleton_enabled", False)):
            min_score = min(
                min_score,
                float(getattr(self.settings, "duplicate_merge_same_gender_singleton_min_score", 0.80)),
            )
        if bool(getattr(self.settings, "duplicate_merge_cross_device_enabled", False)):
            min_score = min(
                min_score,
                float(getattr(self.settings, "duplicate_merge_cross_device_min_score", 0.50)),
            )
        min_score = min(
            min_score,
            float(getattr(self.settings, "duplicate_merge_soft_split_override_threshold", 0.75)),
        )
        max_retries = int(getattr(self.settings, "duplicate_merge_max_retries", 3))
        tried: set[int] = set()
        last_result_person_id = person_id

        for _attempt in range(max_retries):
            candidate = await asyncio.to_thread(
                self.qdrant_store.find_duplicate_candidate,
                last_result_person_id,
                min_score=min_score,
                exclude_person_ids=tried,
            )
            if candidate is None:
                return last_result_person_id
            result = await self._try_merge_candidate(
                last_result_person_id, candidate
            )
            if result.merged:
                # Stop after one successful merge — chain-merging is left to the
                # next tracklet's invocation to keep merges traceable and bounded.
                return result.person_id
            if result.gender_blocked or result.retryable_blocked:
                tried.add(candidate[0])
                continue
            return last_result_person_id
        return last_result_person_id

    async def _try_merge_candidate(
        self, person_id: int, candidate: tuple[int, float, float | None]
    ) -> "_MergeAttemptResult":
        candidate_person_id, score, runner_up_score = candidate

        current_count, candidate_count = await asyncio.gather(
            self.mongo.count_tracklets(person_id),
            self.mongo.count_tracklets(candidate_person_id),
        )
        canonical_counter = getattr(self.mongo, "count_canonical_tracklets", None)
        if callable(canonical_counter):
            current_canonical_count, candidate_canonical_count = await asyncio.gather(
                canonical_counter(person_id),
                canonical_counter(candidate_person_id),
            )
        else:
            current_canonical_count, candidate_canonical_count = 0, 0
        weak_limit = int(getattr(self.settings, "duplicate_merge_weak_max_tracklets", 2))
        standard_min_score = float(getattr(self.settings, "duplicate_merge_min_score", 0.535))
        singleton_min_score = float(getattr(self.settings, "duplicate_merge_singleton_min_score", 0.49))
        singleton_min_target_tracklets = int(getattr(self.settings, "duplicate_merge_singleton_min_target_tracklets", 3))
        min_margin = float(getattr(self.settings, "duplicate_merge_min_margin", 0.08))
        is_singleton_merge = current_count <= 1 or candidate_count <= 1
        margin = float("inf") if runner_up_score is None else float(score - runner_up_score)
        established_min_score = float(getattr(
            self.settings, "duplicate_merge_established_min_score", 0.78
        ))
        temporal_continuity_merge = False
        adjacent_fragment_merge = False
        occlusion_reentry_merge = False
        same_gender_singleton_merge = False
        soft_split_reason = await self._persons_have_soft_split_transition(
            person_id,
            candidate_person_id,
            score=score,
            current_count=current_count,
            candidate_count=candidate_count,
        )
        soft_split_merge = soft_split_reason is not None
        # Cross-device (cross-camera) re-link: allow a margin-driven merge below
        # the established-identity threshold when the two identities together span
        # >= 2 cameras and have a clearly-dominant nearest-neighbour relationship.
        # Cross-view same-person similarity is modest (~0.5-0.6) yet the correct
        # identity still ranks #1 with a strong margin; the absolute 0.78 bar would
        # otherwise reject it. Scoped to multi-camera (single-stream spans one
        # device → never fires). The gender / attribute / cooccurrence guards below
        # still apply unchanged, so genuinely-different people remain blocked.
        cross_device_merge = False
        if (
            bool(getattr(self.settings, "duplicate_merge_cross_device_enabled", False))
            and score >= float(getattr(self.settings, "duplicate_merge_cross_device_min_score", 0.50))
            and margin >= float(getattr(self.settings, "duplicate_merge_cross_device_min_margin", 0.12))
            and max(int(current_count), int(candidate_count))
            <= int(getattr(self.settings, "duplicate_merge_cross_device_max_tracklets", 8))
        ):
            cross_device_merge = (
                await self.mongo.persons_distinct_device_count(person_id, candidate_person_id)
            ) >= 2
        low_score_canonical_bridge_reasons = {
            "reentry_bridge",
            "supported_spatial_reentry",
            "occlusion_spatial_rejoin",
            "clothing_supported_reentry",
            "spatial_only_weak_fragment",
            "spatial_continuation",
            "scale_aware_reentry",
        }
        scale_aware_weak_bridge = (
            soft_split_reason == "scale_aware_reentry"
            and min(int(current_count), int(candidate_count)) <= weak_limit
            and max(int(current_count), int(candidate_count))
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_scale_aware_reentry_max_tracklets",
                    8,
                )
            )
        )
        # Unambiguous-match carve-out: a moderate appearance score is trustworthy
        # for a canonical re-entry merge when the candidate is the clear mutual
        # nearest neighbour (large margin to the runner-up). osnet different people
        # sit ~0.5-0.6 and are rarely each other's NN with a big gap, so this links
        # a genuinely-split person (e.g. weak-appearance grey clothing, score 0.76
        # but margin 0.19) without lowering the bar for ambiguous pairs.
        canonical_bridge_margin_ok = (
            float(score) >= float(getattr(self.settings, "similarity_threshold", 0.73))
            and margin >= float(getattr(self.settings, "duplicate_merge_canonical_bridge_min_margin", 0.12))
        )
        if (
            soft_split_reason in low_score_canonical_bridge_reasons
            and int(current_canonical_count) > 0
            and int(candidate_canonical_count) > 0
            # Block a spatial-bridge merge between two canonical identities when
            # appearance is below the established bar (osnet 0.78) AND the match is
            # ambiguous. The margin carve-out below (canonical_bridge_margin_ok)
            # still admits an UNAMBIGUOUS mutual-NN re-entry (e.g. weak-appearance
            # grey clothing at 0.76 with a 0.19 margin) so one person isn't split.
            and float(score) < established_min_score
            and not scale_aware_weak_bridge
            and not canonical_bridge_margin_ok
        ):
            log.info(
                "duplicate_merge_rejected_low_score_bridge_between_canonical_identities",
                person_id=person_id,
                candidate_person_id=candidate_person_id,
                similarity_score=round(float(score), 4),
                soft_split_reason=soft_split_reason,
                current_tracklet_count=int(current_count),
                candidate_tracklet_count=int(candidate_count),
                current_canonical_count=int(current_canonical_count),
                candidate_canonical_count=int(candidate_canonical_count),
            )
            return _MergeAttemptResult(person_id=person_id, retryable_blocked=True)
        if soft_split_reason == "reentry_bridge":
            reentry_bridge_max_tracklets = int(
                getattr(self.settings, "duplicate_merge_reentry_bridge_max_tracklets", 4)
            )
            if max(int(current_count), int(candidate_count)) > reentry_bridge_max_tracklets:
                supported_min_score = float(
                    getattr(
                        self.settings,
                        "duplicate_merge_reentry_bridge_supported_min_score",
                        0.70,
                    )
                )
                supported_min_margin = float(
                    getattr(
                        self.settings,
                        "duplicate_merge_reentry_bridge_supported_min_margin",
                        0.12,
                    )
                )
                if float(score) < supported_min_score or margin < supported_min_margin:
                    log.info(
                        "duplicate_merge_rejected_weak_to_supported_reentry_bridge",
                        person_id=person_id,
                        candidate_person_id=candidate_person_id,
                        similarity_score=round(float(score), 4),
                        runner_up_score=None if runner_up_score is None else round(float(runner_up_score), 4),
                        margin_to_runner_up=None if runner_up_score is None else round(float(margin), 4),
                        current_count=int(current_count),
                        candidate_count=int(candidate_count),
                        min_score=supported_min_score,
                        min_margin=supported_min_margin,
                    )
                    return _MergeAttemptResult(person_id=person_id, retryable_blocked=True)
        if score < standard_min_score:
            target_support = max(current_count, candidate_count)
            is_supported_singleton_merge = (
                is_singleton_merge
                and score >= singleton_min_score
                and target_support >= singleton_min_target_tracklets
            )
            same_gender_singleton_merge = (
                bool(getattr(self.settings, "duplicate_merge_same_gender_singleton_enabled", False))
                and is_singleton_merge
                and score >= float(getattr(self.settings, "duplicate_merge_same_gender_singleton_min_score", 0.80))
                and target_support >= singleton_min_target_tracklets
            )
            frame_gap = None
            if (
                bool(getattr(self.settings, "duplicate_merge_temporal_continuity_enabled", False))
                or bool(getattr(self.settings, "duplicate_merge_adjacent_fragment_enabled", False))
                or bool(getattr(self.settings, "duplicate_merge_occlusion_reentry_enabled", False))
            ):
                frame_gap = await self.mongo.persons_min_frame_gap(person_id, candidate_person_id)
            if (
                bool(getattr(self.settings, "duplicate_merge_temporal_continuity_enabled", False))
                and score >= float(getattr(self.settings, "duplicate_merge_temporal_continuity_min_score", 0.85))
                and min(current_count, candidate_count) <= weak_limit
            ):
                temporal_continuity_merge = (
                    frame_gap is not None
                    and frame_gap > 0
                    and frame_gap <= int(getattr(
                        self.settings,
                        "duplicate_merge_temporal_continuity_max_gap_frames",
                        15,
                    ))
                )
            if (
                not temporal_continuity_merge
                and bool(getattr(self.settings, "duplicate_merge_adjacent_fragment_enabled", False))
                and is_singleton_merge
                and score >= float(getattr(self.settings, "duplicate_merge_adjacent_fragment_min_score", 0.70))
            ):
                adjacent_fragment_merge = (
                    frame_gap is not None
                    and frame_gap > 0
                    and frame_gap <= int(getattr(
                        self.settings,
                        "duplicate_merge_adjacent_fragment_max_gap_frames",
                        3,
                    ))
                )
            if (
                not temporal_continuity_merge
                and not adjacent_fragment_merge
                and bool(getattr(self.settings, "duplicate_merge_occlusion_reentry_enabled", False))
                and is_singleton_merge
                and min(current_count, candidate_count) <= weak_limit
                and score >= float(getattr(self.settings, "duplicate_merge_occlusion_reentry_min_score", 0.58))
            ):
                occlusion_reentry_merge = (
                    frame_gap is not None
                    and frame_gap > 0
                    and frame_gap <= int(getattr(
                        self.settings,
                        "duplicate_merge_occlusion_reentry_max_gap_frames",
                        180,
                    ))
                    and await self._person_observations_close_for_reentry(person_id, candidate_person_id)
                )
            if (
                not temporal_continuity_merge
                and not adjacent_fragment_merge
                and not occlusion_reentry_merge
                and not same_gender_singleton_merge
                and not soft_split_merge
                and not is_supported_singleton_merge
                and not cross_device_merge
                and (
                    not is_singleton_merge
                    or score < singleton_min_score
                    or margin < min_margin
                )
            ):
                return _MergeAttemptResult(person_id=person_id)

        if current_count <= weak_limit and candidate_count <= weak_limit:
            source_person_id = max(person_id, candidate_person_id)
            target_person_id = min(person_id, candidate_person_id)
        elif current_count <= weak_limit:
            source_person_id = person_id
            target_person_id = candidate_person_id
        elif candidate_count <= weak_limit:
            source_person_id = candidate_person_id
            target_person_id = person_id
        elif soft_split_merge:
            # Detector/tracker splits can accumulate several short fragments
            # before the final reconciler runs. Preserve the better-supported
            # identity, but allow the spatially-proven split below the normal
            # established-identity embedding threshold.
            if candidate_count <= current_count:
                source_person_id = candidate_person_id
                target_person_id = person_id
            else:
                source_person_id = person_id
                target_person_id = candidate_person_id
        elif score >= established_min_score or cross_device_merge:
            # Both sides have accumulated evidence (no weak limit). Embedding
            # similarity must clear a higher bar than the weak-merge path, OR be a
            # margin-qualified cross-device (cross-camera) re-link. The
            # cooccurrence + attribute guards below still apply unchanged. Source =
            # smaller-evidence side so the more-anchored ID is preserved.
            if candidate_count <= current_count:
                source_person_id = candidate_person_id
                target_person_id = person_id
            else:
                source_person_id = person_id
                target_person_id = candidate_person_id
        else:
            return _MergeAttemptResult(person_id=person_id)

        source_tracklet_count = int(
            current_count if source_person_id == person_id else candidate_count
        )
        target_tracklet_count = int(
            candidate_count if target_person_id == candidate_person_id else current_count
        )
        if bool(
            getattr(self.settings, "duplicate_merge_weak_to_supported_guard_enabled", True)
        ):
            min_supported_target = int(
                getattr(
                    self.settings,
                    "duplicate_merge_weak_to_supported_min_target_tracklets",
                    5,
                )
            )
            weak_source_into_supported = (
                source_tracklet_count <= weak_limit
                and target_tracklet_count >= min_supported_target
            )
            hard_geometry_reasons = {
                "duplicate_box",
                "boundary_duplicate_box",
                "same_frame_established_duplicate",
                "overlap_spatial_duplicate",
                "trajectory_reentry",
                "ultra_continuity",
                "tight_spatial_reentry",
                "adjacent_tight_continuation",
                "boundary_weak_continuation",
                "scale_aware_reentry",
            }
            if weak_source_into_supported and soft_split_reason not in hard_geometry_reasons:
                max_supported_target = int(
                    getattr(
                        self.settings,
                        "duplicate_merge_weak_to_supported_max_target_tracklets",
                        8,
                    )
                )
                weak_supported_min_score = float(
                    getattr(
                        self.settings,
                        "duplicate_merge_weak_to_supported_min_score",
                        0.82,
                    )
                )
                weak_supported_min_margin = float(
                    getattr(
                        self.settings,
                        "duplicate_merge_weak_to_supported_min_margin",
                        0.12,
                    )
                )
                strong_override_score = float(
                    getattr(
                        self.settings,
                        "duplicate_merge_weak_to_supported_strong_score",
                        0.89,
                    )
                )
                strong_override_margin = float(
                    getattr(
                        self.settings,
                        "duplicate_merge_weak_to_supported_strong_margin",
                        0.18,
                    )
                )
                strong_embedding_override = (
                    float(score) >= strong_override_score
                    and margin >= strong_override_margin
                )
                if (
                    (target_tracklet_count > max_supported_target and not strong_embedding_override)
                    or float(score) < weak_supported_min_score
                    or margin < weak_supported_min_margin
                ):
                    log.info(
                        "duplicate_merge_rejected_weak_to_supported_guard",
                        source_person_id=source_person_id,
                        target_person_id=target_person_id,
                        similarity_score=round(float(score), 4),
                        runner_up_score=None
                        if runner_up_score is None
                        else round(float(runner_up_score), 4),
                        margin_to_runner_up=None
                        if runner_up_score is None
                        else round(float(margin), 4),
                        source_tracklet_count=source_tracklet_count,
                        target_tracklet_count=target_tracklet_count,
                        soft_split_reason=soft_split_reason,
                        min_score=weak_supported_min_score,
                        min_margin=weak_supported_min_margin,
                        max_supported_target=max_supported_target,
                    )
                    return _MergeAttemptResult(person_id=person_id, retryable_blocked=True)

        # Per-sighting gender disagreement check. Robust against contamination:
        # uses individual sighting confidences rather than the per-person voted
        # gender_confidence, which gets silently muted when a person has already
        # absorbed conflicting tracklets. See
        # persons_have_clear_gender_disagreement docstring for the "pure gender"
        # definition. Never overridden by embedding similarity.
        if await self.mongo.persons_have_clear_gender_disagreement(
            source_person_id,
            target_person_id,
            sighting_confidence_threshold=float(
                getattr(self.settings, "gender_tracklet_flip_confidence",
                        getattr(self.settings, "gender_block_sighting_confidence", 0.90))
            ),
            min_consecutive=int(
                getattr(self.settings, "gender_tracklet_min_consecutive", 2)
            ),
        ):
            # Cooccurrence-safe rescue: if two identities never appeared together
            # and have strong embedding similarity, a noisy attribute conflict can
            # be overridden for weak identities. Do not merge two well-established
            # identities on this rule alone.
            cooccurrence_safe_override_threshold = float(
                getattr(self.settings, "duplicate_merge_gender_cooccurrence_override_threshold", 0.70)
            )
            cooccurred = await self.mongo.persons_cooccur(source_person_id, target_person_id)
            soft_split_can_override_gender = soft_split_reason in {
                "duplicate_box",
                "boundary_duplicate_box",
                "overlap_spatial_duplicate",
            }
            if (
                soft_split_can_override_gender
                or (
                    not cooccurred
                    and score >= cooccurrence_safe_override_threshold
                    and current_count <= weak_limit
                    and candidate_count <= weak_limit
                )
            ):
                log.warning(
                    "duplicate_merge_gender_conflict_override_soft_split"
                    if soft_split_can_override_gender
                    else "duplicate_merge_gender_conflict_override_cooccurrence",
                    source_person_id=source_person_id,
                    target_person_id=target_person_id,
                    similarity_score=round(score, 4),
                    threshold=cooccurrence_safe_override_threshold,
                    current_count=current_count,
                    candidate_count=candidate_count,
                )
            else:
                log.info(
                    "duplicate_merge_rejected_gender_conflict",
                    source_person_id=source_person_id,
                    target_person_id=target_person_id,
                    similarity_score=round(score, 4),
                    cooccurred=bool(cooccurred),
                )
                return _MergeAttemptResult(person_id=person_id, gender_blocked=True)

        # Fetch both persons' attributes in ONE round trip and run the local
        # conflict checks against the returned dicts. Saves a Mongo round-trip
        # per merge attempt vs separate persons_have_*_conflict queries.
        attrs_a, attrs_b = await self.mongo.fetch_two_persons_attributes(
            source_person_id, target_person_id
        )
        if same_gender_singleton_merge and not self._attrs_allow_singleton_merge(
            attrs_a,
            attrs_b,
            score=score,
        ):
            log.info(
                "duplicate_merge_rejected_same_gender_singleton_guard",
                source_person_id=source_person_id,
                target_person_id=target_person_id,
                similarity_score=round(score, 4),
            )
            return _MergeAttemptResult(person_id=person_id, retryable_blocked=True)

        # Hard overrides only. At sim ≥ 0.85 (cooccurrence) / 0.80 (attribute),
        # the embedding match is strong enough to override these guards on its
        # own. Below those thresholds, attribute conflict is GENUINE evidence
        # that the two persons are different.
        attr_conflict_present = MongoPersonStore.attributes_have_strong_conflict(attrs_a, attrs_b)
        attr_override_threshold = float(
            getattr(self.settings, "duplicate_merge_attr_override_threshold", 0.80)
        )
        cooccurrence_override_threshold = float(
            getattr(self.settings, "duplicate_merge_cooccurrence_override_threshold", 0.85)
        )

        cooccurrence_guard_overridden = (
            score >= cooccurrence_override_threshold
            and not temporal_continuity_merge
            and not adjacent_fragment_merge
            and not occlusion_reentry_merge
        )
        if cooccurrence_guard_overridden:
            log.info(
                "duplicate_merge_cooccurrence_override_applied",
                source_person_id=source_person_id,
                target_person_id=target_person_id,
                similarity_score=round(score, 4),
                threshold=cooccurrence_override_threshold,
            )
        soft_split_can_override_cooccurrence = soft_split_reason in {
            "duplicate_box",
            "boundary_duplicate_box",
            "same_frame_established_duplicate",
            "overlap_spatial_duplicate",
            "trajectory_reentry",
            "scale_aware_reentry",
        }
        if soft_split_can_override_cooccurrence:
            log.info(
                "duplicate_merge_cooccurrence_override_soft_split",
                source_person_id=source_person_id,
                target_person_id=target_person_id,
                similarity_score=round(score, 4),
                soft_split_reason=soft_split_reason,
            )
        elif (
            not cooccurrence_guard_overridden
            and await self.mongo.persons_cooccur(source_person_id, target_person_id)
        ):
            log.info(
                "duplicate_merge_rejected_cooccurrence",
                source_person_id=source_person_id,
                target_person_id=target_person_id,
                similarity_score=round(score, 4),
            )
            return _MergeAttemptResult(person_id=person_id, retryable_blocked=True)

        soft_split_can_override_attributes = soft_split_reason in {
            "duplicate_box",
            "boundary_duplicate_box",
            "same_frame_established_duplicate",
            "overlap_spatial_duplicate",
            "trajectory_reentry",
            "scale_aware_reentry",
        }
        if score >= attr_override_threshold or soft_split_can_override_attributes:
            if attr_conflict_present:
                log.info(
                    "duplicate_merge_attr_override_soft_split"
                    if soft_split_can_override_attributes
                    else "duplicate_merge_attr_override_applied",
                    source_person_id=source_person_id,
                    target_person_id=target_person_id,
                    similarity_score=round(score, 4),
                    threshold=attr_override_threshold,
                    soft_split_reason=soft_split_reason,
                )
        elif attr_conflict_present:
            log.info(
                "duplicate_merge_rejected_attribute_conflict",
                source_person_id=source_person_id,
                target_person_id=target_person_id,
                similarity_score=round(score, 4),
                runner_up_score=None if runner_up_score is None else round(runner_up_score, 4),
            )
            return _MergeAttemptResult(person_id=person_id, retryable_blocked=True)

        reason = {
            "method": "temporal_continuity_gallery_merge" if temporal_continuity_merge else "adjacent_fragment_gallery_merge" if adjacent_fragment_merge else "occlusion_reentry_gallery_merge" if occlusion_reentry_merge else "soft_split_gallery_merge" if soft_split_merge else "same_gender_singleton_gallery_merge" if same_gender_singleton_merge else "singleton_reentry_gallery_merge" if is_singleton_merge and score < standard_min_score else "weak_identity_gallery_merge",
            "similarity_score": round(float(score), 4),
            "runner_up_score": None if runner_up_score is None else round(float(runner_up_score), 4),
            "margin_to_runner_up": None if runner_up_score is None else round(float(margin), 4),
            "source_tracklet_count": source_tracklet_count,
            "target_tracklet_count": target_tracklet_count,
        }
        if soft_split_reason is not None:
            reason["soft_split_reason"] = soft_split_reason
        # Final within-camera color veto: even when a soft_split/spatial bridge
        # overrode the cosine gate (e.g. scale_aware_reentry / trajectory_reentry
        # merged at 0.87), don't glue two established persons whose same-camera
        # torso color clearly differs. Color is the last word; cross-camera merges
        # (no shared-device evidence) abstain so cross-view linking is preserved.
        if self._persons_color_conflict(source_person_id, target_person_id):
            log.info(
                "duplicate_merge_rejected_color_conflict",
                source_person_id=source_person_id,
                target_person_id=target_person_id,
                similarity_score=round(float(score), 4),
                soft_split_reason=soft_split_reason,
            )
            return _MergeAttemptResult(person_id=person_id, retryable_blocked=True)
        await asyncio.to_thread(
            self.qdrant_store.merge_person_gallery,
            source_person_id,
            target_person_id,
        )
        await self.mongo.merge_person(
            source_person_id=source_person_id,
            target_person_id=target_person_id,
            reason=reason,
        )
        await asyncio.gather(
            self.redis_cache.invalidate(source_person_id),
            self.redis_cache.invalidate(target_person_id),
        )

        for track_id, mapped_person_id in list(self.track_id_to_person_id.items()):
            if mapped_person_id == source_person_id:
                self.track_id_to_person_id[track_id] = target_person_id
        source_obs = self.person_last_observation.pop(source_person_id, None)
        if source_obs is not None:
            self.person_last_observation[target_person_id] = source_obs
        log.info(
            "duplicate_identity_merged",
            source_person_id=source_person_id,
            target_person_id=target_person_id,
            similarity_score=round(score, 4),
            # Full provenance so a "3-in-1" run can be traced to the exact merges
            # and the guard/path that allowed each one (Phase B seed pinning).
            method=reason.get("method"),
            runner_up_score=reason.get("runner_up_score"),
            margin_to_runner_up=reason.get("margin_to_runner_up"),
            soft_split_reason=reason.get("soft_split_reason"),
            source_tracklet_count=reason.get("source_tracklet_count"),
            target_tracklet_count=reason.get("target_tracklet_count"),
        )
        new_pid = target_person_id if person_id == source_person_id else person_id
        return _MergeAttemptResult(person_id=new_pid, merged=True)

    async def _persons_have_soft_split_transition(
        self,
        person_a: int,
        person_b: int,
        *,
        score: float,
        current_count: int,
        candidate_count: int,
    ) -> str | None:
        appearance_threshold = float(
            getattr(self.settings, "duplicate_merge_soft_split_override_threshold", 0.75)
        )
        max_weak_count = int(
            getattr(self.settings, "duplicate_merge_soft_split_max_weak_tracklets", 4)
        )
        if min(int(current_count), int(candidate_count)) > max_weak_count:
            return None

        try:
            transition = await self.mongo.persons_min_frame_gap_with_bboxes(
                int(person_a),
                int(person_b),
            )
        except AttributeError:
            return None
        if not transition:
            return None

        standard_max_gap = int(
            getattr(
                self.settings,
                "duplicate_merge_temporal_continuity_max_gap_frames",
                15,
            )
        )
        bbox_a = transition.get("bbox_a") or []
        bbox_b = transition.get("bbox_b") or []
        if len(bbox_a) < 4 or len(bbox_b) < 4:
            return None
        gap = int(transition.get("gap", 10**9))
        bbox_a = [float(v) for v in bbox_a]
        bbox_b = [float(v) for v in bbox_b]
        center_ratio = self._center_distance_ratio(
            bbox_a,
            bbox_b,
        )
        # Same-frame duplicate boxes are the tracker/detector equivalent of one
        # physical person being split into two IDs. In that geometry, the crop
        # can differ enough that ReID cosine is low, so require stronger bbox
        # overlap and a weak side instead of a high embedding score.
        weak_limit = int(getattr(self.settings, "duplicate_merge_weak_max_tracklets", 2))
        max_center_ratio = float(
            getattr(self.settings, "duplicate_merge_soft_split_max_center_distance_ratio", 0.35)
        )
        duplicate_iou_threshold = float(
            getattr(self.settings, "duplicate_merge_soft_split_duplicate_iou_threshold", 0.45)
        )
        if gap == 0 and min(int(current_count), int(candidate_count)) <= weak_limit:
            iou = self._bbox_iou(bbox_a, bbox_b)
            duplicate_box_multitrack_min_score = float(
                getattr(
                    self.settings,
                    "duplicate_merge_soft_split_duplicate_box_multitrack_min_score",
                    0.58,
                )
            )
            duplicate_box_score_allowed = (
                min(int(current_count), int(candidate_count)) <= 1
                or float(score) >= duplicate_box_multitrack_min_score
            )
            if (
                duplicate_box_score_allowed
                and (
                    iou >= duplicate_iou_threshold
                    or center_ratio <= (max_center_ratio * 0.6)
                )
            ):
                return "duplicate_box"
            boundary_duplicate_min_score = float(
                getattr(self.settings, "duplicate_merge_boundary_duplicate_min_score", 0.68)
            )
            boundary_duplicate_min_iou = float(
                getattr(self.settings, "duplicate_merge_boundary_duplicate_min_iou", 0.10)
            )
            boundary_duplicate_max_center_ratio = float(
                getattr(
                    self.settings,
                    "duplicate_merge_boundary_duplicate_max_center_distance_ratio",
                    0.45,
                )
            )
            if (
                float(score) >= boundary_duplicate_min_score
                and iou >= boundary_duplicate_min_iou
                and center_ratio <= boundary_duplicate_max_center_ratio
                and self._bboxes_share_frame_boundary(bbox_a, bbox_b)
            ):
                return "boundary_duplicate_box"

        if (
            bool(
                getattr(
                    self.settings,
                    "duplicate_merge_overlap_spatial_duplicate_enabled",
                    True,
                )
            )
            and float(score)
            >= float(
                getattr(
                    self.settings,
                    "duplicate_merge_overlap_spatial_duplicate_min_score",
                    0.58,
                )
            )
            and max(int(current_count), int(candidate_count))
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_overlap_spatial_duplicate_max_tracklets",
                    24,
                )
            )
        ):
            try:
                overlap_transition = (
                    await self.mongo.persons_closest_spatial_transition_with_bboxes(
                        int(person_a),
                        int(person_b),
                        max_gap_frames=int(
                            getattr(
                                self.settings,
                                "duplicate_merge_overlap_spatial_duplicate_max_gap_frames",
                                4,
                            )
                        ),
                    )
                )
            except AttributeError:
                overlap_transition = None
            if overlap_transition:
                overlap_gap = int(overlap_transition.get("gap", 10**9))
                overlap_bbox_a = overlap_transition.get("bbox_a") or []
                overlap_bbox_b = overlap_transition.get("bbox_b") or []
                if len(overlap_bbox_a) >= 4 and len(overlap_bbox_b) >= 4:
                    overlap_bbox_a = [float(v) for v in overlap_bbox_a]
                    overlap_bbox_b = [float(v) for v in overlap_bbox_b]
                    overlap_center_ratio = self._center_distance_ratio(
                        overlap_bbox_a,
                        overlap_bbox_b,
                    )
                    width_a = max(float(overlap_bbox_a[2] - overlap_bbox_a[0]), 1.0)
                    height_a = max(float(overlap_bbox_a[3] - overlap_bbox_a[1]), 1.0)
                    width_b = max(float(overlap_bbox_b[2] - overlap_bbox_b[0]), 1.0)
                    height_b = max(float(overlap_bbox_b[3] - overlap_bbox_b[1]), 1.0)
                    size_a = max(width_a, height_a)
                    size_b = max(width_b, height_b)
                    area_a = width_a * height_a
                    area_b = width_b * height_b
                    size_ratio = max(size_a, size_b) / max(min(size_a, size_b), 1.0)
                    area_ratio = max(area_a, area_b) / max(min(area_a, area_b), 1.0)
                    if (
                        overlap_gap
                        <= int(
                            getattr(
                                self.settings,
                                "duplicate_merge_overlap_spatial_duplicate_max_gap_frames",
                                4,
                            )
                        )
                        and overlap_center_ratio
                        <= float(
                            getattr(
                                self.settings,
                                "duplicate_merge_overlap_spatial_duplicate_max_center_distance_ratio",
                                0.08,
                            )
                        )
                        and size_ratio
                        <= float(
                            getattr(
                                self.settings,
                                "duplicate_merge_overlap_spatial_duplicate_max_size_ratio",
                                1.25,
                            )
                        )
                        and area_ratio
                        <= float(
                            getattr(
                                self.settings,
                                "duplicate_merge_overlap_spatial_duplicate_max_area_ratio",
                                1.60,
                            )
                        )
                    ):
                        return "overlap_spatial_duplicate"

        if (
            bool(getattr(self.settings, "duplicate_merge_trajectory_reentry_enabled", True))
            and float(score)
            >= float(
                getattr(
                    self.settings,
                    "duplicate_merge_trajectory_reentry_min_score",
                    0.60,
                )
            )
            and max(int(current_count), int(candidate_count))
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_trajectory_reentry_max_tracklets",
                    24,
                )
            )
        ):
            try:
                trajectory_transition = (
                    await self.mongo.persons_closest_spatial_transition_with_bboxes(
                        int(person_a),
                        int(person_b),
                        max_gap_frames=int(
                            getattr(
                                self.settings,
                                "duplicate_merge_trajectory_reentry_max_gap_frames",
                                240,
                            )
                        ),
                    )
                )
            except AttributeError:
                trajectory_transition = None
            if trajectory_transition:
                trajectory_gap = int(trajectory_transition.get("gap", 10**9))
                trajectory_bbox_a = trajectory_transition.get("bbox_a") or []
                trajectory_bbox_b = trajectory_transition.get("bbox_b") or []
                if len(trajectory_bbox_a) >= 4 and len(trajectory_bbox_b) >= 4:
                    trajectory_bbox_a = [float(v) for v in trajectory_bbox_a]
                    trajectory_bbox_b = [float(v) for v in trajectory_bbox_b]
                    trajectory_center_ratio = self._center_distance_ratio(
                        trajectory_bbox_a,
                        trajectory_bbox_b,
                    )
                    width_a = max(float(trajectory_bbox_a[2] - trajectory_bbox_a[0]), 1.0)
                    height_a = max(float(trajectory_bbox_a[3] - trajectory_bbox_a[1]), 1.0)
                    width_b = max(float(trajectory_bbox_b[2] - trajectory_bbox_b[0]), 1.0)
                    height_b = max(float(trajectory_bbox_b[3] - trajectory_bbox_b[1]), 1.0)
                    size_a = max(width_a, height_a)
                    size_b = max(width_b, height_b)
                    area_a = width_a * height_a
                    area_b = width_b * height_b
                    size_ratio = max(size_a, size_b) / max(min(size_a, size_b), 1.0)
                    area_ratio = max(area_a, area_b) / max(min(area_a, area_b), 1.0)
                    if (
                        trajectory_gap
                        <= int(
                            getattr(
                                self.settings,
                                "duplicate_merge_trajectory_reentry_max_gap_frames",
                                240,
                            )
                        )
                        and trajectory_center_ratio
                        <= float(
                            getattr(
                                self.settings,
                                "duplicate_merge_trajectory_reentry_max_center_distance_ratio",
                                0.06,
                            )
                        )
                        and size_ratio
                        <= float(
                            getattr(
                                self.settings,
                                "duplicate_merge_trajectory_reentry_max_size_ratio",
                                1.30,
                            )
                        )
                        and area_ratio
                        <= float(
                            getattr(
                                self.settings,
                                "duplicate_merge_trajectory_reentry_max_area_ratio",
                                1.80,
                            )
                        )
                    ):
                        attrs_a, attrs_b = await self.mongo.fetch_two_persons_attributes(
                            int(person_a),
                            int(person_b),
                        )
                        if not MongoPersonStore.attributes_have_strong_conflict(
                            attrs_a,
                            attrs_b,
                        ):
                            return "trajectory_reentry"

        if (
            bool(
                getattr(
                    self.settings,
                    "duplicate_merge_same_frame_established_duplicate_enabled",
                    True,
                )
            )
            and gap == 0
            and float(score)
            >= float(
                getattr(
                    self.settings,
                    "duplicate_merge_same_frame_established_duplicate_min_score",
                    0.50,
                )
            )
        ):
            iou = self._bbox_iou(bbox_a, bbox_b)
            width_a = max(float(bbox_a[2] - bbox_a[0]), 1.0)
            height_a = max(float(bbox_a[3] - bbox_a[1]), 1.0)
            width_b = max(float(bbox_b[2] - bbox_b[0]), 1.0)
            height_b = max(float(bbox_b[3] - bbox_b[1]), 1.0)
            size_a = max(width_a, height_a)
            size_b = max(width_b, height_b)
            area_a = width_a * height_a
            area_b = width_b * height_b
            size_ratio = max(size_a, size_b) / max(min(size_a, size_b), 1.0)
            area_ratio = max(area_a, area_b) / max(min(area_a, area_b), 1.0)
            if (
                iou
                >= float(
                    getattr(
                        self.settings,
                        "duplicate_merge_same_frame_established_duplicate_min_iou",
                        0.75,
                    )
                )
                and center_ratio
                <= float(
                    getattr(
                        self.settings,
                        "duplicate_merge_same_frame_established_duplicate_max_center_distance_ratio",
                        0.05,
                    )
                )
                and size_ratio
                <= float(
                    getattr(
                        self.settings,
                        "duplicate_merge_same_frame_established_duplicate_max_size_ratio",
                        1.15,
                    )
                )
                and area_ratio
                <= float(
                    getattr(
                        self.settings,
                        "duplicate_merge_same_frame_established_duplicate_max_area_ratio",
                        1.25,
                    )
                )
            ):
                attrs_a, attrs_b = await self.mongo.fetch_two_persons_attributes(
                    int(person_a),
                    int(person_b),
                )
                if self._attrs_support_reentry_bridge(attrs_a, attrs_b):
                    return "same_frame_established_duplicate"

        if (
            gap <= standard_max_gap
            and float(score) >= appearance_threshold
            and center_ratio <= max_center_ratio
        ):
            return "appearance_continuity"

        # Boundary/occlusion re-entry can create a singleton with a very poor
        # appearance vector. Allow only a short temporal gap, a singleton side,
        # and tight spatial continuity; this is not a general low-score merge.
        spatial_only_center_ratio = float(
            getattr(
                self.settings,
                "duplicate_merge_soft_split_spatial_only_max_center_distance_ratio",
                0.50,
            )
        )
        spatial_only_min_score = float(
            getattr(
                self.settings,
                "duplicate_merge_soft_split_spatial_only_min_score",
                0.52,
            )
        )
        cooccurred = None
        if (
            gap > 0
            and min(int(current_count), int(candidate_count))
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_ultra_continuity_max_weak_tracklets",
                    weak_limit,
                )
            )
            and max(int(current_count), int(candidate_count))
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_ultra_continuity_max_supported_tracklets",
                    8,
                )
            )
            and gap <= int(getattr(self.settings, "duplicate_merge_ultra_continuity_max_gap_frames", 6))
            and center_ratio <= float(
                getattr(
                    self.settings,
                    "duplicate_merge_ultra_continuity_max_center_distance_ratio",
                    0.12,
                )
            )
            and float(score) >= float(getattr(self.settings, "duplicate_merge_ultra_continuity_min_score", 0.50))
        ):
            cooccurred = await self.mongo.persons_cooccur(int(person_a), int(person_b))
            if not cooccurred:
                return "ultra_continuity"

        if (
            bool(getattr(self.settings, "duplicate_merge_tight_spatial_reentry_enabled", True))
            and gap > 0
            and min(int(current_count), int(candidate_count))
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_tight_spatial_reentry_max_weak_tracklets",
                    weak_limit,
                )
            )
            and gap
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_tight_spatial_reentry_max_gap_frames",
                    6,
                )
            )
            and center_ratio
            <= float(
                getattr(
                    self.settings,
                    "duplicate_merge_tight_spatial_reentry_max_center_distance_ratio",
                    0.12,
                )
            )
            and float(score)
            >= float(
                getattr(
                    self.settings,
                    "duplicate_merge_tight_spatial_reentry_min_score",
                    0.50,
                )
            )
        ):
            width_a = max(float(bbox_a[2] - bbox_a[0]), 1.0)
            height_a = max(float(bbox_a[3] - bbox_a[1]), 1.0)
            width_b = max(float(bbox_b[2] - bbox_b[0]), 1.0)
            height_b = max(float(bbox_b[3] - bbox_b[1]), 1.0)
            size_a = max(width_a, height_a)
            size_b = max(width_b, height_b)
            area_a = width_a * height_a
            area_b = width_b * height_b
            size_ratio = max(size_a, size_b) / max(min(size_a, size_b), 1.0)
            area_ratio = max(area_a, area_b) / max(min(area_a, area_b), 1.0)
            if (
                size_ratio
                <= float(
                    getattr(
                        self.settings,
                        "duplicate_merge_tight_spatial_reentry_max_size_ratio",
                        1.15,
                    )
                )
                and area_ratio
                <= float(
                    getattr(
                        self.settings,
                        "duplicate_merge_tight_spatial_reentry_max_area_ratio",
                        1.25,
                    )
                )
            ):
                if cooccurred is None:
                    cooccurred = await self.mongo.persons_cooccur(int(person_a), int(person_b))
                if not cooccurred:
                    attrs_a, attrs_b = await self.mongo.fetch_two_persons_attributes(
                        int(person_a),
                        int(person_b),
                    )
                    if self._attrs_support_reentry_bridge(attrs_a, attrs_b):
                        return "tight_spatial_reentry"

        reentry_bridge_max_tracklets = int(
            getattr(self.settings, "duplicate_merge_reentry_bridge_max_tracklets", 4)
        )
        reentry_bridge_max_supported_tracklets = int(
            getattr(
                self.settings,
                "duplicate_merge_reentry_bridge_max_supported_tracklets",
                reentry_bridge_max_tracklets,
            )
        )
        reentry_bridge_counts_allowed = (
            max(int(current_count), int(candidate_count)) <= reentry_bridge_max_tracklets
            or (
                min(int(current_count), int(candidate_count)) <= reentry_bridge_max_tracklets
                and max(int(current_count), int(candidate_count))
                <= reentry_bridge_max_supported_tracklets
            )
        )
        if (
            bool(getattr(self.settings, "duplicate_merge_supported_spatial_reentry_enabled", True))
            and gap
            >= int(
                getattr(
                    self.settings,
                    "duplicate_merge_supported_spatial_reentry_min_gap_frames",
                    24,
                )
            )
            and gap
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_supported_spatial_reentry_max_gap_frames",
                    90,
                )
            )
            and max(int(current_count), int(candidate_count))
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_supported_spatial_reentry_max_tracklets",
                    8,
                )
            )
            and float(score)
            >= float(
                getattr(
                    self.settings,
                    "duplicate_merge_supported_spatial_reentry_min_score",
                    0.53,
                )
            )
            and center_ratio
            <= float(
                getattr(
                    self.settings,
                    "duplicate_merge_supported_spatial_reentry_max_center_distance_ratio",
                    0.18,
                )
            )
        ):
            width_a = max(float(bbox_a[2] - bbox_a[0]), 1.0)
            height_a = max(float(bbox_a[3] - bbox_a[1]), 1.0)
            width_b = max(float(bbox_b[2] - bbox_b[0]), 1.0)
            height_b = max(float(bbox_b[3] - bbox_b[1]), 1.0)
            size_a = max(width_a, height_a)
            size_b = max(width_b, height_b)
            area_a = width_a * height_a
            area_b = width_b * height_b
            size_ratio = max(size_a, size_b) / max(min(size_a, size_b), 1.0)
            area_ratio = max(area_a, area_b) / max(min(area_a, area_b), 1.0)
            if (
                size_ratio
                <= float(
                    getattr(
                        self.settings,
                        "duplicate_merge_supported_spatial_reentry_max_size_ratio",
                        1.20,
                    )
                )
                and area_ratio
                <= float(
                    getattr(
                        self.settings,
                        "duplicate_merge_supported_spatial_reentry_max_area_ratio",
                        1.80,
                    )
                )
            ):
                if cooccurred is None:
                    cooccurred = await self.mongo.persons_cooccur(int(person_a), int(person_b))
                if not cooccurred:
                    attrs_a, attrs_b = await self.mongo.fetch_two_persons_attributes(
                        int(person_a),
                        int(person_b),
                    )
                    if self._attrs_support_reentry_bridge(attrs_a, attrs_b):
                        return "supported_spatial_reentry"

        if (
            bool(getattr(self.settings, "duplicate_merge_reentry_bridge_enabled", True))
            and gap > 0
            and reentry_bridge_counts_allowed
            and gap >= int(getattr(self.settings, "duplicate_merge_reentry_bridge_min_gap_frames", 30))
            and gap <= int(getattr(self.settings, "duplicate_merge_reentry_bridge_max_gap_frames", 180))
            and min(int(current_count), int(candidate_count)) <= max_weak_count
            and float(score) >= float(getattr(self.settings, "duplicate_merge_reentry_bridge_min_score", 0.535))
            and center_ratio <= float(
                getattr(
                    self.settings,
                    "duplicate_merge_reentry_bridge_max_center_distance_ratio",
                    0.85,
                )
            )
        ):
            if cooccurred is None:
                cooccurred = await self.mongo.persons_cooccur(int(person_a), int(person_b))
            if not cooccurred:
                attrs_a, attrs_b = await self.mongo.fetch_two_persons_attributes(
                    int(person_a),
                    int(person_b),
                )
                if self._attrs_support_reentry_bridge(attrs_a, attrs_b):
                    return "reentry_bridge"

        if (
            bool(getattr(self.settings, "duplicate_merge_high_conf_reentry_enabled", True))
            and gap
            >= int(
                getattr(
                    self.settings,
                    "duplicate_merge_high_conf_reentry_min_gap_frames",
                    30,
                )
            )
            and gap
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_high_conf_reentry_max_gap_frames",
                    120,
                )
            )
            and max(int(current_count), int(candidate_count))
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_high_conf_reentry_max_tracklets",
                    8,
                )
            )
            and float(score)
            >= float(
                getattr(
                    self.settings,
                    "duplicate_merge_high_conf_reentry_min_score",
                    0.86,
                )
            )
        ):
            if cooccurred is None:
                cooccurred = await self.mongo.persons_cooccur(int(person_a), int(person_b))
            if not cooccurred:
                try:
                    high_conf_transition = (
                        await self.mongo.persons_closest_spatial_transition_with_bboxes(
                            int(person_a),
                            int(person_b),
                            max_gap_frames=int(
                                getattr(
                                    self.settings,
                                    "duplicate_merge_high_conf_reentry_max_gap_frames",
                                    120,
                                )
                            ),
                        )
                    )
                except AttributeError:
                    high_conf_transition = None
                if high_conf_transition:
                    hc_gap = int(high_conf_transition.get("gap", 10**9))
                    hc_bbox_a = high_conf_transition.get("bbox_a") or []
                    hc_bbox_b = high_conf_transition.get("bbox_b") or []
                    if len(hc_bbox_a) >= 4 and len(hc_bbox_b) >= 4:
                        hc_bbox_a = [float(v) for v in hc_bbox_a]
                        hc_bbox_b = [float(v) for v in hc_bbox_b]
                        hc_center_ratio = self._center_distance_ratio(
                            hc_bbox_a,
                            hc_bbox_b,
                        )
                        width_a = max(float(hc_bbox_a[2] - hc_bbox_a[0]), 1.0)
                        height_a = max(float(hc_bbox_a[3] - hc_bbox_a[1]), 1.0)
                        width_b = max(float(hc_bbox_b[2] - hc_bbox_b[0]), 1.0)
                        height_b = max(float(hc_bbox_b[3] - hc_bbox_b[1]), 1.0)
                        size_a = max(width_a, height_a)
                        size_b = max(width_b, height_b)
                        area_a = width_a * height_a
                        area_b = width_b * height_b
                        size_ratio = max(size_a, size_b) / max(min(size_a, size_b), 1.0)
                        area_ratio = max(area_a, area_b) / max(min(area_a, area_b), 1.0)
                        if (
                            hc_gap
                            >= int(
                                getattr(
                                    self.settings,
                                    "duplicate_merge_high_conf_reentry_min_gap_frames",
                                    30,
                                )
                            )
                            and hc_gap
                            <= int(
                                getattr(
                                    self.settings,
                                    "duplicate_merge_high_conf_reentry_max_gap_frames",
                                    120,
                                )
                            )
                            and hc_center_ratio
                            <= float(
                                getattr(
                                    self.settings,
                                    "duplicate_merge_high_conf_reentry_max_center_distance_ratio",
                                    0.35,
                                )
                            )
                            and size_ratio
                            <= float(
                                getattr(
                                    self.settings,
                                    "duplicate_merge_high_conf_reentry_max_size_ratio",
                                    1.70,
                                )
                            )
                            and area_ratio
                            <= float(
                                getattr(
                                    self.settings,
                                    "duplicate_merge_high_conf_reentry_max_area_ratio",
                                    2.80,
                                )
                            )
                        ):
                            attrs_a, attrs_b = await self.mongo.fetch_two_persons_attributes(
                                int(person_a),
                                int(person_b),
                            )
                            if self._attrs_support_high_conf_reentry(attrs_a, attrs_b):
                                return "high_conf_reentry_bridge"

        if (
            bool(getattr(self.settings, "duplicate_merge_scale_aware_reentry_enabled", True))
            and gap > 0
            and gap
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_scale_aware_reentry_max_gap_frames",
                    240,
                )
            )
            and max(int(current_count), int(candidate_count))
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_scale_aware_reentry_max_tracklets",
                    8,
                )
            )
            and float(score)
            >= float(
                getattr(
                    self.settings,
                    "duplicate_merge_scale_aware_reentry_min_score",
                    0.55,
                )
            )
        ):
            if cooccurred is None:
                cooccurred = await self.mongo.persons_cooccur(int(person_a), int(person_b))
            if not cooccurred:
                try:
                    scale_transition = (
                        await self.mongo.persons_closest_spatial_transition_with_bboxes(
                            int(person_a),
                            int(person_b),
                            max_gap_frames=int(
                                getattr(
                                    self.settings,
                                    "duplicate_merge_scale_aware_reentry_max_gap_frames",
                                    240,
                                )
                            ),
                        )
                    )
                except AttributeError:
                    scale_transition = None
                if scale_transition:
                    scale_bbox_a = scale_transition.get("bbox_a") or []
                    scale_bbox_b = scale_transition.get("bbox_b") or []
                    if len(scale_bbox_a) >= 4 and len(scale_bbox_b) >= 4:
                        scale_bbox_a = [float(v) for v in scale_bbox_a]
                        scale_bbox_b = [float(v) for v in scale_bbox_b]
                        scale_center_ratio = self._center_distance_ratio(
                            scale_bbox_a,
                            scale_bbox_b,
                        )
                        width_a = max(float(scale_bbox_a[2] - scale_bbox_a[0]), 1.0)
                        height_a = max(float(scale_bbox_a[3] - scale_bbox_a[1]), 1.0)
                        width_b = max(float(scale_bbox_b[2] - scale_bbox_b[0]), 1.0)
                        height_b = max(float(scale_bbox_b[3] - scale_bbox_b[1]), 1.0)
                        size_a = max(width_a, height_a)
                        size_b = max(width_b, height_b)
                        area_a = width_a * height_a
                        area_b = width_b * height_b
                        size_ratio = max(size_a, size_b) / max(min(size_a, size_b), 1.0)
                        area_ratio = max(area_a, area_b) / max(min(area_a, area_b), 1.0)
                        bottom_delta_ratio = abs(float(scale_bbox_a[3] - scale_bbox_b[3])) / max(
                            height_a,
                            height_b,
                            1.0,
                        )
                        if (
                            scale_center_ratio
                            <= float(
                                getattr(
                                    self.settings,
                                    "duplicate_merge_scale_aware_reentry_max_center_distance_ratio",
                                    1.30,
                                )
                            )
                            and bottom_delta_ratio
                            <= float(
                                getattr(
                                    self.settings,
                                    "duplicate_merge_scale_aware_reentry_max_bottom_delta_ratio",
                                    0.08,
                                )
                            )
                            and size_ratio
                            >= float(
                                getattr(
                                    self.settings,
                                    "duplicate_merge_scale_aware_reentry_min_size_ratio",
                                    1.00,
                                )
                            )
                            and size_ratio
                            <= float(
                                getattr(
                                    self.settings,
                                    "duplicate_merge_scale_aware_reentry_max_size_ratio",
                                    2.20,
                                )
                            )
                            and area_ratio
                            <= float(
                                getattr(
                                    self.settings,
                                    "duplicate_merge_scale_aware_reentry_max_area_ratio",
                                    4.00,
                                )
                            )
                        ):
                            attrs_a, attrs_b = await self.mongo.fetch_two_persons_attributes(
                                int(person_a),
                                int(person_b),
                            )
                            if self._attrs_support_scale_aware_reentry(attrs_a, attrs_b):
                                return "scale_aware_reentry"

        if (
            bool(getattr(self.settings, "duplicate_merge_clothing_reentry_enabled", True))
            and gap
            >= int(
                getattr(
                    self.settings,
                    "duplicate_merge_clothing_reentry_min_gap_frames",
                    30,
                )
            )
            and gap
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_clothing_reentry_max_gap_frames",
                    240,
                )
            )
            and min(int(current_count), int(candidate_count))
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_clothing_reentry_max_weak_tracklets",
                    weak_limit,
                )
            )
            and max(int(current_count), int(candidate_count))
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_clothing_reentry_max_supported_tracklets",
                    8,
                )
            )
            and float(score)
            >= float(
                getattr(
                    self.settings,
                    "duplicate_merge_clothing_reentry_min_score",
                    0.515,
                )
            )
        ):
            if cooccurred is None:
                cooccurred = await self.mongo.persons_cooccur(int(person_a), int(person_b))
            if not cooccurred:
                attrs_a, attrs_b = await self.mongo.fetch_two_persons_attributes(
                    int(person_a),
                    int(person_b),
                )
                if self._attrs_support_clothing_reentry(attrs_a, attrs_b):
                    return "clothing_supported_reentry"

        if (
            bool(getattr(self.settings, "duplicate_merge_supported_spatial_reentry_enabled", True))
            and gap
            >= int(
                getattr(
                    self.settings,
                    "duplicate_merge_supported_spatial_reentry_min_gap_frames",
                    24,
                )
            )
            and gap
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_supported_spatial_reentry_max_gap_frames",
                    90,
                )
            )
            and max(int(current_count), int(candidate_count))
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_supported_spatial_reentry_max_tracklets",
                    8,
                )
            )
            and float(score)
            >= float(
                getattr(
                    self.settings,
                    "duplicate_merge_supported_spatial_reentry_min_score",
                    0.53,
                )
            )
            and center_ratio
            <= float(
                getattr(
                    self.settings,
                    "duplicate_merge_supported_spatial_reentry_max_center_distance_ratio",
                    0.18,
                )
            )
        ):
            width_a = max(float(bbox_a[2] - bbox_a[0]), 1.0)
            height_a = max(float(bbox_a[3] - bbox_a[1]), 1.0)
            width_b = max(float(bbox_b[2] - bbox_b[0]), 1.0)
            height_b = max(float(bbox_b[3] - bbox_b[1]), 1.0)
            size_a = max(width_a, height_a)
            size_b = max(width_b, height_b)
            area_a = width_a * height_a
            area_b = width_b * height_b
            size_ratio = max(size_a, size_b) / max(min(size_a, size_b), 1.0)
            area_ratio = max(area_a, area_b) / max(min(area_a, area_b), 1.0)
            if (
                size_ratio
                <= float(
                    getattr(
                        self.settings,
                        "duplicate_merge_supported_spatial_reentry_max_size_ratio",
                        1.20,
                    )
                )
                and area_ratio
                <= float(
                    getattr(
                        self.settings,
                        "duplicate_merge_supported_spatial_reentry_max_area_ratio",
                        1.80,
                    )
                )
            ):
                if cooccurred is None:
                    cooccurred = await self.mongo.persons_cooccur(int(person_a), int(person_b))
                if not cooccurred:
                    attrs_a, attrs_b = await self.mongo.fetch_two_persons_attributes(
                        int(person_a),
                        int(person_b),
                    )
                    if self._attrs_support_reentry_bridge(attrs_a, attrs_b):
                        return "supported_spatial_reentry"

        if (
            bool(getattr(self.settings, "duplicate_merge_occlusion_spatial_rejoin_enabled", False))
            and gap > 0
            and min(int(current_count), int(candidate_count)) <= max_weak_count
            and float(score)
            >= float(
                getattr(
                    self.settings,
                    "duplicate_merge_occlusion_spatial_rejoin_min_score",
                    0.53,
                )
            )
        ):
            if cooccurred is None:
                cooccurred = await self.mongo.persons_cooccur(int(person_a), int(person_b))
            if not cooccurred:
                try:
                    continuation = await self.mongo.persons_closest_spatial_transition_with_bboxes(
                        int(person_a),
                        int(person_b),
                        max_gap_frames=int(
                            getattr(
                                self.settings,
                                "duplicate_merge_occlusion_spatial_rejoin_max_gap_frames",
                                180,
                            )
                        ),
                    )
                except AttributeError:
                    continuation = None
                if continuation:
                    cont_bbox_a = continuation.get("bbox_a") or []
                    cont_bbox_b = continuation.get("bbox_b") or []
                    if len(cont_bbox_a) >= 4 and len(cont_bbox_b) >= 4:
                        cont_bbox_a = [float(v) for v in cont_bbox_a]
                        cont_bbox_b = [float(v) for v in cont_bbox_b]
                        cont_center_ratio = self._center_distance_ratio(
                            cont_bbox_a,
                            cont_bbox_b,
                        )
                        width_a = max(float(cont_bbox_a[2] - cont_bbox_a[0]), 1.0)
                        height_a = max(float(cont_bbox_a[3] - cont_bbox_a[1]), 1.0)
                        width_b = max(float(cont_bbox_b[2] - cont_bbox_b[0]), 1.0)
                        height_b = max(float(cont_bbox_b[3] - cont_bbox_b[1]), 1.0)
                        size_a = max(width_a, height_a)
                        size_b = max(width_b, height_b)
                        area_a = width_a * height_a
                        area_b = width_b * height_b
                        size_ratio = max(size_a, size_b) / max(min(size_a, size_b), 1.0)
                        area_ratio = max(area_a, area_b) / max(min(area_a, area_b), 1.0)
                        tight_spatial = (
                            cont_center_ratio
                            <= float(
                                getattr(
                                    self.settings,
                                    "duplicate_merge_occlusion_spatial_rejoin_tight_center_distance_ratio",
                                    0.42,
                                )
                            )
                            and size_ratio
                            <= float(
                                getattr(
                                    self.settings,
                                    "duplicate_merge_occlusion_spatial_rejoin_max_size_ratio",
                                    1.55,
                                )
                            )
                            and area_ratio
                            <= float(
                                getattr(
                                    self.settings,
                                    "duplicate_merge_occlusion_spatial_rejoin_max_area_ratio",
                                    2.10,
                                )
                            )
                        )
                        precise_spatial = (
                            cont_center_ratio
                            <= float(
                                getattr(
                                    self.settings,
                                    "duplicate_merge_occlusion_spatial_rejoin_max_center_distance_ratio",
                                    0.50,
                                )
                            )
                            and size_ratio
                            <= float(
                                getattr(
                                    self.settings,
                                    "duplicate_merge_occlusion_spatial_rejoin_tight_size_ratio",
                                    1.10,
                                )
                            )
                            and area_ratio
                            <= float(
                                getattr(
                                    self.settings,
                                    "duplicate_merge_occlusion_spatial_rejoin_tight_area_ratio",
                                    2.00,
                                )
                            )
                        )
                        strong_score = float(score) >= float(
                            getattr(
                                self.settings,
                                "duplicate_merge_occlusion_spatial_rejoin_strong_min_score",
                                0.59,
                            )
                        )
                        if (strong_score and tight_spatial) or precise_spatial:
                            attrs_a, attrs_b = await self.mongo.fetch_two_persons_attributes(
                                int(person_a),
                                int(person_b),
                            )
                            if not MongoPersonStore.attributes_have_strong_conflict(
                                attrs_a,
                                attrs_b,
                            ):
                                return "occlusion_spatial_rejoin"

        if min(int(current_count), int(candidate_count)) > 1:
            spatial_only_min_score = max(
                spatial_only_min_score,
                float(
                    getattr(
                        self.settings,
                        "duplicate_merge_soft_split_spatial_only_multitrack_min_score",
                        0.60,
                    )
                ),
            )
        if (
            gap > 0
            and min(int(current_count), int(candidate_count)) <= weak_limit
            and gap <= standard_max_gap
            and float(score) >= spatial_only_min_score
            and center_ratio <= spatial_only_center_ratio
        ):
            if cooccurred is None:
                cooccurred = await self.mongo.persons_cooccur(int(person_a), int(person_b))
            if cooccurred:
                return None
            return "spatial_only_weak_fragment"

        if (
            bool(getattr(self.settings, "duplicate_merge_adjacent_tight_continuation_enabled", True))
            and gap > 0
            and gap
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_adjacent_tight_continuation_max_gap_frames",
                    4,
                )
            )
            and max(int(current_count), int(candidate_count))
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_adjacent_tight_continuation_max_tracklets",
                    8,
                )
            )
            and float(score)
            >= float(
                getattr(
                    self.settings,
                    "duplicate_merge_adjacent_tight_continuation_min_score",
                    0.50,
                )
            )
            and center_ratio
            <= float(
                getattr(
                    self.settings,
                    "duplicate_merge_adjacent_tight_continuation_max_center_distance_ratio",
                    0.06,
                )
            )
        ):
            width_a = max(float(bbox_a[2] - bbox_a[0]), 1.0)
            height_a = max(float(bbox_a[3] - bbox_a[1]), 1.0)
            width_b = max(float(bbox_b[2] - bbox_b[0]), 1.0)
            height_b = max(float(bbox_b[3] - bbox_b[1]), 1.0)
            size_a = max(width_a, height_a)
            size_b = max(width_b, height_b)
            area_a = width_a * height_a
            area_b = width_b * height_b
            size_ratio = max(size_a, size_b) / max(min(size_a, size_b), 1.0)
            area_ratio = max(area_a, area_b) / max(min(area_a, area_b), 1.0)
            if (
                size_ratio
                <= float(
                    getattr(
                        self.settings,
                        "duplicate_merge_adjacent_tight_continuation_max_size_ratio",
                        1.10,
                    )
                )
                and area_ratio
                <= float(
                    getattr(
                        self.settings,
                        "duplicate_merge_adjacent_tight_continuation_max_area_ratio",
                        1.15,
                    )
                )
            ):
                if cooccurred is None:
                    cooccurred = await self.mongo.persons_cooccur(int(person_a), int(person_b))
                if not cooccurred:
                    attrs_a, attrs_b = await self.mongo.fetch_two_persons_attributes(
                        int(person_a),
                        int(person_b),
                    )
                    if (
                        not MongoPersonStore.attributes_have_strong_conflict(attrs_a, attrs_b)
                        and (
                            not self._attrs_have_stable_identity_evidence(attrs_a)
                            or not self._attrs_have_stable_identity_evidence(attrs_b)
                            or self._attrs_support_reentry_bridge(attrs_a, attrs_b)
                        )
                    ):
                        return "adjacent_tight_continuation"

        if (
            bool(getattr(self.settings, "duplicate_merge_boundary_weak_continuation_enabled", True))
            and gap > 0
            and gap
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_boundary_weak_continuation_max_gap_frames",
                    12,
                )
            )
            and (
                min(int(current_count), int(candidate_count))
                <= int(
                    getattr(
                        self.settings,
                        "duplicate_merge_boundary_weak_continuation_max_weak_tracklets",
                        weak_limit,
                    )
                )
                or max(int(current_count), int(candidate_count))
                <= int(
                    getattr(
                        self.settings,
                        "duplicate_merge_boundary_weak_continuation_max_supported_tracklets",
                        8,
                    )
                )
            )
            and max(int(current_count), int(candidate_count))
            <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_boundary_weak_continuation_max_supported_tracklets",
                    8,
                )
            )
            and float(score)
            >= float(
                getattr(
                    self.settings,
                    "duplicate_merge_boundary_weak_continuation_min_score",
                    0.52,
                )
            )
            and center_ratio
            >= float(
                getattr(
                    self.settings,
                    "duplicate_merge_boundary_weak_continuation_min_center_distance_ratio",
                    0.10,
                )
            )
            and center_ratio
            <= float(
                getattr(
                    self.settings,
                    "duplicate_merge_boundary_weak_continuation_max_center_distance_ratio",
                    0.32,
                )
            )
        ):
            width_a = max(float(bbox_a[2] - bbox_a[0]), 1.0)
            height_a = max(float(bbox_a[3] - bbox_a[1]), 1.0)
            width_b = max(float(bbox_b[2] - bbox_b[0]), 1.0)
            height_b = max(float(bbox_b[3] - bbox_b[1]), 1.0)
            size_a = max(width_a, height_a)
            size_b = max(width_b, height_b)
            area_a = width_a * height_a
            area_b = width_b * height_b
            size_ratio = max(size_a, size_b) / max(min(size_a, size_b), 1.0)
            area_ratio = max(area_a, area_b) / max(min(area_a, area_b), 1.0)
            bottom_delta_ratio = abs(float(bbox_a[3] - bbox_b[3])) / max(
                height_a,
                height_b,
                1.0,
            )
            if (
                size_ratio
                <= float(
                    getattr(
                        self.settings,
                        "duplicate_merge_boundary_weak_continuation_max_size_ratio",
                        1.80,
                    )
                )
                and area_ratio
                <= float(
                    getattr(
                        self.settings,
                        "duplicate_merge_boundary_weak_continuation_max_area_ratio",
                        2.30,
                    )
                )
                and bottom_delta_ratio
                <= float(
                    getattr(
                        self.settings,
                        "duplicate_merge_boundary_weak_continuation_max_bottom_delta_ratio",
                        0.03,
                    )
                )
            ):
                if cooccurred is None:
                    cooccurred = await self.mongo.persons_cooccur(int(person_a), int(person_b))
                if not cooccurred:
                    attrs_a, attrs_b = await self.mongo.fetch_two_persons_attributes(
                        int(person_a),
                        int(person_b),
                    )
                    weak_boundary_side = (
                        min(int(current_count), int(candidate_count))
                        <= int(
                            getattr(
                                self.settings,
                                "duplicate_merge_boundary_weak_continuation_max_weak_tracklets",
                                weak_limit,
                            )
                        )
                    )
                    if (
                        not MongoPersonStore.attributes_have_strong_conflict(attrs_a, attrs_b)
                        and (weak_boundary_side or self._attrs_support_reentry_bridge(attrs_a, attrs_b))
                    ):
                        return "boundary_weak_continuation"

        if (
            gap > 0
            and min(int(current_count), int(candidate_count)) <= 1
            and gap <= int(
                getattr(
                    self.settings,
                    "duplicate_merge_singleton_spatial_continuation_max_gap_frames",
                    standard_max_gap,
                )
            )
            and float(score)
            >= float(
                getattr(
                    self.settings,
                    "duplicate_merge_singleton_spatial_continuation_min_score",
                    0.30,
                )
            )
            and center_ratio
            <= float(
                getattr(
                    self.settings,
                    "duplicate_merge_singleton_spatial_continuation_max_center_distance_ratio",
                    spatial_only_center_ratio,
                )
            )
        ):
            width_a = max(float(bbox_a[2] - bbox_a[0]), 1.0)
            height_a = max(float(bbox_a[3] - bbox_a[1]), 1.0)
            width_b = max(float(bbox_b[2] - bbox_b[0]), 1.0)
            height_b = max(float(bbox_b[3] - bbox_b[1]), 1.0)
            size_a = max(width_a, height_a)
            size_b = max(width_b, height_b)
            area_a = width_a * height_a
            area_b = width_b * height_b
            size_ratio = max(size_a, size_b) / max(min(size_a, size_b), 1.0)
            area_ratio = max(area_a, area_b) / max(min(area_a, area_b), 1.0)
            if (
                size_ratio
                <= float(
                    getattr(
                        self.settings,
                        "duplicate_merge_singleton_spatial_continuation_max_size_ratio",
                        1.80,
                    )
                )
                and area_ratio
                <= float(
                    getattr(
                        self.settings,
                        "duplicate_merge_singleton_spatial_continuation_max_area_ratio",
                        2.20,
                    )
                )
            ):
                if cooccurred is None:
                    cooccurred = await self.mongo.persons_cooccur(int(person_a), int(person_b))
                if cooccurred:
                    return None
                return "singleton_spatial_continuation"

        if (
            bool(getattr(self.settings, "duplicate_merge_spatial_continuation_enabled", False))
            and min(int(current_count), int(candidate_count)) <= weak_limit
            and float(score) >= float(
                getattr(self.settings, "duplicate_merge_spatial_continuation_min_score", 0.20)
            )
        ):
            if cooccurred is None:
                cooccurred = await self.mongo.persons_cooccur(int(person_a), int(person_b))
            if cooccurred:
                return None
            try:
                continuation = await self.mongo.persons_closest_spatial_transition_with_bboxes(
                    int(person_a),
                    int(person_b),
                    max_gap_frames=int(
                        getattr(
                            self.settings,
                            "duplicate_merge_spatial_continuation_max_gap_frames",
                            60,
                        )
                    ),
                )
            except AttributeError:
                continuation = None
            if continuation:
                cont_bbox_a = continuation.get("bbox_a") or []
                cont_bbox_b = continuation.get("bbox_b") or []
                if len(cont_bbox_a) >= 4 and len(cont_bbox_b) >= 4:
                    cont_center_ratio = self._center_distance_ratio(
                        [float(v) for v in cont_bbox_a],
                        [float(v) for v in cont_bbox_b],
                    )
                    if cont_center_ratio <= float(
                        getattr(
                            self.settings,
                            "duplicate_merge_spatial_continuation_max_center_distance_ratio",
                            0.30,
                        )
                    ):
                        return "spatial_continuation"
        return None

    @staticmethod
    def _bboxes_share_frame_boundary(box_a: list[float], box_b: list[float]) -> bool:
        # Finite-video demos use full-frame person crops at the image boundary.
        # If two same-frame boxes both touch the same image edge, a detector split
        # can have low IoU while still describing one truncated person.
        boundary_eps = 2.0
        return (
            abs(float(box_a[0]) - float(box_b[0])) <= boundary_eps
            or abs(float(box_a[1]) - float(box_b[1])) <= boundary_eps
            or abs(float(box_a[2]) - float(box_b[2])) <= boundary_eps
            or abs(float(box_a[3]) - float(box_b[3])) <= boundary_eps
        )

    async def _person_observations_close_for_reentry(self, person_a: int, person_b: int) -> bool:
        obs_a = self.person_last_observation.get(int(person_a))
        obs_b = self.person_last_observation.get(int(person_b))
        max_center_ratio = float(
            getattr(
                self.settings,
                "duplicate_merge_occlusion_reentry_max_center_distance_ratio",
                2.0,
            )
        )
        if obs_a and obs_b:
            bbox_a = obs_a.get("bbox_xyxy")
            bbox_b = obs_b.get("bbox_xyxy")
            if bbox_a and bbox_b and len(bbox_a) >= 4 and len(bbox_b) >= 4:
                center_ratio = self._center_distance_ratio(
                    [float(v) for v in bbox_a],
                    [float(v) for v in bbox_b],
                )
                if center_ratio <= max_center_ratio:
                    return True

        try:
            transition = await self.mongo.persons_min_frame_gap_with_bboxes(
                int(person_a),
                int(person_b),
            )
        except AttributeError:
            return False
        if not transition:
            return False
        bbox_a = transition.get("bbox_a") or []
        bbox_b = transition.get("bbox_b") or []
        if len(bbox_a) < 4 or len(bbox_b) < 4:
            return False
        center_ratio = self._center_distance_ratio(
            [float(v) for v in bbox_a],
            [float(v) for v in bbox_b],
        )
        return center_ratio <= max_center_ratio

    def _attrs_have_same_confident_gender(self, attrs_a: dict, attrs_b: dict) -> bool:
        gender_a = attrs_a.get("gender")
        gender_b = attrs_b.get("gender")
        if gender_a not in {"male", "female"} or gender_b not in {"male", "female"}:
            return False
        if gender_a != gender_b:
            return False
        conf_threshold = float(
            getattr(self.settings, "duplicate_merge_same_gender_singleton_gender_confidence", 0.80)
        )
        return (
            float(attrs_a.get("gender_confidence", 0.0) or 0.0) >= conf_threshold
            and float(attrs_b.get("gender_confidence", 0.0) or 0.0) >= conf_threshold
        )

    def _attrs_support_reentry_bridge(self, attrs_a: dict, attrs_b: dict) -> bool:
        if MongoPersonStore.attributes_have_strong_conflict(attrs_a, attrs_b):
            return False

        gender_a = attrs_a.get("gender")
        gender_b = attrs_b.get("gender")
        if gender_a not in {"male", "female"} or gender_b not in {"male", "female"}:
            return False
        if gender_a != gender_b:
            return False

        gender_conf_threshold = float(
            getattr(self.settings, "duplicate_merge_reentry_bridge_gender_confidence", 0.70)
        )
        if (
            float(attrs_a.get("gender_confidence", 0.0) or 0.0) < gender_conf_threshold
            or float(attrs_b.get("gender_confidence", 0.0) or 0.0) < gender_conf_threshold
        ):
            return False

        attr_conf_threshold = 0.70
        stable_tasks = ("backpack", "hat", "lower", "sleeve")
        matches = 0
        for task in stable_tasks:
            label_a = attrs_a.get(task)
            label_b = attrs_b.get(task)
            if not label_a or not label_b or label_a == "unknown" or label_b == "unknown":
                continue
            if label_a != label_b:
                continue
            conf_a = float(attrs_a.get(f"{task}_confidence", 0.0) or 0.0)
            conf_b = float(attrs_b.get(f"{task}_confidence", 0.0) or 0.0)
            if conf_a >= attr_conf_threshold and conf_b >= attr_conf_threshold:
                matches += 1

        return matches >= int(
            getattr(self.settings, "duplicate_merge_reentry_bridge_min_attr_matches", 2)
        )

    def _attrs_support_high_conf_reentry(self, attrs_a: dict, attrs_b: dict) -> bool:
        if MongoPersonStore.attributes_have_strong_conflict(attrs_a, attrs_b):
            return False

        attr_conf_threshold = float(
            getattr(self.settings, "duplicate_merge_high_conf_reentry_attr_confidence", 0.65)
        )
        matches = 0
        for task in ("backpack", "hat", "lower", "sleeve"):
            label_a = attrs_a.get(task)
            label_b = attrs_b.get(task)
            if not label_a or not label_b or label_a == "unknown" or label_b == "unknown":
                continue
            if label_a != label_b:
                continue
            conf_a = float(attrs_a.get(f"{task}_confidence", 0.0) or 0.0)
            conf_b = float(attrs_b.get(f"{task}_confidence", 0.0) or 0.0)
            if conf_a >= attr_conf_threshold and conf_b >= attr_conf_threshold:
                matches += 1

        return matches >= int(
            getattr(self.settings, "duplicate_merge_high_conf_reentry_min_attr_matches", 2)
        )

    def _attrs_support_scale_aware_reentry(self, attrs_a: dict, attrs_b: dict) -> bool:
        if MongoPersonStore.attributes_have_strong_conflict(attrs_a, attrs_b):
            return False

        gender_a = attrs_a.get("gender")
        gender_b = attrs_b.get("gender")
        if (
            gender_a in {"male", "female"}
            and gender_b in {"male", "female"}
            and gender_a != gender_b
        ):
            return False

        attr_conf_threshold = float(
            getattr(self.settings, "duplicate_merge_scale_aware_reentry_attr_confidence", 0.65)
        )
        matches = 0
        for task in ("hat", "lower", "sleeve", "glasses"):
            label_a = attrs_a.get(task)
            label_b = attrs_b.get(task)
            if not label_a or not label_b or label_a == "unknown" or label_b == "unknown":
                continue
            if label_a != label_b:
                continue
            conf_a = float(attrs_a.get(f"{task}_confidence", 0.0) or 0.0)
            conf_b = float(attrs_b.get(f"{task}_confidence", 0.0) or 0.0)
            if conf_a >= attr_conf_threshold and conf_b >= attr_conf_threshold:
                matches += 1

        return matches >= int(
            getattr(self.settings, "duplicate_merge_scale_aware_reentry_min_attr_matches", 2)
        )

    def _attrs_support_clothing_reentry(self, attrs_a: dict, attrs_b: dict) -> bool:
        if MongoPersonStore.attributes_have_strong_conflict(attrs_a, attrs_b):
            return False

        attr_conf_threshold = float(
            getattr(self.settings, "duplicate_merge_clothing_reentry_attr_confidence", 0.70)
        )
        matches = 0
        for task in ("backpack", "hat", "lower", "sleeve"):
            label_a = attrs_a.get(task)
            label_b = attrs_b.get(task)
            if not label_a or not label_b or label_a == "unknown" or label_b == "unknown":
                continue
            if label_a != label_b:
                continue
            conf_a = float(attrs_a.get(f"{task}_confidence", 0.0) or 0.0)
            conf_b = float(attrs_b.get(f"{task}_confidence", 0.0) or 0.0)
            if conf_a >= attr_conf_threshold and conf_b >= attr_conf_threshold:
                matches += 1

        return matches >= int(
            getattr(self.settings, "duplicate_merge_clothing_reentry_min_attr_matches", 3)
        )

    def _attrs_have_stable_identity_evidence(self, attrs: dict) -> bool:
        attr_conf_threshold = 0.70
        for task in ("gender", "backpack", "hat", "lower", "sleeve"):
            label = attrs.get(task)
            if not label or label == "unknown":
                continue
            if float(attrs.get(f"{task}_confidence", 0.0) or 0.0) >= attr_conf_threshold:
                return True
        return False

    def _mask_ambiguous_gender_conflict(
        self,
        person_attrs: dict[str, tuple[str, float]],
        tracklet_attrs: dict[str, tuple[str, float]],
    ) -> dict[str, tuple[str, float]]:
        if not bool(getattr(self.settings, "gender_ambiguous_conflict_enabled", True)):
            return person_attrs
        p_gender, p_conf = person_attrs.get("gender", ("unknown", 0.0))
        t_gender, t_conf = tracklet_attrs.get("gender", ("unknown", 0.0))
        if (
            p_gender in {"male", "female"}
            and t_gender in {"male", "female"}
            and p_gender != t_gender
            and float(t_conf) >= float(
                getattr(self.settings, "gender_ambiguous_conflict_tracklet_confidence", 0.70)
            )
            and float(p_conf) <= float(
                getattr(self.settings, "gender_ambiguous_conflict_max_person_confidence", 0.80)
            )
        ):
            masked = dict(person_attrs)
            masked["gender"] = ("unknown", 0.0)
            return masked
        return person_attrs

    def _attrs_allow_singleton_merge(
        self,
        attrs_a: dict,
        attrs_b: dict,
        *,
        score: float,
    ) -> bool:
        if self._attrs_have_same_confident_gender(attrs_a, attrs_b):
            return True

        gender_a = attrs_a.get("gender")
        gender_b = attrs_b.get("gender")
        if (
            gender_a in {"male", "female"}
            and gender_b in {"male", "female"}
            and gender_a != gender_b
        ):
            return False

        # Missing/low-confidence attributes are not negative identity evidence.
        # For singleton fragments, appearance + no cooccurrence should be allowed
        # to repair splits when the cosine is high enough; attributes will be
        # re-voted after merge from the stronger person's evidence.
        unknown_attr_min_score = float(
            getattr(self.settings, "duplicate_merge_singleton_unknown_attr_min_score", 0.88)
        )
        return float(score) >= unknown_attr_min_score

    async def _persist_occlusion_candidate(
        self,
        tracklet,
        *,
        reason: str,
        matching: dict | None = None,
        selected_entries: list[TrackletEntry] | None = None,
        embedding_consistency: float | None = None,
        min_entries: int | None = None,
        candidate_id_override: str | None = None,
    ) -> None:
        if not getattr(self.settings, "occlusion_candidates_enabled", True):
            return
        max_lag_s = float(getattr(self.settings, "max_new_identity_lag_seconds", 0.0) or 0.0)
        if not self._tracklet_is_fresh_enough(
            tracklet,
            max_lag_s=max_lag_s,
            log_event="occlusion_candidate_blocked_stale_tracklet",
        ):
            return
        # Mirror the identity-mint quiescence guard at
        # _can_mint_new_identity: once stream_quiescence_seconds of
        # wall-clock have passed since the last Kafka message, the
        # stream is over and the worker is just draining backlog. New
        # occlusion-candidate rows in that window are evidence the user
        # cannot act on — and visually they make the UI count keep
        # creeping up for minutes after the video ends. Silently drop
        # them; legitimate mid-stream evidence is already captured.
        quiescence_s = float(getattr(self.settings, "stream_quiescence_seconds", 0.0) or 0.0)
        last_ns = getattr(self, "last_message_time_ns", None)
        if (
            quiescence_s > 0
            and last_ns
            and not getattr(self, "_stream_finalizing", False)
        ):
            if (time.time_ns() - int(last_ns)) > int(quiescence_s * 1e9):
                return
        if not hasattr(self, "occlusion_candidate_track_ids"):
            self.occlusion_candidate_track_ids = set()
        if candidate_id_override is None and int(tracklet.track_id) in self.occlusion_candidate_track_ids:
            return
        entries = list(tracklet.entries or [])
        required_entries = (
            int(min_entries)
            if min_entries is not None
            else int(getattr(self.settings, "occlusion_candidate_min_entries", 2))
        )
        if len(entries) < required_entries:
            return
        max_visibility = max((float(entry.v_score) for entry in entries), default=0.0)
        if max_visibility < float(getattr(self.settings, "occlusion_candidate_min_visibility", 0.45)):
            return

        if candidate_id_override is None:
            self.occlusion_candidate_track_ids.add(int(tracklet.track_id))
        candidate_id = candidate_id_override or (
            f"{self._current_device_id}:track:{int(tracklet.track_id)}:"
            f"{reason}:{int(entries[0].frame_idx)}:{int(entries[-1].frame_idx)}"
        )
        selected_entries = selected_entries or []
        selected_frame_idxs = {int(entry.frame_idx) for entry in selected_entries}
        best_entry = max(entries, key=lambda entry: (entry.v_score, -entry.overlap_ratio))
        preview_entries = sorted(
            entries,
            key=lambda entry: (
                entry.frame_idx not in selected_frame_idxs,
                -entry.v_score,
                entry.overlap_ratio,
            ),
        )[:5]
        if best_entry not in preview_entries:
            preview_entries = [best_entry, *preview_entries[:4]]

        crop_key = ""
        snapshot_crop = (
            best_entry.attribute_crop
            if best_entry.attribute_crop is not None and best_entry.attribute_crop.size > 0
            else best_entry.crop
        )
        if snapshot_crop is not None and snapshot_crop.size > 0:
            ok, buf = cv2.imencode(".jpg", snapshot_crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if ok:
                crop_key = await asyncio.to_thread(
                    self.minio.upload_tracklet_snapshot,
                    candidate_id,
                    buf.tobytes(),
                )

        frame_crop_keys: dict[int, str] = {}
        for entry in preview_entries:
            if entry.crop.size <= 0:
                continue
            ok, buf = cv2.imencode(".jpg", entry.crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            frame_key = await asyncio.to_thread(
                self.minio.upload_tracklet_frame_snapshot,
                candidate_id,
                int(entry.frame_idx),
                buf.tobytes(),
            )
            if frame_key:
                frame_crop_keys[int(entry.frame_idx)] = frame_key

        consistency = compute_tracklet_consistency(entries)
        emb_consistency = float(embedding_consistency or 0.0)
        quality = {
            "v_avg": round(sum(float(entry.v_score) for entry in entries) / len(entries), 4),
            "embedding_consistency": round(emb_consistency, 4),
            "bbox_size_stability": round(consistency.bbox_size_stability, 4),
            "position_stability": round(consistency.position_stability, 4),
            "good_frame_ratio": round(consistency.good_frame_ratio, 4),
            "overall_consistency": round(consistency.overall, 4),
        }
        evidence = {
            "selected_frame_count": len(selected_frame_idxs),
            "selected_frame_indices": sorted(selected_frame_idxs),
            "frame_samples": [
                {
                    "frame_idx": int(entry.frame_idx),
                    "bbox_xyxy": [float(round(v, 2)) for v in entry.bbox_xyxy],
                    "bbox_center_xy": [
                        float(round((entry.bbox_xyxy[0] + entry.bbox_xyxy[2]) / 2.0, 2)),
                        float(round((entry.bbox_xyxy[1] + entry.bbox_xyxy[3]) / 2.0, 2)),
                    ],
                    "bbox_size_wh": [
                        float(round(entry.bbox_xyxy[2] - entry.bbox_xyxy[0], 2)),
                        float(round(entry.bbox_xyxy[3] - entry.bbox_xyxy[1], 2)),
                    ],
                    "visibility_score": float(round(entry.v_score, 4)),
                    "overlap_ratio": float(round(entry.overlap_ratio, 4)),
                    "selected": int(entry.frame_idx) in selected_frame_idxs,
                    "selection_reason": (
                        "selected_consensus_frame"
                        if int(entry.frame_idx) in selected_frame_idxs
                        else reason
                    ),
                    "crop_key": frame_crop_keys.get(int(entry.frame_idx)),
                }
                for entry in preview_entries
            ],
        }
        await self.mongo.add_occlusion_candidate(
            candidate_id=candidate_id,
            track_id=int(tracklet.track_id),
            device_id=self._current_device_id,
            reason=reason,
            status=(
                "attached"
                if isinstance(matching, dict)
                and bool(matching.get("provisional"))
                and matching.get("reuse_person_id") is not None
                else "unconfirmed"
            ),
            frame_start=int(entries[0].frame_idx),
            frame_end=int(entries[-1].frame_idx),
            entry_count=len(entries),
            quality=quality,
            evidence=evidence,
            best_crop_key=crop_key or None,
            matching=matching or {},
        )
        log.info(
            "occlusion_candidate_persisted",
            candidate_id=candidate_id,
            track_id=tracklet.track_id,
            reason=reason,
            entry_count=len(entries),
            max_visibility=round(max_visibility, 4),
        )

    async def _persist_attached_occlusion_evidence(
        self,
        *,
        tracklet,
        tracklet_id: str,
        person_id: int,
        consistency,
        v_avg: float,
        emb_consistency: float,
        selected: list[TrackletEntry],
        matching: dict,
        tracklet_attrs=None,
    ) -> None:
        """Persist occlusion ReID under a person without touching canonical gallery.

        These rows are first-class evidence in the person's detail page, but
        they deliberately do not update the person snapshot, person attributes,
        or Qdrant gallery anchors. Occluded crops are useful for audit/reID
        continuity, not as clean identity exemplars.
        """
        entries = list(tracklet.entries or [])
        if not entries:
            return
        device_id = self._current_device_id
        started_at = datetime.fromtimestamp(entries[0].timestamp_ns / 1e9, tz=timezone.utc)
        ended_at = datetime.fromtimestamp(entries[-1].timestamp_ns / 1e9, tz=timezone.utc)
        selected = list(selected or [])
        selected_frame_idxs = {int(entry.frame_idx) for entry in selected}
        if not selected_frame_idxs:
            selected = [max(entries, key=lambda entry: (entry.v_score, -entry.overlap_ratio))]
            selected_frame_idxs = {int(selected[0].frame_idx)}

        quality = {
            "v_avg": round(float(v_avg), 4),
            "embedding_consistency": round(float(emb_consistency), 4),
            "bbox_size_stability": round(consistency.bbox_size_stability, 4),
            "position_stability": round(consistency.position_stability, 4),
            "good_frame_ratio": round(consistency.good_frame_ratio, 4),
            "overall_consistency": round(consistency.overall, 4),
        }
        evidence = {
            "selected_frame_count": len(selected_frame_idxs),
            "selected_frame_indices": sorted(selected_frame_idxs),
            "frame_samples": [
                {
                    "frame_idx": int(entry.frame_idx),
                    "visibility_score": float(round(entry.v_score, 4)),
                    "overlap_ratio": float(round(entry.overlap_ratio, 4)),
                    "bbox_xyxy": [float(round(v, 2)) for v in entry.bbox_xyxy],
                    "selected": int(entry.frame_idx) in selected_frame_idxs,
                    "selection_reason": (
                        "selected_occlusion_frame"
                        if int(entry.frame_idx) in selected_frame_idxs
                        else "occlusion_context"
                    ),
                    "crop_key": None,
                }
                for entry in entries
            ],
        }

        await asyncio.gather(
            self.mongo.add_tracklet_record(
                tracklet_id=tracklet_id,
                track_id=tracklet.track_id,
                person_id=person_id,
                device_id=device_id,
                state="occlusion_attached",
                frame_start=entries[0].frame_idx,
                frame_end=entries[-1].frame_idx,
                frame_indices=[int(e.frame_idx) for e in entries],
                entry_count=len(entries),
                quality=quality,
                matching={
                    **(matching or {}),
                    "method": (matching or {}).get("method") or "occlusion_provisional_match",
                    "canonical_update_applied": False,
                    "provisional": True,
                },
                evidence=evidence,
                first_bbox_xyxy=[float(v) for v in entries[0].bbox_xyxy],
                last_bbox_xyxy=[float(v) for v in entries[-1].bbox_xyxy],
                best_crop_key=None,
            ),
            self.mongo.add_sighting(
                person_id=person_id,
                device_id=device_id,
                tracklet_id=tracklet_id,
                started_at=started_at,
                ended_at=ended_at,
                entry_count=len(entries),
                quality_score=round(consistency.overall, 4),
                snapshot_key=None,
                attributes={},
            ),
            self.mongo.add_timeline_event(
                person_id=person_id,
                event_type="occlusion_evidence_attached",
                device_id=device_id,
                details={
                    "tracklet_id": tracklet_id,
                    "track_id": int(tracklet.track_id),
                    "similarity_score": (matching or {}).get("similarity_score"),
                    "source": (matching or {}).get("source"),
                },
            ),
            self.redis_cache.invalidate(person_id),
            return_exceptions=True,
        )

        best_entry = max(selected, key=lambda entry: (entry.v_score, -entry.overlap_ratio))
        crop_key = ""
        snapshot_crop = (
            best_entry.attribute_crop
            if best_entry.attribute_crop is not None and best_entry.attribute_crop.size > 0
            else best_entry.crop
        )
        if snapshot_crop is not None and snapshot_crop.size > 0:
            ok, buf = cv2.imencode(".jpg", snapshot_crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if ok:
                crop_key = await asyncio.to_thread(
                    self.minio.upload_tracklet_snapshot,
                    tracklet_id,
                    buf.tobytes(),
                )

        context_preview_entries = sorted(
            (entry for entry in entries if int(entry.frame_idx) not in selected_frame_idxs),
            key=lambda entry: (-float(entry.v_score), float(entry.overlap_ratio), int(entry.frame_idx)),
        )[:4]
        frame_upload_idxs = selected_frame_idxs | {
            int(entry.frame_idx) for entry in context_preview_entries
        }

        frame_crop_keys: dict[int, str] = {}
        for entry in entries:
            if int(entry.frame_idx) not in frame_upload_idxs:
                continue
            if entry.crop.size <= 0:
                continue
            ok, buf = cv2.imencode(".jpg", entry.crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                continue
            frame_key = await asyncio.to_thread(
                self.minio.upload_tracklet_frame_snapshot,
                tracklet_id,
                int(entry.frame_idx),
                buf.tobytes(),
            )
            if frame_key:
                frame_crop_keys[int(entry.frame_idx)] = frame_key

        if crop_key or frame_crop_keys:
            evidence_with_assets = {
                **evidence,
                "frame_samples": [
                    {
                        **sample,
                        "crop_key": frame_crop_keys.get(int(sample["frame_idx"])),
                    }
                    for sample in evidence["frame_samples"]
                ],
            }
            await asyncio.gather(
                self.mongo.update_tracklet_assets(
                    tracklet_id,
                    best_crop_key=crop_key or None,
                    evidence=evidence_with_assets,
                ),
                self.mongo.update_sighting_snapshot(
                    tracklet_id,
                    snapshot_key=crop_key,
                ) if crop_key else asyncio.sleep(0),
                return_exceptions=True,
            )
        log.info(
            "occlusion_evidence_attached",
            track_id=tracklet.track_id,
            tracklet_id=tracklet_id,
            person_id=person_id,
            similarity_score=(matching or {}).get("similarity_score"),
        )

    async def _run_identity_serial(self, coro_fn):
        """Serialize identity-mutating work under one lock so concurrent
        fire-and-forget tasks cannot interleave their match/gallery-update steps
        (the source of run-to-run nondeterminism). The lock is acquired in task
        creation order (asyncio.Lock is FIFO), making processing order
        deterministic. _IDENTITY_SERIAL_ACTIVE (per-task ContextVar) lets a call
        already inside the serialized section run nested helpers without
        re-acquiring."""
        lock = getattr(self, "_identity_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._identity_lock = lock
        if _IDENTITY_SERIAL_ACTIVE.get():
            return await coro_fn()
        async with lock:
            token = _IDENTITY_SERIAL_ACTIVE.set(True)
            try:
                return await coro_fn()
            finally:
                _IDENTITY_SERIAL_ACTIVE.reset(token)

    async def _process_short_fragment_tracklet(self, tracklet, *, reason: str) -> int | None:
        if not getattr(self.settings, "deterministic_processing_enabled", True):
            return await self._process_short_fragment_tracklet_impl(tracklet, reason=reason)
        return await self._run_identity_serial(
            lambda: self._process_short_fragment_tracklet_impl(tracklet, reason=reason)
        )

    async def _process_short_fragment_tracklet_impl(self, tracklet, *, reason: str) -> int | None:
        entries = list(tracklet.entries or [])
        if not entries:
            return None
        if not getattr(self.settings, "fragment_recovery_enabled", True):
            await self._persist_occlusion_candidate(
                tracklet,
                reason=reason,
                matching={"method": "unconfirmed", "source": reason, "similarity_score": None},
            )
            return None

        best_entry = max(entries, key=lambda entry: (entry.v_score, -entry.overlap_ratio))
        if float(best_entry.v_score) < float(getattr(self.settings, "fragment_recovery_min_visibility", 0.72)):
            await self._persist_occlusion_candidate(
                tracklet,
                reason=reason,
                matching={"method": "unconfirmed", "source": reason, "similarity_score": None},
            )
            return None

        ok, buf = cv2.imencode(".jpg", best_entry.crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            await self._persist_occlusion_candidate(
                tracklet,
                reason=reason,
                matching={"method": "unconfirmed", "source": reason, "similarity_score": None},
            )
            return None

        try:
            _, result = await self.model_client.extract_features(
                buf.tobytes(),
                model=getattr(self.settings, "embedding_model", "osnet"),
            )
            embedding = np.array(result["embedding"], dtype=np.float32)
            norm = float(np.linalg.norm(embedding))
            if norm <= 1e-8:
                raise ValueError("zero-norm embedding")
            embedding = embedding / norm
        except Exception as err:
            log.warning("fragment_recovery_feature_failed", track_id=tracklet.track_id, error=str(err))
            await self._persist_occlusion_candidate(
                tracklet,
                reason=reason,
                matching={"method": "unconfirmed", "source": reason, "similarity_score": None},
            )
            return None

        attr_crop = (
            best_entry.attribute_crop
            if best_entry.attribute_crop is not None and best_entry.attribute_crop.size > 0
            else best_entry.crop
        )
        ok_attr, attr_buf = cv2.imencode(".jpg", attr_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok_attr:
            try:
                attrs = await self.model_client.classify_attributes(attr_buf.tobytes())
                if attrs:
                    self.attribute_voter.vote_frame(tracklet.track_id, attrs)
            except Exception:
                pass

        consistency = compute_tracklet_consistency(entries)
        v_avg = float(sum(float(entry.v_score) for entry in entries) / len(entries))
        person_id, matching = self._add_fragment_recovery_candidate(
            tracklet=tracklet,
            embedding=embedding,
            v_avg=v_avg,
            emb_consistency=1.0,
        )
        t_attrs = self.attribute_voter.resolve_tracklet(tracklet.track_id)
        if person_id is None and matching:
            person_id, matching = self._maybe_accept_occlusion_provisional_match(
                tracklet=tracklet,
                matching=matching,
                v_avg=v_avg,
                tracklet_attrs=t_attrs,
                forbidden_person_ids=set(),
                recent_incompatible_person_ids=set(),
                blocked_person_ids=set(),
            )
        if person_id is None:
            await self._persist_occlusion_candidate(
                tracklet,
                reason=reason,
                selected_entries=[best_entry],
                embedding_consistency=1.0,
                matching=matching or {"method": "unconfirmed", "source": reason, "similarity_score": None},
            )
            return None

        tracklet_id = str(uuid.uuid4())
        tracklet.person_id = person_id
        tracklet.state = TrackletState.MATCHED
        self.track_id_to_person_id[tracklet.track_id] = person_id
        self._update_person_last_observation_from_tracklet(person_id, tracklet)
        if bool((matching or {}).get("provisional")):
            p_attrs = self.attribute_voter.person_snapshot(person_id)
        else:
            p_attrs = self.attribute_voter.resolve_person(person_id, t_attrs)
        self.track_metadata[tracklet.track_id] = {
            "tracklet_id": tracklet_id,
            "tracklet_state": tracklet.state.value,
            "snapshot_key": None,
            "visibility_score": round(v_avg, 4),
            "quality": {
                "v_avg": float(round(v_avg, 4)),
                "embedding_consistency": 1.0,
                "overall_consistency": float(round(consistency.overall, 4)),
                "good_frame_ratio": float(round(consistency.good_frame_ratio, 4)),
            },
            "matching": matching or {"method": "new_identity", "source": "fragment_recovery"},
            "attributes": {task: label for task, (label, _) in p_attrs.items()},
        }
        if bool((matching or {}).get("provisional")):
            await self._persist_attached_occlusion_evidence(
                tracklet=tracklet,
                tracklet_id=tracklet_id,
                person_id=person_id,
                consistency=consistency,
                v_avg=v_avg,
                emb_consistency=1.0,
                selected=[best_entry],
                matching=matching,
                tracklet_attrs=t_attrs,
            )
            await self._persist_occlusion_candidate(
                tracklet,
                reason=str(
                    (matching or {}).get("provisional_reason")
                    or (matching or {}).get("source")
                    or "provisional_occlusion_match"
                ),
                selected_entries=[best_entry],
                embedding_consistency=1.0,
                matching=matching,
            )
            log.info(
                "fragment_recovery_provisional_attached",
                track_id=tracklet.track_id,
                person_id=person_id,
                entries=len(entries),
                v_avg=round(v_avg, 4),
            )
            return person_id
        await self._persist_tracklet(
            tracklet=tracklet,
            tracklet_id=tracklet_id,
            person_id=person_id,
            consistency=consistency,
            v_avg=v_avg,
            emb_consistency=1.0,
            best_entry=best_entry,
            selected=[best_entry],
            matching=matching or {"method": "new_identity", "source": "fragment_recovery"},
            person_attrs=p_attrs,
        )
        log.info(
            "fragment_recovery_promoted",
            track_id=tracklet.track_id,
            person_id=person_id,
            entries=len(entries),
            v_avg=round(v_avg, 4),
        )
        return person_id

    async def _process_tracklet(
        self,
        tracklet,
        reserved_person_ids: set[int] | None = None,
        allow_tentative_fallback: bool = True,
    ) -> int | None:
        if not getattr(self.settings, "deterministic_processing_enabled", True):
            return await self._process_tracklet_impl(
                tracklet, reserved_person_ids, allow_tentative_fallback
            )
        return await self._run_identity_serial(
            lambda: self._process_tracklet_impl(
                tracklet, reserved_person_ids, allow_tentative_fallback
            )
        )

    async def _process_tracklet_impl(
        self,
        tracklet,
        reserved_person_ids: set[int] | None = None,
        allow_tentative_fallback: bool = True,
    ) -> int | None:
        synthetic_fast_ready = self._is_synthetic_fast_tracklet_ready(tracklet)
        if not self.topk_selector.is_tracklet_ready(tracklet.entries) and not synthetic_fast_ready:
            tracklet.state = TrackletState.ACTIVE  # allow re-evaluation as new frames arrive
            recent_v = [round(e.v_score, 3) for e in tracklet.entries[-5:]]
            log.warning("tracklet_quality_gate_fail",
                        track_id=tracklet.track_id,
                        entries=len(tracklet.entries),
                        recent_v_scores=recent_v,
                        threshold=getattr(self.topk_selector, "high_quality_threshold", None))
            self._track_inflight(asyncio.ensure_future(
                self._persist_occlusion_candidate(
                    tracklet,
                    reason="quality_gate_fail",
                    matching={"method": "unconfirmed", "source": "quality_gate_fail"},
                )
            ))
            return None
        consistency = compute_tracklet_consistency(tracklet.entries)
        # PDF Bước 2 — reject tracklets that are dimensionally / spatially
        # incoherent before paying for embedding extraction. A bouncy bbox
        # or jumping centroid signals either a polluted track (ByteTrack
        # ID-swap during occlusion blended two people into one track_id)
        # or a detection-quality problem; either way the aggregated
        # embedding cannot be trusted to seed a canonical or steal a match.
        consistency_threshold = float(
            getattr(self.settings, "tracklet_readiness_consistency_threshold", 0.0)
        )
        if consistency.overall < consistency_threshold:
            tracklet.state = TrackletState.ACTIVE
            log.warning(
                "tracklet_consistency_gate_fail",
                track_id=tracklet.track_id,
                entries=len(tracklet.entries),
                consistency_overall=consistency.overall,
                bbox_size_stability=consistency.bbox_size_stability,
                position_stability=consistency.position_stability,
                good_frame_ratio=consistency.good_frame_ratio,
                threshold=consistency_threshold,
            )
            self._track_inflight(asyncio.ensure_future(
                self._persist_occlusion_candidate(
                    tracklet,
                    reason="consistency_gate_fail",
                    matching={"method": "unconfirmed", "source": "consistency_gate_fail"},
                )
            ))
            return None
        selected = self.topk_selector.select(tracklet.entries)

        embeddings, v_scores, overlap_ratios = [], [], []
        best_entry = selected[0] if selected else None
        best_entry_attrs: dict[str, dict] | None = None
        async def _extract_one(entry):
            ok, buf = cv2.imencode(".jpg", entry.crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return None, None, None
            img_bytes = buf.tobytes()
            attr_crop = entry.attribute_crop if entry.attribute_crop is not None and entry.attribute_crop.size > 0 else entry.crop
            ok_attr, attr_buf = cv2.imencode(".jpg", attr_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            attr_bytes = attr_buf.tobytes() if ok_attr else img_bytes
            emb_vec = None
            attrs = None
            try:
                _, result = await self.model_client.extract_features(
                    img_bytes,
                    model=getattr(self.settings, "embedding_model", "osnet"),
                )
                emb_vec = np.array(result["embedding"], dtype=np.float32)
                norm = np.linalg.norm(emb_vec)
                if norm > 1e-8:
                    emb_vec /= norm
                else:
                    emb_vec = None
            except Exception as err:
                log.warning("feature_extraction_failed", error=str(err))
            try:
                attrs = await self.model_client.classify_attributes(attr_bytes)
            except Exception:
                pass
            return emb_vec, entry, attrs

        extracted = []
        for emb_vec, entry, attrs in await asyncio.gather(*[_extract_one(e) for e in selected]):
            if emb_vec is None:
                continue
            extracted.append((emb_vec, entry, attrs))

        if not extracted:
            log.warning("tracklet_no_embeddings", track_id=tracklet.track_id, selected=len(selected))
            tracklet.state = TrackletState.ACTIVE
            self._track_inflight(asyncio.ensure_future(
                self._persist_occlusion_candidate(
                    tracklet,
                    reason="feature_extraction_failed",
                    selected_entries=selected,
                    matching={"method": "unconfirmed", "source": "feature_extraction_failed"},
                )
            ))
            return

        raw_embeddings = [item[0] for item in extracted]
        raw_v_scores = [item[1].v_score for item in extracted]
        consensus_threshold = float(
            getattr(self.settings, "embedding_consensus_threshold", 0.72)
        )
        consensus_indices = _select_embedding_consensus_indices(
            raw_embeddings,
            raw_v_scores,
            similarity_threshold=consensus_threshold,
        )
        min_consensus = int(getattr(self.settings, "min_consensus_embeddings", 2))
        consensus_failed = (
            len(raw_embeddings) >= min_consensus and len(consensus_indices) < min_consensus
        )
        if consensus_failed:
            raw_consistency = WeightedEmbeddingAggregator.compute_embedding_consistency(raw_embeddings)
            log.warning(
                "tracklet_embedding_consensus_fail",
                track_id=tracklet.track_id,
                selected=len(selected),
                embedded=len(raw_embeddings),
                consensus_count=len(consensus_indices),
                raw_embedding_consistency=round(raw_consistency, 4),
                threshold=consensus_threshold,
            )
            # Graceful degrade: pick the top-v_score embedding and let the matcher
            # attempt a gallery match in match-only mode (no new identity). If no
            # match is found, the post-match flow persists it as occlusion candidate.
            top_idx = max(range(len(extracted)), key=lambda i: extracted[i][1].v_score)
            consensus_indices = [top_idx]

        # Loop A — ReID embedding path: stays restricted to consensus_indices.
        # Embedding aggregation + best_entry tracking feed the matcher and the
        # glasses_best_frame_override; touching either would shift ReID quality.
        for idx in consensus_indices:
            emb_vec, entry, attrs = extracted[idx]
            embeddings.append(emb_vec)
            v_scores.append(entry.v_score)
            overlap_ratios.append(entry.overlap_ratio)
            if best_entry is None or entry.v_score > best_entry.v_score:
                best_entry = entry
                best_entry_attrs = attrs
            elif best_entry is entry:
                best_entry_attrs = attrs

        # Loop B — PAR voting: broaden coverage to every extracted entry with
        # sufficient visibility. AttributeVoter does its own confidence-weighted
        # majority, so more frames → more robust against single-frame errors
        # under occlusion. consensus_indices filtering above is an embedding-
        # similarity gate, which is the wrong criterion for attribute coverage.
        par_vote_all = bool(getattr(self.settings, "par_vote_all_extracted", True))
        par_min_v = float(getattr(self.settings, "par_min_v_score", 0.55))
        if par_vote_all:
            vote_entries = [
                (entry, attrs) for _, entry, attrs in extracted
                if attrs and entry.v_score >= par_min_v
            ]
        else:
            vote_entries = [
                (extracted[idx][1], extracted[idx][2])
                for idx in consensus_indices
                if extracted[idx][2]
            ]
        for _entry, _attrs in vote_entries:
            self.attribute_voter.vote_frame(tracklet.track_id, _attrs)

        consensus_entries = [extracted[idx][1] for idx in consensus_indices]
        self.embedded_tracklets += 1
        emb_consistency = WeightedEmbeddingAggregator.compute_embedding_consistency(embeddings)
        tracklet_embedding = self.aggregator.aggregate(embeddings, v_scores, overlap_ratios)
        v_avg = sum(v_scores) / len(v_scores)

        # Resolve tracklet-level attributes (all 8 tasks).
        t_attrs = self.attribute_voter.resolve_tracklet(tracklet.track_id)
        attrs_unreliable = self._is_occlusion_attribute_unreliable(tracklet, v_avg=v_avg)

        glasses_from_best = None
        if isinstance(best_entry_attrs, dict):
            glasses_from_best = best_entry_attrs.get("glasses")
        if isinstance(glasses_from_best, dict):
            glasses_label = glasses_from_best.get("label")
            glasses_conf = float(glasses_from_best.get("confidence", 0.0))
            if (
                isinstance(glasses_label, str)
                and glasses_label in {"glasses", "no_glasses"}
                and glasses_conf >= self.settings.glasses_best_frame_override_threshold
            ):
                t_attrs["glasses"] = (glasses_label, round(glasses_conf, 4))

        try:
            if reserved_person_ids is None:
                reserved_person_ids = set()
            current_person_id = self.track_id_to_person_id.get(tracklet.track_id)
            if current_person_id is None:
                current_person_id = self._find_recent_track_identity(tracklet)
                if current_person_id is not None:
                    self.track_id_to_person_id[tracklet.track_id] = current_person_id
                    log.info(
                        "track_identity_memory_restored",
                        track_id=tracklet.track_id,
                        person_id=current_person_id,
                        frame_start=tracklet.entries[0].frame_idx if tracklet.entries else None,
                    )
            identity_shift_risk = None
            if current_person_id is not None:
                identity_shift_risk = self._tracklet_identity_shift_risk(tracklet)
                if identity_shift_risk is not None:
                    forbidden_person_ids_for_shift = self.track_forbidden_person_ids.setdefault(
                        tracklet.track_id,
                        set(),
                    )
                    forbidden_person_ids_for_shift.add(current_person_id)
                    self.track_id_to_person_id.pop(tracklet.track_id, None)
                    log.warning(
                        "current_identity_shift_risk_rejected",
                        track_id=tracklet.track_id,
                        rejected_person_id=current_person_id,
                        **identity_shift_risk,
                    )
                    current_person_id = None
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
            blocked_duplicate_person_ids = set()
            if tracklet.entries:
                blocked_duplicate_person_ids = self._find_blocked_duplicate_person_ids(
                    tracklet.entries[-1].bbox_xyxy,
                    blocked_person_ids,
                )
            recent_incompatible_person_ids = set()
            forbidden_person_ids = set(self.track_forbidden_person_ids.get(tracklet.track_id, set()))
            if self._has_current_identity_attribute_conflict(t_attrs, current_person_id):
                assert current_person_id is not None
                forbidden_person_ids.add(current_person_id)
                self.track_forbidden_person_ids.setdefault(tracklet.track_id, set()).add(
                    current_person_id
                )
                self.track_id_to_person_id.pop(tracklet.track_id, None)
                log.warning(
                    "current_identity_attribute_conflict_rejected",
                    track_id=tracklet.track_id,
                    rejected_person_id=current_person_id,
                    tracklet_gender=t_attrs.get("gender"),
                    person_snapshot=self.attribute_voter.person_snapshot(current_person_id),
                )
                current_person_id = None
            elif current_person_id is not None:
                forbidden_person_ids.discard(current_person_id)
            if attrs_unreliable:
                log.info(
                    "attribute_incompatible_guard_skipped_occlusion",
                    track_id=tracklet.track_id,
                    tracklet_gender=t_attrs.get("gender"),
                    v_avg=round(float(v_avg), 4),
                )
            else:
                forbidden_person_ids.update(
                    self._find_attribute_incompatible_person_ids(t_attrs, current_person_id)
                )
            # Within-camera color guard at PRIMARY match: forbid same-camera persons
            # whose reference torso color clearly differs (clean-frame gated; abstains
            # on noisy color). Keeps each identity's color anchor pure and stops
            # different-colored people gluing via gallery_match at the embedding ceiling.
            _color_incompatible = self._find_color_incompatible_person_ids(
                tracklet, current_person_id, getattr(self, "_current_device_id", "")
            )
            if _color_incompatible:
                forbidden_person_ids.update(_color_incompatible)
                log.info(
                    "color_incompatible_persons_forbidden",
                    track_id=tracklet.track_id,
                    forbidden=sorted(_color_incompatible),
                    device_id=getattr(self, "_current_device_id", ""),
                )
            # A2: temporal co-active guard — any person_id already bound to a
            # live track in the last 600ms can't also be this tracklet.
            forbidden_person_ids.update(
                self._find_co_active_person_ids(
                    tracklet.track_id,
                    current_person_id,
                    tracklet.entries[-1].timestamp_ns if tracklet.entries else 0,
                )
            )
            static_artifact = self._should_suppress_new_identity(tracklet)
            if static_artifact:
                self.track_id_to_person_id.pop(tracklet.track_id, None)
                self.track_metadata.pop(tracklet.track_id, None)
                if hasattr(self, "tracklet_buffer"):
                    self.tracklet_buffer.remove(tracklet.track_id)
                log.warning(
                    "tracklet_static_artifact_rejected",
                    track_id=tracklet.track_id,
                    current_person_id=current_person_id,
                    entry_count=len(tracklet.entries),
                )
                return None
            allow_new_identity = True
            if current_person_id is None:
                allow_new_identity = (
                    not static_artifact
                    and identity_shift_risk is None
                    and not self._identity_cap_reached()
                    and self._can_allocate_new_identity(tracklet)
                )
                if (
                    allow_new_identity
                    and int(tracklet.track_id) < 0
                    and float(consistency.overall)
                    < float(
                        getattr(
                            self.settings,
                            "synthetic_new_identity_min_overall_consistency",
                            0.75,
                        )
                    )
                ):
                    allow_new_identity = False
                    log.warning(
                        "synthetic_new_identity_blocked_low_consistency",
                        track_id=tracklet.track_id,
                        overall_consistency=round(float(consistency.overall), 4),
                        threshold=float(
                            getattr(
                                self.settings,
                                "synthetic_new_identity_min_overall_consistency",
                                0.75,
                            )
                        ),
                    )
            else:
                # A live track_id already carries temporal evidence for its
                # assigned person. If its current crop no longer clears the
                # gallery/continuity threshold, keep it tentative/occlusion
                # evidence instead of minting a second person from the same
                # tracker identity. Actual ID switches are still handled above
                # by strong gallery matches or explicit attribute conflicts.
                allow_new_identity = False
            # Distinguish the two reasons minting can be denied: an actual
            # identity cap (where falling back to capped_soft_match at the
            # low threshold is correct) vs an unreliable tracklet
            # (consensus failure / static-artifact suppression / cannot
            # allocate). In the unreliable case the right thing is to
            # defer, not to force-merge into the nearest existing person.
            allow_capped_soft_match = self._identity_cap_reached()
            if consensus_failed:
                allow_new_identity = False
            if tracklet.entries:
                recent_incompatible_person_ids = self._find_recent_incompatible_person_ids(
                    tracklet.entries[-1].bbox_xyxy,
                    tracklet.entries[-1].timestamp_ns,
                    current_person_id,
                )

            # Register the new person's attributes in attribute_voter
            # synchronously at allocation time. Without this, concurrent
            # _process_tracklet tasks running their conflict check between
            # this match_tracklet's allocation and the worker's later
            # resolve_person call would see an empty voter entry for the
            # just-allocated pid and miss the attribute-conflict guard —
            # leading to wrong-gender tracklets matching the new identity.
            def _register_new_identity_attrs(new_pid: int) -> None:
                if self._is_occlusion_attribute_unreliable(tracklet, v_avg=v_avg):
                    return
                try:
                    self.attribute_voter.resolve_person(new_pid, t_attrs)
                except Exception:
                    log.warning("on_new_identity_register_failed", person_id=new_pid)

            # PDF Bước 2: max consecutive good frames. Used as an alternative
            # promotion signal inside the matcher when v_avg / consistency dip
            # (the heavily-occluded boundary person scenario).
            good_streak = _compute_max_good_streak(
                tracklet.entries,
                float(getattr(self.settings, "high_quality_threshold", 0.55)),
            )
            # PDF Bước 5 gate #2: count the high-quality frames in the full
            # tracklet buffer (not the consensus-filtered set). The matcher
            # enforces num_high_quality_frames >= min_high_quality_frames
            # as one of the promote-tentative conditions; we compute it here
            # using the same threshold as the readiness selector so both
            # gates see the same metric.
            high_quality_threshold = float(
                getattr(self.settings, "high_quality_threshold", 0.55)
            )
            num_high_quality_frames = sum(
                1 for entry in tracklet.entries if entry.v_score >= high_quality_threshold
            )
            effective_tracklet_len = len(tracklet.entries)
            if synthetic_fast_ready:
                effective_tracklet_len = max(
                    effective_tracklet_len,
                    int(getattr(self.settings, "new_identity_min_tracklet_len", 6)),
                )
            selected_max_overlap = max(
                (float(entry.overlap_ratio or 0.0) for entry in consensus_entries),
                default=0.0,
            )
            allow_gallery_update = (
                selected_max_overlap
                <= float(getattr(self.settings, "gallery_update_max_overlap_ratio", 0.25))
                and float(consistency.overall)
                >= float(getattr(self.settings, "gallery_update_min_overall_consistency", 0.80))
            )
            allow_scale_aux_match = (
                bool(getattr(self.settings, "scale_aux_gallery_enabled", False))
                and current_person_id is None
                and int(tracklet.track_id) < 0
                and float(v_avg) >= float(getattr(self.settings, "scale_aux_min_v", 0.70))
                and float(emb_consistency)
                >= float(getattr(self.settings, "scale_aux_min_consistency", 0.80))
                and effective_tracklet_len
                >= int(getattr(self.settings, "scale_aux_min_tracklet_len", 5))
                and selected_max_overlap
                <= float(getattr(self.settings, "scale_aux_max_overlap_ratio", 0.35))
            )
            # Query the upper-body auxiliary gallery with the current full
            # tracklet embedding. Near-camera crops are already upper-body
            # dominated; cropping them again removes too much discriminative
            # clothing signal.
            scale_aux_embedding = tracklet_embedding if allow_scale_aux_match else None
            person_id = self.matcher.match_tracklet(
                track_id=tracklet.track_id,
                embedding=tracklet_embedding,
                v_avg=v_avg,
                embedding_consistency=emb_consistency,
                tracklet_len=effective_tracklet_len,
                num_high_quality_frames=num_high_quality_frames,
                blocked_person_ids=blocked_person_ids,
                current_person_id=current_person_id,
                reuse_person_id=reuse_person_id,
                blocked_duplicate_person_ids=blocked_duplicate_person_ids,
                forbidden_person_ids=forbidden_person_ids,
                recent_incompatible_person_ids=recent_incompatible_person_ids,
                allow_new_identity=allow_new_identity,
                allow_capped_soft_match=allow_capped_soft_match,
                on_new_identity=_register_new_identity_attrs,
                good_streak=good_streak,
                allow_tentative_fallback=allow_tentative_fallback,
                allow_gallery_update=allow_gallery_update,
                scale_aux_embedding=scale_aux_embedding,
                allow_scale_aux_match=allow_scale_aux_match,
            )
            pop_last_decision = getattr(self.matcher, "pop_last_decision", None)
            matching = pop_last_decision(tracklet.track_id) if callable(pop_last_decision) else {}
            matching = matching or {}
        except PersonIdAllocationError:
            log.error("person_id_allocation_failed", track_id=tracklet.track_id, exc_info=True)
            self.tracklet_buffer.remove(tracklet.track_id)
            return None

        if person_id is None and current_person_id is None:
            provisional_person_id, provisional_matching = self._maybe_accept_occlusion_provisional_match(
                tracklet=tracklet,
                matching=matching,
                v_avg=v_avg,
                tracklet_attrs=t_attrs,
                forbidden_person_ids=forbidden_person_ids,
                recent_incompatible_person_ids=recent_incompatible_person_ids,
                blocked_person_ids=blocked_person_ids,
            )
            if provisional_person_id is not None:
                person_id = provisional_person_id
                matching = provisional_matching

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

        if person_id is None and current_person_id is None:
            recovery_person_id, recovery_matching = self._add_fragment_recovery_candidate(
                tracklet=tracklet,
                embedding=tracklet_embedding,
                v_avg=v_avg,
                emb_consistency=emb_consistency,
            )
            if recovery_matching:
                matching = recovery_matching
            if recovery_person_id is not None:
                person_id = recovery_person_id
            else:
                provisional_person_id, provisional_matching = self._maybe_accept_occlusion_provisional_match(
                    tracklet=tracklet,
                    matching=matching,
                    v_avg=v_avg,
                    tracklet_attrs=t_attrs,
                    forbidden_person_ids=forbidden_person_ids,
                    recent_incompatible_person_ids=recent_incompatible_person_ids,
                    blocked_person_ids=blocked_person_ids,
                )
                if provisional_person_id is not None:
                    person_id = provisional_person_id
                    matching = provisional_matching

        tracklet_id = str(uuid.uuid4())

        if person_id is not None:
            self.matched_tracklets += 1
            tracklet.person_id = person_id
            tracklet.state = TrackletState.MATCHED
            self.track_id_to_person_id[tracklet.track_id] = person_id
            self._update_person_last_observation_from_tracklet(person_id, tracklet)
            reserved_person_ids.add(person_id)  # make visible to other concurrent tasks

            # Emit one attachment_decision line per match so identity decisions
            # can be audited from appearance, attribute, and guard signals.
            try:
                _pre_snap = self.attribute_voter.person_snapshot(person_id) or {}
                _pre_gender, _pre_gconf = _pre_snap.get("gender", ("unknown", 0.0))
                _pre_support = self.attribute_voter.person_task_stable_support(person_id, "gender")
                _t_gender, _t_gconf = (t_attrs or {}).get("gender", ("unknown", 0.0))
                _guard_thresh = float(self.settings.attribute_conflict_tracklet_confidence)
                log.warning(
                    "attachment_decision",
                    track_id=tracklet.track_id,
                    person_id=person_id,
                    method=str(matching.get("method", "unknown")),
                    similarity_score=matching.get("similarity_score"),
                    runner_up_score=matching.get("runner_up_score"),
                    margin=matching.get("margin_to_runner_up"),
                    tracklet_gender=_t_gender,
                    tracklet_gender_conf=round(float(_t_gconf), 3),
                    person_gender_before=_pre_gender,
                    person_gender_conf_before=round(float(_pre_gconf), 3),
                    person_gender_support_before=int(_pre_support),
                    attribute_guard_active=bool(_t_gconf >= _guard_thresh),
                )
            except Exception:
                log.warning("attachment_decision_log_failed", track_id=tracklet.track_id, exc_info=True)

            # Resolve person-level attributes with per-task hysteresis.
            is_provisional_occlusion = bool(matching.get("provisional"))
            if is_provisional_occlusion or attrs_unreliable:
                # Keep the tracklet/sighting evidence, but don't let a partial
                # occlusion crop alter person-level attributes. The sighting
                # still stores tracklet_attrs below for audit/debug.
                p_attrs = self.attribute_voter.person_snapshot(person_id)
            else:
                p_attrs = self.attribute_voter.resolve_person(person_id, t_attrs)
            p_attrs = self._mask_ambiguous_gender_conflict(p_attrs, t_attrs)
            p_gender, p_gender_conf = p_attrs.get("gender", ("unknown", 0.0))
            self.track_metadata[tracklet.track_id] = {
                "tracklet_id": tracklet_id,
                "tracklet_state": tracklet.state.value,
                "snapshot_key": None,
                "visibility_score": round(v_avg, 4),
                "quality": {
                    "v_avg": float(round(v_avg, 4)),
                    "embedding_consistency": float(round(emb_consistency, 4)),
                    "overall_consistency": float(round(consistency.overall, 4)),
                    "good_frame_ratio": float(round(consistency.good_frame_ratio, 4)),
                },
                "matching": matching,
                # Compact label-only summary for the optional Avro `attributes` map.
                "attributes": {task: label for task, (label, _) in p_attrs.items()},
            }

            # ── Persistence (fire-and-forget) ─────────────────────────
            try:
                if is_provisional_occlusion:
                    await self._persist_occlusion_candidate(
                        tracklet,
                        reason=str(
                            matching.get("provisional_reason")
                            or matching.get("source")
                            or "provisional_occlusion_match"
                        ),
                        selected_entries=consensus_entries,
                        embedding_consistency=emb_consistency,
                        matching=matching,
                    )
                    await self._persist_attached_occlusion_evidence(
                        tracklet=tracklet,
                        tracklet_id=tracklet_id,
                        person_id=person_id,
                        consistency=consistency,
                        v_avg=v_avg,
                        emb_consistency=emb_consistency,
                        selected=consensus_entries,
                        matching=matching,
                        tracklet_attrs=t_attrs,
                    )
                else:
                    await self._persist_tracklet(
                        tracklet=tracklet,
                        tracklet_id=tracklet_id,
                        person_id=person_id,
                        consistency=consistency,
                        v_avg=v_avg,
                        emb_consistency=emb_consistency,
                        best_entry=best_entry,
                        selected=consensus_entries,
                        matching=matching,
                        person_attrs=p_attrs,
                        tracklet_attrs=t_attrs,
                    )
                    await self._maybe_persist_scale_aux_embedding(
                        person_id=person_id,
                        tracklet=tracklet,
                        entry=best_entry,
                        v_avg=v_avg,
                        emb_consistency=emb_consistency,
                        overall_consistency=consistency.overall,
                        selected_max_overlap=selected_max_overlap,
                        matching=matching,
                    )
            except Exception:
                log.error("persistence_failed", tracklet_id=tracklet_id, exc_info=True)

            if not is_provisional_occlusion:
                person_id = await self._maybe_merge_duplicate_person(person_id)
            tracklet.person_id = person_id
            self.track_id_to_person_id[tracklet.track_id] = person_id
            self._remember_track_identity(person_id, tracklet)

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
            if (
                getattr(self, "_stream_finalizing", False)
                and allow_tentative_fallback
                and len(tracklet.entries) >= int(getattr(self.settings, "tracklet_min_entries", 4))
            ):
                self.tracklet_buffer.tracklets.setdefault(tracklet.track_id, tracklet)
            tent = getattr(self.matcher, "tentative", {}).get(tracklet.track_id, {})
            log.warning("tracklet_tentative_pending",
                        track_id=tracklet.track_id,
                        attempts=tent.get("attempts", 0),
                        v_avg=round(v_avg, 4),
                        consistency=round(emb_consistency, 4))
            self._track_inflight(asyncio.ensure_future(
                self._persist_occlusion_candidate(
                    tracklet,
                    reason=str(matching.get("source") or "tentative_unconfirmed"),
                    selected_entries=consensus_entries,
                    embedding_consistency=emb_consistency,
                    matching=matching,
                )
            ))

        return person_id

    async def _persist_tracklet(
        self, *, tracklet, tracklet_id, person_id, consistency,
        v_avg, emb_consistency, best_entry, selected, matching, person_attrs,
        tracklet_attrs=None,
    ) -> None:
        """Write to MongoDB, MinIO, and invalidate Redis — all async.

        ``person_attrs`` is the per-task ``{task: (label, confidence)}`` snapshot
        from the AttributeVoter, written to ``persons.attributes.*``.

        ``tracklet_attrs`` is the raw classifier output for THIS tracklet (before
        voter aggregation). Stored on ``sightings.attributes`` so the per-sighting
        gender-disagreement check in `_maybe_merge_duplicate_person` sees the
        per-tracklet classifier evidence, not the post-voted person gender (which
        is "unknown" for low-confidence tracklets and gives the merge guard nothing
        to work with). Falls back to ``person_attrs`` when not supplied.
        """
        device_id = self._current_device_id
        entries = tracklet.entries
        started_at = datetime.fromtimestamp(entries[0].timestamp_ns / 1e9, tz=timezone.utc)
        ended_at = datetime.fromtimestamp(entries[-1].timestamp_ns / 1e9, tz=timezone.utc)
        quality = {
            "v_avg": round(v_avg, 4),
            "embedding_consistency": round(emb_consistency, 4),
            "bbox_size_stability": round(consistency.bbox_size_stability, 4),
            "position_stability": round(consistency.position_stability, 4),
            "good_frame_ratio": round(consistency.good_frame_ratio, 4),
            "overall_consistency": round(consistency.overall, 4),
        }
        selected_frame_idxs = {entry.frame_idx for entry in selected}
        evidence = {
            "selected_frame_count": len(selected_frame_idxs),
            "selected_frame_indices": sorted(selected_frame_idxs),
            "frame_samples": [
                {
                    "frame_idx": int(entry.frame_idx),
                    "visibility_score": float(round(entry.v_score, 4)),
                    "overlap_ratio": float(round(entry.overlap_ratio, 4)),
                    "bbox_xyxy": [float(round(v, 2)) for v in entry.bbox_xyxy],
                    "selected": entry.frame_idx in selected_frame_idxs,
                    "selection_reason": (
                        "selected_consensus_frame"
                        if entry.frame_idx in selected_frame_idxs
                        else "not_selected"
                    ),
                    "crop_key": None,
                }
                for entry in entries
            ],
        }
        max_snapshot_overlap = float(
            getattr(self.settings, "person_snapshot_max_overlap_ratio", 0.35)
        )
        person_snapshot_entry = _choose_person_snapshot_entry(
            entries,
            selected,
            max_overlap_ratio=max_snapshot_overlap,
        )
        snapshot_overlap_ratio = (
            float(getattr(person_snapshot_entry, "overlap_ratio", 0.0) or 0.0)
            if person_snapshot_entry is not None
            else (
                float(getattr(best_entry, "overlap_ratio", 1.0) or 1.0)
                if best_entry is not None
                else 1.0
            )
        )
        snapshot_score = _compute_person_snapshot_score(
            v_avg=v_avg,
            overall_consistency=consistency.overall,
            embedding_consistency=emb_consistency,
            overlap_ratio=snapshot_overlap_ratio,
        )
        is_provisional_occlusion = bool((matching or {}).get("provisional"))
        # A confirmed person should always get an avatar. Prefer a clean (low-overlap)
        # entry, but fall back to the best available entry when every frame overlapped
        # (an occluded-throughout person) so the identity isn't left with a NULL
        # snapshot. update_person_snapshot is score-gated (high overlap -> low score),
        # so this only FILLS a missing avatar; it never overrides a better one.
        allow_person_snapshot = (
            not is_provisional_occlusion
            and (person_snapshot_entry is not None or best_entry is not None)
        )
        # Record torso color evidence for the within-camera color guard, but only
        # for CONFIRMED (non-provisional) assignments — provisional occlusion
        # evidence must not define a person's reference color.
        if not is_provisional_occlusion:
            try:
                self._update_person_color_evidence(person_id, device_id, selected or entries)
            except Exception:
                log.debug("person_color_evidence_update_failed", exc_info=True)

        # Log every sighting write so attribute evidence can be audited against
        # later merge or attachment decisions.
        _stored_attrs = tracklet_attrs if tracklet_attrs is not None else person_attrs
        _sighting_gender, _sighting_gconf = (_stored_attrs or {}).get("gender", ("unknown", 0.0))
        log.warning(
            "sighting_persisted",
            person_id=person_id,
            tracklet_id=tracklet_id,
            track_id=tracklet.track_id,
            gender=_sighting_gender,
            gender_conf=round(float(_sighting_gconf), 3),
            quality_score=round(consistency.overall, 4),
            entry_count=len(entries),
            frame_start=entries[0].frame_idx,
            frame_end=entries[-1].frame_idx,
        )

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
                frame_indices=[int(e.frame_idx) for e in entries],
                entry_count=len(entries),
                quality=quality,
                matching=matching,
                evidence=evidence,
                first_bbox_xyxy=[float(v) for v in entries[0].bbox_xyxy],
                last_bbox_xyxy=[float(v) for v in entries[-1].bbox_xyxy],
                best_crop_key=None,
            ),
            self.mongo.upsert_person(
                person_id,
                attributes=person_attrs,
                device_id=device_id,
                snapshot_key=None,
                snapshot_score=snapshot_score,
            ),
            self.mongo.add_sighting(
                person_id=person_id,
                device_id=device_id,
                tracklet_id=tracklet_id,
                started_at=started_at,
                ended_at=ended_at,
                entry_count=len(entries),
                quality_score=round(consistency.overall, 4),
                snapshot_key=None,
                attributes=tracklet_attrs if tracklet_attrs is not None else person_attrs,
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
        await self.mongo.recompute_person_attributes(person_id)
        await self.redis_cache.invalidate(person_id)

        # Asset upload is secondary to person persistence. Write Mongo first so a
        # newly-assigned identity appears in DB/UI immediately even when MinIO is
        # slow or backlogged.
        crop_key = ""
        snapshot_crop = None
        asset_entry = person_snapshot_entry or best_entry
        if asset_entry is not None:
            snapshot_crop = (
                asset_entry.attribute_crop
                if asset_entry.attribute_crop is not None and asset_entry.attribute_crop.size > 0
                else asset_entry.crop
            )
        if snapshot_crop is not None and snapshot_crop.size > 0:
            ok, buf = cv2.imencode(".jpg", snapshot_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                crop_key = await asyncio.to_thread(
                    self.minio.upload_tracklet_snapshot, tracklet_id, buf.tobytes(),
                )
                if tracklet.track_id in self.track_metadata:
                    self.track_metadata[tracklet.track_id]["snapshot_key"] = crop_key

        context_preview_entries = sorted(
            (entry for entry in entries if int(entry.frame_idx) not in selected_frame_idxs),
            key=lambda entry: (-float(entry.v_score), float(entry.overlap_ratio), int(entry.frame_idx)),
        )[:4]
        frame_upload_idxs = selected_frame_idxs | {
            int(entry.frame_idx) for entry in context_preview_entries
        }

        frame_crop_keys: dict[int, str] = {}
        for entry in entries:
            if int(entry.frame_idx) not in frame_upload_idxs:
                continue
            if entry.crop.size <= 0:
                continue
            ok, buf = cv2.imencode(".jpg", entry.crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                continue
            frame_key = await asyncio.to_thread(
                self.minio.upload_tracklet_frame_snapshot,
                tracklet_id,
                int(entry.frame_idx),
                buf.tobytes(),
            )
            if frame_key:
                frame_crop_keys[int(entry.frame_idx)] = frame_key

        if crop_key or frame_crop_keys:
            evidence_with_assets = {
                **evidence,
                "frame_samples": [
                    {
                        **sample,
                        "crop_key": frame_crop_keys.get(int(sample["frame_idx"])),
                    }
                    for sample in evidence["frame_samples"]
                ],
            }
            await asyncio.gather(
                self.mongo.update_tracklet_assets(
                    tracklet_id,
                    best_crop_key=crop_key or None,
                    evidence=evidence_with_assets,
                ),
                (
                    self.mongo.update_person_snapshot(
                        person_id,
                        snapshot_key=crop_key,
                        snapshot_score=snapshot_score,
                    )
                    if crop_key and allow_person_snapshot
                    else asyncio.sleep(0)
                ),
                self.mongo.update_sighting_snapshot(
                    tracklet_id,
                    snapshot_key=crop_key,
                ) if crop_key else asyncio.sleep(0),
                return_exceptions=True,
            )


def run() -> None:
    retry_delay_s = 3.0
    while True:
        try:
            WorkerPipeline().run()
            return
        except (NoBrokersAvailable, KafkaError, OSError):
            log.warning(
                "worker_start_retry",
                retry_delay_s=retry_delay_s,
                exc_info=True,
            )
            time.sleep(retry_delay_s)
