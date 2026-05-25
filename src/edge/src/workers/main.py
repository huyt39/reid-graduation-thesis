from __future__ import annotations

import time
from pathlib import Path

import cv2
import structlog

from src.core.config import settings
from src.detection.detection import DetectionModel
from src.filtering.post_skip import PostFrameSkipper
from src.filtering.pre_skip import PreFrameSkipper
from src.kafka.producer import EdgeKafkaProducer
from src.scoring.tagging import tag_detection
from src.scoring.visibility import (
    compute_overlap_ratio,
    compute_subscores,
    compute_visibility_score,
)

log = structlog.get_logger()


class EdgePipeline:
    def __init__(self) -> None:
        self.settings = settings
        self.detector = DetectionModel(
            model_path=self._resolve_local_path(self.settings.model_path),
            conf_threshold=self.settings.yolo_conf_threshold,
            imgsz=self.settings.yolo_imgsz,
        )
        self.pre_skipper = PreFrameSkipper(
            max_skip_with_boxes=self.settings.pre_skip_max_detected,
            max_skip_without_boxes=self.settings.pre_skip_max_empty,
            box_count_weight=self.settings.pre_skip_box_count_weight,
            criterion_scale=self.settings.pre_skip_criterion_scale,
            gray_size=self.settings.pre_skip_gray_size,
        )
        self.post_skipper = PostFrameSkipper(
            rates={
                "good": self.settings.post_skip_good,
                "mid": self.settings.post_skip_mid,
                "bad": self.settings.post_skip_bad,
            },
            drop_floor=self.settings.drop_floor,
        )
        self.producer = EdgeKafkaProducer(
            bootstrap_servers=self.settings.kafka_bootstrap_servers,
            topic=self.settings.reid_topic,
            schema_path=self.settings.schema_path,
        )
        self.detect_every_n_frames = max(1, self.settings.detect_every_n_frames)

    @staticmethod
    def _resolve_local_path(path_str: str) -> str:
        path = Path(path_str)
        if path.is_absolute():
            return str(path)

        cwd_candidate = Path.cwd() / path
        if cwd_candidate.exists():
            return str(cwd_candidate)

        parents = Path(__file__).resolve().parents
        if len(parents) > 4:
            repo_root = parents[4]
            repo_candidate = repo_root / path
            if repo_candidate.exists():
                return str(repo_candidate)

        return str(cwd_candidate)

    @staticmethod
    def _bbox_area_ratio(bbox: list[float], frame_w: int, frame_h: int) -> float:
        x1, y1, x2, y2 = bbox
        if frame_w <= 0 or frame_h <= 0 or x2 <= x1 or y2 <= y1:
            return 0.0
        return ((x2 - x1) * (y2 - y1)) / float(frame_w * frame_h)

    def _should_force_send(
        self,
        *,
        confidence: float,
        visibility_score: float,
        overlap_ratio: float,
        cutoff_score: float,
        bbox: list[float],
        frame_w: int,
        frame_h: int,
    ) -> bool:
        return (
            confidence >= self.settings.always_send_conf_threshold
            and visibility_score >= self.settings.always_send_visibility_threshold
            and overlap_ratio <= self.settings.always_send_max_overlap_ratio
            and cutoff_score >= self.settings.always_send_min_cutoff_score
            and self._bbox_area_ratio(bbox, frame_w, frame_h)
            >= self.settings.always_send_min_area_ratio
        )

    @staticmethod
    def _make_spatial_key(bbox: list[float], frame_w: int, frame_h: int) -> str:
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        x_bin = int(cx // 64)
        y_bin = int(cy // 64)
        w_bin = int((w / max(frame_w, 1)) * 20)
        h_bin = int((h / max(frame_h, 1)) * 20)
        return f"{x_bin}_{y_bin}_{w_bin}_{h_bin}"

    @staticmethod
    def _open_capture(source_url: str) -> cv2.VideoCapture:
        source: str | int = source_url
        if source_url.isdigit():
            source = int(source_url)
        return cv2.VideoCapture(source)

    @staticmethod
    def _synthetic_detection(frame_w: int, frame_h: int) -> dict:
        x1 = frame_w * 0.30
        y1 = frame_h * 0.15
        x2 = frame_w * 0.70
        y2 = frame_h * 0.92
        return {
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": 0.99,
            "class_id": 0,
        }

    def _prepare_outbound_frame(
        self,
        frame,
        detections: list[dict],
    ) -> tuple:
        max_encode_dim = self.settings.max_encode_dim
        if max_encode_dim <= 0:
            return frame, detections

        frame_h, frame_w = frame.shape[:2]
        longest_dim = max(frame_h, frame_w)
        if longest_dim <= max_encode_dim:
            return frame, detections

        scale = max_encode_dim / float(longest_dim)
        resized_w = max(1, int(round(frame_w * scale)))
        resized_h = max(1, int(round(frame_h * scale)))
        resized_frame = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

        resized_detections = []
        for det in detections:
            resized_detections.append(
                {
                    **det,
                    "bbox": [float(coord * scale) for coord in det["bbox"]],
                }
            )

        return resized_frame, resized_detections

    def run(self) -> None:
        cap = self._open_capture(self.settings.source_url)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.settings.source_url}")

        frame_idx = 0
        processed_frames = 0
        published_messages = 0
        fps_start = time.time()
        total_detect_ms = 0.0
        total_encode_ms = 0.0
        total_publish_ms = 0.0
        total_raw_detections = 0
        total_outbound_detections = 0
        log.info(
            "edge_started",
            service=self.settings.service_name,
            source_url=self.settings.source_url,
            device_id=self.settings.device_id,
            topic=self.settings.reid_topic,
            demo_mode=self.settings.demo_mode,
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    log.info(
                        "edge_stream_ended",
                        frame_idx=frame_idx,
                        processed_frames=processed_frames,
                        published_messages=published_messages,
                    )
                    self.producer.send_end_of_stream(
                        device_id=self.settings.device_id,
                        frame_number=frame_idx,
                        timestamp_ns=time.time_ns(),
                    )
                    break

                frame_idx += 1
                if (
                    self.settings.log_every_n_frames > 0
                    and frame_idx % self.settings.log_every_n_frames == 0
                ):
                    log.info("edge_frame_read", frame_idx=frame_idx)

                if (
                    not self.settings.demo_mode
                    and self.detect_every_n_frames > 1
                    and frame_idx % self.detect_every_n_frames != 0
                ):
                    continue

                if (
                    not self.settings.demo_mode
                    and self.detect_every_n_frames <= 1
                    and not self.pre_skipper.should_process(frame)
                ):
                    continue

                detect_started_at = time.perf_counter()
                detections = self.detector.infer(frame)
                detect_ms = (time.perf_counter() - detect_started_at) * 1000
                total_detect_ms += detect_ms
                if not self.settings.demo_mode:
                    if self.detect_every_n_frames <= 1:
                        self.pre_skipper.update_after_detection(detections)
                    if not detections:
                        continue
                elif not detections:
                    frame_h, frame_w = frame.shape[:2]
                    detections = [self._synthetic_detection(frame_w, frame_h)]
                    log.info(
                        "edge_demo_synthetic_detection",
                        frame_idx=frame_idx,
                        bbox=detections[0]["bbox"],
                    )

                log.info(
                    "edge_detections",
                    frame_idx=frame_idx,
                    detection_count=len(detections),
                )
                total_raw_detections += len(detections)

                frame_h, frame_w = frame.shape[:2]
                timestamp_ns = time.time_ns()
                all_bboxes = [det["bbox"] for det in detections]
                outbound_detections: list[dict] = []

                for det in detections:
                    bbox = det["bbox"]
                    confidence = det["confidence"]
                    subscores = compute_subscores(
                        bbox,
                        confidence,
                        frame_w,
                        frame_h,
                        all_bboxes=all_bboxes,
                    )
                    visibility_score = compute_visibility_score(subscores)
                    overlap_ratio = compute_overlap_ratio(bbox, all_bboxes)
                    tag = tag_detection(
                        visibility_score,
                        good_thresh=self.settings.v_good_threshold,
                        mid_thresh=self.settings.v_mid_threshold,
                    )

                    spatial_key = self._make_spatial_key(bbox, frame_w, frame_h)
                    if not self.settings.demo_mode:
                        force_send = self._should_force_send(
                            confidence=confidence,
                            visibility_score=visibility_score,
                            overlap_ratio=overlap_ratio,
                            cutoff_score=subscores["cut_off"],
                            bbox=bbox,
                            frame_w=frame_w,
                            frame_h=frame_h,
                        )
                        should_send = self.post_skipper.should_send(
                            tag,
                            visibility_score,
                            spatial_key,
                            frame_idx=frame_idx,
                        )
                        if not force_send and not should_send:
                            continue

                    outbound_detections.append(
                        {
                            "bbox": bbox,
                            "confidence": confidence,
                            "class_id": det["class_id"],
                            "visibility_score": round(visibility_score, 4),
                            "overlap_ratio": round(overlap_ratio, 4),
                            "visibility_tag": tag.value if hasattr(tag, "value") else str(tag),
                        }
                    )

                if not outbound_detections:
                    continue

                frame_to_encode, outbound_detections = self._prepare_outbound_frame(
                    frame,
                    outbound_detections,
                )
                encode_started_at = time.perf_counter()
                ok, img_encoded = cv2.imencode(
                    ".jpg",
                    frame_to_encode,
                    [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality],
                )
                encode_ms = (time.perf_counter() - encode_started_at) * 1000
                total_encode_ms += encode_ms
                if not ok:
                    log.warning("frame_encode_failed", frame_idx=frame_idx)
                    continue

                publish_started_at = time.perf_counter()
                self.producer.send(
                    device_id=self.settings.device_id,
                    frame_number=frame_idx,
                    detections=outbound_detections,
                    image_data=img_encoded.tobytes(),
                    timestamp_ns=timestamp_ns,
                )
                publish_ms = (time.perf_counter() - publish_started_at) * 1000
                total_publish_ms += publish_ms
                processed_frames += 1
                published_messages += 1
                total_outbound_detections += len(outbound_detections)
                log.info(
                    "edge_published",
                    frame_idx=frame_idx,
                    detection_count=len(outbound_detections),
                    published_messages=published_messages,
                )

                if (
                    self.settings.log_every_n_processed_frames > 0
                    and processed_frames % self.settings.log_every_n_processed_frames == 0
                ):
                    elapsed = max(time.time() - fps_start, 1e-6)
                    log.info(
                        "edge_progress",
                        processed_frames=processed_frames,
                        frame_idx=frame_idx,
                        fps=round(processed_frames / elapsed, 2),
                        source_fps=round(frame_idx / elapsed, 2),
                        avg_detect_ms=round(total_detect_ms / processed_frames, 2),
                        avg_encode_ms=round(total_encode_ms / processed_frames, 2),
                        avg_publish_ms=round(total_publish_ms / processed_frames, 2),
                        avg_raw_detections=round(total_raw_detections / processed_frames, 2),
                        avg_outbound_detections=round(total_outbound_detections / processed_frames, 2),
                    )
        finally:
            cap.release()
            self.producer.close()
            log.info(
                "edge_stopped",
                processed_frames=processed_frames,
                published_messages=published_messages,
                frame_idx=frame_idx,
            )


def run() -> None:
    pipeline = EdgePipeline()
    pipeline.run()
