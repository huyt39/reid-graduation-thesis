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

    @staticmethod
    def _resolve_local_path(path_str: str) -> str:
        path = Path(path_str)
        if path.is_absolute():
            return str(path)

        cwd_candidate = Path.cwd() / path
        if cwd_candidate.exists():
            return str(cwd_candidate)

        repo_root = Path(__file__).resolve().parents[4]
        return str(repo_root / path)

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

    def run(self) -> None:
        cap = self._open_capture(self.settings.source_url)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.settings.source_url}")

        frame_idx = 0
        processed_frames = 0
        fps_start = time.time()
        log.info(
            "edge_started",
            service=self.settings.service_name,
            source_url=self.settings.source_url,
            device_id=self.settings.device_id,
            topic=self.settings.reid_topic,
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    log.info("edge_stream_ended", frame_idx=frame_idx)
                    break

                frame_idx += 1

                if not self.pre_skipper.should_process(frame):
                    continue

                detections = self.detector.infer(frame)
                self.pre_skipper.update_after_detection(detections)
                if not detections:
                    continue

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
                    if not self.post_skipper.should_send(
                        tag,
                        visibility_score,
                        spatial_key,
                        frame_idx=frame_idx,
                    ):
                        continue

                    outbound_detections.append(
                        {
                            "bbox": bbox,
                            "confidence": confidence,
                            "class_id": det["class_id"],
                            "visibility_score": round(visibility_score, 4),
                            "visibility_tag": tag.value,
                            "overlap_ratio": round(overlap_ratio, 4),
                            "subscores": subscores,
                        }
                    )

                if not outbound_detections:
                    continue

                ok, img_encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality],
                )
                if not ok:
                    log.warning("frame_encode_failed", frame_idx=frame_idx)
                    continue

                self.producer.send(
                    device_id=self.settings.device_id,
                    frame_number=frame_idx,
                    detections=outbound_detections,
                    image_data=img_encoded.tobytes(),
                    timestamp_ns=timestamp_ns,
                )
                processed_frames += 1

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
                    )
        finally:
            cap.release()
            self.producer.close()
            log.info("edge_stopped", processed_frames=processed_frames, frame_idx=frame_idx)


def run() -> None:
    pipeline = EdgePipeline()
    pipeline.run()
