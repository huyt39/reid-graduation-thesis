import os
import time
import io

import cv2
import numpy as np
from dotenv import load_dotenv

from src.config import EdgeSettings
from src.detection.yolo import YoloModel
from src.scoring.visibility import compute_visibility_score, compute_subscores, compute_overlap_ratio
from src.scoring.tagging import VisibilityTag, tag_detection
from src.filtering.pre_skip import PreFrameSkipper
from src.filtering.post_skip import PostFrameSkipper
from src.kafka.producer import EdgeKafkaProducer

load_dotenv()


class EdgePipeline:
    def __init__(self, settings: EdgeSettings | None = None):
        self.settings = settings or EdgeSettings()
        self.yolo = YoloModel(
            model_path=self.settings.model_path,
            conf_threshold=self.settings.yolo_conf_threshold,
        )
        self.pre_skipper = PreFrameSkipper(skip_rate=self.settings.pre_skip_rate)
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

    def run(self):
        cap = cv2.VideoCapture(self.settings.source_url)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.settings.source_url}")

        frame_idx = 0
        fps_start = time.time()
        processed = 0

        print(f"[Edge] Starting pipeline on {self.settings.source_url}")
        print(f"[Edge] Device ID: {self.settings.device_id}")
        print(f"[Edge] Pre-skip rate: {self.settings.pre_skip_rate}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Pre-frame-skipping
                if not self.pre_skipper.should_process(frame_idx):
                    continue

                # YOLO detection
                detections = self.yolo.infer(frame)
                if not detections:
                    continue

                frame_h, frame_w = frame.shape[:2]
                timestamp_ns = time.time_ns()

                # Collect all bboxes for person-person overlap scoring
                all_bboxes = [det["bbox"] for det in detections]

                # Score and filter each detection
                messages = []
                for det in detections:
                    bbox = det["bbox"]
                    confidence = det["confidence"]

                    # Compute visibility subscores (including person overlap)
                    subscores = compute_subscores(
                        bbox, confidence, frame_w, frame_h, all_bboxes=all_bboxes
                    )
                    v_score = compute_visibility_score(subscores)

                    # Compute raw overlap ratio for downstream worker use
                    overlap_ratio = compute_overlap_ratio(bbox, all_bboxes)

                    # Tag detection
                    tag = tag_detection(
                        v_score,
                        good_thresh=self.settings.v_good_threshold,
                        mid_thresh=self.settings.v_mid_threshold,
                    )

                    # Post-frame-skipping by tag
                    spatial_key = f"{int(bbox[0]//50)}_{int(bbox[1]//50)}"
                    if not self.post_skipper.should_send(tag, v_score, spatial_key):
                        continue

                    messages.append({
                        "bbox": bbox,
                        "confidence": confidence,
                        "class_id": det["class_id"],
                        "visibility_score": round(v_score, 4),
                        "visibility_tag": tag.value,
                        "overlap_ratio": round(overlap_ratio, 4),
                        "subscores": subscores,
                    })

                if messages:
                    # Encode frame as JPEG
                    _, img_encoded = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                    )
                    image_bytes = img_encoded.tobytes()

                    self.producer.send(
                        device_id=self.settings.device_id,
                        frame_number=frame_idx,
                        detections=messages,
                        image_data=image_bytes,
                        timestamp_ns=timestamp_ns,
                    )

                processed += 1
                if processed % 100 == 0:
                    elapsed = time.time() - fps_start
                    print(f"[Edge] Processed {processed} frames, FPS: {processed/elapsed:.1f}")

        finally:
            cap.release()
            self.producer.close()
            print(f"[Edge] Pipeline finished. Total frames processed: {processed}")


if __name__ == "__main__":
    pipeline = EdgePipeline()
    pipeline.run()
