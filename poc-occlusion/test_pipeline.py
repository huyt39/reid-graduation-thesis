"""Standalone test script to visualize the edge scoring pipeline.

Runs YOLO + visibility scoring on a video and displays annotated frames
with visibility scores, tags, and overlap ratios.

Usage:
    cd poc-occlusion
    PYTHONPATH=edge python3 test_pipeline.py
"""

import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "edge")

from src.detection.yolo import YoloModel
from src.scoring.visibility import compute_subscores, compute_visibility_score, compute_overlap_ratio
from src.scoring.tagging import tag_detection
from src.filtering.pre_skip import PreFrameSkipper

# Config
VIDEO_PATH = "data/18156284-hd_1080_1920_25fps.mp4"
MODEL_PATH = "yolo11n.pt"
PRE_SKIP_RATE = 2
CONF_THRESHOLD = 0.25

# Colors by tag
TAG_COLORS = {
    "good": (0, 255, 0),   # green
    "mid": (0, 255, 255),   # yellow
    "bad": (0, 0, 255),     # red
}


def main():
    print(f"Loading YOLO from {MODEL_PATH}...")
    yolo = YoloModel(model_path=MODEL_PATH, conf_threshold=CONF_THRESHOLD, imgsz=1280)
    pre_skipper = PreFrameSkipper(skip_rate=PRE_SKIP_RATE)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {VIDEO_PATH}")
        return

    frame_idx = 0
    fps_start = time.time()
    processed = 0

    print("Press 'q' to quit, SPACE to pause/resume")
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Video ended, restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            frame_idx += 1

            if not pre_skipper.should_process(frame_idx):
                continue

            # Resize for display if too large
            display = frame.copy()
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                display = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                scale = 1.0

            # Detect
            detections = yolo.infer(frame)
            if frame_idx <= 10 or frame_idx % 50 == 0:
                print(f"Frame {frame_idx}: shape={frame.shape}, detections={len(detections)}")
                for d in detections[:3]:
                    print(f"  conf={d['confidence']:.3f} bbox={[int(x) for x in d['bbox']]}")
            all_bboxes = [d["bbox"] for d in detections]

            # Score and draw
            for det in detections:
                bbox = det["bbox"]
                conf = det["confidence"]

                subscores = compute_subscores(bbox, conf, w, h, all_bboxes=all_bboxes)
                v_score = compute_visibility_score(subscores)
                overlap = compute_overlap_ratio(bbox, all_bboxes)
                tag = tag_detection(v_score)

                # Scale bbox for display
                x1, y1, x2, y2 = [int(c * scale) for c in bbox]
                color = TAG_COLORS[tag.value]

                # Draw bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = f"v={v_score:.2f} [{tag.value}] ovlp={overlap:.2f}"
                sub_label = (
                    f"cut={subscores['cut_off']:.1f} area={subscores['area_ratio']:.1f} "
                    f"asp={subscores['aspect_ratio']:.1f} conf={subscores['det_conf']:.1f} "
                    f"povlp={subscores['person_overlap']:.1f}"
                )

                # Background for text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display, (x1, y1 - th - 25), (x1 + tw + 4, y1), color, -1)
                cv2.putText(display, label, (x1 + 2, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                (tw2, th2), _ = cv2.getTextSize(sub_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.rectangle(display, (x1, y1 - 13), (x1 + tw2 + 4, y1), color, -1)
                cv2.putText(display, sub_label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # Frame info
            processed += 1
            elapsed = time.time() - fps_start
            fps = processed / elapsed if elapsed > 0 else 0
            info = f"Frame {frame_idx} | Detections: {len(detections)} | FPS: {fps:.1f}"
            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("PoC Occlusion Pipeline - Edge Scoring", display)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {processed} frames")


if __name__ == "__main__":
    main()
