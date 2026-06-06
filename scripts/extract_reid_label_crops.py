#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract person crops from demo videos for ReID labeling."
    )
    parser.add_argument(
        "--output-dir",
        default="reid_label_crops/unlabeled_target_eval",
        help="Directory where crops and manifest.csv will be written.",
    )
    parser.add_argument("--model", default="infer/best_26.pt", help="YOLO detector weights.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--stride", type=int, default=15, help="Sample every N frames.")
    parser.add_argument("--min-height", type=int, default=70)
    parser.add_argument("--max-crops-per-video", type=int, default=450)
    parser.add_argument(
        "videos",
        nargs="*",
        default=[
            "infer/vid3.mp4",
            "infer/demo/device_1.mp4",
            "infer/demo/device_2.mp4",
            "infer/demo/device_3.mp4",
        ],
    )
    return parser.parse_args()


def padded_box(
    xyxy: list[float],
    frame_w: int,
    frame_h: int,
    *,
    top: float = 0.22,
    side: float = 0.08,
    bottom: float = 0.04,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    px = width * side
    py_top = height * top
    py_bottom = height * bottom
    return (
        max(0, int(round(x1 - px))),
        max(0, int(round(y1 - py_top))),
        min(frame_w, int(round(x2 + px))),
        min(frame_h, int(round(y2 + py_bottom))),
    )


def extract_video(
    *,
    model: YOLO,
    video_path: Path,
    output_dir: Path,
    writer: csv.DictWriter,
    imgsz: int,
    conf: float,
    stride: int,
    min_height: int,
    max_crops: int,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_name = video_path.stem
    source_dir = output_dir / source_name
    source_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    saved = 0

    while saved < max_crops:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        frame_h, frame_w = frame.shape[:2]
        result = model(frame, verbose=False, conf=conf, imgsz=imgsz, classes=[0])[0]
        boxes = sorted(
            result.boxes,
            key=lambda box: float(box.xyxy[0][0]),
        )
        det_idx = 0
        for box in boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            if (y2 - y1) < min_height:
                continue
            px1, py1, px2, py2 = padded_box([x1, y1, x2, y2], frame_w, frame_h)
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                continue
            filename = f"{source_name}_f{frame_idx:06d}_d{det_idx:02d}_c{float(box.conf):.2f}.jpg"
            rel_path = f"{source_name}/{filename}"
            cv2.imwrite(str(source_dir / filename), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            writer.writerow(
                {
                    "file": rel_path,
                    "source_video": str(video_path),
                    "frame_idx": frame_idx,
                    "det_idx": det_idx,
                    "confidence": round(float(box.conf), 4),
                    "bbox_x1": round(x1, 2),
                    "bbox_y1": round(y1, 2),
                    "bbox_x2": round(x2, 2),
                    "bbox_y2": round(y2, 2),
                    "crop_x1": px1,
                    "crop_y1": py1,
                    "crop_x2": px2,
                    "crop_y2": py2,
                }
            )
            saved += 1
            det_idx += 1
            if saved >= max_crops:
                break
        frame_idx += 1

    cap.release()
    return saved


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model, task="detect")

    manifest_path = output_dir / "manifest.csv"
    fields = [
        "file",
        "source_video",
        "frame_idx",
        "det_idx",
        "confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "crop_x1",
        "crop_y1",
        "crop_x2",
        "crop_y2",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        total = 0
        for video in args.videos:
            saved = extract_video(
                model=model,
                video_path=Path(video),
                output_dir=output_dir,
                writer=writer,
                imgsz=args.imgsz,
                conf=args.conf,
                stride=args.stride,
                min_height=args.min_height,
                max_crops=args.max_crops_per_video,
            )
            total += saved
            print(f"{video}: saved {saved} crops")
    print(f"total: saved {total} crops")
    print(f"output: {output_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
