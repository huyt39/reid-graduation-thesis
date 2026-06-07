#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
import requests


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate current ReID embeddings on a labeled crop folder.")
    parser.add_argument(
        "--label-dir",
        default="reid_label_crops/unlabeled_target_eval/label",
        help="Folder containing one subfolder per real person.",
    )
    parser.add_argument("--endpoint", default="http://localhost:8001/embedding/batch")
    parser.add_argument("--single-endpoint", default="http://localhost:8001/embedding")
    parser.add_argument("--backend", choices=["endpoint", "onnx"], default="endpoint")
    parser.add_argument("--onnx-model", default="triton_models/osnet/1/model.onnx")
    parser.add_argument("--model", default="osnet")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        default="reid_label_crops/unlabeled_target_eval/eval_current_model",
    )
    return parser.parse_args()


def load_image_paths(label_dir: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for person_dir in sorted([p for p in label_dir.iterdir() if p.is_dir()]):
        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                rows.append((person_dir.name, image_path))
    if not rows:
        raise RuntimeError(f"No labeled images found under {label_dir}")
    return rows


def embed_single(endpoint: str, model: str, item: tuple[str, Path]) -> list[float]:
    _, path = item
    with path.open("rb") as handle:
        response = requests.post(
            endpoint,
            files={"image": (path.name, handle, "image/jpeg")},
            data={"model": model},
            timeout=180,
        )
    response.raise_for_status()
    return response.json()["embedding"]


def embed_batch(
    endpoint: str,
    single_endpoint: str,
    model: str,
    items: list[tuple[str, Path]],
) -> list[list[float]]:
    files = []
    handles = []
    try:
        for _, path in items:
            handle = path.open("rb")
            handles.append(handle)
            files.append(("images", (path.name, handle, "image/jpeg")))
        response = requests.post(endpoint, files=files, data={"model": model}, timeout=180)
        if response.status_code == 503:
            return [embed_single(single_endpoint, model, item) for item in items]
        response.raise_for_status()
        payload = response.json()
        return payload["embeddings"]
    finally:
        for handle in handles:
            handle.close()


def preprocess_for_onnx(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((128, 256), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    chw = arr.transpose(2, 0, 1)
    return (chw - IMAGENET_MEAN) / IMAGENET_STD


def embed_onnx(session, items: list[tuple[str, Path]]) -> list[list[float]]:
    batch = np.stack([preprocess_for_onnx(path) for _, path in items], axis=0).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: batch})[0].astype(np.float32)
    output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-8)
    return output.tolist()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float32), p))


def main() -> None:
    args = parse_args()
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_paths = load_image_paths(label_dir)
    onnx_session = None
    if args.backend == "onnx":
        import onnxruntime as ort
        onnx_session = ort.InferenceSession(args.onnx_model, providers=["CPUExecutionProvider"])

    labels: list[str] = []
    paths: list[str] = []
    embeddings: list[list[float]] = []
    for start in range(0, len(labeled_paths), args.batch_size):
        batch = labeled_paths[start:start + args.batch_size]
        if args.backend == "onnx":
            embeddings.extend(embed_onnx(onnx_session, batch))
        else:
            embeddings.extend(embed_batch(args.endpoint, args.single_endpoint, args.model, batch))
        labels.extend(label for label, _ in batch)
        paths.extend(str(path) for _, path in batch)
        print(f"embedded {min(start + args.batch_size, len(labeled_paths))}/{len(labeled_paths)}")

    emb = np.asarray(embeddings, dtype=np.float32)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    sims = emb @ emb.T

    same_scores: list[float] = []
    diff_scores: list[float] = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sims[i, j])
            if labels[i] == labels[j]:
                same_scores.append(score)
            else:
                diff_scores.append(score)

    thresholds = [round(x, 2) for x in np.arange(0.45, 0.91, 0.01)]
    threshold_rows = []
    best_f1 = (-1.0, None)
    for threshold in thresholds:
        tp = sum(score >= threshold for score in same_scores)
        fn = len(same_scores) - tp
        fp = sum(score >= threshold for score in diff_scores)
        tn = len(diff_scores) - fp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
        row = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fp / max(fp + tn, 1),
            "false_negative_rate": fn / max(tp + fn, 1),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        threshold_rows.append(row)
        if f1 > best_f1[0]:
            best_f1 = (f1, threshold)

    per_person = {}
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)
    for label, idxs in sorted(by_label.items()):
        intra = [
            float(sims[i, j])
            for pos, i in enumerate(idxs)
            for j in idxs[pos + 1:]
        ]
        nearest_negative = []
        for i in idxs:
            neg_scores = [float(sims[i, j]) for j in range(n) if labels[j] != label]
            nearest_negative.append(max(neg_scores) if neg_scores else float("nan"))
        per_person[label] = {
            "count": len(idxs),
            "same_mean": mean(intra) if intra else float("nan"),
            "same_p05": percentile(intra, 5),
            "same_p50": percentile(intra, 50),
            "nearest_negative_mean": mean(nearest_negative),
            "nearest_negative_p95": percentile(nearest_negative, 95),
        }

    summary = {
        "label_dir": str(label_dir),
        "image_count": n,
        "person_count": len(by_label),
        "same_pair_count": len(same_scores),
        "different_pair_count": len(diff_scores),
        "same": {
            "mean": mean(same_scores),
            "p05": percentile(same_scores, 5),
            "p10": percentile(same_scores, 10),
            "p50": percentile(same_scores, 50),
            "p90": percentile(same_scores, 90),
        },
        "different": {
            "mean": mean(diff_scores),
            "p50": percentile(diff_scores, 50),
            "p90": percentile(diff_scores, 90),
            "p95": percentile(diff_scores, 95),
            "p99": percentile(diff_scores, 99),
            "max": max(diff_scores),
        },
        "best_f1_threshold": best_f1[1],
        "best_f1": best_f1[0],
        "per_person": per_person,
    }

    np.save(output_dir / "embeddings.npy", emb)
    with (output_dir / "items.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "path"])
        writer.writeheader()
        writer.writerows({"label": label, "path": path} for label, path in zip(labels, paths))
    with (output_dir / "thresholds.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(threshold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(threshold_rows)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
