#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FRAME_RE = re.compile(r"_f(\d+)_")
PERSON_RE = re.compile(r"(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reproducible Vast.ai OSNet tuning package from labeled ReID crops."
    )
    parser.add_argument(
        "--label-dir",
        default="reid_label_crops/unlabeled_target_eval/label",
        help="Folder containing one subfolder per identity.",
    )
    parser.add_argument(
        "--eval-dir",
        default="reid_label_crops/unlabeled_target_eval/eval_local_endpoint_20260518_151433",
        help="Folder containing items.csv and embeddings.npy from eval_reid_labeled_set.py.",
    )
    parser.add_argument("--output-dir", default="reid_tuning_package")
    parser.add_argument("--dataset-name", default="occlusion_reid_market1501")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--query-ratio", type=float, default=0.15)
    parser.add_argument("--top-k-hard", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def frame_number(path: Path) -> int:
    match = FRAME_RE.search(path.name)
    return int(match.group(1)) if match else 0


def person_id(label: str, fallback: int) -> int:
    match = PERSON_RE.search(label)
    return int(match.group(1)) if match else fallback


def camera_id(index: int, split_name: str) -> int:
    # Torchreid expects Market1501-style cam IDs. These are pseudo-cameras derived
    # from temporal split buckets so query/gallery do not collapse into one camera.
    split_offset = {"bounding_box_train": 1, "query": 2, "bounding_box_test": 3}[split_name]
    return ((index + split_offset - 1) % 6) + 1


def load_labeled_images(label_dir: Path) -> dict[str, list[Path]]:
    by_label: dict[str, list[Path]] = {}
    for person_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
        images = sorted(
            [p for p in person_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES],
            key=lambda p: (frame_number(p), p.name),
        )
        if images:
            by_label[person_dir.name] = images
    if not by_label:
        raise RuntimeError(f"No labeled images found under {label_dir}")
    return by_label


def split_images(images: list[Path], train_ratio: float, query_ratio: float) -> dict[str, list[Path]]:
    n = len(images)
    if n < 3:
        return {
            "bounding_box_train": images[:],
            "query": images[:1],
            "bounding_box_test": images[-1:],
        }

    train_n = max(1, int(round(n * train_ratio)))
    query_n = max(1, int(round(n * query_ratio)))
    if train_n + query_n >= n:
        train_n = max(1, n - 2)
        query_n = 1

    return {
        "bounding_box_train": images[:train_n],
        "query": images[train_n:train_n + query_n],
        "bounding_box_test": images[train_n + query_n:],
    }


def build_dataset(args: argparse.Namespace, by_label: dict[str, list[Path]]) -> dict[str, int]:
    output_dir = Path(args.output_dir)
    dataset_dir = output_dir / args.dataset_name
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split_name in ("bounding_box_train", "query", "bounding_box_test"):
        (dataset_dir / split_name).mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, int] = defaultdict(int)
    manifest_rows: list[dict[str, str | int]] = []
    for label_index, (label, images) in enumerate(sorted(by_label.items()), start=1):
        pid = person_id(label, label_index)
        splits = split_images(images, args.train_ratio, args.query_ratio)
        for split_name, split_images_ in splits.items():
            for index, src in enumerate(split_images_):
                camid = camera_id(index, split_name)
                dst_name = f"{pid:04d}_c{camid}s1_{index:06d}_00.jpg"
                dst = dataset_dir / split_name / dst_name
                shutil.copy2(src, dst)
                split_counts[split_name] += 1
                manifest_rows.append(
                    {
                        "label": label,
                        "pid": pid,
                        "split": split_name,
                        "source_path": str(src),
                        "dataset_path": str(dst.relative_to(output_dir)),
                    }
                )

    with (output_dir / "dataset_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "pid", "split", "source_path", "dataset_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return dict(split_counts)


def load_eval(eval_dir: Path) -> tuple[list[str], list[str], np.ndarray]:
    with (eval_dir / "items.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = [row["label"] for row in rows]
    paths = [row["path"] for row in rows]
    embeddings = np.load(eval_dir / "embeddings.npy").astype(np.float32)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    return labels, paths, embeddings


def write_hard_pair_report(args: argparse.Namespace) -> dict[str, int]:
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    labels, paths, embeddings = load_eval(eval_dir)
    sims = embeddings @ embeddings.T

    hard_positive: list[dict[str, str | float]] = []
    hard_negative: list[dict[str, str | float]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            row = {
                "score": float(sims[i, j]),
                "label_a": labels[i],
                "path_a": paths[i],
                "label_b": labels[j],
                "path_b": paths[j],
            }
            if labels[i] == labels[j]:
                hard_positive.append(row)
            else:
                hard_negative.append(row)

    hard_positive.sort(key=lambda row: float(row["score"]))
    hard_negative.sort(key=lambda row: float(row["score"]), reverse=True)
    hard_positive = hard_positive[:args.top_k_hard]
    hard_negative = hard_negative[:args.top_k_hard]

    for name, rows in (("hard_positive_low_similarity.csv", hard_positive), ("hard_negative_high_similarity.csv", hard_negative)):
        with (output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["score", "label_a", "path_a", "label_b", "path_b"],
            )
            writer.writeheader()
            writer.writerows(rows)

    return {
        "hard_positive_count": len(hard_positive),
        "hard_negative_count": len(hard_negative),
    }


def write_vast_files(output_dir: Path) -> None:
    (output_dir / "requirements-vast.txt").write_text(
        "\n".join(
            [
                "numpy>=1.26.4,<2.7",
                "scipy>=1.13.0",
                "h5py>=3.11.0",
                "Cython>=3.0.0",
                "wheel>=0.43.0",
                "setuptools>=70.0.0",
                "yacs>=0.1.8",
                "gdown>=5.2.0",
                "future>=1.0.0",
                "six>=1.16.0",
                "imageio>=2.34.0",
                "Pillow>=10.0.0",
                "tqdm>=4.66.0",
                "tensorboard>=2.16.0",
                "onnx>=1.16.0",
                "onnxruntime>=1.18.0",
                "opencv-python-headless>=4.9.0",
                "",
            ]
        )
    )
    (output_dir / "run_vast_train.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

# V100 needs CUDA 12.1 wheels; newer cu13 wheels warn that sm_70 is unsupported.
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-vast.txt
pip install --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git

mkdir -p data
tar -xzf occlusion_reid_market1501.tar.gz -C data

python train_osnet_occlusion.py --data-root data --save-dir runs/osnet_occlusion --max-epoch 60 --batch-size 32 --lr 2e-5 --stepsize 30 --eval-freq 5
"""
    )


def create_archives(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    dataset_tar = output_dir / f"{args.dataset_name}.tar.gz"
    package_tar = output_dir / "vast_osnet_tuning_package.tar.gz"
    for tar_path in (dataset_tar, package_tar):
        if tar_path.exists():
            tar_path.unlink()

    with tarfile.open(dataset_tar, "w:gz") as tar:
        tar.add(output_dir / args.dataset_name, arcname=args.dataset_name)

    package_members = [
        dataset_tar,
        output_dir / "requirements-vast.txt",
        output_dir / "run_vast_train.sh",
        output_dir / "train_osnet_occlusion.py",
        output_dir / "export_osnet_onnx.py",
        output_dir / "dataset_manifest.csv",
        output_dir / "hard_positive_low_similarity.csv",
        output_dir / "hard_negative_high_similarity.csv",
        output_dir / "tuning_package_summary.json",
    ]
    with tarfile.open(package_tar, "w:gz") as tar:
        for member in package_members:
            if member.exists():
                tar.add(member, arcname=member.name)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_label = load_labeled_images(Path(args.label_dir))
    split_counts = build_dataset(args, by_label)
    hard_pair_counts = write_hard_pair_report(args)
    write_vast_files(output_dir)

    summary = {
        "label_dir": args.label_dir,
        "eval_dir": args.eval_dir,
        "person_count": len(by_label),
        "image_count": sum(len(images) for images in by_label.values()),
        "split_counts": split_counts,
        **hard_pair_counts,
        "upload_archive": str(output_dir / "vast_osnet_tuning_package.tar.gz"),
    }
    (output_dir / "tuning_package_summary.json").write_text(json.dumps(summary, indent=2))
    create_archives(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
