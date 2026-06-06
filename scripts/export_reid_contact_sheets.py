#!/usr/bin/env python3
"""Export visual contact sheets for current ReID persons/tracklets."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from pymongo import MongoClient


BUCKET = "reid-snapshots"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create person-level ReID contact sheets from MinIO crops.")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27018")
    parser.add_argument("--mongo-db", default="reid_production")
    parser.add_argument("--minio-alias", default="local")
    parser.add_argument("--out-dir", default="/private/tmp/reid_contact_sheets")
    parser.add_argument("--max-tracklets-per-person", type=int, default=20)
    parser.add_argument("--thumb-width", type=int, default=160)
    parser.add_argument("--thumb-height", type=int, default=260)
    return parser.parse_args()


def _frame_range(doc: dict[str, Any]) -> str:
    frame_range = doc.get("frame_range") or {}
    return f"{int(frame_range.get('start', -1))}-{int(frame_range.get('end', -1))}"


def _copy_object(alias: str, key: str, dest: Path) -> bool:
    if not key:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = f"{alias}/{BUCKET}/{key}"
    result = subprocess.run(["mc", "cp", "--quiet", source, str(dest)], check=False)
    return result.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def _load_thumb(path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((width, height - 48))
    canvas = Image.new("RGB", (width, height), "white")
    x = (width - image.width) // 2
    canvas.paste(image, (x, 0))
    return canvas


def _draw_label(tile: Image.Image, label: str) -> None:
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, tile.height - 46, tile.width, tile.height), fill=(245, 245, 245))
    draw.text((6, tile.height - 42), label, fill=(20, 20, 20))


def _make_sheet(person_id: int, tracklets: list[dict[str, Any]], image_paths: list[Path], args: argparse.Namespace) -> Image.Image:
    cols = min(5, max(1, len(image_paths)))
    rows = (len(image_paths) + cols - 1) // cols
    header_h = 42
    sheet = Image.new("RGB", (cols * args.thumb_width, header_h + rows * args.thumb_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), f"Person {person_id} | tracklets={len(tracklets)}", fill=(0, 0, 0))
    for idx, (tracklet, path) in enumerate(zip(tracklets, image_paths)):
        tile = _load_thumb(path, args.thumb_width, args.thumb_height)
        matching = tracklet.get("matching") or {}
        sim = matching.get("similarity_score")
        sim_text = "new" if sim is None else f"sim={float(sim):.3f}"
        label = f"tr={tracklet.get('track_id')} fr={_frame_range(tracklet)}\n{matching.get('source') or '?'} {sim_text}"
        _draw_label(tile, label)
        x = (idx % cols) * args.thumb_width
        y = header_h + (idx // cols) * args.thumb_height
        sheet.paste(tile, (x, y))
    return sheet


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    crops_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=2000)
    db = client[args.mongo_db]
    person_ids = sorted(
        int(row["person_id"])
        for row in db.persons.find({}, {"_id": 0, "person_id": 1})
    )
    generated = []
    for person_id in person_ids:
        tracklets = list(
            db.tracklets.find(
                {"person_id": person_id, "best_crop_key": {"$nin": [None, ""]}},
                {"_id": 0, "tracklet_id": 1, "track_id": 1, "frame_range": 1, "best_crop_key": 1, "matching": 1},
            )
            .sort("frame_range.start", 1)
            .limit(args.max_tracklets_per_person)
        )
        image_paths = []
        kept_tracklets = []
        for tracklet in tracklets:
            dest = crops_dir / f"person_{person_id}" / f"track_{tracklet.get('track_id')}.jpg"
            if _copy_object(args.minio_alias, str(tracklet.get("best_crop_key") or ""), dest):
                image_paths.append(dest)
                kept_tracklets.append(tracklet)
        if not image_paths:
            continue
        sheet = _make_sheet(person_id, kept_tracklets, image_paths, args)
        sheet_path = out_dir / f"person_{person_id}_sheet.jpg"
        sheet.save(sheet_path, quality=92)
        generated.append(sheet_path)
        print(sheet_path)

    index_path = out_dir / "index.md"
    with index_path.open("w", encoding="utf-8") as handle:
        handle.write("# ReID Contact Sheets\n\n")
        for path in generated:
            handle.write(f"![{path.name}]({path})\n\n")
    print(f"index={index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
