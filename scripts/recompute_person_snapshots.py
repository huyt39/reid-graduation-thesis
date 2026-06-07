from __future__ import annotations

import argparse

from pymongo import MongoClient


def compute_snapshot_score(tracklet: dict) -> float:
    quality = tracklet.get("quality") or {}
    v_avg = float(quality.get("v_avg", 0.0))
    overall = float(quality.get("overall_consistency", 0.0))
    emb = float(quality.get("embedding_consistency", 0.0))
    return round((0.5 * v_avg) + (0.3 * overall) + (0.2 * emb), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute best person snapshot keys from tracklet quality.")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--mongo-db", default="reid_production")
    args = parser.parse_args()

    client = MongoClient(args.mongo_uri)
    db = client[args.mongo_db]

    updated = 0
    for person in db.persons.find({}, {"_id": 0, "person_id": 1}):
        person_id = person["person_id"]
        best_tracklet = None
        best_score = -1.0
        for tracklet in db.tracklets.find(
            {"person_id": person_id, "best_crop_key": {"$exists": True, "$ne": ""}},
            {"_id": 0, "best_crop_key": 1, "quality": 1},
        ):
            score = compute_snapshot_score(tracklet)
            if score > best_score:
                best_score = score
                best_tracklet = tracklet

        if best_tracklet is None:
            continue

        db.persons.update_one(
            {"person_id": person_id},
            {
                "$set": {
                    "snapshot_key": best_tracklet["best_crop_key"],
                    "stats.best_snapshot_score": best_score,
                }
            },
        )
        updated += 1

    print(f"updated_persons={updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
