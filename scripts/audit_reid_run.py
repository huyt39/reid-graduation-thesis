#!/usr/bin/env python3
"""Audit a demo ReID run without mutating identity state.

The goal is to make duplicate/missed-person debugging data-driven before
changing online matching logic again.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient


STABLE_ATTRIBUTE_TASKS = ("gender", "backpack", "hat", "lower", "sleeve")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit current Mongo/Qdrant ReID state.")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27018")
    parser.add_argument("--mongo-db", default="reid_production")
    parser.add_argument("--qdrant-url", default="http://localhost:16333")
    parser.add_argument("--collection", default="persons")
    parser.add_argument("--duplicate-threshold", type=float, default=0.72)
    parser.add_argument("--candidate-v-threshold", type=float, default=0.50)
    parser.add_argument("--json-out", default="", help="Optional path to write machine-readable audit JSON.")
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def _frame_range(doc: dict[str, Any]) -> tuple[int, int]:
    frame_range = doc.get("frame_range") or {}
    return int(frame_range.get("start", -1)), int(frame_range.get("end", -1))


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _persons_cooccur(tracklets_by_person: dict[int, list[dict[str, Any]]], person_a: int, person_b: int) -> bool:
    for tracklet_a in tracklets_by_person.get(person_a, []):
        for tracklet_b in tracklets_by_person.get(person_b, []):
            if str(tracklet_a.get("device_id", "")) != str(tracklet_b.get("device_id", "")):
                continue
            if _ranges_overlap(_frame_range(tracklet_a), _frame_range(tracklet_b)):
                return True
    return False


def _attribute_conflicts(attrs_a: dict[str, Any], attrs_b: dict[str, Any], min_conf: float = 0.85) -> list[str]:
    conflicts: list[str] = []
    for task in STABLE_ATTRIBUTE_TASKS:
        label_a = attrs_a.get(task)
        label_b = attrs_b.get(task)
        conf_a = float(attrs_a.get(f"{task}_confidence", 0.0) or 0.0)
        conf_b = float(attrs_b.get(f"{task}_confidence", 0.0) or 0.0)
        if (
            label_a
            and label_b
            and label_a != "unknown"
            and label_b != "unknown"
            and label_a != label_b
            and conf_a >= min_conf
            and conf_b >= min_conf
        ):
            conflicts.append(f"{task}:{label_a}!={label_b}")
    return conflicts


def _quality_value(doc: dict[str, Any], key: str) -> float:
    return float(((doc.get("quality") or {}).get(key)) or 0.0)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _post_json(url: str, payload: dict[str, Any], timeout_s: float = 2.0) -> dict[str, Any] | None:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None


def _scroll_qdrant_vectors(qdrant_url: str, collection: str) -> dict[int, list[list[float]]]:
    vectors_by_person: dict[int, list[list[float]]] = defaultdict(list)
    offset: Any = None
    while True:
        payload: dict[str, Any] = {
            "limit": 256,
            "with_payload": True,
            "with_vector": True,
        }
        if offset is not None:
            payload["offset"] = offset
        response = _post_json(f"{qdrant_url.rstrip('/')}/collections/{collection}/points/scroll", payload)
        if not response:
            break
        result = response.get("result") or {}
        points = result.get("points") or []
        for point in points:
            payload_doc = point.get("payload") or {}
            person_id = payload_doc.get("person_id")
            vector = point.get("vector")
            if person_id is None or not isinstance(vector, list):
                continue
            vectors_by_person[int(person_id)].append([float(v) for v in vector])
        offset = result.get("next_page_offset")
        if not offset:
            break
    return dict(vectors_by_person)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _best_gallery_similarity(vectors_a: list[list[float]], vectors_b: list[list[float]]) -> float:
    best = 0.0
    for vector_a in vectors_a:
        for vector_b in vectors_b:
            best = max(best, _cosine(vector_a, vector_b))
    return best


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _audit(args: argparse.Namespace) -> dict[str, Any]:
    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=2000)
    db = client[args.mongo_db]
    persons = list(db.persons.find({}, {"_id": 0}).sort("person_id", 1))
    tracklets = list(db.tracklets.find({}, {"_id": 0}).sort([("person_id", 1), ("frame_range.start", 1)]))
    candidates = list(db.occlusion_candidates.find({}, {"_id": 0}).sort("frame_range.start", 1))

    tracklets_by_person: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for tracklet in tracklets:
        person_id = tracklet.get("person_id")
        if person_id is not None:
            tracklets_by_person[int(person_id)].append(tracklet)

    person_docs = {int(person["person_id"]): person for person in persons}
    vectors_by_person = _scroll_qdrant_vectors(args.qdrant_url, args.collection)

    person_rows = []
    for person in persons:
        person_id = int(person["person_id"])
        person_tracklets = tracklets_by_person.get(person_id, [])
        ranges = [_frame_range(tracklet) for tracklet in person_tracklets]
        quality = {
            "v_avg": _mean([_quality_value(tracklet, "v_avg") for tracklet in person_tracklets]),
            "embedding_consistency": _mean([_quality_value(tracklet, "embedding_consistency") for tracklet in person_tracklets]),
            "overall_consistency": _mean([_quality_value(tracklet, "overall_consistency") for tracklet in person_tracklets]),
        }
        methods = defaultdict(int)
        for tracklet in person_tracklets:
            matching = tracklet.get("matching") or {}
            methods[str(matching.get("source") or matching.get("method") or "unknown")] += 1
        person_rows.append(
            {
                "person_id": person_id,
                "sightings": int(((person.get("stats") or {}).get("sighting_count")) or 0),
                "tracklets": len(person_tracklets),
                "gallery_vectors": len(vectors_by_person.get(person_id, [])),
                "frame_span": [min([r[0] for r in ranges], default=-1), max([r[1] for r in ranges], default=-1)],
                "quality": {key: round(value, 4) for key, value in quality.items()},
                "match_sources": dict(sorted(methods.items())),
                "attributes": person.get("attributes") or {},
                "merged_person_ids": person.get("merged_person_ids") or [],
            }
        )

    mongo_person_ids = set(person_docs)
    qdrant_person_ids = set(vectors_by_person)
    consistency_warnings = []
    orphan_qdrant_person_ids = sorted(qdrant_person_ids - mongo_person_ids)
    missing_qdrant_person_ids = sorted(mongo_person_ids - qdrant_person_ids)
    if orphan_qdrant_person_ids:
        consistency_warnings.append(
            {
                "type": "orphan_qdrant_vectors",
                "message": "Qdrant contains gallery vectors for person ids not present in Mongo.",
                "person_ids": orphan_qdrant_person_ids,
            }
        )
    if missing_qdrant_person_ids:
        consistency_warnings.append(
            {
                "type": "missing_qdrant_vectors",
                "message": "Mongo contains persons without any Qdrant gallery vector.",
                "person_ids": missing_qdrant_person_ids,
            }
        )

    duplicate_candidates = []
    person_ids = sorted(person_docs)
    for index, person_a in enumerate(person_ids):
        for person_b in person_ids[index + 1:]:
            vectors_a = vectors_by_person.get(person_a, [])
            vectors_b = vectors_by_person.get(person_b, [])
            similarity = _best_gallery_similarity(vectors_a, vectors_b) if vectors_a and vectors_b else 0.0
            conflicts = _attribute_conflicts(
                person_docs[person_a].get("attributes") or {},
                person_docs[person_b].get("attributes") or {},
            )
            cooccur = _persons_cooccur(tracklets_by_person, person_a, person_b)
            if similarity >= args.duplicate_threshold or (similarity >= 0.60 and not conflicts and not cooccur):
                duplicate_candidates.append(
                    {
                        "person_a": person_a,
                        "person_b": person_b,
                        "gallery_similarity": round(similarity, 4),
                        "cooccur_range_overlap": cooccur,
                        "attribute_conflicts": conflicts,
                        "tracklets_a": len(tracklets_by_person.get(person_a, [])),
                        "tracklets_b": len(tracklets_by_person.get(person_b, [])),
                    }
                )
    duplicate_candidates.sort(key=lambda row: row["gallery_similarity"], reverse=True)

    unconfirmed_candidates = []
    for candidate in candidates:
        quality = candidate.get("quality") or {}
        v_avg = float(quality.get("v_avg", 0.0) or 0.0)
        if str(candidate.get("status", "")) == "unconfirmed" and v_avg >= args.candidate_v_threshold:
            unconfirmed_candidates.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "track_id": candidate.get("track_id"),
                    "reason": candidate.get("reason"),
                    "frame_range": list(_frame_range(candidate)),
                    "entry_count": int(candidate.get("entry_count", 0) or 0),
                    "v_avg": round(v_avg, 4),
                    "embedding_consistency": round(float(quality.get("embedding_consistency", 0.0) or 0.0), 4),
                    "overall_consistency": round(float(quality.get("overall_consistency", 0.0) or 0.0), 4),
                    "matching": candidate.get("matching") or {},
                }
            )

    reason_counts: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        reason_counts[str(candidate.get("reason") or "unknown")] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mongo": {"uri": args.mongo_uri, "db": args.mongo_db},
        "qdrant": {
            "url": args.qdrant_url,
            "collection": args.collection,
            "persons_with_vectors": len(vectors_by_person),
            "total_vectors": sum(len(vectors) for vectors in vectors_by_person.values()),
        },
        "counts": {
            "persons": len(persons),
            "tracklets": len(tracklets),
            "occlusion_candidates": len(candidates),
            "unconfirmed_candidates_above_threshold": len(unconfirmed_candidates),
        },
        "consistency_warnings": consistency_warnings,
        "persons": person_rows,
        "duplicate_candidates": duplicate_candidates[: args.top_k],
        "occlusion_candidate_reason_counts": dict(sorted(reason_counts.items())),
        "unconfirmed_occlusion_candidates": unconfirmed_candidates[: args.top_k],
    }


def _print_report(report: dict[str, Any]) -> None:
    counts = report["counts"]
    qdrant = report["qdrant"]
    print("# ReID Run Audit")
    print(f"generated_at: {report['generated_at']}")
    print(
        "counts: "
        f"persons={counts['persons']} tracklets={counts['tracklets']} "
        f"occlusion_candidates={counts['occlusion_candidates']} "
        f"unconfirmed_high_quality={counts['unconfirmed_candidates_above_threshold']}"
    )
    print(f"qdrant: persons_with_vectors={qdrant['persons_with_vectors']} total_vectors={qdrant['total_vectors']}")

    print("\n## Consistency Warnings")
    if not report["consistency_warnings"]:
        print("- none")
    for warning in report["consistency_warnings"]:
        ids = ",".join(str(person_id) for person_id in warning["person_ids"])
        print(f"- {warning['type']}: {warning['message']} ids=[{ids}]")

    print("\n## Persons")
    for row in report["persons"]:
        quality = row["quality"]
        span = row["frame_span"]
        sources = ", ".join(f"{key}:{value}" for key, value in row["match_sources"].items()) or "none"
        print(
            f"- id={row['person_id']} tracklets={row['tracklets']} sightings={row['sightings']} "
            f"vectors={row['gallery_vectors']} frames={span[0]}-{span[1]} "
            f"v={_format_float(quality['v_avg'])} emb_cons={_format_float(quality['embedding_consistency'])} "
            f"overall={_format_float(quality['overall_consistency'])} sources=[{sources}]"
        )

    print("\n## Duplicate Candidates")
    if not report["duplicate_candidates"]:
        print("- none above audit thresholds")
    for row in report["duplicate_candidates"]:
        conflicts = ",".join(row["attribute_conflicts"]) or "none"
        print(
            f"- {row['person_a']} <-> {row['person_b']} "
            f"sim={_format_float(row['gallery_similarity'])} "
            f"cooccur={row['cooccur_range_overlap']} conflicts={conflicts} "
            f"tracklets=({row['tracklets_a']},{row['tracklets_b']})"
        )

    print("\n## Occlusion Candidate Reasons")
    for reason, count in report["occlusion_candidate_reason_counts"].items():
        print(f"- {reason}: {count}")

    print("\n## High-Quality Unconfirmed Occlusion Candidates")
    if not report["unconfirmed_occlusion_candidates"]:
        print("- none above threshold")
    for row in report["unconfirmed_occlusion_candidates"]:
        frame_range = row["frame_range"]
        print(
            f"- candidate={row['candidate_id']} track={row['track_id']} reason={row['reason']} "
            f"frames={frame_range[0]}-{frame_range[1]} entries={row['entry_count']} "
            f"v={_format_float(row['v_avg'])} emb_cons={_format_float(row['embedding_consistency'])} "
            f"overall={_format_float(row['overall_consistency'])}"
        )


def main() -> int:
    args = _parse_args()
    try:
        report = _audit(args)
    except Exception as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 1
    _print_report(report)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
