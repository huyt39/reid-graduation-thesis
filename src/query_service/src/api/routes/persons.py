from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query

from src.api.deps import get_mongo, get_redis, get_minio_urls

from src.schemas.query import (
    PersonResponse,
    PaginatedPersonsResponse,
    PaginatedSightingsResponse,
    PaginatedTrackletsResponse,
    PaginatedTimelineResponse,
    SimilarPersonsResponse,
    )

router = APIRouter(prefix="/persons", tags=["persons"])

def _attach_snapshot_url(item: dict, minio_urls) -> dict:
    enriched = dict(item)
    enriched["snapshot_url"] = minio_urls.presigned_url(item.get("snapshot_key"))
    return enriched


def _attach_best_crop_url(item: dict, minio_urls) -> dict:
    enriched = dict(item)
    enriched["best_crop_url"] = minio_urls.presigned_url(item.get("best_crop_key"))
    evidence = dict(enriched.get("evidence") or {})
    frame_samples = []
    for sample in evidence.get("frame_samples") or []:
        enriched_sample = dict(sample)
        enriched_sample["crop_url"] = minio_urls.presigned_url(enriched_sample.get("crop_key"))
        enriched_sample.pop("crop_key", None)
        frame_samples.append(enriched_sample)
    evidence["frame_samples"] = frame_samples
    enriched["evidence"] = evidence
    return enriched


@router.get("", response_model = PaginatedPersonsResponse)
async def list_persons(
    gender: str | None = None,
    device: str | None = None,
    is_active: bool | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    mongo = get_mongo()
    filters: dict = {}
    if gender:
        filters["attributes.gender"] = gender
    if device:
        filters["stats.last_seen_device"] = device
    if is_active is not None:
        filters["is_active"] = is_active

    minio_urls = get_minio_urls()
    items, total = await mongo.search_persons(
        filters=filters, skip=(page - 1) * page_size, limit=page_size,
    )
    enriched_items = [_attach_snapshot_url(item, minio_urls) for item in items]
    return {"items": enriched_items, "total": total, "page": page, "page_size": page_size}


@router.get("/{person_id}", response_model = PersonResponse)
async def get_person(person_id: int):
    from fastapi import HTTPException

    mongo = get_mongo()
    redis_cache = get_redis()

    cached = await redis_cache.get_person(person_id)
    if cached:
        return cached

    minio_urls = get_minio_urls()

    person = await mongo.get_person(person_id)
    if person is None:
        raise HTTPException(404, f"Person {person_id} not found")

    enriched_person = _attach_snapshot_url(person, minio_urls)
    await redis_cache.set_person(person_id, enriched_person)
    return enriched_person


@router.get("/{person_id}/sightings", response_model = PaginatedSightingsResponse)
async def get_sightings(
    person_id: int,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    mongo = get_mongo()
    minio_urls = get_minio_urls()
    items, total = await mongo.get_sightings(
        person_id, start_time=start_time, end_time=end_time,
        skip=(page - 1) * page_size, limit=page_size,
    )
    enriched_items = [_attach_snapshot_url(item, minio_urls) for item in items]
    return {"items": enriched_items, "total": total, "page": page, "page_size": page_size}


@router.get("/{person_id}/tracklets", response_model = PaginatedTrackletsResponse)
async def get_tracklets(
    person_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    mongo = get_mongo()
    minio_urls = get_minio_urls()
    items, total = await mongo.get_tracklets(
        person_id,
        skip=(page - 1) * page_size,
        limit=page_size,
    )
    enriched_items = [_attach_best_crop_url(item, minio_urls) for item in items]
    return {"items": enriched_items, "total": total, "page": page, "page_size": page_size}


@router.get("/{person_id}/timeline", response_model = PaginatedTimelineResponse)
async def get_timeline(
    person_id: int,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    event_types: list[str] | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
):
    mongo = get_mongo()
    items, total = await mongo.get_timeline(
        person_id, start_time=start_time, end_time=end_time, event_types=event_types,
        skip=(page - 1) * page_size, limit=page_size,
    )
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/{person_id}/similar", response_model = SimilarPersonsResponse)
async def get_similar(
    person_id: int,
    top_k: int = Query(10, ge=1, le=50),
    min_score: float = Query(0.5, ge=0.0, le=1.0),
):
    from src.api.deps import get_qdrant
    from fastapi import HTTPException
    qdrant = get_qdrant()
    mongo = get_mongo()
    minio_urls = get_minio_urls()

    source_person = await mongo.get_person(person_id)
    if source_person is None:
        raise HTTPException(404, f"Person {person_id} not found")

    results = qdrant.search_similar(person_id, top_k=top_k, min_score=min_score)

    enriched = []
    for r in results:
        person = await mongo.get_person(r["person_id"])
        item = dict(r)
        if person is not None:
            item["person"] = _attach_snapshot_url(person, minio_urls)
        enriched.append(item)

    return {"similar_persons": enriched}
