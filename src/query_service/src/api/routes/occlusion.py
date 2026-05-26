from __future__ import annotations

from fastapi import APIRouter, Query

from src.api.deps import get_minio_urls, get_mongo
from src.api.routes.persons import _attach_best_crop_url
from src.schemas.query import PaginatedOcclusionCandidatesResponse

router = APIRouter(prefix="/occlusion-candidates", tags=["occlusion"])


@router.get("", response_model=PaginatedOcclusionCandidatesResponse)
async def list_occlusion_candidates(
    status: str | None = "unconfirmed",
    device: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    mongo = get_mongo()
    minio_urls = get_minio_urls()
    items, total = await mongo.get_occlusion_candidates(
        status=status,
        device_id=device,
        skip=(page - 1) * page_size,
        limit=page_size,
    )
    enriched_items = [_attach_best_crop_url(item, minio_urls) for item in items]
    return {"items": enriched_items, "total": total, "page": page, "page_size": page_size}
