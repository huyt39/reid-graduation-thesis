from __future__ import annotations

from fastapi import APIRouter

from src.api.deps import get_mongo

router = APIRouter(tags=["stats"])


@router.get("/stats")
async def get_stats():
    mongo = get_mongo()
    return await mongo.get_stats()
