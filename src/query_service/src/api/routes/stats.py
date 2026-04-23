from __future__ import annotations
from typing import Literal
from datetime import datetime

from fastapi import APIRouter
from src.schemas.query import StatsResponse, AggregationResponse
from src.api.deps import get_mongo

router = APIRouter(tags=["stats"])


@router.get("/stats", response_model = StatsResponse)
async def get_stats():
    mongo = get_mongo()
    return await mongo.get_stats()


@router.get("/stats/aggregate", response_model = AggregationResponse)
async def aggregate_stats(
    person_id: int | None = None,
    device_id: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    group_by: Literal["hour", "day", "device"] = "hour",
):
    mongo = get_mongo()
    aggregation = await mongo.aggregate_sightings(
        person_id=person_id,
        device_id=device_id,
        start_time=start_time,
        end_time=end_time,
        group_by=group_by,
    )
    return {"aggregation": aggregation}
