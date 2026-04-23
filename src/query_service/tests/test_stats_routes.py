from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.api.routes import stats as stats_routes


@pytest.mark.asyncio
async def test_aggregate_stats_route_calls_mongo_with_query_params(monkeypatch):
    mongo = AsyncMock()
    mongo.aggregate_sightings.return_value = [{"_id": "cam_01", "count": 3}]

    monkeypatch.setattr(stats_routes, "get_mongo", lambda: mongo)

    result = await stats_routes.aggregate_stats(
        person_id=7,
        device_id="cam_01",
        start_time=None,
        end_time=None,
        group_by="device",
    )

    assert result == {"aggregation": [{"_id": "cam_01", "count": 3}]}
    mongo.aggregate_sightings.assert_awaited_once_with(
        person_id=7,
        device_id="cam_01",
        start_time=None,
        end_time=None,
        group_by="device",
    )


@pytest.mark.asyncio
async def test_get_stats_route_returns_mongo_stats(monkeypatch):
    mongo = AsyncMock()
    mongo.get_stats.return_value = {
        "total_persons": 5,
        "active_persons": 3,
        "total_sightings": 12,
        "total_devices": 2,
    }

    monkeypatch.setattr(stats_routes, "get_mongo", lambda: mongo)

    result = await stats_routes.get_stats()

    assert result == {
        "total_persons": 5,
        "active_persons": 3,
        "total_sightings": 12,
        "total_devices": 2,
    }
    mongo.get_stats.assert_awaited_once()


def test_aggregation_response_defaults_to_empty_list():
    from src.schemas.query import AggregationResponse

    response = AggregationResponse()

    assert response.aggregation == []
