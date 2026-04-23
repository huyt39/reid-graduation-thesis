from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from src.api import deps
from src.api.routes import devices as devices_routes
from src.api.routes import persons as persons_routes
from src.api.routes import search as search_routes
from src.api.routes import stats as stats_routes


@pytest.mark.asyncio
async def test_get_person_uses_cache(monkeypatch):
    mongo = AsyncMock()
    redis_cache = AsyncMock()
    redis_cache.get_person.return_value = {"person_id": 7}

    monkeypatch.setattr(persons_routes, "get_mongo", lambda: mongo)
    monkeypatch.setattr(persons_routes, "get_redis", lambda: redis_cache)

    result = await persons_routes.get_person(7)

    assert result["person_id"] == 7
    mongo.get_person.assert_not_awaited()
    redis_cache.set_person.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_similar_enriches_persons(monkeypatch):
    mongo = AsyncMock()
    mongo.get_person = AsyncMock(side_effect=[
        {"person_id": 1},
        {"person_id": 2, "attributes": {"gender": "male"}},
    ])
    qdrant = MagicMock()
    qdrant.search_similar.return_value = [{"person_id": 2, "score": 0.91}]

    monkeypatch.setattr(persons_routes, "get_mongo", lambda: mongo)
    monkeypatch.setattr(deps, "get_qdrant", lambda: qdrant)

    result = await persons_routes.get_similar(person_id=1, top_k=5, min_score=0.5)

    assert result["similar_persons"][0]["person_id"] == 2
    assert result["similar_persons"][0]["person"]["person_id"] == 2


@pytest.mark.asyncio
async def test_get_similar_raises_404_for_missing_source_person(monkeypatch):
    mongo = AsyncMock()
    mongo.get_person = AsyncMock(return_value=None)
    qdrant = MagicMock()

    monkeypatch.setattr(persons_routes, "get_mongo", lambda: mongo)
    monkeypatch.setattr(deps, "get_qdrant", lambda: qdrant)

    with pytest.raises(HTTPException) as exc:
        await persons_routes.get_similar(person_id=999, top_k=5, min_score=0.5)

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_natural_language_query_returns_error_result_without_validation(monkeypatch):
    parser = AsyncMock()
    parser.parse.return_value = {"query_type": "error", "message": "Could not parse: ???"}
    executor = AsyncMock()

    monkeypatch.setattr(search_routes, "get_nl_parser", lambda: parser)
    monkeypatch.setattr(search_routes, "get_executor", lambda: executor)

    result = await search_routes.natural_language_query(search_routes.NLQueryRequest(query="???"))

    assert result == {
        "parsed_query": {"query_type": "error", "message": "Could not parse: ???"},
        "result": {"query_type": "error", "message": "Could not parse: ???"},
    }
    executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_natural_language_query_validates_and_executes_structured_query(monkeypatch):
    parser = AsyncMock()
    parser.parse.return_value = {"query_type": "person_lookup", "params": {"person_id": 7}}
    executor = AsyncMock()
    executor.execute.return_value = {"person": {"person_id": 7}}

    monkeypatch.setattr(search_routes, "get_nl_parser", lambda: parser)
    monkeypatch.setattr(search_routes, "get_executor", lambda: executor)

    result = await search_routes.natural_language_query(search_routes.NLQueryRequest(query="person 7"))

    assert result == {
        "parsed_query": {"query_type": "person_lookup", "params": {"person_id": 7}},
        "result": {"person": {"person_id": 7}},
    }
    executor.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_natural_language_query_returns_validation_error_result(monkeypatch):
    parser = AsyncMock()
    parser.parse.return_value = {
        "query_type": "person_lookup",
        "params": ["not", "a", "dict"],
    }
    executor = AsyncMock()

    monkeypatch.setattr(search_routes, "get_nl_parser", lambda: parser)
    monkeypatch.setattr(search_routes, "get_executor", lambda: executor)

    result = await search_routes.natural_language_query(
        search_routes.NLQueryRequest(query="bad parsed query")
    )

    assert result["parsed_query"] == {
        "query_type": "person_lookup",
        "params": ["not", "a", "dict"],
    }
    assert result["result"]["query_type"] == "error"
    assert result["result"]["message"] == "Parsed query failed validation"
    assert "details" in result["result"]
    executor.execute.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_get_device_route_raises_404_for_missing_device(monkeypatch):
    mongo = AsyncMock()
    mongo.get_device.return_value = None

    monkeypatch.setattr(devices_routes, "get_mongo", lambda: mongo)

    with pytest.raises(HTTPException) as exc:
        await devices_routes.get_device("cam_404")

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_list_devices_route_returns_devices_wrapper(monkeypatch):
    mongo = AsyncMock()
    mongo.list_devices.return_value = [
        {
            "device_id": "cam_01",
            "sighting_count": 4,
            "unique_person_count": 2,
        }
    ]

    monkeypatch.setattr(devices_routes, "get_mongo", lambda: mongo)

    result = await devices_routes.list_devices()

    assert result == {
        "devices": [
            {
                "device_id": "cam_01",
                "sighting_count": 4,
                "unique_person_count": 2,
            }
        ]
    }
    mongo.list_devices.assert_awaited_once()


def test_aggregation_response_defaults_to_empty_list():
    from src.schemas.query import AggregationResponse, DevicesListResponse

    response = AggregationResponse()
    devices = DevicesListResponse()

    assert response.aggregation == []
    assert devices.devices == []
