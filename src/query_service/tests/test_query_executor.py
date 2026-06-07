"""Tests for QueryExecutor with mocked DB clients."""
from __future__ import annotations

from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
from pydantic import TypeAdapter, ValidationError

from src.services.query_executor import QueryExecutor
from src.schemas.query import (
    PersonLookupQuery,
    SightingAggregationParams,
    StructuredSearchQuery,
)


@pytest.fixture
def mongo():
    m = AsyncMock()
    m.get_person = AsyncMock(return_value={"person_id": 1, "attributes": {"gender": "male"}})
    m.search_persons = AsyncMock(return_value=([{"person_id": 1}], 1))
    m.get_timeline = AsyncMock(return_value=([{"event_type": "sighting_start"}], 1))
    m.list_devices = AsyncMock(return_value=[{"device_id": "cam-1"}])
    m.get_stats = AsyncMock(return_value={"total_persons": 5})
    return m


@pytest.fixture
def qdrant():
    q = MagicMock()
    q.search_similar = MagicMock(return_value=[{"person_id": 2, "score": 0.8}])
    return q


@pytest.fixture
def redis_cache():
    r = AsyncMock()
    r.get_person = AsyncMock(return_value=None)
    r.set_person = AsyncMock()
    return r


@pytest.fixture
def executor(mongo, qdrant, redis_cache):
    return QueryExecutor(mongo, qdrant, redis_cache)


@pytest.mark.asyncio
async def test_person_lookup(executor, mongo):
    result = await executor.execute({"query_type": "person_lookup", "params": {"person_id": 1}})
    assert result["person"]["person_id"] == 1
    mongo.get_person.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_person_lookup_attaches_snapshot_url():
    mongo = AsyncMock()
    qdrant = MagicMock()
    redis_cache = AsyncMock()
    minio_urls = MagicMock()

    mongo.get_person = AsyncMock(return_value={"person_id": 1, "snapshot_key": "persons/1.jpg"})
    redis_cache.get_person = AsyncMock(return_value=None)
    redis_cache.set_person = AsyncMock()
    minio_urls.presigned_url.return_value = "https://snapshots.example/persons/1.jpg"

    executor = QueryExecutor(mongo, qdrant, redis_cache, minio_urls)

    result = await executor.execute({"query_type": "person_lookup", "params": {"person_id": 1}})

    assert result["person"]["snapshot_url"] == "https://snapshots.example/persons/1.jpg"
    minio_urls.presigned_url.assert_called_once_with("persons/1.jpg")
    redis_cache.set_person.assert_awaited_once_with(1, result["person"])


@pytest.mark.asyncio
async def test_execute_accepts_structured_query_request(executor):
    query = PersonLookupQuery(query_type="person_lookup", params={"person_id": 1})

    result = await executor.execute(query)

    assert result["person"]["person_id"] == 1


@pytest.mark.asyncio
async def test_person_lookup_uses_cache(executor, redis_cache, mongo):
    redis_cache.get_person.return_value = {"person_id": 1, "attributes": {"gender": "male"}}
    result = await executor.execute({"query_type": "person_lookup", "params": {"person_id": 1}})
    assert result["person"]["person_id"] == 1
    mongo.get_person.assert_not_awaited()
    redis_cache.set_person.assert_not_awaited()


@pytest.mark.asyncio
async def test_person_search(executor, mongo):
    result = await executor.execute({
        "query_type": "person_search",
        "params": {"filters": {"gender": "male"}},
    })
    assert result["total"] == 1


@pytest.mark.asyncio
async def test_person_search_attaches_snapshot_urls():
    mongo = AsyncMock()
    qdrant = MagicMock()
    redis_cache = AsyncMock()
    minio_urls = MagicMock()

    mongo.search_persons = AsyncMock(
        return_value=(
            [
                {"person_id": 1, "snapshot_key": "persons/1.jpg"},
                {"person_id": 2, "snapshot_key": None},
            ],
            2,
        )
    )
    minio_urls.presigned_url.side_effect = (
        lambda key: f"https://snapshots.example/{key}" if key else None
    )

    executor = QueryExecutor(mongo, qdrant, redis_cache, minio_urls)

    result = await executor.execute({
        "query_type": "person_search",
        "params": {"filters": {}},
    })

    assert result["items"][0]["snapshot_url"] == "https://snapshots.example/persons/1.jpg"
    assert result["items"][1]["snapshot_url"] is None


@pytest.mark.asyncio
async def test_person_search_builds_filters_and_pagination():
    mongo = AsyncMock()
    qdrant = MagicMock()
    redis_cache = AsyncMock()

    mongo.search_persons = AsyncMock(return_value=([{"person_id": 1}], 1))

    executor = QueryExecutor(mongo, qdrant, redis_cache)

    result = await executor.execute({
        "query_type": "person_search",
        "params": {
            "filters": {
                "gender": "male",
                "gender_confidence_min": 0.8,
                "glasses": "glasses",
                "backpack": "backpack",
                "hat": "hat",
                "last_seen_device": "cam_01",
                "is_active": True,
            },
            "page": 2,
            "page_size": 10,
        },
    })

    assert result["items"] == [{"person_id": 1}]
    assert result["total"] == 1
    assert result["page"] == 2
    assert result["page_size"] == 10

    mongo.search_persons.assert_awaited_once_with(
        filters={
            "attributes.gender": "male",
            "attributes.gender_confidence": {"$gte": 0.8},
            "attributes.glasses": "glasses",
            "attributes.backpack": "backpack",
            "attributes.hat": "hat",
            "stats.last_seen_device": "cam_01",
            "is_active": True,
        },
        skip=10,
        limit=10,
    )


@pytest.mark.asyncio
async def test_person_search_supports_person_id_filter():
    mongo = AsyncMock()
    qdrant = MagicMock()
    redis_cache = AsyncMock()

    mongo.search_persons = AsyncMock(return_value=([{"person_id": 7}], 1))

    executor = QueryExecutor(mongo, qdrant, redis_cache)

    result = await executor.execute({
        "query_type": "person_search",
        "params": {
            "filters": {
                "person_id": 7,
            },
        },
    })

    assert result["items"] == [{"person_id": 7}]
    assert result["total"] == 1

    mongo.search_persons.assert_awaited_once_with(
        filters={"person_id": 7},
        skip=0,
        limit=20,
    )


@pytest.mark.asyncio
async def test_person_search_supports_first_seen_range_filters():
    mongo = AsyncMock()
    qdrant = MagicMock()
    redis_cache = AsyncMock()

    mongo.search_persons = AsyncMock(return_value=([{"person_id": 9}], 1))

    executor = QueryExecutor(mongo, qdrant, redis_cache)

    result = await executor.execute({
        "query_type": "person_search",
        "params": {
            "filters": {
                "first_seen_after": "2026-04-01T00:00:00Z",
                "first_seen_before": "2026-04-30T23:59:59Z",
            },
        },
    })

    assert result["items"] == [{"person_id": 9}]
    assert result["total"] == 1

    mongo.search_persons.assert_awaited_once_with(
        filters={
            "stats.first_seen_at": {
                "$gte": ANY,
                "$lte": ANY,
            }
        },
        skip=0,
        limit=20,
    )


@pytest.mark.asyncio
async def test_person_search_supports_min_sighting_count():
    mongo = AsyncMock()
    qdrant = MagicMock()
    redis_cache = AsyncMock()

    mongo.search_persons = AsyncMock(return_value=([{"person_id": 11}], 1))

    executor = QueryExecutor(mongo, qdrant, redis_cache)

    result = await executor.execute({
        "query_type": "person_search",
        "params": {
            "filters": {
                "min_sighting_count": 3,
            },
        },
    })

    assert result["items"] == [{"person_id": 11}]
    assert result["total"] == 1

    mongo.search_persons.assert_awaited_once_with(
        filters={"stats.sighting_count": {"$gte": 3}},
        skip=0,
        limit=20,
    )


@pytest.mark.asyncio
async def test_similarity_search(executor, qdrant):
    result = await executor.execute({
        "query_type": "similarity_search",
        "params": {"person_id": 1, "top_k": 5},
    })
    assert len(result["similar_persons"]) == 1
    qdrant.search_similar.assert_called_once()


@pytest.mark.asyncio
async def test_sighting_aggregation_uses_typed_params():
    mongo = AsyncMock()
    qdrant = MagicMock()
    redis_cache = AsyncMock()

    mongo.aggregate_sightings = AsyncMock(return_value=[{"bucket": "2026-04-20T10:00:00Z", "count": 3}])

    executor = QueryExecutor(mongo, qdrant, redis_cache)

    result = await executor.execute({
        "query_type": "sighting_aggregation",
        "params": {
            "person_id": 7,
            "device_id": "cam_01",
            "start_time": "2026-04-20T10:00:00Z",
            "end_time": "2026-04-20T11:00:00Z",
            "group_by": "hour",
        },
    })

    assert result["aggregation"] == [{"bucket": "2026-04-20T10:00:00Z", "count": 3}]

    mongo.aggregate_sightings.assert_awaited_once_with(
        person_id=7,
        device_id="cam_01",
        start_time=ANY,
        end_time=ANY,
        group_by="hour",
    )


@pytest.mark.asyncio
async def test_device_lookup_all(executor, mongo):
    result = await executor.execute({"query_type": "device_lookup", "params": {}})
    assert len(result["devices"]) == 1


@pytest.mark.asyncio
async def test_device_lookup_one(executor, mongo):
    mongo.get_device = AsyncMock(return_value={"device_id": "cam-1"})
    result = await executor.execute({
        "query_type": "device_lookup",
        "params": {"device_id": "cam-1"},
    })
    assert result["device"]["device_id"] == "cam-1"


@pytest.mark.asyncio
async def test_unknown_query_type(executor):
    result = await executor.execute({"query_type": "invalid_type", "params": {}})
    assert "error" in result


def test_structured_query_request_rejects_unknown_query_type():
    adapter = TypeAdapter(StructuredSearchQuery)
    with pytest.raises(ValidationError):
        adapter.validate_python({"query_type": "invalid_type", "params": {}})


def test_structured_query_request_rejects_wrong_param_shape():
    adapter = TypeAdapter(StructuredSearchQuery)
    with pytest.raises(ValidationError):
        adapter.validate_python({
            "query_type": "person_lookup",
            "params": {"device_id": "cam-1"},
        })


def test_sighting_aggregation_params_reject_invalid_group_by():
    with pytest.raises(ValidationError):
        SightingAggregationParams(group_by="month")
