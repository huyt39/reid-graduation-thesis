from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from src.api import deps
from src.api.routes import persons as persons_routes
from src.api.routes import search as search_routes


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
