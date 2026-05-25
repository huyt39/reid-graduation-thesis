from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.api.routes import search as search_routes


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
        "summary": "Found person 7.",
    }
    executor.execute.assert_awaited_once()
    parser.summarize.assert_not_awaited()


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
