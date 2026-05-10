"""Unit tests for QueryParser using a stub LLM that returns fixed strings."""
from __future__ import annotations

import asyncio

import pytest

from src.services.query_parser import VALID_QUERY_TYPES, QueryParser


class StubLLM:
    """Minimal LLMClient stand-in: returns a preset string from chat()."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0

    async def chat(self, *args, **kwargs):  # noqa: D401, ANN001
        self.call_count += 1
        return self.response


def _run(coro):
    return asyncio.run(coro)


def test_person_lookup_passthrough():
    llm = StubLLM('{"query_type": "person_lookup", "params": {"person_id": 42}}')
    parser = QueryParser(llm)  # type: ignore[arg-type]
    out = _run(parser.parse("show me person 42"))
    assert out == {"query_type": "person_lookup", "params": {"person_id": 42}}
    assert llm.call_count == 1


def test_person_search_with_filters():
    llm = StubLLM(
        '{"query_type": "person_search", '
        '"params": {"filters": {"gender": "female", "last_seen_device": "cam-1"}}}'
    )
    out = _run(QueryParser(llm).parse("women last seen at cam-1"))  # type: ignore[arg-type]
    assert out["query_type"] == "person_search"
    assert out["params"]["filters"]["gender"] == "female"
    assert out["params"]["filters"]["last_seen_device"] == "cam-1"


def test_timeline_query():
    llm = StubLLM('{"query_type": "timeline", "params": {"person_id": 100}}')
    out = _run(QueryParser(llm).parse("where was 100"))  # type: ignore[arg-type]
    assert out["query_type"] == "timeline"
    assert out["params"]["person_id"] == 100


def test_similarity_search():
    llm = StubLLM('{"query_type": "similarity_search", '
                  '"params": {"person_id": 7, "top_k": 5}}')
    out = _run(QueryParser(llm).parse("similar to 7"))  # type: ignore[arg-type]
    assert out["query_type"] == "similarity_search"
    assert out["params"]["top_k"] == 5


def test_sighting_aggregation():
    llm = StubLLM('{"query_type": "sighting_aggregation", '
                  '"params": {"person_id": 5, "group_by": "hour"}}')
    out = _run(QueryParser(llm).parse("how many times by hour for 5"))  # type: ignore[arg-type]
    assert out["query_type"] == "sighting_aggregation"
    assert out["params"]["group_by"] == "hour"


def test_device_lookup_empty_params():
    llm = StubLLM('{"query_type": "device_lookup", "params": {}}')
    out = _run(QueryParser(llm).parse("list cameras"))  # type: ignore[arg-type]
    assert out["query_type"] == "device_lookup"
    assert out["params"] == {}


def test_six_query_types_all_valid():
    """Sanity: the QueryParser's allowed set matches what query_service expects."""
    expected = {
        "person_lookup", "person_search", "timeline",
        "similarity_search", "sighting_aggregation", "device_lookup",
    }
    assert VALID_QUERY_TYPES == expected


def test_unknown_query_type_returns_error():
    llm = StubLLM('{"query_type": "delete_everything", "params": {}}')
    out = _run(QueryParser(llm).parse("nuke it"))  # type: ignore[arg-type]
    assert out["query_type"] == "error"
    assert "Unknown query_type" in out["params"]["reason"]


def test_explicit_error_passthrough():
    llm = StubLLM('{"query_type": "error", "params": {"reason": "ambiguous"}}')
    out = _run(QueryParser(llm).parse("blah"))  # type: ignore[arg-type]
    assert out == {"query_type": "error", "params": {"reason": "ambiguous"}}


def test_invalid_json_returns_error():
    llm = StubLLM("this is not json")
    out = _run(QueryParser(llm).parse("hi"))  # type: ignore[arg-type]
    assert out["query_type"] == "error"
    assert "Invalid JSON" in out["params"]["reason"]


def test_code_fence_wrapped_json_is_handled():
    """Some models wrap JSON in ```json ... ``` despite instructions."""
    llm = StubLLM(
        '```json\n{"query_type": "person_lookup", "params": {"person_id": 1}}\n```'
    )
    out = _run(QueryParser(llm).parse("person 1"))  # type: ignore[arg-type]
    assert out == {"query_type": "person_lookup", "params": {"person_id": 1}}


def test_non_dict_params_returns_error():
    llm = StubLLM('{"query_type": "person_lookup", "params": [1, 2, 3]}')
    out = _run(QueryParser(llm).parse("person 1"))  # type: ignore[arg-type]
    assert out["query_type"] == "error"
    assert "JSON object" in out["params"]["reason"]


def test_empty_query_returns_error_without_calling_llm():
    llm = StubLLM("won't be called")
    out = _run(QueryParser(llm).parse(""))  # type: ignore[arg-type]
    assert out["query_type"] == "error"
    assert llm.call_count == 0


def test_llm_exception_returns_error():
    class ExplodingLLM:
        async def chat(self, *args, **kwargs):
            raise RuntimeError("upstream down")

    out = _run(QueryParser(ExplodingLLM()).parse("anything"))  # type: ignore[arg-type]
    assert out["query_type"] == "error"
    assert "LLM call failed" in out["params"]["reason"]


@pytest.mark.parametrize("qtype", sorted(VALID_QUERY_TYPES))
def test_each_query_type_passes_through(qtype):
    """Smoke-test that each valid query_type round-trips correctly."""
    llm = StubLLM(f'{{"query_type": "{qtype}", "params": {{}}}}')
    out = _run(QueryParser(llm).parse("blah"))  # type: ignore[arg-type]
    assert out["query_type"] == qtype
    assert out["params"] == {}
