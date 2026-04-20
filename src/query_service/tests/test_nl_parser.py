"""Tests for the NL query parser fallback keyword matching."""
import httpx

import pytest

from src.services.nl_parser import NLQueryParser


@pytest.fixture
def parser():
    return NLQueryParser(vllm_url="")  # no vLLM — pure fallback


def test_person_lookup(parser):
    r = parser._fallback_keyword_parse("show me person 42")
    assert r["query_type"] == "person_lookup"
    assert r["params"]["person_id"] == 42


def test_person_lookup_hash(parser):
    r = parser._fallback_keyword_parse("person #7")
    assert r["query_type"] == "person_lookup"
    assert r["params"]["person_id"] == 7


def test_similarity_search(parser):
    r = parser._fallback_keyword_parse("who looks like person 15")
    assert r["query_type"] == "similarity_search"
    assert r["params"]["person_id"] == 15


def test_timeline(parser):
    r = parser._fallback_keyword_parse("where was person 3 today?")
    assert r["query_type"] == "timeline"
    assert r["params"]["person_id"] == 3


def test_aggregation(parser):
    r = parser._fallback_keyword_parse("how many times was person 5 seen grouped by hour")
    assert r["query_type"] == "sighting_aggregation"
    assert r["params"]["person_id"] == 5
    assert r["params"]["group_by"] == "hour"


def test_device_lookup(parser):
    r = parser._fallback_keyword_parse("list all cameras")
    assert r["query_type"] == "device_lookup"


def test_camera_keyword_alone_does_not_trigger_device_lookup(parser):
    r = parser._fallback_keyword_parse("camera")
    assert r["query_type"] == "error"


def test_gender_search(parser):
    r = parser._fallback_keyword_parse("find all males seen today")
    assert r["query_type"] == "person_search"
    assert r["params"]["filters"]["gender"] == "male"


def test_unparseable_returns_error(parser):
    r = parser._fallback_keyword_parse("what is the weather?")
    assert r["query_type"] == "error"


@pytest.mark.asyncio
async def test_parse_via_vllm_rejects_non_dict_params():
    parser = NLQueryParser(vllm_url="http://fake-vllm")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "query_type": "person_lookup",
                "params": ["not", "a", "dict"],
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json):
            return FakeResponse()

    original_client = httpx.AsyncClient
    httpx.AsyncClient = lambda *args, **kwargs: FakeClient()
    try:
        result = await parser._parse_via_vllm("person 7")
    finally:
        httpx.AsyncClient = original_client

    assert result == {
        "query_type": "error",
        "message": "Invalid params shape from vLLM",
    }
