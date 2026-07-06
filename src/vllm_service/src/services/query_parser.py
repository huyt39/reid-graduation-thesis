from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import structlog

from src.services.llm_client import LLMClient

log = structlog.get_logger()


VALID_QUERY_TYPES = {
    "person_lookup", # find a person by id
    "person_search", # find people by attributes
    "timeline", # check a person by time
    "similarity_search", # find similar people
    "sighting_aggregation", # count sightings
    "device_lookup", # device info
}

# regex: rule to parse simple sentence without calling llm
PERSON_COUNT_PATTERNS = [
    r"\bhow many\s+(people|persons)\b",
    r"\b(total|count|number of)\s+(people|persons)\b",
    r"\bhow many\s+(unique\s+)?(people|persons)\s+(are there|were seen|seen)\b",
]

GENDER_TERMS = {
    "female": [r"\bfemale\b", r"\bwomen\b", r"\bwoman\b"],
    "male": [r"\bmale\b", r"\bmen\b", r"\bman\b"],
}

ATTRIBUTE_PATTERNS: list[tuple[str, str, list[str]]] = [
    ("glasses", "no_glasses", [r"\bno glasses\b", r"\bwithout glasses\b"]),
    ("glasses", "glasses", [r"\bglasses\b", r"\bwearing glasses\b"]),
    ("backpack", "no_backpack", [r"\bno backpack\b", r"\bwithout backpack\b"]),
    ("backpack", "backpack", [r"\bbackpack\b", r"\bwith backpack\b"]),
    ("sidebag", "no_sidebag", [r"\bno sidebag\b", r"\bwithout sidebag\b"]),
    ("sidebag", "sidebag", [r"\bsidebag\b", r"\bside bag\b"]),
    ("hat", "no_hat", [r"\bno hat\b", r"\bwithout hat\b"]),
    ("hat", "hat", [r"\bhat\b", r"\bwearing hat\b"]),
    ("sleeve", "short_sleeve", [r"\bshort sleeve\b", r"\bshort sleeves\b"]),
    ("sleeve", "long_sleeve", [r"\blong sleeve\b", r"\blong sleeves\b"]),
    ("lower", "trousers", [r"\btrousers\b", r"\bpants\b"]),
    ("lower", "shorts", [r"\bshorts\b"]),
    ("age_child", "child", [r"\bchild\b", r"\bchildren\b", r"\bkid\b", r"\bkids\b"]),
    ("age_child", "adult", [r"\badult\b", r"\badults\b"]),
]

REQUIRED_PERSON_ID_QUERY_TYPES = {
    "person_lookup",
    "timeline",
    "similarity_search",
}


SYSTEM_PROMPT = """You are a query parser for a person re-identification surveillance system. \
Convert the user's natural-language request into a JSON object describing what kind of database \
query to run.

Output ONLY a JSON object — no prose, no code fences, no markdown. The JSON has exactly two fields:
  - "query_type": exactly one of:
      "person_lookup", "person_search", "timeline",
      "similarity_search", "sighting_aggregation", "device_lookup"
  - "params": a JSON object whose shape depends on query_type.

Schemas:
  1. person_lookup — look up a single person by ID.
     params: {"person_id": <int>}
  2. person_search — search for persons by attributes.
     params: {"filters": {"gender"?: "male"|"female",
                            "age_child"?: "adult"|"child",
                            "backpack"?: "backpack"|"no_backpack",
                            "sidebag"?: "sidebag"|"no_sidebag",
                            "hat"?: "hat"|"no_hat",
                            "glasses"?: "glasses"|"no_glasses",
                            "sleeve"?: "short_sleeve"|"long_sleeve",
                            "lower"?: "trousers"|"shorts",
                            "last_seen_device"?: <str>,
                            "first_seen_after"?: <ISO datetime>,
                            "first_seen_before"?: <ISO datetime>,
                            "last_seen_after"?: <ISO datetime>,
                            "last_seen_before"?: <ISO datetime>,
                            "min_sighting_count"?: <int>,
                            "is_active"?: <bool>},
              "page"?: <int>, "page_size"?: <int>}
     Use person_search with empty filters {} for questions asking how many people/persons exist,
     how many unique people were seen, total people, or list/show all people. The query result's
     "total" field is the count.
  3. timeline — events for a specific person over time.
     params: {"person_id": <int>,
              "start_time"?: <ISO datetime>,
              "end_time"?: <ISO datetime>,
              "event_types"?: [<str>]}
  4. similarity_search — visually similar persons.
     params: {"person_id": <int>, "top_k"?: <int>, "min_score"?: <float>}
  5. sighting_aggregation — count/group sightings.
     params: {"person_id"?: <int>, "device_id"?: <str>,
              "start_time"?: <ISO datetime>, "end_time"?: <ISO datetime>,
              "group_by"?: "hour"|"day"|"device"}
  6. device_lookup — info about cameras/devices.
     params: {"device_id"?: <str>}  (empty params {} is allowed for "list all")

Resolve relative dates against the current datetime: {now}.
- "today"     → start_time = current date at 00:00:00 UTC
- "yesterday" → start_time = (current_date - 1 day) at 00:00:00 UTC, end_time = same date at 23:59:59
- "last hour" → start_time = (now - 1h)
- "last 24h"  → start_time = (now - 24h)

If you genuinely cannot map the query to one of the 6 types, return:
{"query_type": "error", "params": {"reason": "<why>"}}

Do not invent person_ids or device_ids that the user did not mention. If a person_id is
required by the schema but the user did not provide one, return an error."""


FEW_SHOT_EXAMPLES: list[dict] = [
    {"role": "user", "content": "show me person 42"},
    {"role": "assistant",
     "content": '{"query_type": "person_lookup", "params": {"person_id": 42}}'},

    {"role": "user", "content": "find all women"},
    {"role": "assistant",
     "content": '{"query_type": "person_search", "params": {"filters": {"gender": "female"}}}'},

    {"role": "user", "content": "how many people are there"},
    {"role": "assistant",
     "content": '{"query_type": "person_search", "params": {"filters": {}, "page": 1, "page_size": 20}}'},

    {"role": "user", "content": "count all persons"},
    {"role": "assistant",
     "content": '{"query_type": "person_search", "params": {"filters": {}, "page": 1, "page_size": 20}}'},

    # The system prompt provides the current datetime; the LLM is expected to
    # substitute the right ISO value. The example shows the output *format* only.
    {"role": "user", "content": "where was person 100 between Jan 1 and Jan 2 2024?"},
    {"role": "assistant",
     "content": ('{"query_type": "timeline", "params": '
                 '{"person_id": 100, '
                 '"start_time": "2024-01-01T00:00:00+00:00", '
                 '"end_time": "2024-01-02T23:59:59+00:00"}}')},

    {"role": "user", "content": "people similar to person 7"},
    {"role": "assistant",
     "content": '{"query_type": "similarity_search", "params": {"person_id": 7, "top_k": 10}}'},

    {"role": "user", "content": "how many times did person 5 appear by hour"},
    {"role": "assistant",
     "content": ('{"query_type": "sighting_aggregation", "params": '
                 '{"person_id": 5, "group_by": "hour"}}')},

    {"role": "user", "content": "list all cameras"},
    {"role": "assistant",
     "content": '{"query_type": "device_lookup", "params": {}}'},

    {"role": "user", "content": "show me men last seen at camera-1"},
    {"role": "assistant",
     "content": ('{"query_type": "person_search", "params": '
                 '{"filters": {"gender": "male", "last_seen_device": "camera-1"}}}')},

    {"role": "user", "content": "how many people wear glasses"},
    {"role": "assistant",
     "content": '{"query_type": "person_search", "params": {"filters": {"glasses": "glasses"}, "page": 1, "page_size": 20}}'},

    {"role": "user", "content": "find women with backpack and hat"},
    {"role": "assistant",
     "content": ('{"query_type": "person_search", "params": '
                 '{"filters": {"gender": "female", "backpack": "backpack", "hat": "hat"}}}')},
]

# helper functions:
# drop fence to keep json loadss
def _strip_code_fence(content: str) -> str:
    """Some models wrap JSON in ```json ... ``` despite instructions; strip it."""
    s = content.strip()
    if s.startswith("```"):
        # Drop opening fence (with optional language) and trailing fence.
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

# get person id from params
def _person_id_from_params(params: dict[str, Any]) -> int | None:
    person_id = params.get("person_id")
    return person_id if isinstance(person_id, int) else None

# no use llm, just detect
def _deterministic_parse(text: str) -> dict[str, Any] | None:
    q = text.lower().strip()
    filters: dict[str, Any] = {}

    if any(re.search(pattern, q) for pattern in GENDER_TERMS["female"]):
        filters["gender"] = "female"
    elif any(re.search(pattern, q) for pattern in GENDER_TERMS["male"]):
        filters["gender"] = "male"

    for attr, value, patterns in ATTRIBUTE_PATTERNS:
        if attr not in filters and any(re.search(pattern, q) for pattern in patterns):
            filters[attr] = value

    has_count_intent = any(
        phrase in q
        for phrase in [
            "how many",
            "count",
            "number of",
            "total",
        ]
    )
    has_person_id = re.search(r"\bperson\s*#?\s*\d+\b", q) is not None
    is_sighting_count = any(term in q for term in ["times", "sighting", "appear"])
    if has_count_intent and filters and not has_person_id and not is_sighting_count:
        return {
            "query_type": "person_search",
            "params": {"filters": filters, "page": 1, "page_size": 20},
        }

    has_search_intent = any(
        phrase in q
        for phrase in [
            "find",
            "show",
            "list",
            "search",
            "people with",
            "persons with",
            "wearing",
            "who have",
        ]
    )
    if filters and has_search_intent and not has_person_id:
        return {"query_type": "person_search", "params": {"filters": filters}}

    if any(re.search(pattern, q) for pattern in PERSON_COUNT_PATTERNS):
        return {
            "query_type": "person_search",
            "params": {"filters": {}, "page": 1, "page_size": 20},
        }

    if any(phrase in q for phrase in ["list all people", "list all persons", "show all people", "show all persons"]):
        return {
            "query_type": "person_search",
            "params": {"filters": {}, "page": 1, "page_size": 20},
        }

    return None


def _validate_required_params(qtype: str, params: dict[str, Any]) -> dict[str, Any] | None:
    if qtype in REQUIRED_PERSON_ID_QUERY_TYPES and _person_id_from_params(params) is None:
        return {
            "query_type": "error",
            "params": {"reason": f"{qtype} requires person_id"},
        }
    return None


class QueryParser:
    def __init__(self, llm_client: LLMClient, *, temperature: float = 0.0,
                 max_tokens: int = 512) -> None:
        self.llm = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens

    # build message for llm
    def _messages(self, text: str) -> list[dict]:
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        system = SYSTEM_PROMPT.replace("{now}", now_iso)
        return [
            {"role": "system", "content": system},
            *FEW_SHOT_EXAMPLES,
            {"role": "user", "content": text},
        ]

    async def parse(self, text: str) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {"query_type": "error", "params": {"reason": "empty query"}}

        deterministic = _deterministic_parse(text)
        if deterministic is not None:
            return deterministic

        try:
            content = await self.llm.chat(
                self._messages(text),
                response_format_json=True,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            log.warning("query_parser.llm_call_failed", error=str(exc))
            return {"query_type": "error",
                    "params": {"reason": f"LLM call failed: {exc}"}}

        cleaned = _strip_code_fence(content)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            log.warning("query_parser.json_decode_failed",
                        content_preview=cleaned[:200])
            return {"query_type": "error",
                    "params": {"reason": f"Invalid JSON from LLM: {exc}"}}

        qtype = parsed.get("query_type")
        params = parsed.get("params", {})

        if qtype == "error":
            # LLM explicitly signalled it couldn't parse — pass through.
            return {"query_type": "error",
                    "params": params if isinstance(params, dict) else {}}

        if qtype not in VALID_QUERY_TYPES:
            return {"query_type": "error",
                    "params": {"reason": f"Unknown query_type: {qtype!r}"}}

        if not isinstance(params, dict):
            return {"query_type": "error",
                    "params": {"reason": "params must be a JSON object"}}

        invalid = _validate_required_params(qtype, params)
        if invalid is not None:
            return invalid

        return {"query_type": qtype, "params": params}
