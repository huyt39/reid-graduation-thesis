from __future__ import annotations

from fastapi import APIRouter
from pydantic import TypeAdapter, ValidationError

from src.api.deps import get_executor, get_nl_parser
from src.schemas.query import NLQueryRequest, StructuredSearchQuery

router = APIRouter(tags=["search"])

structured_search_query_adapter = TypeAdapter(StructuredSearchQuery)

ATTRIBUTE_LABELS = {
    "gender": {"male": "male", "female": "female"},
    "age_child": {"child": "child", "adult": "adult"},
    "backpack": {"backpack": "with backpack", "no_backpack": "without backpack"},
    "sidebag": {"sidebag": "with sidebag", "no_sidebag": "without sidebag"},
    "hat": {"hat": "with hat", "no_hat": "without hat"},
    "glasses": {"glasses": "with glasses", "no_glasses": "without glasses"},
    "sleeve": {"short_sleeve": "with short sleeves", "long_sleeve": "with long sleeves"},
    "lower": {"trousers": "wearing trousers", "shorts": "wearing shorts"},
}


def _deterministic_summary(query_type: str, params: dict, result: dict) -> str:
    if query_type == "person_search" and "total" in result:
        total = result.get("total", 0)
        filters = params.get("filters", {}) if isinstance(params, dict) else {}
        qualifiers: list[str] = []
        if isinstance(filters, dict):
            for attr, label_map in ATTRIBUTE_LABELS.items():
                value = filters.get(attr)
                if value:
                    qualifiers.append(label_map.get(str(value), str(value)))
            device = filters.get("last_seen_device")
        else:
            device = None
        if device:
            qualifiers.append(f"last seen at {device}")
        if qualifiers:
            return f"Found {total} people matching: {', '.join(qualifiers)}."
        return f"Found {total} people matching the query."
    if query_type == "person_lookup":
        person = result.get("person")
        if isinstance(person, dict) and "person_id" in person:
            return f"Found person {person['person_id']}."
        if "error" in result:
            return str(result["error"])
    if query_type == "device_lookup":
        if "devices" in result and isinstance(result["devices"], list):
            return f"Found {len(result['devices'])} devices."
        device = result.get("device")
        if isinstance(device, dict):
            return "Found the requested device."
        if "error" in result:
            return str(result["error"])
    if query_type == "sighting_aggregation" and isinstance(result.get("aggregation"), list):
        return f"Found {len(result['aggregation'])} aggregation buckets."
    if query_type == "timeline" and "total" in result:
        return f"Found {result.get('total', 0)} timeline events."
    if query_type == "similarity_search" and isinstance(result.get("similar_persons"), list):
        return f"Found {len(result['similar_persons'])} similar people."
    return ""


@router.post("/search")
async def structured_search(query: StructuredSearchQuery):
    """Execute a structured JSON query."""
    executor = get_executor()
    return await executor.execute(query)


@router.post("/query/natural")
async def natural_language_query(body: NLQueryRequest):
    """Parse natural language query and execute it."""
    parser = get_nl_parser()
    executor = get_executor()

    parsed = await parser.parse(body.query)
    if parsed.get("query_type") == "error":
        return {"parsed_query": parsed, "result": parsed}

    try:
        structured = structured_search_query_adapter.validate_python(parsed)
    except ValidationError as exc:
        error = {
            "query_type": "error",
            "message": "Parsed query failed validation",
            "details": exc.errors(),
        }
        return {"parsed_query": parsed, "result": error}

    result = await executor.execute(structured)
    structured_payload = structured.model_dump()
    summary = _deterministic_summary(
        structured_payload["query_type"],
        structured_payload["params"],
        result,
    )
    if not summary:
        summary = await parser.summarize(
            question=body.query,
            query_type=structured_payload["query_type"],
            params=structured_payload["params"],
            results=result,
        )
    return {
        "parsed_query": structured_payload,
        "result": result,
        "summary": summary,
    }
