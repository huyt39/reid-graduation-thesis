from __future__ import annotations

from fastapi import APIRouter
from pydantic import TypeAdapter, ValidationError

from src.api.deps import get_executor, get_nl_parser
from src.schemas.query import NLQueryRequest, StructuredSearchQuery

router = APIRouter(tags=["search"])

structured_search_query_adapter = TypeAdapter(StructuredSearchQuery)


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
    return {"parsed_query": structured.model_dump(), "result": result}
