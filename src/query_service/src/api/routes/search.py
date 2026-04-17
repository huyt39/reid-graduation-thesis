from __future__ import annotations

from fastapi import APIRouter

from src.api.deps import get_executor, get_nl_parser
from src.schemas.query import NLQueryRequest, StructuredQueryRequest

router = APIRouter(tags=["search"])


@router.post("/search")
async def structured_search(query: StructuredQueryRequest):
    """Execute a structured JSON query."""
    executor = get_executor()
    return await executor.execute(query.model_dump())


@router.post("/query/natural")
async def natural_language_query(body: NLQueryRequest):
    """Parse natural language query and execute it."""
    parser = get_nl_parser()
    executor = get_executor()
    parsed = await parser.parse(body.query)
    result = await executor.execute(parsed)
    return {"parsed_query": parsed, "result": result}
