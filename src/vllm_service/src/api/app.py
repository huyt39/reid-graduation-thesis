from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import structlog
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.core.config import settings
from src.services.llm_client import LLMClient
from src.services.query_parser import QueryParser
from src.services.result_summarizer import ResultSummarizer

logger = structlog.get_logger()

# shared singletons populated by lifespan => init = none, app restart => req reuse prepared object, no need to create new
_llm_client: LLMClient | None = None
_query_parser: QueryParser | None = None
_summarizer: ResultSummarizer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm_client, _query_parser, _summarizer
    _llm_client = LLMClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_seconds,
    )
    _query_parser = QueryParser(
        _llm_client,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
    _summarizer = ResultSummarizer(_llm_client)
    logger.info("vllm_service.started", llm_base_url=settings.llm_base_url,
                model=settings.llm_model)
    yield
    logger.info("vllm_service.stopped")


app = FastAPI(title=settings.service_name, lifespan=lifespan)


@app.middleware("http") #req in, if header has request id=>reuse; else: create new uuid
async def attach_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid4()))
    request.state.request_id = request_id
    started_at = time.perf_counter()
    response = await call_next(request)
    response.headers.setdefault("x-request-id", request_id)
    logger.info(
        "vllm_service.http_response",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
    )
    return response


# schemas

class ParseRequest(BaseModel):
    text: str = Field(..., min_length=1)


class ParseResponse(BaseModel):
    query_type: str
    params: dict


class SummarizeRequest(BaseModel):
    question: str
    query_type: str
    params: dict = Field(default_factory=dict)
    results: Any = None #raw result has many shapes depend on query type


class SummarizeResponse(BaseModel):
    summary: str


# health

@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": settings.service_name}


@app.get("/readyz")
async def readyz():
    llm_ok = False
    if _llm_client is not None:
        llm_ok = await _llm_client.healthy()

    ready = (not settings.require_llm_for_ready) or llm_ok
    payload = {
        "status": "ready" if ready else "not_ready",
        "service": settings.service_name,
        "checks": {"llm_reachable": llm_ok},
        "model": settings.llm_model,
    }
    if not ready:
        raise HTTPException(status_code=503, detail=payload)
    return payload


# business endpoint: parse and summary

@app.post("/parse", response_model=ParseResponse)
async def parse(req: ParseRequest):
    assert _query_parser is not None
    return await _query_parser.parse(req.text)


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    assert _summarizer is not None
    text = await _summarizer.summarize(
        original_question=req.question,
        query_type=req.query_type,
        params=req.params,
        results=req.results,
    )
    return SummarizeResponse(summary=text)
