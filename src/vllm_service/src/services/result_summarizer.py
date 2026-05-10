"""Plain-English summarizer for structured query results.

Optional companion to ``query_parser`` — the UI can call ``POST /summarize`` with
the original query + the structured response and get a short human-readable blurb.
"""
from __future__ import annotations

import json
from typing import Any

import structlog

from src.services.llm_client import LLMClient

log = structlog.get_logger()


SYSTEM_PROMPT = """You are a concise assistant that summarizes structured query \
results from a person re-identification surveillance system. Given the user's \
original question, the structured query that was run, and the raw results, \
write 1-2 sentences explaining what was found. Be specific with counts and \
identifiers. Do not invent data. If results are empty, say so plainly."""


class ResultSummarizer:
    def __init__(self, llm_client: LLMClient, *, max_tokens: int = 200) -> None:
        self.llm = llm_client
        self.max_tokens = max_tokens

    async def summarize(
        self,
        original_question: str,
        query_type: str,
        params: dict[str, Any],
        results: Any,
    ) -> str:
        user_msg = (
            f"User asked: {original_question!r}\n"
            f"Query type: {query_type}\n"
            f"Params: {json.dumps(params, default=str)}\n"
            f"Results: {json.dumps(results, default=str)[:4000]}"
        )
        try:
            return (await self.llm.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=self.max_tokens,
            )).strip()
        except Exception as exc:
            log.warning("result_summarizer.failed", error=str(exc))
            return ""
