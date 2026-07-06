from __future__ import annotations

import json
from typing import Any

import structlog

from src.services.llm_client import LLMClient

log = structlog.get_logger()

# optional because if summarizer error => query service still has deterministic summary fallback
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a \
person re-identification surveillance system. You are given the user's original \
question, the structured query that was run, and the raw JSON results.

Write a clear, natural-language answer that directly addresses the question.

Guidelines:
- Use Markdown. Open with one short sentence that answers the question, then add \
a bullet list when reporting multiple people, devices, time buckets, or matches. \
Keep it concise — no preamble, no headings, no restating the query.
- Ground every statement in the provided results only. Be specific with counts, \
IDs, attributes, cameras, and timestamps. Never invent data not present in the results.
- If the results are empty or contain an error, say so plainly in one sentence.
- Answer in the same language as the user's question."""


class ResultSummarizer:
    def __init__(self, llm_client: LLMClient, *, max_tokens: int = 512) -> None:
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
            f"Results: {json.dumps(results, default=str)[:8000]}"
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
