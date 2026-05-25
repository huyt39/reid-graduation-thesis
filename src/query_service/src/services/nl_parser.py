"""Natural language to structured query parser.

Calls vLLM for parsing when available, falls back to regex keyword matching.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import httpx
import structlog
from fastapi.encoders import jsonable_encoder

log = structlog.get_logger()

VALID_QUERY_TYPES = {
    "person_lookup", "person_search", "timeline",
    "similarity_search", "sighting_aggregation", "device_lookup", "error",
}


class NLQueryParser:
    def __init__(self, vllm_url: str = "") -> None:
        self.vllm_url = vllm_url.rstrip("/") if vllm_url else ""

    async def parse(self, text: str) -> dict:
        if self.vllm_url:
            try:
                return await self._parse_via_vllm(text)
            except Exception:
                log.warning("nl_parser.vllm_fallback", exc_info=True)
        return self._fallback_keyword_parse(text)

    async def _parse_via_vllm(self, text: str) -> dict:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{self.vllm_url}/parse",
                json={"text": text},
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("query_type") in VALID_QUERY_TYPES:
                params = result.get("params", {})
                if isinstance(params, dict):
                    return result
                return {"query_type": "error", "message": "Invalid params shape from vLLM"}
            return {"query_type": "error", "message": "Unknown query type from vLLM"}

    async def summarize(
        self,
        *,
        question: str,
        query_type: str,
        params: dict,
        results: object,
    ) -> str:
        if not self.vllm_url:
            return ""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{self.vllm_url}/summarize",
                    json=jsonable_encoder({
                        "question": question,
                        "query_type": query_type,
                        "params": params,
                        "results": results,
                    }),
                )
                resp.raise_for_status()
                data = resp.json()
                summary = data.get("summary", "")
                return summary if isinstance(summary, str) else ""
        except Exception:
            log.warning("nl_parser.summary_failed", exc_info=True)
            return ""

    @staticmethod
    def _fallback_keyword_parse(text: str) -> dict:
        q = text.lower().strip()

        # Person lookup: "person 42", "show person #42"
        match = re.search(r"person\s*#?\s*(\d+)", q)
        pid = int(match.group(1)) if match else None

        # Similarity
        if pid and any(kw in q for kw in ["similar", "looks like", "resembles", "like person"]):
            return {"query_type": "similarity_search", "params": {"person_id": pid, "top_k": 10}}

        # Timeline
        if pid and any(kw in q for kw in ["where was", "timeline", "history", "where is"]):
            return {"query_type": "timeline", "params": {"person_id": pid}}

        # Aggregation
        if pid and any(kw in q for kw in ["how many", "count", "aggregate", "times"]):
            params: dict = {"person_id": pid}
            if "hour" in q:
                params["group_by"] = "hour"
            elif "day" in q:
                params["group_by"] = "day"
            elif "device" in q or "camera" in q:
                params["group_by"] = "device"
            return {"query_type": "sighting_aggregation", "params": params}

        # Plain person lookup
        if pid and not any(kw in q for kw in ["male", "female", "search", "find all"]):
            return {"query_type": "person_lookup", "params": {"person_id": pid}}

        # Device lookup
        if any(
            phrase in q for phrase in [
                "list cameras",
                "list devices",
                "show devices",
                "show cameras",
                "all devices",
                "all cameras",
                ]):
            return {"query_type": "device_lookup", "params": {}}

        # Person search with filters
        filters: dict = {}
        if "male" in q and "female" not in q:
            filters["gender"] = "male"
        elif "female" in q:
            filters["gender"] = "female"
        if "today" in q:
            now = datetime.now(timezone.utc)
            filters["last_seen_after"] = now.replace(hour=0, minute=0, second=0).isoformat()
        if filters:
            return {"query_type": "person_search", "params": {"filters": filters}}

        return {"query_type": "error", "message": f"Could not parse: {text}"}
