"""Thin OpenAI-compatible HTTP client.

Sends chat-completion requests to any endpoint that implements the OpenAI
``/chat/completions`` shape: vLLM's OpenAI server, Ollama's /v1 layer, OpenAI
itself, etc. The model is configured per-service via ``VLLM_LLM_MODEL``.
"""
from __future__ import annotations

import httpx
import structlog

log = structlog.get_logger()


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def chat(
        self,
        messages: list[dict],
        *,
        response_format_json: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """POST /chat/completions and return the assistant message content."""
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format_json:
            # Honoured by OpenAI + vLLM; harmless on backends that ignore it.
            body["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response shape: {data}") from exc

    async def healthy(self) -> bool:
        """Best-effort readiness probe — checks the upstream /models endpoint."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self.base_url}/models", headers=self._headers(),
                )
            return resp.status_code < 500
        except Exception as exc:
            log.warning("llm_client.health_probe_failed", error=str(exc))
            return False
