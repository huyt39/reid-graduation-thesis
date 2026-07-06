# vllm_service

LLM-backed natural-language layer for the ReID system. It powers the UI **Search**
tab: turning a user's plain-language question into a structured query, and turning the
query results back into a readable answer.

It is an OpenAI-compatible *client* — it does not host a model itself. Point it at any
OpenAI-compatible endpoint (a local vLLM container, Ollama, OpenAI, etc.) via env.

## Responsibilities

1. **Query parsing** (`POST /parse`) — convert NL → a structured query JSON
   (`query_parser.py`). Tries a deterministic regex parse first (fast, LLM-free) and
   only calls the LLM when no pattern matches. Output is validated against 6 query
   types: `person_lookup`, `person_search`, `timeline`, `similarity_search`,
   `sighting_aggregation`, `device_lookup`. Runs at `temperature=0.0` in JSON mode.
2. **Result summarization** (`POST /summarize`) — given the original question, the
   structured query, and the raw results, produce a natural-language **Markdown**
   answer (`result_summarizer.py`).

## Endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET | `/healthz` | — | `{"status":"ok","service":...}` |
| GET | `/readyz` | — | readiness + `{"checks":{"llm_reachable":bool},"model":...}` (503 if `require_llm_for_ready` and LLM down) |
| POST | `/parse` | `{"text": str}` | `{"query_type": str, "params": dict}` |
| POST | `/summarize` | `{"question": str, "query_type": str, "params": dict, "results": any}` | `{"summary": str}` (Markdown; `""` on LLM failure) |

## How it fits the Search flow

```
UI (search tab) → gateway → query_service POST /query/natural
   ├─ query_service → vllm_service POST /parse   → structured query
   ├─ query_service executes query (Mongo + Qdrant)
   └─ query_service → vllm_service POST /summarize → Markdown answer
```

- The **answer shown to the user is the `summary`**, rendered as Markdown by the UI
  (`src/ui/app/search/_components/search-form.tsx`, via `react-markdown`). The parsed
  query and raw DB result are kept in a collapsible "Details" block.
- Summary ordering lives in `query_service` (`src/api/routes/search.py`): it calls
  this service's `/summarize` **first** (rich natural answer), and only falls back to
  its own deterministic template strings (`_deterministic_summary`) when `/summarize`
  returns empty — i.e. when the LLM is unreachable or errors. This keeps the stack
  usable with no live LLM.

### Summary prompt contract (`result_summarizer.py`)

The system prompt instructs the model to: answer in Markdown (one lead sentence, then
bullets for multiple items), stay **grounded** in the provided results (never invent
data), say so plainly on empty/error results, and **reply in the same language as the
question**. Generation uses `max_tokens=512` and sends up to the first 8000 chars of
the results JSON.

## Env (`.env`, prefix `VLLM_`)

| Var | Default (code) | Notes |
|-----|----------------|-------|
| `VLLM_LLM_BASE_URL` | `http://vllm:8000/v1` | OpenAI-compatible endpoint |
| `VLLM_LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | model name on that endpoint |
| `VLLM_LLM_API_KEY` | `""` | optional |
| `VLLM_LLM_TIMEOUT_SECONDS` | `30.0` | per-request timeout |
| `VLLM_TEMPERATURE` | `0.0` | parse generation temperature |
| `VLLM_MAX_TOKENS` | `512` | parse generation cap (summarizer caps itself at 512) |
| `VLLM_REQUIRE_LLM_FOR_READY` | `false` | if true, `/readyz` 503s when the LLM is unreachable |
| `SERVICE_PORT` | `8100` | uvicorn port (host-exposed as `18100` in compose) |

Local dev (`src/vllm_service/.env`) points at Ollama on the host
(`http://host.docker.internal:11434/v1`). Production uses `Qwen2.5-7B-Instruct`.

## Runtime / Deploy Notes

- **Keep the local model light.** This box can't run large models — a 14B model
  (~9 GB) froze the dev machine. The local default is a small model (e.g.
  `llama3.2` ~2 GB or `qwen2.5:0.5b` ~0.4 GB); answer quality scales with model size,
  but heavy models are not an option locally. To unload a model from Ollama:
  `curl http://localhost:11434/api/generate -d '{"model":"<name>","keep_alive":0}'`.
- `query_service` has `depends_on: vllm_service`, and its NL parser also has a regex
  fallback, so a flaky LLM should not take the stack down (`VLLM_REQUIRE_LLM_FOR_READY=false`).
- **`vllm_service` is NOT in `scripts/demo.sh`'s rebuilt app-services list.** After
  editing code here, `demo.sh up --build` will **not** rebuild it. Rebuild manually:
  `docker compose -f src/deploy/docker-compose.yml build vllm_service` then
  `docker compose -f src/deploy/docker-compose.yml up -d --force-recreate vllm_service`
  (a model/env change only needs the second command).
