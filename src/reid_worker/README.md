# reid_worker
Worker service skeleton with uv.

## Env
- `WORKER_PERSON_ID_SEQ_KEY`: Redis key used to allocate globally unique `person_id` via `INCR`.
