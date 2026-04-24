# reid_worker
Worker service skeleton with uv.

## Env
- `WORKER_PERSON_ID_SEQ_KEY`: Redis key used to allocate globally unique `person_id` via `INCR`.

## Runtime Notes
- `RedisPersonIdAllocator` is used to allocate globally unique `person_id` values across worker instances.
- The worker uses fail-fast behavior for ID allocation: if Redis allocation fails, it logs the error and skips creating a new person for that tracklet instead of falling back to an in-memory counter.
