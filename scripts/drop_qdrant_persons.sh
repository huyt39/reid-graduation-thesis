#!/usr/bin/env bash
# Drop the Qdrant `persons` collection so the worker recreates it on next start.
# Required when switching embedding backbone (e.g. OSNet -> LMBN) since vectors
# from different models cannot share a collection. Also resets the Redis
# person_id sequence so new persons start from 1 again.
#
# Usage:
#   scripts/drop_qdrant_persons.sh                 # uses localhost defaults
#   QDRANT_URL=http://localhost:6333 REDIS_URL=redis://localhost:6379 scripts/drop_qdrant_persons.sh
set -euo pipefail

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
COLLECTION="${QDRANT_COLLECTION:-persons}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_DB="${REDIS_DB:-0}"
SEQ_KEY="${PERSON_ID_SEQ_KEY:-reid:seq:person_id}"

echo "==> Dropping Qdrant collection '${COLLECTION}' at ${QDRANT_URL}"
http_code=$(curl -s -o /tmp/qdrant_drop.json -w "%{http_code}" -X DELETE "${QDRANT_URL}/collections/${COLLECTION}")
echo "    HTTP ${http_code}"
cat /tmp/qdrant_drop.json && echo
if [[ "${http_code}" != "200" && "${http_code}" != "404" ]]; then
  echo "ERROR: unexpected status from Qdrant" >&2
  exit 1
fi

echo "==> Resetting Redis person_id sequence (${SEQ_KEY}) on ${REDIS_HOST}:${REDIS_PORT}/${REDIS_DB}"
if command -v redis-cli >/dev/null 2>&1; then
  redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" -n "${REDIS_DB}" DEL "${SEQ_KEY}" || true
else
  echo "    (redis-cli not found, skipping. Reset manually if needed.)"
fi

echo "Done. The collection will be recreated automatically when reid_worker next starts."
