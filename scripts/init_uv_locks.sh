#!/usr/bin/env bash
set -euo pipefail
for d in src/edge src/reid_worker src/inference_engine src/streaming src/gateway src/query_service; do
  if command -v uv >/dev/null 2>&1; then
    echo "Locking $d"
    (cd "$d" && uv lock)
  else
    echo "uv not installed; skipping lock for $d"
  fi
done
