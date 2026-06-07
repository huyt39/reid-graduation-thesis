#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/macbook/Documents/reid-production"

run_service_tests() {
  local service_dir="$1"
  shift

  echo
  echo "==> ${service_dir}"
  (
    cd "$ROOT_DIR/$service_dir"
    pytest "$@"
  )
}

run_service_tests "src/inference_engine" tests/test_healthz.py tests/test_endpoints.py
run_service_tests "src/query_service" tests/test_healthz.py
run_service_tests "src/streaming" tests/test_healthz.py
run_service_tests "src/gateway" tests/test_healthz.py

echo
echo "All readiness test groups passed."
