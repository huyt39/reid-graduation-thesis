#!/usr/bin/env bash
#
# Run the FULL pipeline with the inference_engine on the Apple Silicon GPU (MPS).
# The engine runs natively on the host (Docker on macOS has no Metal passthrough);
# every other service stays in Docker and talks to it via host.docker.internal.
#
# The engine serves on :8000 (demo.sh's MPS_NATIVE mode points workers there).
# Make sure nothing else holds :8000 before running.
#
# Usage:
#   scripts/run_mps_demo.sh                 # engine on MPS + full stack + reset
#   scripts/run_mps_demo.sh --build         # rebuild app images first (apply code changes)
#   scripts/run_mps_demo.sh --no-reset      # keep identity stores
#   flags combine, e.g.: scripts/run_mps_demo.sh --build --no-reset
#
# NOTE: MPS is NOT bit-identical with CPU → ReID results are non-deterministic.
# Good for live/throughput demos, NOT for reproducible A/B evaluation.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MPS_PORT="${MPS_PORT:-8000}"
ENGINE_LOG="/tmp/mps_engine_${MPS_PORT}.log"
RESET_FLAG="--reset"
BUILD_FLAG=""
for arg in "$@"; do
  case "$arg" in
    --no-reset) RESET_FLAG="" ;;
    --build)    BUILD_FLAG="--build" ;;
    *) echo "unknown arg: $arg (use --build / --no-reset)" >&2; exit 1 ;;
  esac
done

# 1. Free the engine port — but ONLY kill our own MPS engine process (a leftover
#    `python -m src` / uvicorn from a previous run). NEVER kill Docker Desktop's
#    backend or any other server that happens to hold the port: killing
#    com.docker.backend takes Docker down with it.
kill_our_engine_on_port() {
  local pid cmd killed=0
  # only the LISTENing process — not stray CLOSED/ESTABLISHED client sockets
  # (e.g. Docker's backend keeps closed connections to the port and must be ignored)
  for pid in $(lsof -ti ":${MPS_PORT}" -sTCP:LISTEN 2>/dev/null); do
    cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    case "$cmd" in
      *com.docker*|*Docker.app*|*vpnkit*)
        echo "ERROR: :${MPS_PORT} is held by Docker (pid $pid) — refusing to kill it." >&2
        echo "       Pick a different host port (MPS_PORT=8200 $0) or free :${MPS_PORT} manually." >&2
        exit 1 ;;
      *-m\ src*|*uvicorn*|*.venv-mps*|*run_native_mps*)
        echo "  killing previous MPS engine (pid $pid) ..."
        kill "$pid" 2>/dev/null || true; killed=1 ;;
      *)
        echo "ERROR: :${MPS_PORT} is held by an unknown process (pid $pid): ${cmd}" >&2
        echo "       Not killing it. Free :${MPS_PORT} yourself or change MPS_PORT." >&2
        exit 1 ;;
    esac
  done
  [ "$killed" = 1 ] || return 0
  for _ in 1 2 3 4 5; do lsof -ti ":${MPS_PORT}" -sTCP:LISTEN >/dev/null 2>&1 || { echo "  port ${MPS_PORT} freed."; return 0; }; sleep 1; done
  lsof -ti ":${MPS_PORT}" -sTCP:LISTEN | xargs kill -9 2>/dev/null || true; sleep 1
}
kill_our_engine_on_port

# 2. Start the native MPS engine in the background.
echo "Starting native MPS inference_engine on :${MPS_PORT} (log: ${ENGINE_LOG}) ..."
ENGINE_PORT="${MPS_PORT}" nohup "${ROOT_DIR}/scripts/run_native_mps.sh" >"${ENGINE_LOG}" 2>&1 &
ENGINE_PID=$!
echo "  engine pid=${ENGINE_PID}"

# 3. Wait until it is serving.
echo "Waiting for engine readiness ..."
for i in $(seq 1 60); do
  if curl -sf "http://localhost:${MPS_PORT}/readyz" >/dev/null 2>&1; then
    echo "  engine ready: $(curl -s http://localhost:${MPS_PORT}/readyz)"
    break
  fi
  if ! kill -0 "${ENGINE_PID}" 2>/dev/null; then
    echo "ERROR: engine process died. Last log lines:" >&2
    tail -20 "${ENGINE_LOG}" >&2
    exit 1
  fi
  sleep 5
  [ "$i" = "60" ] && { echo "ERROR: engine not ready after 5 min. See ${ENGINE_LOG}" >&2; exit 1; }
done

# 4. Bring up the full stack in MPS mode (skips the engine container, points
#    workers at the host engine on :8000).
echo "Starting full pipeline (MPS_NATIVE) ..."
MPS_NATIVE=true "${ROOT_DIR}/scripts/demo.sh" up ${BUILD_FLAG} ${RESET_FLAG}

cat <<EOF

============================================================
Full pipeline is up with the engine on GPU (MPS, :${MPS_PORT}).

Measure detector FPS per camera (let it run ~30s first):
  docker logs deploy-edge_cam1-1 2>&1 | grep edge_progress | tail -3
  docker logs deploy-edge_cam2-1 2>&1 | grep edge_progress | tail -3
  # detector fps ≈ 1000 / avg_detect_ms

Live CPU per container:
  docker stats --no-stream | grep -E 'edge_cam|worker_cam'

Stop the native engine when done:
  kill ${ENGINE_PID}    # or: lsof -ti :${MPS_PORT} | xargs kill
Return to CPU mode:
  scripts/demo.sh up --reset
============================================================
EOF
