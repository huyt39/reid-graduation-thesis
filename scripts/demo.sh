#!/usr/bin/env bash
#
# Unified demo driver. Replaces demo-fast.sh, demo-run.sh, demo-replay.sh,
# demo-reset-identities.sh.
#
# Usage:
#   scripts/demo.sh up                 # bring stack up (no build, no DB reset)
#   scripts/demo.sh up --build         # rebuild app images first
#   scripts/demo.sh up --reset         # clear identity stores first
#   scripts/demo.sh up --build --reset # both
#   scripts/demo.sh replay             # restart edge to re-play the source video
#   scripts/demo.sh reset              # clear identity stores only (Mongo/Qdrant/Redis/MinIO/Kafka)
#   scripts/demo.sh diag               # print mongo histogram + occlusion / person counts
#   scripts/demo.sh down               # docker compose down -v
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/src/deploy/docker-compose.yml"
VIEWER_FILE="$ROOT_DIR/demo/stream-view.html"
MONGO_DB="reid_production"
QDRANT_COLLECTION="persons"
QDRANT_AUX_COLLECTION="person_aux_upper"
QDRANT_VECTOR_SIZE="512"
REDIS_DB="0"
KAFKA_TOPICS=("reid_input_cam1" "reid_input_cam2" "reid_output" "edge_preview")
EDGE_SERVICES=("edge_cam1" "edge_cam2")
KAFKA_BOOTSTRAP="localhost:9092"

# ── helpers ────────────────────────────────────────────────────────────────

wait_for_http() {
  local name="$1" url="$2" max_attempts="${3:-60}" attempt=1
  until curl --silent --show-error --fail "$url" >/dev/null 2>&1; do
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "Timed out waiting for $name at $url" >&2
      return 1
    fi
    sleep 2; attempt=$((attempt + 1))
  done
}

wait_for_http_optional() {
  local name="$1" url="$2" max_attempts="${3:-15}"
  if ! wait_for_http "$name" "$url" "$max_attempts"; then
    echo "Warning: $name did not become ready in time; continuing anyway." >&2
    return 1
  fi
}

wait_for_tcp_port() {
  local name="$1" host="$2" port="$3" max_attempts="${4:-60}" attempt=1
  until bash -c "echo > /dev/tcp/$host/$port" >/dev/null 2>&1; do
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "Timed out waiting for $name on $host:$port" >&2
      return 1
    fi
    sleep 2; attempt=$((attempt + 1))
  done
}

wait_for_mongo() {
  local max_attempts="${1:-60}" attempt=1
  until docker compose -f "$COMPOSE_FILE" exec -T mongo \
      mongosh --quiet --eval "db.adminCommand({ ping: 1 }).ok" >/dev/null 2>&1; do
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "Timed out waiting for MongoDB to accept connections" >&2
      return 1
    fi
    sleep 2; attempt=$((attempt + 1))
  done
}

# ── subcommands ────────────────────────────────────────────────────────────

cmd_reset() {
  echo "Starting identity stores if needed..."
  docker compose -f "$COMPOSE_FILE" up -d kafka mongo qdrant redis minio >/dev/null

  echo "Waiting for MongoDB, Qdrant..."
  wait_for_mongo 30
  wait_for_http "qdrant" "http://localhost:16333/readyz" 30

  echo "Clearing MongoDB identity collections..."
  docker compose -f "$COMPOSE_FILE" exec -T mongo mongosh --quiet --eval "
const dbh = db.getSiblingDB('$MONGO_DB');
for (const name of ['persons', 'tracklets', 'sightings', 'timeline', 'occlusion_candidates']) {
  const result = dbh.getCollection(name).deleteMany({});
  print(name + ': ' + result.deletedCount);
}
"

  echo "Clearing Redis identity/cache state..."
  docker compose -f "$COMPOSE_FILE" exec -T redis redis-cli -n "$REDIS_DB" FLUSHDB >/dev/null

  echo "Clearing MinIO ReID snapshot objects..."
  if command -v mc >/dev/null 2>&1; then
    mc alias set local http://localhost:9002 minio minio123 >/dev/null
    mc rm --recursive --force local/reid-snapshots >/dev/null 2>&1 || true
  else
    docker compose -f "$COMPOSE_FILE" exec -T minio \
      mc alias set local http://localhost:9000 minio minio123 >/dev/null
    docker compose -f "$COMPOSE_FILE" exec -T minio \
      mc rm --recursive --force local/reid-snapshots >/dev/null 2>&1 || true
  fi

  echo "Resetting Qdrant identity collections..."
  for collection in "$QDRANT_COLLECTION" "$QDRANT_AUX_COLLECTION"; do
    curl --silent --show-error -X DELETE "http://localhost:16333/collections/$collection" >/dev/null || true
    curl --silent --show-error -X PUT "http://localhost:16333/collections/$collection" \
      -H "Content-Type: application/json" \
      -d "{\"vectors\":{\"size\":$QDRANT_VECTOR_SIZE,\"distance\":\"Cosine\"}}" >/dev/null
  done

  echo "Resetting Kafka replay topics..."
  for topic in "${KAFKA_TOPICS[@]}"; do
    docker compose -f "$COMPOSE_FILE" exec -T kafka /opt/kafka/bin/kafka-topics.sh \
      --bootstrap-server "$KAFKA_BOOTSTRAP" \
      --delete --if-exists --topic "$topic" >/dev/null 2>&1 || true
  done
  sleep 2
  for topic in "${KAFKA_TOPICS[@]}"; do
    docker compose -f "$COMPOSE_FILE" exec -T kafka /opt/kafka/bin/kafka-topics.sh \
      --bootstrap-server "$KAFKA_BOOTSTRAP" \
      --create --if-not-exists --topic "$topic" \
      --partitions 1 --replication-factor 1 >/dev/null
  done

  echo "Identity state has been reset."
}

cmd_up() {
  local build_flag="" reset="false"
  for arg in "$@"; do
    case "$arg" in
      --build) build_flag="--build" ;;
      --reset|--reset-identities) reset="true" ;;
      *) echo "Unknown flag for 'up': $arg" >&2; usage; exit 1 ;;
    esac
  done

  echo "Starting infrastructure if needed..."
  docker compose -f "$COMPOSE_FILE" up -d kafka mongo qdrant redis minio

  echo "Waiting for infrastructure..."
  wait_for_tcp_port "kafka" "localhost" "19092"

  if [ "$reset" = "true" ]; then
    echo "Resetting identity history for a clean run..."
    cmd_reset
  fi

  # Native MPS demo mode: the inference_engine runs natively on the host (Apple
  # Silicon GPU via PyTorch MPS — Docker on macOS has no Metal passthrough), so
  # we skip its container and point workers at it via host.docker.internal.
  # Start the engine first with: scripts/run_native_mps.sh
  #   MPS_NATIVE=true scripts/demo.sh up --build --reset
  local app_services=(inference_engine query_service streaming raw_stream worker_cam1 worker_cam2 gateway ui)
  local extra_up=""
  if [ "${MPS_NATIVE:-false}" = "true" ]; then
    echo "MPS_NATIVE=true → skipping inference_engine container; workers use host-native MPS engine."
    export WORKER_MODEL_SERVICE_URL="http://host.docker.internal:8000"
    app_services=(query_service streaming raw_stream worker_cam1 worker_cam2 gateway ui)
    # --no-deps so workers' depends_on does not pull the inference_engine
    # container back up. Infra was already started above.
    extra_up="--no-deps"
  fi

  if [ -n "$build_flag" ]; then
    echo "Rebuilding app services..."
    if [ "$reset" = "true" ]; then
      docker compose -f "$COMPOSE_FILE" up -d $extra_up --build --force-recreate "${app_services[@]}"
    else
      docker compose -f "$COMPOSE_FILE" up -d $extra_up --build "${app_services[@]}"
    fi
  else
    echo "Starting app services..."
    if [ "$reset" = "true" ]; then
      docker compose -f "$COMPOSE_FILE" up -d $extra_up --force-recreate "${app_services[@]}"
    else
      docker compose -f "$COMPOSE_FILE" up -d $extra_up "${app_services[@]}"
    fi
  fi

  echo "Waiting for services..."
  if [ "${MPS_NATIVE:-false}" = "true" ]; then
    wait_for_http "inference_engine (native MPS)" "http://localhost:8000/healthz" 30 || {
      echo "Native MPS engine not reachable on :8000. Start it first: scripts/run_native_mps.sh" >&2
    }
  else
    wait_for_http_optional "inference_engine" "http://localhost:8001/healthz" 30 || true
  fi
  wait_for_http "query_service" "http://localhost:18090/readyz"
  wait_for_http "streaming"     "http://localhost:8765/readyz"
  wait_for_http "gateway"       "http://localhost:18080/readyz"
  wait_for_http "ui"            "http://localhost:3000"

  echo "Replaying edge streams (${EDGE_SERVICES[*]})..."
  docker compose -f "$COMPOSE_FILE" up -d $build_flag --force-recreate "${EDGE_SERVICES[@]}" >/dev/null

  cat <<EOF

Demo stack is ready.

Viewer:
  $VIEWER_FILE
  or:  cd $ROOT_DIR/demo && python3 -m http.server 8088
       then open http://localhost:8088/stream-view.html

Useful commands:
  scripts/demo.sh up                 # restart + replay (no build, no reset)
  scripts/demo.sh up --build         # rebuild app images first
  scripts/demo.sh up --reset         # clear identity stores first
  scripts/demo.sh replay             # just restart edge to replay
  scripts/demo.sh diag               # mongo histogram + person counts
  scripts/demo.sh down               # tear it all down
EOF
}

cmd_replay() {
  echo "Restarting edges for replay (${EDGE_SERVICES[*]})..."
  docker compose -f "$COMPOSE_FILE" restart "${EDGE_SERVICES[@]}"
}

cmd_diag() {
  docker compose -f "$COMPOSE_FILE" exec -T mongo mongosh --quiet --eval "
const db = db.getSiblingDB('$MONGO_DB');
print('=== persons.count: ' + db.persons.countDocuments({}) + ' ===');
print('\n--- tracklets.matching histogram ---');
db.tracklets.aggregate([
  { \$group: { _id: { method: '\$matching.method', source: '\$matching.source' }, count: { \$sum: 1 } } },
  { \$sort: { count: -1 } }
]).forEach(d => printjson(d));
print('\n--- occlusion_candidates.matching.source histogram ---');
db.occlusion_candidates.aggregate([
  { \$group: { _id: '\$matching.source', count: { \$sum: 1 } } },
  { \$sort: { count: -1 } }
]).forEach(d => printjson(d));
print('\n--- tracklets per person ---');
db.tracklets.aggregate([
  { \$group: { _id: '\$person_id', n: { \$sum: 1 } } },
  { \$sort: { _id: 1 } }
]).forEach(d => printjson(d));
print('\n--- sightings per device (multi-camera) ---');
db.sightings.aggregate([
  { \$group: { _id: '\$device_id', sightings: { \$sum: 1 }, persons: { \$addToSet: '\$person_id' } } },
  { \$project: { sightings: 1, distinct_persons: { \$size: '\$persons' } } },
  { \$sort: { _id: 1 } }
]).forEach(d => printjson(d));
print('\n--- CROSS-CAMERA persons (same person_id seen on >=2 devices) ---');
let xcam = 0;
db.sightings.aggregate([
  { \$group: { _id: '\$person_id', devices: { \$addToSet: '\$device_id' } } },
  { \$project: { devices: 1, n_devices: { \$size: '\$devices' } } },
  { \$match: { n_devices: { \$gte: 2 } } },
  { \$sort: { _id: 1 } }
]).forEach(d => { xcam++; printjson(d); });
print('cross_camera_person_count: ' + xcam);
"
}

cmd_down() {
  echo "Tearing down stack (including volumes)..."
  docker compose -f "$COMPOSE_FILE" down -v
}

usage() {
  cat <<EOF
Usage:
  scripts/demo.sh up [--build] [--reset]
  scripts/demo.sh replay
  scripts/demo.sh reset
  scripts/demo.sh diag
  scripts/demo.sh down
EOF
}

# ── dispatch ───────────────────────────────────────────────────────────────

cmd="${1:-}"
shift || true
case "$cmd" in
  up)     cmd_up "$@" ;;
  replay) cmd_replay ;;
  reset)  cmd_reset ;;
  diag)   cmd_diag ;;
  down)   cmd_down ;;
  ""|-h|--help) usage ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
