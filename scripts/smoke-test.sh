#!/usr/bin/env bash
# scripts/smoke-test.sh — end-to-end health check for the ReID pipeline demo.
#
# Runs a fresh demo replay and verifies that:
#   1. Edge processes the bundled video to completion
#   2. The worker creates a plausible number of persons (5-9 for vid3.mp4)
#   3. Triton served > 50 OSNet inferences (proves the inference path works)
#   4. Streaming exposes a usable WebSocket
#
# Exit code 0 on PASS, 1 on FAIL. Prints a short summary regardless.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
COMPOSE_FILE="$ROOT_DIR/src/deploy/docker-compose.yml"

PASS=0
FAIL=0
SUMMARY=()

record_pass() { PASS=$((PASS + 1)); SUMMARY+=("  ✅  $1"); }
record_fail() { FAIL=$((FAIL + 1)); SUMMARY+=("  ❌  $1"); }

print_header() { echo; echo "── $1 ──"; }

print_header "Step 1: triggering fresh demo replay"
"$ROOT_DIR/scripts/demo-fast.sh" --reset-identities >/dev/null 2>&1 \
  && record_pass "demo-fast.sh --reset-identities completed" \
  || record_fail "demo-fast.sh exited non-zero"

print_header "Step 2: waiting for edge to finish replaying vid3.mp4"
attempts=0
while [ $attempts -lt 120 ]; do
  state=$(docker compose -f "$COMPOSE_FILE" ps -a --format json edge_cam1 2>/dev/null | grep -o '"State":"[^"]*"' | head -1)
  if [[ "$state" == *'"exited"'* ]]; then
    break
  fi
  sleep 5
  attempts=$((attempts + 1))
done
if [[ "$state" == *'"exited"'* ]]; then
  record_pass "edge container exited cleanly"
else
  record_fail "edge did not exit within 10 minutes (state=$state)"
fi

print_header "Step 3: querying Mongo for person count"
PERSONS=$(docker compose -f "$COMPOSE_FILE" exec -T mongo \
  mongosh --quiet --eval 'print(db.getSiblingDB("reid_production").persons.countDocuments({}))' 2>/dev/null | tail -1 | tr -d '[:space:]')
if [[ "$PERSONS" =~ ^[0-9]+$ ]] && [ "$PERSONS" -ge 5 ] && [ "$PERSONS" -le 9 ]; then
  record_pass "persons=$PERSONS (within tuned band 5..9 for vid3.mp4)"
else
  record_fail "persons=$PERSONS (expected 5..9 for vid3.mp4; tune .env if drift)"
fi

print_header "Step 4: checking Triton inference count"
OSNET_COUNT=$(curl -s http://localhost:8013/metrics 2>/dev/null \
  | awk -F'[ {}]' '/^nv_inference_count\{model="osnet"/ {print $NF}' | head -1)
if [[ "$OSNET_COUNT" =~ ^[0-9]+$ ]] && [ "$OSNET_COUNT" -gt 50 ]; then
  record_pass "Triton served $OSNET_COUNT OSNet inferences"
else
  record_fail "Triton OSNet inference count=$OSNET_COUNT (expected > 50)"
fi

print_header "Step 5: checking Triton batching efficiency"
OSNET_EXEC=$(curl -s http://localhost:8013/metrics 2>/dev/null \
  | awk -F'[ {}]' '/^nv_inference_exec_count\{model="osnet"/ {print $NF}' | head -1)
if [[ "$OSNET_EXEC" =~ ^[0-9]+$ ]] && [ "$OSNET_EXEC" -gt 0 ]; then
  BATCH_RATIO=$(awk "BEGIN { printf \"%.2f\", $OSNET_COUNT / $OSNET_EXEC }")
  AT_LEAST_TWO=$(awk "BEGIN { print ($OSNET_COUNT / $OSNET_EXEC >= 2.0) ? 1 : 0 }")
  if [ "$AT_LEAST_TWO" -eq 1 ]; then
    record_pass "Triton avg batch size = ${BATCH_RATIO}x (>= 2x; dynamic batching working)"
  else
    record_fail "Triton avg batch size = ${BATCH_RATIO}x (< 2x; not batching effectively)"
  fi
else
  record_fail "Triton exec count missing or zero"
fi

print_header "Step 6: streaming readyz check"
if curl -fsS http://localhost:8765/readyz >/dev/null 2>&1; then
  record_pass "streaming /readyz returned 200"
else
  record_fail "streaming /readyz failed"
fi

echo
echo "═══════════ Smoke-test summary ═══════════"
for line in "${SUMMARY[@]}"; do
  echo "$line"
done
echo
echo "Pass: $PASS    Fail: $FAIL"
echo

if [ "$FAIL" -eq 0 ]; then
  exit 0
fi
exit 1
