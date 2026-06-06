#!/usr/bin/env bash
# Tạo service mới từ template: worker (reid_worker) hoặc api (query_service).
# Usage:
#   ./scripts/new_service.sh <tên_snake_case> <worker|api> [--port CỔNG]
#
# Ví dụ:
#   ./scripts/new_service.sh analytics_worker worker
#   ./scripts/new_service.sh analytics_api api --port 8100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_ROOT="$REPO_ROOT/src"

usage() {
  sed -n '1,12p' "$0" | tail -n +2
  exit "${1:-0}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage 0
fi

NAME="${1:-}"
TYPE="${2:-}"
PORT=""

if [[ -z "$NAME" || -z "$TYPE" ]]; then
  usage 1
fi

shift 2 || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="${2:?--port cần một số}"
      shift 2
      ;;
    *)
      echo "Tham số không hợp lệ: $1" >&2
      usage 1
      ;;
  esac
done

if ! [[ "$NAME" =~ ^[a-z][a-z0-9_]*$ ]]; then
  echo "Tên service không hợp lệ: '$NAME' (chỉ chữ thường, số, gạch dưới; bắt đầu bằng chữ)" >&2
  exit 1
fi

case "$TYPE" in
  worker|api) ;;
  *)
    echo "Loại phải là worker hoặc api, nhận được: $TYPE" >&2
    exit 1
    ;;
esac

if [[ "$TYPE" == "api" && -z "$PORT" ]]; then
  PORT=8091
fi
if [[ "$TYPE" == "worker" && -n "$PORT" ]]; then
  echo "--port chỉ dùng với loại api" >&2
  exit 1
fi

DEST="$SRC_ROOT/$NAME"
if [[ -e "$DEST" ]]; then
  echo "Đã tồn tại: $DEST — xóa hoặc đổi tên trước khi chạy lại." >&2
  exit 1
fi

TEMPLATE=""
case "$TYPE" in
  worker) TEMPLATE="$SRC_ROOT/reid_worker" ;;
  api) TEMPLATE="$SRC_ROOT/query_service" ;;
esac

if [[ ! -d "$TEMPLATE" ]]; then
  echo "Không tìm thấy template: $TEMPLATE" >&2
  exit 1
fi

echo "Copy template $TYPE từ $(basename "$TEMPLATE") -> $NAME"
cp -R "$TEMPLATE" "$DEST"
rm -f "$DEST/uv.lock"

OLD_PY_NAME=""
OLD_SERVICE_TITLE=""
case "$TYPE" in
  worker)
    OLD_PY_NAME="reid_worker"
    OLD_SERVICE_TITLE="reid_worker"
    ;;
  api)
    OLD_PY_NAME="query_service"
    OLD_SERVICE_TITLE="query_service"
    ;;
esac

replace_in_file() {
  local file="$1"
  local from="$2"
  local to="$3"
  if [[ ! -f "$file" ]]; then
    return 0
  fi
  # macOS và GNU sed
  if sed --version >/dev/null 2>&1; then
    sed -i "s/${from}/${to}/g" "$file"
  else
    sed -i '' "s/${from}/${to}/g" "$file"
  fi
}

# pyproject.toml + README
replace_in_file "$DEST/pyproject.toml" "$OLD_PY_NAME" "$NAME"
replace_in_file "$DEST/pyproject.toml" "Worker: ${OLD_PY_NAME}" "Worker: ${NAME}"
replace_in_file "$DEST/pyproject.toml" "Service: ${OLD_PY_NAME}" "Service: ${NAME}"
replace_in_file "$DEST/README.md" "$OLD_PY_NAME" "$NAME"

# .env.example
replace_in_file "$DEST/.env.example" "SERVICE_NAME=${OLD_PY_NAME}" "SERVICE_NAME=${NAME}"

if [[ "$TYPE" == "api" ]]; then
  replace_in_file "$DEST/.env.example" "SERVICE_PORT=8090" "SERVICE_PORT=${PORT}"
fi

# core/config.py
replace_in_file "$DEST/src/core/config.py" "\"${OLD_PY_NAME}\"" "\"${NAME}\""
if [[ "$TYPE" == "worker" ]]; then
  replace_in_file "$DEST/src/core/config.py" 'service_name: str = "worker"' "service_name: str = \"${NAME}\""
fi
if [[ "$TYPE" == "api" ]]; then
  replace_in_file "$DEST/src/core/config.py" "service_port: int = 8090" "service_port: int = ${PORT}"
fi

# Dockerfile: EXPOSE cho api
if [[ "$TYPE" == "api" ]]; then
  replace_in_file "$DEST/Dockerfile" "EXPOSE 8000" "EXPOSE ${PORT}"
fi

python3 <<PY
from __future__ import annotations

import os
from pathlib import Path

repo = Path(r"$REPO_ROOT")
name = r"$NAME"
rel = f"src/{name}"
port = int(r"${PORT:-0}" or 0)
svc_type = r"$TYPE"

def add_for_loop(path: Path, needle: str) -> None:
    text = path.read_text(encoding="utf-8")
    if needle in text:
        return
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        if "for d in" in line and "src/edge" in line and "; do" in line and needle not in line:
            line = line.replace("; do", f" {needle}; do", 1)
        out.append(line)
    path.write_text("".join(out), encoding="utf-8")

add_for_loop(repo / "Makefile", rel)
add_for_loop(repo / "scripts" / "init_uv_locks.sh", rel)

compose = repo / "src" / "deploy" / "docker-compose.yml"
ct = compose.read_text(encoding="utf-8")
block_key = f"  {name}:"
if block_key in ct:
    print(f"docker-compose: đã có khối {name}, bỏ qua.")
else:
    if svc_type == "worker":
        block = f"""  {name}:
    build: ../{name}
    env_file: ../{name}/.env.example
    depends_on: [kafka, mongo, qdrant, redis]

"""
    else:
        block = f"""  {name}:
    build: ../{name}
    env_file: ../{name}/.env.example
    depends_on: [mongo, qdrant, redis]
    ports: ["{port}:{port}"]

"""
    if not ct.endswith("\n"):
        ct += "\n"
    compose.write_text(ct + "\n" + block, encoding="utf-8")
    print(f"Đã thêm service vào {compose}")

PY

echo ""
echo "Hoàn tất. Thư mục: $DEST"
echo "Bước tiếp theo (local):"
echo "  cd $DEST && uv lock && uv sync"
echo "Chạy worker:  uv run python -m src"
echo "Chạy api:     uv run python -m src   (port trong .env hoặc SERVICE_PORT)"
echo "Docker stack: make -C $REPO_ROOT up   (hoặc docker compose -f $REPO_ROOT/src/deploy/docker-compose.yml up -d --build)"
