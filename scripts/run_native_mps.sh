#!/usr/bin/env bash
#
# Run the inference_engine natively on the macOS host so it can use the Apple
# Silicon GPU via PyTorch MPS (~24x faster than CPU on osnet_ain). Docker on
# macOS has no Metal passthrough, so the engine must run OUTSIDE the container;
# the rest of the stack still runs in Docker and reaches this engine over
# host.docker.internal (see `MPS_NATIVE=true scripts/demo.sh up`).
#
# IMPORTANT: MPS is for live/demo only. It is NOT bit-identical with CPU, so
# evaluation / A-B runs must keep using the in-container CPU engine.
#
# Usage:
#   scripts/run_native_mps.sh            # set up venv (first run) + serve on :8000
#   ENGINE_PORT=8000 scripts/run_native_mps.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="$ROOT_DIR/src/inference_engine"
VENV_DIR="$ENGINE_DIR/.venv-mps"
ENGINE_PORT="${ENGINE_PORT:-8000}"

cd "$ENGINE_DIR"

# 1. venv with MPS-capable torch (default PyPI wheel; pyproject's [tool.uv.sources]
#    CPU pin is uv-only and ignored by pip, so pip gets the macOS MPS wheel).
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating MPS venv at $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip >/dev/null
  echo "Installing inference_engine deps (torch from default index → MPS) ..."
  pip install -e .
else
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

# 2. Fail fast if MPS is not actually available in this env.
python -c "
import sys, torch
ok = torch.backends.mps.is_available() and torch.backends.mps.is_built()
print(f'torch {torch.__version__}  mps_available={torch.backends.mps.is_available()}  mps_built={torch.backends.mps.is_built()}')
sys.exit(0 if ok else 1)
" || { echo 'ERROR: MPS not available in this venv.' >&2; exit 1; }

# 3. Point weight paths at host-local files (the in-container .env uses /app/...
#    paths that do not exist on the host). Real env vars take precedence over
#    the .env file in pydantic-settings, so these override it. Paths are
#    relative to ENGINE_DIR (== cwd).
export INFERENCE_DEVICE=mps
export INFERENCE_OSNET_WEIGHTS="src/assets/models/osnet/model.pth.tar-150"
export INFERENCE_OSNET_AIN_WEIGHTS="src/assets/models/osnet_ain/osnet_ain_msmt17.pth"
export INFERENCE_LMBN_WEIGHTS="src/assets/models/lmbn/lmbn_n_finetuned.pth"
export INFERENCE_EFFICIENTNET_WEIGHTS=""
export INFERENCE_MULTI_ATTR_WEIGHTS="src/assets/models/multi_attr/best_model_multi_attr_b0.pth"
export INFERENCE_STANDALONE_GENDER_WEIGHTS="src/assets/models/gender/gender_model.pth"
export SERVICE_PORT="$ENGINE_PORT"

echo "Starting native MPS inference_engine on http://localhost:$ENGINE_PORT (device=mps)"
echo "Check: curl -s http://localhost:$ENGINE_PORT/readyz   →   \"device\":\"mps\""
exec python -m src
