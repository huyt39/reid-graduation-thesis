#!/usr/bin/env bash
# Build lmbn_ft_bundle.tgz for Colab LMBN multi-scale fine-tuning.
#
# Extract on Colab so these land at the CWD the script expects:
#   models/                 LMBN_n + deps (package; relative imports resolve here)
#   dataset/personN/*.jpg   labeled target-domain crops (12 IDs, multi-scale d00..d04)
#   lmbn_n_cuhk03_d.pth     init weights
#   lmbn_finetune_colab.py  the training script (paste into a Colab cell)
#
# Colab cell order:
#   !pip install -q torch torchvision pillow tqdm scikit-learn structlog   # NOTE: structlog (osnet.py imports it)
#   !tar xzf lmbn_ft_bundle.tgz
#   # then paste lmbn_finetune_colab.py into the next cell and run
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$ROOT/.lmbn_ft_stage"
LABEL_DIR="$ROOT/reid_label_crops/unlabeled_target_eval/label"
MODELS_SRC="$ROOT/src/inference_engine/src/models"

rm -rf "$STAGE"
mkdir -p "$STAGE/models" "$STAGE/dataset"

# Model package (LMBN_n + its relative-import deps). __init__.py makes it a package.
for f in __init__.py lightmbn_n.py attention.py bnneck.py gem_pooling.py osnet.py; do
  cp "$MODELS_SRC/$f" "$STAGE/models/$f"
done

# Init weights + training script.
cp "$ROOT/src/inference_engine/src/assets/models/lmbn/lmbn_n_cuhk03_d.pth" "$STAGE/lmbn_n_cuhk03_d.pth"
cp "$ROOT/scripts/lmbn_finetune_colab.py" "$STAGE/lmbn_finetune_colab.py"

# Labeled crops → dataset/personN/ (script reads DATA_ROOT=dataset).
cp -R "$LABEL_DIR/." "$STAGE/dataset/"

tar czf "$ROOT/lmbn_ft_bundle.tgz" -C "$STAGE" models dataset lmbn_n_cuhk03_d.pth lmbn_finetune_colab.py
rm -rf "$STAGE"

echo "Built $ROOT/lmbn_ft_bundle.tgz"
echo "  IDs: $(ls "$LABEL_DIR" | grep -c '^person')  crops: $(find "$LABEL_DIR" -type f | grep -icE '\.jpe?g$')"
echo "  size: $(du -h "$ROOT/lmbn_ft_bundle.tgz" | cut -f1)"
