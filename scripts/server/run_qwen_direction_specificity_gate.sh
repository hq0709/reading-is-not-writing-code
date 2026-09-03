#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZED_HOME=/home/qingchan
readonly SOURCE_RUN_ID=20260903T000321Z-caaae3ef346d-qwen-full
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
derived_run_dir=$(realpath -- "$source_dir/..")
run_dir=$(realpath -- "${RUN_DIR:?direction-specificity gate refused: trusted RUN_DIR is missing}")
case "$run_dir" in
  "$AUTHORIZED_HOME"/data/concept-flow/runs/*) ;;
  *) echo "direction-specificity gate refused: run directory is outside registered runs" >&2; exit 64 ;;
esac
test "$run_dir" = "$derived_run_dir" || {
  echo "direction-specificity gate refused: RUN_DIR does not match archived source" >&2
  exit 64
}
grep -Fx 'GPU_COUNT=1' "$run_dir/metadata.env" >/dev/null || {
  echo "direction-specificity gate refused: immutable receipt must declare one GPU" >&2
  exit 64
}
gpu=$(sed -n 's/^GPU_IDS=//p' "$run_dir/metadata.env")
[[ "$gpu" =~ ^[0-9]+$ ]] || {
  echo "direction-specificity gate refused: GPU identity is invalid" >&2
  exit 64
}

manifest="$AUTHORIZED_HOME/data/concept-flow/datasets/nih-chestxray14/manifest.csv"
source_run="$AUTHORIZED_HOME/data/concept-flow/runs/$SOURCE_RUN_ID"
model_root="$AUTHORIZED_HOME/data/concept-flow/models/qwen7b-huggingface"
artifacts="$run_dir/artifacts"
intervention="$artifacts/intervention"
mkdir -p "$artifacts"
export HF_HOME="$model_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python "$source_dir/src/qwen_direction_specificity_gate.py" validate-source \
  --source-run "$source_run" --manifest "$manifest" \
  --out "$artifacts/source-reference.json" \
  --rows-out "$artifacts/registered-rows.json"
snapshot=$(python - "$artifacts/source-reference.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["model_source"])
PY
)

python "$source_dir/src/intervene.py" \
  --acts "$source_run/artifacts/activations" --manifest "$manifest" \
  --arch qwen7b --concept Effusion --loci vis.last \
  --alphas=0,0.25 --alpha-mode reltoken --control-alphas=0.25 \
  --n-random 20 --n-eval 400 --eval-split test --batch-size 16 --gpu "$gpu" \
  --model-path "$snapshot" --directions "$source_run/artifacts/probe/directions.npz" \
  --eval-row-ids-file "$artifacts/registered-rows.json" \
  --per-image-out "$intervention/per-image.csv" \
  --prompt 'Is there a pleural effusion in this chest radiograph? Answer yes or no.' \
  --out "$intervention"

python "$source_dir/src/qwen_direction_specificity_gate.py" summarize \
  --per-image "$intervention/per-image.csv" --meta "$intervention/meta.json" \
  --done "$intervention/DONE" --source-reference "$artifacts/source-reference.json" \
  --registered-rows "$artifacts/registered-rows.json" \
  --bootstrap "$artifacts/direction-specificity-bootstrap.npz" \
  --out "$artifacts/direction-specificity-summary.json"
