#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZED_HOME=/home/qingchan
readonly SOURCE_RUN_ID=20260903T000321Z-caaae3ef346d-qwen-full
readonly OWNERSHIP_RUN_ID=20260904T125710Z-a3bd883540eb-causal-ownership

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
derived_run_dir=$(realpath -- "$source_dir/..")
run_dir=$(realpath -- "${RUN_DIR:?Consolidation input-closure gate refused: RUN_DIR is missing}")
case "$run_dir" in
  "$AUTHORIZED_HOME"/data/concept-flow/runs/*) ;;
  *) echo "Consolidation input-closure gate refused: run path is outside registered runs" >&2; exit 64 ;;
esac
test "$run_dir" = "$derived_run_dir" || {
  echo "Consolidation input-closure gate refused: RUN_DIR does not match archived source" >&2
  exit 64
}
grep -Fx 'GPU_COUNT=1' "$run_dir/metadata.env" >/dev/null || {
  echo "Consolidation input-closure gate refused: immutable receipt must declare one GPU" >&2
  exit 64
}
gpu=$(sed -n 's/^GPU_IDS=//p' "$run_dir/metadata.env")
[[ "$gpu" =~ ^[0-9]+$ ]] || {
  echo "Consolidation input-closure gate refused: GPU identity is invalid" >&2
  exit 64
}

data_root="$AUTHORIZED_HOME/data/concept-flow"
source_run="$data_root/runs/$SOURCE_RUN_ID"
ownership_run="$data_root/runs/$OWNERSHIP_RUN_ID"
manifest="$data_root/datasets/nih-chestxray14/manifest.csv"
model_root="$data_root/models/qwen7b-huggingface"
artifacts="$run_dir/artifacts"
mkdir -p "$artifacts/probe" "$artifacts/intervention"
export HF_HOME="$model_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python "$source_dir/src/qwen_consolidation_input_closure.py" validate-source \
  --source-run "$source_run" \
  --ownership-run "$ownership_run" \
  --manifest "$manifest" \
  --out "$artifacts/source-reference.json" \
  --pairs-out "$artifacts/registered-pairs.json"

python "$source_dir/src/qwen_consolidation_input_closure.py" probe \
  --acts "$source_run/artifacts/activations" \
  --manifest "$manifest" \
  --out "$artifacts/probe"

python "$source_dir/src/qwen_consolidation_input_closure.py" evaluate \
  --acts "$source_run/artifacts/activations" \
  --manifest "$manifest" \
  --source-reference "$artifacts/source-reference.json" \
  --pairs "$artifacts/registered-pairs.json" \
  --directions "$artifacts/probe/directions.npz" \
  --gpu "$gpu" \
  --out "$artifacts/intervention"

python "$source_dir/src/qwen_consolidation_input_closure.py" summarize \
  --per-pair "$artifacts/intervention/per-pair.csv" \
  --meta "$artifacts/intervention/meta.json" \
  --done "$artifacts/intervention/DONE" \
  --probe "$artifacts/probe/probe.json" \
  --pairs "$artifacts/registered-pairs.json" \
  --bootstrap "$artifacts/input-closure-bootstrap.npz" \
  --out "$artifacts/input-closure-summary.json"
