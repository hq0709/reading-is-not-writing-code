#!/usr/bin/env bash
set -euo pipefail

readonly data_root=/home/qingchan/data/concept-flow
readonly ownership_run="$data_root/runs/20260904T125710Z-a3bd883540eb-causal-ownership"
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
run_dir=$(realpath -- "${RUN_DIR:?trusted RUN_DIR is required}")
case "$run_dir" in
  "$data_root"/runs/*) ;;
  *) echo 'answer-encoding run must be under the project run store' >&2; exit 64 ;;
esac
test "$run_dir" = "$(realpath -- "$source_dir/..")"
grep -Fx 'GPU_COUNT=1' "$run_dir/metadata.env" >/dev/null
gpu=$(sed -n 's/^GPU_IDS=//p' "$run_dir/metadata.env")
[[ "$gpu" =~ ^[0-9]+$ ]]
export HF_HOME="$data_root/models/qwen7b-huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
args=()
case "${1:-full}" in
  full) ;;
  preflight) args+=(--preflight-only) ;;
  *) echo 'usage: run_qwen_answer_encoding_gate.sh [full|preflight]' >&2; exit 64 ;;
esac
python "$source_dir/src/qwen_answer_encoding.py" run \
  --ownership-run "$ownership_run" \
  --manifest "$data_root/datasets/nih-chestxray14/manifest.csv" \
  --gpu "$gpu" --out "$run_dir/artifacts" "${args[@]}"
if test "${1:-full}" = full; then
  python "$source_dir/src/qwen_answer_encoding.py" summarize \
    --per-image "$run_dir/artifacts/per-image.csv" \
    --out "$run_dir/artifacts/answer-encoding-summary.json"
fi
