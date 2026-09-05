#!/usr/bin/env bash
set -euo pipefail

readonly data_root=/home/qingchan/data/concept-flow
readonly ownership_run="$data_root/runs/20260904T125710Z-a3bd883540eb-causal-ownership"
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
run_dir=$(realpath -- "${RUN_DIR:?trusted RUN_DIR is required}")
case "$run_dir" in
  "$data_root"/runs/*) ;;
  *) echo 'Paired opportunity run must be under the project run store' >&2; exit 64 ;;
esac
test "$run_dir" = "$(realpath -- "$source_dir/..")"
export HF_HOME="$data_root/models/qwen7b-huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd -- "$source_dir"
args=()
command=run
case "${1:-full}" in
  full) ;;
  preflight) args+=(--preflight-only) ;;
  cohort) command=cohort ;;
  summarize)
    exec python -B -m src.run_qwen_paired_opportunity summarize \
      --per-image "$run_dir/artifacts/per-image.csv" \
      --out "$run_dir/artifacts/paired-opportunity-summary.json" ;;
  *) echo 'usage: run_qwen_paired_opportunity_gate.sh [full|preflight|cohort|summarize] [Mass|Consolidation]' >&2; exit 64 ;;
esac
case "${2:?accepted selected concept is required}" in
  Mass|Consolidation) args+=(--concept "$2") ;;
  *) echo 'selected concept must be Mass or Consolidation' >&2; exit 64 ;;
esac
if test "$command" = run; then
  grep -Fx 'GPU_COUNT=1' "$run_dir/metadata.env" >/dev/null
  gpu=$(sed -n 's/^GPU_IDS=//p' "$run_dir/metadata.env")
  [[ "$gpu" =~ ^[0-9]+$ ]]
  args+=(--gpu "$gpu" --ownership-run "$ownership_run")
fi
exec timeout --signal=TERM --kill-after=10 900 \
  python -B -m src.run_qwen_paired_opportunity "$command" \
    --dataset-root "$data_root/datasets/nih-chestxray14" \
    --runs-root "$data_root/runs" --out "$run_dir/artifacts" \
    --source-commit "${SOURCE_COMMIT:?immutable source commit is required}" "${args[@]}"
