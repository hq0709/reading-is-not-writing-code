#!/usr/bin/env bash
set -euo pipefail

readonly data_root=/home/qingchan/data/concept-flow
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
run_dir=$(realpath -- "${RUN_DIR:?trusted RUN_DIR is required}")
case "$run_dir" in
  "$data_root"/runs/*) ;;
  *) echo 'LLaVA opportunity run must be under the project run store' >&2; exit 64 ;;
esac
test "$source_dir" = "$run_dir/source"
test "$run_dir" = "$(realpath -- "$source_dir/..")"
expected_sha=${2:?expected immutable source SHA is required}
[[ "$expected_sha" =~ ^[0-9a-f]{40}$ ]]
test "${SOURCE_COMMIT:?immutable source commit is required}" = "$expected_sha"
grep -Fx "SOURCE_COMMIT=$expected_sha" "$run_dir/metadata.env" >/dev/null
grep -Fx "SOURCE_PATH=$source_dir" "$run_dir/metadata.env" >/dev/null
grep -Fx "RUN_ID=${run_dir##*/}" "$run_dir/metadata.env" >/dev/null
cd -- "$source_dir"
source scripts/server/activate_env.sh
export HF_HOME="$data_root/models/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
args=()
case "${1:-full}" in
  full) ;;
  preflight) args+=(--preflight-only) ;;
  summarize)
    exec timeout --signal=TERM --kill-after=10 900 \
      python -B -m src.run_llava_paired_opportunity summarize \
        --per-image "$run_dir/artifacts/per-image.csv" \
        --out "$run_dir/artifacts/paired-opportunity-summary.json" ;;
  *) echo 'usage: run_llava_paired_opportunity_gate.sh [full|preflight|summarize] EXPECTED_SHA' >&2; exit 64 ;;
esac
grep -Fx 'GPU_COUNT=1' "$run_dir/metadata.env" >/dev/null
gpu=$(sed -n 's/^GPU_IDS=//p' "$run_dir/metadata.env")
[[ "$gpu" =~ ^[0-9]+$ ]]
exec timeout --signal=TERM --kill-after=10 900 \
  python -B -m src.run_llava_paired_opportunity run \
    --dataset-root "$data_root/datasets/nih-chestxray14" \
    --runs-root "$data_root/runs" --model-root "$HF_HOME" \
    --out "$run_dir/artifacts" --source-commit "$expected_sha" --gpu "$gpu" "${args[@]}"
