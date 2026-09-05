#!/usr/bin/env bash
set -euo pipefail

readonly data_root=/home/qingchan/data/concept-flow
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
run_dir=$(realpath -- "${RUN_DIR:?trusted RUN_DIR is required}")
case "$run_dir" in
  "$data_root"/runs/*) ;;
  *) echo 'LLaVA validation run must be under the project run store' >&2; exit 64 ;;
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
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
common=(--data-root "$data_root" --out "$run_dir/artifacts" --source-commit "$expected_sha")
mode=${1:-full}
args=()
summarize() {
  timeout --signal=TERM --kill-after=10 5m \
    python -B -m src.run_llava_validation_opportunity summarize "${common[@]}"
}
case "$mode" in
  full) ;;
  preflight) args+=(--preflight-only) ;;
  summarize) summarize; exit ;;
  *) echo 'usage: run_llava_validation_opportunity_gate.sh [full|preflight|summarize] EXPECTED_SHA' >&2; exit 64 ;;
esac
grep -Fx 'GPU_COUNT=1' "$run_dir/metadata.env" >/dev/null
gpu=$(sed -n 's/^GPU_IDS=//p' "$run_dir/metadata.env")
[[ "$gpu" =~ ^[0-9]+$ ]]
timeout --signal=TERM --kill-after=10 30m \
  python -B -m src.run_llava_validation_opportunity prepare "${common[@]}"
timeout --signal=TERM --kill-after=60 90m \
  python -B -m src.run_llava_validation_opportunity run "${common[@]}" --gpu "$gpu" "${args[@]}"
if test "$mode" = full; then
  summarize
fi
