#!/usr/bin/env bash
set -euo pipefail

if test "${LLAVA_TEXT_CALIBRATION_OUTER:-0}" != 1; then
  export LLAVA_TEXT_CALIBRATION_OUTER=1
  exec timeout --signal=TERM --kill-after=60 14m bash "$0" "$@"
fi

readonly data_root=/home/qingchan/data/concept-flow
readonly stop_sentinel="$data_root/STOP"
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
run_dir=$(realpath -- "${RUN_DIR:?trusted RUN_DIR is required}")
case "$run_dir" in
  "$data_root"/runs/*) ;;
  *) echo 'LLaVA text calibration must be under the project run store' >&2; exit 64 ;;
esac
test ! -e "$stop_sentinel"
test "$source_dir" = "$run_dir/source"
test "$run_dir" = "$(realpath -- "$source_dir/..")"
expected_sha=${1:?expected immutable source SHA is required}
[[ "$expected_sha" =~ ^[0-9a-f]{40}$ ]]
test "${SOURCE_COMMIT:?immutable source commit is required}" = "$expected_sha"
grep -Fx "SOURCE_COMMIT=$expected_sha" "$run_dir/metadata.env" >/dev/null
grep -Fx "SOURCE_PATH=$source_dir" "$run_dir/metadata.env" >/dev/null
grep -Fx "RUN_ID=${run_dir##*/}" "$run_dir/metadata.env" >/dev/null
grep -Fx 'GPU_COUNT=1' "$run_dir/metadata.env" >/dev/null
cd -- "$source_dir"
source scripts/server/activate_env.sh
export HF_HOME="$data_root/models/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
gpu=$(sed -n 's/^GPU_IDS=//p' "$run_dir/metadata.env")
[[ "$gpu" =~ ^[0-9]+$ ]]
used=$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits \
  | awk '{gsub(/[^0-9]/, "", $0); if ($0 != "") s += $0} END {print s+0}')
test "$used" -lt 500 || { echo "registered GPU uses ${used} MiB immediately before launch" >&2; exit 64; }
common=(--data-root "$data_root" --out "$run_dir/artifacts" --source-commit "$expected_sha")
python -B -m src.run_llava_text_semantic_calibration run "${common[@]}" --gpu "$gpu"
python -B -m src.run_llava_text_semantic_calibration replay "${common[@]}"
