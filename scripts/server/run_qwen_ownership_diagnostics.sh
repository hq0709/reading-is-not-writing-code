#!/usr/bin/env bash
set -euo pipefail

readonly data_root=/home/qingchan/data/concept-flow
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
run_dir=$(realpath -- "${RUN_DIR:?trusted RUN_DIR is required}")
case "$run_dir" in
  "$data_root"/runs/*) ;;
  *) echo 'diagnostic run must be under the project run store' >&2; exit 64 ;;
esac
test "$run_dir" = "$(realpath -- "$source_dir/..")"
python "$source_dir/src/qwen_ownership_diagnostics.py" \
  --ownership-run "$data_root/runs/20260904T125710Z-a3bd883540eb-causal-ownership" \
  --manifest "$data_root/datasets/nih-chestxray14/manifest.csv" \
  --out "$run_dir/artifacts/ownership-diagnostics.json"
