#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZED_HOME=/home/qingchan
readonly SOURCE_RUN_ID=20260902T191411Z-ffd523c464c8-f99e2f39
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
derived_run_dir=$(realpath -- "$source_dir/..")
run_dir=$(realpath -- "${RUN_DIR:?cross-cell gate refused: trusted RUN_DIR is missing}")
case "$run_dir" in
  "$AUTHORIZED_HOME"/data/concept-flow/runs/*) ;;
  *) echo "cross-cell gate refused: run directory is outside registered runs" >&2; exit 64 ;;
esac
test "$run_dir" = "$derived_run_dir" || {
  echo "cross-cell gate refused: RUN_DIR does not match archived source" >&2
  exit 64
}
grep -Fx 'GPU_COUNT=1' "$run_dir/metadata.env" >/dev/null || {
  echo "cross-cell gate refused: immutable receipt must declare exactly one allocated GPU" >&2
  exit 64
}
gpu=$(sed -n 's/^GPU_IDS=//p' "$run_dir/metadata.env")
[[ "$gpu" =~ ^[0-9]+$ ]] || {
  echo "cross-cell gate refused: immutable receipt must declare one concrete GPU ID" >&2
  exit 64
}

manifest="$AUTHORIZED_HOME/data/concept-flow/datasets/nih-chestxray14/manifest.csv"
model_root="$AUTHORIZED_HOME/data/concept-flow/models/huggingface"
source_run="$AUTHORIZED_HOME/data/concept-flow/runs/$SOURCE_RUN_ID"
source_acts="$source_run/artifacts/activations"
source_input="$source_run/artifacts/input-verification.json"
source_hook="$source_run/artifacts/hook-verification.json"
artifacts="$run_dir/artifacts"
reuse="$artifacts/activation-reuse.json"
probe="$artifacts/probe"
intervention="$artifacts/intervention"
mkdir -p "$artifacts"
export HF_HOME="$model_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python "$source_dir/src/cross_cell_gate.py" validate-source \
  --source-run "$source_run" --out "$reuse"
python "$source_dir/src/first_gate.py" validate-hook-receipt \
  --hook "$source_hook" --input-verification "$source_input"
snapshot=$(python - "$source_input" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["snapshot"])
PY
)

python "$source_dir/src/cross_cell_gate.py" probe \
  --acts "$source_acts" --manifest "$manifest" --reuse-receipt "$reuse" \
  --out "$probe" --n-boot 2000
python "$source_dir/src/intervene.py" \
  --acts "$source_acts" --manifest "$manifest" --arch llava15_7b --concept Edema \
  --loci vis.last --alphas=-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1 \
  --alpha-mode reltoken --control-alphas=-1,-0.5,-0.25,0.25,0.5,1 \
  --n-random 20 --n-eval 200 --eval-split test --batch-size 16 --gpu "$gpu" \
  --model-path "$snapshot" --directions "$probe/directions.npz" \
  --prompt 'Is there pulmonary edema in this chest radiograph? Answer yes or no.' \
  --out "$intervention"
python "$source_dir/src/cross_cell_gate.py" summarize-intervention \
  --csv "$intervention/intervention.csv" --meta "$intervention/meta.json" \
  --reuse-receipt "$reuse" --probe "$probe/probe.json" \
  --bootstrap "$probe/probe_scores_and_bootstrap.npz" \
  --directions "$probe/directions.npz" --done "$intervention/DONE" \
  --out "$artifacts/intervention-summary.json"
