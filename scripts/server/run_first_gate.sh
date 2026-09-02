#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZED_HOME=/home/qingchan
mode=${1:?usage: run_first_gate.sh hook|full GPU [HOOK_RECEIPT]}
gpu=${2:?usage: run_first_gate.sh hook|full GPU [HOOK_RECEIPT]}
run_dir=${RUN_DIR:?first gate refused: trusted RUN_DIR is missing}
case "$(realpath -m -- "$run_dir")/" in
  "$AUTHORIZED_HOME"/*) ;;
  *) echo "first gate refused: run directory escapes authorized home" >&2; exit 64 ;;
esac
case "$mode" in hook|full) ;; *) echo "first gate refused: mode must be hook or full" >&2; exit 64;; esac
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "first gate refused: GPU must be numeric" >&2; exit 64; }
grep -Fx 'GPU_COUNT=1' "$run_dir/metadata.env" >/dev/null || {
  echo "first gate refused: immutable receipt must declare exactly one allocated GPU" >&2
  exit 64
}

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
manifest="$AUTHORIZED_HOME/data/concept-flow/datasets/nih-chestxray14/manifest.csv"
dataset_receipt="$AUTHORIZED_HOME/data/concept-flow/datasets/nih-chestxray14/asset-receipt.json"
model_root="$AUTHORIZED_HOME/data/concept-flow/models/huggingface"
artifacts="$run_dir/artifacts"
mkdir -p "$artifacts"
export HF_HOME="$model_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python "$source_dir/src/first_gate.py" verify-inputs \
  --manifest "$manifest" --dataset-receipt "$dataset_receipt" \
  --model-root "$model_root" --out "$artifacts/input-verification.json"
snapshot=$(python - "$artifacts/input-verification.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["snapshot"])
PY
)

if test "$mode" = hook; then
  python "$source_dir/src/first_gate.py" verify-hook \
    --model-root "$model_root" --input-verification "$artifacts/input-verification.json" \
    --gpu "$gpu" --out "$artifacts/hook-verification.json"
  exit 0
fi

hook_receipt=${3:?full mode requires a prior immutable hook-verification.json}
hook_receipt=$(realpath -- "$hook_receipt")
case "$hook_receipt" in
  "$AUTHORIZED_HOME"/data/concept-flow/runs/*/artifacts/hook-verification.json) ;;
  *) echo "first gate refused: hook receipt is outside immutable runs" >&2; exit 64;;
esac
hook_run=${hook_receipt%/artifacts/hook-verification.json}
test "$(cat "$hook_run/command_exit_status")" -eq 0
test "$(cat "$hook_run/exit_status")" -eq 0
grep -Fx 'GPU_COUNT=1' "$hook_run/metadata.env" >/dev/null
hook_line=$(grep -F '  ./artifacts/hook-verification.json' "$hook_run/SHA256SUMS")
test -n "$hook_line"
printf '%s\n' "$hook_line" | (cd "$hook_run" && sha256sum -c -)
hook_run_source=$(sed -n 's/^SOURCE_COMMIT=//p' "$hook_run/metadata.env")
hook_receipt_source=$(python - "$hook_receipt" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["source_commit"])
PY
)
test "$hook_run_source" = "$hook_receipt_source"
python "$source_dir/src/first_gate.py" validate-hook-receipt \
  --hook "$hook_receipt" --input-verification "$artifacts/input-verification.json"
cp -- "$hook_receipt" "$artifacts/hook-verification.json"

acts="$artifacts/activations"
probe="$artifacts/probe"
intervention="$artifacts/intervention"
python "$source_dir/src/extract.py" \
  --manifest "$manifest" --arch llava15_7b --gpus "$gpu" --out "$acts" \
  --batch-size 8 --loci vis.last --model-path "$snapshot" \
  --prompt 'Is there a pleural effusion in this chest radiograph? Answer yes or no.'
python "$source_dir/src/first_gate.py" probe \
  --acts "$acts" --manifest "$manifest" --out "$probe" --n-boot 2000
python "$source_dir/src/intervene.py" \
  --acts "$acts" --manifest "$manifest" --arch llava15_7b --concept Effusion \
  --loci vis.last --alphas=-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1 \
  --alpha-mode reltoken --control-alphas=-1,-0.5,-0.25,0.25,0.5,1 \
  --n-random 20 --n-eval 200 --eval-split test --batch-size 16 --gpu "$gpu" \
  --model-path "$snapshot" --directions "$probe/directions.npz" \
  --prompt 'Is there a pleural effusion in this chest radiograph? Answer yes or no.' \
  --out "$intervention"
python "$source_dir/src/first_gate.py" summarize-intervention \
  --csv "$intervention/intervention.csv" --meta "$intervention/meta.json" \
  --input-verification "$artifacts/input-verification.json" \
  --hook-verification "$artifacts/hook-verification.json" \
  --extract-meta "$acts/meta.json" \
  --probe "$probe/probe.json" --directions "$probe/directions.npz" \
  --done "$intervention/DONE" \
  --out "$artifacts/intervention-summary.json"
