#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZED_HOME=/home/qingchan
readonly LLAVA_SOURCE_RUN_ID=20260902T191411Z-ffd523c464c8-f99e2f39
mode=${1:?usage: run_qwen_effusion_vislast_gate.sh hook|full [HOOK_RECEIPT]}
case "$mode" in hook|full) ;; *) echo "Qwen gate refused: mode must be hook or full" >&2; exit 64;; esac
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
derived_run_dir=$(realpath -- "$source_dir/..")
run_dir=$(realpath -- "${RUN_DIR:?Qwen gate refused: trusted RUN_DIR is missing}")
case "$run_dir" in
  "$AUTHORIZED_HOME"/data/concept-flow/runs/*) ;;
  *) echo "Qwen gate refused: run directory is outside registered runs" >&2; exit 64 ;;
esac
test "$run_dir" = "$derived_run_dir" || {
  echo "Qwen gate refused: RUN_DIR does not match archived source" >&2
  exit 64
}
grep -Fx 'GPU_COUNT=1' "$run_dir/metadata.env" >/dev/null || {
  echo "Qwen gate refused: immutable receipt must declare exactly one allocated GPU" >&2
  exit 64
}
gpu=$(sed -n 's/^GPU_IDS=//p' "$run_dir/metadata.env")
[[ "$gpu" =~ ^[0-9]+$ ]] || {
  echo "Qwen gate refused: immutable receipt must declare one concrete GPU ID" >&2
  exit 64
}

manifest="$AUTHORIZED_HOME/data/concept-flow/datasets/nih-chestxray14/manifest.csv"
dataset_receipt="$AUTHORIZED_HOME/data/concept-flow/datasets/nih-chestxray14/asset-receipt.json"
model_root="$AUTHORIZED_HOME/data/concept-flow/models/qwen7b-huggingface"
llava_source_run="$AUTHORIZED_HOME/data/concept-flow/runs/$LLAVA_SOURCE_RUN_ID"
artifacts="$run_dir/artifacts"
mkdir -p "$artifacts"
export HF_HOME="$model_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python "$source_dir/src/qwen_effusion_gate.py" verify-inputs \
  --manifest "$manifest" --dataset-receipt "$dataset_receipt" \
  --model-root "$model_root" --out "$artifacts/input-verification.json"
snapshot=$(python - "$artifacts/input-verification.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["snapshot"])
PY
)

if test "$mode" = hook; then
  python "$source_dir/src/qwen_effusion_gate.py" verify-hook \
    --model-root "$model_root" --input-verification "$artifacts/input-verification.json" \
    --gpu "$gpu" --out "$artifacts/hook-verification.json"
  exit 0
fi

hook_receipt=${2:?full mode requires a prior immutable Qwen hook-verification.json}
hook_receipt=$(realpath -- "$hook_receipt")
case "$hook_receipt" in
  "$AUTHORIZED_HOME"/data/concept-flow/runs/*/artifacts/hook-verification.json) ;;
  *) echo "Qwen gate refused: hook receipt is outside immutable runs" >&2; exit 64;;
esac
hook_run=${hook_receipt%/artifacts/hook-verification.json}
verify_receipt_file() {
  local relative=$1 lines count
  lines=$(awk -v path="$relative" '$2 == path {print}' "$hook_run/SHA256SUMS")
  count=$(grep -c . <<<"$lines" || true)
  test "$count" -eq 1 || {
    echo "Qwen gate refused: expected one checksum for $relative" >&2
    exit 64
  }
  printf '%s\n' "$lines" | (cd "$hook_run" && sha256sum -c - >/dev/null)
}
for relative in ./metadata.env ./command.sh ./command_exit_status ./exit_status ./artifacts/hook-verification.json; do
  verify_receipt_file "$relative"
done
python - "$hook_run/command.sh" <<'PY'
import shlex, sys
lines = open(sys.argv[1], encoding="utf-8").read().splitlines()
if len(lines) != 3 or lines[:2] != ["#!/usr/bin/env bash", "set -euo pipefail"]:
    raise SystemExit("Qwen gate refused: hook command envelope mismatch")
tokens = shlex.split(lines[2])
if tokens != ["exec", "bash", "scripts/server/run_qwen_effusion_vislast_gate.sh", "hook"]:
    raise SystemExit("Qwen gate refused: hook command mismatch")
PY
test "$(cat "$hook_run/command_exit_status")" -eq 0
test "$(cat "$hook_run/exit_status")" -eq 0
grep -Fx 'GPU_COUNT=1' "$hook_run/metadata.env" >/dev/null
hook_run_source=$(sed -n 's/^SOURCE_COMMIT=//p' "$hook_run/metadata.env")
hook_receipt_source=$(python - "$hook_receipt" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["source_commit"])
PY
)
test "$hook_run_source" = "$hook_receipt_source"
python "$source_dir/src/qwen_effusion_gate.py" validate-hook-receipt \
  --hook "$hook_receipt" --input-verification "$artifacts/input-verification.json"
cp -- "$hook_receipt" "$artifacts/hook-verification.json"

llava_reference="$artifacts/llava-reference.json"
acts="$artifacts/activations"
probe="$artifacts/probe"
intervention="$artifacts/intervention"
python "$source_dir/src/cross_cell_gate.py" validate-source \
  --source-run "$llava_source_run" --out "$llava_reference"
python "$source_dir/src/extract.py" \
  --manifest "$manifest" --arch qwen7b --gpus "$gpu" --out "$acts" \
  --batch-size 8 --loci vis.last --model-path "$snapshot" \
  --prompt 'Is there a pleural effusion in this chest radiograph? Answer yes or no.'
python "$source_dir/src/qwen_effusion_gate.py" probe \
  --acts "$acts" --manifest "$manifest" --llava-reference "$llava_reference" \
  --out "$probe" --n-boot 2000
python "$source_dir/src/intervene.py" \
  --acts "$acts" --manifest "$manifest" --arch qwen7b --concept Effusion \
  --loci vis.last --alphas=-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1 \
  --alpha-mode reltoken --control-alphas=-1,-0.5,-0.25,0.25,0.5,1 \
  --n-random 20 --n-eval 200 --eval-split test --batch-size 16 --gpu "$gpu" \
  --model-path "$snapshot" --directions "$probe/directions.npz" \
  --prompt 'Is there a pleural effusion in this chest radiograph? Answer yes or no.' \
  --out "$intervention"
python "$source_dir/src/qwen_effusion_gate.py" summarize-intervention \
  --csv "$intervention/intervention.csv" --meta "$intervention/meta.json" \
  --input-verification "$artifacts/input-verification.json" \
  --hook-verification "$artifacts/hook-verification.json" \
  --extract-meta "$acts/meta.json" \
  --probe "$probe/probe.json" --bootstrap "$probe/probe_scores_and_bootstrap.npz" \
  --directions "$probe/directions.npz" --done "$intervention/DONE" \
  --llava-reference "$llava_reference" \
  --out "$artifacts/intervention-summary.json"
