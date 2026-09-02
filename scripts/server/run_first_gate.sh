#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZED_HOME=/home/qingchan
mode=${1:?usage: run_first_gate.sh hook|full RUN_DIR GPU}
run_dir=${2:?usage: run_first_gate.sh hook|full RUN_DIR GPU}
gpu=${3:?usage: run_first_gate.sh hook|full RUN_DIR GPU}
case "$(realpath -m -- "$run_dir")/" in
  "$AUTHORIZED_HOME"/*) ;;
  *) echo "first gate refused: run directory escapes authorized home" >&2; exit 64 ;;
esac
case "$mode" in hook|full) ;; *) echo "first gate refused: mode must be hook or full" >&2; exit 64;; esac
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "first gate refused: GPU must be numeric" >&2; exit 64; }

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
python "$source_dir/src/first_gate.py" verify-hook \
  --model-root "$model_root" --gpu "$gpu" --out "$artifacts/hook-verification.json"

if test "$mode" = hook; then
  exit 0
fi

acts="$artifacts/activations"
probe="$artifacts/probe"
intervention="$artifacts/intervention"
python "$source_dir/src/extract.py" \
  --manifest "$manifest" --arch llava15_7b --gpus "$gpu" --out "$acts" \
  --batch-size 8 --loci vis.last
python "$source_dir/src/first_gate.py" probe \
  --acts "$acts" --manifest "$manifest" --out "$probe" --n-boot 2000
python "$source_dir/src/intervene.py" \
  --acts "$acts" --manifest "$manifest" --arch llava15_7b --concept Effusion \
  --loci vis.last --alphas=-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1 \
  --alpha-mode reltoken --control-alphas=-1,-0.5,-0.25,0.25,0.5,1 \
  --n-random 20 --n-eval 200 --eval-split test --batch-size 16 --gpu "$gpu" \
  --out "$intervention"
python "$source_dir/src/first_gate.py" summarize-intervention \
  --csv "$intervention/intervention.csv" --meta "$intervention/meta.json" \
  --out "$artifacts/intervention-summary.json"
