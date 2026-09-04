#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZED_HOME=/home/qingchan
readonly SOURCE_RUN_ID=20260903T000321Z-caaae3ef346d-qwen-full
readonly PREVIOUS_RUN_ID=20260903T103508Z-456c81bad460-7a3edc4b
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
derived_run_dir=$(realpath -- "$source_dir/..")
run_dir=$(realpath -- "${RUN_DIR:?causal-ownership gate refused: trusted RUN_DIR is missing}")
case "$run_dir" in
  "$AUTHORIZED_HOME"/data/concept-flow/runs/*) ;;
  *) echo "causal-ownership gate refused: run directory is outside registered runs" >&2; exit 64 ;;
esac
test "$run_dir" = "$derived_run_dir" || {
  echo "causal-ownership gate refused: RUN_DIR does not match archived source" >&2
  exit 64
}
grep -Fx 'GPU_COUNT=1' "$run_dir/metadata.env" >/dev/null || {
  echo "causal-ownership gate refused: immutable receipt must declare one GPU" >&2
  exit 64
}
gpu=$(sed -n 's/^GPU_IDS=//p' "$run_dir/metadata.env")
[[ "$gpu" =~ ^[0-9]+$ ]] || {
  echo "causal-ownership gate refused: GPU identity is invalid" >&2
  exit 64
}

manifest="$AUTHORIZED_HOME/data/concept-flow/datasets/nih-chestxray14/manifest.csv"
source_run="$AUTHORIZED_HOME/data/concept-flow/runs/$SOURCE_RUN_ID"
previous_run="$AUTHORIZED_HOME/data/concept-flow/runs/$PREVIOUS_RUN_ID"
model_root="$AUTHORIZED_HOME/data/concept-flow/models/qwen7b-huggingface"
artifacts="$run_dir/artifacts"
questions="$artifacts/questions"
mkdir -p "$questions"
export HF_HOME="$model_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python "$source_dir/src/qwen_causal_ownership_gate.py" validate-source \
  --source-run "$source_run" --previous-run "$previous_run" --manifest "$manifest" \
  --out "$artifacts/source-reference.json" \
  --rows-out "$artifacts/registered-rows.json"
snapshot=$(python - "$artifacts/source-reference.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["model_source"])
PY
)

concepts='Effusion|Atelectasis|Pneumothorax|Cardiomegaly|Mass|Nodule'
while IFS= read -r concept; do
  case "$concept" in
    Effusion) prompt='Is there a pleural effusion in this chest radiograph? Answer yes or no.' ;;
    Atelectasis) prompt='Is there atelectasis in this chest radiograph? Answer yes or no.' ;;
    Pneumothorax) prompt='Is there a pneumothorax in this chest radiograph? Answer yes or no.' ;;
    Cardiomegaly) prompt='Is there cardiomegaly in this chest radiograph? Answer yes or no.' ;;
    Mass) prompt='Is there a lung mass in this chest radiograph? Answer yes or no.' ;;
    Nodule) prompt='Is there a lung nodule in this chest radiograph? Answer yes or no.' ;;
    *) echo "unregistered concept: $concept" >&2; exit 64 ;;
  esac
  question_dir="$questions/${concept,,}"
  mkdir -p "$question_dir"
  python "$source_dir/src/intervene.py" \
    --acts "$source_run/artifacts/activations" --manifest "$manifest" \
    --arch qwen7b --concept "$concept" --loci vis.last \
    --alphas=0,0.25 --alpha-mode reltoken --control-alphas=0.25 \
    --n-random 119 --n-eval 400 --eval-split test --batch-size 16 --gpu "$gpu" \
    --model-path "$snapshot" --directions "$source_run/artifacts/probe/directions.npz" \
    --eval-row-ids-file "$artifacts/registered-rows.json" \
    --per-image-out "$question_dir/per-image.csv" \
    --prompt "$prompt" --out "$question_dir"
done < <(printf '%s\n' "$concepts" | tr '|' '\n')

python "$source_dir/src/qwen_causal_ownership_gate.py" summarize \
  --questions-root "$questions" \
  --source-reference "$artifacts/source-reference.json" \
  --registered-rows "$artifacts/registered-rows.json" \
  --bootstrap "$artifacts/causal-ownership-bootstrap.npz" \
  --out "$artifacts/causal-ownership-summary.json"

(
  cd "$source_dir"
  python -m src.qwen_mechanism_route \
    --summary "$artifacts/causal-ownership-summary.json" \
    --output "$artifacts/mechanism-route.json"
)
