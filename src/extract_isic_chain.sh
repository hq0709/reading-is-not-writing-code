#!/usr/bin/env bash
# Extract dermoscopy activations for every model, one model at a time, on a fixed pair of cards.
#
# Sequential rather than supervised: extraction is inference only, so it is the cheapest thing running and
# has no reason to compete with the intervention and D2 sweeps for the large cards. It restarts cleanly
# because each model's output directory is skipped once its meta.json records a finished run.
#
#   GPUS=2,3 nohup setsid bash src/extract_isic_chain.sh > runs/extract_isic.log 2>&1 < /dev/null &

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source "$PWD/scripts/server/activate_env.sh" || exit 1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

GPUS="${GPUS:-2,3}"
BATCH="${BATCH:-8}"
ARCHS="${ARCHS:-lingshu7b qwen7b llavamed7b llava15_7b internvl3_8b}"
MANIFEST="${MANIFEST:-data/isic_manifest.csv}"

# The question has to be the modality's question. Asking a dermoscopy image about a chest radiograph would
# make the prompt itself the strongest signal in every representation.
PROMPT="${PROMPT:-Is this skin lesion a melanoma? Answer yes or no.}"

for arch in $ARCHS; do
  out="runs/isicact_$arch"
  if [[ -f "$out/meta.json" ]] && ls "$out"/shard*.npz >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] skip $arch, already extracted"; continue
  fi
  echo "[$(date +%H:%M:%S)] extracting $arch -> $out"
  python src/extract.py --manifest "$MANIFEST" --arch "$arch" --gpus "$GPUS" \
      --out "$out" --batch-size "$BATCH" --prompt "$PROMPT"
  echo "[$(date +%H:%M:%S)] finished $arch (exit $?)"
done
echo "[$(date +%H:%M:%S)] all dermoscopy extractions attempted"
