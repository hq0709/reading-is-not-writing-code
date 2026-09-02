#!/usr/bin/env bash
# D0 for every model: the AUROC of the model's own answer, against the same labels the probes use.
# One model per card; inference only, so it co-exists with whatever else is on the box.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source "$PWD/scripts/server/activate_env.sh" || exit 1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
N="${N:-1000}"
i=0
for spec in "lingshu7b 0" "qwen7b 1" "llavamed7b 2" "llava15_7b 3" "internvl3_8b 4"; do
  set -- $spec; arch=$1; gpu=$2
  [[ -f "runs/behav_$arch/DONE" ]] && { echo "skip $arch"; continue; }
  echo "[$(date +%H:%M:%S)] gpu$gpu <- behav_$arch"
  nohup setsid python src/behaviour.py --arch "$arch" --n-images "$N" --batch-size 16 \
      --gpu "$gpu" --out "runs/behav_$arch" > "runs/behav_$arch.log" 2>&1 < /dev/null &
  disown; sleep 15
done
echo "launched"
