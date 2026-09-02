#!/usr/bin/env bash
# Run D3 (planted ground truth) for each model and concept, once that cell's D2 run has finished.
#
# The dependency is real rather than bookkeeping: D3's central number is the cosine between the activation
# displacement a planted lesion actually causes and each candidate direction, and `v_cad` is one of the
# candidates. Running D3 first would produce the probe comparison and silently omit the one that matters.
#
#   nohup setsid bash src/supervise_plant.sh > runs/supervise_plant.log 2>&1 < /dev/null &

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source "$PWD/scripts/server/activate_env.sh" || exit 1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
MIN_FREE_MB="${MIN_FREE_MB:-30000}"
POLL="${POLL:-120}"
NIMG="${NIMG:-96}"
BATCH="${BATCH:-8}"
ARCHS="${ARCHS:-llava15_7b llavamed7b internvl3_8b lingshu7b qwen7b}"
CONCEPTS="${CONCEPTS:-Effusion Nodule Consolidation}"

log () { echo "[$(date +%H:%M:%S)] $*"; }

others_for () {
  case "$1" in
    Effusion)      echo "Cardiomegaly,Nodule" ;;
    Nodule)        echo "Effusion,Cardiomegaly" ;;
    Consolidation) echo "Effusion,Cardiomegaly" ;;
    *)             echo "Effusion,Cardiomegaly" ;;
  esac
}

busy_gpus () {
  ps -eo cmd= | grep -E "[i]ntervene\.py|[c]ad\.py|[p]lant\.py" \
    | grep -oE -- "--gpu [0-9]+" | awk '{print $2}' | sort -u
}
gpu_free_mb () {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F', ' -v g="$1" '$1==g {print $3-$2}'
}

log "plant supervisor up"
while true; do
  pending=0; waiting=0
  mapfile -t busy < <(busy_gpus)
  for concept in $CONCEPTS; do
    lc=$(echo "$concept" | tr '[:upper:]' '[:lower:]')
    for arch in $ARCHS; do
      name="plant_${arch}_${lc}"
      cad="runs/cad_${arch}_${lc}"
      [[ -f "runs/$name/DONE" ]] && continue
      if ps -eo cmd= | grep "[p]lant.py" | grep -q -- "--out runs/$name\$"; then continue; fi
      # D2 ran on Effusion and Cardiomegaly. For the other findings there is no v_cad to score, so those
      # cells run with the probe and random comparison only rather than blocking forever.
      cadarg=""
      if [[ -f "$cad/DONE" ]]; then cadarg="$cad"
      elif [[ "$concept" == "Effusion" || "$concept" == "Cardiomegaly" ]]; then
        waiting=$((waiting + 1)); continue
      fi
      pending=$((pending + 1))
      for g in $GPUS; do
        skip=0; for b in "${busy[@]}"; do [[ "$b" == "$g" ]] && skip=1; done
        [[ $skip -eq 1 ]] && continue
        free=$(gpu_free_mb "$g")
        [[ -z "$free" || "$free" -lt "$MIN_FREE_MB" ]] && continue
        log "gpu$g <- $name  (${free} MiB free)"
        nohup setsid python src/plant.py --arch "$arch" --concept "$concept" \
            --others "$(others_for "$concept")" --cad-dir "$cadarg" \
            --n-images "$NIMG" --batch-size "$BATCH" --gpu "$g" \
            --out "runs/$name" > "runs/$name.log" 2>&1 < /dev/null &
        disown; busy+=("$g"); sleep 20; break
      done
    done
  done
  running=$(ps -eo cmd= | grep -c "[p]lant.py")
  log "running=$running done=$(ls runs/plant_*/DONE 2>/dev/null | wc -l) ready=$pending waiting_on_d2=$waiting"
  [[ $pending -eq 0 && $running -eq 0 && $waiting -eq 0 ]] && break
  sleep "$POLL"
done
log "plant supervisor finished"
