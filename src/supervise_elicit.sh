#!/usr/bin/env bash
# Elicitation sweep: does a better readout close the probe-versus-answer gap?
# Runs the models with a large gap first, because they are the ones the question is about.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source "$PWD/scripts/server/activate_env.sh" || exit 1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
MIN_FREE_MB="${MIN_FREE_MB:-24000}"
POLL="${POLL:-90}"
N="${N:-500}"
BATCH="${BATCH:-16}"
# gap-bearing models first, no-gap models kept as controls
ARCHS="${ARCHS:-llavamed7b llava15_7b qwen7b internvl3_8b lingshu7b}"
CONCEPTS="${CONCEPTS:-Effusion,Cardiomegaly,Pneumothorax,Edema}"

log () { echo "[$(date +%H:%M:%S)] $*"; }
busy_gpus () {
  ps -eo cmd= | grep -E "[i]ntervene\.py|[c]ad\.py|[p]lant\.py|[e]licit\.py" \
    | grep -oE -- "--gpu [0-9]+" | awk '{print $2}' | sort -u
}
gpu_free_mb () {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F', ' -v g="$1" '$1==g {print $3-$2}'
}

log "elicit supervisor up. n=$N concepts=$CONCEPTS"
while true; do
  pending=0
  mapfile -t busy < <(busy_gpus)
  for arch in $ARCHS; do
    name="elicit_$arch"
    [[ -f "runs/$name/DONE" ]] && continue
    if ps -eo cmd= | grep "[e]licit.py" | grep -q -- "--out runs/$name\$"; then continue; fi
    pending=$((pending + 1))
    for g in $GPUS; do
      skip=0; for b in "${busy[@]}"; do [[ "$b" == "$g" ]] && skip=1; done
      [[ $skip -eq 1 ]] && continue
      free=$(gpu_free_mb "$g")
      [[ -z "$free" || "$free" -lt "$MIN_FREE_MB" ]] && continue
      log "gpu$g <- $name  (${free} MiB free)"
      # Append, never truncate, and record the exit status. Overwriting on relaunch is how a crash loop
      # stayed invisible for four hours in the D2 sweep: sixty launches, zero completions, no traceback
      # left on disk to say why.
      { echo; echo "===== launch $(date +%FT%T) on gpu$g ====="; } >> "runs/$name.log"
      nohup setsid bash -c 'python "$@"; echo "===== exit $? at $(date +%FT%T) ====="' _ src/elicit.py --arch "$arch" --n-images "$N" --batch-size "$BATCH" \
          --concepts "$CONCEPTS" --gpu "$g" --out "runs/$name" >> "runs/$name.log" 2>&1 < /dev/null &
      disown; busy+=("$g"); sleep 20; break
    done
  done
  running=$(ps -eo cmd= | grep -c "[e]licit.py")
  log "running=$running done=$(ls runs/elicit_*/DONE 2>/dev/null | wc -l) pending=$pending"
  [[ $pending -eq 0 && $running -eq 0 ]] && break
  sleep "$POLL"
done
log "elicit supervisor finished"
