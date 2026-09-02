#!/usr/bin/env bash
# Fill spare capacity with D2 (causally-optimised direction) jobs.
#
# Deliberately yields to the D1 sweep: it skips any card already carrying an intervene.py or cad.py job of
# ours, and asks for more free memory than D1 does, because D2 backpropagates through the model and D1 does
# not. Same stateless design as src/supervise.sh, so it can be killed and restarted at any point.
#
#   nohup setsid bash src/supervise_cad.sh > runs/supervise_cad.log 2>&1 < /dev/null &

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source "$PWD/scripts/server/activate_env.sh" || exit 1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
MIN_FREE_MB="${MIN_FREE_MB:-45000}"

# Backward from an early locus has to hold the forward activations of the vision tower as well, and the
# towers differ by an order of magnitude in how many positions they carry: the Qwen family runs 4608
# patches through full attention where LLaVA's CLIP runs 576 tokens. One threshold for all of them either
# starves the small models of usable cards or hands the large ones a card they will die on.
mem_for () {
  case "$1" in
    lingshu7b|qwen7b) echo 55000 ;;
    *)                echo "$MIN_FREE_MB" ;;
  esac
}
POLL="${POLL:-90}"
STEPS="${STEPS:-40}"
NFIT="${NFIT:-64}"
NEVAL="${NEVAL:-64}"
SCREEN="${SCREEN:-1}"
BATCH="${BATCH:-2}"
ACCUM="${ACCUM:-4}"
# Light vision towers first, so the small cards are useful while the big ones are still busy
ARCHS="${ARCHS:-llava15_7b llavamed7b internvl3_8b lingshu7b qwen7b}"
CONCEPTS="${CONCEPTS:-Effusion Cardiomegaly}"

log () { echo "[$(date +%H:%M:%S)] $*"; }

# The locus set for D2 is not the D1 set. Two of the D1 loci cannot support this measurement at all:
# llm.L{n-1}.vis is causally severed, and at llm.L{n-1}.ans the only downstream computation is the norm and
# the readout, so every yes/no question shares one gradient direction and no concept-specific direction can
# exist. The last of those is kept, as the degenerate end of the separability profile.
loci_for () {
  python - "$1" <<'PY'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from registry import REGISTRY
n = REGISTRY[sys.argv[1]].n_llm_layers
print(",".join(["connector", f"llm.L{n//4}.vis", f"llm.L{n//2}.vis",
                f"llm.L{(3*n)//4}.vis", f"llm.L{n-1}.ans"]))
PY
}

others_for () {   # the questions the direction must leave alone
  case "$1" in
    Effusion)     echo "Cardiomegaly,Pneumothorax" ;;
    Cardiomegaly) echo "Effusion,Pneumothorax" ;;
    *)            echo "Effusion,Cardiomegaly" ;;
  esac
}

busy_gpus () {
  ps -eo cmd= | grep -E "[i]ntervene\.py|[c]ad\.py" | grep -oE -- "--gpu [0-9]+" | awk '{print $2}' | sort -u
}

gpu_free_mb () {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F', ' -v g="$1" '$1==g {print $3-$2}'
}

log "cad supervisor up. min_free=${MIN_FREE_MB}MiB steps=$STEPS n_fit=$NFIT n_eval=$NEVAL"

while true; do
  pending=0
  mapfile -t busy < <(busy_gpus)

  for concept in $CONCEPTS; do
    for arch in $ARCHS; do
      name="cad_${arch}_$(echo "$concept" | tr '[:upper:]' '[:lower:]')"
      [[ -f "runs/$name/DONE" ]] && continue
      if ps -eo cmd= | grep "[c]ad.py" | grep -q -- "--out runs/$name\$"; then continue; fi
      pending=$((pending + 1))

      for g in $GPUS; do
        skip=0
        for b in "${busy[@]}"; do [[ "$b" == "$g" ]] && skip=1; done
        [[ $skip -eq 1 ]] && continue
        free=$(gpu_free_mb "$g")
        need=$(mem_for "$arch")
        [[ -z "$free" || "$free" -lt "$need" ]] && continue

        log "gpu$g <- $name  (${free} MiB free, needs ${need})"
        { echo; echo "===== launch $(date +%FT%T) on gpu$g ====="; } >> "runs/$name.log"
        nohup setsid bash -c 'python "$@"; echo "===== exit $? at $(date +%FT%T) ====="' _ src/cad.py \
            --acts "runs/act_$arch" --arch "$arch" --concept "$concept" \
            --others "$(others_for "$concept")" --loci "$(loci_for "$arch")" \
            --steps "$STEPS" --accum "$ACCUM" --n-fit "$NFIT" --n-eval "$NEVAL" \
            --batch-size "$BATCH" --frontier 5 --checkpoint "${CKPT:-0}" --screen "$SCREEN" --gpu "$g" \
            --out "runs/$name" >> "runs/$name.log" 2>&1 < /dev/null &
        disown
        busy+=("$g")
        sleep 25
        break
      done
    done
  done

  running=$(ps -eo cmd= | grep -c "[c]ad.py")
  done_n=$(ls runs/cad_*/DONE 2>/dev/null | wc -l)
  log "running=$running done=$done_n pending=$pending"
  [[ $pending -eq 0 && $running -eq 0 ]] && break
  sleep "$POLL"
done
log "cad supervisor finished: $(ls runs/cad_*/DONE 2>/dev/null | wc -l) jobs complete"
