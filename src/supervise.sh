#!/usr/bin/env bash
# Keep every GPU busy with intervention jobs until the queue is empty.
#
# Idempotent and restartable by design. It holds no state in memory: on every poll it recomputes which
# jobs still lack an output CSV and which GPUs have none of our jobs on them, then fills the gaps. Kill it
# and restart it at any time and it picks up where it left off, which matters because a previous Python
# supervisor was taken down with the shell that started it and orphaned seven queued jobs.
#
#   nohup setsid bash src/supervise.sh > runs/supervise.log 2>&1 < /dev/null &
#
# Env: GPUS (default 0-7), NEVAL, BATCH, POLL.

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
ROOT="$PWD"
eval "$(conda shell.bash hook)"
conda activate base
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
NEVAL="${NEVAL:-96}"
BATCH="${BATCH:-16}"
POLL="${POLL:-60}"
MIN_FREE_MB="${MIN_FREE_MB:-28000}"

ARCHS="lingshu7b qwen7b llavamed7b llava15_7b internvl3_8b"
CONCEPTS="Effusion Cardiomegaly Pneumothorax"

log () { echo "[$(date +%H:%M:%S)] $*"; }

loci_for () {   # matched relative depths, so curves compare across 28- and 32-layer models
  python - "$1" <<'PY'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from registry import REGISTRY
n = REGISTRY[sys.argv[1]].n_llm_layers
# Two channels, because they are not equivalent. Perturbing the VISUAL positions at layer L can only
# reach the answer through the attention of layers L+1..n-1, so at the final layer it cannot reach it at
# all: measured on Lingshu-7B the logits move by exactly 0.00000 at llm.L27.vis, while llm.L26.vis moves
# them by 1.03 and llm.L27.ans by 11.11. The deepest visual locus is kept deliberately, as a structural
# zero: a place where the causal path is severed by construction, so any non-zero reading there would
# indict the measurement rather than describe the model.
# Density is a statistical requirement, not a nicety. The headline is whether decodability RANKS
# steerability across loci, and the standard error of a rank correlation falls as 1/sqrt(k-1), so twelve
# loci left the pooled test at p = 0.05 and could not separate a weak real effect from none. Twenty loci
# is the coarsest grid that gives the correlation room to be measured.
#
# The argmax version of the claim, "the peak of one is never the peak of the other", is deliberately NOT
# the headline: with k loci and two independent argmaxes it holds by chance with probability
# (1 - 1/k)^15, which is 0.27 at k = 12 and 0.81 at k = 70. It is what independence predicts, not
# evidence against it.
eighth = max(n // 8, 1)
vis = sorted({0} | {i * eighth for i in range(1, 8)} | {n - 2})
ans = sorted({i * eighth for i in range(1, 8)} | {n - 1})
print(",".join(["vis.last", "connector"]
               + [f"llm.L{i}.vis" for i in vis]
               + [f"llm.L{i}.ans" for i in ans]
               + [f"llm.L{n - 1}.vis"]))
PY
}

# GPUs that currently carry one of our intervention jobs. The [i] in the pattern keeps grep from
# matching its own command line, which is how three earlier cleanups killed the calling shell.
busy_gpus () {
  ps -eo cmd= | grep "[i]ntervene.py" | grep -oE -- "--gpu [0-9]+" | awk '{print $2}' | sort -u
}

gpu_free_mb () {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F', ' -v g="$1" '$1==g {print $3-$2}'
}

log "supervisor up. gpus [$GPUS] n_eval=$NEVAL batch=$BATCH"

while true; do
  pending=0
  mapfile -t busy < <(busy_gpus)

  for concept in $CONCEPTS; do
    for arch in $ARCHS; do
      name="${PREFIX:-int}_${arch}_$(echo "$concept" | tr '[:upper:]' '[:lower:]')"
      [[ -f "runs/$name/DONE" ]] && continue
      # already running?
      if ps -eo cmd= | grep "[i]ntervene.py" | grep -q -- "--out runs/$name\$"; then continue; fi
      pending=$((pending + 1))

      for g in $GPUS; do
        skip=0
        for b in "${busy[@]}"; do [[ "$b" == "$g" ]] && skip=1; done
        [[ $skip -eq 1 ]] && continue
        free=$(gpu_free_mb "$g")
        [[ -z "$free" || "$free" -lt "$MIN_FREE_MB" ]] && continue

        loci=$(loci_for "$arch")
        log "gpu$g <- $name  (${free} MiB free)"
        nohup setsid python src/intervene.py \
            --acts "runs/act_$arch" --arch "$arch" --concept "$concept" \
            --loci "$loci" --n-eval "$NEVAL" --batch-size "$BATCH" --gpu "$g" \
            --n-random "${NRANDOM:-16}" --alpha-mode "${ALPHAMODE:-reltoken}" \
            --out "runs/$name" > "runs/$name.log" 2>&1 < /dev/null &
        disown
        busy+=("$g")
        sleep 20            # let it claim memory before the next placement decision
        break
      done
    done
  done

  running=$(ps -eo cmd= | grep -c "[i]ntervene.py")
  done_n=$(ls runs/int_*/DONE 2>/dev/null | wc -l)
  log "running=$running done=$done_n pending=$pending"
  [[ $pending -eq 0 && $running -eq 0 ]] && break
  sleep "$POLL"
done

log "supervisor finished: $(ls runs/int_*/DONE 2>/dev/null | wc -l) jobs complete"
