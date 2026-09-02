#!/usr/bin/env bash
# Keep the reserved GPUs working without supervision.
#
# The box is shared and has no scheduler, so an idle card is a card someone else takes. This chains the
# jobs that are known to be needed so the cards stay busy from one to the next.
#
#   bash src/run_chain.sh 4,5,6
#
# Each stage writes into runs/ and skips itself if its output already exists, so the chain is
# restartable after an interruption.
set -uo pipefail

GPUS="${1:-4,5,6}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"
eval "$(conda shell.bash hook)"
conda activate base
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false

log () { echo "[$(date +%H:%M:%S)] $*"; }

stage_extract () {   # stage_extract <arch-key>
  local arch="$1" out="runs/act_$1"
  if [[ -f "$out/shard0.json" ]]; then log "skip extract $arch, already done"; return 0; fi
  log "extract $arch on gpus $GPUS"
  PYTHONUNBUFFERED=1 python src/extract.py --manifest data/manifest.csv --arch "$arch" \
      --gpus "$GPUS" --out "$out" --batch-size 8 >> "runs/extract_$arch.log" 2>&1
  log "extract $arch exit $?"
}

stage_probe () {     # stage_probe <arch-key>
  local acts="runs/act_$1" out="runs/probe_$1"
  if [[ -f "$out/probe_results.csv" ]]; then log "skip probe $1, already done"; return 0; fi
  if [[ ! -f "$acts/shard0.npz" ]]; then log "no activations for $1, skipping probe"; return 0; fi
  log "probe $1"
  # --proj-dim is not a speed hack, it is the design. Vision loci are 1280-wide and LLM loci are
  # 3584-wide, so without a common projection a deeper locus wins on width rather than on information,
  # and the survival curve is confounded by dimensionality.
  PYTHONUNBUFFERED=1 python src/probe.py --acts "$acts" --manifest data/manifest.csv \
      --concepts Effusion,Atelectasis,Pneumothorax,Consolidation,Cardiomegaly,Infiltration,Mass,Nodule,Edema \
      --proj-dim 512 --out "$out" >> "runs/probe_$1.log" 2>&1
  log "probe $1 exit $?"
}

ARCHS="${ARCHS:-qwen7b lingshu7b llavamed7b llava15_7b internvl3_8b}"

log "chain start on gpus $GPUS over: $ARCHS"

for a in $ARCHS; do stage_extract "$a"; done
for a in $ARCHS; do stage_probe   "$a"; done

log "chain done"
python src/gpu_hold.py status | tail -12
