#!/usr/bin/env bash
# Fan one (model, dataset, module) block out over N single-GPU Slurm jobs (row shards), each resumable.
#   bash scripts/mayo/submit_block.sh MODEL DATASET MODULE N_SHARDS [BATCH] [TIME] [GPUS_PER_JOB] [PARTITION]
set -euo pipefail
M=$1; D=$2; MOD=$3; N=$4; B=${5:-32}; T=${6:-12:00:00}; G=${7:-1}; P=${8:-gen-a100.p}
DM="cuda:0"; [ "$G" -gt 1 ] && DM="auto"
for ((s=0; s<N; s++)); do
  bash "$(dirname "$0")/sbatch_py.sh" -J "cf-$M-$D-$MOD-$s" -g "$G" -t "$T" -p "$P" -- \
    python -m cftransfer.runner --model-key "$M" --dataset "$D" --module "$MOD" --shard "$s" --n-shards "$N" --batch "$B" --device-map "$DM"
done
