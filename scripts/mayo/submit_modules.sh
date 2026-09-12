#!/usr/bin/env bash
# Submit every requested module of one (model, dataset) as sharded single-GPU Slurm jobs.
#   bash scripts/mayo/submit_modules.sh MODEL DATASET SHARDS_CORE [BATCH] [TIME] [GPUS_PER_JOB] [PARTITION] [MODULES]
# Shard counts scale with each module's row budget relative to CORE (PROMPT 5/3x, DOSE ~1/3x, REFIT ~1/9x, LOCUS 1x).
set -euo pipefail
M=$1; D=$2; NC=$3; B=${4:-32}; T=${5:-12:00:00}; G=${6:-1}; P=${7:-gen-a100.p}; MODS=${8:-"CALIBRATION,CORE,PROMPT,DOSE,REFIT,LOCUS,LOCUS_CALIBRATION"}
declare -A SCALE=( [CORE]=1 [PROMPT]=2 [DOSE]=1 [REFIT]=1 [LOCUS]=1 [CALIBRATION]=1 [LOCUS_CALIBRATION]=1 )
declare -A NSH
NSH[CORE]=$NC; NSH[LOCUS]=$NC; NSH[PROMPT]=$(( NC * 5 / 3 + 1 )); NSH[DOSE]=$(( NC / 3 + 1 )); NSH[REFIT]=$(( NC / 6 + 1 )); NSH[CALIBRATION]=1; NSH[LOCUS_CALIBRATION]=1
IFS=, read -ra LIST <<< "$MODS"
for MOD in "${LIST[@]}"; do
  n=${NSH[$MOD]}
  echo "== $M $D $MOD -> $n shard(s)"
  bash "$(dirname "$0")/submit_block.sh" "$M" "$D" "$MOD" "$n" "$B" "$T" "$G" "$P"
done
