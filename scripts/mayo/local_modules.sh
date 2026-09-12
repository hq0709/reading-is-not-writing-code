#!/usr/bin/env bash
# Run a sequence of modules (with shard specs) locally on the current GPU:  MODEL DATASET "MOD:N:S0-S1,MOD:N:S0-S1,..." [BATCH]
set -euo pipefail
M=$1; D=$2; SPEC=$3; B=${4:-32}
cd /rodata/azradonc_dev/m253405/concept-flow/src
IFS=, read -ra ITEMS <<< "$SPEC"
for it in "${ITEMS[@]}"; do
  MOD=${it%%:*}; rest=${it#*:}; N=${rest%%:*}; range=${rest#*:}; S0=${range%-*}; S1=${range#*-}
  if [ "$S0" -le "$S1" ]; then seq_list=$(seq "$S0" "$S1"); else seq_list=$(seq "$S0" -1 "$S1"); fi
  for s in $seq_list; do
    python -m cftransfer.runner --model-key "$M" --dataset "$D" --module "$MOD" --shard "$s" --n-shards "$N" --batch "$B"
  done
done
echo LOCAL_MODULES_DONE $M $D $SPEC
