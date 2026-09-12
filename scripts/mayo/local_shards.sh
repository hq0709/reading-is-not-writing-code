#!/usr/bin/env bash
# Run shards s0..s1 of one block sequentially on the current CUDA_VISIBLE_DEVICES (login-node GPU).
#   bash scripts/mayo/local_shards.sh MODEL DATASET MODULE N_SHARDS S0 S1 [BATCH]
set -euo pipefail
M=$1; D=$2; MOD=$3; N=$4; S0=$5; S1=$6; B=${7:-32}
cd /rodata/azradonc_dev/m253405/concept-flow/src
for ((s=S0; s<=S1; s++)); do
  python -m cftransfer.runner --model-key "$M" --dataset "$D" --module "$MOD" --shard "$s" --n-shards "$N" --batch "$B"
done
echo LOCAL_SHARDS_DONE $M $D $MOD $S0-$S1
