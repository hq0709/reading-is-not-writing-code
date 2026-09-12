#!/usr/bin/env bash
# Generic Slurm launcher for cftransfer modules on one A100-80GB (or more via --gpus).
#   bash scripts/mayo/sbatch_py.sh -J name [-g N_GPUS] [-t HH:MM:SS] [-p partition] -- python -m cftransfer.<module> args...
set -euo pipefail
NAME=cf; GPUS=1; TIME=04:00:00; PART=gen-a100.p; MEM=""
while [ $# -gt 0 ]; do
  case "$1" in
    -J) NAME=$2; shift 2;; -g) GPUS=$2; shift 2;; -t) TIME=$2; shift 2;; -p) PART=$2; shift 2;; --mem) MEM=$2; shift 2;;
    --) shift; break;; *) echo "unknown arg $1"; exit 2;;
  esac
done
LOGS=/rodata/azradonc_dev/m253405/cf-transfer/logs/slurm
GRES="gpu:a100:$GPUS"; [ "$PART" = "gen-h100" ] && GRES="gpu:h100:$GPUS"
[ -z "$MEM" ] && MEM="$((64 * GPUS))G"   # never the partition default (= whole node)
MEMOPT="--mem=$MEM"
sbatch -p "$PART" --gres="$GRES" -t "$TIME" -J "$NAME" -o "$LOGS/$NAME.%j.out" -e "$LOGS/$NAME.%j.err" \
  --cpus-per-task=$((8 * GPUS)) $MEMOPT --wrap "
set -euo pipefail
ulimit -u 65536
source /rodata/azradonc_dev/m253405/myconda/etc/profile.d/conda.sh && conda activate research
export HF_HOME=/rodata/azradonc_dev/m253405/cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
cd /rodata/azradonc_dev/m253405/concept-flow/src
echo node=\$(hostname) job=\$SLURM_JOB_ID gpus=\$CUDA_VISIBLE_DEVICES start=\$(date -u +%FT%TZ)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$*
echo end=\$(date -u +%FT%TZ)
"
