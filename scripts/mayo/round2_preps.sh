#!/bin/bash
# Round-2 preps and enqueue: ALTDIRD (displacement families appended to altdir files) and ANSDIRT at prefix 11,
# ATTR (view/sex/age directions; chest blocks) at prefix 4. One block at a time; each block is enqueued only after
# its prep files exist. Idempotent: skips blocks whose prep files already carry the new keys.
set -u
ulimit -u 65536
source /rodata/azradonc_dev/m253405/myconda/etc/profile.d/conda.sh && conda activate research
cd /rodata/azradonc_dev/m253405/concept-flow
export PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
RUNS=/rodata/azradonc_dev/m253405/cf-transfer/runs
python - <<'PY' > /tmp/r2_blocks.txt
import csv
seen=set()
for r in csv.DictReader(open('/rodata/azradonc_dev/m253405/cf-transfer/runs/manifest.csv')):
    if r['block_included'].lower() in ('true','1') and (r['model_key'],r['dataset']) not in seen:
        seen.add((r['model_key'],r['dataset'])); print(r['model_key'], r['dataset'])
PY
echo "$(date -u +%FT%TZ) blocks: $(wc -l < /tmp/r2_blocks.txt)"
while read m d; do
  alt=$RUNS/$m/$d/fits/vis.last/altdir_seed0.npz
  [ -f "$alt" ] || alt=""
  if [ -n "$alt" ] && python -c "import numpy as np,sys; z=np.load('$alt'); sys.exit(0 if any(k.startswith('dom_disp') for k in z.files) else 1)"; then
    echo "$(date -u +%FT%TZ) ALTDIRD prep present $m $d"
  elif [ -n "$alt" ]; then
    echo "$(date -u +%FT%TZ) ALTDIRD prep $m $d"; python -m cftransfer.altdir --model-key $m --dataset $d > /dev/null 2>&1 || { echo "  altdir FAILED $m $d"; continue; }
  else
    echo "$(date -u +%FT%TZ) no altdir file yet for $m $d (ALTDIR pending); skipping ALTDIRD"; continue
  fi
  python -m cftransfer.enqueue --models $m --datasets $d --modules ALTDIRD --no-prep --prefix 11 2>&1 | tail -1
done < /tmp/r2_blocks.txt
# ANSDIRT on the blocks that have the answer-direction prep
for p in $RUNS/*/*/fits/vis.last/ansdir_seed0.npz; do
  m=$(echo $p | awk -F/ "{print \$(NF-4)}"); d=$(echo $p | awk -F/ "{print \$(NF-3)}")
  echo "$(date -u +%FT%TZ) ANSDIRT enqueue $m $d"; python -m cftransfer.enqueue --models $m --datasets $d --modules ANSDIRT --no-prep --prefix 11 2>&1 | tail -1
done
# ATTR on chest blocks
while read m d; do
  [ "$d" = coco ] && continue
  if ls $RUNS/$m/$d/fits/vis.last/attr_seed0.npz > /dev/null 2>&1; then echo "$(date -u +%FT%TZ) ATTR prep present $m $d"
  else echo "$(date -u +%FT%TZ) ATTR prep $m $d"; python -m cftransfer.attr --model-key $m --dataset $d > /dev/null 2>&1 || { echo "  attr FAILED $m $d"; continue; }; fi
  python -m cftransfer.enqueue --models $m --datasets $d --modules ATTR --no-prep --prefix 4 2>&1 | tail -1
done < /tmp/r2_blocks.txt
echo "$(date -u +%FT%TZ) round2_preps done"
