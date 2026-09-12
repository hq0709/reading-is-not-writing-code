#!/usr/bin/env bash
# Once the COCO subset is extracted, queue Slurm preparation (features -> fits -> preflight) for each model on coco.
set -euo pipefail
LOG=/rodata/azradonc_dev/m253405/cf-transfer/logs/coco-extract.log
until grep -q "final missing 0" "$LOG"; do
  if grep -q "final missing [1-9]\|Traceback" "$LOG"; then echo "COCO extraction failed"; exit 1; fi
  sleep 60
done
cd /rodata/azradonc_dev/m253405/concept-flow
for m in q25-7 q3-8 medgemma-4 llava15-7 iv35-8 llavamed-7 q25-3 q3-4 lingshu-7 iv35-14 llava15-13; do
  bash scripts/mayo/sbatch_py.sh -J prep-$m-coco -t 06:00:00 -- bash /rodata/azradonc_dev/m253405/concept-flow/scripts/mayo/prep_model.sh $m coco 32 cuda:0
done
for m in q25-32 q3-32 medgemma-27 lingshu-32; do
  bash scripts/mayo/sbatch_py.sh -J prep-$m-coco -g 2 -t 08:00:00 -- bash /rodata/azradonc_dev/m253405/concept-flow/scripts/mayo/prep_model.sh $m coco 16 auto
done
echo COCO_PREPS_SUBMITTED
