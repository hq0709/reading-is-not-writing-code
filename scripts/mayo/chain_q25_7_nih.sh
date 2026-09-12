#!/usr/bin/env bash
# Wait for the train-feature extraction, then merge -> fit -> preflight (both loci) -> settings -> CORE trial (3 rows).
set -euo pipefail
LOG=/rodata/azradonc_dev/m253405/cf-transfer/logs/feat-q25-7-nih-train.log
until grep -q "exit=0" "$LOG"; do sleep 20; done
RUN=/rodata/azradonc_dev/m253405/cf-transfer/runs/q25-7/nih
cd /rodata/azradonc_dev/m253405/concept-flow/src
python /rodata/azradonc_dev/m253405/concept-flow/scripts/mayo/merge_features.py "$RUN"
python -m cftransfer.fit --model-key q25-7 --dataset nih
python -m cftransfer.preflight --model-key q25-7 --dataset nih --locus primary --device-map cuda:0
python -m cftransfer.preflight --model-key q25-7 --dataset nih --locus connector --device-map cuda:0
python - <<PY
from cftransfer.adapters import get_adapter
import json
ad = get_adapter("q25-7").load(device_map="cuda:0")
json.dump(ad.processing_settings(), open("$RUN/processing_settings.json", "w"), indent=1, default=str)
print("processing_settings.json written")
PY
python -m cftransfer.runner --model-key q25-7 --dataset nih --module CORE --shard 0 --n-shards 200 --batch 32 --rows-per-part 1 --limit-rows 600
echo CHAIN_DONE
