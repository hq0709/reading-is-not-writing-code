#!/usr/bin/env bash
# One-GPU preparation for (model, dataset): features (all roles) -> fits (both loci, seeds 0/1/2) -> preflight (both loci).
#   bash scripts/mayo/prep_model.sh MODEL DATASET [BATCH] [DEVICE_MAP]
set -euo pipefail
M=$1; D=$2; B=${3:-32}; DM=${4:-cuda:0}
RUN=/rodata/azradonc_dev/m253405/cf-transfer/runs/$M/$D
cd /rodata/azradonc_dev/m253405/concept-flow/src
if [ ! -f "$RUN/features/vis.last.npz" ] || ! python - <<PY
import numpy as np, sys; f=np.load("$RUN/features/vis.last.npz"); sys.exit(0 if (f["fit_role"]=="train").any() else 1)
PY
then
  python -m cftransfer.features --model-key "$M" --dataset "$D" --roles train,preflight,calibration,test --batch-size "$B" --device-map "$DM"
fi
python -m cftransfer.fit --model-key "$M" --dataset "$D"
python -m cftransfer.preflight --model-key "$M" --dataset "$D" --locus primary --device-map "$DM"
python -m cftransfer.preflight --model-key "$M" --dataset "$D" --locus connector --device-map "$DM"
python - <<PY
from cftransfer.adapters import get_adapter
import json, torch
ad = get_adapter("$M").load(device_map="$DM")
json.dump(ad.processing_settings(), open("$RUN/processing_settings.json", "w"), indent=1, default=str)
print("processing_settings.json written")
PY
echo PREP_DONE $M $D
