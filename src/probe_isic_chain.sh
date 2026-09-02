#!/usr/bin/env bash
# Wait for the dermoscopy extractions, then run the same validity battery on them that the chest
# radiographs got. Probing is CPU work, so it does not contend with anything on the cards.
#
#   nohup setsid bash src/probe_isic_chain.sh > runs/probe_isic.log 2>&1 < /dev/null &

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source "$PWD/scripts/server/activate_env.sh" || exit 1
export PYTHONUNBUFFERED=1

ARCHS="${ARCHS:-lingshu7b qwen7b llavamed7b llava15_7b internvl3_8b}"
CONCEPTS="${CONCEPTS:-Nevus,Melanoma,BasalCellCarcinoma,SeborrheicKeratosis,SquamousCellCarcinoma,ActinicKeratosis}"
# Same crossing as the chest radiographs, with anatomic site standing in for view position.
TYPECOLS="${TYPECOLS:-site,sex_M,age}"
NUIS="${NUIS:-site_Trunk,site_Headandneck,sex_M}"

extraction_running () { ps -eo cmd= | grep -q "[e]xtract_isic_chain.sh"; }

while true; do
  for arch in $ARCHS; do
    acts="runs/isicact_$arch"
    out="runs/isicprobe_$arch"
    [[ -f "$out/probe_results.csv" ]] && continue
    # A finished extraction has a per-shard json for every shard file it wrote.
    n_npz=$(ls "$acts"/shard*.npz 2>/dev/null | wc -l)
    n_json=$(ls "$acts"/shard*.json 2>/dev/null | wc -l)
    [[ "$n_npz" -eq 0 || "$n_npz" -ne "$n_json" ]] && continue

    echo "[$(date +%H:%M:%S)] probing $arch ($n_npz shards)"
    python src/probe.py --acts "$acts" --manifest data/isic_manifest.csv \
        --concepts "$CONCEPTS" --type-cols "$TYPECOLS" --nuisance "$NUIS" \
        --proj-dim 512 --out "$out"
    echo "[$(date +%H:%M:%S)] finished $arch (exit $?)"
  done
  done_n=$(ls -d runs/isicprobe_*/probe_results.csv 2>/dev/null | wc -l)
  n_arch=$(echo $ARCHS | wc -w)
  echo "[$(date +%H:%M:%S)] probed $done_n/$n_arch"
  [[ "$done_n" -ge "$n_arch" ]] && break
  extraction_running || { sleep 120; extraction_running || { echo "extraction chain gone; last pass"; }; }
  sleep 60
done
echo "[$(date +%H:%M:%S)] dermoscopy probing complete"
