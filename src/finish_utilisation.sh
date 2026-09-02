#!/usr/bin/env bash
# Wait for the untrained-floor extractions, probe them, then rebuild the utilisation table and figure.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source "$PWD/scripts/server/activate_env.sh" || exit 1
export PYTHONUNBUFFERED=1
CONC="Effusion,Cardiomegaly,Pneumothorax,Edema,Atelectasis,Consolidation,Infiltration,Mass,Nodule"

wait_for () {   # a finished extraction has one json per npz
  local d="$1"
  while true; do
    n=$(ls "$d"/shard*.npz 2>/dev/null | wc -l); m=$(ls "$d"/shard*.json 2>/dev/null | wc -l)
    [[ "$n" -gt 0 && "$n" -eq "$m" ]] && return 0
    ps -eo cmd= | grep -q "[e]xtract_random_init.py" || { echo "extraction gone, $d has $n npz $m json"; return 1; }
    sleep 60
  done
}

for arch in qwen7b internvl3_8b; do
  d="runs/rndact_$arch"
  echo "[$(date +%H:%M:%S)] waiting for $d"
  wait_for "$d" || continue
  [[ -f "runs/rndprobe_$arch/probe_results.csv" ]] && { echo "  already probed"; continue; }
  echo "[$(date +%H:%M:%S)] probing $arch untrained floor"
  python src/probe.py --acts "$d" --manifest data/manifest.csv --concepts "$CONC" \
      --proj-dim 512 --out "runs/rndprobe_$arch"
done

# Lingshu shares Qwen's architecture, so the same untrained model is its floor. LLaVA-1.5 shares
# LLaVA-Med's. Link rather than re-extract: an identical architecture with the same seed is the same floor.
for pair in "lingshu7b:qwen7b" "llava15_7b:llavamed7b"; do
  tgt="runs/rndprobe_${pair%%:*}"; src="runs/rndprobe_${pair##*:}"
  [[ -d "$src" && ! -e "$tgt" ]] && { ln -s "$(basename "$src")" "$tgt"; echo "linked $tgt -> $src"; }
done

echo "[$(date +%H:%M:%S)] rebuilding utilisation"
python src/analyze_utilisation.py
python src/figures_utilisation.py 2>/dev/null || true
echo "[$(date +%H:%M:%S)] done"
