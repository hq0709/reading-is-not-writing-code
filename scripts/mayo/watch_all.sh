#!/bin/bash
# One watcher loop for every analysis trigger: addendum modules (ANSDIR, EXTCOMP, TOKENW, PRECISION, ALTDIRD, ATTR,
# ANSDIRT, VALID), ALTDIR, fully scored blocks, and complete-but-unanalysed CORE blocks. Each hit packages the block
# and runs the matching analysis; output goes to logs/analysis/<model>-<dataset>-<what>.log and one summary line to stdout.
set -u
ulimit -u 65536
source /rodata/azradonc_dev/m253405/myconda/etc/profile.d/conda.sh && conda activate research
cd /rodata/azradonc_dev/m253405/concept-flow
export PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
L=/rodata/azradonc_dev/m253405/cf-transfer/logs/analysis
HOURS=${1:-24}
end=$(( $(date +%s) + HOURS * 3600 ))
declare -A seen
run() {  # model dataset what
  local key="$1/$2/$3"
  local n=${seen[$key]:-0}
  if [ "$n" -ge 2 ]; then return; fi      # a block that keeps re-appearing after two attempts is reported once, not looped
  seen[$key]=$((n + 1))
  python -m cftransfer.package --model-key $1 --dataset $2 > $L/$1-$2-package.log 2>&1
  if [ "$3" = all ]; then python -m cftransfer.analysis --model-key $1 --dataset $2 > $L/$1-$2-$3.log 2>&1
  else python -m cftransfer.analysis --model-key $1 --dataset $2 --what $3 > $L/$1-$2-$3.log 2>&1; fi
  local rc=$?
  echo "$(date -u +%FT%TZ) $1 $2 $3 rc=$rc $(grep -ciE 'traceback|error' $L/$1-$2-$3.log) error-lines"
}
while [ $(date +%s) -lt $end ]; do
  while read m d mod; do [ -n "$mod" ] && run $m $d $(echo $mod | tr 'A-Z' 'a-z'); done < <(python3 scripts/mayo/new_addendum_blocks.py 2>/dev/null)
  for b in $(python3 scripts/mayo/new_altdir_blocks.py 2>/dev/null | tr ' ' ':'); do run ${b%%:*} ${b##*:} altdir; done
  while read m d; do [ -n "$d" ] && run $m $d all; done < <(python3 scripts/mayo/new_core_blocks.py 2>/dev/null)
  while read m d; do
    [ -n "$d" ] || continue; k="$m/$d/pack"; n=${seen[$k]:-0}; [ "$n" -ge 2 ] && continue; seen[$k]=$((n + 1))
    python -m cftransfer.package --model-key $m --dataset $d > $L/$m-$d-package.log 2>&1
    echo "$(date -u +%FT%TZ) $m $d packaged (fully scored): $(python3 -c "import json;print(json.load(open('/rodata/azradonc_dev/m253405/cf-transfer/runs/$m/$d/run.json')).get('status'))")"
  done < <(python3 scripts/mayo/new_complete_blocks.py 2>/dev/null)
  sleep 600
done
echo "watch_all finished"
