#!/bin/bash
# Full numbers refresh: manifest -> robustness analyses -> paper tables, macros, consistency check, figures, PDF.
set -u
ulimit -u 65536
source /rodata/azradonc_dev/m253405/myconda/etc/profile.d/conda.sh && conda activate research
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /rodata/azradonc_dev/m253405/concept-flow
export PYTHONPATH=src
t() { local s=$(date +%s); "$@"; local rc=$?; echo "$(date -u +%FT%TZ) rc=$rc $(( $(date +%s) - s ))s: $*" >&2; return $rc; }
# run.json's completed_modules is derived from the outcomes on disk, and watch_all.sh stops packaging a
# block after two attempts, so a block that gains a module later keeps a stale list and drops out of the
# coverage counts while its shards sit on disk. Repackaging every block first costs ~17s each and makes
# the refresh independent of whether the watcher happened to catch the block.
# The loop walks outcomes/ and not run.json: a block that was never packaged at all has no run.json,
# so keying on that file makes exactly the blocks that most need packaging invisible to the sweep.
repackage_all() {
  local o d n=0 bad=0
  for o in /rodata/azradonc_dev/m253405/cf-transfer/runs/*/*/outcomes; do
    d=$(dirname "$o")
    python -m cftransfer.package --model-key "$(basename "$(dirname "$d")")" --dataset "$(basename "$d")" > /dev/null 2>&1 \
      && n=$((n + 1)) || { bad=$((bad + 1)); echo "package failed: $d" >&2; }
  done
  [ "$bad" -eq 0 ] && echo "repackaged $n blocks" >&2 || echo "repackaged $n blocks, $bad FAILED" >&2
  [ "$bad" -eq 0 ]
}
t repackage_all
# summary.json feeds every table and figure and is written by cftransfer.analysis, which the same watcher
# stops re-running after two attempts. Analysis costs ~6 minutes a block, so only blocks whose outcomes are
# newer than their summary are redone; a block that gained a module since its last analysis is exactly that.
reanalyse_stale() {
  local o d m ds s n=0 bad=0
  local L=/rodata/azradonc_dev/m253405/cf-transfer/logs/analysis
  mkdir -p "$L"
  for o in /rodata/azradonc_dev/m253405/cf-transfer/runs/*/*/outcomes; do
    d=$(dirname "$o"); ds=$(basename "$d"); m=$(basename "$(dirname "$d")"); s="$d/summary.json"
    if [ -f "$s" ] && [ -z "$(find "$d/outcomes" -name 'part-*' -newer "$s" -print -quit 2>/dev/null)" ]; then
      continue
    fi
    python -m cftransfer.analysis --model-key "$m" --dataset "$ds" > "$L/$m-$ds-all.log" 2>&1 \
      && n=$((n + 1)) || { bad=$((bad + 1)); echo "analysis failed: $m/$ds" >&2; }
  done
  [ "$bad" -eq 0 ] && echo "re-analysed $n stale blocks" >&2 || echo "re-analysed $n stale blocks, $bad FAILED" >&2
  [ "$bad" -eq 0 ]
}
t reanalyse_stale
t python -m cftransfer.manifest > /dev/null
for r in geometry scale pairs validation refit round2 validfit; do t python scripts/mayo/robustness_$r.py > /rodata/azradonc_dev/m253405/cf-transfer/logs/robustness_$r.log 2>&1; done
cd /rodata/azradonc_dev/m253405/concept-flow-paper
t python scripts/build_cf_transfer_tables.py > /dev/null
t python scripts/build_robustness_tables.py > /dev/null
t python scripts/build_numbers.py > /rodata/azradonc_dev/m253405/cf-transfer/logs/build_numbers.log 2>&1
grep -i "warn" /rodata/azradonc_dev/m253405/cf-transfer/logs/build_numbers.log
t python scripts/plot_paper_figures.py > /rodata/azradonc_dev/m253405/cf-transfer/logs/plot_figures.log 2>&1; grep -E "OVERLAP|overlap check: [1-9]|Traceback" /rodata/azradonc_dev/m253405/cf-transfer/logs/plot_figures.log | head
t latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > /dev/null 2>&1
echo "errors $(grep -c '^!' main.log) undefined $(grep -ci undefined main.log) pages $(pdfinfo main.pdf 2>/dev/null | awk '/Pages/{print $2}') conclusion-p10 $(pdftotext -f 10 -l 10 main.pdf - 2>/dev/null | grep -c Conclusion)"
t python scripts/check_numbers.py > /rodata/azradonc_dev/m253405/cf-transfer/logs/check_numbers.log 2>&1; tail -1 /rodata/azradonc_dev/m253405/cf-transfer/logs/check_numbers.log
grep -v "^  ok" /rodata/azradonc_dev/m253405/cf-transfer/logs/check_numbers.log | grep -iv "all consistent" | head -20
echo "refresh done"
