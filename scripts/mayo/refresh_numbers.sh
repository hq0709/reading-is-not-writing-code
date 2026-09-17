#!/bin/bash
# Full numbers refresh: manifest -> robustness analyses -> paper tables, macros, consistency check, figures, PDF.
set -u
ulimit -u 65536
source /rodata/azradonc_dev/m253405/myconda/etc/profile.d/conda.sh && conda activate research
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /rodata/azradonc_dev/m253405/concept-flow
export PYTHONPATH=src
t() { local s=$(date +%s); "$@"; local rc=$?; echo "$(date -u +%FT%TZ) rc=$rc $(( $(date +%s) - s ))s: $*" >&2; return $rc; }
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
