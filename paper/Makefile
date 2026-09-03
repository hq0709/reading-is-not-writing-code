ENGINE ?= tectonic
PYTHON ?= python
.DEFAULT_GOAL := main.pdf

ASSET_SCRIPTS := scripts/build_paper_data.py scripts/gen_fig1.py scripts/gen_fig2.py scripts/gen_fig3.py scripts/gen_tables.py
ASSETS := data/accepted_results.json figures/fig1_evidence_ladder.pdf figures/fig2_dose_responses.pdf figures/fig3_direction_specificity.pdf tables/table_decoding.tex tables/table_gates.tex
SECTIONS := $(wildcard sections/*.tex)

assets: $(ASSET_SCRIPTS)
	$(PYTHON) scripts/build_paper_data.py
	$(PYTHON) scripts/gen_fig1.py
	$(PYTHON) scripts/gen_fig2.py
	$(PYTHON) scripts/gen_fig3.py
	$(PYTHON) scripts/gen_tables.py

main.pdf: main.tex math_commands.tex refs.bib $(SECTIONS) $(ASSETS)
	$(ENGINE) -X compile main.tex --keep-logs

check: main.pdf
	@echo "pages:    $$(pdfinfo main.pdf | awk '/Pages/{print $$2}')"
	@echo "overfull: $$(grep -c Overfull main.log || true)"
	@echo "undefined:$$(grep -ci undefined main.log || true)"

clean:
	rm -f main.aux main.log main.blg main.bbl main.out

.PHONY: assets check clean
