# Tectonic needs no local TeX tree; latexmk works too if you have one.
ENGINE ?= tectonic

main.pdf: main.tex refs.bib tables.tex $(wildcard figures/*.png)
	$(ENGINE) -X compile main.tex --keep-logs

check: main.pdf
	@echo "pages:    $$(pdfinfo main.pdf | awk '/Pages/{print $$2}')"
	@echo "overfull: $$(grep -c Overfull main.log || true)"
	@echo "undefined:$$(grep -ci undefined main.log || true)"

clean:
	rm -f main.aux main.log main.blg main.bbl main.out

.PHONY: check clean
