# Medical Vision-Language Models Do Not Use What They Encode

LaTeX source for the paper. The experiment and analysis code, and the result tables every number is
computed from, live at the root of the same [`concept-flow`](https://github.com/wy-coliney/concept-flow)
repository.

## The claim

A probe score is the sum of what any network of that shape provides, what training added, and nothing at
all about whether the model can use either. Separating the three needs one control the field has almost
never run — the same probe on the **same architecture with randomly initialised weights**:

```
utilisation  U = (B - F) / (T - F)
```

Across five 7B–8B medical and general vision-language models and nine chest-radiograph findings, the
model's own zero-shot answer captures a **median 9%** of what training added, and falls **below** the
untrained floor in **22 of 45** cells. Probe AUROC spans 0.033 across the five models — the half-width of
a single bootstrap interval — while their answers span 0.179.

## Building

Needs a TeX distribution with `newtxtext`, `tcolorbox` and `booktabs`.

```bash
make          # or: tectonic -X compile main.tex
```

`tables.tex` is **generated**, not written: `src/make_tables.py` in the code repository emits it from the
released CSVs so the appendix tables cannot drift from the data.

## A note on the fonts

The preamble uses `newtxtext,newtxmath` rather than `\usepackage{times}`. Under this toolchain `times`
resolves to Latin Modern Roman with no bold and no italic face, so every `\textbf` and `\emph` renders at
regular weight **without raising an error**. It was found with `pdffonts`, not by reading.

## Licence

Paper text and figures: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
`iclr2025_conference.{sty,bst}`, `fancyhdr.sty`, `natbib.sty` and `math_commands.tex` are the ICLR 2025
author kit, redistributed under its own terms.
