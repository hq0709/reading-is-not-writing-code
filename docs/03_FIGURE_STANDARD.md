# Concept Flow figure standard

This standard is derived from quantitative figures in Darcet et al., *Vision Transformers Need
Registers*; Schaeffer et al., *Are Emergent Abilities of Large Language Models a Mirage?*; Elhage et
al., *Toy Models of Superposition*; Laban et al.'s ICLR 2026 outstanding paper; and a 16-paper reference
set spanning ICLR, NeurIPS, ICML, CVPR and ICCV.

## Core rule

Each figure shows one scientific claim in the data geometry best suited to it. The caption states the
claim, defines the encoding, walks through the panels and names the uncertainty and controls.

## Page geometry

- Author figures at their final width: 3.3 inches for a column or 6.9 inches for full width; use 5.5
  inches when the venue template sets that text width.
- Keep ordinary main figures below 2.5 inches tall. Use additional height only when it exposes individual
  cells that a summary would hide.
- Do not scale a figure after export. Inspect the compiled PDF at final size.
- Use panel letters and short titles; keep sentence-length claims out of the axes.
- Axis labels use one or two words. Data, ticks and compact identifiers are the only axis furniture.

## Typography and marks

- Use the vendored Lato family through `src/figstyle.py`; do not rely on Matplotlib's default font.
- Keep all final-size text at 6 pt or larger. Axis labels should be visibly larger than tick labels.
- Use left and bottom spines for minimal plots. A boxed frame is acceptable when a benchmark-style panel
  needs exact visual lookup.
- Use solid lines with round joins and caps. Markers have a thin white edge where series cross.
- Prefer no grid. When value lookup matters, use a pale horizontal grid (`#e4e6e9`) and omit vertical
  rules.
- Use Okabe–Ito colours. Positive/negative comparisons use blue `#0072B2` and vermillion `#D55E00`, not
  red and green.
- Use sequential ramps for ordered quantities and categorical hues only for unordered groups.

## Labels and legends

- Direct-label a small number of series when their identities carry the claim.
- For multi-panel figures, use one frameless shared legend above the panels in a single row.
- Use one shared x-axis label below a grid and write repeated row or column headers once.
- Teach a non-obvious glyph or encoding in a compact key or constructed example before presenting the
  full data.
- Replace raw implementation identifiers with readable phrases when space permits.

## What each result should look like

| Result | Figure form | Required evidence in the ink |
|---|---|---|
| utilisation across models and findings | cell-level bullet grid | floor, trained probe and behaviour values remain individually recoverable; utilisation sign visible |
| depth profile | one shared-axis line chart or aligned small multiples | zero/reference line, matched loci and uncertainty at the reported locus |
| before/after readout change | sorted row-wise dot-to-dot plot | direction and magnitude of every paired change |
| intervention selectivity | dose-response curve with control band | alpha=0, both signs, random/sham band and held-out interval |
| matched concept/random effect | paired log-log scatter with identity line | every matched cell and departures from equality |
| planted anchor | aligned image-space and representation-space dose curves | dose, location control and monotonic region |
| dense model-by-concept table | booktabs table with group banners | domain, matched pairs, metric groups and uncertainty |

The figure shows individual cells when exceptions matter. A five-dot median summary does not replace a
45-cell result. A table does not repeat a figure that already exposes the same values.

## Captions

The first sentence is the result in bold. Subsequent sentences:

1. identify panels with bold `(a)`, `(b)`, and so on;
2. define every line, point, band and reference;
3. state the patient/sample unit and interval construction;
4. name the matched control and registered gate;
5. give the run or evidence identifier when space permits.

The caption reports the observation before interpretation. Scope is stated once through the positive
question answered by the next gate.

## Tables

- Use booktabs: top, header and bottom rules; no vertical rules.
- Right-align numbers and align decimals.
- Use `\rowcolor` section banners only when the grouping is part of the result.
- Group related columns with spacing and `\cmidrule`; write the group header once.
- Tint only the focal subset, such as medical models, and mark architecture-matched pairs with one shared
  symbol.
- Put uncertainty beside the estimate as an interval or `±`; do not create a standalone forest plot for
  five models when the interval belongs naturally in the result table.
- Load table colour support with `\PassOptionsToPackage{table}{xcolor}` before `\documentclass` when the
  template loads `xcolor` itself.

## Automated checks

Run `src/figcheck.py` on the drawn figure before saving. It checks:

- text against text and text against data;
- text outside its axes or the canvas;
- legends crossing sampled line segments;
- adjacent tick labels closer than 1.5 pt.

Run `src/lint_tex.py` on the paper source. It checks loose prose inside floats, references without labels,
unreferenced float labels and missing image files.

The uncertainty note uses the interval attached to each estimate. For the existing probe analysis, the
median per-estimate 95% interval width is 0.066, half-width 0.033; it is not computed by subtracting the
median lower endpoint from the median upper endpoint.

## Release checklist

- [ ] the plotted form makes the claim visible without reconstructing it from prose
- [ ] all registered controls and uncertainty appear in the figure or caption
- [ ] the panel is authored and inspected at final size
- [ ] no text is smaller than 6 pt
- [ ] every series is identifiable without repeated boxed legends
- [ ] no label, legend or tick collides with data or another label
- [ ] the caption opens with the result and states the evidence unit
- [ ] every image file exists and every float is referenced
- [ ] no table duplicates a figure's numbers
