# Figure standard, taken from published figures rather than from taste

Three papers were rendered page by page and read as figures, not as text: Darcet et al., *Vision
Transformers Need Registers* (ICLR 2024 outstanding paper); Schaeffer et al., *Are Emergent Abilities of
Large Language Models a Mirage?* (NeurIPS 2023 outstanding paper); Elhage et al., *Toy Models of
Superposition*. What follows are the rules their figures actually obey, with the violation each one fixes
in the first draft of ours.

## Size and furniture

1. **A main figure occupies about a sixth of a page.** The registers paper's Figure 7, its headline
   quantitative result, is three panels roughly 1.5 by 1.2 inches each. Our first draft used a full page
   for one figure.
2. **Nothing inside the axes except data, tick labels and a one-word axis label.** "norm". "density".
   "proportion". Our first draft put a sentence-long title inside every panel and a second sentence as an
   annotation.
3. **No panel titles.** Panel letters and the entire explanation live in the caption. Where a column
   header is unavoidable it is one or two words in body font, not bold and not a claim.
4. **7 to 8 point type throughout, sparse ticks, no gridlines, no boxes.**
5. **Legends are the last resort.** Three of the registers paper's quantitative figures have none.

## The caption does the work

6. **The caption's first sentence is the claim, in bold.** Schaeffer et al.: "**Claimed emergent abilities
   evaporate upon changing the metric.**" Then the panels are walked through in normal type. Our first
   draft had the claim in the axes and a caption that described the axes.
7. **Panel references are bold inside the caption**: "**(a)**: Distribution of output token norms along
   layers."

## What to plot

8. **Show the phenomenon, not a summary of it.** The registers paper argues that some patches carry
   abnormal norms and proves it with a *histogram of norms* that is visibly bimodal, next to a heat map
   with visible outlier patches. Our first draft only ever plotted means and medians of AUROC. The figure
   that replaced it plots the two score distributions directly, and the claim is legible without reading a
   number.
9. **Small multiples with shared structure.** Schaeffer et al. use three tasks across the columns and two
   metrics down the rows, so the reader compares by looking down a column. Our Figure 1 uses five models
   across and two quantities down.
10. **Progressive series where a parameter varies.** Elhage et al. show seven sparsity levels side by side
    so the transition is visible rather than asserted.

## Tables

11. **Booktabs.** Three horizontal rules and no vertical ones, with a thin rule separating groups of
    paired rows. Numbers right aligned.

## What we did not copy

Elhage et al. place explanatory prose inside the figure, beside the element it explains, with a coloured
accent bar. That works in a web article with unlimited width and does not survive a two-column PDF, so the
prose stays in the caption here.

## Checklist before a figure ships

- [ ] under 2.5 inches tall, and 3.3 or 6.9 inches wide
- [ ] no title anywhere inside the axes
- [ ] every axis label is one or two words
- [ ] the caption opens with the claim in bold
- [ ] the claim is visible in the ink, not only in the numbers printed on it
- [ ] no label collides with another at final size

---

# Round two: typography, measured at 400 dpi

The first round fixed size and restraint. It did not fix how the figures *look*, and the reason was
typographic. Rendering the first figure of Laban et al. (ICLR 2026 outstanding paper) at 400 dpi and
inspecting it gives the following, each with the default it replaces.

| detail | published figure | matplotlib default we were shipping |
|--------|------------------|-------------------------------------|
| font | a humanist sans (Roboto) | **DejaVu Sans**, narrow apertures and a heavy stem, clumsy beside Times |
| size hierarchy | axis label about **1.7x** the tick label | 6.6 against 6.0, which reads as no hierarchy and is the clearest giveaway of a default plot |
| tick marks | **none**; the numbers sit away from the axis with padding | 2pt ticks on both axes |
| spines | left and bottom only, drawn heavier than the data | thin, and often four of them |
| markers | solid, **no edge**, large enough to see | 13pt with a white edge, which disappears at print size |
| series names | **next to the data**, bold, in the series colour | a legend box |
| colour | two saturated hues plus pale tints of the same hues as regions | five categorical colours |
| weight | mixed within a line: the word carrying the claim is bold | one weight throughout |
| padding | data occupies about two thirds of the panel | data fills the panel |

Lato is used here because it is open-licensed, is a humanist sans of the same character as Roboto, and was
the best available; the family is vendored under `fonts/` and registered by `src/figstyle.py`, which is now
the single place any of these decisions is made.

## Two failures this round, both found by looking rather than by reasoning

Making the span bars longer to separate two tick labels made them worse: **widening an axis range pushes
the plotted columns together in figure coordinates**, because the data occupies a smaller fraction of a
fixed panel. The fix was the opposite of the instinct.

A figure environment was deleted along with the section text it sat in, leaving two `\ref` calls with no
target and no LaTeX warning, because the label went with it. Listing the image files referenced in the
main text and in the appendix, and comparing the count against the figures the PDF actually contains, is
now part of the check below.

## Checklist, extended

- [ ] the figure is authored at the width it will occupy, 5.5 inches for ICLR, never scaled by `\linewidth`
- [ ] the font is the vendored humanist sans, not the matplotlib default
- [ ] the axis label is visibly larger than the tick labels
- [ ] there are no tick marks, and only two spines
- [ ] markers are solid and edgeless
- [ ] series are named beside the data unless a legend is genuinely unavoidable
- [ ] no label collides with another at final size, checked on the compiled PDF and not on the png
- [ ] every figure file referenced by the tex exists, and every figure in the tex is referenced by the text

---

# Round three: form, not typography

Rounds one and two fixed size and type. They left the figures showing the wrong *things*. Rendering
Figure 6 of Laban et al. (ICLR 2026 outstanding paper) at 200 dpi and counting what is on it:

- 15 models x 3 conditions = **45 box plots** in one panel
- **three numbers annotated on every box**: the upper value above, the lower value below, and the
  percentage drop rotated beside it, each coloured to match its row
- rows carry a colour and a **vertical row label** in that colour; columns carry **one shared set of
  rotated headers** serving all three rows
- a **narrow left panel that teaches the encoding** on constructed examples before the reader meets any
  data, with its own sub-caption
- a narrow right panel of small multiples, again with every value annotated

Ours, at the same point in the paper, showed the **median of 45 cells as five dots**. The 45 cells were
the result and none of them was visible.

| what a reader can do | published figure | ours before |
|----------------------|------------------|-------------|
| look up one model and one finding | yes, with the value printed | no |
| see the spread within a model | yes | no |
| see which cells are exceptions | yes | no |
| learn the encoding before reading data | yes, panel (a) | no |

The replacement puts all 45 cells on one shared AUROC scale as bullet glyphs, prints utilisation in every
cell, shades rows by sign, rotates the nine finding names once across the top, and adds a left panel that
draws three constructed cells to teach the glyph. It also made the summary table redundant, which was
removed: a table and a figure carrying the same five numbers is a page of the nine spent twice.

ROME (Meng et al., NeurIPS 2022) supplies a third form we do not yet use: **lettered call-outs placed as
coloured boxes directly on a single continuous diagram**, with the caption reading as a walk through the
letters, rather than separate lettered panels.

## Checklist, extended again

- [ ] the figure shows the data, not a summary of the data, unless the summary *is* the claim
- [ ] a reader can look up an individual cell and read its value
- [ ] rows or columns carry colour and a label, and shared headers are written once
- [ ] the encoding is taught somewhere, either in a panel or unambiguously in the caption
- [ ] no table repeats numbers a figure already carries

---

## Round 4 — the CVPR line-chart school, and the coloured-banner table

Two reference artefacts were supplied directly: a three-panel line chart (VQA/MME benchmarks, serif type,
boxed axes) and a `table*` using `\rowcolor` section banners. Both are the CVPR/ICLR house style rather
than the minimal school of Rounds 1–3, and the paper now follows them.

### Lines

| detail | before | now |
|---|---|---|
| dash | non-emphasised series dashed `(0,(4,2))` | **all solid**; at 1.0 pt a 4-on-2 dash is a row of blocks |
| weight | 1.9 emphasis / 1.15 rest | 1.25 / 1.0 — a 1.9× ratio reads as two different chart types |
| joins | mitre (default) | **round join and cap** — a mitre throws a spike off every acute corner, which is most of what made thin lines look ragged |
| markers | filled, no edge | white edge 0.55 pt, so crossings stay legible |
| grid | both axes | **y only** — a vertical rule under every steep segment crosshatches the zigzag |
| aspect | 5.5 × 1.95 in over 9 x-steps | 5.5 × 2.35 — the same data over more height turns through gentler angles |
| legend | inside panel 1, holding a corner open | one `fig.legend` below all panels; holding a corner open cost a fifth of the plotted range in *every* panel |

### Colour

Red-against-green is out everywhere, replaced by Okabe–Ito. It is the one pair a colourblind reader cannot
separate, and neither reference figure uses it. `GOOD/BAD` are now blue `#0072B2` / vermillion `#D55E00`.

### Type floor

Nothing under **6 pt** at final size. The grid figure had annotation at `ANNOT - 1.5` = 5.1 pt, which
survives on screen and dies in print.

### The table

Devices taken from the reference and what each is doing:

- `\rowcolor` **section banner rows** via `\multicolumn` — the grouping *is* the result, so the dichotomy
  (positive vs negative utilisation) is the table's spine rather than an alphabetical model list
- **row tint** for the subset the paper is about (the two medical models)
- `@{\hskip 13pt}` **column grouping** — {F, T, T−F} what a probe sees | {B, U} what the model does |
  {below F, Δelicit} controls, with a `\cmidrule` and an italic group header over each
- **small parenthetical after the name** (the reference's `{\scriptsize(CVPR25)}` venue tag) carrying the
  medical/general domain
- `\textcolor` on the one column whose *sign* is the claim
- small rank numerals in `T` and `B`, which put the paper's first claim inside the table: the two orderings
  cross (LLaVA-Med is 2nd by probe and 4th by answer)
- daggers marking the two architecture-matched pairs, which is how the design becomes visible in the table

`\usepackage[table]{xcolor}` clashes because another package loads xcolor first; use
`\PassOptionsToPackage{table}{xcolor}` before `\documentclass`.

### What this cost

The table pushed the main text from 9 pages to 10. Reclaimed by moving §11.1–11.3 to Appendix F in full
and keeping a two-paragraph version in the body, plus compression in Related Work and §8. No result,
number or claim was dropped.

### A fault the compiler did not catch

Figure 1's caption had 630 characters of a *different* figure's caption sitting loose inside the float
after `\label`, left over from an earlier caption replacement. It typeset as an unattributed truncated
paragraph under the teaser and raised no warning. `src/lint_tex.py` now fails the build on loose prose
inside a float, on any `\ref` without a `\label`, on any float `\label` never referenced, and on a missing
image file.

---

## Round 5 — a 16-paper visualisation survey, 2026-08-29

Reference set widened from 6 to 16 PDFs in `refs/`, chosen to cover every figure type this paper uses:
Chinchilla, Emergent Abilities (TMLR 2022), Are Emergent Abilities a Mirage (NeurIPS 2023 outstanding),
Language Models (Mostly) Know What They Know, The Platonic Representation Hypothesis (ICML 2024), Do ViTs
See Like CNNs (NeurIPS 2021), A ConvNet for the 2020s (CVPR 2022), Segment Anything (ICCV 2023), IOI,
Representation Engineering, plus the six already held. Figure pages rendered and read individually.

### What recurs, and what we were missing

| device | seen in | our state before |
|---|---|---|
| one shared legend for all panels, **above** them, in a single **frameless** row | Emergent Fig. 2; Mirage Fig. 1 | below the panels, framed → **fixed** |
| one x-axis label centred under the whole grid, not per panel | Emergent Fig. 2 | removed entirely → **restored** |
| two groups on one y-axis separated by a **hairline**, not a gap | ConvNeXt Fig. 1 | separated by a gap → **fixed** |
| the y quantity named as a small **corner caption inside the frame** rather than rotated up the side | ConvNeXt Fig. 1 | rotated label → **fixed in fig5** |
| **direct labels instead of a legend** when the series names are the point | ConvNeXt Fig. 1 | legend → **fixed in fig5**; teaser already did it |
| panel letter + short title, letter set apart by weight | Emergent "(A) Mod. arithmetic"; Registers | already done |
| **hero panel + small multiples** — the claim at full size, the replication at a glance | Platonic Fig. 3 | not used; our three panels are equal-status, so not adopted |
| inline glyph key only for the encoding that is *not* obvious (bubble area, cell glyph) | ConvNeXt; ICLR-26 outstanding Fig. 6 | fig_grid panel (a) already does it |
| **sequential ramp** for an ordered series; categorical hue only when unordered | Platonic (dino small→giant); Mirage (str len 1–5); KnowKnow (params, colourbar) | our series are unordered, so categorical is right |
| a named reference line carried in the legend | Emergent "Random"; KnowKnow "Perfect Calibration" | `figstyle.reference_line` exists; not needed in these panels |

### Counterexamples worth keeping

Being a strong paper does not make the figures strong, and two of the set are useful as things not to do.

- **Do ViTs See Like CNNs (NeurIPS 2021)**: legends carry raw variable names (`encoder_block0`,
  `block1unit1`), panel titles wrap to two lines, and one panel title is **clipped at the frame edge**
  (`R152` cut off). Never put an identifier in a legend where a phrase will fit; always check the render
  for clipping.
- **Are Emergent Abilities a Mirage (NeurIPS 2023 outstanding)**: the same legend is drawn **six times**,
  boxed, at roughly a quarter of each panel's width, and the panels carry no titles at all. A shared key
  drawn once is worth more than a house style.

### The one genuine disagreement in the field

**Grid or no grid.** The pasted CVPR reference uses a full box over a pale grid; Emergent, ConvNeXt and
Platonic use no grid or a barely-visible horizontal one. Both are defensible and the deciding question is
whether the reader must read values off the axis. Ours must: the whole claim is whether a blue point sits
above or below a violet point in the same column, and whether that ordering survives across panels. We
keep a **horizontal-only** grid at `#e4e6e9` — the middle position, and the vertical rules are dropped
because they crosshatch a zigzag series.

---

## Round 6 — the forest plot, deleted

A standalone dot-and-interval plot of five models × two quantities was built, and then removed. Two
reasons, in order of importance.

**Nobody in this literature draws one.** Searching the reference set for how uncertainty is presented:
Meng et al. (ROME) is the only paper of the sixteen that quantifies it seriously, and they put 95%
intervals as **shaded bands on curves they already had** (their Fig. 7/9), not as a figure of their own.
Everyone else puts it in a table as $\pm$, on error bars attached to an existing bar or scatter, or not
at all. The forest plot is a medical meta-analysis form; imported here without its conventions — a
numeric table beside it, a vertical null line, no floating text — it reads as something generated rather
than designed.

**The specific tells, worth remembering.** Coloured series names floating in the top-left of the axes,
which is a legend refusing to admit it is a legend. Interval end-caps hand-drawn with a second `plot`
call and NaN separators. An annotation auto-placed into a collision with the last row and the axis. Two
rows whose bars ran off the left spine. Any one of these is a bug; together they are a form nobody chose.

**Where the uncertainty went instead.** A note row in Table 1 giving the half-width of a 95% interval on
a single probe estimate, and a single error bar on the depth figure at the connector, which is the locus
the paper reports. A band along the whole depth curve was tried first and discarded: at $\pm0.033$ it is
comparable to the $T-F$ slab it sits on, so two orange fills with different meanings merged into one.

**The number this exercise fixed.** The first version computed the interval as median(high endpoint) −
median(low endpoint) = 0.047. Those medians are over different findings, so it is not the width of any
interval. The per-estimate median width is 0.066, half-width 0.033 — which happens to equal the whole
spread of $T$ across the five models, 0.033. The claim is therefore stronger than first written: any two
of the five probe estimates have overlapping intervals. `src/verify_draft.py` caught this and now checks
both the half-width and the spread expressed in units of it.

---

## Round 7 — Figures 3, 4 and 5 reconceived, and an overlap checker

Restyling had run out of road. The three figures were rebuilt around a different question — *what claim
does this panel make, and what shape makes that claim visible* — rather than around how they looked.

### `src/figcheck.py`

Overlapping labels were the recurring defect across every round, and eyeballing a 400-dpi render does not
catch a collision that is fatal at 5.5 inches. It is now arithmetic, run on the drawn figure before every
save:

| check | what it catches |
|---|---|
| text vs text | two labels whose ink rectangles intersect |
| **text vs data** | a label placed in what looked like empty space and then buried when the data changed |
| text outside its axes / the canvas | silently clipped at save |
| legend over data | segments sampled, not only vertices — a straight line through a legend has no vertex inside it |
| tick crowding | adjacent tick labels closer than 1.5 pt, which prints as a smear |

On the first run it found, in figures that had already been reviewed by eye: the nine rotated finding
labels in the heatmap **overlapping by 4–5 pt**; tick labels 0.7 pt apart in the profile figure; and an
AUROC annotation sitting on 39 histogram bars. None had been noticed.

### What each figure became

| was | is | why |
|---|---|---|
| five narrow panels of raw AUROC vs depth, half of each empty because AUROC must span 0.5–1.0 while the data occupies 0.65–0.78, and one x tick | **utilisation vs depth, one panel, five curves, zero drawn** | the section claims something about the *derived* quantity, so plot that. The claim is now a shape: no curve crosses zero, 0 sign changes in 192 model-by-locus points |
| scatter of gap-before against gap-after on the identity line | **one row per cell, dot to dot, sorted** | the claim is about *movement*; a diagonal scatter makes the reader verify twenty points one at a time, twenty near-zero bars say it at once |
| a panel showing a non-significant correlation ($r=0.65$, $p=0.06$, $n=9$) | **deleted** | an association we already downgraded in the text does not belong in a main figure; it hands a reviewer the objection |
| histogram of a log ratio | **paired log-log scatter with the identity line** | the histogram threw away the pairing — each cell has a matched random control, so a paired comparison should be drawn as pairs |
| the dose-response, buried in an appendix | **promoted to Figure 4a** | it is the only panel in the paper with ground truth, and a quantity plotted against a dose is evidence where the same quantity at one dose is a coincidence |

### The cost, paid honestly

The rebuilt figures are taller, and the main body went to ten pages. Reclaimed by moving the depth figure
to the appendix — it answers a methodological objection about locus choice, which is what an appendix is
for, and the body now states the number in one sentence — and by folding the Implications section into
the Conclusion, its full text already being in Appendix F. No result, number or claim was dropped.
