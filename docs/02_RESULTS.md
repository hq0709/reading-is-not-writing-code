# Results ledger

Every number the paper stands on, the experiment that produced it, and the control that makes it
interpretable. Rewritten 2026-08-29 after an independent Codex review and my own re-checks retired three
claims and corrected four numbers. Retired claims are kept below, with why, because the temptation to make
them is what the paper is about.

---

## The thesis

Not "the probing inference chain fails at every link". That was the framing until the review, and two of
the links were not supported. What the evidence carries is narrower and sharper:

> **What probing measures is largely not what the model uses, and a large part of it is not about the
> model at all.**

Two legs and a consequence.

| leg | claim | evidence |
|-----|-------|----------|
| 1 | the model's own output uses a small share of what training adds | utilisation, with an untrained floor |
| 2 | much of what a probe reports needs no training at all | an untrained network decodes view position at 0.997 |
| ⇒ | probing cannot rank models | probe AUROC spans 0.033, behaviour spans 0.206 |

Constructive half: the probe normal is not the causal direction, and a direction fitted on behaviour is
closer to the one a real image change moves along.

---

## D0. Utilisation  *(complete for LLaVA-Med; two more untrained floors in flight)*

### The measurement

Three numbers on the same held-out rows and the same labels.

    floor      a probe on the SAME architecture with randomly initialised weights
    trained    the same probe on the trained model
    behaviour  the AUROC of the model's own P(yes) for "is there a c?"

    utilisation = (behaviour - floor) / (trained - floor)

Zero when the answer is no better than an untrained network's readout. One when the answer is as good as
any linear readout of the model's own activations. The floor is what makes the comparison well-posed: a
probe fitted on 18,212 labelled radiographs beats a zero-shot prompt for reasons that have nothing to do
with what the model learned, and the floor carries exactly that advantage in both numerator and
denominator.

### Two objections, both closed

**"It is just a bad prompt."** Four readouts on identical images: the original yes/no question, a different
phrasing, the log-probability of "Yes, there is a *c*." against "No, there is no *c*.", and a
negation-symmetrised score. Across 20 model-by-finding cells the best of the four closes a median of
**7%** of the gap, moving it from 0.177 to 0.170; the largest gain in any cell is 0.075; in **7 of 20**
cells the original question is already the best. On Lingshu the best gain is exactly 0.000 on all four
findings, because there is no gap to close.

**"It is a calibration artefact."** AUROC is invariant to any strictly monotone transform of the score, so
a constant tendency to answer yes cannot lower it. LLaVA-Med answering yes on 100% of radiographs is a
statement about its threshold; its AUROC is a statement about its ranking.

### Result: four models, nine findings, connector locus fixed in advance

`utilisation = (best answer - untrained floor) / (trained probe - untrained floor)`, where the untrained
floor is the same probe on the same architecture with randomly initialised weights, and the answer is the
best of the four readouts. Lingshu and Qwen share an architecture and therefore share a floor; LLaVA-Med
and LLaVA-1.5 share the other.

| model | domain | median utilisation | answer below the untrained floor |
|-------|--------|-------------------:|---------------------------------:|
| **Lingshu-7B** | medical | **99%** | **0 / 9** |
| Qwen2.5-VL-7B | general | **-52%** | 7 / 9 |
| LLaVA-Med-7B | medical | **-57%** | 8 / 9 |
| LLaVA-1.5-7B | general | **-72%** | 7 / 9 |

Over 33 measurable cells the median utilisation is **-31%**, and in **22 of 36** cells the model's own
zero-shot answer separates the labels *worse than a supervised linear probe on an untrained network of the
same architecture*.

**Lingshu and Qwen are the same architecture and share the same untrained floor.** One reaches 99% and the
other -52%. The entire difference is what training did, and it is invisible to a probe: their probe AUROCs
at the connector differ by less than 0.01 on six of nine findings.

That contrast is also what rules out an instrument failure. A measure that returned a negative number for
every model would be suspect. This one returns 99% for one model and negative values for three.

Per finding on the two matched-architecture medical models:

| finding | floor | Lingshu trained / answer / util | LLaVA-Med trained / answer / util |
|---------|------:|--------------------------------:|----------------------------------:|
| Pneumothorax | 0.612 / 0.632 | 0.857 / 0.866 / **104%** | 0.783 / 0.482 / **-100%** |
| Effusion | 0.660 / 0.647 | 0.823 / 0.856 / **120%** | 0.792 / 0.661 / +9% |
| Atelectasis | 0.621 / 0.613 | 0.787 / 0.726 / +63% | 0.736 / 0.577 / -30% |
| Mass | 0.546 / 0.540 | 0.697 / 0.695 / +99% | 0.687 / 0.495 / -31% |
| Cardiomegaly | 0.706 / 0.667 | 0.891 / 0.708 / +1% | 0.741 / 0.650 / -24% |

Figures: `figures/utilisation.png`, `figures/utilisation_by_model.png`.

---

## Leg 2. A large part of "decodability" needs no training

The untrained network is not a blank slate. It decodes the acquisition variable essentially perfectly.

| | view position | sex |
|---|---:|---:|
| randomly initialised LLaVA-Med | **0.9974** | 0.828 |
| trained LLaVA-Med | 0.9992 | 0.989 |

And the height of each finding's untrained floor tracks how strongly that finding's label correlates with
view position, at **r = +0.612** across the nine findings:

| finding | untrained floor | corr(label, view position) |
|---------|----------------:|---------------------------:|
| Edema | **0.793** | **+0.164** |
| Infiltration | 0.669 | +0.157 |
| Consolidation | 0.673 | +0.135 |
| Effusion | 0.674 | +0.033 |
| Mass | **0.568** | -0.012 |
| Nodule | 0.608 | -0.075 |

Oedema reads at 0.793 from a network that was never trained, and training adds 0.039. Portable AP films are
taken of sicker patients, view position is free to decode, and the label correlates with it.

An untrained Qwen-architecture network shows the same picture: view position at **0.996**, and a
*positive* control-task selectivity of **+0.174 for cardiomegaly** and **+0.206** for oedema, higher than
most trained models achieve. Cardiomegaly is the size of a shape, a low-level image statistic, and it does
not correlate with view position (r = -0.020), so the randomised control task does not absorb it.

**Even selectivity, the strongest control the field has, does not separate what a model learned from what
its input statistics provide.** Only the untrained-network floor does. A study that reports "the model
encodes this finding" without one cannot distinguish that from "this finding is a low-level image
statistic, or correlates with how the image was taken". The survey behind this project found such a
floor in 5 of 121 applicable studies and a randomised control task in 0 of 135.

---

## D3. Planted ground truth  *(complete, 15 cells, 9 pass the gate)*

Plant a smooth opacity of known intensity in radiographs the model reads as normal, at the anatomical site
and at a control site outside the thorax, and compare the activation displacement to each candidate
direction.

Gate: monotone dose response, rise above 0.02, and more than three times the location control. **9 of 15
cells pass.** The location control separates by factors of 4.4 to 754. The six failures are LLaVA-Med on
all three findings, whose answer is pinned above 0.92 with no headroom, plus three non-monotone cells.

| candidate direction | mean abs cosine with the displacement the lesion actually causes |
|---------------------|----------------------------------------------------------------:|
| probe normal | **0.0041** |
| random vector | 0.0126 |
| behaviour-fitted direction | **0.0442** |

**The probe normal is three times less aligned with the true causal displacement than a random vector.** It
beats a random vector in only 46 of 270 locus-by-dose rows. The behaviour-fitted direction beats the probe
normal in 71 of 72.

---

## D2. Read directions are not write directions  *(complete, 50 cells)*

`v_probe` is fitted on labels and never sees behaviour. `v_cad` is fitted on behaviour under a
concept-specificity penalty and never sees a label.

| statement | value | tautological? |
|-----------|------:|---------------|
| `v_cad` steers more than `v_probe` | 45 / 50 | **yes, it is optimised for this** |
| `v_probe` steers no better than a random vector | **26 / 50** | no |
| median abs cos(`v_probe`, `v_cad`) | **0.0135** | no |
| median abs cos(`v_probe`, random) | 0.0091 | reference |
| `v_cad` reads worse than `v_probe` | 47 / 50 | no, the objective never mentions decoding |
| specificity ratio, penalty on against off (paired, connector) | 11.3 vs 1.9 | no |

---

## P2. A second modality  *(complete, 12,012 dermoscopy images, 4,109 patients)*

| modality | median selectivity | best finding | strongest acquisition nuisance | nuisance outranks every finding |
|----------|-------------------:|-------------:|-------------------------------:|--------------------------------:|
| chest radiograph | +0.050 | 0.897 | view position **0.999** | **5 / 5 models** |
| dermoscopy | +0.308 | 0.973 | anatomic site 0.947 | **1 / 5 models** |

Not a prevalence artefact: in dermoscopy the rarest finding has the highest selectivity (squamous cell
carcinoma, 3.4% prevalence, n=410, +0.453) against Nevus at 63.5% (+0.361), while in radiographs the
comparably rare oedema (2.8%) reaches +0.268 and Nodule is negative on all five models.

Caveat to state: two dermoscopy extractions (Lingshu, Qwen) completed 8,684 of 12,012 images with
diagnosis-dependent missingness, and are being re-extracted.

---

## Downgraded to descriptive

| claim | why |
|-------|-----|
| decodability ranks steerability across loci | median Spearman rho -0.245, negative in 11 of 14 cells, pooled Fisher-z p = 0.050. Raw AUROC against steerability is p = 0.74. Borderline at 12 loci per cell |
| causal separability rises with depth | strictly non-decreasing in **3 of 10** curves, not ten |

## Retired, with the reason

| claim | why it was retired |
|-------|--------------------|
| "the peak of decodability is never the peak of steerability, 15/15" | with *k* loci and two independent argmaxes this holds by chance with probability (1-1/k)^15, which is 0.27 at k=12 and 0.81 at k=70. It is what the null predicts |
| "29 selective intervention effects, 14 sign-inverted" | a leave-one-random-out audit puts the null pass rate at 8.21%, or 29.6 expected rows out of 360, against 29 observed |
| "separability of 1.0 means no concept-specific direction can exist" | the code averaged the magnitude of a **signed** cosine. The per-step sequence at the readout is [-1, +1, -1, -1, +1]; +1 is conflict and -1 is synergy, and averaging magnitudes merged them. Near-one there is also an architectural necessity, since the only computation downstream is a norm and a yes-minus-no readout shared by every question |

---

## Design faults found and fixed

| fault | what it would have shown |
|-------|--------------------------|
| control task over 2 input types | selectivity -0.19, "probes are anti-informative" |
| `alpha / sqrt(D)` scaling | "no locus is steerable", from a sweep never exceeding 13% of ‖h‖ |
| magnitude scaled by the pooled norm, applied per token | cross-locus comparisons confounded with depth (ratio 0.42x to 1.64x) |
| a structurally severed locus in the steering set | a dead locus scored as "not steerable" |
| Adam step of norm 3.0 on a unit vector | specificity weights 3.0 and 0.0 gave byte-identical runs |
| `gradient_checkpointing_enable()` under `.eval()` | a silent no-op; three jobs died reporting checkpointing was on |
| `h.detach()` inside a checkpointed block | every D2 job died at the second locus |
| images screened by dataset label, not model baseline | "LLaVA-Med's representation is unsteerable", when it had 0.033 of headroom |
| D0 joined a 1000-row behavioural score to a 5343-row probe score | a gap inflated by comparing different samples |
| all activations cached under the effusion prompt | answer-position probes for the other eight findings read the wrong prompt's activations. Visual, connector and vision-tower loci are prompt-independent to 0.000000 and unaffected |
| the modality analyser read only `site_Trunk` | dermoscopy nuisance dominance reported as 0/5 instead of 1/5 |
| the probe direction refitted 12x per locus in D3 | three hours of arithmetic already done |
