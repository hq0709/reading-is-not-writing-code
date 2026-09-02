# Current scientific findings

This file states the current result set. Detailed measurements, run provenance and failed-run evidence
are recorded in `docs/10_RESULTS.md`.

---

## The thesis

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

## D0. Utilisation

### The measurement

Three numbers on the same held-out rows and the same labels.

    floor      a probe on the SAME architecture with randomly initialised weights
    trained    the same probe on the trained model
    behaviour  the AUROC of the model's own P(yes) for "is there a c?"

    utilisation = (behaviour - floor) / (trained - floor)

Zero when the answer is no better than an untrained network's readout. One when the answer is as good as
any linear readout of the model's own activations. The floor makes the comparison well-posed: a probe
fitted on 18,212 labelled radiographs has training-independent advantages over a zero-shot prompt, and
the floor carries that advantage in both numerator and denominator.

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

The 99% result for Lingshu and negative values for the other three models show that the measure
discriminates among models rather than imposing one common outcome.

Per finding on the two matched-architecture medical models:

| finding | floor | Lingshu trained / answer / util | LLaVA-Med trained / answer / util |
|---------|------:|--------------------------------:|----------------------------------:|
| Pneumothorax | 0.612 / 0.632 | 0.857 / 0.866 / **104%** | 0.783 / 0.482 / **-100%** |
| Effusion | 0.660 / 0.647 | 0.823 / 0.856 / **120%** | 0.792 / 0.661 / +9% |
| Atelectasis | 0.621 / 0.613 | 0.787 / 0.726 / +63% | 0.736 / 0.577 / -30% |
| Mass | 0.546 / 0.540 | 0.697 / 0.695 / +99% | 0.687 / 0.495 / -31% |
| Cardiomegaly | 0.706 / 0.667 | 0.891 / 0.708 / +1% | 0.741 / 0.650 / -24% |

Figures: `figures/utilisation.png`, `figures/utilisation_by_model.png`.

### Readout robustness and calibration

Four readouts were evaluated on identical images: the original yes/no question, a different phrasing,
the log-probability of "Yes, there is a *c*." against "No, there is no *c*.", and a
negation-symmetrised score. Across 20 model-by-finding cells the best of the four closes a median of
**7%** of the gap, moving it from 0.177 to 0.170; the largest gain in any cell is 0.075; in **7 of 20**
cells the original question is already the best. On Lingshu the best gain is exactly 0.000 on all four
findings, because there is no gap to close.

AUROC makes the calibration comparison threshold-independent: it is invariant to any strictly monotone
transform of the score. LLaVA-Med answering yes on 100% of radiographs describes its threshold, while
its AUROC describes its ranking.

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

**Selectivity alone does not separate what training added from structure already available in the input
and architecture.** The untrained-network floor measures that distinction directly. The survey behind
this project found such a floor in 5 of 121 applicable studies and a randomised control task in 0 of 135.

---

## D3. Planted ground truth

Plant a smooth opacity of known intensity in radiographs the model reads as normal, at the anatomical site
and at a control site outside the thorax, and compare the activation displacement to each candidate
direction.

Gate: monotone dose response, rise above 0.02, and more than three times the location control. **9 of 15
cells pass.** The location control separates by factors of 4.4 to 754.

| candidate direction | mean abs cosine with the displacement the lesion actually causes |
|---------------------|----------------------------------------------------------------:|
| probe normal | **0.0041** |
| random vector | 0.0126 |
| behaviour-fitted direction | **0.0442** |

**The probe normal is three times less aligned with the true causal displacement than a random vector.** It
beats a random vector in only 46 of 270 locus-by-dose rows. The behaviour-fitted direction beats the probe
normal in 71 of 72.

---

## D2. Read directions and write directions

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

## P2. Cross-modality result

| modality | median selectivity | best finding | strongest acquisition nuisance | nuisance outranks every finding |
|----------|-------------------:|-------------:|-------------------------------:|--------------------------------:|
| chest radiograph | +0.050 | 0.897 | view position **0.999** | **5 / 5 models** |
| dermoscopy | +0.308 | 0.973 | anatomic site 0.947 | **1 / 5 models** |

The prevalence comparison points in the same direction: in dermoscopy the rarest finding has the highest selectivity (squamous cell
carcinoma, 3.4% prevalence, n=410, +0.453) against Nevus at 63.5% (+0.361), while in radiographs the
comparably rare oedema (2.8%) reaches +0.268 and Nodule is negative on all five models.

## Evidence decisions

| Observation | Decision |
|-------|-----|
| median Spearman rho between decodability and steerability is -0.245, negative in 11 of 14 cells; pooled Fisher-z p=0.050, while raw AUROC against steerability has p=0.74 | descriptive association; the current locus count does not support a ranking claim |
| causal separability is strictly non-decreasing in 3 of 10 curves | descriptive profile rather than a depth trend |

## Next gate

Run: `llava-effusion-vislast-reltoken` under the immutable server protocol.

Observation required: the exact consumed `encoder.layers.22` hook, 2,000 patient bootstraps, control seeds
0–19 and the token-relative intervention grid on 200 held-out images with 20 random directions.

Gate decision: the corrected immutable implementation must obtain its exact hook receipt before the
full probe and intervention run is dispatched.

Next: determine whether Effusion at the consumed LLaVA visual locus passes the registered probe and
intervention controls. Detailed provenance and all failure records remain in `docs/10_RESULTS.md`.
