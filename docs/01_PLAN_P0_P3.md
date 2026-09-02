# Plan: from a diagnostic result to a landmark paper

Written 2026-08-27, after the first complete intervention sweep. This supersedes the framing in
`00_PAPER_PLAN.md` §0 and records the four work packages that carry the paper.

---

## 1. The thesis

The field's interpretability practice rests on a chain of four inferences:

> a probe decodes concept *c* at locus λ → *c* is represented at λ → λ is where *c* lives → adding the
> probe normal at λ manipulates *c*

Every link is assumed, none is tested. This paper tests all four, and each one fails.

| link | what fails | measurement in hand |
|------|------------|---------------------|
| ① decodable ⇒ represented | most decodability is probe capacity | surface AUROC median 0.75, control task 0.69, **selectivity +0.084**; Nodule negative on all five models |
| ② represented ⇒ clinical content | the most decodable variable is the scanner, not the patient | view position **0.997–0.999 at every locus**, clinical concepts at best 0.90 |
| ③ decodable here ⇒ intervene here | the two peaks do not coincide | **15/15** model × concept pairs over the full locus set |
| ④ probe normal ⇒ causal direction | reading and writing are different directions | sign inversion in 4 of 6 significant effects; **D2 is the decisive test** |

**The positive control that makes the negatives credible.** A pure null invites the reply that the
instrument is too blunt. It is not: the same instrument cleanly separates an architecture-matched medical
and general pair. Lingshu-7B peaks at `llm.L27` for all three concepts with selectivity +0.144 to +0.180;
Qwen2.5-VL-7B, identical in architecture, scatters across `llm.L7` / `llm.L14` / `vis.last` at +0.058 to
+0.080. The instrument detects medical pretraining. It does not detect the thing the field assumes.

Working title: **Decodable Is Not Used: Where the Probing Inference Breaks in Medical Vision-Language
Models.**

---

## 2. Work packages

### P0 — remove the magnitude confound  *(done, rerunning)*

The perturbation was scaled by the norm of the **pooled** activation while being added to the **per-token**
activations. Those differ by a factor that varies with depth, which is the axis the paper compares along.
Calibration on Lingshu-7B:

| locus | per-token median | per-token mean | p95 | pooled median |
|-------|-----------------:|---------------:|----:|--------------:|
| vis.last | 2171 | 5261 | 18317 | 5188 |
| connector | 63.7 | 65.3 | — | 38.8 |
| llm.L13.vis | 109.0 | 110.6 | — | 84.1 |
| llm.L27.vis | 328.5 | 353.5 | — | 224.1 |

Two consequences. The pooled norm runs from 0.42x to 1.64x the per-token norm across stages, so a nominal
"50% of ‖h‖" was a different physical perturbation at each locus. And the vision tower's patch norms are
heavily right-skewed (mean 2.4x median, p95 8.6x median on Qwen-family towers, but only 1.1x on LLaVA's
CLIP tower), so no single scalar magnitude can be right for both a skewed and an unskewed stage.

Fix: steering applies `alpha · ‖h_t‖ · v` **token by token**, so alpha is the same relative perturbation
everywhere. Superseded runs archived under `runs/_pooled_scale_v1/` with `WHY.md`.

Side finding worth reporting: the pooled representation that nearly every probing study uses is dominated
by a handful of outlier patches in Qwen-family vision towers, and the degree of domination is
architecture-dependent. Cross-architecture comparisons of pooled features inherit that distortion.

### P0b — the structurally inert locus  *(found and fixed today)*

Steering the **visual** positions at layer L can only reach the answer through the attention of layers
L+1…n−1. At the final layer there are none, so the intervention is causally inert by construction.
Measured on Lingshu-7B with a random unit direction at alpha = 1.0:

| locus | max abs change in logits |
|-------|-------------------------:|
| llm.L26.vis | 1.03 |
| **llm.L27.vis** | **0.00000** |
| llm.L27.ans | 11.11 |

The first sweep put this dead locus among only seven, and omitted the answer-position channel entirely,
even though it carries effects five to ten times larger. The locus set is now twelve:

    vis.last, connector,
    llm.L{0, n/4, n/2, 3n/4, n-2}.vis,
    llm.L{n/4, n/2, 3n/4, n-1}.ans,
    llm.L{n-1}.vis          <- kept deliberately as a structural zero control

The last entry stays in the design. It is a place where the causal path is severed by construction, so a
non-zero reading there would indict the measurement rather than describe the model. Almost no paper in this
literature carries an internal control of that kind.

### P1 — the causally-optimised direction (D2)  *(implemented, `src/cad.py`)*

The constructive half. At a locus, alongside the probe normal `v_probe`, fit a unit vector `v_cad` by
gradient descent on behaviour:

    maximise  mean_i [ m_c(x_i; v) − m_c(x_i; 0) ]                             move question c
    minus  L · mean_{c'≠c} mean_i | m_{c'}(x_i; v) − m_{c'}(x_i; 0) |          leave the others alone

where m is the yes-minus-no logit margin. **The objective uses no labels.** `v_probe` is built from labels
and never sees behaviour; `v_cad` is built from behaviour and never sees labels. That is the cleanest
possible contrast between reading and writing.

The specificity penalty is load-bearing: without it the optimiser converges on a generic "say yes"
direction, which would move question c strongly while meaning nothing about c. `--lambda-spec 0` is run as
an ablation, and the survey's finding that 5 of 8 targeted medical interventions reported no
concept-specificity control is exactly why both are shown.

Reported on held-out test images: effect, specificity cost, the AUROC of the 1-D projection `x·v` (how well
the direction **reads**), cosine to the probe normal, and the frontier along the interpolation
`v(t) = normalise((1−t)·v_probe + t·v_cad)`.

Prediction: `v_cad` steers far better and decodes far worse, and the frontier is a genuine trade-off. If so,
"read directions are not write directions" stops being an observation about loci and becomes a property of
the representation.

### P1b — causal separability, a measurement that fell out of P1  *(new)*

Fitting `v_cad` requires the gradient of the concept's effect and the gradient of the other concepts'
effects. The cosine between them is a quantity in its own right, and it answers a question no probe can:

> at this locus, can the concept's causal pathway be separated from another concept's at all?

At 1.0 the two objectives are the same objective, no specificity weight can pull them apart, and no
concept-specific write direction exists there regardless of what any fit reports. Measured on Lingshu-7B:

| locus | cos(grad_gain, grad_cost) | interpretation |
|-------|--------------------------:|----------------|
| connector | 0.24 to 0.60 | separable; a concept-specific direction exists |
| llm.L{n-1}.ans | **1.000** | one causal axis only |

The final-layer answer position is degenerate for a structural reason. The only computation downstream of
it is the final norm and the readout, and the yes-minus-no readout direction is the *same vector* for every
question. So every yes/no question there shares one gradient, and a direction that changes the answer about
effusion changes the answer about cardiomegaly by the identical mechanism. That locus is kept in the design
as the degenerate end of the profile: a positive control on the measure, since a separability instrument
that failed to report 1.0 there would be broken.

Early evidence that the instrument discriminates, from the first two D2 runs at the connector:

| model | cad effect | specificity cost | ratio | mean \|grad_align\| |
|-------|-----------:|-----------------:|------:|-------------------:|
| LLaVA-1.5-7B (general) | 0.38 | 0.018 | 21:1 | 0.29 to 0.50 |
| LLaVA-Med-7B (medical) | 0.024 | 0.008 | 3:1 | 0.73 to 0.83 |

and from the first full scoring, also LLaVA-1.5 at the connector:

| direction | effect | reads (AUROC) | cos to probe |
|-----------|-------:|--------------:|-------------:|
| probe | 0.031 | **0.760** | +1.000 |
| cad | **0.163** | 0.540 | −0.005 |
| random | 0.054 | 0.609 | 0.000 |
| sham | 0.019 | 0.543 | −0.022 |

The probe normal reads the concept well and moves it *less than a random vector does*. The fitted write
direction moves it three times better than random and reads at near chance. Their cosine is at the floor
for two unrelated vectors in 4096 dimensions. These are 8-image smoke numbers and the sweep will replace
them, but the shape is the thesis.

### P2 — a second modality  *(images downloaded, extraction running)*

One modality supports "we found this in chest radiographs". Two support "we found this in medical VLMs".

Dermoscopy from the ISIC archive, chosen because its metadata mirrors the chest-radiograph design almost
exactly: `anatom_site_1` stands in for view position as the acquisition nuisance, with `sex` and
`age_approx` completing the control-task type. It is maximally unlike a chest radiograph in every other
respect: colour, surface, lesion-centred framing. `acquisition.image_type` gives a second, stronger
acquisition nuisance that chest radiographs have no equivalent of.

Settled: `clinical.patient_id` is present in 88.6% of the 553,019 records, over 6,772 patients, so the
split is patient-level by the same hash and the same salt as the chest radiographs, and the same assertion
that no patient spans a split passes. The usable subset is **12,012 dermoscopic images over 4,109 patients**:
8,426 train, 1,283 val, 2,303 test, with six diagnoses at 300 or more examples (Nevus 7,622, Melanoma 2,083,
basal cell carcinoma 1,003, seborrheic keratosis 581, squamous cell carcinoma 410, actinic keratosis 313),
four anatomic sites as the acquisition nuisance, and 67 control-task types.

### P3 — planted ground truth (D3)  *(not started)*

With `v_cad` in hand, D3 changes role from a nice-to-have into the anchor. Plant synthetic lesions of known
location and known intensity into radiographs that the model reads as normal, and compare two dose-response
curves: image-space (the planted lesion grows) against representation-space (alpha grows along a direction).
Then ask which direction reproduces the image-space curve. The answer converts link ④ from "the probe
direction behaves oddly" into "here is what the correct direction looks like, and the probe normal is not
it".

---

## 3. Order and gates

| priority | package | gate to pass before it counts |
|---|---|---|
| P0 | per-token scaling, 12-locus sweep | structural-zero locus must read exactly 0.000 in every run |
| P1 | CAD | `v_cad` must generalise to held-out patients, or it is an adversarial perturbation, not a direction |
| P2 | dermoscopy | control task must be non-degenerate (≥8 types, selectivity not driven by the nuisance) |
| P3 | planted lesions | image-space dose-response must be monotone before any representation-space comparison is meaningful |

**Standing rule for this project.** When a result contradicts the intended contribution, interrogate the
experiment before weakening the claim. Two nulls on this project were artefacts of the design rather than
properties of the models: a control task over two input types that collapsed into decoding view position,
and a magnitude sweep that never exceeded 13% of the activation norm. A third, the structurally inert
locus, was caught today. Only after the design is adequate does a persistent null count as evidence, and
then it is reported plainly.
