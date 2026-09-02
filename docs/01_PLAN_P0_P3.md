# Plan: controlled availability, use and semantic anchoring

## Thesis

Concept Flow separates four questions that a probe curve often compresses into one:

1. What can a matched linear readout recover?
2. What did training add beyond the same untrained architecture?
3. Which internal changes alter the model's behaviour specifically?
4. Which direction follows the representation change caused by the clinical finding itself?

The paper answers them with an untrained floor, a controlled depth profile, matched interventions and a
planted image-space anchor.

## Work packages

### P0: calibrated intervention geometry

Run: steer at registered visual and answer-position loci using a unit direction multiplied by each
token's activation norm.

Observation motivating the calibration:

| Lingshu-7B locus | per-token median | per-token mean | p95 | pooled median |
|---|---:|---:|---:|---:|
| `vis.last` | 2,171 | 5,261 | 18,317 | 5,188 |
| `connector` | 63.7 | 65.3 | — | 38.8 |
| `llm.L13.vis` | 109.0 | 110.6 | — | 84.1 |
| `llm.L27.vis` | 328.5 | 353.5 | — | 224.1 |

The pooled-to-token ratio spans 0.42x to 1.64x across stages, and Qwen-family vision-tower token norms
are strongly right-skewed. Token-relative scaling therefore defines alpha consistently across model
stages and architectures.

Gate decision: a dose-response comparison is admissible when the same alpha denotes the same relative
token perturbation at every locus.

Next: apply the registered grid with both signs and matched control directions.

### P0b: forward-path and structural controls

Run: verify every intervention locus by measuring its downstream logit effect.

Observed on Lingshu-7B with a random unit direction at alpha 1.0:

| Locus | maximum absolute logit change |
|---|---:|
| `llm.L26.vis` | 1.03 |
| `llm.L27.vis` | 0.00000 |
| `llm.L27.ans` | 11.11 |

The final-layer visual position is a structural zero because no later attention layer can carry the
edited visual token into the answer. It remains in the design as an internal negative control. For
LLaVA models, `vis.last` is the consumed block selected by `vision_feature_layer`, currently
`encoder.layers.22` for `vision_feature_layer=-2`.

Gate decision: an active locus must change downstream logits; the structural-zero control must remain at
zero.

Next: complete the focused hook preflight before any immutable scientific run.

### P1: behaviour-fitted causal direction

Run: fit a unit direction `v_cad` at each locus by maximising target behaviour while penalising changes
to other concepts:

`mean target effect - lambda * mean absolute off-target effect`.

`v_probe` uses labels and never sees behaviour. `v_cad` uses behaviour and never sees labels. The
specificity-off ablation (`lambda=0`) measures the generic-answer component.

Report on held-out patients:

- target effect and specificity cost;
- AUROC of the one-dimensional projection onto the direction;
- cosine with the probe normal;
- the frontier along the normalised interpolation from `v_probe` to `v_cad`.

Early connector evidence on LLaVA-1.5-7B established that the instrument can separate reading and
writing: the probe direction had effect 0.031 and AUROC 0.760, while the behaviour-fitted direction had
effect 0.163, AUROC 0.540 and cosine -0.005 to the probe. These eight-image measurements are smoke
evidence; the paper-level estimate comes from the registered patient-held-out gate.

Gate decision: the behaviour-fitted direction must generalise to held-out patients and exceed matched
random and sham controls at a registered dose.

Next: run the full patient-held-out grid after P0 passes.

### P1b: causal separability profile

Run: measure the cosine between the target-effect gradient and off-target gradients at each locus.

Observation: Lingshu-7B connector gradients ranged from 0.24 to 0.60, while the final-layer answer
position reached absolute cosine 1.0. The final answer position shares a yes-minus-no readout direction
across questions, making it the registered degenerate endpoint of the profile.

Gate decision: report signed gradient relations and their uncertainty; preserve positive and negative
alignment rather than averaging magnitudes.

Next: compare the separability profile with held-out causal-direction specificity.

### P2: cross-modality replication

Run: apply the same patient-level design to dermoscopy.

The usable ISIC subset contains 12,012 images from 4,109 patients: 8,426 train, 1,283 validation and
2,303 test. Six diagnoses have at least 300 examples: Nevus 7,622; Melanoma 2,083; basal cell carcinoma
1,003; seborrheic keratosis 581; squamous cell carcinoma 410; actinic keratosis 313. Four anatomical
sites serve as the acquisition nuisance, and the metadata yields 67 control-task types.

Gate decision: patient partitions remain disjoint, at least eight control types recur across splits and
the clinical result is evaluated alongside acquisition nuisances.

Next: replicate the primary controlled measurements across all five models.

### P3: planted ground truth

Run: insert a synthetic finding of known location and intensity into radiographs that the model reads as
normal, then measure the representation displacement and the model's response.

Observation target: compare the image-space dose response with steering along `v_probe` and `v_cad` on
the same axes, using an outside-thorax placement control.

Gate decision: the image-space response is monotone, rises above its registered minimum and exceeds the
location control before representation-space equivalence is assessed.

Next: identify which learned direction best follows the displacement and behavioural curve caused by the
finding.

## Execution order

| Priority | Run | Gate | Next authorised experiment |
|---|---|---|---|
| 1 | exact LLaVA `vis.last` hook preflight | consumed block changes logits | immutable Effusion probe |
| 2 | Effusion probe, 2,000 patient bootstraps, control seeds 0–19 | interval and control spread complete | matched intervention grid |
| 3 | token-relative intervention, 20 random directions, 200 held-out images | concept effect passes registered controls | full primary-model locus profile |
| 4 | five-model and matched-pair replication | same protocol complete across models | dermoscopy replication |
| 5 | planted lesion dose response | monotonicity and location control pass | image/representation comparison |

Design corrections and failed runs remain in the evidence ledger in `docs/10_RESULTS.md`; this plan
contains the current protocol and next gate only.
