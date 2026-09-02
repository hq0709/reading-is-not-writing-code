# Concept Flow: measurement and evidence ledger

This ledger records measurements, their evidence source and the gate decision they support. The current
paper-level findings are in `docs/02_RESULTS.md`; volatile run status belongs in `docs/RESEARCH_STATE.md`.

## 1. Current immutable input evidence

### NIH ChestX-ray14 trust bootstrap

Run: stage the official 12 image archives and label CSV, then compare their content against an independent
distribution and pinned label source.

Observation: the staged dataset contains 112,120 images and a 26,229-row patient-split research manifest.
The 9,239 SHA-1 pieces in Academic Torrents infohash
`557481faacd824c83fbf57dcf7b6da9383b3235a` cover 45,089,461,497 bytes and match the downloaded archive
content. The pinned Hugging Face label commit `a607951b44ac6c14bbce3a03e50eb6fcd657474e`
contains 112,120 rows byte-identical to the official label body; the header names `Patient Gender` and
`Patient Sex` differ.

Gate decision: `nih-chestxray14-trust-bootstrap` is `PASS`. Its terminal receipts are reused as the
dataset evidence:

- `/home/qingchan/data/concept-flow/state/nih-chestxray14-independent-verification-5342b4219127/receipt.json`
- `/home/qingchan/data/concept-flow/datasets/nih-chestxray14/asset-receipt.json`
- `/home/qingchan/data/concept-flow/datasets/nih-chestxray14/manifest.csv`

Next: the active scientific gate tests the exact LLaVA `encoder.layers.22` forward path. The data trust
chain is reopened only after concrete contamination evidence or a new terminal asset gate.

## 2. Exploratory measurement cohort

The measurements below come from the local exploratory cohort. They motivate the registered server gate;
their run directories predate the immutable dispatcher and do not substitute for the active gate receipt.

### Data

`data/manifest.csv` contains 26,229 NIH ChestX-ray14 images from 8,297 patients. The deterministic split
hashes `sha256("concept-flow-v1:" + patient_id)`:

| Split | Images | Patients |
|---|---:|---:|
| train | 18,212 | 5,783 |
| validation | 2,674 | 855 |
| test | 5,343 | 1,659 |

No patient crosses a split. Nine clinical labels are evaluated; view position and sex are nuisance
labels. Test positives range from 179 for Edema to 872 for Infiltration.

### Models and loci

| Key | Domain | Measured parameters | Vision blocks | LLM layers | Hidden | Visual tokens | Evidence |
|---|---|---:|---:|---:|---:|---:|---|
| `qwen7b` | general | 8.292B | 32 | 28 | 3,584 | 144 | `runs/hook_demo_qwen7b.json` |
| `lingshu7b` | medical | 8.29B | 32 | 28 | 3,584 | 144 | `runs/smoke_lingshu.log` |
| `llavamed7b` | medical | 7.57B | 24 | 32 | 4,096 | 576 | `runs/arch_verification.json` |
| `llava15_7b` | general | 7.063B | 24 | 32 | 4,096 | 576 | `runs/hook_demo_llava7b.json` |
| `internvl3_8b` | general | 7.94B | 24 | 28 | 3,584 | 256 | `runs/arch_verification.json` |

Lingshu/Qwen is the clean architecture-matched pair. LLaVA-Med/LLaVA-1.5 shares the framework but differs
in base LLM and by 0.51B parameters, so that pair is interpreted as a combined domain/base-model contrast.

The exploratory locus set contains mean-pooled vision blocks, `vis.last`, the connector, every language
layer at visual positions and every language layer at the answer position. The probe uses standardised
features, a 512-dimensional Gaussian projection and logistic regression (`C=1`, `max_iter=2000`, seed 0).
Mean pooling over 144, 256, or 576 visual tokens can dilute a local lesion signal; this cohort does not
include bounding-box or otherwise localised readout evidence.

### Probe evidence sources

| Source | Rows |
|---|---:|
| `runs/probe_lingshu7b/probe_results.csv` | 594 |
| `runs/probe_qwen7b/probe_results.csv` | 594 |
| `runs/probe_llavamed7b/probe_results.csv` | 648 |
| `runs/probe_llava15_7b/probe_results.csv` | 648 |
| `runs/probe_internvl3_8b/probe_results.csv` | 576 |

Each of the 3,060 cells records real-label AUROC, a within-train permutation fit, one fixed random label
per recurring input type, selectivity (`real-control`) and the two nuisance AUROCs. The input type is view
position × sex × age decade, yielding 39 types; 37 recur in both train and test.

### Aggregate evidence sources

| Measurement | Evidence source |
|---|---|
| D0 utilisation | `runs/utilisation_connector.csv` |
| D2 read/write directions | `runs/cad_summary_v1.csv` |
| D3 planted-anchor gate | `runs/plant_gate.csv` |
| D3 direction alignment | `runs/plant_alignment.csv` |
| cross-modality comparison | `runs/modality_comparison.csv` |

## 3. Controlled availability measurements

### Cross-model summary

Run: choose the highest-real-AUROC row per model and concept on the registered visual-position track
(`vis.last`, connector and `llm.*.vis`).

| Statistic over 45 peaks | Median | Mean | Min | Max |
|---|---:|---:|---:|---:|
| real AUROC | 0.7507 | 0.7578 | 0.6511 | 0.8966 |
| permutation AUROC | 0.5021 | 0.5039 | 0.4397 | 0.5579 |
| control AUROC | 0.6854 | 0.6847 | 0.5945 | 0.7716 |
| selectivity | **+0.0764** | +0.0732 | -0.1132 | +0.2388 |

Observation: seven of 45 peaks have selectivity at or below zero. Across all 3,060 cells, median real
AUROC is 0.7168, control 0.6707 and selectivity +0.0623; 21.6% of cells have selectivity at or below zero.
The permutation median is 0.5028, with 95.8% of cells in [0.45, 0.55].

Gate decision: the fitting pipeline passes its permutation check, while the control task removes most of
the surface AUROC. The single control draw does not support concept rankings; the active gate therefore
uses control seeds 0–19 and reports the mean and spread.

### Concept profile

| Concept | Positive-selectivity loci | Median selectivity |
|---|---:|---:|
| Edema | 340/340 | +0.1966 |
| Infiltration | 340/340 | +0.1010 |
| Consolidation | 340/340 | +0.0865 |
| Cardiomegaly | 331/340 | +0.0685 |
| Effusion | 340/340 | +0.0620 |
| Pneumothorax | 317/340 | +0.0483 |
| Atelectasis | 319/340 | +0.0291 |
| Mass | 72/340 | -0.0207 |
| Nodule | 0/340 | -0.1104 |

Observation: Nodule has median real AUROC 0.637 and median control 0.7475; Mass has median real 0.649 and
control 0.667. Because every model shared the same concept-specific control map, these are one-draw
observations rather than five independent control replications.

Gate decision: retain Mass and Nodule as hypotheses for the repeated-control run.

### Nuisance profile

Run: fit view-position and sex probes at the same loci and capacity.

| Representation | View position AUROC | Sex AUROC |
|---|---:|---:|
| randomly initialised LLaVA-Med | 0.9974 | 0.828 |
| trained LLaVA-Med | 0.9992 | 0.989 |

Across the five trained models and all loci, view-position AUROC spans 0.996415–0.999367. The untrained
finding floor correlates with label/view correlation at `r=+0.612` across nine findings. Untrained Qwen
reaches view-position AUROC 0.996 and positive selectivity +0.174 for Cardiomegaly and +0.206 for Edema.

Gate decision: acquisition structure is a dominant reference signal. Availability claims are paired with
the untrained floor and nuisance measurements.

## 4. Architecture-matched comparisons

### Lingshu-7B versus Qwen2.5-VL-7B

Run: compare the clean pair at matched loci and at each concept's peak.

Observation: Lingshu has higher peak AUROC and peak selectivity in 9/9 concepts. The median peak
differences are +0.047 AUROC and +0.050 selectivity. At matched loci, median AUROC differences grow from
+0.0319 at `vis.last`, to +0.0263 at the connector, +0.0334 at `llm.L0.vis`, +0.0657 at
`llm.L13.vis`, +0.0635 at `llm.L20.vis`, and +0.0790 at `llm.L27.vis`.

Gate decision: the exploratory data support a medical-domain advantage concentrated in later language
layers for one architecture-matched pair.

### LLaVA-Med-7B versus LLaVA-1.5-7B

Run: compare the shared-framework pair at matched loci and at each concept's peak.

Observation: median peak differences are +0.006 AUROC and +0.001 selectivity. Same-locus AUROC
differences are -0.0013 at `vis.last`, -0.0056 at the connector, -0.0036 at `llm.L0.vis`, +0.0058 at
`llm.L15.vis`, +0.0123 at `llm.L23.vis` and +0.0095 at `llm.L31.vis`.

Gate decision: this pair supplies a weak, confounded replication of the late-layer pattern; it does not
estimate a pure medical-training effect.

## 5. Utilisation and causal measurements

### D0 utilisation

Run: on identical held-out rows, compute `F` from a probe on the randomly initialised architecture, `T`
from the trained representation and `B` from the best of four behavioural readouts. Report
`(B-F)/(T-F)`.

| Model | Domain | Median utilisation | Behaviour below floor |
|---|---|---:|---:|
| Lingshu-7B | medical | **99%** | 0/9 |
| Qwen2.5-VL-7B | general | **-52%** | 7/9 |
| LLaVA-Med-7B | medical | **-57%** | 8/9 |
| LLaVA-1.5-7B | general | **-72%** | 7/9 |

Across 33 measurable cells, median utilisation is -31%; in 22/36 cells the model's zero-shot answer
ranks the labels below the same supervised probe on an untrained architecture. Four prompt/readout forms
close a median 7% of the gap, from 0.177 to 0.170, and AUROC removes constant calibration bias.

Gate decision: availability gained through training and behavioural use differ sharply across models;
Lingshu provides the positive model contrast.

### D2 read and write directions

Run: compare label-fitted `v_probe` with behaviour-fitted, specificity-penalised `v_cad` over 50 cells.

| Measurement | Value |
|---|---:|
| `v_cad` steers more than `v_probe` | 45/50 |
| `v_probe` steers no better than random | 26/50 |
| median abs cosine (`v_probe`, `v_cad`) | 0.0135 |
| median abs cosine (`v_probe`, random) | 0.0091 |
| `v_cad` reads worse than `v_probe` | 47/50 |
| specificity ratio, penalty on versus off at connector | 11.3 versus 1.9 |

Gate decision: the exploratory measurement supports a distinction between read and write directions. The
45/50 optimisation result is descriptive of the objective; the label-free readout and cosine comparisons
carry the independent evidence.

### D3 planted anchor

Run: plant a smooth opacity of known intensity at an anatomical site and at an outside-thorax control
site, then compare the resulting activation displacement with candidate directions.

Observation: 9/15 cells pass the registered monotonicity, minimum-rise and 3× location-control gate. The
location control separates by factors 4.4–754.

| Candidate direction | Mean absolute cosine with planted displacement |
|---|---:|
| probe normal | 0.0041 |
| random vector | 0.0126 |
| behaviour-fitted direction | 0.0442 |

The behaviour-fitted direction exceeds the probe normal in 71/72 rows; the probe normal exceeds a random
vector in 46/270 locus-dose rows.

Gate decision: for the passing cells, the behaviour-fitted direction is closer to the displacement caused
by the lesion than the probe normal.

## 6. Cross-modality measurement

| Modality | Median selectivity | Best finding AUROC | Strongest acquisition nuisance | Nuisance outranks every finding |
|---|---:|---:|---:|---:|
| chest radiograph | +0.050 | 0.897 | view position 0.999 | 5/5 models |
| dermoscopy | +0.308 | 0.973 | anatomic site 0.947 | 1/5 models |

The dermoscopy cohort contains 12,012 images from 4,109 patients. Squamous cell carcinoma has 3.4%
prevalence, `n=410`, and selectivity +0.453; Nevus has 63.5% prevalence and selectivity +0.361.

Gate decision: the acquisition-nuisance dominance observed in radiographs is not uniform across medical
image modalities. `DERM-MISSING-01` excludes incomplete Lingshu and Qwen extractions from cross-model
claims until their 12,012-image receipts exist.

## 7. Failure and evidence ledger

Failures are recorded once here and referenced by evidence id.

| Evidence id | Observation | Scientific consequence |
|---|---|---|
| `CTRL-TYPE-02` | two-type control equals view-position decoding; median selectivity -0.3029 for Lingshu and -0.3062 for Qwen | use recurring 39-type construction and repeated control seeds |
| `CTRL-DRAW-01` | one shared control draw explains 55.4% of cross-concept selectivity variation (`R²=0.554`; correlation -0.744) | concept rankings require repeated draws |
| `LOCUS-PATH-01` | historical LLaVA `vis.last` pointed to discarded layer 23 while the connector consumed layer 22; interventions were exact no-ops | active registry maps `vis.last` to `encoder.layers.22` and requires a forward-path hook test |
| `SCALE-REL-01` | `alpha/sqrt(D)` reached only 13.4% of the median activation norm | use token-relative scale |
| `SCALE-POOL-01` | pooled norm applied per token varied 0.42x–1.64x across stages | scale from each token's norm |
| `ARGMAX-NULL-01` | decodability and steerability peaks differ in 15/15 cells, an event with chance probability 0.27 at 12 loci | peak mismatch is not a result |
| `LOO-RANDOM-01` | 29 observed selective rows versus 29.6 expected under leave-one-random-out | count-based selectivity result rejected |
| `GRAD-SIGN-01` | absolute cosine merged signed gradient relations | retain signed gradient relations |
| `PROMPT-CACHE-01` | all answer-position activations used the Effusion prompt | use concept-specific prompts for language loci; visual and connector loci were invariant to 0.000000 |
| `D0-JOIN-01` | a 1,000-row behaviour score was joined to a 5,343-row probe score | compare identical row ids |
| `SCREEN-BASE-01` | label-based image screening left LLaVA-Med with 0.033 headroom | screen on model baseline |
| `DERM-MISSING-01` | two dermoscopy extractions stopped at 8,684/12,012 with diagnosis-dependent missingness | exclude until complete |
| `NIH-LABEL-01` | NIH labels are report-mined; Nodule and Mass are small, noise-sensitive targets | preserve label-noise uncertainty in concept interpretation |
| `CAD-STEP-01` | Adam step norm 3.0 made specificity weights 3.0 and 0.0 byte-identical | bound the optimisation step and verify the specificity ablation |
| `CKPT-EVAL-01` | checkpointing under `.eval()` was inactive; three jobs stopped | remove the inactive configuration |
| `CKPT-DETACH-01` | `h.detach()` broke the checkpointed gradient path at the second locus | preserve the gradient path |
| `DERM-SITE-01` | modality analysis read only `site_Trunk` | aggregate all registered anatomical-site nuisance columns |
| `D3-REFIT-01` | probe direction was refitted 12 times per locus | fit once and reuse the immutable direction artifact |
| `D3-GATE-01` | six of 15 planted-anchor cells miss the gate: three LLaVA-Med cells have baseline answer probability above 0.92 and three are non-monotone | anchored interpretation uses the nine passing cells |

## 8. First hard-gate result

Run: `llava-effusion-vislast-reltoken` completed in immutable run
`20260902T191411Z-ffd523c464c8-f99e2f39` from commit `ffd523c464c84417a93c5a6d0a34e5b74e55e76e`.

Observation: exact layer-22 capture fired once, alpha zero was a bitwise no-op, and the hook changed the
downstream connector and final logits. The 512-dimensional `C=1`, seed-0 Effusion probe reaches AUROC
0.7788 (95% patient-bootstrap CI 0.7538–0.8031). The control-seed mean is 0.6649 and selectivity is
0.1139 (95% CI 0.0875–0.1404). Across the registered 200-image intervention grid, the largest
concept-consistent change is 0.1470 at alpha -1; it exceeds random p95 0.1202 but not maximum absolute
sham 0.1546. The response is monotone over `[-0.5,0.5]` with Spearman rho 1.0.

Gate decision: `PASS`; scientific state `OBSERVED`, gate disposition `READY`, and
`selective_cell=false`. The terminal receipt, deterministic replay, provisional same-family integrity
audit, and pinned read-only Claude review support this decision.

Next: determine which fully specified cross-cell experiment should test whether decodability rank predicts
selective causal influence, then register its multiplicity handling and immutable dispatch command.
