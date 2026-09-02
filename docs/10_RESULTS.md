# Concept Flow: results record

Written 2026-08-27, independently re-verified the same day. This is the organised record of every result
the paper draws on. Every number in it was recomputed from the run directories named beside it, not copied
from a summary. Where a number in an earlier note disagrees with what the files say, the file wins and the
disagreement is recorded.

**Verification pass, 2026-08-27 15:08 EDT.** Every number below was recomputed a second time, from the raw
CSVs, by a reader who did not write the first draft. `src/figures.py` and `src/summarize_interventions.py`
were re-run. The manifest's split rule was re-derived from `patient_id` and matches the recorded `split` in
all 26,229 rows. All 45 peak-locus rows in section 2.1, every cross-model statistic in section 2.2, the
whole of sections 2.3, 3 and 4, and every dose-response table in section 5 reproduced to the digit --
all 45 peak rows in 2.1, and all 234 printed numbers in the five dose-response tables in 5.2 to 5.4. Nine
things did not survive the check, and are corrected in place below with the correction marked:

| Where | Was | Is | Why |
|---|---|---|---|
| §1.4 | `probe_act_qwen7b` "bit-identical" to `probe_qwen7b` | identical in `real`, `perm` and both nuisance columns; `control` and `selectivity` differ by up to 0.4479 | the parenthetical checked only `real`. That run also used the superseded 2-type control |
| §1.4 | `probe_act_lingshu7b` superseded because of the control scheme | superseded for two reasons; it also has `proj_dim: 0` | its `real` differs from the current run by up to 0.140. It is a different fit, not a control variant |
| §2.2 | Nodule's control "reaches 0.752" | 0.748 | 0.752 is in no file. The median control over Nodule's 340 cells is 0.7475 |
| §2.2, §3.1 | "visual-position loci" used for two different scopes | both scopes named and both numbers given | the peak search excludes `vis.block*`; these two summaries included them. +0.0656 against +0.0652 |
| §2.3 | peak-depth table's two depth columns unqualified | stated as medians over in-LLM peaks only | for `qwen7b` they describe 1 concept of 9 |
| §5.0 | status as of 14:50, 12 complete loci | 15:08 snapshot, 18 complete loci, and the whole block restructured | the queue is live. Volatile counts are now in one timestamped place |
| §5.0, §7 | 12 (later 18) complete loci "out of a planned 105" | 13 of the 18 count against the 105; 5 are an out-of-queue run | two of those 5 loci are in no queued job's list. Crediting all of them understated the outstanding work |
| §5.2 | InternVL3 sham max effect 0.0176 | 0.0217 | the maximum is at alpha -8, not +8 |
| §5.4, §5.6 | "meets the pre-registered criterion", one reading given, sham named as the only larger direction | two readings, both reported, both passing; a random direction is also larger | `summarize_interventions.py` disagreed with the prose. The script was wrong; see §5.4 |

Two defects were found in `src/summarize_interventions.py` itself and are fixed there, not here: its
completeness check accepted a locus truncated to a rectangular subset of the grid, and it tested
selectivity against the maximum of the random effects rather than their 95th percentile. See §5.0 and §5.4.

Nothing in the audit changed a headline: the median peak AUROC, the median selectivity, the permutation
null, the nuisance saturation, both matched-pair results and the Nodule and Mass failures all stand exactly
as first written.

Analysis scripts that produced these tables read only:

| Source | What it holds |
|---|---|
| `data/manifest.csv` | 26,229 rows, patient-level split, 9 clinical labels, 2 nuisance labels |
| `runs/act_<arch>/meta.json`, `shard{0,1,2}.npz` | pooled activations, 26,229 rows per model, one vector per locus |
| `runs/probe_<arch>/probe_results.csv` | D2, one row per concept x locus with the full battery |
| `runs/probe_<arch>/probe_meta.json` | probe hyperparameters and split sizes |
| `figures/headline.csv` | the peak-locus summary emitted by `src/figures.py` |
| `runs/arch_verification.json`, `runs/hook_demo_*.json`, `runs/smoke_lingshu.log` | measured architecture facts |
| `runs/int_*/intervention.csv` | D1, partial. See section 5 |

---

## 1. What was run

### 1.1 Models

Five models carry every result. Parameter counts below are measured at load time, not read off a model
card. The measurement source is given per row.

| Key | HF id | Domain | Params (B), measured | Vision blocks | Connector | LLM layers | Hidden | Visual tokens | Loci | Source of the measurement |
|---|---|---|---|---|---|---|---|---|---|---|
| `qwen7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | general | 8.292 | 32 | PatchMerger (`model.visual.merger`), 2x2 spatial merge | 28 | 3584 | 144 | 66 | `runs/hook_demo_qwen7b.json` |
| `lingshu7b` | `lingshu-medical-mllm/Lingshu-7B` | medical | 8.29 | 32 | PatchMerger, identical to `qwen7b` | 28 | 3584 | 144 | 66 | `runs/smoke_lingshu.log` |
| `llavamed7b` | `models/llava-med-7b-hf` (local convert) | medical | 7.57 | 24 | linear/MLP projector (`model.multi_modal_projector`) | 32 | 4096 | 576 | 72 | `runs/arch_verification.json` |
| `llava15_7b` | `llava-hf/llava-1.5-7b-hf` | general | 7.063 | 24 | linear/MLP projector | 32 | 4096 | 576 | 72 | `runs/hook_demo_llava7b.json`, `runs/arch_verification.json` |
| `internvl3_8b` | `OpenGVLab/InternVL3-8B-hf` | general | 7.94 | 24 | pixel shuffle 0.5 then LayerNorm then MLP | 28 | 3584 | 256 | 64 | `runs/arch_verification.json` |

`src/registry.py` records `internvl3_8b` at 8.08B; the loaded model counts 7.94B
(`runs/arch_verification.json`). Use 7.94.

Two models in the registry are excluded and the exclusion is evidenced:

| Key | Why excluded | Evidence |
|---|---|---|
| `huatuo7b` | does not load. `model type llava_qwen2` is unknown to transformers 5.2 | `runs/arch_verification.json`, `ok: false` |
| `llavaov7b` | architecture check failed: 26 of 27 claimed vision-block indices exist, and it emits 3,699 visual tokens against the registry's 1,485 | `runs/arch_verification.json`, `ok: false` |
| `chexagent3b` | 3B, below the study's 7B floor; also needs transformers 4.40 | `src/registry.py`, `enabled=False` |

The architecture-matched pairs:

| Pair | Members | Cleanliness |
|---|---|---|
| A | `lingshu7b` (medical) against `qwen7b` (general) | clean. Same family, same 32 vision blocks, same PatchMerger, same 28 LLM layers, same hidden 3584, same 144 visual tokens, same parameter count to two decimals. The loci are literally the same module paths, so a same-locus comparison is meaningful |
| B | `llavamed7b` (medical) against `llava15_7b` (general) | confounded. Same framework, same connector class, same 24 vision blocks, same 32 layers, same hidden 4096, but different base LLM: LLaVA-Med is Mistral-derived and LLaVA-1.5 is Vicuna-derived, and the parameter counts differ by 0.51B. Any difference between them is domain training plus base LLM, and this record does not attempt to separate the two |

`internvl3_8b` is unpaired. It serves only as a third connector type.

### 1.2 Data

NIH ChestX-ray14, 112,120 images unzipped locally, built into a manifest by `src/build_manifest.py`
(`data/manifest.csv`, written 2026-08-26).

Construction. For each of the 9 concepts the builder samples up to 4,000 positives and an equal number of
`No Finding` negatives, then takes the union of all 9 selections and deduplicates by image id. The union,
not the per-concept selection, is what gets extracted and probed, so the per-concept 1:1 balance does not
survive into the probe (see section 6).

Splitting. `patient_split()` hashes `sha256("concept-flow-v1:" + patient_id)` and assigns test below 0.2,
val below 0.3, train otherwise. It is patient-level, deterministic and machine-independent.

| Quantity | Value |
|---|---|
| Rows | 26,229 |
| Unique image ids | 26,229 |
| Unique patients | 8,297 |
| Train | 18,212 rows, 5,783 patients |
| Val | 2,674 rows, 855 patients |
| Test | 5,343 rows, 1,659 patients |
| Patients spanning more than one split | **0** |

Val is built but never used: `src/probe.py` fits on `split == "train"` and scores on `split == "test"`.

Positives per concept per split, and the resulting class prevalence the probe actually sees:

| Concept | Train pos | Val pos | Test pos | Train prevalence | Test prevalence |
|---|---|---|---|---|---|
| Infiltration | 2,948 | 439 | 872 | 0.1619 | 0.1632 |
| Effusion | 2,423 | 357 | 700 | 0.1330 | 0.1310 |
| Atelectasis | 2,311 | 319 | 626 | 0.1269 | 0.1172 |
| Nodule | 1,549 | 257 | 431 | 0.0851 | 0.0807 |
| Consolidation | 1,288 | 128 | 301 | 0.0707 | 0.0563 |
| Mass | 1,263 | 222 | 372 | 0.0693 | 0.0696 |
| Pneumothorax | 1,254 | 160 | 468 | 0.0689 | 0.0876 |
| Cardiomegaly | 881 | 105 | 221 | 0.0484 | 0.0414 |
| Edema | 467 | 90 | 179 | 0.0256 | 0.0335 |
| view_AP (nuisance) | 7,628 | 1,064 | 2,168 | 0.4188 | 0.4058 |
| sex_M (nuisance) | 9,643 | 1,561 | 2,995 | 0.5295 | 0.5605 |
| No Finding | 8,922 | 1,321 | 2,624 | 0.4899 | 0.4911 |

Every row that is not `No Finding` carries at least one of the 9 labels; 12,867 rows carry none, which is
exactly the `No Finding` count across the three splits. Labels co-occur: 7,859 rows carry one label, 4,079
carry two, 1,127 carry three, 256 carry four, 37 carry five, 4 carry six.

Each concept's negative class is therefore mixed rather than healthy. In train, between 50.3% (Edema) and
58.5% (Infiltration) of a concept's negatives are `No Finding` rows and the rest carry some other finding.

### 1.3 Loci and pooling

`src/loci.py` defines the locus set and `src/extract.py` fills every locus in one forward pass per image.
The locus set has the same shape in every model: a subsample of vision blocks every 4, the last vision
block, the connector output, then every LLM layer read twice.

| Locus name | Module | Positions pooled |
|---|---|---|
| `vis.block{0,4,8,...}` | vision block i | mean over all tokens the block emits |
| `vis.last` | last vision block | mean over all tokens |
| `connector` | the module whose output is the visual token sequence the LLM consumes | mean over visual tokens |
| `llm.L{i}.vis` | LLM decoder layer i | mean over the sequence positions whose input id equals the image token id |
| `llm.L{i}.ans` | LLM decoder layer i | the final prompt position, no pooling |

Pooling is mean, fixed in `ActivationCache._pool`, identical at every locus and every model. Locus counts:
66 for `qwen7b` and `lingshu7b`, 72 for `llavamed7b` and `llava15_7b`, 64 for `internvl3_8b`. The counts
differ only because the models have 28 or 32 LLM layers and 24 or 32 vision blocks.

Extraction used a single fixed prompt for every concept and every model:

> `Is there a pleural effusion in this chest radiograph? Answer yes or no.`

recorded in all five `runs/act_*/meta.json`. Every probe in section 2, including the Cardiomegaly and
Nodule probes, reads activations produced while the model was answering an effusion question. Consequences
in section 6.

A prior extraction was discarded and kept as documentation in `runs/_bad_vision_pooling/` with
`WHY.md`: Qwen's vision tower emits `(sum_of_patches, D)` with no batch axis, and the pooler averaged the
whole batch into one row, giving 3,279 rows against 26,229 ids. It was caught by `probe.py`'s row-count
assertion, never reached a figure, and is fixed by passing the batch size into the cache.

### 1.4 The validity battery

`src/probe.py` fits a logistic regression, `C = 1.0`, `max_iter = 2000`, `random_state = 0`, on
standardised features, after a random Gaussian projection to 512 dimensions
(`proj_dim: 512` in every `runs/probe_*/probe_meta.json`). The projection is drawn once per locus and
reused for every concept at that locus. Native locus widths are 1024, 1280, 3584 and 4096, so the
projection is a genuine down-projection everywhere and the widest locus cannot win on width.

At every concept x locus cell the file records four AUROCs and one difference:

| Column | What is fitted | What it rules out |
|---|---|---|
| `real` | true labels on train, scored against true labels on test | nothing; this is the surface number the field reports |
| `perm` | train labels permuted within the train split, scored against **true** test labels | that the fit is an artefact of the pipeline. This is a chance floor for this exact fit, not a shuffled-test score |
| `control` | a fixed random binary label per input **type**, fitted and scored on that random label | that the number is decoder capacity. This is the Hewitt and Liang selectivity denominator |
| `selectivity` | `real - control` | the headline quantity. The field has no equivalent number |
| `nuis_view_AP`, `nuis_sex_M` | view position and sex, from the **same** features at the **same** locus | that a finding probe is reading acquisition or demography rather than pathology |

The input type for the control task is `view position x sex x age decade`. Measured on the manifest that
gives **39 types**, median 629 rows, largest 1,970, **smallest 3**. 37 of the 39 recur in both train and
test; the two that do not are the two smallest, `PA_F_9` (3 rows, test only) and `AP_M_9` (5 rows, train
only).

An earlier scheme used view position alone, 2 types. It is degenerate: with two types the random
assignment is one of four functions, two constant and two equal to the type up to a flip, so the control
just measures whether view position is decodable. The surviving evidence for this is
`runs/probe_act_lingshu7b/probe_results.csv`, whose `control` column equals its `nuis_view_AP` column to
13 decimal places (0.9972528111107883 against 0.9972528111107885 at `vis.block0`), giving a median
control of 0.9983 and a median selectivity of **-0.3029**. That run is superseded and must not be quoted
as a result; it is the reason the 39-type scheme exists. It differs from `runs/probe_lingshu7b` in **two**
ways, not one, and the second matters: its `probe_meta.json` records `proj_dim: 0`, so it was fitted on the
full native locus width rather than the 512-dimensional projection every current run uses. Its `real`
column consequently differs from `probe_lingshu7b`'s by up to 0.140, and its `perm` by up to 0.095. It is
not a control-scheme variant of the current run; it is a different fit.

`runs/probe_act_qwen7b/probe_results.csv` was described in the first draft of this record as
"bit-identical" to `runs/probe_qwen7b/probe_results.csv`. **Corrected: it is not.** Over the 594 shared
cells the two agree exactly, to a maximum absolute difference of 0.0, in `real`, `perm`, `nuis_view_AP` and
`nuis_sex_M`; they differ in `control` and therefore in `selectivity` by up to **0.4479**. The reason is
that `probe_act_qwen7b` is also a 2-type-control run: its `control` column tracks its `nuis_view_AP` column
to 3.6e-05, its median control is 0.9981 and its median selectivity is -0.3062, and its CSV has no
`n_types` column because it predates the field. So it is a **duplicate** of `probe_qwen7b` on the four
columns that do not depend on the control task, and a **superseded** run on the two that do. Neither half
may be quoted as a second observation: the first half is the same fit re-run, the second half is the
degenerate control.

Two battery elements promised in `docs/00_PAPER_PLAN.md` are **not implemented**: there is no
random-encoder floor anywhere in `src/`, and `--bootstrap` is declared in `probe.py` but never referenced
after argument parsing, so no confidence interval exists on any AUROC in this document.

---

## 2. The decodability result

Definition used throughout this section, matching `src/figures.py`: the **peak locus** for a
(model, concept) is the locus with the highest `real` AUROC among the visual-position track, that is
`vis.last`, `connector` and every `llm.L{i}.vis`. Answer-position loci and intermediate vision blocks are
excluded from the peak search. Where a different scope changes a number, both are given.

### 2.1 Per model, at the peak locus

Source: `runs/probe_<model>/probe_results.csv`. Reproduced in `figures/headline.csv`.

#### qwen7b (general)

| concept | peak locus | AUROC | perm | control | selectivity | view AUROC | sex AUROC |
|---|---|---|---|---|---|---|---|
| Atelectasis | `connector` | 0.725 | 0.470 | 0.672 | +0.053 | 0.9986 | 0.968 |
| Cardiomegaly | `vis.last` | 0.801 | 0.440 | 0.734 | +0.067 | 0.9987 | 0.967 |
| Consolidation | `vis.last` | 0.726 | 0.489 | 0.624 | +0.103 | 0.9987 | 0.967 |
| Edema | `vis.last` | 0.826 | 0.504 | 0.599 | +0.227 | 0.9987 | 0.967 |
| Effusion | `llm.L0.vis` | 0.780 | 0.539 | 0.704 | +0.075 | 0.9986 | 0.970 |
| Infiltration | `vis.last` | 0.712 | 0.502 | 0.618 | +0.094 | 0.9987 | 0.967 |
| Mass | `vis.last` | 0.652 | 0.536 | 0.676 | **-0.024** | 0.9987 | 0.967 |
| Nodule | `vis.last` | 0.651 | 0.506 | 0.764 | **-0.113** | 0.9987 | 0.967 |
| Pneumothorax | `vis.last` | 0.779 | 0.545 | 0.721 | +0.058 | 0.9987 | 0.967 |

#### lingshu7b (medical)

| concept | peak locus | AUROC | perm | control | selectivity | view AUROC | sex AUROC |
|---|---|---|---|---|---|---|---|
| Atelectasis | `llm.L23.vis` | 0.767 | 0.478 | 0.693 | +0.074 | 0.9972 | 0.971 |
| Cardiomegaly | `llm.L20.vis` | 0.897 | 0.538 | 0.733 | +0.163 | 0.9979 | 0.972 |
| Consolidation | `llm.L19.vis` | 0.751 | 0.499 | 0.636 | +0.114 | 0.9979 | 0.971 |
| Edema | `vis.last` | 0.869 | 0.488 | 0.630 | +0.239 | 0.9979 | 0.981 |
| Effusion | `llm.L22.vis` | 0.831 | 0.558 | 0.685 | +0.145 | 0.9971 | 0.970 |
| Infiltration | `llm.L3.vis` | 0.740 | 0.522 | 0.637 | +0.103 | 0.9983 | 0.979 |
| Mass | `llm.L16.vis` | 0.754 | 0.536 | 0.690 | +0.064 | 0.9988 | 0.971 |
| Nodule | `llm.L23.vis` | 0.698 | 0.505 | 0.761 | **-0.063** | 0.9972 | 0.971 |
| Pneumothorax | `llm.L23.vis` | 0.869 | 0.465 | 0.722 | +0.148 | 0.9972 | 0.971 |

#### llavamed7b (medical)

| concept | peak locus | AUROC | perm | control | selectivity | view AUROC | sex AUROC |
|---|---|---|---|---|---|---|---|
| Atelectasis | `llm.L9.vis` | 0.713 | 0.477 | 0.686 | +0.027 | 0.9988 | 0.977 |
| Cardiomegaly | `llm.L3.vis` | 0.836 | 0.499 | 0.740 | +0.096 | 0.9986 | 0.979 |
| Consolidation | `llm.L12.vis` | 0.727 | 0.489 | 0.641 | +0.087 | 0.9986 | 0.979 |
| Edema | `llm.L1.vis` | 0.813 | 0.535 | 0.619 | +0.194 | 0.9988 | 0.981 |
| Effusion | `llm.L0.vis` | 0.784 | 0.516 | 0.716 | +0.068 | 0.9984 | 0.979 |
| Infiltration | `llm.L7.vis` | 0.729 | 0.512 | 0.639 | +0.090 | 0.9990 | 0.979 |
| Mass | `llm.L22.vis` | 0.681 | 0.503 | 0.683 | **-0.002** | 0.9988 | 0.972 |
| Nodule | `llm.L12.vis` | 0.677 | 0.480 | 0.771 | **-0.094** | 0.9986 | 0.979 |
| Pneumothorax | `llm.L4.vis` | 0.807 | 0.531 | 0.736 | +0.071 | 0.9986 | 0.980 |

#### llava15_7b (general)

| concept | peak locus | AUROC | perm | control | selectivity | view AUROC | sex AUROC |
|---|---|---|---|---|---|---|---|
| Atelectasis | `llm.L3.vis` | 0.716 | 0.487 | 0.671 | +0.045 | 0.9985 | 0.977 |
| Cardiomegaly | `llm.L0.vis` | 0.826 | 0.485 | 0.742 | +0.085 | 0.9984 | 0.979 |
| Consolidation | `connector` | 0.715 | 0.486 | 0.649 | +0.066 | 0.9986 | 0.977 |
| Edema | `llm.L8.vis` | 0.810 | 0.513 | 0.616 | +0.194 | 0.9982 | 0.975 |
| Effusion | `llm.L3.vis` | 0.780 | 0.518 | 0.717 | +0.062 | 0.9985 | 0.977 |
| Infiltration | `llm.L4.vis` | 0.723 | 0.502 | 0.625 | +0.098 | 0.9983 | 0.977 |
| Mass | `llm.L27.vis` | 0.670 | 0.512 | 0.667 | +0.003 | 0.9980 | 0.968 |
| Nodule | `llm.L2.vis` | 0.662 | 0.493 | 0.772 | **-0.110** | 0.9986 | 0.980 |
| Pneumothorax | `llm.L4.vis` | 0.803 | 0.495 | 0.727 | +0.076 | 0.9983 | 0.977 |

#### internvl3_8b (general)

| concept | peak locus | AUROC | perm | control | selectivity | view AUROC | sex AUROC |
|---|---|---|---|---|---|---|---|
| Atelectasis | `llm.L2.vis` | 0.724 | 0.458 | 0.674 | +0.051 | 0.9979 | 0.966 |
| Cardiomegaly | `vis.last` | 0.851 | 0.492 | 0.743 | +0.108 | 0.9988 | 0.979 |
| Consolidation | `llm.L7.vis` | 0.737 | 0.493 | 0.624 | +0.112 | 0.9980 | 0.963 |
| Edema | `llm.L11.vis` | 0.821 | 0.538 | 0.595 | +0.227 | 0.9979 | 0.953 |
| Effusion | `vis.last` | 0.806 | 0.551 | 0.698 | +0.108 | 0.9988 | 0.979 |
| Infiltration | `llm.L14.vis` | 0.721 | 0.499 | 0.615 | +0.106 | 0.9979 | 0.947 |
| Mass | `llm.L18.vis` | 0.682 | 0.470 | 0.661 | +0.021 | 0.9978 | 0.939 |
| Nodule | `llm.L5.vis` | 0.658 | 0.517 | 0.768 | **-0.109** | 0.9982 | 0.965 |
| Pneumothorax | `llm.L2.vis` | 0.799 | 0.506 | 0.714 | +0.084 | 0.9979 | 0.966 |

### 2.2 Cross-model summary

Over the 45 (model, concept) cells at their peak locus:

| Statistic | Median | Mean | Min | Max |
|---|---|---|---|---|
| `real` | 0.7507 | 0.7578 | 0.6511 | 0.8966 |
| `perm` | 0.5021 | 0.5039 | 0.4397 | 0.5579 |
| `control` | 0.6854 | 0.6847 | 0.5945 | 0.7716 |
| `selectivity` | **+0.0764** | +0.0732 | -0.1132 | +0.2388 |

7 of the 45 cells have selectivity at or below zero.

The headline sentence, stated with its exact scope: **the median peak AUROC is 0.751 and the control task
at the same locus with the same capacity reaches 0.685, so the median selectivity is +0.076.** The peak
AUROC is 0.751 of the way from chance to perfect, but only 0.076 of it survives the capacity control.

An earlier project note gave this as +0.084. That number is the same statistic computed with the peak
search widened to **all** loci including answer positions and intermediate vision blocks; recomputed that
way the median is +0.0881. The narrower, pre-registered visual-position track gives +0.0764. Both are in
this record so the paper can pick one and state its scope.

Pooling every cell rather than only the peaks, over all 3,060 concept x locus x model rows:

| Statistic | Value |
|---|---|
| median `real` | 0.7168 |
| median `control` | 0.6707 |
| median `selectivity` | +0.0623 |
| fraction of cells with selectivity at or below 0 | 0.216 |
| median `selectivity`, all non-answer loci (1,728 cells, includes `vis.block*`) | +0.0656 |
| median `selectivity`, the visual-position track proper (1,422 cells, no `vis.block*`) | +0.0652 |
| median `selectivity`, answer-position loci only (1,332 cells) | +0.0544 |

The permutation null is well behaved everywhere: across all 3,060 cells it runs 0.4046 to 0.5869 with
median 0.5028, and 95.8% of cells fall in [0.45, 0.55]. Nothing in this study is an artefact of the fitting
pipeline. Per model the medians are 0.5042, 0.5036, 0.5027, 0.5034, 0.4993.

Which concepts are never selective and which always. Counting over every locus in every model:

| Concept | Loci with selectivity > 0 | Median selectivity | Verdict |
|---|---|---|---|
| Edema | 340 / 340 | +0.1966 | always selective |
| Effusion | 340 / 340 | +0.0620 | always selective |
| Consolidation | 340 / 340 | +0.0865 | always selective |
| Infiltration | 340 / 340 | +0.1010 | always selective |
| Cardiomegaly | 331 / 340 | +0.0685 | selective except at 9 loci |
| Atelectasis | 319 / 340 | +0.0291 | selective but small |
| Pneumothorax | 317 / 340 | +0.0483 | selective but small |
| Mass | 72 / 340 | -0.0207 | not selective. Positive only in `lingshu7b` (34/66 loci) and marginally in `internvl3_8b` (18/64) and `llavamed7b` (16/72); 0/66 in `qwen7b`, 4/72 in `llava15_7b` |
| Nodule | **0 / 340** | -0.1104 | **never selective, at any locus, in any of the five models** |

Nodule's surface AUROC is not zero: its median `real` across models and loci is 0.637, and its peak
reaches 0.698 in `lingshu7b`. Its control, on the same 340 cells and at the same capacity, has median
**0.748** (the first draft said 0.752; that value is in no file and is corrected here — the median over
Nodule's 340 cells is 0.7475, the maximum is 0.790, and the control at each model's Nodule peak runs 0.761
to 0.772). The apparent decodability of
Nodule is entirely probe capacity by this measurement. The same statement holds more weakly for Mass
(median `real` 0.649, median control 0.667).

Section 6 explains why "never selective, in all five models" is a much weaker claim than five independent
replications, and it should not be written as one.

### 2.3 Shape of the curve

Median over the 9 concepts of `real` at fixed landmark loci:

| Model | `vis.last` | `connector` | `llm.L0.vis` | mid LLM `.vis` | last LLM `.vis` | last LLM `.ans` |
|---|---|---|---|---|---|---|
| qwen7b | 0.7263 | 0.7248 | 0.7149 | 0.6986 | 0.6868 | 0.6763 |
| lingshu7b | 0.7544 | 0.7491 | 0.7561 | 0.7491 | 0.7511 | 0.7691 |
| llavamed7b | 0.7208 | 0.7269 | 0.7196 | 0.7242 | 0.7151 | 0.6985 |
| llava15_7b | 0.7216 | 0.7157 | 0.7174 | 0.7118 | 0.7032 | 0.6884 |
| internvl3_8b | 0.7225 | 0.7223 | 0.7232 | 0.7358 | 0.7218 | 0.7193 |

Same landmarks, median `selectivity`:

| Model | `vis.last` | `connector` | `llm.L0.vis` | mid LLM `.vis` | last LLM `.vis` | last LLM `.ans` |
|---|---|---|---|---|---|---|
| qwen7b | 0.0583 | 0.0560 | 0.0625 | 0.0605 | 0.0661 | 0.0322 |
| lingshu7b | 0.0880 | 0.0869 | 0.0842 | 0.1091 | 0.1130 | 0.1365 |
| llavamed7b | 0.0477 | 0.0524 | 0.0683 | 0.0676 | 0.0369 | 0.0173 |
| llava15_7b | 0.0502 | 0.0611 | 0.0623 | 0.0488 | 0.0547 | 0.0203 |
| internvl3_8b | 0.0890 | 0.0863 | 0.0920 | 0.1064 | 0.1010 | 0.0893 |

The connector is not a bottleneck. Across all 45 cells the change from `vis.last` to `connector` has median
**-0.0021** AUROC and is negative in 27 of 45. The largest single drop is -0.0379 (`qwen7b`, Nodule) and
the largest single gain is +0.0179 (`llava15_7b`, Mass). Whatever the survey's single record F8b was
seeing, this measurement does not reproduce a step down at the connector.

Where the peak sits, per model. The two depth columns are medians **over the concepts that peak inside the
LLM only** — a concept peaking at `vis.last` or the connector has no LLM layer to average — so for `qwen7b`
they describe 1 concept of 9 and should not be read as a summary of that model. Relative depth is
`(layer + 1) / n_layers`.

| Model | Concepts peaking before the LLM | Concepts peaking inside the LLM | Median LLM layer of the in-LLM peaks | Median relative depth of the in-LLM peaks |
|---|---|---|---|---|
| qwen7b | 8 | 1 | L0 | 0.036 |
| lingshu7b | 1 | 8 | L21 | 0.786 |
| llavamed7b | 0 | 9 | L7 | 0.250 |
| llava15_7b | 1 | 8 | L3.5 | 0.141 |
| internvl3_8b | 2 | 7 | L7 | 0.286 |

The medical model of the clean pair peaks near the end of the language model (`lingshu7b`, median layer 21
of 28) while its general twin peaks at the vision tower or the connector (`qwen7b`, 8 of 9 concepts). In
the confounded pair the same contrast appears but is much weaker (`llavamed7b` median layer 7 of 32
against `llava15_7b` median layer 3.5 of 32).

---

## 3. The nuisance result

Every row of every `probe_results.csv` carries `nuis_view_AP` and `nuis_sex_M`, fitted on the same
features at the same locus with the same capacity as the clinical probe.

### 3.1 View position

| Model | All loci: min | median | max | Non-answer loci range | Answer-position loci range |
|---|---|---|---|---|---|
| qwen7b | 0.9968 | 0.9981 | 0.9994 | 0.9968 to 0.9994 | 0.9969 to 0.9985 |
| lingshu7b | 0.9964 | 0.9980 | 0.9993 | 0.9964 to 0.9993 | 0.9973 to 0.9987 |
| llavamed7b | 0.9972 | 0.9985 | 0.9992 | 0.9979 to 0.9992 | 0.9972 to 0.9987 |
| llava15_7b | 0.9972 | 0.9982 | 0.9991 | 0.9972 to 0.9991 | 0.9974 to 0.9984 |
| internvl3_8b | 0.9969 | 0.9980 | 0.9991 | 0.9969 to 0.9991 | 0.9972 to 0.9986 |

The fourth column is the range over every locus that is not an answer position, `vis.block*` included; it
is not the narrower "visual-position track" that section 2 uses for the peak search. On that narrower
definition the ranges are 0.9972-0.9987 (`qwen7b`), 0.9971-0.9988 (`lingshu7b`), 0.9979-0.9990
(`llavamed7b`), 0.9972-0.9987 (`llava15_7b`) and 0.9969-0.9988 (`internvl3_8b`). Nothing in the conclusion
turns on which is used.

**View position decodes at 0.9964 to 0.9994 at every one of the 66, 72 or 64 loci in every one of the five
models.** There is no depth dependence worth reporting: the total range across 3,060 cells is 0.003 AUROC.
It is already at 0.9964 at the very first sampled vision block (`lingshu7b`, `vis.block0`) and it is still
at 0.9969 at the last LLM layer's answer position (`qwen7b`, `llm.L27.ans`).

The comparison that matters: the best clinical concept at its best locus in the best model is Cardiomegaly
in `lingshu7b` at 0.897. View position beats that by 0.10 AUROC **at every locus in every model**, and beats
the median clinical peak of 0.751 by 0.25.

### 3.2 Sex

Sex is the one nuisance variable with a real depth profile.

| Model | `vis.block0` | `vis.last` | `connector` | `llm.L0.vis` | LLM `.vis` median | LLM `.ans` median | last `.ans` |
|---|---|---|---|---|---|---|---|
| qwen7b | 0.7864 | 0.9673 | 0.9684 | 0.9696 | 0.9542 | 0.8938 | 0.8493 |
| lingshu7b | 0.7693 | 0.9812 | 0.9817 | 0.9800 | 0.9742 | 0.9413 | 0.9469 |
| llavamed7b | 0.8782 | 0.9777 | 0.9785 | 0.9795 | 0.9762 | 0.9478 | 0.9304 |
| llava15_7b | 0.8778 | 0.9781 | 0.9766 | 0.9786 | 0.9735 | 0.9389 | 0.9221 |
| internvl3_8b | 0.8123 | 0.9787 | 0.9689 | 0.9678 | 0.9477 | 0.8952 | 0.8561 |

Sex rises from 0.77 to 0.88 at the first vision block to 0.985 to 0.989 by the middle vision blocks, holds
near 0.97 to 0.98 through the connector and the early LLM, and decays through LLM depth, most at the answer
position. The minimum over all loci and models is 0.7693 (`lingshu7b`, `vis.block0`) and the maximum is
0.9886 (`llava15_7b` and `llavamed7b`, mid vision blocks).

Even its lowest value, 0.769, is above the median clinical peak's **selectivity-corrected** standing: the
median clinical peak is 0.751 raw against a 0.685 control. Sex at its worst locus outperforms the median
clinical concept at its best.

### 3.3 What this means for the finding probes

Both nuisance variables are decodable at every locus at levels no clinical concept approaches. This does
not by itself prove a finding probe is reading them, because the nuisance probes are separate fits on the
same features and a high nuisance AUROC is compatible with a clinically valid finding probe in an
orthogonal subspace. What the numbers do establish is that the representation is saturated with
acquisition and demographic information, so any claim that a 0.75 finding AUROC reflects clinical content
has to survive that. This record does not contain the test that would settle it: no finding probe was
refitted with view position and sex partialled out, and no concept-versus-nuisance subspace angle was
computed. Section 7 lists it.

---

## 4. The matched-pair result

### 4.1 Pair A: lingshu7b (medical) against qwen7b (general). Clean.

Identical architecture, so the loci are the same module paths and both a peak comparison and a same-locus
comparison are meaningful.

| Concept | lingshu peak | AUROC | sel | qwen peak | AUROC | sel | dAUROC | dSel |
|---|---|---|---|---|---|---|---|---|
| Atelectasis | `llm.L23.vis` | 0.767 | +0.074 | `connector` | 0.725 | +0.053 | +0.042 | +0.021 |
| Cardiomegaly | `llm.L20.vis` | 0.897 | +0.163 | `vis.last` | 0.801 | +0.067 | +0.095 | +0.097 |
| Consolidation | `llm.L19.vis` | 0.751 | +0.114 | `vis.last` | 0.726 | +0.103 | +0.024 | +0.012 |
| Edema | `vis.last` | 0.869 | +0.239 | `vis.last` | 0.826 | +0.227 | +0.042 | +0.011 |
| Effusion | `llm.L22.vis` | 0.831 | +0.145 | `llm.L0.vis` | 0.780 | +0.075 | +0.051 | +0.070 |
| Infiltration | `llm.L3.vis` | 0.740 | +0.103 | `vis.last` | 0.712 | +0.094 | +0.029 | +0.010 |
| Mass | `llm.L16.vis` | 0.754 | +0.064 | `vis.last` | 0.652 | -0.024 | +0.102 | +0.088 |
| Nodule | `llm.L23.vis` | 0.698 | -0.063 | `vis.last` | 0.651 | -0.113 | +0.047 | +0.050 |
| Pneumothorax | `llm.L23.vis` | 0.869 | +0.148 | `vis.last` | 0.779 | +0.058 | +0.090 | +0.090 |
| **median** | | | | | | | **+0.047** | **+0.050** |

Lingshu is higher on peak AUROC in **9 of 9** concepts and on peak selectivity in **9 of 9**.

Same-locus comparison, median over the 9 concepts:

| Locus | median dAUROC | median dSel | lingshu higher AUROC |
|---|---|---|---|
| `vis.last` | +0.0319 | +0.0113 | 8 / 9 |
| `connector` | +0.0263 | +0.0229 | 9 / 9 |
| `llm.L0.vis` | +0.0334 | +0.0189 | 9 / 9 |
| `llm.L13.vis` | +0.0657 | +0.0311 | 9 / 9 |
| `llm.L20.vis` | +0.0635 | +0.0434 | 9 / 9 |
| `llm.L27.vis` | +0.0790 | +0.0443 | 9 / 9 |

The medical advantage exists at the vision tower (+0.032) and roughly doubles by the end of the language
model (+0.079). Medical training changed both towers, and it changed the language model more. Because
`lingshu7b` and `qwen7b` share their architecture, token count and parameter count exactly, this
comparison is not confounded by capacity or by connector type.

### 4.2 Pair B: llavamed7b (medical) against llava15_7b (general). Confounded.

The base LLM differs (Mistral-derived against Vicuna-derived) and the parameter counts differ by 0.51B.
Any difference below is domain training **and** base LLM, and this record makes no attempt to attribute it.

| Concept | llavamed peak | AUROC | sel | llava15 peak | AUROC | sel | dAUROC | dSel |
|---|---|---|---|---|---|---|---|---|
| Atelectasis | `llm.L9.vis` | 0.713 | +0.027 | `llm.L3.vis` | 0.716 | +0.045 | -0.003 | -0.018 |
| Cardiomegaly | `llm.L3.vis` | 0.836 | +0.096 | `llm.L0.vis` | 0.826 | +0.085 | +0.010 | +0.012 |
| Consolidation | `llm.L12.vis` | 0.727 | +0.087 | `connector` | 0.715 | +0.066 | +0.012 | +0.021 |
| Edema | `llm.L1.vis` | 0.813 | +0.194 | `llm.L8.vis` | 0.810 | +0.194 | +0.003 | +0.001 |
| Effusion | `llm.L0.vis` | 0.784 | +0.068 | `llm.L3.vis` | 0.780 | +0.062 | +0.004 | +0.006 |
| Infiltration | `llm.L7.vis` | 0.729 | +0.090 | `llm.L4.vis` | 0.723 | +0.098 | +0.006 | -0.008 |
| Mass | `llm.L22.vis` | 0.681 | -0.002 | `llm.L27.vis` | 0.670 | +0.003 | +0.012 | -0.005 |
| Nodule | `llm.L12.vis` | 0.677 | -0.094 | `llm.L2.vis` | 0.662 | -0.110 | +0.015 | +0.016 |
| Pneumothorax | `llm.L4.vis` | 0.807 | +0.071 | `llm.L4.vis` | 0.803 | +0.076 | +0.003 | -0.006 |
| **median** | | | | | | | **+0.006** | **+0.001** |

LLaVA-Med is higher on peak AUROC in 8 of 9 and on peak selectivity in only 5 of 9. The median selectivity
advantage is +0.001, which is nothing.

Same-locus comparison:

| Locus | median dAUROC | median dSel | llavamed higher AUROC |
|---|---|---|---|
| `vis.last` | -0.0013 | -0.0015 | 3 / 9 |
| `connector` | -0.0056 | -0.0087 | 1 / 9 |
| `llm.L0.vis` | -0.0036 | -0.0012 | 4 / 9 |
| `llm.L15.vis` | +0.0058 | -0.0005 | 7 / 9 |
| `llm.L23.vis` | +0.0123 | +0.0123 | 8 / 9 |
| `llm.L31.vis` | +0.0095 | +0.0044 | 6 / 9 |

LLaVA-Med is **worse** than LLaVA-1.5 at its vision tower and its connector (1 of 9 concepts at the
connector) and better only in the second half of the language model, by about +0.012 AUROC.

### 4.3 What the two pairs jointly support and do not

Both pairs agree on the direction of the depth effect: the medical model's advantage is concentrated in
the later language model, and in both pairs the pre-LLM stages show the smallest or no medical advantage.
The magnitudes are an order apart, +0.079 against +0.010 at the deepest compared locus, and only pair A is
architecturally clean. A claim of the form "medical training moves clinical information deeper into the
language model" is supported at full strength by one clean pair and at weak strength by one confounded
pair. It is not supported by five models: `internvl3_8b` has no medical twin.

---

## 5. The intervention result. PRELIMINARY

### 5.0 Status, and how completeness is decided

**This section describes a queue that is still running.** Every count in it is a snapshot and every count
below will be wrong within minutes of being written. Only the per-locus dose-response tables in 5.1 to 5.4
are stable, because a locus that has completed its grid never changes afterwards. Regenerate the counts
with `python src/summarize_interventions.py` rather than quoting them from here.

Snapshot at **2026-08-27 15:08 EDT**, from `src/summarize_interventions.py --markdown`:

| run | rows | complete loci | partial or absent | `DONE` |
|---|---|---|---|---|
| `int_internvl3_8b_effusion` | 486 | `vis.last`, `connector` | not started: `llm.L0.vis`, `llm.L7.vis`, `llm.L14.vis`, `llm.L21.vis`, `llm.L27.vis` | **no** |
| `int_lingshu7b_cardiomegaly` | 234 | `vis.last` | not started: 6 loci | **no** |
| `int_lingshu7b_effusion` | 243 | `vis.last` | not started: 6 loci | **no** |
| `int_lingshu_effusion` | 1,215 | `vis.last`, `connector`, `llm.L0.vis`, `llm.L13.vis`, `llm.L22.vis` | none | **no** |
| `int_llava15_7b_effusion` | 729 | `vis.last`, `connector`, `llm.L0.vis` | not started: `llm.L8.vis`, `llm.L16.vis`, `llm.L24.vis`, `llm.L31.vis` | **no** |
| `int_llavamed7b_cardiomegaly` | 702 | `vis.last`, `connector`, `llm.L0.vis` | not started: `llm.L8.vis`, `llm.L16.vis`, `llm.L24.vis`, `llm.L31.vis` | **no** |
| `int_llavamed7b_effusion` | 243 | `vis.last` | not started: 6 loci | **no** |
| `int_qwen7b_cardiomegaly` | 234 | `vis.last` | not started: 6 loci | **no** |
| `int_qwen7b_effusion` | 243 | `vis.last` | not started: 6 loci | **no** |
| the 5 pneumothorax jobs, `int_llava15_7b_cardiomegaly`, `int_internvl3_8b_cardiomegaly` | queued, no directory yet | none | **no** |

**Zero of the 15 queued jobs have written a `DONE` marker.** Eight are running on GPUs 0 through 7 and
seven are still queued (`runs/launcher.log`, `runs/supervise.log`). `runs/int_lingshu_effusion` is a ninth
directory that is **not one of the 15**: it is an earlier standalone run that crashed, described below.

The plan's denominator is now read out of `runs/launcher.log` rather than asserted: 15 jobs, 7 loci each,
105 loci. At this snapshot **18 loci have a complete grid, of which 13 count against the plan** and the
other 5 are the out-of-queue crashed run's, so **92 planned loci are outstanding**. Two of the crashed
run's five loci, `llm.L13.vis` and `llm.L22.vis`, are not in any queued job's locus list at all, so they
will not be reproduced by finishing the queue. Crediting all 18 against the 105 would have understated the
outstanding work by five loci; `src/summarize_interventions.py` now separates the two counts. Earlier
drafts of this section recorded 12 complete loci at 14:50 EDT; the queue moved.

**How a locus is judged complete, and a defect that was found in that judgement.** `src/intervene.py`
rewrites `intervention.csv` after every locus, so a partial CSV is normal and its existence says nothing
about completion. Only the `DONE` marker speaks to whether a *run* finished, and none exists yet.
`src/summarize_interventions.py` decides whether a *locus* is complete. Its first version checked only
that the rows present formed a full directions x magnitudes rectangle, and that check is not sufficient:
fed a locus truncated to a rectangular subset of itself, 5 directions x 9 magnitudes out of 27 x 9, it
reported the locus as **complete** and summarised it. Fed one cut down to 2 random directions it reported
the locus as complete **and** as meeting the selectivity criterion, because a 5th-to-95th percentile band
computed over 2 samples is not a band. Both were demonstrated against synthetic truncations of
`int_qwen7b_effusion`. The check now validates the grid against the protocol — 20 random directions, 4 or 5
unrelated-concept directions, a sham, and the declared 9 magnitudes — and names the reason when a locus
fails. All three synthetic truncations are now rejected, and all 18 real loci still pass, so no number in
5.1 to 5.4 was affected. The defect mattered for what could have been reported next, not for what was.

`runs/int_lingshu_effusion` is a special case. Its five loci each completed the full grid, and the process
then died on the last line of `main()` with `AttributeError: 'Namespace' object has no attribute 'model'`
(`runs/int_lingshu_effusion.log`), a stale reference from before `--model` was renamed to `--arch`. The
bug is already fixed in the current `src/intervene.py`, which writes `args.arch`. The five loci are usable
data; the run simply never got to write its marker. It is the only multi-locus D1 result that exists.

That the crashed run's data is trustworthy is now checked rather than assumed. The relaunched job
`runs/int_lingshu7b_effusion` re-ran the same model, concept and locus in a separate process on a different
GPU, and its `vis.last` grid is **identical to `int_lingshu_effusion`'s in all 243 cells**, to the last
digit of both `mean_p_yes` and `sd`. That is the only genuine cross-process determinism check anywhere in
this record. It confirms the crashed run's numbers; it says nothing about sampling variance, because the
evaluation set, the direction fit and the random directions are all seeded identically.

Protocol as actually executed: directions fitted on all 18,212 train rows, evaluated on 160 held-out test
images, 20 random equal-norm directions, up to 5 unrelated-concept directions, 1 sham (coordinate
permutation of `v_c`), 9 magnitudes from -8 to +8. Per locus that is 27 directions x 9 magnitudes x 160
images and it takes about 570 s.

### 5.1 lingshu7b, Effusion, 5 loci

`P(yes)` is the softmax of the best single-token `yes` logit against the best single-token `no` logit at
the first generated position, averaged over the 160 held-out images. Baseline `P(yes)` with no
intervention is 0.4140 at every locus, as it must be.

Magnitude, stated precisely because the code and the plan disagree: `src/intervene.py` adds
`alpha * median(||x||) / sqrt(D) * v` where `v` is unit norm. The plan says "multiples of the layer's
median activation norm". They are not the same. At `alpha = +8` and `D = 3584` the injected vector is
**0.134 of the median activation norm**, and at `D = 1280` it is 0.224.

| Locus | median activation norm | injected norm at alpha=+8, as a fraction of it | concept, max abs change in P(yes) | sham, max abs change | random directions, max abs change of the mean | concept outside the 5th to 95th random percentile band |
|---|---|---|---|---|---|---|
| `vis.last` | 3858.7 | 0.224 | 0.3308 | 0.3737 | 0.2938 | at alpha +4 and +8, but on the **inside**: the concept moved the output *less* than random directions did |
| `connector` | 35.9 | 0.134 | 0.0044 | 0.0025 | 0.0014 | at alpha -4 and -2, at an effect size of 0.004 |
| `llm.L0.vis` | 39.7 | 0.134 | 0.0022 | 0.0091 | 0.0015 | at alpha -1 and +2, at an effect size of 0.002 |
| `llm.L13.vis` | 71.4 | 0.134 | 0.0039 | 0.0111 | 0.0039 | no |
| `llm.L22.vis` | 131.9 | 0.134 | 0.0021 | 0.0017 | 0.0013 | no |

Read carefully, this is a negative result at every locus, and for two different reasons.

At `vis.last` the intervention has a large effect: `P(yes)` falls from 0.414 to 0.083 at alpha = -8. But the
sham falls further (to 0.040) and the mean of the 20 random directions falls to 0.147. The concept
direction is not doing better than a coordinate permutation of itself. At alpha = +4 and +8 the concept
direction sits outside the random band on the side of *less* disruption, which is the opposite of the
pre-registered selectivity criterion. What is being measured at this locus is a norm perturbation
degrading the vision tower, not a concept.

At the connector and the three LLM loci nothing moves at all. Across all 27 directions and all 9
magnitudes at those four loci the largest change in absolute probability is **0.0143**, from `random16` at
`llm.L13.vis` and alpha +8. The concept direction's largest effect anywhere in that set is 0.0044, and the
sham's is 0.0111. Three cells land outside the random band, but the bands
there are 0.002 to 0.006 wide, so the exceedance is statistical noise at a behaviourally meaningless
effect size. The right reading is that a perturbation at 13.4% of the median residual norm, applied to
every visual token position, does not move this model's answer. That may mean the model does not use the
direction; it may equally mean the dose is too small. The two are not separable from this data.

### 5.2 internvl3_8b, Effusion, `vis.last` only

Baseline `P(yes)` 0.3671, 160 held-out images, 27 directions.

| Direction | alpha -8 | -4 | -2 | -1 | 0 | +1 | +2 | +4 | +8 |
|---|---|---|---|---|---|---|---|---|---|
| concept | 0.3701 | 0.3692 | 0.3683 | 0.3673 | 0.3671 | 0.3660 | 0.3653 | 0.3644 | 0.3604 |
| sham | 0.3454 | 0.3565 | 0.3612 | 0.3631 | 0.3671 | 0.3688 | 0.3712 | 0.3761 | 0.3847 |
| random, mean | 0.3656 | 0.3664 | 0.3666 | 0.3662 | 0.3671 | 0.3663 | 0.3664 | 0.3659 | 0.3646 |
| random, 95th pct | 0.3924 | 0.3795 | 0.3727 | 0.3687 | 0.3671 | 0.3693 | 0.3725 | 0.3780 | 0.3884 |
| random, 5th pct | 0.3419 | 0.3543 | 0.3607 | 0.3623 | 0.3671 | 0.3629 | 0.3592 | 0.3530 | 0.3400 |
| unrelated, mean | 0.3608 | 0.3640 | 0.3655 | 0.3657 | 0.3671 | 0.3665 | 0.3674 | 0.3676 | 0.3673 |

The concept direction's maximum absolute effect is **0.0066** and it stays inside the random band at every
magnitude. The sham's maximum effect is **0.0217**, at alpha -8, more than three times larger. (The first
draft gave the sham as 0.0176, which is its excursion at alpha +8, not its maximum. Corrected.) The 95th
percentile of the 20 random directions' own maximum effects is 0.0305, so the concept direction fails the
pre-registered test by a factor of five. This is a clean preliminary
negative: at the locus where InternVL3's Effusion probe peaks (`vis.last`, AUROC 0.806, selectivity
+0.108), steering along the probe normal does not change the model's answer more than a random direction
of the same norm does.

### 5.3 llava15_7b Effusion and llavamed7b Cardiomegaly at the connector

The `vis.last` results for these two are void, for the reason given in 5.5. Their `connector` loci are
live and both are negative.

`llava15_7b`, Effusion, `connector`, baseline `P(yes)` 0.5789:

| Direction | -8 | -4 | -2 | -1 | 0 | +1 | +2 | +4 | +8 |
|---|---|---|---|---|---|---|---|---|---|
| concept | 0.5716 | 0.5760 | 0.5770 | 0.5777 | 0.5789 | 0.5794 | 0.5801 | 0.5804 | 0.5754 |
| sham | 0.5758 | 0.5786 | 0.5803 | 0.5802 | 0.5789 | 0.5794 | 0.5793 | 0.5773 | 0.5743 |
| random, mean | 0.5690 | 0.5745 | 0.5770 | 0.5779 | 0.5789 | 0.5802 | 0.5805 | 0.5816 | 0.5823 |
| random, 95th pct | 0.5890 | 0.5864 | 0.5832 | 0.5817 | 0.5789 | 0.5829 | 0.5860 | 0.5915 | 0.6014 |
| random, 5th pct | 0.5464 | 0.5638 | 0.5710 | 0.5753 | 0.5789 | 0.5768 | 0.5746 | 0.5700 | 0.5594 |

Concept maximum absolute effect 0.0072, sham 0.0045, largest of any single direction 0.0327. The concept
direction is inside the 5th to 95th random band at every magnitude. Not selective.

`llavamed7b`, Cardiomegaly, `connector`, baseline `P(yes)` 0.8711:

| Direction | -8 | -4 | -2 | -1 | 0 | +1 | +2 | +4 | +8 |
|---|---|---|---|---|---|---|---|---|---|
| concept | 0.8552 | 0.8633 | 0.8674 | 0.8704 | 0.8711 | 0.8736 | 0.8743 | 0.8771 | 0.8832 |
| sham | 0.8727 | 0.8715 | 0.8705 | 0.8713 | 0.8711 | 0.8712 | 0.8721 | 0.8735 | 0.8789 |
| random, mean | 0.8720 | 0.8714 | 0.8715 | 0.8716 | 0.8711 | 0.8713 | 0.8709 | 0.8702 | 0.8690 |
| random, 95th pct | 0.9009 | 0.8881 | 0.8796 | 0.8758 | 0.8711 | 0.8762 | 0.8802 | 0.8889 | 0.9084 |
| random, 5th pct | 0.8426 | 0.8546 | 0.8628 | 0.8672 | 0.8711 | 0.8677 | 0.8628 | 0.8523 | 0.8373 |

This is the only cell in the current D1 data with the qualitative shape the plan predicts: the concept row
is **monotone in alpha across all nine magnitudes** and moves in the right direction, adding the
cardiomegaly direction raises `P(yes)` from 0.8711 to 0.8832 and subtracting it lowers it to 0.8552, while
the sham and the random mean are flat. It still fails the pre-registered test. The effect is 0.0159 and the
random band at alpha=8 spans 0.8373 to 0.9084, so the concept direction sits comfortably inside it; the
largest effect from any single direction is 0.0615, four times the concept's. Monotone and correctly signed
is a real signal, but the random directions are noisier than the concept direction is strong, so the
criterion "must exceed the 95th percentile of the random effects" is not met.

### 5.4 qwen7b at `vis.last`: the one cell that meets the criterion, and the one that does not

Qwen-family vision towers are live loci, unlike the LLaVA case in 5.5, and both qwen7b cells are complete.
They behave differently from each other, which is itself informative.

**qwen7b, Cardiomegaly, `vis.last`**, baseline `P(yes)` 0.3083:

| Direction | -8 | -4 | -2 | -1 | 0 | +1 | +2 | +4 | +8 |
|---|---|---|---|---|---|---|---|---|---|
| concept | 0.0089 | 0.3502 | 0.1506 | 0.2720 | 0.3083 | 0.2962 | 0.1926 | 0.1719 | 0.1260 |
| sham | 0.0303 | 0.4398 | 0.4052 | 0.3332 | 0.3083 | 0.2173 | 0.1350 | 0.0278 | 0.0028 |
| random, mean | 0.0688 | 0.2474 | 0.2184 | 0.2947 | 0.3083 | 0.2997 | 0.2253 | 0.2522 | 0.0619 |
| random, 95th pct | 0.2804 | 0.5172 | 0.4596 | 0.4413 | 0.3083 | 0.4230 | 0.3641 | 0.5384 | 0.2147 |
| random, 5th pct | 0.0011 | 0.0281 | 0.0689 | 0.1853 | 0.3083 | 0.1892 | 0.1156 | 0.0569 | 0.0011 |

The concept direction is inside the random band at every magnitude and the response is not monotone: it is
higher at alpha=-4 (0.3502) than at alpha=-1 (0.2720), and adding the cardiomegaly direction *lowers*
`P(yes)`, which is the wrong sign. The sham produces a cleaner dose-response than the concept direction
does. Not selective.

**qwen7b, Effusion, `vis.last`**, baseline `P(yes)` 0.1825:

| Direction | -8 | -4 | -2 | -1 | 0 | +1 | +2 | +4 | +8 |
|---|---|---|---|---|---|---|---|---|---|
| concept | 0.0111 | 0.1564 | 0.0929 | 0.1265 | 0.1825 | 0.2588 | 0.3743 | 0.5322 | 0.3380 |
| sham | 0.0052 | 0.2989 | 0.2975 | 0.2095 | 0.1825 | 0.2352 | 0.2645 | 0.5013 | 0.6563 |
| random, mean | 0.0804 | 0.2353 | 0.2195 | 0.1968 | 0.1825 | 0.1958 | 0.2141 | 0.2462 | 0.0770 |
| random, 95th pct | 0.2354 | 0.4270 | 0.4483 | 0.3118 | 0.1825 | 0.2700 | 0.3306 | 0.4322 | 0.3401 |
| random, 5th pct | 0.0038 | 0.0419 | 0.1168 | 0.1253 | 0.1825 | 0.1258 | 0.0911 | 0.0887 | 0.0050 |

**This is the only cell in the current D1 data that meets the pre-registered criterion, and it meets it on
both available readings of that criterion.** The plan's wording, `docs/00_PAPER_PLAN.md` §5.4, is "the
effect of `v_c` must exceed the 95th percentile of the random-direction effects". That does not fix whether
"effect" means the signed response at each magnitude or one sign-blind effect size per direction, and the
two readings do not always agree, so both are reported here and in the summariser's output:

| Reading | What is compared | This cell |
|---|---|---|
| per-magnitude | at each alpha, `P(yes)` under `v_c` against the 5th-to-95th percentile of the 20 random `P(yes)` values at that same alpha. Keeps the sign | **passes.** Above the 95th at alpha +2 (0.3743 against 0.3306) and +4 (0.5322 against 0.4322); below the 5th at alpha -2 (0.0929 against 0.1168). The sign is right in all three: adding the effusion direction raises `P(yes)`, subtracting it lowers it |
| effect-size | `max_alpha` \|dP(yes)\| for `v_c`, 0.3497, against the 95th percentile of the same quantity over the 20 random directions, 0.3277. Sign-blind | **passes**, by 0.022 |

Over alpha=-1 to +4 the response is monotone and steep, 0.1265 to 0.5322.

The first draft of this record asserted the pass while `src/summarize_interventions.py` reported the same
cell as failing. The script was wrong: it compared the concept's effect against the **maximum** of the
random effects rather than their 95th percentile, which is the 100th percentile and a strictly harder test
than the one pre-registered. It has been fixed and now agrees with the prose. This is exactly the kind of
disagreement that should be resolved against the plan, not against whichever artefact was written last, and
it is recorded rather than quietly reconciled.

Four reasons not to call this a result yet. The response breaks at alpha=+8, falling back to 0.3380, so it
is not monotone over the full range. The sham reaches 0.6563 at alpha=+8, a larger absolute excursion
(0.4738) than the concept direction's maximum of 0.3497. **And so does one of the random directions, whose
largest effect is 0.4277** — the concept direction clears the 95th percentile of the random effects but is
not the largest effect in the set, and the first draft mentioned only the sham here. And this is one cell
out of eighteen, uncorrected for the multiplicity of eight magnitudes x eighteen loci, with 160 evaluation
images and no confidence interval. What it does establish is that the instrument can produce a signal,
which the near-zero LLM-locus results in 5.1 leave open.

### 5.5 A bug the intervention data exposed: `vis.last` is a dead branch in both LLaVA models

`runs/int_llava15_7b_effusion`, `runs/int_llavamed7b_cardiomegaly` and `runs/int_llavamed7b_effusion` all
report `P(yes)` **identical to every printed digit** for all 27 (or 26) directions at all 9 magnitudes:
0.5789 for llava15 Effusion, 0.8711 for llavamed Cardiomegaly, 0.9669 for llavamed Effusion. In each case
the entire 243-cell (or 234-cell) grid holds exactly one distinct value of `mean_p_yes` and exactly one of
`sd`. A perturbation at 25% of the median activation norm (D = 1024, so `8/sqrt(1024)` = 0.25 exactly)
cannot leave the logits bit-identical. The intervention is a numerically exact no-op. The third run,
`int_llavamed7b_effusion`, completed after the first draft of this section and is an independent
confirmation on a second concept: the no-op is a property of the locus, not of the concept or the fit.

The cause is in the model configs, confirmed by loading them:

| Model | `vision_feature_layer` |
|---|---|
| `llava-hf/llava-1.5-7b-hf` | **-2** |
| `models/llava-med-7b-hf` | **-2** |
| `OpenGVLab/InternVL3-8B-hf` | -1 |

Both LLaVA models take `hidden_states[-2]` as the visual feature, which is the output of vision encoder
layer 22, not layer 23. `src/registry.py` sets `n_vision_blocks = 24`, so `src/loci.py` defines
`vis.last = encoder.layers.23`, whose output the connector never reads. Writing to it changes nothing
downstream, exactly as observed.

This has two consequences and both matter:

1. **D1.** Every `vis.last` intervention on `llavamed7b` and `llava15_7b` is void. Six of the fifteen
   queued jobs are LLaVA-family and each will spend a locus producing a guaranteed null; three of the six
   already have. The locus must be replaced with `encoder.layers.22` before any LLaVA vision-tower steering
   result is reported.
2. **D2.** The `vis.last` probe numbers for `llavamed7b` and `llava15_7b` in section 2 read a
   representation the language model does not consume. They are not wrong as probe measurements, but they
   cannot be described as "the encoder output entering the connector". Worse, the vision-block subsample
   is `0, 4, 8, 12, 16, 20`, so **layer 22, the representation those two models actually pass forward, is
   not sampled at any locus.** Every LLaVA-family conclusion about the vision tower in this document is
   about a branch the model discards.

`qwen7b`, `lingshu7b` and `internvl3_8b` are unaffected: the Qwen merger consumes the last block's output
and InternVL3's `vision_feature_layer` is -1.

### 5.6 Summary of D1 so far

Eighteen complete loci at the 15:08 EDT snapshot, none of them from a run that has written a `DONE`
marker. **One meets the pre-registered selectivity criterion on both of its readings**, `qwen7b` Effusion
at `vis.last`, at three of eight magnitudes and with the four caveats in 5.4. The other seventeen do not,
in three distinct ways that need different remedies:

| Failure mode | Where | Remedy |
|---|---|---|
| The perturbation is too strong and disrupts the representation non-specifically. Every direction, sham included, moves the output as much as or more than the concept direction | Qwen-family `vis.last`: `lingshu7b` Effusion, `qwen7b` Cardiomegaly, and partly `qwen7b` Effusion, where the median activation norm is 3,859 to 4,451 and the injected fraction is 0.224 | a finer magnitude grid below alpha=4, plus the norm and principal-subspace check on the edited activation that `docs/00_PAPER_PLAN.md` §4 requires and `src/intervene.py` does not implement |
| The perturbation is too weak and nothing moves. The largest effect from any of the 26 or 27 directions is 0.0112 to 0.0615, and the concept direction's own is 0.0013 to 0.0159 | every `connector` and every LLM locus, injected fraction 0.134 | larger magnitudes, and fixing the scaling so alpha means what the plan says it means |
| The locus is not in the model's forward path, so the intervention is an exact no-op | `llavamed7b` and `llava15_7b` at `vis.last` | see 5.5 |

Two cells show a monotone correctly signed dose-response that is nevertheless inside the random band:
`llavamed7b` Cardiomegaly at the connector (effect 0.0159, random reaches 0.0615) and the alpha=-1 to +4
arm of `qwen7b` Effusion. That is the shape to look for once the magnitudes are fixed.

One cell shows the shape in the wrong place, and it is the most uncomfortable single number in the D1 data.
At `llavamed7b` Cardiomegaly `llm.L0.vis` the **sham** produces a clean monotone dose-response across all
nine magnitudes, 0.9197 down to 0.8126, an effect of 0.0585, while the concept direction is flat at 0.0013.
The sham is a coordinate permutation of the concept direction: same norm, same marginal distribution of
coordinates, none of the structure. It moves this model's answer 44 times as much as the direction it was
built from (0.0585 against 0.0013). Whatever that locus is responding to, it is not the probe's geometry.

No claim about decodability predicting steerability can be made from this data. The one selective cell is
at `vis.last` in `qwen7b`, which is also where `qwen7b`'s Effusion probe is close to its peak (0.772
against a peak of 0.780 at `llm.L0.vis`), so the single positive is consistent with decodability and
steerability tracking each other. One cell decides nothing.

### 5.7 Loci completed after the first draft of this section

These six loci finished between the 14:50 and 15:08 snapshots. None changes any conclusion; all are
recorded so the section is not silently selective about which completed loci it reports.

| Run | Locus | base `P(yes)` | concept max effect | sham max effect | random 95th pct | random max | outside the band | selective |
|---|---|---|---|---|---|---|---|---|
| `int_lingshu7b_effusion` | `vis.last` | 0.4140 | 0.3308 | 0.3737 | 0.3948 | 0.4048 | +4, +8, both above | no. Cell-for-cell identical to `int_lingshu_effusion`; see 5.0 |
| `int_lingshu7b_cardiomegaly` | `vis.last` | 0.2128 | 0.1226 | 0.1547 | 0.2152 | 0.3936 | never | no |
| `int_internvl3_8b_effusion` | `connector` | 0.3671 | 0.0042 | 0.0049 | 0.0107 | 0.0112 | never | no |
| `int_llava15_7b_effusion` | `llm.L0.vis` | 0.5789 | 0.0099 | 0.0114 | 0.0317 | 0.0363 | never | no |
| `int_llavamed7b_cardiomegaly` | `llm.L0.vis` | 0.8711 | 0.0013 | **0.0585** | 0.0337 | 0.0612 | never | no. The sham cell discussed above |
| `int_llavamed7b_effusion` | `vis.last` | 0.9669 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | never | void. Third exact no-op; see 5.5 |

---

## 6. Threats to validity

Listed in order of how much they could change a headline claim.

**1. The control task is one random draw per concept, shared across all five models.**
`control_labels()` is seeded with `args.seed + concept_index`, and every model was run with `seed = 0` and
the same concept list. So Nodule receives the *same* random type-to-label map in all five models. The
consequence is measurable. Median control AUROC per concept, pooled over models and loci, ranges from
0.601 (Edema's draw) to 0.748 (Nodule's draw), a spread of 0.147 that has nothing to do with the concepts.
Across the 9 concepts, the correlation between median control and median selectivity is **-0.744**, and
regressing selectivity on control alone gives **R² = 0.554**. Over half the cross-concept variation in the
paper's headline quantity is explained by which random label map that concept happened to draw.
"Nodule is never selective in five models" is therefore **one** unlucky control draw observed five times,
not five replications. The fix is to average `control` over many independent draws per concept and report
the selectivity with the draw-to-draw spread. Until that is run, no concept-level ranking by selectivity
should appear in the paper.

**2. The control task is not a strict Hewitt and Liang control.** Their construction assigns a random
label per *word type*, and the probe's ability to memorise it measures capacity because word identity is
exactly recoverable from the input. Images have no word-identity analogue. The substitute here,
`view x sex x age decade`, gives 39 types. Two problems follow. The types are not exactly recoverable from
the representation, so the control is not a pure capacity measure; and worse, they are *nearly* exactly
recoverable, because view position decodes at 0.998 (section 3), which means the control partly measures
"how well does this locus encode acquisition" rather than "how much can this decoder memorise". The
39-type scheme is a strict improvement on the 2-type scheme it replaced, which was degenerate (its control
column equals its view-position column to 13 decimals, `runs/probe_act_lingshu7b/probe_results.csv`), but
it is a proxy and the paper has to say so.

**3. The type distribution is severely unbalanced and two types fail to cross the split.** 39 types,
median 629 rows, largest 1,970, **smallest 3 rows**. The five smallest are `PA_F_0` (56), `AP_F_8` (55),
`PA_M_9` (8), `AP_M_9` (5), `PA_F_9` (3). `PA_F_9` appears only in test and `AP_M_9` only in train, so 37
of 39 types recur across the split. A control task whose smallest type has 3 rows is dominated by the
large types, which makes it closer to a coarse view-and-sex control than the type count suggests.

**4. NIH ChestX-ray14 labels are report-mined and noisy.** The 14 labels were extracted from free-text
reports by NLP with reported per-label accuracy well below a reader standard, and the original reports are
not public, so nothing here can be re-adjudicated. Nodule and Mass, the two concepts that fail the control,
are also two of the smallest and least reliably mined findings, and their poor showing is consistent with
label noise as well as with the capacity explanation. This record cannot distinguish the two.

**5. Mean pooling over visual tokens may hide localised signal.** Every locus is one vector per image,
mean-pooled over 144 (Qwen family), 256 (InternVL3) or 576 (LLaVA family) visual token positions. A
pneumothorax occupying a few percent of the field contributes a few percent of the mean. Max pooling was
specified in `docs/00_PAPER_PLAN.md` §3.3 and is not implemented. The 984 `BBox_List_2017` boxes in the
NIH release exist and would allow a localised readout, and none of this is used.

**6. One prompt was used for all nine concepts.** All five `runs/act_*/meta.json` record the same
extraction prompt, `Is there a pleural effusion in this chest radiograph? Answer yes or no.` The
Cardiomegaly, Nodule and Mass probes therefore read a representation formed while the model was answering
an *effusion* question. This is defensible for the vision tower and the connector, which do not see the
text. It is not defensible for the LLM loci, and it is least defensible at the `.ans` positions, which are
the answer to the effusion question specifically. Any statement about where Cardiomegaly lives inside the
language model is conditional on that prompt.

**7. The `vis.last` locus is the wrong module in two of five models.** See section 5.5. `llavamed7b` and
`llava15_7b` consume vision encoder layer 22 and this study probes and steers layer 23, and never samples
layer 22. Confirmed three times over in the D1 data, on two concepts, by bit-identical output grids.

**8. No confidence intervals anywhere.** `--bootstrap` exists in `probe.py`'s argument parser and is never
referenced afterwards. Every AUROC in this document is a point estimate from a single fit on a single
split with a single projection seed. Test positives run from 179 (Edema) to 872 (Infiltration), so the
sampling error on an AUROC is on the order of 0.02 to 0.04 for the small concepts. Several differences
reported as results in section 4.2, including the entire pair-B effect at +0.006 median dAUROC, are inside
that band.

**9. Single seed for everything.** One projection seed, one probe seed, one control draw, one split salt.
Nothing in this study has been re-run with a different seed. Two things were re-run and both are
determinism checks, not variance estimates. `probe_act_qwen7b` against `probe_qwen7b` agrees exactly in
`real`, `perm` and both nuisance columns and differs in `control` only because it used the superseded
2-type control (section 1.4); the first draft called the pair "bit-identical", which is wrong, though the
determinism conclusion it was used for stands for the four columns that match. `int_lingshu7b_effusion`
against `int_lingshu_effusion` at `vis.last` agrees in all 243 cells to the last digit, across separate
processes on different GPUs (section 5.0). Both re-runs used the same seed, so neither says anything about
sampling variance, and the study still has no estimate of it.

**10. Class imbalance is not what the manifest builder intended.** The builder balances positives to
negatives 1:1 *within* each concept's selection, then takes the union across nine concepts and
deduplicates. The union's per-concept prevalence is 2.6% to 16.2% (section 1.2), not 50%. AUROC is
prevalence-insensitive so the metric survives, but the L2-regularised logistic fit at fixed `C = 1.0` is
not: the effective regularisation strength per positive example differs by a factor of six between Edema
and Infiltration, at a capacity that is supposed to be matched. Edema is the concept with the highest
selectivity in all five models and the fewest positives.

**11. The negative class is mixed.** Between 50% and 59% of each concept's negatives are `No Finding`
images and the rest carry a different finding. A probe scoring 0.75 may be separating "this finding" from
"healthy" as much as from "a different finding", and the design contains no lesion-versus-lesion contrast
that would separate the two.

**12. Only three of eleven promised validity classes are actually implemented.** `docs/00_PAPER_PLAN.md`
§2.2 lists eleven. This record contains a permutation null, a randomised control task, a nuisance probe
set and a matched-architecture comparison model. It contains **no** random-encoder floor (nothing in
`src/` implements one), no planted ground truth, no input-space corroboration and no localisation check.
The paper cannot claim the full battery on this evidence.

**13. D1's magnitude is smaller than the plan states.** The plan pre-specifies magnitudes as multiples of
the layer's median activation norm. The code injects `alpha * norm / sqrt(D)`. At the strongest magnitude
run, `alpha = 8`, the perturbation is 13.4% of the median norm at a 3584-wide locus. A null result at that
dose is not a null result at the dose the plan specifies, and section 5 says so.

**14. D1 evaluates 160 images per locus.** `--n-eval 160`, so a mean `P(yes)` has a standard error of
roughly `sd / sqrt(160)`. The reported `sd` at `lingshu7b` `vis.last` alpha=-4 is 0.154, giving a standard
error near 0.012. The sub-0.005 effects in section 5.1's LLM loci are far below that, so those cells are
correctly read as "no measurable effect", not as "a small effect".

---

## 7. What is still missing before the paper can be written

Ordered by whether the paper can be written without it.

### Blocking

1. **Finish D1.** 15 jobs queued, 0 with a `DONE` marker; at the 15:08 EDT snapshot 13 of the planned 105
   loci were complete, plus 5 more from the out-of-queue crashed run. Nothing in section 5 can be reported
   as final until the markers exist, and the counts must be regenerated with
   `src/summarize_interventions.py` rather than quoted from this document.
2. **Fix `vis.last` for the LLaVA family** to `encoder.layers.22`, add layer 22 to the vision-block
   subsample, and re-extract and re-probe `llavamed7b` and `llava15_7b`. Every LLaVA vision-tower number
   in section 2 and every LLaVA `vis.last` intervention is currently reading a discarded branch.
3. **Repeat the control task over many draws.** At minimum 20 independent type-to-label assignments per
   concept, reporting selectivity as `real - mean(control)` with the control's spread. Threat 1 shows this
   is not a refinement: it decides whether "Nodule is never selective" is a result or an artefact of one
   draw.
4. **Bootstrap confidence intervals on every AUROC.** The `--bootstrap` flag is dead code. The pair-B
   result (+0.006 median dAUROC) cannot be reported at all without them.
5. **Re-run D1 at the dose the plan specifies.** Either fix the scaling to true multiples of the median
   norm, or keep the current scaling and restate the pre-registered magnitudes. A null at 13% of the norm
   does not falsify use of the direction.

### Needed for the claims the plan makes

6. **The random-encoder floor.** Promised in the battery, absent from `src/`. Without it the paper cannot
   claim the full eleven-class battery, and it is one of the two classes the survey found in only 5 of 121
   records.
7. **D3, the planted anchor, in all three parts.** Not started, no code. D3c, the image-space against
   representation-space dose comparison, is the plan's strongest single result and nothing exists for it.
   `runs/smoke_lingshu.log` shows a planted-image smoke test ran and produced per-locus cosines
   (`connector` 0.9193, `llm.L0.vis` 0.9284, `llm.L14.vis` 0.9815, `llm.L27.ans` 0.9975), which is
   encouraging but is one image and not an experiment.
8. **A nuisance-controlled finding probe.** Section 3 shows the representation is saturated with view and
   sex. Nothing yet tests whether the finding probe's direction is separable from them. The two obvious
   tests are refitting the finding probe on features with the view and sex directions projected out, and
   reporting the angle between `v_finding` and `v_view`.
9. **Concept-appropriate extraction prompts,** or at least a sensitivity run on one model showing how much
   the LLM-locus curves move when the prompt names the concept being probed. Threat 6 currently applies to
   every LLM number for eight of the nine concepts.
10. **A second dataset.** The plan requires every headline claim to replicate on a second dataset. Only
    NIH ChestX-ray14 exists here. VinDr-CXR and PadChest were the candidates and neither is downloaded.
11. **Max pooling and a localised readout,** to test threat 5. The 984 NIH bounding boxes are on disk and
    unused.

### Needed for the paper to be positioned

12. **Read arXiv:2604.08333 and arXiv:2603.06054 in full.** `README.md` still lists this as not done. The
    related-work delta table cannot be written without it, and the plan's Claim C explicitly depends on
    verifying whether Theodoridis et al. already test decodability against steerability.
13. **A statement of what the current evidence actually supports.** As of this record, the defensible
    claims are: (a) the controlled curve is far below the uncontrolled curve, median selectivity +0.076
    against a median AUROC of 0.751; (b) nuisance variables dominate at every locus, view position 0.996
    to 0.999 everywhere; (c) one clean architecture-matched pair shows a medical advantage that grows with
    LLM depth, +0.032 at the vision tower to +0.079 at the last layer; (d) two concepts, Nodule always and
    Mass usually, fail the capacity control, subject to threat 1. The dissociation claim (Claim C) has 18
    complete loci, no finished run, one selective cell out of eighteen, three of those eighteen void
    because the locus is off the forward path, and a magnitude scaling that is wrong in both directions at
    once, so it currently supports nothing either way. Claims D and E cannot be made at all yet.
