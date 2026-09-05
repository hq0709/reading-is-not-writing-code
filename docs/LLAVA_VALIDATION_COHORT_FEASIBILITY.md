# LLaVA validation cohort feasibility

## Run

Read-only metadata inspection used the accepted NIH manifest, LLaVA probe assignments, sixteen fixed implementation-preflight rows and the Qwen ownership cohort receipt. Sources are `/home/qingchan/data/concept-flow/datasets/nih-chestxray14/manifest.csv`, first-gate run `20260902T191411Z-ffd523c464c8-f99e2f39` and ownership run `20260904T125710Z-a3bd883540eb-causal-ownership`.

## Observation

The manifest contains 18,212 training rows from 5,783 patients, 2,674 `val` rows from 855 patients and 5,343 test rows from 1,659 patients. The sixteen preflight rows identify sixteen distinct validation patients. Their exclusion leaves 839 eligible patients.

The fixed seed-20260913 permutation of sorted string patient IDs allocates 700 patients to calibration and 100 to the write screen, with 39 unused. Each selected row is the patient's lexicographically smallest validation row. Preflight, calibration, write and the accepted ownership cohort are pairwise disjoint in both patients and rows.

| Concept | Calibration positive / negative | Write positive / negative |
|---|---:|---:|
| Effusion | 34 / 666 | 7 / 93 |
| Atelectasis | 43 / 657 | 7 / 93 |
| Pneumothorax | 12 / 688 | 1 / 99 |
| Cardiomegaly | 25 / 675 | 3 / 97 |
| Mass | 39 / 661 | 8 / 92 |
| Nodule | 48 / 652 | 12 / 88 |

All twenty accepted type-control assignments cover the calibration rows, with at least 147 patients in the smaller assigned class. The fixed allocation's calibration anchors are `9691 / 00009691_000` and `2995 / 00002995_000`; its write anchors are `6752 / 00006752_000` and `7202 / 00007202_002`.

The accepted activation archive contains a `(26229, 1024)` float16 pooled `vis.last` array. Its direction bundle stores projection, scale, clinical coefficients and unit normals. Clinical ranking scores therefore require no refits. Twenty shared train-only type-control fits supply the missing validation control scores. Existing test-only bootstrap arrays retain their accepted evidentiary role.

## Gate decision

Metadata feasibility is `PASS`: all six calibration questions meet the minimum ten-patients-per-class requirement before outcomes. Sparse positive labels in the write cohort limit its descriptive AUROC precision; the registered write endpoint uses paired perturbation responses over all 100 patients. Scientific eligibility and write opportunity await the prospective experiment.

## Next step

Implement the allocation, reader/capability qualification and controlled write screen specified in `docs/RESEARCH_PLAN.md#llava-validation-intervention-opportunity`, then complete independent internal and pinned registration review.
