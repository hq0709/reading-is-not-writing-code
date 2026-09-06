# LLaVA reader transfer across sampling frames

## Run

This descriptive synthesis combines accepted pilot `20260905T093108Z-42a43207c848-llava-validation` and independent-patient diagnostic `20260905T232313Z-2a95dfbed888-llava-yesno`. It reads existing summaries and cohort records, with no additional model evaluation, fit or bootstrap.

## Observation

| Reader quantity | Pilot | Independent cohort |
|---|---:|---:|
| Clinical AUROC | 0.8196 | 0.6664 |
| Mean of twenty control-task AUROCs | 0.6758 | 0.6775 |
| Selectivity | 0.1438 | -0.0111 |
| Selectivity 95% interval | [0.0658, 0.2163] | [-0.1541, 0.1185] |
| Effusion-positive / total patients | 34 / 700 | 25 / 700 |

At the point-estimate level, the selectivity decrease is principally a reduction in clinical AUROC; the mean control-task benchmark is nearly unchanged. The controls predict fixed random binary labels assigned to acquisition/demographic types (`view_AP`, `sex_M`, clipped age decade). Their AUROCs evaluate their own synthetic labels, not Effusion. Selectivity is relative to this control family, rather than a direct estimate of clinical information after removing confounding.

The pilot selects a representative image per patient within the original manifest. That manifest samples disease-positive images and `No Finding` negatives across concepts. The independent cohort instead selects the lowest image identifier per validation patient absent from the entire manifest. Identifier allocation is label-blind, but the sampling frames and eligible representative images differ.

Selected recorded characteristics among Effusion-negative patients are:

| Characteristic | Pilot (n=666) | Independent cohort (n=675) |
|---|---:|---:|
| AP | 119 (17.9%) | 79 (11.7%) |
| Male | 360 (54.1%) | 364 (53.9%) |
| No Finding | 477 (71.6%) | 490 (72.6%) |
| Infiltration | 54 (8.1%) | 89 (13.2%) |
| Mass | 35 (5.3%) | 19 (2.8%) |
| Nodule | 45 (6.8%) | 22 (3.3%) |
| Pneumothorax | 12 (1.8%) | 2 (0.3%) |

Cofindings overlap. Counts describe composition and are not an exhaustive adjustment or an inferential comparison.

## Gate decision

The accepted diagnostic remains `PASS / OBSERVED`, with reader replication false, opportunity not established and wording dependence unresolved. The pilot's readable-direction/weak-answer separation is a cohort-specific observation; it did not become a replicated controlled-availability result on the independent patients. Existing intervention evidence retains its registered scope.

## Limitations

There is no accepted interval for the between-cohort AUROC or selectivity difference. Sampling variation, clinical/acquisition composition, transfer of the frozen ranking and selection-related optimism from following up the stronger pilot anchor remain unresolved explanations. Fewer positive patients reduce precision; prevalence alone does not explain an AUROC difference. Stable mean control performance does not establish stable disease difficulty. The cohort JSONs have different coverage of additional diagnoses, so only shared named fields are compared here. Marginal counts cannot attribute the clinical-AUROC change to composition, and nonreplication does not establish absent information or equivalence.

## Next step

How should this sampling-frame boundary qualify the pilot-to-independent-cohort narrative? Integrate it with the accepted diagnostic in the manuscript; a causal or standardized comparison would require a separately specified analysis.

## Evidence

- Pilot `artifacts/qualification.json`: `questions[0]` (`question=Effusion`), `reader_auroc=0.8195548489666137`, `reader_auroc_ci95=[0.7424204891906874, 0.8920370150136474]`, `selectivity=0.14376553503387168`. The arithmetic mean of its twenty `control_auroc` entries is 0.6757893139327421.
- Pilot `artifacts/cohort.json`, `calibration`; independent run `artifacts/cohort.json`, `index`. All quoted counts use existing rows stratified by their recorded Effusion label.
- The run store is `/home/qingchan/data/concept-flow/runs/`; accepted reports are `docs/LLAVA_VALIDATION_OPPORTUNITY_RESULTS.md` and `docs/LLAVA_YESNO_IMAGE_DIAGNOSTIC_RESULTS.md`, with their linked terminal and reviewer receipts.
- Sampling and control definitions: `src/build_manifest.py`, `src/probe.py`, `src/llava_readout_diagnostic.py`.
