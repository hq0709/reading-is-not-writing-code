# Mass direction specificity across prompts

Run: `20260905T015725Z-1c9820d7b576-mass-prompts` evaluated seven prompt conditions on 200 independent confirmation patients and 200 patient-disjoint calibration patients. It completed 179,200 registered outcomes in 8,783 seconds on one A100, finishing at 2026-09-05 04:24:13 UTC. The immutable source is `1c9820d7b5765c2ec220d9f4afaafd4c743e43e4`.

## Observation

Mass exceeds all five fixed clinical directions under the two original `show` A/B prompts, with simultaneous lower bounds of at least 0.1675 and 0.1462. Both prompts have larger random-direction effects, so the registered specificity conjunction does not hold. Exchanging `show` for `is` reverses Mass's clinical competition margin in both coded mappings. Effusion is the largest clinical competitor in every cell.

Effects and clinical margins are in finding-present logit units. The clinical margin subtracts the largest of the five other clinical effects. Its displayed intervals are descriptive paired-patient percentile intervals; the coded-cell simultaneous bounds belong to the fixed 20-comparison family.

| Condition | Mass effect | Clinical margin [95% interval] | Minimum simultaneous lower bound | Largest random effect | Absolute sham effect |
|---|---:|---:|---:|---:|---:|
| `show/yes_no` | 2.3800 | 0.2369 [0.1362, 0.3394] | — | 3.0531 | 0.1225 |
| `show/A_present` | 1.7019 | 0.2525 [0.1869, 0.3175] | 0.1675 | 2.2100 | 0.2931 |
| `show/B_present` | 1.8188 | 0.2338 [0.1675, 0.3006] | 0.1462 | 2.2169 | 0.2850 |
| `is/yes_no` | 2.6600 | −0.4556 [−0.5350, −0.3794] | — | 3.6031 | 0.6556 |
| `is/A_present` | 1.9225 | −0.1087 [−0.1606, −0.0575] | −0.1751 | 2.4750 | 0.6075 |
| `is/B_present` | 2.0019 | −0.2006 [−0.2531, −0.1487] | −0.2686 | 2.4944 | 0.5944 |
| `anchor` | 2.9987 | −0.7438 [−0.8338, −0.6575] | — | 4.1312 | 0.9856 |

`random95` is the largest random direction in every cell; `random90` is also larger than Mass in both original coded prompts. Their same-cell reference ranks for Mass are 3/120 in both `show` coded cells, 4/120 in `is/A_present`, and 5/120 in `is/B_present`. The registered decision requires Mass to exceed all 119 random effects.

### Wording and response instructions

Under `show`, explicit yes/no and the two A/B mappings have similar observed clinical margins. Their direct paired differences include zero. Under `is`, A/B instructions improve the clinical margin relative to explicit yes/no, while both margins remain negative. Explicit yes/no instructions also improve the `is` margin relative to the original anchor. This factorial comparison resolves the wording/instruction confounding in the discovery prompts.

| Paired clinical-margin contrast | Difference [descriptive 95% interval] |
|---|---:|
| `show/A_present − show/yes_no` | 0.0156 [−0.0250, 0.0544] |
| `show/B_present − show/yes_no` | −0.0031 [−0.0425, 0.0350] |
| `is/A_present − is/yes_no` | 0.3469 [0.3150, 0.3788] |
| `is/B_present − is/yes_no` | 0.2550 [0.2219, 0.2881] |
| `is/yes_no − show/yes_no` | −0.6925 [−0.7369, −0.6494] |
| `is/A_present − show/A_present` | −0.3612 [−0.3906, −0.3319] |
| `is/B_present − show/B_present` | −0.4344 [−0.4644, −0.4056] |
| `is/yes_no − anchor` | 0.2881 [0.2631, 0.3137] |

### Capability and predictive behavior

All seven cells meet the registered capability criterion on the independent calibration cohort, which contains 15 positive and 185 negative patients. Calibration AUROCs range from 0.6420 to 0.6542; all 2,000 bootstrap draws are valid, and one-sided lower bounds range from 0.5043 to 0.5231. Eligibility follows this predeclared calibration rule.

The confirmation cohort contains 12 positive and 188 negative patients. Its descriptive baseline AUROCs range from 0.4938 to 0.5055, with broad intervals. Mass steering raises finding-present probabilities but increases Brier error in every cell; every paired AUROC-change interval spans zero.

| Condition | Baseline AUROC | Steered AUROC | Baseline Brier | Steered Brier | Brier change [95% interval] |
|---|---:|---:|---:|---:|---:|
| `show/yes_no` | 0.4938 | 0.5288 | 0.0601 | 0.1530 | 0.0929 [0.0691, 0.1163] |
| `show/A_present` | 0.5055 | 0.5461 | 0.0858 | 0.2949 | 0.2091 [0.1829, 0.2338] |
| `show/B_present` | 0.4962 | 0.5468 | 0.0943 | 0.3131 | 0.2188 [0.1911, 0.2445] |
| `is/yes_no` | 0.4942 | 0.4989 | 0.0644 | 0.2238 | 0.1594 [0.1305, 0.1871] |
| `is/A_present` | 0.4951 | 0.5202 | 0.0985 | 0.3866 | 0.2881 [0.2589, 0.3153] |
| `is/B_present` | 0.4940 | 0.5441 | 0.1067 | 0.4015 | 0.2948 [0.2644, 0.3227] |
| `anchor` | 0.4956 | 0.4807 | 0.0630 | 0.2189 | 0.1559 [0.1269, 0.1836] |

## Gate decision

The complete registered analysis gives `cross_wording_specificity=false` and `discovery_wording_replication=false`, with `capability_status=eligible`. Clinical-direction competition reproduces under the original coded wording, while the larger random-control family prevents the registered specificity claim. The scientific result is `OBSERVED`; internal verification and pinned result review passed. The prescribed next concept is Consolidation.

## Next step

Does a fresh, metadata-matched Consolidation cohort supply a positive paired behavioral opportunity? Register the 100-patient clean-response gate before component/complement interventions, using the available original NIH image pool and full source disease labels.

## Limitations

This comparison fixes one model, visual locus, dose and finding. Descriptive contrasts retain their pointwise uncertainty, and intervals spanning zero do not establish equivalence. The small number of positive confirmation patients limits discrimination precision. Prompt-conditioned clinical competition and mean answer shifts concern these measurements; the random controls and predictive metrics determine the supported interpretation.

## Evidence

The immutable run's `artifacts/mass-prompt-specificity-summary.json` contains all 20 simultaneous comparisons, all direction effects, paired contrasts, calibration metrics, baseline correlations and predictive-metric changes. The protocol is `docs/RESEARCH_PLAN.md#qwen-mass-prompt-conditioned-specificity`.

Internal receipt: `/home/qingchan/data/concept-flow/state/mass-confirmation-internal-20260905T042600Z/smallreceipt.json`. Nine decision-bearing checksum entries passed once; all 179,200 rows, cohort regeneration, labels and score orientations passed. Independent clinical point estimates agree within 4.44e−16. Replay preserves all simultaneous bounds and route flags exactly; 114 other numeric fields differ by at most 1.67e−16 under the recorded single-thread replay environment.

Pinned acceptance: `docs/reviews/qwen7b-mass-prompt-specificity-results.md`; receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T051404447778Z.json`.
