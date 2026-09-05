# Answer-encoding response

Run: `20260904T230152Z-21dad2f72726-answer-encoding` evaluates the registered Qwen answer-encoding crossover from source `21dad2f727261491c372bf9ffccf2dafd7c01413`. The 200 intervention patients receive six questions, three encodings, and two signed doses; 200 patient-disjoint calibration patients receive clean A/B questions. All 198,000 intervention outcomes and 2,400 calibration outcomes completed in 6,402 seconds on one A100. These patients come from the accepted ownership cohort, making this a paired sensitivity study.

## Calibration and mapping response

Four questions meet the registered capability criterion in both A/B mappings. All calibration AUROC estimates use the finding-present logit margin and 2,000 patient bootstrap draws. Eligibility uses the one-sided 95% lower bound and at least ten patients in each class.

| Question | Positive patients | A-present AUROC | B-present AUROC | A-present lower bound | B-present lower bound | Eligibility |
|---|---:|---:|---:|---:|---:|---|
| Effusion | 11 | 0.5125 | 0.5111 | 0.3872 | 0.3967 | Baseline ineligible |
| Atelectasis | 15 | 0.6586 | 0.6441 | 0.5448 | 0.5372 | Eligible |
| Pneumothorax | 13 | 0.6508 | 0.6524 | 0.5088 | 0.5113 | Eligible |
| Cardiomegaly | 8 | 0.5661 | 0.5811 | 0.4677 | 0.4757 | Insufficient labels |
| Mass | 15 | 0.6533 | 0.6542 | 0.5198 | 0.5231 | Eligible |
| Nodule | 21 | 0.6347 | 0.6327 | 0.5210 | 0.5196 | Eligible |

Observation: the primary mapping-even minus mapping-odd response-energy contrast is **-2.1786**, with a joint patient-bootstrap 95% interval of **[-2.4061, -1.9810]**. The endpoint sums over the four independently eligible questions and averages over the six clinical directions, using the unbiased squared-mean estimator. The corresponding all-question descriptive contrast is -2.9329 [-3.2374, -2.6636].

The negative primary contrast means that, in this aggregate, squared mean responses are larger in the component that reverses raw A-minus-B sign when the finding-to-letter mapping is exchanged. The registered encoding-even-beyond-controls condition is false. All 20 random-direction contrasts are also negative, ranging from -5.4773 to -0.1769; the sham contrast is -2.3848. On the intervention patients, clean clinically oriented A/B rankings have Spearman correlations from 0.9594 to 0.9804 across the six questions.

Calibration mean raw A-minus-B margins have opposite signs across the mappings for every question, with absolute differences between their magnitudes ranging from 0.0050 to 0.1200; this baseline asymmetry remains part of the component interpretation.

On the independent calibration patients, clinically oriented A/B Spearman correlations are 0.976613 for Effusion, 0.970729 for Atelectasis, 0.960770 for Pneumothorax, 0.979586 for Cardiomegaly, 0.988116 for Mass, and 0.979468 for Nodule. These use the clean margins in `calibration.csv`; the summary JSON's `baseline_spearman` field describes the intervention cohort.

## Clinical competition across prompts

The descriptive Mass clinical-competition margin changes sign across the standard and coded prompts. Each margin subtracts the largest of the five other clinical-direction mean effects from the matched Mass-direction effect, in clinically oriented logit coordinates at alpha +0.25.

| Prompt | Mass clinical margin | Pointwise bootstrap 95% interval | Mass effect | Largest random effect | Absolute sham effect |
|---|---:|---|---:|---:|---:|
| Standard yes/no | -0.7181 | [-0.7956, -0.6412] | 2.8625 | 3.0412 | 0.5869 |
| A-present | 0.2825 | [0.2269, 0.3406] | 1.6125 | 1.1650 | 0.7550 |
| B-present | 0.2900 | [0.2300, 0.3506] | 1.7469 | 1.2581 | 1.1475 |

The Mass response therefore motivates a prospective confirmation of prompt-conditioned clinical specificity. The raw A/B exchange preserves a positive clinical margin in both mappings, while the standard question places a competing direction above Mass.

## Interpretation limits

The mapping decomposition concerns aggregate response energy; generic random perturbations also produce mapping-odd-dominant responses. It does not establish clinical-direction specificity or encoding invariance. Effusion and Cardiomegaly remain in the descriptive tables but do not enter the primary endpoint. The standard and coded prompts differ in wording and response instructions as well as answer symbols, so the Mass contrast describes the prompt package. Mass was identified from six questions on reused patients, and its pointwise intervals with 20 random controls are not the original simultaneous ownership test.

The complete registered descriptive outputs remain in `answer-encoding-summary.json`: `encodings.<encoding>.doses` contains both signed doses, all clinical response matrices, clinical-competition margins and random/sham effects; `encodings.<encoding>.metric_changes` contains steered AUROC and Brier summaries; `crossover_plus_025` contains both component matrices, the all-question contrast, the naive-energy diagnostic and direction-wise control intervals. Baseline discrimination uses answer scores against the original image labels.

Gate decision: `PASS`; scientific state `OBSERVED`. Independent internal verification confirmed complete grids, patient separation, score mapping and the primary point estimate. The pinned `claude-fable-5-1` reviewer accepted the interpretation with no required actions using medium effort and read-only transport. Receipt: `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T005436988285Z.json`.

Next step: does Mass retain its prompt-conditioned clinical advantage on independent patients under a prospectively locked control family? Register the confirmation design before collecting new outcomes, then use its result to select the concept and readout for the natural-displacement comparison.

Evidence: the immutable run's `artifacts/answer-encoding-summary.json`, `artifacts/per-image.csv`, `artifacts/calibration.csv`, `artifacts/meta.json`, `artifacts/preflight.json`, terminal statuses, and `metadata.env`.
