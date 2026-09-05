# Paired behavioral opportunity across model configurations

Run: accepted Qwen and LLaVA configurations evaluate the same 100 within-patient Consolidation-positive/negative pairs with the prompt `Is there consolidation in this chest radiograph? Answer yes or no.` Each model contributes 200 image outcomes. The LLaVA run completed in 38 seconds on one A100, with command, dispatcher and cleanup statuses zero.

## Observation

Each model's opportunity criterion is a strictly positive one-sided 95% lower bound on its patient-weighted positive-minus-negative yes/no margin. All patients remain in the analysis, using the same 5,000 stored patient-bootstrap draws.

| Model configuration | Mean paired margin gap | One-sided 95% lower bound | Descriptive 95% interval | Within-patient concordance |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-7B-Instruct | -0.1175 | -0.3987 | [-0.4500, 0.2088] | 0.545 |
| LLaVA-1.5-7B | -0.0712 | -0.1338 | [-0.1463, 0.0012] | 0.455 |

The direct patient-paired concordance difference, LLaVA minus Qwen, is **-0.090**, with descriptive 95% interval **[-0.210, 0.025]**. This interval spans zero. LLaVA has 36 positive gaps, 45 negative gaps and 19 ties; Qwen has 54 positive gaps, 45 negative gaps and one tie. Concordance gives half credit to ties.

LLaVA's mean sigmoid-margin gap is -0.01508. The 52 pairs with a `No Finding` negative image have mean margin gap -0.15385; the other 48 pairs have mean gap 0.01823. These are descriptive strata of the complete cohort.

## Gate decision

Both model-specific opportunity flags are false. Positive paired Consolidation opportunity remains unresolved for these configurations in this matched population. Independent internal terminal verification and pinned result review passed.

## Next step

How does direction-by-question semantic competition transfer to the accepted LLaVA configuration? Register its controlled direction/question comparison using the accepted readout and intervention evidence.

## Limitations

The paired population is defined by within-patient agreement on view, sex, integer age and the other thirteen disease labels. The comparison follows the Qwen result and uses the models' native training and preprocessing. Margins retain their within-model coordinates; the direct model contrast uses concordance, with half credit for ties at the finite precision of the registered BF16 scoring. The intervals support uncertainty in positive opportunity and in the model difference, rather than equivalence or a zero-response mechanism.

## Evidence

- LLaVA run: `20260905T064036Z-1af90218a1d8-llava-opportunity`; immutable source `1af90218a1d8022bce66452826fa408ef146e432`.
- LLaVA artifacts: `/home/qingchan/data/concept-flow/runs/20260905T064036Z-1af90218a1d8-llava-opportunity/artifacts/`.
- Qwen run and acceptance: `docs/QWEN_PAIRED_OPPORTUNITY_RESULTS.md`; `docs/reviews/qwen7b-paired-opportunity-results.md`.
- Registration: `docs/reviews/llava-paired-opportunity-registration.md`.
- Internal terminal receipt: `/home/qingchan/data/concept-flow/state/llava-opportunity-internal-20260905T064202Z/receipt.json`. Eleven decision-bearing new-run files passed the immutable checksum manifest once; source-record identities, saved reference copies, complete patient arrays and shared bootstrap indices passed. Maximum independent numerical discrepancy was `5.55e-17`.
- Pinned result receipt: `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T064601848008Z.json`; canonical `claude-fable-5-1`, medium effort, valid and read-only, exit 0, clean source `1af90218a1d8022bce66452826fa408ef146e432`.
