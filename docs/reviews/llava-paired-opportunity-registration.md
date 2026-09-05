# LLaVA paired behavioral opportunity registration

Run: implementation and registration review for the accepted LLaVA configuration on the fixed 100-patient Consolidation cohort.

Observation: the accepted source interfaces provide the complete ordered cohort, Qwen image outcomes, sixteen validation identities and 5,000 patient-bootstrap draws. The LLaVA asset receipt identifies revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`; its stored processor uses RGB, shortest-edge resize and center crop at 336 pixels, CLIP normalization, patch size 14 and default visual feature selection. Source-interface inspection used the existing receipts and small metadata artifacts.

The registered within-model endpoint is the mean positive-minus-negative yes/no margin, with opportunity requiring a strictly positive fifth-percentile patient-bootstrap bound. The direct model comparison is the mean patient-paired concordance difference, LLaVA minus Qwen, with a descriptive two-sided interval from the same draws. Each endpoint retains all 100 patients. Model-specific preprocessing is part of the compared configurations.

Local verification completed 15 focused tests and a 238-test full suite in 90.517 seconds, with zero failures and six environment-specific skips. The focused tests cover the real CPU serialization chain, cohort and model identities, shared draws, strict gate boundaries, concordance ties, scoring orientation, preflight stops and immutable launcher binding. Bash syntax validation passed. Independent implementation verification passed the focused suite in 13.018 seconds. Actual server CPU integration accepted the cohort, model source, sixteen validation rows and native processor configuration before any scientific image scoring.

Gate decision: `PASS`, ready for dispatch. The independent terminal validator passed a valid synthetic baseline and rejected 23 targeted mutants. Pinned review returned `LLAVA_PAIRED_OPPORTUNITY_REGISTRATION PASS` for executable source `398dc0c197650e1bec7dacae97813fceeefc6057`.

Next step: does LLaVA supply positive paired Consolidation opportunity on the fixed cohort? Execute the 200-outcome gate through the immutable dispatcher, then accept the complete result through internal and pinned terminal review.

Evidence:

- Protocol: `docs/RESEARCH_PLAN.md#llava-paired-behavioral-opportunity`.
- Ordered cohort: `/home/qingchan/data/concept-flow/runs/20260905T052025Z-9bc918b414ff-paired-opportunity/artifacts/registered-pairs.json`.
- Accepted Qwen reference: `/home/qingchan/data/concept-flow/runs/20260905T053520Z-f0317eaf3f4e-paired-opportunity/artifacts/`; source `f0317eaf3f4e3d97a18c266eb3d84b0f1c09baec`; acceptance in `docs/reviews/qwen7b-paired-opportunity-results.md`.
- Model asset: `/home/qingchan/data/concept-flow/models/huggingface/asset-receipt.json`.
- Accepted LLaVA input identity: `/home/qingchan/data/concept-flow/runs/20260902T191411Z-ffd523c464c8-f99e2f39/artifacts/input-verification.json`; source `ffd523c464c84417a93c5a6d0a34e5b74e55e76e`.
- Pinned registration receipt: `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T063744124554Z.json`; canonical `claude-fable-5-1`, medium effort, valid and read-only, tools disabled, exit 0. Server checkout stayed clean at the reviewed source.
