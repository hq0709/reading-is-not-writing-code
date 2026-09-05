# Research state

Updated: 2026-09-05. Scientific state uses `PLANNED`, `RUNNING`, `FAILED`, and `OBSERVED`; gate disposition uses `READY`, `BLOCKED`, and `UNAVAILABLE`.

## Active research stage

- Run: `20260905T015725Z-1c9820d7b576-mass-prompts`, source `1c9820d7b5765c2ec220d9f4afaafd4c743e43e4`, completed all 179,200 outcomes in 8,783 seconds at 04:24:13 UTC.
- Observation: all seven calibration cells are eligible. The original `show` A/B conditions have positive clinical-family simultaneous bounds, but their Mass effects remain below the registered random maxima. Both `is` coded clinical margins are negative. Direct paired comparisons show prompt-conditioned clinical competition.
- Scientific state: independent-patient Mass confirmation `OBSERVED`; Consolidation run `20260905T052025Z-9bc918b414ff-paired-opportunity` is `FAILED` at measurement preflight, before scientific scoring. Its fixed 100-patient opportunity remains unevaluated.
- Gate disposition: Mass internal verification and pinned result review `PASS`; `cross_wording_specificity=false`, `discovery_wording_replication=false`. Consolidation opportunity protocol and implementation passed independent internal and pinned registration review. The local suite passed 219 tests with six environment-specific skips.
- Next step: does the fixed, metadata-matched Consolidation cohort supply a positive paired behavioral opportunity? Complete the pre-scientific-outcome mapping-validation amendment review, then run the unchanged 200-outcome image comparison on one A100 with a 15-minute cap.
- Writer ownership: project task `01a06ec5-fec9-7241-b066-72c1bc5632b8` coordinates research, evidence and delivery and owns code and experiment dispatch. Paper task `01a06afe-5c66-7a80-b922-b3ede71eab3e` owns the separate paper repository. Existing unrelated local edits remain with their author.
- Supervision: heartbeat `concept-flow` is ACTIVE and bound to the project task. Its current 10-minute interval reflects short review and experiment handoffs; adjust it to the next decision window as work changes, remaining quiet on unchanged state.
- Evidence: `docs/MASS_PROMPT_SPECIFICITY_RESULTS.md`; `docs/reviews/qwen7b-mass-prompt-specificity-results.md`; internal receipt `/home/qingchan/data/concept-flow/state/mass-confirmation-internal-20260905T042600Z/smallreceipt.json`; pinned result receipt `review-20260905T051404447778Z.json`.
- Paired-cohort availability: accepted source metadata and inventory give 120 eligible Consolidation patients absent from the entire original manifest, matched within patient on view, sex, age and 13 other disease labels. Deterministic selection precedes model loading; `No Finding` remains a descriptive stratum. Source evidence: `docs/PAIRED_OPPORTUNITY_FEASIBILITY.md` and `docs/reviews/paired-opportunity-preparation.md`.
- Paper handoff: accepted Mass evidence and writing objectives have been sent for integration into paper main `4706ea5`. The paper task owns manuscript changes, build and independent writing review; project coordination tracks any resulting scientific gaps.
- Protocol: `docs/RESEARCH_PLAN.md#qwen-paired-behavioral-opportunity`.
- Registration: `docs/reviews/qwen7b-paired-opportunity-registration.md`; pinned receipt `review-20260905T051705042400Z.json`.
- Current validation: `docs/reviews/qwen7b-paired-opportunity-validation.md`; internal design review accepts reuse of the pre-existing generic semantic-mapping standard. Implementation and pinned review are pending; the original run remains in the run ledger.

## Bootstrap gate

- Run: infrastructure bootstrap and immutable smoke run `20260902T071539Z-7a4e7552d9e1-75d3510c`.
- Observation: Git handoff, fixed environment, reviewer transport, ARIS pin, tmux persistence, immutable dispatch, fetch, and safety acceptance have terminal receipts.
- Gate decision: `PASS`.
- Gate disposition: `READY`.

## First hard gate and accepted scientific sequence

- Run: the first hard gate, two registered cross-cell gates, and the independent-row Qwen direction-specificity supplement completed from immutable commits with fixed model and dataset identities, forward-path hook evidence, patient bootstrap and control protocols, probe-normal intervention, terminal receipts, and pinned read-only review.
- Observation: all three final-block cells have positive controlled selectivity. LLaVA Effusion and Edema stop at the random/sham rung. Qwen Effusion replicates above random/sham on 400 independent patients and stops at the fixed clinical-direction rung because Nodule is larger; the familywise margin is `-0.0573` (95% CI `[-0.0694, -0.0450]`).
- Scientific state: `OBSERVED`.
- Gate decision: `PASS`; deterministic replays, decision-bearing terminal checksums, and the configured pinned read-only Claude reviews passed.
- Gate disposition: `READY` for the accepted paper claim.
- Evidence: immutable runs `20260902T191411Z-ffd523c464c8-f99e2f39`, `20260902T221534Z-110b84618d1b-edema`, `20260903T000321Z-caaae3ef346d-qwen-full`, and `20260903T103508Z-456c81bad460-7a3edc4b`; `CLAIMS_FROM_RESULTS.md`; `.aris/evidence_precheck.json`.

## ICLR 6+ causal-ownership gate

- Run: immutable run `20260904T125710Z-a3bd883540eb-causal-ownership` crosses the six fixed clinical probe directions with six matching questions on the registered third, label-blind cohort of 400 patients.
- Observation: all six diagonal ownership margins are negative and all six diagonals lose to at least one of 119 same-column random directions. Effusion is the strongest shared-alias candidate, with relative-dominance lower bound `0.1332` and off-diagonal-effect lower bound `0.2808`, but its raw off-diagonal effect `0.2909` remains below the registered global random maximum `0.3402`; no shared alias is detected.
- Scientific state: `OBSERVED`.
- Gate decision: `PASS`; the 304,800-outcome summary, 5,000-draw bootstrap, and mechanism route replayed byte-identically, 21 targeted tests passed, and the configured pinned read-only Claude review accepted the result and `consolidation_fallback` route.
- Gate disposition: `READY`.
- Evidence: `/home/qingchan/data/concept-flow/runs/20260904T125710Z-a3bd883540eb-causal-ownership/`; `/home/qingchan/data/concept-flow/state/causal-ownership-internal-validation-20260904T160302Z-Uu2oNg/`; reviewer receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260904T160448231526Z.json`; `docs/reviews/qwen7b-vislast-causal-ownership.md`.

## Qwen Consolidation input-closure gate

- Run: immutable run `20260904T162317Z-8a55a4c2f6b6-consolidation-closure` evaluates 50 fresh within-patient positive/negative Consolidation pairs using the train-fitted capacity-matched probe normal and displacement-matched `reltoken` writes against 119 random, sham, and six clinical controls.
- Observation: the probe is eligible with selectivity `0.0642` (95% CI `[0.0302, 0.0976]`). Mean displacement is `0.1091`, but its one-sided lower bound is `-0.0364`; concept closure gain `0.0022` is below random maximum `0.0025`, and the clinical-familywise lower bound is `-0.0015`. The immutable run completed 6,450 outcomes in 445 seconds. The validator-only ordering correction reused the artifact, all 257 arrays match, and the pinned read-only Claude review accepted `input_closure=false`.
- Scientific state: `OBSERVED`.
- Gate decision: `PASS`; the six-gate result-to-claim pass reports `claim_supported=yes`, confidence `high`, integrity `pass`, and routing action `confirm`.
- Gate disposition: `READY`; accepted evidence is available for the manuscript and the active evidence-expansion stage.
- Next step: how does input-supported intervention geometry explain the paired response? Develop the registered component/complement comparison after the answer-encoding crossover.
- Writer ownership: immutable run `20260904T162317Z-8a55a4c2f6b6-consolidation-closure` owns the accepted experimental artifacts; current writer ownership is recorded in the active research stage.
- Evidence: `/home/qingchan/data/concept-flow/runs/20260904T162317Z-8a55a4c2f6b6-consolidation-closure/`; `/home/qingchan/data/concept-flow/state/consolidation-input-closure-validator-replay-a-20260904T1634XXZ-F1ruMl/receipt.json`; reviewer receipts `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260904T163729487366Z.json` and `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260904T164410087766Z.json`; `docs/reviews/qwen7b-consolidation-vislast-input-closure.md`; `docs/reviews/six-gate-result-to-claim.md`.

## Accepted paper delivery

- Run: the six-gate evidence passed paper planning, drafting, immutable compilation and anonymous-bundle validation.
- Observation: the accepted baseline compiled from `e477a63` as an 11-page PDF with a nine-page main body, all 27 adjudicated values and embedded fonts. Bundle run `20260904T171115Z-3625aeb-sixgate-submission-bundle` reproduced that PDF byte-identically from 22 source assets.
- Gate decision: baseline delivery `PASS`; all required internal and pinned read-only reviews passed. Current manuscript integration belongs to the independent paper task.
- Evidence: `docs/reviews/paper-plan.md`, `docs/reviews/six-gate-evidence-locked-manuscript.md`, `docs/reviews/six-gate-paper-compilation.md`, `docs/reviews/six-gate-anonymous-submission-bundle.md`; exact run identities and receipts remain in `docs/EXPERIMENT_REGISTRY.md`.

## Transition rules

`PLANNED -> RUNNING` requires a clean pushed commit and dispatch receipt for experiments. `RUNNING -> FAILED` records implementation, measurement, or infrastructure invalidity in the run/failure ledger. `RUNNING -> OBSERVED` requires valid scientific evidence. A passed gate advances immediately to the next authorised action; validator-only changes reuse the original artifact.
