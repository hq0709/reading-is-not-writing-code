# Research state

Updated: 2026-09-05. Scientific state uses `PLANNED`, `RUNNING`, `FAILED`, and `OBSERVED`; gate disposition uses `READY`, `BLOCKED`, and `UNAVAILABLE`.

## Active research stage

- Run: `20260905T015725Z-1c9820d7b576-mass-prompts`, source `1c9820d7b5765c2ec220d9f4afaafd4c743e43e4`, started at 01:57:50 UTC on GPU 0. The seven-condition Mass confirmation has 179,200 registered outcomes and a three-hour execution cap.
- Observation: validation preflight passed all twelve mapping cases, exact zero-dose behavior and a nonzero logit change of 3.5. The 1,024-equivalent pilot predicts 8,834.675 seconds, below the 9,900-second allowance. At 03:35 UTC, calibration, the three `show` cells and `is/yes_no` were complete and `is/A_present` was progressing; scientific conclusions await the complete registered grid.
- Scientific state: answer-encoding `OBSERVED`; independent-patient Mass confirmation `RUNNING`.
- Gate disposition: Mass registration, execution-budget review and validation preflight `PASS`. Internal design/implementation review, the 177-test local suite (six skips), syntax checks and pinned review passed; the budget-only change passed 11 CPU tests and independent/pinned review. The fixed 200-patient confirmation selection follows metadata-only exclusion from 642 eligible test patients.
- Next step: does Mass retain its prompt-conditioned clinical advantage on independent patients across the two question wordings? Complete the immutable grid and registered analysis, then perform internal and pinned result review before selecting the natural-displacement gate and handing findings to the paper task. The paired-opportunity preparation distinguishes the pooled-reader projection from norm-weighted writes and binds cohort feasibility to `docs/PAIRED_OPPORTUNITY_FEASIBILITY.md`.
- Writer ownership: project task `01a06ec5-fec9-7241-b066-72c1bc5632b8` coordinates research, evidence and delivery and owns code and experiment dispatch. Paper task `01a06afe-5c66-7a80-b922-b3ede71eab3e` owns the separate paper repository. Existing unrelated local edits remain with their author.
- Supervision: heartbeat `concept-flow` is ACTIVE, bound to the project task every 30 minutes, advancing research and paper coordination and remaining quiet on unchanged state.
- Evidence: active run `metadata.env`, `artifacts/registered-rows.json` and `artifacts/preflight.json`. Accepted answer-encoding findings remain in `docs/ANSWER_ENCODING_RESULTS.md` and `docs/reviews/qwen7b-vislast-answer-encoding.md`, with pinned receipt `review-20260905T005436988285Z.json`.
- Registration evidence: `docs/reviews/qwen7b-mass-prompt-specificity-registration.md` and `docs/reviews/qwen7b-mass-prompt-specificity-budget.md`; pinned receipts `review-20260905T014243639442Z.json` and `review-20260905T015535962388Z.json`. Run-level feasibility outcomes remain in `docs/EXPERIMENT_REGISTRY.md`.
- Paired-cohort availability: original NIH metadata and available files provide 4,480 test patients absent from the entire probe manifest. Same-patient, view, sex, age and 13-other-disease matching yields 124 Mass or 120 Consolidation patients; exact `No Finding` matching gives 60 or 73. Internal review, source-bound replay, calibration coverage, full retained-field agreement and pinned review passed; see `docs/PAIRED_OPPORTUNITY_FEASIBILITY.md` and `docs/reviews/paired-opportunity-preparation.md`. Opportunity registration follows Mass routing.
- Paper handoff: the paper task integrated answer-encoding and ownership diagnostics into paper main `4706ea5`, with a 24-page PDF and nine-page main body. Its internal, build, visual and pinned writing reviews passed; see paper `docs/reviews/answer-encoding-integration.md`.
- Protocol: `docs/RESEARCH_PLAN.md#qwen-mass-prompt-conditioned-specificity`.

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
