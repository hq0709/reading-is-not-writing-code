# Research state

Updated: 2026-09-05. Scientific state uses `PLANNED`, `RUNNING`, `FAILED`, and `OBSERVED`; gate disposition uses `READY`, `BLOCKED`, and `UNAVAILABLE`.

## Active research stage

- Run: `20260904T230152Z-21dad2f72726-answer-encoding`, source `21dad2f`, completed all 200,400 outcomes in 6,402 seconds on GPU 0. Both terminal exit statuses are zero; no scientific run is active.
- Observation: four independently calibrated questions are eligible. The mapping-even minus mapping-odd energy contrast is -2.1786 (95% CI [-2.4061, -1.9810]); random/sham responses are also mapping-odd dominant. Mass's descriptive clinical logit margin changes from -0.7181 under standard yes/no to +0.2825 and +0.2900 under the two coded prompts. Under both coded prompts, the matched Mass-direction effect exceeds all 20 random effects and the absolute sham effect.
- Scientific state: answer-encoding `OBSERVED`; independent-patient Mass confirmation `PLANNED`.
- Gate disposition: answer-encoding and Mass registration `PASS`. The new confirmation passed independent design/implementation review, the 177-test local suite (six skips), syntax checks and pinned read-only registration review. Metadata-only exclusion leaves 642 eligible test patients and 1,334 images for the fixed 200-patient selection.
- Next step: does Mass retain its prompt-conditioned clinical advantage on independent patients across the two question wordings? Dispatch the registered seven-condition, 179,200-outcome confirmation from the clean pushed commit; its validation preflight gates the scientific grid.
- Writer ownership: project task `01a06ec5-fec9-7241-b066-72c1bc5632b8` coordinates research, evidence and delivery and owns code and experiment dispatch. Paper task `01a06afe-5c66-7a80-b922-b3ede71eab3e` owns the separate paper repository. Existing unrelated local edits remain with their author.
- Supervision: heartbeat `concept-flow` is ACTIVE, bound to the project task every 30 minutes, advancing research and paper coordination and remaining quiet on unchanged state.
- Evidence: `docs/ANSWER_ENCODING_RESULTS.md`, `docs/reviews/qwen7b-vislast-answer-encoding.md`; pinned receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T005436988285Z.json`; immutable run artifacts include complete signed-dose matrices, AUROC/Brier summaries and calibration data.
- Registration evidence: `docs/reviews/qwen7b-mass-prompt-specificity-registration.md`; pinned receipt `review-20260905T014243639442Z.json`.
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
