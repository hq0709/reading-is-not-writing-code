# Research state

Updated: 2026-09-05. Scientific state uses `PLANNED`, `RUNNING`, `FAILED`, and `OBSERVED`; gate disposition uses `READY`, `BLOCKED`, and `UNAVAILABLE`.

## Active research stage

- Run: immutable run `20260905T232313Z-2a95dfbed888-llava-yesno`, source `2a95dfbed888fce12b7481812065e5d68b712e8a`; all 2,800 image and two text outcomes completed in 260 dispatcher seconds before post-execution summarization stopped.
- Observation: the registered native CSV, candidate logits, partitions and frozen-reader activations are complete. NumPy replay of the Torch float32 log-sum-exp values differs by at most `3.814697265625e-6`, while the original summarizer required `1e-6`.
- Scientific state: `FAILED`; the model outcomes remain unadjudicated pending validator-only recovery from the preserved immutable artifact.
- Gate decision: gate disposition `BLOCKED` at summary validation; registration remains `PASS` and no scientific rerun is authorised.
- Next step: can a source-bound validator-only recovery reproduce and independently accept the complete registered result from the unchanged native artifact?
- Evidence: immutable run `/home/qingchan/data/concept-flow/runs/20260905T232313Z-2a95dfbed888-llava-yesno/`; registration `docs/reviews/llava-yesno-image-registration.md`; pinned registration receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T232034186616Z.json`; failure ledger in `docs/EXPERIMENT_REGISTRY.md`.
- Writer ownership: local project task `01a06ec5-fec9-7241-b066-72c1bc5632b8` owns the validator-only recovery on `research/llava-yesno-image`; the immutable run owns the scientific artifact. Paper task `01a06afe-5c66-7a80-b922-b3ede71eab3e` retains the independent manuscript repository.
- Image diagnostic preparation: registration `PASS` at `2a95dfbed888fce12b7481812065e5d68b712e8a`; independent executable review and pinned read-only registration review accepted the exact two-prompt grid, frozen reader and shared family `(A0-0.5, G0, G1-G0)`. Protocol: `docs/RESEARCH_PLAN.md#llava-yesno-image-diagnostic`; reviews: `docs/reviews/llava-yesno-image-design.md`, `docs/reviews/llava-yesno-image-implementation.md`, and `docs/reviews/llava-yesno-image-registration.md`.
- Diagnostic execution: immutable run `20260905T211508Z-d24f2a7fe05e-llava-readout` stopped at the registered pre-scientific mapping gate before new cohort outcomes. Scientific state remains `PLANNED`; gate disposition `BLOCKED`. Independent terminal verification and canonical pinned read-only measurement review agree on failure class `MEASUREMENT` and require no rerun. Evidence: failure ledger in `docs/EXPERIMENT_REGISTRY.md`; internal receipt `/home/qingchan/data/concept-flow/state/llava-readout-diagnostic-preflight-internal-20260905T211744Z/receipt.json`; pinned receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T211847245097Z.json`.
- Supervision: heartbeat `concept-flow` is ACTIVE and bound to the project task at a verified 30-minute interval. Its updated cadence policy uses 30 minutes for ordinary continuation, shortening only for a concrete time-sensitive decision window and lengthening for stable long waits. Active work continues within a turn; unchanged external state remains quiet.
- Evidence: `docs/MASS_PROMPT_SPECIFICITY_RESULTS.md`; `docs/reviews/qwen7b-mass-prompt-specificity-results.md`; internal receipt `/home/qingchan/data/concept-flow/state/mass-confirmation-internal-20260905T042600Z/smallreceipt.json`; pinned result receipt `review-20260905T051404447778Z.json`.
- Paired evidence: `docs/QWEN_PAIRED_OPPORTUNITY_RESULTS.md`; `docs/reviews/qwen7b-paired-opportunity-results.md`; internal receipt `/home/qingchan/data/concept-flow/state/paired-opportunity-internal-20260905T054119Z/smallreceipt.json`; pinned result receipt `review-20260905T054422273733Z.json`.
- Paired-cohort availability: accepted source metadata and inventory give 120 eligible Consolidation patients absent from the entire original manifest, matched within patient on view, sex, age and 13 other disease labels. Deterministic selection precedes model loading; `No Finding` remains a descriptive stratum. Source evidence: `docs/PAIRED_OPPORTUNITY_FEASIBILITY.md` and `docs/reviews/paired-opportunity-preparation.md`.
- Paper handoff: the paper task reports validation-readout integration delivered at main and origin/main `4babe4ccaefafda5f42094f157e1fd9c3d99ffd9`. The PDF has 31 pages, with main text and AI-use statement ending on page nine; Appendix K contains the complete six-question reader and capability results. All 27 original values, final build, visual checks and independent reviews passed. Evidence record: paper `docs/reviews/validation-readout-integration.md`; pinned receipt `review-20260905T202602751758Z.json`. The temporary branch is merged and removed, one canonical worktree remains and pre-existing `.omc/` is preserved.
- Reviewer coordination: the paper task received the accepted finite-suite measurement and its scope. It will integrate measurement selection with the subsequent accepted image evidence; the Y-only server assignment owns the available pinned transport for its registration and result gates.
- Protocol: `docs/RESEARCH_PLAN.md#llava-paired-behavioral-opportunity`. The accepted LLaVA snapshot is `/home/qingchan/data/concept-flow/models/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9`; reuse its asset receipt and the accepted LLaVA activation archive if a reader gate becomes justified.
- LLaVA registration: `docs/reviews/llava-paired-opportunity-registration.md`; pinned receipt `review-20260905T063744124554Z.json`.
- LLaVA result: `docs/PAIRED_OPPORTUNITY_ARCHITECTURE_RESULTS.md`; `docs/reviews/llava-paired-opportunity-results.md`; internal receipt `/home/qingchan/data/concept-flow/state/llava-opportunity-internal-20260905T064202Z/receipt.json`; pinned result receipt `review-20260905T064601848008Z.json`.
- Registration: `docs/reviews/qwen7b-paired-opportunity-registration.md`; pinned receipt `review-20260905T051705042400Z.json`.
- Current validation: `docs/reviews/qwen7b-paired-opportunity-validation.md`; independent design/implementation and pinned amendment review passed, receipt `review-20260905T053453798343Z.json`. The full suite passed 223 tests with six environment-specific skips. The recovery run reproduced all four generic mapping signs, exact clean repeat and source-bound 100-pair identities before image scoring.

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
