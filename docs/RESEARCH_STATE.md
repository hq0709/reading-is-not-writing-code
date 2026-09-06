# Research state

Updated: 2026-09-06 UTC. Scientific state uses `PLANNED`, `RUNNING`, `FAILED`, and `OBSERVED`; gate disposition uses `READY`, `BLOCKED`, and `UNAVAILABLE`.

## Active research stage

- Run: immutable run `20260905T232313Z-2a95dfbed888-llava-yesno`, source `2a95dfbed888fce12b7481812065e5d68b712e8a`, completed all 2,800 image and two text outcomes in 260 dispatcher seconds. Validator-only recovery at `06277a1fb8a487a54b9220373957b1f1d988e2f5` reused the unchanged artifact; terminal validation at `1da1ca3c6da95218a99304f0edb65e3ab407e0de` independently replayed the result.
- Observation: accepted-wording image AUROC is `0.5259`, donor AUROC against index labels is `0.4072`, and image advantage is `0.1187`; alternate-wording values are `0.5089`, `0.3975`, and `0.1113`. The registered simultaneous intervals for `A0-0.5`, `G0`, and `G1-G0` are (`-0.1045`, `0.1563`), (`-0.0117`, `0.2491`), and (`-0.1377`, `0.1231`). Frozen-reader selectivity is `-0.0111` (95% CI `-0.1541`, `0.1185`).
- Scientific state: `OBSERVED`; the registered evidence is a valid negative/unresolved result.
- Gate decision: `PASS`, gate disposition `READY`; `opportunity=false`, `wording=unresolved`, and `reader_replicated=false`. Pinned read-only Claude result review accepted the terminal packet and evidence-synthesis route.
- Next step: can a medical-domain model provide a faithful first-answer measurement for independently qualifying image-linked discrimination before direction-control comparisons? Read-only asset triage selected LLaVA-Med for the bounded CPU-only preparation in `docs/RESEARCH_PLAN.md#medical-domain-runtime-feasibility`. Public pinned metadata is available; its custom model class is absent from the fixed runtime's native registry, so compatibility remains unresolved. Evidence: `docs/MEDICAL_MODEL_FEASIBILITY.md`. The project task owns preparation; scientific image evaluation and weight staging await separate registration.
- Evidence: `docs/LLAVA_YESNO_IMAGE_DIAGNOSTIC_RESULTS.md`; immutable run `/home/qingchan/data/concept-flow/runs/20260905T232313Z-2a95dfbed888-llava-yesno/`; terminal receipt `/home/qingchan/data/concept-flow/state/llava-yesno-image-internal-20260905T235215Z/receipt.json`; pinned result receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T235348704249Z.json`; result-to-claim trace `.aris/traces/result-to-claim/2026-09-05_run02/`.
- Writer ownership: project task `01a06ec5-fec9-7241-b066-72c1bc5632b8` prepares the next handoff in `docs/LLAVA_MED_FEASIBILITY_SERVER_ASSIGNMENT.md`. After clean GitHub synchronization and launcher acceptance, server Codex owns its CPU-only feasibility branch and the local task supervises read-only. Paper task `01a06afe-5c66-7a80-b922-b3ede71eab3e` retains manuscript ownership. Native outcomes and accepted recovery/terminal receipts remain unchanged.
- Image diagnostic preparation: registration `PASS` at `2a95dfbed888fce12b7481812065e5d68b712e8a`; independent executable review and pinned read-only registration review accepted the exact two-prompt grid, frozen reader and shared family `(A0-0.5, G0, G1-G0)`. Protocol: `docs/RESEARCH_PLAN.md#llava-yesno-image-diagnostic`; reviews: `docs/reviews/llava-yesno-image-design.md`, `docs/reviews/llava-yesno-image-implementation.md`, and `docs/reviews/llava-yesno-image-registration.md`.
- Diagnostic execution: immutable run `20260905T211508Z-d24f2a7fe05e-llava-readout` stopped at the registered pre-scientific mapping gate before new cohort outcomes. Scientific state remains `PLANNED`; gate disposition `BLOCKED`. Independent terminal verification and canonical pinned read-only measurement review agree on failure class `MEASUREMENT` and require no rerun. Evidence: failure ledger in `docs/EXPERIMENT_REGISTRY.md`; internal receipt `/home/qingchan/data/concept-flow/state/llava-readout-diagnostic-preflight-internal-20260905T211744Z/receipt.json`; pinned receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T211847245097Z.json`.
- Supervision: heartbeat `concept-flow` is ACTIVE and bound to the project task at a verified 30-minute interval. Its updated cadence policy uses 30 minutes for ordinary continuation, shortening only for a concrete time-sensitive decision window and lengthening for stable long waits. Active work continues within a turn; unchanged external state remains quiet.
- Evidence: `docs/MASS_PROMPT_SPECIFICITY_RESULTS.md`; `docs/reviews/qwen7b-mass-prompt-specificity-results.md`; internal receipt `/home/qingchan/data/concept-flow/state/mass-confirmation-internal-20260905T042600Z/smallreceipt.json`; pinned result receipt `review-20260905T051404447778Z.json`.
- Paired evidence: `docs/QWEN_PAIRED_OPPORTUNITY_RESULTS.md`; `docs/reviews/qwen7b-paired-opportunity-results.md`; internal receipt `/home/qingchan/data/concept-flow/state/paired-opportunity-internal-20260905T054119Z/smallreceipt.json`; pinned result receipt `review-20260905T054422273733Z.json`.
- Paired-cohort availability: accepted source metadata and inventory give 120 eligible Consolidation patients absent from the entire original manifest, matched within patient on view, sex, age and 13 other disease labels. Deterministic selection precedes model loading; `No Finding` remains a descriptive stratum. Source evidence: `docs/PAIRED_OPPORTUNITY_FEASIBILITY.md` and `docs/reviews/paired-opportunity-preparation.md`.
- Paper handoff: image/readout integration is delivered at verified paper main and origin/main `ffa6e66edf4202088e1974e60e7347e9b868d648`. The PDF has 36 pages: main text and AI-use statement end on page nine, references occupy page ten, appendices start on page eleven and Appendix L occupies pages 32–36. The paper review record reports all 27 original values, final build, visual checks and independent reviews passed (`PAPER_IMAGE_READOUT_INTEGRATION PASS`). Evidence: paper `docs/reviews/image-readout-integration.md` and `REPRODUCE.md`; pinned receipt `review-20260906T001834586719Z.json`. The paper task reports its temporary branch removed and one canonical worktree; pre-existing `.omc/` remains untouched.
- Reviewer coordination: the paper task released the pinned transport after its accepted review against clean server source `3cd386c8fd582eb50f1a59e73ae3b48a9eaee90d`. Server main then fast-forwarded to the docs-only synthesis checkpoint `cf0b505`; the project task owns subsequent coordination.
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
