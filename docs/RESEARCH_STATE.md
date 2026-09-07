# Research state

Updated: 2026-09-06 UTC. Scientific state uses `PLANNED`, `RUNNING`, `FAILED`, and `OBSERVED`; gate disposition uses `READY`, `BLOCKED`, and `UNAVAILABLE`.

## Current delivery and decision

- Run: the registered experiment batch, two-finding manuscript integration, and bounded statistical/label-conditioned evidence integration are complete.
- Observation: the current scientific manuscript is paper commit `a9a5e6e5397a380f05dcb8115086a055c063b50a` (39 pages). Paper delivery commit `534e31e085f01c53b5b83161821e62f8f3ca3d85` includes that manuscript, research reviews, and the frozen 35-page review package.
- Gate decision: accepted result gates and `PAPER_STATISTICAL_CLARITY PASS` are recorded. The final integration includes 36 label-conditioned rows checked against the accepted full-precision JSON, explicit dose-selection rules, and the implemented max-T definition. These are evidence integration and verification, not new model outcomes.
- Next step: prepare external replication of the phenomenon across 22 checkpoints, three datasets and six concepts per dataset. The [external execution package](external-replication/README.md) specifies three main evidence tables, fixed measurements, resource planning and return artifacts. Scientific state is `PLANNED`; family adapters, CheXpert/COCO manifests, revision pins and resource assignments precede test execution. Literature coverage remains paused at 16 cited works.
- Supervision: heartbeat `concept-flow` is **PAUSED** by user request. All assigned experiment and paper batches are terminal; no experiment, paper repair, or review is awaiting completion.
- Ownership: the project task maintains research state and the external protocol; the user forwards the execution package to collaborators. External runners are unassigned until the hardware/model allocation is agreed. The paper task owns the independent manuscript repository. Final reviewer coordination is released.
- Evidence: [statistical integration and verification](https://github.com/wy-coliney/concept-flow-paper/blob/534e31e085f01c53b5b83161821e62f8f3ca3d85/docs/reviews/statistical-clarity.md). The final receipt is `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260906T043708128134Z.json`, requested/actual `claude-fable-5-1`, medium, valid read-only review against clean code `bb1205ca57c83ac48916937b6508a08f2d2e2a5e`.

## Accepted research and evidence map

| Completed work | Observation and decision | Authoritative report |
| --- | --- | --- |
| Controlled readout and independent Qwen Effusion steering | Independent 400-patient steering exceeds random/sham references; Nodule produces the larger clinical-direction response. Accepted evidence. | [Direction-specificity results](reviews/qwen7b-effusion-vislast-direction-specificity.md) |
| Six-direction by six-question ownership matrix | All six diagonal ownership margins are negative in the fixed family; the registered shared-alias criterion is unmet. Accepted evidence. | [Ownership review](reviews/qwen7b-vislast-causal-ownership.md) |
| Ownership response diagnostics | All 36 label-conditioned cells are available; 35 gap point estimates are negative. These are retrospective descriptive estimates with pointwise intervals. | [Ownership diagnostics](OWNERSHIP_DIAGNOSTICS.md) |
| Answer encoding and independent Mass confirmation | Clinical-direction advantage depends on the tested wording; the fixed 119-random reference gate is unmet. Accepted evidence. | [Encoding](ANSWER_ENCODING_RESULTS.md), [Mass confirmation](MASS_PROMPT_SPECIFICITY_RESULTS.md) |
| Within-patient opportunity and LLaVA reader/image checks | Paired-response, clean discrimination, reader selectivity, and write opportunity have separate measured boundaries. Accepted reports retain each gate's disposition. | [Paired models](PAIRED_OPPORTUNITY_ARCHITECTURE_RESULTS.md), [Reader validation](LLAVA_VALIDATION_OPPORTUNITY_RESULTS.md), [Image diagnostic](LLAVA_YESNO_IMAGE_DIAGNOSTIC_RESULTS.md) |
| LLaVA-Med runtime feasibility | CPU-only tensor mapping and runtime preparation passed; scientific state remains `PLANNED`. Clinical validation is a prospective option. | [Runtime feasibility](LLAVA_MED_RUNTIME_FEASIBILITY.md) |

The accepted contribution concerns the distinction between readable information, steering efficacy, and fixed-family label-specific advantage. Direction-refit stability, disease-conditional mechanism, and transfer across models/loci remain open scientific questions. Detailed run identities, execution failures, commands and artifact paths belong to the [experiment registry](EXPERIMENT_REGISTRY.md); protocols and prospective gates belong to the [research plan](RESEARCH_PLAN.md).

## Paper, reviews and reproducibility

- [Current paper repository](https://github.com/wy-coliney/concept-flow-paper): formal manuscript sources and `main.pdf`; publication ownership stays separate from this code repository's evidence snapshot in `paper/`.
- [Review reports](https://github.com/wy-coliney/concept-flow-paper/tree/534e31e085f01c53b5b83161821e62f8f3ca3d85/docs/reviews): manuscript integration records, the historical ICLR assessment, external-review triage and final statistical verification. Historical assessments retain their assessed commits; verification PASS is not a new acceptance score.
- [Frozen 35-page review package](https://github.com/wy-coliney/concept-flow-paper/tree/534e31e085f01c53b5b83161821e62f8f3ca3d85/review-artifacts/2026-09-06): PDF, source archive and layout wrapper for paper baseline `2602850ce64d9984fd80e24ce21b1e76b58b442e`. A new length-limited copy of the current 39-page manuscript is a separate pending deliverable if requested.
- [Reproduction and data dependencies](https://github.com/wy-coliney/concept-flow-paper/blob/534e31e085f01c53b5b83161821e62f8f3ca3d85/REPRODUCE.md): accepted run provenance and figure inputs. Full model outputs and receipts remain in the immutable server archive under `/home/qingchan/data/concept-flow/` and `/home/qingchan/.codex/state/claude-review-concept-flow/`; GitHub carries source, reports and selected aggregate evidence.
- Reviewer routing follows [the registered policy](RESEARCH_PLAN.md#reviewer-routing-policy). Before a future server research launch, refresh the ARIS wrapper audit using the existing installer.
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
- Gate disposition: `READY`; accepted evidence is available for the manuscript. Follow-up decisions are recorded in the current delivery section above.
- Writer ownership: immutable run `20260904T162317Z-8a55a4c2f6b6-consolidation-closure` owns the accepted experimental artifacts; current writer ownership is recorded in the active research stage.
- Evidence: `/home/qingchan/data/concept-flow/runs/20260904T162317Z-8a55a4c2f6b6-consolidation-closure/`; `/home/qingchan/data/concept-flow/state/consolidation-input-closure-validator-replay-a-20260904T1634XXZ-F1ruMl/receipt.json`; reviewer receipts `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260904T163729487366Z.json` and `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260904T164410087766Z.json`; `docs/reviews/qwen7b-consolidation-vislast-input-closure.md`; `docs/reviews/six-gate-result-to-claim.md`.

## Accepted paper delivery

- Run: the six-gate evidence passed paper planning, drafting, immutable compilation and anonymous-bundle validation.
- Observation: the accepted baseline compiled from `e477a63` as an 11-page PDF with a nine-page main body, all 27 adjudicated values and embedded fonts. Bundle run `20260904T171115Z-3625aeb-sixgate-submission-bundle` reproduced that PDF byte-identically from 22 source assets.
- Gate decision: baseline delivery `PASS`; all required internal and pinned read-only reviews passed. Current manuscript integration belongs to the independent paper task.
- Evidence: `docs/reviews/paper-plan.md`, `docs/reviews/six-gate-evidence-locked-manuscript.md`, `docs/reviews/six-gate-paper-compilation.md`, `docs/reviews/six-gate-anonymous-submission-bundle.md`; exact run identities and receipts remain in `docs/EXPERIMENT_REGISTRY.md`.

## Transition rules

`PLANNED -> RUNNING` requires a clean pushed commit and dispatch receipt for experiments. `RUNNING -> FAILED` records implementation, measurement, or infrastructure invalidity in the run/failure ledger. `RUNNING -> OBSERVED` requires valid scientific evidence. A passed gate advances immediately to the next authorised action; validator-only changes reuse the original artifact.
