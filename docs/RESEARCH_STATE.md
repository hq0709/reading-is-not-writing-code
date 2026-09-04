# Research state

Updated: 2026-09-04. Scientific state uses `PLANNED`, `RUNNING`, `FAILED`, and `OBSERVED`; gate disposition uses `READY`, `BLOCKED`, and `UNAVAILABLE`.

## Active research stage

- Run: `qwen7b-vislast-answer-encoding` registration and implementation; accepted ownership outcomes also enter descriptive score, discrimination, and label-conditioned diagnostics.
- Observation: the existing Qwen ownership matrix covers six clinical questions at one yes/no encoding and one positive dose. The available LLaVA causal comparison covers two concepts. Encoding dependence, input-supported intervention geometry, and architecture transfer are the next evidence targets.
- Scientific state: `PLANNED`.
- Gate disposition: `BLOCKED` for GPU dispatch until the registration changes and implementation review pass. The pinned reviewer is available and requested independent capability calibration, an unbiased response-energy estimator, matched-control interpretation, and a validation throughput pilot.
- Diagnostic run: `20260904T224225Z-1ab7576193a7-ownership-diagnostics` completed on CPU from pushed source `1ab7576`. All six ownership margins remain negative in logit coordinates; all six matched-steering AUROC point estimates are below their baselines. Descriptive findings are in `docs/OWNERSHIP_DIAGNOSTICS.md`, pending pinned result review.
- Next step: does the direction-by-question response follow clinical meaning when the answer-code mapping is exchanged? Complete the registered crossover implementation, run its validation-only preflight, and dispatch the full paired sensitivity experiment.
- Writer ownership: local Codex owns the code repository and experiment protocol; the separate paper task owns the Overleaf manuscript. The server checkout was clean at `d0a4265`, with GPU 0 available at the initial inventory. GPU availability is checked again by the dispatcher.
- Protocol: `docs/RESEARCH_PLAN.md#iclr-evidence-expansion` and `#qwen-answer-encoding-crossover`.

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

## Paper-planning gate

- Run: the accepted four-gate evidence was organized into the six-section ICLR outline in `PAPER_PLAN.md`.
- Observation: all 18 adjudicated values map to primary artifacts, and the outline preserves the registered availability, random/sham, and clinical-direction rungs.
- Gate decision: `PASS`; the pinned read-only Claude review returned disposition `READY` with no required actions.
- Evidence: `PAPER_PLAN.md`; reviewer receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260903T112856162325Z.json`; `docs/reviews/paper-plan.md`.

## Manuscript-drafting gate

- Run: reconcile the evidence-locked manuscript and generated evidence snapshot with the accepted six-gate sequence, validate and render it, inspect the decision-bearing pages, and request pinned read-only review at pushed commit `7e70d1db3ca95178abc1c0663c8030af65ea8005`.
- Observation: the draft includes the Qwen ownership matrix and paired Consolidation closure gate throughout the claim chain. Static validation finds all 27 adjudicated values and 15 citations; the 11-page render ends the main body on page 8, embeds all fonts, and has no unresolved reference, overfull box, or clipped table. The reviewer returned `PASS`, `READY`, and no required actions; its optional clarity notes are incorporated.
- Scientific state: accepted experimental evidence remains `OBSERVED`.
- Gate decision: `PASS`.
- Gate disposition: `READY` for immutable compilation.
- Next step: can the accepted six-gate source compile reproducibly from an immutable commit while preserving the nine-page main-body limit, all 27 adjudicated values, readable tables, and fully embedded fonts? Run the registered immutable compile gate.
- Evidence: `paper/`; `CLAIMS_FROM_RESULTS.md`; reviewer receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260904T165826704963Z.json`; `docs/reviews/six-gate-evidence-locked-manuscript.md`.

## Paper-compilation gate

- Run: immutable no-GPU build `20260904T170512Z-e477a63-sixgate-paper-compile` compiled clean pushed source `e477a634410f9867d9aa2be70b0795ad04f691e2` with pinned Tectonic 0.17.0.
- Observation: the 11-page PDF has SHA-256 `91ab6dd2ecbb745a238e616a7fffd5827fb80c648d21887a040a1630a9c23102`, conservative main-body page 9 of 9, all 30 fonts embedded, all 27 adjudicated values, no unresolved marker or overfull box, and readable decision-bearing figures and tables.
- Scientific state: accepted experimental evidence remains `OBSERVED`.
- Gate decision: `PASS`; terminal manifest validation, compile checks, visual inspection, and the configured pinned read-only Claude review passed with no required actions.
- Gate disposition: `READY` for anonymous bundle validation.
- Next step: can the accepted six-gate PDF and its complete minimal source reproduce byte-identically as an anonymous submission bundle? Build and validate the registered immutable bundle.
- Evidence: `/home/qingchan/data/concept-flow/runs/20260904T170512Z-e477a63-sixgate-paper-compile/`; reviewer receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260904T170638706411Z.json`; `docs/reviews/six-gate-paper-compilation.md`.
## Submission-bundle gate

- Run: immutable no-GPU bundle run `20260904T171115Z-3625aeb-sixgate-submission-bundle` packaged the accepted six-gate source and PDF, then rebuilt it with pinned Tectonic 0.17.0.
- Observation: the 24-member archive has SHA-256 `11ae2a9e4cc41cde85bd2c1e65a0b03d92f4e73cf4f46eee0357d1521aa252e0`; all 22 source assets pass the manifest, every member is regular and safely rooted, and clean-room reproduction is byte-identical to the accepted PDF at SHA-256 `91ab6dd2ecbb745a238e616a7fffd5827fb80c648d21887a040a1630a9c23102`. Anonymous authorship and the identity-marker scan pass.
- Scientific state: accepted experimental evidence remains `OBSERVED`.
- Gate decision: `PASS`; the deterministic internal check and configured pinned read-only Claude review passed with no required actions.
- Gate disposition: `READY` for submission handoff.
- Next step: integrate accepted findings from the active evidence-expansion stage through the independent paper task.
- Evidence: `/home/qingchan/data/concept-flow/runs/20260904T171115Z-3625aeb-sixgate-submission-bundle/`; reviewer receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260904T171201801981Z.json`; `docs/reviews/six-gate-anonymous-submission-bundle.md`. The Overleaf-facing source remains in the separate `concept-flow-paper` repository.

## Transition rules

`PLANNED -> RUNNING` requires a clean pushed commit and dispatch receipt for experiments. `RUNNING -> FAILED` records implementation, measurement, or infrastructure invalidity in the run/failure ledger. `RUNNING -> OBSERVED` requires valid scientific evidence. A passed gate advances immediately to the next authorised action; validator-only changes reuse the original artifact.
