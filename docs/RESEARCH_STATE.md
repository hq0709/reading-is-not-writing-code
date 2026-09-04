# Research state

Updated: 2026-09-04. Scientific state uses `PLANNED`, `RUNNING`, `FAILED`, and `OBSERVED`; gate disposition uses `READY`, `BLOCKED`, and `UNAVAILABLE`.

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
- Gate disposition: `READY` for manuscript reconciliation; no further experiment is authorized.
- Next step: can the six-gate evidence be integrated into the evidence-locked manuscript while preserving exact scope, provenance, and the nine-page main-body budget? Update and cross-review the manuscript.
- Writer ownership: immutable run `20260904T162317Z-8a55a4c2f6b6-consolidation-closure` owns experimental artifacts; Codex owns claim and manuscript reconciliation. No experiment process is active.
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

- Run: immutable paper build `20260903T123856Z-00f0adc-paper-compile` compiled the evidence-locked source with the pinned Tectonic 0.17.0 toolchain after a fresh agent followed the documented invocation verbatim.
- Observation: the PDF has 10 total pages, with the main body ending on page 8 against the nine-page limit, references beginning on page 8, and the appendix beginning on page 9. All 28 fonts are embedded; static validation covers 18 adjudicated values and 15 citations; no unresolved PDF marker or overfull box remains; visual inspection confirms readable tables and figures.
- Scientific state: accepted experimental evidence remains `OBSERVED`.
- Gate decision: `PASS`; deterministic compilation checks and the configured pinned read-only Claude review passed with no required actions.
- Gate disposition: `READY`.
- Evidence: `/home/qingchan/data/concept-flow/runs/20260903T123856Z-00f0adc-paper-compile/`; reviewer receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260903T124039710820Z.json`; `docs/reviews/evidence-locked-paper-compilation.md`; `paper/main.pdf`.

## Submission-bundle gate

- Run: immutable bundle run `20260903T125459Z-51b652c-submission-bundle` packaged the accepted paper source and assets, then rebuilt it with the pinned Tectonic toolchain.
- Observation: the 23-member archive has SHA-256 `302025a07282f46d74b28746fe86a90aaab945cb34476baf360f8bdfe6792517`; its manifest covers all 21 source assets, all members are regular files under one safe archive root, and the clean-room PDF is byte-identical to the accepted PDF at SHA-256 `e88c7f9a42baa4f04c7ccdbfdf9653501e4f96a81bd1c1ee238163671e5106e2`. Anonymous authorship and the identity-marker scan pass.
- Scientific state: accepted experimental evidence remains `OBSERVED`.
- Gate decision: `PASS`; the deterministic internal check and configured pinned read-only Claude review passed with no required actions.
- Gate disposition: `READY` for submission handoff.
- Evidence: `/home/qingchan/data/concept-flow/runs/20260903T125459Z-51b652c-submission-bundle/`; `/home/qingchan/data/concept-flow/state/submission-bundle-internal-validation-20260903T125553Z/receipt.json`; reviewer receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260903T125705202059Z.json`; `docs/reviews/anonymous-submission-bundle.md`.
- The accepted bundle remains the reproducible baseline while the causal-ownership gate raises the paper's scientific ceiling. The Overleaf-facing source remains in the separate `concept-flow-paper` repository.

## Transition rules

`PLANNED -> RUNNING` requires a clean pushed commit and dispatch receipt for experiments. `RUNNING -> FAILED` records implementation, measurement, or infrastructure invalidity in the run/failure ledger. `RUNNING -> OBSERVED` requires valid scientific evidence. A passed gate advances immediately to the next authorised action; validator-only changes reuse the original artifact.
