# Research state

Updated: 2026-09-03. Scientific state uses `PLANNED`, `RUNNING`, `FAILED`, and `OBSERVED`; gate disposition uses `READY`, `BLOCKED`, and `UNAVAILABLE`.

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

## Paper-planning gate

- Run: the accepted four-gate evidence was organized into the six-section ICLR outline in `PAPER_PLAN.md`.
- Observation: all 18 adjudicated values map to primary artifacts, and the outline preserves the registered availability, random/sham, and clinical-direction rungs.
- Gate decision: `PASS`; the pinned read-only Claude review returned disposition `READY` with no required actions.
- Evidence: `PAPER_PLAN.md`; reviewer receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260903T112856162325Z.json`; `docs/reviews/paper-plan.md`.

## Manuscript-drafting gate

- Run: draft the evidence-locked manuscript and generate Figures 1--3 and Tables 1--2 from the four accepted receipts.
- Observation: the manuscript, compact evidence snapshot, three vector figures, and two generated tables pass static validation with 18/18 adjudicated values, 15/15 cited bibliography entries, matched labels/references, anonymous authorship, and no stale section files or draft markers. The repository suite passes 83 tests with one platform skip. The pinned read-only Claude action-closure review reports `REQUIRED_ACTIONS: NONE`.
- Scientific state: accepted experimental evidence remains `OBSERVED`.
- Gate decision: `PASS` at pushed source commit `d3f7663674331e04e6c97474885995edfc205abe`.
- Gate disposition: `READY`.
- Evidence: `paper/`; `/home/qingchan/data/concept-flow/state/manuscript-draft-validation-20260903T120926Z/receipt.json`; reviewer receipts `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260903T120710736411Z.json` and `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260903T120851250065Z.json`; `docs/reviews/evidence-locked-manuscript-draft.md`.

## Paper-compilation gate

- Run: inspect the fixed environment for the registered manuscript build prerequisites.
- Observation: the reviewed source and reproducible figure/table generators are ready, but the fixed environment contains no `tectonic`, `latexmk`, `pdflatex`, or `bibtex`; therefore no current PDF, page-count, font-embedding, or final table-width observation exists.
- Gate decision: `BLOCKED`; the source gate passed, but the paper-compilation gate cannot be evaluated without changing the pinned toolchain.
- Gate disposition: `BLOCKED`.
- Next step: can the evidence-locked source compile within the nine-page ICLR main-body budget with readable tables and fully embedded fonts? Add an audited user-local TeX compiler to the pinned environment, then compile and validate the unchanged reviewed source.
- Writer ownership: server Codex/ARIS; no experiment or manuscript-build process is active.

## Transition rules

`PLANNED -> RUNNING` requires a clean pushed commit and dispatch receipt for experiments. `RUNNING -> FAILED` records implementation, measurement, or infrastructure invalidity in the run/failure ledger. `RUNNING -> OBSERVED` requires valid scientific evidence. A passed gate advances immediately to the next authorised action; validator-only changes reuse the original artifact.
