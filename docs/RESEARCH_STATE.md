# Research state

Updated: 2026-09-02. Scientific state uses `PLANNED`, `RUNNING`, `FAILED`, and `OBSERVED`; gate disposition uses `READY`, `BLOCKED`, and `UNAVAILABLE`.

## Bootstrap gate

- Run: infrastructure bootstrap and immutable smoke run `20260902T071539Z-7a4e7552d9e1-75d3510c`.
- Observation: Git handoff, fixed environment, reviewer transport, ARIS pin, tmux persistence, immutable dispatch, fetch, and safety acceptance have terminal receipts.
- Gate decision: `PASS`.
- Gate disposition: `READY`.
- Next step: can the registered `vis.last` synthetic GPU preflight prove exact forward-path capture?
- Evidence: `/home/qingchan/data/concept-flow/state/aris-audit.env`, `/home/qingchan/data/concept-flow/state/health.env`, `/home/qingchan/data/concept-flow/state/bootstrap-safety-acceptance.env`, `/home/qingchan/data/concept-flow/state/setup-receipt.tsv`, and `/home/qingchan/data/concept-flow/runs/20260902T071539Z-7a4e7552d9e1-75d3510c/`.

## First hard gate: `llava-effusion-vislast-reltoken`

- Run: the corrected exact hook preflight `20260902T191425Z-ffd523c464c8-hook` passed, and immutable full run `20260902T191522Z-ffd523c464c8-firstgate` is extracting the registered NIH manifest from the staged model snapshot with one A100.
- Observation: exact `encoder.layers.22` capture fired once; relative-token steering changed the downstream connector by `1.1875` and final logits by `0.1875`, while alpha zero was a bitwise no-op. The full run is producing activation shards from the staged snapshot.
- Scientific state: `RUNNING`; no probe or intervention observation is yet terminal.
- Gate decision: `BLOCKED` pending terminal probe, intervention, receipt, internal verification, and pinned read-only review evidence.
- Gate disposition: `BLOCKED`.
- Evidence: trust receipt `/home/qingchan/data/concept-flow/state/nih-chestxray14-independent-verification-5342b4219127/receipt.json`; dataset receipt `/home/qingchan/data/concept-flow/datasets/nih-chestxray14/asset-receipt.json`; model receipt `/home/qingchan/data/concept-flow/models/huggingface/asset-receipt.json`; research manifest `/home/qingchan/data/concept-flow/datasets/nih-chestxray14/manifest.csv`.
- Evidence reuse: the trust chain and asset receipts are terminal inputs. Heartbeats and validators read them without rehashing the full dataset, archives, model, shards, or run unless concrete contamination evidence appears.
- Release evidence: model and dataset receipts, exact hook preflight, patient bootstrap, repeated control, intervention sweep, immutable run receipt, and pinned read-only review.
- Next step: does the registered intervention produce a selective concept-consistent change relative to same-alpha controls? Complete and verify the active full run, then request the pinned read-only review.
- Writer ownership: server Codex/ARIS while the persistent server session is active; handoff follows the single-writer Git protocol.

## Transition rules

`PLANNED -> RUNNING` requires a clean pushed commit and dispatch receipt. `RUNNING -> FAILED` records implementation, measurement, or infrastructure invalidity in the run/failure ledger. `RUNNING -> OBSERVED` requires valid positive or negative scientific evidence. A passed gate advances immediately to the next authorised experiment; validator-only changes reuse the original artifact.
