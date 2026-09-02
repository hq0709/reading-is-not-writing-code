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

- Run: exact hook preflight `20260902T191229Z-ffd523c464c8-db7dce6e` and immutable full run `20260902T191411Z-ffd523c464c8-f99e2f39` completed the registered NIH protocol from commit `ffd523c464c84417a93c5a6d0a34e5b74e55e76e` on one A100 in 7,022 seconds.
- Observation: exact `encoder.layers.22` capture fired once, alpha zero was a bitwise no-op, and relative-token steering changed the downstream connector by `1.1875` and final logits by `0.1875`. Effusion AUROC was `0.7788` (95% patient-bootstrap CI `0.7538–0.8031`); the 20-seed control mean was `0.6649`, giving selectivity `0.1139` (95% CI `0.0875–0.1404`). The maximum concept-consistent behavioural change was `0.1470` at alpha `-1`, above the same-alpha random 95th percentile `0.1202` and below the maximum absolute sham effect `0.1546`; `selective_cell=false`. The response over `[-0.5,0.5]` was monotone with Spearman rho `1.0`.
- Scientific state: `OBSERVED`; this registered cell contains linearly decodable Effusion information and no selective causal effect under the registered threshold.
- Gate decision: `PASS`; deterministic replay was byte-identical, the decision-bearing dispatcher checksums passed, and the pinned read-only Claude review returned `PASS` with no required actions.
- Gate disposition: `READY`.
- Evidence: run receipt `/home/qingchan/data/concept-flow/runs/20260902T191411Z-ffd523c464c8-f99e2f39/`; internal replay `/home/qingchan/data/concept-flow/state/first-gate-internal-validation-20260902T211900Z/intervention-summary.json`; reviewer receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260902T212456605398Z.json`; review record `docs/reviews/llava-effusion-vislast-reltoken.md`.
- Evidence reuse: the terminal trust, asset, hook, run, internal-validation, and reviewer receipts are inputs to later gates. Validators read them without rehashing the dataset, model, activation shard, or complete run unless a new terminal gate or concrete contamination evidence requires it.
- Next step: does the higher-controlled-decoding Edema cell at the same consumed LLaVA visual locus show selective causal influence under the unchanged threshold? Run the registered second hard gate.
- Writer ownership: server Codex/ARIS while the persistent server session is active; handoff follows the single-writer Git protocol.

## Second hard gate: `llava-edema-vislast-reltoken`

- Run: the fixed Edema protocol reuses the accepted prompt-independent `vis.last` activation artifact and the same 200 held-out intervention rows from `20260902T191411Z-ffd523c464c8-f99e2f39`; new probe and intervention evidence will use the current immutable source commit.
- Observation: none; the cell is registered and awaiting immutable dispatch.
- Scientific state: `PLANNED`.
- Gate decision: the protocol, one paired ordering contrast, control family, and unchanged selective-cell threshold are registered.
- Gate disposition: `READY`.
- Evidence: `docs/RESEARCH_PLAN.md#second-hard-gate`; source activation receipt `/home/qingchan/data/concept-flow/runs/20260902T191411Z-ffd523c464c8-f99e2f39/`.
- Next step: does Edema's stronger controlled decoding at `vis.last` produce selective causal influence? Dispatch `["bash","scripts/server/run_cross_cell_gate.sh"]` from clean pushed `main` on one registered GPU.
- Writer ownership: local Codex until the registration and runner commit is pushed; the server runner then owns immutable evidence generation.

## Transition rules

`PLANNED -> RUNNING` requires a clean pushed commit and dispatch receipt. `RUNNING -> FAILED` records implementation, measurement, or infrastructure invalidity in the run/failure ledger. `RUNNING -> OBSERVED` requires valid positive or negative scientific evidence. A passed gate advances immediately to the next authorised experiment; validator-only changes reuse the original artifact.
