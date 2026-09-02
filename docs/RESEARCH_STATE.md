# Research state

Last bootstrap update: 2026-09-02. This file distinguishes scientific state (`PLANNED`, `RUNNING`, `FAILED`, `OBSERVED`) from gate disposition (`READY`, `BLOCKED`, `UNAVAILABLE`). Existing evidence remains in `docs/02_RESULTS.md` and `docs/10_RESULTS.md`; this bootstrap does not invent new observations.

## Bootstrap gate

- Scientific state: `PLANNED`
- Gate disposition: `READY`
- Accepted implementation: pushed commit `5c6506f75ee3de0a26a95275af4e3a60596e24e5`.
- Environment, capability, and safety audits: `/home/qingchan/data/concept-flow/state/aris-audit.env`, `/home/qingchan/data/concept-flow/state/health.env`, and `/home/qingchan/data/concept-flow/state/bootstrap-safety-acceptance.env`.
- Toolchain provenance is defined by `config/toolchain-provenance.tsv`; its atomic server receipt is `/home/qingchan/data/concept-flow/state/setup-receipt.tsv`.
- Reviewer evidence: fresh-shell receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260902T074946088752Z.json`; detached-tmux receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260902T075031584500Z.json`.
- Executor-to-reviewer evidence: `/home/qingchan/data/concept-flow/state/codex-mcp-probe-20260902T075246Z.jsonl`.
- Immutable GPU/fetch evidence: run `20260902T071539Z-7a4e7552d9e1-75d3510c` under `/home/qingchan/data/concept-flow/runs/`.
- Writer ownership: local writer through this state commit; server Codex becomes writer when the persistent ARIS session starts.

## First hard gate

- Scientific state: `PLANNED`
- Gate disposition: `BLOCKED`
- Measurement: corrected LLaVA vision hook, corrected registered intervention scale, repeated control draws, and bootstrap intervals on the smallest informative registered cell.
- Supported interpretation: none until a valid immutable run is `OBSERVED`.
- Unresolved hypothesis: whether peak decodability predicts selective causal use after correcting known measurement faults.
- Excluded claims: current incomplete D1 runs do not establish either dissociation or alignment.
- Prerequisite status: the registered LLaVA model and NIH release are staged and verified under `/home/qingchan/data/concept-flow/`; the registered NIH manifest has 26,229 rows at SHA-256 `837ca37acce72cdf1e7c4a43e559a6d34d99edf2eb5c76a8f5e53295e99c6440`. Independent release verification and its pinned read-only review are recorded under `/home/qingchan/data/concept-flow/state/nih-chestxray14-independent-verification-5342b4219127/` and `/home/qingchan/data/concept-flow/state/nih-checksum-review-20260902T171417Z.txt`.
- Blocker: immutable synthetic validation run `20260902T173900Z-358463289f6e-hook` failed before data access because the fixed Transformers build exposes the CLIP encoder at `model.vision_tower.encoder.layers`, while the registry retained an older nested namespace. The corrected `encoder.layers.22` path must pass a new synthetic forward-path proof before measurement dispatch.
- Release condition: model and dataset checks pass, then the registered hook, patient bootstrap, repeated control, intervention sweep, immutable receipt, and pinned read-only Claude review all complete successfully.
- Next action: push the environment-specific module-path correction and rerun the immutable synthetic hook proof; do not dispatch the one-shot test measurement until it passes.

## Transition rules

`PLANNED -> RUNNING` requires a clean pushed commit and dispatch receipt. `RUNNING -> FAILED` is reserved for implementation, measurement, or infrastructure invalidity. `RUNNING -> OBSERVED` requires valid positive or negative scientific evidence. A reviewer, command, plan, or absent run is not an observation. Gate disposition becomes `UNAVAILABLE` when the pinned reviewer identity/read-only proof cannot be established, and `BLOCKED` for external prerequisites.
