# Research state

Last bootstrap update: 2026-09-02. This file distinguishes scientific state (`PLANNED`, `RUNNING`, `FAILED`, `OBSERVED`) from gate disposition (`READY`, `BLOCKED`, `UNAVAILABLE`). Existing evidence remains in `docs/02_RESULTS.md` and `docs/10_RESULTS.md`; this bootstrap does not invent new observations.

## Bootstrap gate

- Scientific state: `PLANNED`
- Gate disposition: `READY`
- Accepted implementation: pushed commit `958c33ef90223da29f93dc7c14477ca5651c5102`.
- Environment, capability, and safety audits: `/home/qingchan/data/concept-flow/state/aris-audit.env`, `/home/qingchan/data/concept-flow/state/health.env`, and `/home/qingchan/data/concept-flow/state/bootstrap-safety-acceptance.env`.
- Reviewer evidence: fresh-shell receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260902T074946088752Z.json`; detached-tmux receipt `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260902T075031584500Z.json`.
- Executor-to-reviewer evidence: `/home/qingchan/data/concept-flow/state/codex-mcp-probe-20260902T075246Z.jsonl`.
- Immutable GPU/fetch evidence: run `20260902T071539Z-7a4e7552d9e1-75d3510c` under `/home/qingchan/data/concept-flow/runs/`.
- Writer ownership: local writer through this state commit; server Codex becomes writer when the persistent ARIS session starts.

## First hard gate

- Scientific state: `PLANNED`
- Gate disposition: `READY`
- Measurement: corrected LLaVA vision hook, corrected registered intervention scale, repeated control draws, and bootstrap intervals on the smallest informative registered cell.
- Supported interpretation: none until a valid immutable run is `OBSERVED`.
- Unresolved hypothesis: whether peak decodability predicts selective causal use after correcting known measurement faults.
- Excluded claims: current incomplete D1 runs do not establish either dissociation or alignment.
- Next action: stage the registered model and dataset assets below `/home/qingchan/data/concept-flow/`, then dispatch the smallest registered validity cell from a clean pushed commit.

## Transition rules

`PLANNED -> RUNNING` requires a clean pushed commit and dispatch receipt. `RUNNING -> FAILED` is reserved for implementation, measurement, or infrastructure invalidity. `RUNNING -> OBSERVED` requires valid positive or negative scientific evidence. A reviewer, command, plan, or absent run is not an observation. Gate disposition becomes `UNAVAILABLE` when the pinned reviewer identity/read-only proof cannot be established, and `BLOCKED` for external prerequisites.
