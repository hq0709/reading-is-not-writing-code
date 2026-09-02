# Research state

Last bootstrap update: 2026-09-02. This file distinguishes scientific state (`PLANNED`, `RUNNING`, `FAILED`, `OBSERVED`) from gate disposition (`READY`, `BLOCKED`, `UNAVAILABLE`). Existing evidence remains in `docs/02_RESULTS.md` and `docs/10_RESULTS.md`; this bootstrap does not invent new observations.

## Bootstrap gate

- Scientific state: `PLANNED`
- Gate disposition: `BLOCKED`
- Goal: complete environment, Git, CLI, reviewer, ARIS, immutable-dispatch and receipt acceptance.
- Blockers: interactive Codex/Claude authentication and live verification that Claude returns the pinned Fable canonical identity for biomedical review; server-local research datasets/model assets for full experiments. Infrastructure smoke does not require the latter.
- Evidence: to be filled only with pushed commit SHA and `/home/qingchan/data/concept-flow/...` receipt paths.

## First hard gate

- Scientific state: `PLANNED`
- Gate disposition: `BLOCKED` until bootstrap acceptance and required model/data assets exist.
- Measurement: corrected LLaVA vision hook, corrected registered intervention scale, repeated control draws, and bootstrap intervals on the smallest informative registered cell.
- Supported interpretation: none until a valid immutable run is `OBSERVED`.
- Unresolved hypothesis: whether peak decodability predicts selective causal use after correcting known measurement faults.
- Excluded claims: current incomplete D1 runs do not establish either dissociation or alignment.

## Transition rules

`PLANNED -> RUNNING` requires a clean pushed commit and dispatch receipt. `RUNNING -> FAILED` is reserved for implementation, measurement, or infrastructure invalidity. `RUNNING -> OBSERVED` requires valid positive or negative scientific evidence. A reviewer, command, plan, or absent run is not an observation. Gate disposition becomes `UNAVAILABLE` when the pinned reviewer identity/read-only proof cannot be established, and `BLOCKED` for external prerequisites.
