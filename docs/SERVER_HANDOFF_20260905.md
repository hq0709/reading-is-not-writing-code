# Server research handoff — 5 September 2026

## Objective and ownership

The user is shutting down the local computer and asked for already-authorized long-running work to be sent directly to the server. After this branch is pushed, checked out cleanly on the server and the project launcher starts, server Codex + ARIS owns code changes on `research/llava-validation-opportunity`. The local project task suspends code writes during that lease. The separate paper repository remains owned by paper task `01a06afe-5c66-7a80-b922-b3ede71eab3e`.

The bounded assignment is to finish executable acceptance, registration, execution and terminal review of the LLaVA validation-only intervention opportunity pilot, then deliver an evidence report and the next research decision. Accepted prior experiments and the manuscript remain valid inputs.

Completion: the bounded assignment is complete. Immutable run `20260905T093108Z-42a43207c848-llava-validation` completed the registered calibration, independent terminal replay and pinned result review. The valid `K=0` result routes to evidence synthesis; `docs/LLAVA_VALIDATION_OPPORTUNITY_RESULTS.md` records the evidence and the next prospective registration question. No follow-on GPU experiment is registered. The server writer returns the repository through a clean pushed Git handoff after integrating this branch into `main`.

## References and current maturity

- `docs/RESEARCH_STATE.md`: current stage, accepted evidence and writer ownership.
- `docs/RESEARCH_PLAN.md#llava-validation-intervention-opportunity`: prospective scientific protocol, population, controls, statistics and budgets. This gate is planned, not an observed result.
- `docs/LLAVA_VALIDATION_COHORT_FEASIBILITY.md`: accepted metadata-only 700 calibration / 100 write patient allocation, all six question class counts and exclusion identities.
- `docs/reviews/llava-validation-opportunity-design.md`: independent design review.
- `docs/reviews/llava-validation-opportunity-implementation.md`: version-bound implementation review, existing tests and four required terminal-checker changes.
- `src/llava_validation_opportunity.py` and `src/run_llava_validation_opportunity.py`: completed pilot implementation with offline preparation, native model preflight, qualification persisted before writes, and deterministic summary.
- `scripts/server/run_llava_validation_opportunity_gate.sh`: immutable source-bound phase launcher.
- `scripts/server/validate_llava_validation_opportunity.py`: independent terminal checker awaiting the four recorded fixes. It is not yet an accepted validator.

## Execution sequence

1. Repair the four terminal-checker findings, retaining independent reference calculations: use dispatcher `SHA256SUMS`; verify NPZ source/route/eligible-question identities and exact write-question set; verify reported qualification and write diagnostics; require exactly twenty calibration controls with coherent shapes before iteration and averaging. Add proportional synthetic regression checks and obtain independent internal review of the fixes.
2. Run complete tests in `/home/qingchan/miniforge3/envs/conceptflow` with OMP/OpenBLAS/MKL threads set to four. The previous local access-denied validation attempt is historical environment evidence, not a scientific result. Verify current server behavior directly.
3. Inspect real input interfaces through metadata/source loading, then obtain pinned read-only registration review of the executable and prospective protocol before preparation fits or new model outcomes. Preserve the configured Codex profile and canonical `claude-fable-5-1`, medium-effort reviewer transport. Record both verdict and actual structured identity/read-only receipt.
4. Commit and push the verified source. Immediately before scientific dispatch, require a clean pushed full SHA, `scripts/server/supervisor.sh --gate-only`, resource availability, stop and budget gates. Use the immutable dispatcher rather than launching from a mutable checkout.
5. Run `scripts/server/run_llava_validation_opportunity_gate.sh full <FULL_SHA>` under an outer `timeout --signal=TERM --kill-after=60 120m`. The registered pilot allows one A100, at most two GPU-hours and two wall-clock hours, a 75-minute preflight projection, and 4,200 + 4,500*K outcomes for K=0..6. Retain all registered patient splits, control construction, dose/sign rules and shared bootstrap seeds.
6. Validate only this new terminal artifact against its dispatcher manifest, independently replay decision-bearing arrays and statistics, and obtain pinned result review. Persist receipts, observed results, gate decision and a concise paper-ready report. Eligible questions lead to a separately registered test-cohort competition proposal; no eligible question leads to evidence synthesis. Future scientific execution requires its own prospective protocol and review gates.

## Persistence and delivery

The project launcher supplies detached `concept-flow-aris` and experiment/supervisor sessions on `/home/qingchan/.cache/tmux/concept-flow.sock`. All managed paths and processes remain under `/home/qingchan/`; source synchronization uses GitHub only. Local Desktop heartbeat availability is not required for the server assignment.

At each safe checkpoint commit and push scoped work and maintain current state. On completion or an external blocker, leave source and receipts discoverable, push research state and record the next action. Integrate the verified branch into main at a safe clean checkpoint; retain this branch while it owns active work. Before local editing resumes, the server must finish the writer handoff with pushed work and a clean checkout. Deliver accepted evidence through code-repository reports for the paper owner to consume when available.
