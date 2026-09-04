# Concept Flow agent contract

The local workspace topology is fixed: `E:\projects\concept-flow-project\concept-flow-code` and `E:\projects\concept-flow-project\concept-flow-paper` are sibling, independent Git repositories. The paper repository is the Overleaf-facing manuscript source. Preserve this two-repository layout.

Read and follow `docs/RESEARCH_WRITING_AND_RECORDING.md` for research judgment, human-facing prose, AutoResearch state updates, experiment summaries, and evidence records. It routes to the source-authoritative contract in `auto-research-contracts/`. The supervisor also advances the research: when environment and gate requirements pass, resume or launch the next registered experiment instead of stopping at inspection or reporting. Reuse immutable dispatcher checksums, metadata, receipts, and terminal verification; hash only the necessary manifest at an explicit terminal hard gate or when concrete contamination evidence exists. A validator-only change reuses the original artifact, a passed gate advances immediately, and meaningful local changes are committed and pushed at a safe checkpoint.

Write every report as `run -> observation -> gate decision -> next step`. State scope once as the positive question for the next gate. Keep failures in the designated ledger and provenance in receipts; do not repeat defensive claim-denial lists, revision history, or resolved blockers in plans, trackers, reviews, state updates, findings, commits, PRs, or papers.

Read `docs/REMOTE_RESEARCH_OPERATIONS.md` for machines, Git handoff, environment, immutable runs, recovery, and the `/home/qingchan/` boundary. On the server, all managed writes, temporary files, tmux sockets, project processes, repositories, environments, caches, logs, data, and models must resolve below `/home/qingchan/`; never use elevated privileges, system paths/services, or other users' resources.

Read `docs/RESEARCH_PLAN.md` for the current problem, fixed protocol, gates, and permitted optimisation. Read `docs/RESEARCH_STATE.md` for the current stage, evidence, blockers, next action, and writer ownership. Full experiments run only on the server; GitHub is the sole source synchronization channel.

Codex is the executor. Claude is used only through the pinned read-only reviewer transport. ARIS must be pinned to a full SHA and installed from the audited allowlist; Superpowers and competing autonomous research pipelines are forbidden.
