# Concept Flow agent contract

Read and follow `docs/RESEARCH_WRITING_AND_RECORDING.md` for research judgment, human-facing prose, AutoResearch state updates, experiment summaries, and evidence records. State the current object directly; keep failures, blockers, limitations, and provenance only in their designated evidence locations.

Read `docs/REMOTE_RESEARCH_OPERATIONS.md` for machines, Git handoff, environment, immutable runs, recovery, and the `/home/qingchan/` boundary. On the server, all managed writes, temporary files, tmux sockets, project processes, repositories, environments, caches, logs, data, and models must resolve below `/home/qingchan/`; never use elevated privileges, system paths/services, or other users' resources.

Read `docs/RESEARCH_PLAN.md` for the current problem, fixed protocol, gates, and permitted optimisation. Read `docs/RESEARCH_STATE.md` for the current stage, evidence, blockers, next action, and writer ownership. Full experiments run only on the server; GitHub is the sole source synchronization channel.

Codex is the executor. Claude is used only through the pinned read-only reviewer transport. ARIS must be pinned to a full SHA and installed from the audited allowlist; Superpowers and competing autonomous research pipelines are forbidden.
