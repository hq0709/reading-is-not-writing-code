# Concept Flow AutoResearch startup

This project applies the bootstrap contract in `../auto-research-contracts/autoresearch-bootstrap-contract.md`. Research writing and record hygiene are routed through `docs/RESEARCH_WRITING_AND_RECORDING.md`; infrastructure is routed through `docs/REMOTE_RESEARCH_OPERATIONS.md`.

## Project card

- Project: Concept Flow; repository `wy-coliney/concept-flow`; default branch `main`.
- Problem: measure how much clinical information a medical VLM uses, rather than treating linear decodability as use.
- Source material: `README.md`, `docs/00_PAPER_PLAN.md`, `docs/02_RESULTS.md`, `docs/10_RESULTS.md`, and `docs/11_RELATED_WORK.md`.
- Core question and protocol: `docs/RESEARCH_PLAN.md`.
- Current gate: `qwen7b-vislast-causal-ownership`; the next decisive experiment is the registered six-direction by six-question matrix on a third patient-disjoint Qwen cohort.
- Evidence root: `/home/qingchan/data/concept-flow/runs`.
- Runtime: local `E:/projects/concept-flow-project`; server `/home/qingchan/work/concept-flow`; data `/home/qingchan/data/concept-flow`.
- Environment: `source scripts/server/activate_env.sh`; Python 3.13; lock `uv.lock`.
- tmux: `concept-flow-aris` and `concept-flow-exp`, using the project socket below the authorized home.
- Executor: Codex profile `concept-flow`, model `gpt-5.6-sol`, high reasoning, Fast disabled and service tier omitted.
- Reviewer: `claude-review-concept-flow`, `claude-fable-5-1`, medium, read-only, with runtime canonical-model verification.
- ARIS: `94d8093ed21d20a790830318190095b9f5036ce8`; allow/forbidden lists in `config/`.
- Limits: 168 wall-clock hours, 336 GPU-hours, 24 hours per run, 500 GB minimum free disk, three consecutive failures, stop sentinel `/home/qingchan/data/concept-flow/STOP`. API spend budget is USD 0 until a paid API path is explicitly authorized; subscription CLI use does not create an API-spend allowance.

Bootstrap is complete only when the acceptance evidence in `docs/RESEARCH_STATE.md` points to passing Git, environment, CLI, reviewer, ARIS, tmux, immutable-run, fetch, safety, and recording checks.

After bootstrap, immutable dispatcher checksums, metadata, asset/run receipts, and terminal verification are reused. Heartbeats and validator-only changes do not rehash complete runs, shards, models, or datasets or rerun the scientific artifact. A passed gate advances immediately to the next authorised experiment. Project reports follow `run -> observation -> gate decision -> next step`, with scope stated once as the next gate's positive question and failures kept in the run/failure ledger.
