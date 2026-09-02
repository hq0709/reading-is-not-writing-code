# Remote research operations

The generic infrastructure contract is [`../auto-research-contracts/local-codex-remote-server-setup.md`](../auto-research-contracts/local-codex-remote-server-setup.md). The values below are the Concept Flow binding.

## Fixed topology and environment

- Source: `wy-coliney/concept-flow`; server checkout `/home/qingchan/work/concept-flow`.
- Data/runs: `/home/qingchan/data/concept-flow`; ARIS: `/home/qingchan/aris_repo`.
- Environment: `/home/qingchan/miniforge3/envs/conceptflow`; activate with `source scripts/server/activate_env.sh`; install with `UV_PROJECT_ENVIRONMENT` equal to that prefix and `uv sync --frozen`.
- Python 3.13; PyTorch 2.7.0 from the official cu126 index. Import smoke verifies CUDA availability and an A100 device.
- Tests: `python -m unittest discover -s tests -v && python -m compileall -q src`.
- tmux sessions `concept-flow-aris` and `concept-flow-exp` use `/home/qingchan/.cache/tmux/concept-flow.sock`.

Every server launcher resolves declared paths below `/home/qingchan`, activates the environment explicitly, and fails closed on dirty/divergent Git, wrong SHA, missing login/reviewer identity, stop sentinel, disk, or budget gates. Source moves only through GitHub. Runs use an archived immutable commit snapshot, a separate process group, and receipts under the data root. Fetch stages to `.partial-*`, verifies SHA-256 and required metadata, and refuses overwrite.

Codex CLI is pinned to 0.152.1; Claude Code is pinned to 2.1.258; ARIS is pinned to the SHA in `config/autoresearch.env`. Installer sources, hashes, runtime versions, and acceptance commands are stored in the server setup receipt. Authentication uses official interactive commands and is never recorded.

`paper/` is authoritative. `paper-overleaf/` is an ignored separate publishing clone and is `CONDITIONAL N/A` until configured.
