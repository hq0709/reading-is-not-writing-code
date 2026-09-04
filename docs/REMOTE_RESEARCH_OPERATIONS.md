# Remote research operations

The generic infrastructure contract is [`../auto-research-contracts/local-codex-remote-server-setup.md`](../auto-research-contracts/local-codex-remote-server-setup.md). The values below are the Concept Flow binding.
The reset-specific inventory, download list, fixed paths, and ordered rebuild procedure are in [`SERVER_RESET_RECOVERY.md`](SERVER_RESET_RECOVERY.md).

## Fixed topology and environment

- Research source: `wy-coliney/concept-flow`; local checkout `concept-flow-code/`; server checkout `/home/qingchan/work/concept-flow`.
- Publication source: `wy-coliney/concept-flow-paper`; local sibling checkout `concept-flow-paper/`; `main.tex` stays at its repository root for user-managed GitHub-to-Overleaf import.
- Data/runs: `/home/qingchan/data/concept-flow`; ARIS: `/home/qingchan/aris_repo`.
- Environment: `/home/qingchan/miniforge3/envs/conceptflow`; activate with `source scripts/server/activate_env.sh`; install with `UV_PROJECT_ENVIRONMENT` equal to that prefix and `uv sync --frozen`.
- Python 3.13; PyTorch 2.7.0 from the official cu126 index. Import smoke verifies CUDA availability and an A100 device.
- Tests: `python -m unittest discover -s tests -v && python -m compileall -q src`.
- tmux sessions `concept-flow-aris` and `concept-flow-exp` use `/home/qingchan/.cache/tmux/concept-flow.sock`.

Every server launcher resolves declared paths below `/home/qingchan`, activates the environment explicitly, and fails closed on dirty/divergent Git, wrong SHA, missing login/reviewer identity, stop sentinel, disk, or budget gates. Source moves only through GitHub. Runs use an archived immutable commit snapshot, a separate process group, and receipts under the data root. Fetch stages to `.partial-*`, verifies SHA-256 and required metadata, and refuses overwrite.

Here, immutable means the dispatcher writes a terminal snapshot that the single-writer workflow treats as append-only. The run-local checksum is an audit and corruption-detection record, not a cryptographic boundary against deliberate tampering by the same Unix account; that stronger threat model requires separately protected storage or signatures.

Codex CLI is pinned to 0.152.1; Claude Code is pinned to 2.1.258; ARIS is pinned to the SHA in `config/autoresearch.env`. `config/toolchain-provenance.tsv` owns official source URLs, verified installer hashes, expected versions and binary hashes, and sanitized installation commands. From a clean server checkout, `bash scripts/server/write_setup_receipt.sh` verifies those pins and writes `/home/qingchan/data/concept-flow/state/setup-receipt.tsv` atomically with mode 600. The receipt adds UTC, host, current commit, observed versions and installed binary hashes, and the live ARIS full SHA. Authentication uses official interactive commands and is never recorded.

`paper/` is the accepted evidence-linked snapshot retained with the research code. The sibling `concept-flow-paper/` repository is the Overleaf-facing manuscript source. Keep the two Git repositories separate and synchronize only through their GitHub remotes.
