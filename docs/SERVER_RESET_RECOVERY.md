# asimov1 reset recovery

This runbook restores Concept Flow after `/home/qingchan` is reset. The platform
keeps the GPU and host system; the account-owned environment, tools, data, caches,
credentials, repositories, tmux sessions, and receipts are treated as lost.

The recovery target is the current project state, not a byte-for-byte copy of the
old home directory. Source and sanitized configuration come from GitHub. Public
models and datasets are downloaded directly on the server. Credentials are
re-created interactively.

## Single pre-reset checkpoint: T-12 hours

This document is preparation only. Keep the current server and research processes
running until approximately twelve hours before the confirmed reset. Because models,
datasets, caches, and the Conda environment are not copied, one checkpoint is
enough; there is no multi-day or multi-checkpoint schedule.

Prepare the small verified installer `bootstrap-cache` once before the reset window
and refresh it only when `config/toolchain-provenance.tsv` changes. The checkpoint
verifies and copies that cache; it does not rediscover installer URLs while the
server is being retired.

At T-12 hours, create the human-owned dispatch freeze first. This does not terminate
an experiment that is already running. The reserved
`AUTORESEARCH_SUPERVISOR_BLOCK_V1` content belongs only to the supervisor and must
never be used for this manual freeze:

```bash
stop=/home/qingchan/data/concept-flow/STOP
tmp=$(mktemp /home/qingchan/data/concept-flow/.STOP.reset-freeze.XXXXXX)
printf '%s\n' 'MANUAL_RESET_FREEZE: no new dispatches before account reset' >"$tmp"
chmod 600 "$tmp"
ln "$tmp" "$stop"
rm -f -- "$tmp"
test "$(stat -c %a "$stop")" = 600
grep -Fx 'MANUAL_RESET_FREEZE: no new dispatches before account reset' "$stop"
```

If `ln` reports that `STOP` already exists, audit the existing sentinel instead of
overwriting it. Keep the manual sentinel in place through the reset; the reset
itself removes the old account-owned filesystem.

Then:

1. let an active run continue only when its registered completion bound leaves a
   six-hour reset margin; otherwise stop it at its registered safe point;
2. after the active run reaches a terminal receipt, have the server writer commit
   and push, then return ownership to the local checkout with `pull --ff-only`;
3. fetch only selected immutable receipts, small results, and other irreplaceable
   evidence;
4. copy the verified installer `bootstrap-cache` and its manifest locally;
5. verify the Git commit and SHA-256 values of the retained files.

Expected operator time is roughly 15–30 minutes, excluding any wait for an active
experiment to finish and any missing installer download. The rest of the twelve-hour
window is buffer for an earlier-than-announced reset, a safe experiment stop, or an
unexpected authentication issue.
No earlier checkpoint or bulk server copy is required.

## Fixed recovery card

| Item | Fixed value |
|---|---|
| SSH alias | `asimov1` |
| Account root | `/home/qingchan` |
| Source authority | private GitHub repository `wy-coliney/concept-flow` |
| Server checkout | `/home/qingchan/work/concept-flow` |
| Data and run root | `/home/qingchan/data/concept-flow` |
| ARIS checkout | `/home/qingchan/aris_repo` |
| Miniforge root | `/home/qingchan/miniforge3` |
| Project environment | `/home/qingchan/miniforge3/envs/conceptflow` |
| Python | `3.13` |
| PyTorch | `2.7.0` from the official CUDA 12.6 wheel index |
| User executables | `/home/qingchan/.local/bin` |
| Codex profile | `/home/qingchan/.codex/concept-flow.config.toml` |
| Agent tmux | `concept-flow-aris` |
| Experiment tmux | `concept-flow-exp` |
| tmux socket | `/home/qingchan/.cache/tmux/concept-flow.sock` |
| Stop sentinel | `/home/qingchan/data/concept-flow/STOP` |

`config/autoresearch.env` is the machine-readable authority for these values.
`config/toolchain-provenance.tsv` is the authority for installer hashes, installed
versions, installed binary hashes, official sources, and sanitized install
commands. `uv.lock` and `pyproject.toml` are the environment authority. Do not
replace them with a copied Conda directory or an ad-hoc `pip freeze`.

## What to retain before reset

Do not copy the full home directory. Before reset:

1. Stop new dispatches and let any valuable active run reach a safe terminal state.
2. Commit and push all source, configuration, paper, and research-state changes.
3. Fetch any valid immutable run that cannot be recomputed economically with
   `.\scripts\fetch_results.ps1 -RunId <RUN_ID>`.
4. Copy only small, irreplaceable evidence that is not already represented in Git:
   completed run receipts and checksums, accepted state receipts, and reviewer
   receipts required for historical audit.
5. Build a small local `bootstrap-cache` containing the exact verified installer
   files used for the versions in `config/toolchain-provenance.tsv`. Complete this
   once before the reset window and refresh it only when the tracked provenance
   changes; do not defer installer discovery until the T-12 checkpoint or recovery.
   Store a local manifest with the columns `component`, `artifact_url`, `filename`,
   `bytes`, and `sha256`, and require each SHA-256 to match the tracked provenance
   manifest. This may include the approximately 101 MiB Miniforge installer; it
   must not include environments, credentials, models, datasets, or ordinary
   caches.
6. Record which evidence was retained and its SHA-256 outside the server.

Do not retain or commit `.codex/auth.json`, Claude credentials, SSH private keys,
tokens, model caches, the Conda environment, tmux state, public model weights,
public dataset archives, `.partial-*` staging trees, or ordinary caches. Login and
SSH credentials are re-created after reset.

A planning-time read-only estimate on 2026-09-02 found approximately 14 GiB of
model files and a 25 GiB incomplete NIH staging tree that can be discarded and
downloaded again. It was not saved as a backup receipt. Re-run `du` during the
pre-reset backup session and let that inventory determine the actual selection;
these figures are not quotas or evidence of a completed backup.

## Server downloads after reset

Only the following public research assets are required for the current first gate.
The staging script owns source URLs, exact sizes, revisions, integrity checks,
partial-download behavior, and atomic publication; this table is an operator index,
not a second manifest.

| Asset | Server download | Final path | Staging path | Verification and receipt |
|---|---:|---|---|---|
| `llava-hf/llava-1.5-7b-hf` at revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` | about 14 GiB | `/home/qingchan/data/concept-flow/models/huggingface` | `/home/qingchan/data/concept-flow/models/.partial-huggingface-b234b804b114d9e37bb655e11cbbb5f5e971b7a9` | revision, four registered file hashes, local config/processor load, and `asset-receipt.json` |
| NIH ChestX-ray14: `images_001.tar.gz` through `images_012.tar.gz` plus `Data_Entry_2017_v2020.csv` | 45,088,866,280 bytes (41.99 GiB compressed) | `/home/qingchan/data/concept-flow/datasets/nih-chestxray14` | `/home/qingchan/data/concept-flow/datasets/.partial-nih-chestxray14` | official Box metadata, exact bytes, gzip/tar safety, 112,120 label rows, committed SHA-256 manifest, registered manifest, and `asset-receipt.json` |

The NIH trust bootstrap is deliberately two-stage:

```bash
cd /home/qingchan/work/concept-flow
source scripts/server/activate_env.sh
python scripts/server/stage_first_gate_assets.py nih-bootstrap
```

This downloads or resumes the 13 official files, validates their structure and
metadata, and writes a same-source candidate bundle under
`/home/qingchan/data/concept-flow/state/nih-chestxray14-sha256-candidate/`. Review
that evidence, then obtain or verify all 13 hashes against an independently trusted
source. The Box downloads and their candidate hashes cannot establish independent
trust for one another. Only after that independent check may the accepted values be
committed to `config/nih-chestxray14-sha256.tsv`, pushed, and fast-forwarded into
the clean server checkout. Then publish the dataset:

```bash
cd /home/qingchan/work/concept-flow
source scripts/server/activate_env.sh
python scripts/server/stage_first_gate_assets.py nih
```

Stage the model independently with:

```bash
cd /home/qingchan/work/concept-flow
source scripts/server/activate_env.sh
python scripts/server/stage_first_gate_assets.py model
```

The downloads are resumable in their `.partial-*` directories. Never transfer
these assets through the local computer and never treat a partial tree as accepted
data.

## Ordered recovery procedure

Run every server command as `qingchan`, without `sudo`. All created paths, temporary
files, environments, caches, logs, repositories, and processes must remain below
`/home/qingchan`.

### 1. Verify the retained platform

```bash
test "$(id -un)" = qingchan
test "$HOME" = /home/qingchan
nvidia-smi
git --version
curl --version
tmux -V
df -h /home/qingchan
```

The GPU is not restored by this runbook. This gate only confirms that the retained
platform still exposes CUDA and the required base commands. If a base command or
the GPU is unavailable, stop and ask the platform administrator; do not modify
system paths or use another user's libraries.

### 2. Restore GitHub access and clone the source

Create a repository-scoped deploy key at
`/home/qingchan/.ssh/concept-flow-deploy`, grant it write access to only
`wy-coliney/concept-flow`, and configure this exact SSH host alias:

```sshconfig
Host github-concept-flow
  HostName github.com
  User git
  IdentityFile /home/qingchan/.ssh/concept-flow-deploy
  IdentitiesOnly yes
```

Require mode `700` on `/home/qingchan/.ssh` and mode `600` on the private key and
SSH config. The public key may use mode `644`.

Verify GitHub's current official SSH host-key fingerprints before populating
`/home/qingchan/.ssh/known_hosts`. Then verify the alias and complete the documented
reversible read/write probe. Clone through the alias so Git cannot silently select
another account key:

```bash
mkdir -p /home/qingchan/work /home/qingchan/data/concept-flow
git clone git@github-concept-flow:wy-coliney/concept-flow.git /home/qingchan/work/concept-flow
cd /home/qingchan/work/concept-flow
git switch main
git pull --ff-only
test -z "$(git status --porcelain --untracked-files=all)"
test "$(git rev-parse HEAD)" = "$(git rev-parse '@{upstream}')"
```

GitHub is the only source synchronization channel. Do not restore a source-tree
archive with SCP or rsync.

### 3. Install the pinned user toolchain

Upload the small, locally retained `bootstrap-cache` to
`/home/qingchan/.cache/autoresearch-setup/`. If a required installer was not
retained, stop: the pre-reset backup session was not completed. The official
project/documentation URLs in `config/toolchain-provenance.tsv` are provenance,
not immutable artifact URLs and must not be rediscovered during recovery. Verify
each cached artifact's filename, byte count, and SHA-256 against the local cache
manifest and tracked provenance before executing the sanitized install command.
The expected results are:

- Miniforge/Conda `26.3.2` at `/home/qingchan/miniforge3/bin/conda`;
- uv and uvx `0.12.8` at `/home/qingchan/.local/bin/`;
- Codex CLI `0.152.1` at `/home/qingchan/.local/bin/codex`;
- Claude Code `2.1.258` at `/home/qingchan/.local/bin/claude`.

The manifest also fixes the installed binary SHA-256 values. An installer or binary
hash mismatch is a stop condition, not permission to accept a nearby version.

Create the project environment once, then let uv reproduce the locked packages:

```bash
source /home/qingchan/miniforge3/etc/profile.d/conda.sh
conda create --yes --prefix /home/qingchan/miniforge3/envs/conceptflow python=3.13
cd /home/qingchan/work/concept-flow
export UV_PROJECT_ENVIRONMENT=/home/qingchan/miniforge3/envs/conceptflow
/home/qingchan/.local/bin/uv sync --frozen
source scripts/server/activate_env.sh
test "$CONDA_PREFIX" = /home/qingchan/miniforge3/envs/conceptflow
test ! -e /home/qingchan/work/concept-flow/.venv
python -c 'import torch; n=torch.cuda.get_device_name(0); assert torch.__version__.startswith("2.7.0"); assert torch.cuda.is_available(); assert "A100" in n; print(torch.__version__, torch.version.cuda, n)'
python -m unittest discover -s tests -v
python -m compileall -q src
```

### 4. Restore ARIS and project-owned configuration

```bash
git clone https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep.git /home/qingchan/aris_repo
git -C /home/qingchan/aris_repo checkout --detach 94d8093ed21d20a790830318190095b9f5036ce8
test "$(git -C /home/qingchan/aris_repo rev-parse HEAD)" = 94d8093ed21d20a790830318190095b9f5036ce8
cd /home/qingchan/work/concept-flow
bash scripts/server/install_profile.sh
bash scripts/server/install_aris.sh
bash scripts/server/install_aris.sh --apply
```

`install_profile.sh` recreates the Codex profile and strict reviewer route from
tracked configuration. `install_aris.sh` dry-runs first, then installs and audits
only the pinned allowlist and reviewer overlay.

### 5. Restore interactive credentials

Run the official Codex and Claude interactive login commands yourself. Never paste
tokens, device codes, passwords, `.codex/auth.json`, or Claude credential files
into Git, a run receipt, or chat. After login, repeat both the fresh-shell and
detached-tmux reviewer probes required by `docs/REMOTE_RESEARCH_OPERATIONS.md`.

### 6. Produce one setup receipt and restore public assets

```bash
cd /home/qingchan/work/concept-flow
bash scripts/server/write_setup_receipt.sh
```

This is the single toolchain acceptance gate. It verifies the tracked provenance,
installed versions and binary hashes, source commit, and pinned ARIS checkout, then
writes `/home/qingchan/data/concept-flow/state/setup-receipt.tsv`. Do not repeat
individual version investigations after this receipt passes.

Run the model and NIH staging commands from the download table. A complete
accepted NIH hash manifest in Git remains authoritative and permits normal `nih`
staging without re-deciding the hashes. If the tracked manifest is incomplete,
run `nih-bootstrap`, complete the approved trust review, commit and push the
accepted manifest, fast-forward the clean server checkout, and only then run
normal `nih` staging.

### 7. Final end-to-end acceptance

The wall-clock and GPU budgets belong to the rebuilt instance and are not restored
from an old receipt. After the user explicitly authorizes a new research budget
window, initialize its epoch once:

```bash
cd /home/qingchan/work/concept-flow
mkdir -p /home/qingchan/data/concept-flow/state
umask 077
target=/home/qingchan/data/concept-flow/state/bootstrap_started_epoch
test ! -e "$target"
tmp=$(mktemp /home/qingchan/data/concept-flow/state/.bootstrap_started_epoch.XXXXXX)
trap 'rm -f -- "$tmp"' EXIT
date +%s > "$tmp"
chmod 600 "$tmp"
grep -Eq '^[0-9]+$' "$tmp"
ln "$tmp" "$target"
rm -f -- "$tmp"
trap - EXIT
test "$(stat -c %a "$target")" = 600
```

Do not run this command merely to clear an expired budget. Its meaning is “the user
approved a new `MAX_WALL_CLOCK_HOURS=168` window on this rebuilt instance.” Without
the file, the supervisor and dispatcher must continue to fail closed.

Then run, in order:

1. the fresh-shell and detached-tmux Claude reviewer probes;
2. `scripts/server/supervisor.sh --gate-only`;
3. one harmless immutable GPU smoke run;
4. `.\scripts\fetch_results.ps1 -RunId <RUN_ID>` from the local checkout;
5. the fail-closed probes for dirty Git, wrong commit, wrong ARIS SHA, missing
   login, wrong reviewer identity, stop sentinel, low disk, and path escape.

Record the new receipts and immutable smoke run ID in
`docs/RESEARCH_STATE.md` and `docs/EXPERIMENT_REGISTRY.md`, preserving the
scientific state as `PLANNED` unless a valid scientific run says otherwise. Commit
and push those updates, then require the clean server checkout to fast-forward to
that commit. Only after this writer handoff is complete may the server run
`scripts/start_aris.sh`. Historical receipts may be kept locally for audit, but
paths deleted by the reset must not remain the current infrastructure evidence and
historical receipts do not prove the rebuilt instance is healthy.

## Recovery completion conditions

Recovery is complete when:

- local, GitHub, and server commits match and the server checkout is clean;
- the fixed Conda prefix is active, no repository `.venv` exists, and the test,
  compile, Torch, and CUDA smokes pass;
- toolchain, ARIS, profile, login, and reviewer receipts are fresh;
- required public assets are atomically staged and their receipts pass;
- a detached immutable smoke run completes and can be fetched and verified;
- the supervisor gate passes and exactly one project agent session can start.
