# Working guide

Two repositories. This one holds the campaign — adapters, protocol, runners, analyses, and the slice of
results that rebuilds the paper. The other holds the paper — LaTeX, tables, figures, and the scripts that
turn results into numbers.

| | repository | branch |
|---|---|---|
| code | `hq0709/reading-is-not-writing-code` | `main` |
| paper | `hq0709/reading-is-not-writing` | `main` |

Both are public. `REPRODUCING.md` covers what `runs/` contains and what it deliberately does not.

---

## 1. Rebuild the paper

```bash
export CF_RUNS=/path/to/code-repo/runs        # omit on the machine the campaign ran on
cd /path/to/paper-repo
bash /path/to/code-repo/scripts/mayo/refresh_numbers.sh
```

That one script is the whole chain, in the order the dependencies demand:

1. **repackage every block** — `run.json`'s module list is derived from the outcomes on disk, and the
   watcher stops packaging a block after two attempts, so a block that gained a module later keeps a
   stale list
2. **re-analyse the stale blocks** — only those whose outcomes are newer than their `summary.json`
3. `cftransfer.manifest` → `runs/manifest.csv`
4. the seven robustness analyses → `runs/robustness/*.json`
5. `build_cf_transfer_tables.py`, `build_robustness_tables.py`, `build_numbers.py`
6. `plot_paper_figures.py`
7. `latexmk`
8. `check_numbers.py`

**Step 8 is the gate. If it does not print `ALL CONSISTENT`, the PDF is wrong.** It compares the prose,
the tables, the figure caches and the macros against `manifest.csv` — three independent paths to the same
number, and it fails when they disagree.

Running the steps by hand is fine; running them out of order is not. A table built before the manifest
is a table of yesterday's numbers.

---

## 2. Change one thing

**A number.** Never type it. Every number in the prose is a macro from `tables/cf_numbers.tex`. Find the
macro, change what produces it in `build_numbers.py`, re-run the chain. If a number appears in the text
without a macro, that is the bug.

**A figure.** Edit `plot_paper_figures.py`, then `python scripts/plot_paper_figures.py <name>` to rebuild
one. It refuses to finish if labels collide — the overlap check is not advisory.

**A table's look.** Headers are normalised by one pass, `scripts/table_headers.py`, which both builders
run over everything they write. Edit the pass, not the forty-nine inline headers.

**Prose.** Paragraph-by-paragraph through `scripts/gpt6_polish.py`, never by hand and never by a model
other than the one that pass uses. `--only <substring>` targets single paragraphs.

---

## 3. Add a checkpoint

1. **Register it** in `docs/external-replication/protocol.json` (`planned_models`) with the model id and
   the resolved commit, and in `src/cftransfer/adapters/__init__.py` (`REVISIONS`, `FAMILY_ADAPTER`).
2. **Write the adapter.** It answers four questions and nothing else: how to load the checkpoint, which
   module output is the consumed visual block, which tensor entries are that image's tokens, and how to
   read the answer-position logits. If the family already exists, subclass it — `adapters/echo.py` is
   thirty lines because the vision side is Qwen2.5-VL renamed.
3. **Stage it** if the released code needs patching. `scripts/mayo/convert_echo.py` is the pattern:
   symlink the weights, patch only what blocks loading, and write a `conversion.json` that records every
   change, its reason, and its arithmetic effect. Assert that each patch target appears exactly once.
4. **Preflight**, on a compute node:
   ```bash
   bash scripts/mayo/sbatch_py.sh -J prep-<key> -g 1 -t 08:00:00 -- \
        bash scripts/mayo/prep_model.sh <key> nih 16 cuda:0
   ```
   This extracts features for all four roles, fits the probes, and runs the seven gates at both loci.
5. **Read the verdict before running anything else.** `preflight.vis.last.json` → `pass`. Gate E is the
   one that decides whether the checkpoint can be graded at all: it states the finding in words and
   requires the answer to follow. A report generator fails it — `maira2-7` and `echo-7` both did, and
   both are in the protocol with their measured numbers.
6. **Run the modules** and refresh.

---

## 4. Run modules at scale

```bash
python -m cftransfer.enqueue --models <keys> --datasets <sets> --modules <names> [--no-prep] [--prefix 2]
bash scripts/mayo/sbatch_py.sh -J worker -g 1 -t 14:00:00 -- \
     python -m cftransfer.worker --lane gpu1 --max-hours 13 --exit-when-empty
```

Tasks are files in `queue/{pending,running,done,failed}`, with `hold` and `cancelled` beside them.
Workers claim them in **filename order**, and
`--prefix` is how priority is expressed: `--prefix 2` sorts after everything already queued. Dependencies
are recorded as **file paths**, not task names, so renaming a pending task to reprioritise it is safe —
which is the trick for pulling a CORE shard ahead of a PROMPT sweep.

Signals are handled: SIGTERM requeues the task, so cancelling a worker loses at most its current task.

**A 4-GPU job needs a whole free node.** Single-GPU workers scattered across nodes will starve it
indefinitely — the fix is to free a node, not to wait.

---

## 5. The environment will fight you

**Thread cap.** `ulimit -u` is 1024 and counts *threads across the whole account*. An interactive agent
session holds ~500. Cross the line and every `fork` fails, including the `pkill` you would use to fix it.

- start every interactive command with `ulimit -u 65536`
- launch long jobs through `scripts/jobwrap.sh`, which refuses above 800 live threads
- **anything that loads a model onto a GPU goes to `sbatch`**, never to the login node
- `pgrep -f` and `pkill -f` match your own command line: match on a bracketed pattern or kill by PID

**GPU size.** Compute nodes have 80 GB cards; the login node has 40 GB. Size the batch for 80.

**Credentials.** The Hugging Face token and the Redivis token are 0600 files read by path. Never print,
log or commit them. The research env exports a stale `HF_TOKEN` that overrides the cached credential —
unset it in any download script.

---

## 6. What the guards are for

Each of these exists because it caught something real:

| guard | catches |
|---|---|
| `check_numbers.py` | prose, tables and macros disagreeing about the same quantity |
| overlap check in the figure script | labels colliding, which a glance at a thumbnail misses |
| manifest-vs-summaries check | numbers built from a half-written snapshot |
| dose-curve cache check | a cache that predates the blocks it claims to summarise |
| coverage check | prose describing a module coverage the runs do not have |
| `conversion.json` | a patched checkpoint whose deviation from the release is undocumented |

`check_numbers.py` passing does **not** mean the numbers are right — it means the three paths agree. They
agreed once over a manifest that was missing five blocks. When a count looks wrong, check the block count
first.

---

## 7. Boundaries

**CheXpert Plus** travels under a use agreement. Every grade is in `runs/`; the cohort and label manifests
and the images are not. Figure A2 is the one figure that needs them.

**The 75 GB** — per-sample outcome shards and activations — is not in any repository. It is needed only to
re-derive a grade from raw rows or to refit a probe.

**Checkpoints** download from the Hub, pinned by commit. Nothing here mirrors them.
