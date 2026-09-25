# Reproducing the numbers and the figures

The campaign root is 75 GB: 27 GB of activations, 35 GB of per-sample outcome shards, 4.3 GB of probe
fits. None of it is needed to rebuild a table or a figure. `runs/` in this repository is the part that
is — the packaged per-block records, the robustness analyses, the figure caches, the shard metadata,
and the per-sample rows the two qualitative figures read, sliced to the columns and rows they read.

## What is here

| path | what it is |
|---|---|
| `runs/manifest.csv` | one row per (checkpoint, dataset, module, concept): every grade the paper reports |
| `runs/<model>/<dataset>/run.json` | completed modules, revisions, processing settings, GPU hours |
| `runs/<model>/<dataset>/summary.json` | the block's grades: calibration, write matrix, every module |
| `runs/<model>/<dataset>/preflight.*.json` | the seven acceptance gates at both loci |
| `runs/<model>/<dataset>/outcomes/*/meta-*.json` | rows, seeds, templates and timings of every shard |
| `runs/robustness/*.json` | the analyses behind the appendix tables |
| `runs/figures/*.json` | dose curves and the counts the overview figure draws |
| `runs/_manifests/<dataset>/` | one cohort and label manifest per dataset, not one per block |

Two blocks carry no outcomes and are here anyway, because the paper reports why: `maira2-7` fails every
template at the image-free mapping gate, and `echo-7` fails four of six.

## Rebuilding

```bash
export CF_RUNS=/path/to/this/repo/runs          # the scripts default to the path the campaign ran at
cd /path/to/the/paper/repo
python scripts/build_cf_transfer_tables.py      # Table 1 and the appendix sheets
python scripts/build_robustness_tables.py       # the robustness tables
python scripts/build_numbers.py                 # every \cf... macro the prose uses
python scripts/plot_paper_figures.py            # the figures
python scripts/check_numbers.py                 # must print ALL CONSISTENT
```

## What is not here, and why

**The 75 GB.** Per-sample outcome shards and activations belong in an archive, not a repository. They are
needed only to re-derive a grade from raw rows or to refit a probe.

**CheXpert Plus cohort and label manifests, and CheXpert images.** CheXpert Plus travels under a use
agreement that this repository cannot redistribute around. Every CheXpert *grade* is here — the manifests
are not. Rebuild them from the licensed source with `scripts/mayo/fetch_chexpert_zips.py`.

Figure A2 reads those manifests, so it is the one figure that needs the licensed data; the other eight
rebuild from this repository alone. NIH ChestX-ray14 and COCO manifests are included.

**Checkpoints.** Twenty-five public Hugging Face checkpoints, each pinned to a commit in
`src/cftransfer/adapters/__init__.py` and in `docs/external-replication/protocol.json`. They download
from the Hub; nothing here mirrors them.
