"""Copy the part of cf-transfer/runs that regenerates every number in the paper.

The campaign root is 75 GB: 27 GB of activations, 35 GB of per-sample outcome shards, 4.3 GB of probe fits.
None of that is needed to rebuild a table. The builders read the packaged per-block records, the robustness
analyses and the figure caches, which come to a few tens of megabytes, so those are what ships.

Two things are deliberately left out. The per-sample outcome shards and the activations, because they are the
75 GB and belong in an archive rather than a repository. And the CheXpert Plus cohort and label manifests,
because CheXpert Plus is distributed under a use agreement that this repository cannot redistribute around;
NIH ChestX-ray14 and COCO manifests ship, and the CheXpert ones rebuild from the licensed source with
scripts/mayo/fetch_chexpert_zips.py.

The manifests are also deduplicated: a cohort is a property of the dataset, and the campaign stored one copy
per block, so the same NIH labels.csv sits in the tree twenty-five times.
"""
import shutil
import sys
from pathlib import Path

SRC = Path("/rodata/azradonc_dev/m253405/cf-transfer/runs")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/rodata/azradonc_dev/m253405/concept-flow/runs")
PER_BLOCK = ("run.json", "summary.json", "coverage.csv", "preflight.vis.last.json",
             "preflight.connector.json", "template_eligibility.json", "processing_settings.json",
             "answer_interface_probe.json", "prompts.json")
REDISTRIBUTABLE = ("nih", "coco")          # CheXpert Plus travels under its own use agreement

def copy(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return src.stat().st_size

total, files = 0, 0
for name in ("manifest.csv", "leaderboard.csv", "leaderboard.md"):
    if (SRC / name).exists():
        total += copy(SRC / name, OUT / name); files += 1
for sub in ("robustness", "figures"):
    for f in sorted((SRC / sub).glob("*.json")):
        total += copy(f, OUT / sub / f.name); files += 1
# every block directory, not only the ones with a run.json: a checkpoint that produced no outcomes has no
# run.json and would vanish from the bundle, and MAIRA-2 is exactly that -- the paper reports its gate failure
# from template_eligibility.json and answer_interface_probe.json, which is all it ever wrote.
dirs = sorted({f.parent for name in PER_BLOCK for f in SRC.glob(f"*/*/{name}")})
blocks = 0
for d in dirs:
    blocks += 1
    for name in PER_BLOCK:
        if (d / name).exists():
            total += copy(d / name, OUT / d.parent.name / d.name / name); files += 1
# the outcome shards' meta-*.json: a few kilobytes each, recording the rows, seeds, templates and timings of
# every shard. The parquet beside them is the 35 GB and stays out; the analyses read only these.
for meta in sorted(SRC.glob("*/*/outcomes/*/meta-*.json")):
    rel = meta.relative_to(SRC)
    total += copy(meta, OUT / rel); files += 1

# one copy of each dataset's cohort, not one per block
for ds in REDISTRIBUTABLE:
    for name in ("cohort.csv", "labels.csv"):
        found = sorted(SRC.glob(f"*/{ds}/manifests/{name}"))
        if found:
            total += copy(found[0], OUT / "_manifests" / ds / name); files += 1
print(f"{files} files, {blocks} blocks, {total / 1e6:.1f} MB -> {OUT}")
