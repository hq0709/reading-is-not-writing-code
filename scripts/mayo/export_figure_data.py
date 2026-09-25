"""Export exactly the per-sample rows the figure scripts read, and nothing else.

Figures 3 and A2 are the two that leave the packaged records and go back to the scored rows: they draw
P(yes) for named images under the clean pass and under each concept write. The merged CORE.parquet of one
block is 24 MB over 457,200 rows and 33 columns, and those figures read 25,200 rows over 8 columns of it --
one template, one fit seed, the baseline and concept directions. Sliced that way a block is a quarter of a
megabyte, so the figure data travels with the repository instead of needing the 35 GB archive.

CheXpert Plus rows are exported like the others, but its cohort and label manifests and its images are not:
the dataset travels under a use agreement. The CheXpert panels of Figure A2 need that data to redraw.
"""
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC = Path("/rodata/azradonc_dev/m253405/cf-transfer/runs")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/rodata/azradonc_dev/m253405/concept-flow/runs")

# the blocks named in plot_paper_figures.py (EXAMPLES for fig3, EXAMPLE_SPEC for figA2)
BLOCKS = [("q25-7", "nih"), ("q25-7", "chexpert"), ("q25-7", "coco"),
          ("lingshu-32", "chexpert"), ("lingshu-32", "coco"),
          ("q25-72", "nih"), ("medgemma-4", "coco")]
CORE_COLS = ["row_id", "concept", "template_id", "fit_seed", "direction_id", "direction_kind",
             "p_present", "sample_status"]
PROBE_COLS = ["row_id", "concept", "probe_kind", "fit_seed", "role", "probability"]

def slim_core(src: Path, dst: Path) -> int:
    t = pq.read_table(src, columns=CORE_COLS,
                      filters=[("template_id", "=", "IY"), ("fit_seed", "=", 0), ("sample_status", "=", "OK")])
    t = t.filter(pc.is_in(t["direction_kind"], value_set=pa.array(["baseline", "concept"])))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, dst, compression="zstd")
    return dst.stat().st_size

def slim_probe(src: Path, dst: Path) -> int:
    t = pq.read_table(src, columns=PROBE_COLS, filters=[("probe_kind", "=", "real"), ("fit_seed", "=", 0), ("role", "=", "test")])
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, dst, compression="zstd")
    return dst.stat().st_size

total, n = 0, 0
for mk, ds in BLOCKS:
    d = SRC / mk / ds
    core = d / "outcomes" / "CORE.parquet"
    if core.exists():
        total += slim_core(core, OUT / mk / ds / "outcomes" / "CORE.parquet"); n += 1
    ps = d / "probe_scores.vis.last.parquet"
    if ps.exists():
        total += slim_probe(ps, OUT / mk / ds / "probe_scores.vis.last.parquet"); n += 1
    # the cohort and label manifests are not copied per block: export_runs_bundle.py already keeps one copy
    # per dataset under _manifests/, and the readers fall back to it
print(f"{n} files, {total / 1e6:.1f} MB -> {OUT}")
