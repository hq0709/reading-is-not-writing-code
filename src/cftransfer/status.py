"""One-screen campaign status: per (model, dataset) what exists and how complete each module is."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from .protocol import DATASETS, MODEL_ORDER, MODULES, expected_rows, primary_template
from .runpaths import RUN_ROOT


def module_progress(rd: Path, module: str, dataset_id: str) -> str:
    exp = expected_rows(module, dataset_id)
    if exp == 0:
        return "-"
    d = rd / "outcomes" / module
    if not d.exists():
        return "0%"
    n = 0
    for p in d.glob("part-*.parquet"):
        n += pq.read_metadata(p).num_rows
    return f"{100 * n / exp:.0f}%"


def main():
    print(f"{'model':12s} {'ds':8s} {'feat':5s} {'fit':4s} {'pre':4s} " + " ".join(f"{m[:7]:>7s}" for m in MODULES))
    for m in MODEL_ORDER:
        for ds in DATASETS:
            rd = RUN_ROOT / m / ds
            if not rd.exists():
                continue
            feat = (rd / "features" / "vis.last.npz").exists()
            fit = (rd / "fits" / "vis.last" / "seed2.npz").exists() and (rd / "fits" / "connector" / "seed0.npz").exists()
            pre = "-"
            if (rd / "preflight.vis.last.json").exists():
                r = json.loads((rd / "preflight.vis.last.json").read_text())
                c = r["checks"]
                cases = c["E_semantic_direction"]["cases"]
                tpl = primary_template(rd) if primary_template(rd) in cases else "IY"    # the template the block is scored on
                core_ok = (c["A_determinism_max_abs_diff"] == 0 and c["B_alpha0_max_abs_diff"] == 0
                           and c["C_reach"]["locus_change_outside_consumed_tokens"] == 0 and c["G_fp32_vs_model_logits"]["pass"]
                           and c["D2_replicated_batch_vs_single"]["within_batch_spread_pass"]
                           and all(x["ok"] for x in cases[tpl]))
                pre = "ok" if r.get("pass") else ("elig" if core_ok else "FAIL")   # elig: some templates ineligible
            print(f"{m:12s} {ds:8s} {'yes' if feat else '-':5s} {'yes' if fit else '-':4s} {pre:4s} "
                  + " ".join(f"{module_progress(rd, mod, ds):>7s}" for mod in MODULES))


if __name__ == "__main__":
    main()
