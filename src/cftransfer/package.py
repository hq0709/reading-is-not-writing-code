"""Assemble the return package for one <model_key>/<dataset_id>: run.json, environment.txt, prompts.json,
coverage.csv, merged outcomes/<MODULE>.parquet, features/ and fits/ (return-format.md)."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .fit import run_id_for
from .images import DATA_ROOT
from .protocol import (CONCEPTS, LOCI, MODULES, MODULES_ADDED_LATER, PROTOCOL_ID, TEMPLATE_ORDER, MODELS, conditions_for,
                       expected_rows, primary_template, question_list, render_question)
from .runner import KEY
from .runpaths import outcomes_dir, run_dir

COVERAGE_COLS = ["run_id", "model_key", "dataset_id", "module", "locus_id", "fit_seed", "template_id", "concept",
                 "expected_rows", "actual_unique_rows", "failed_rows", "execution_status", "baseline_module",
                 "baseline_fit_seed", "reason"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip()
    except Exception:
        return "unknown"


def merge_module(model_key: str, dataset_id: str, module: str) -> tuple[Path | None, dict]:
    part_dir = outcomes_dir(model_key, dataset_id) / module
    parts = sorted(part_dir.glob("part-*.parquet")) if part_dir.exists() else []
    if not parts:
        return None, {}
    table = pa.concat_tables([pq.read_table(p) for p in parts])
    # de-duplicate on the unique key (resumed shards can legitimately re-score a chunk; keep the last)
    df = table.to_pandas()
    before = len(df)
    df = df.drop_duplicates(subset=KEY, keep="last")
    out = outcomes_dir(model_key, dataset_id) / f"{module}.parquet"
    pq.write_table(pa.Table.from_pandas(df, schema=table.schema, preserve_index=False), out, compression="zstd")
    stats = {"parts": len(parts), "rows_raw": before, "rows_unique": len(df),
             "failed": int((df["sample_status"] == "FAILED").sum())}
    return out, stats


def requested_modules(dataset_id: str, rd: Path) -> list[str]:
    """Modules a block must cover to be COMPLETE: every module the protocol requests for the dataset, except that a
    module added to the dataset after the first wave (protocol.MODULES_ADDED_LATER) counts only once it has started
    (its outcomes directory exists), so blocks packaged before the extension keep their COMPLETE status."""
    return [m for m in MODULES if dataset_id in MODULES[m].datasets and expected_rows(m, dataset_id) > 0
            and (m not in MODULES_ADDED_LATER.get(dataset_id, ()) or (rd / "outcomes" / m).exists())]


def coverage_rows(model_key: str, dataset_id: str, module: str, merged: Path | None, primary: str | None = None) -> list[dict]:
    """One row per (question, fit seed) of the module; `primary` is the block's primary template (default: resolved
    from template_eligibility.json), so the expected cells of the single-template modules are the ones actually scored."""
    spec = MODULES[module]
    run_id = run_id_for(model_key, dataset_id)
    primary = primary or primary_template(run_dir(model_key, dataset_id))
    rows = []
    n_role = {"test": 600, "calibration": 400}[spec.role]
    n_rows = min(n_role, spec.row_limit) if spec.row_limit else n_role
    if dataset_id not in spec.datasets:
        return [{"run_id": run_id, "model_key": model_key, "dataset_id": dataset_id, "module": module, "locus_id": LOCI[spec.locus],
                 "fit_seed": "", "template_id": "", "concept": "", "expected_rows": 0, "actual_unique_rows": 0, "failed_rows": 0,
                 "execution_status": "NOT_REQUESTED", "baseline_module": spec.baseline_module or "", "baseline_fit_seed": "",
                 "reason": "module not requested for this dataset by protocol"}]
    counts = defaultdict(lambda: [0, 0])
    if merged is not None:
        t = pq.read_table(merged, columns=["concept", "template_id", "fit_seed", "sample_status"]).to_pandas()
        for (c, tpl, s, st), n in t.groupby(["concept", "template_id", "fit_seed", "sample_status"]).size().items():
            counts[(c, tpl, int(s))][0 if st == "OK" else 1] += int(n)
    elig_path = run_dir(model_key, dataset_id) / "template_eligibility.json"
    elig = json.loads(elig_path.read_text()) if elig_path.exists() else {}
    for concept, tpl in question_list(dataset_id, module, primary):
        for seed in spec.fit_seeds:
            exp = len(conditions_for(module, dataset_id, concept)) * n_rows
            ok, failed = counts.get((concept, tpl, seed), [0, 0])
            status = "NOT_STARTED" if ok + failed == 0 else ("COMPLETE" if ok == exp and failed == 0 else ("FAILED" if failed else "RUNNING"))
            if not elig.get(tpl, {}).get("eligible", True) and ok + failed == 0:
                rows.append({"run_id": run_id, "model_key": model_key, "dataset_id": dataset_id, "module": module,
                             "locus_id": LOCI[spec.locus], "fit_seed": seed, "template_id": tpl, "concept": concept,
                             "expected_rows": exp, "actual_unique_rows": 0, "failed_rows": 0, "execution_status": "NOT_STARTED",
                             "baseline_module": spec.baseline_module or "", "baseline_fit_seed": 0 if spec.baseline_module else "",
                             "reason": "INELIGIBLE: " + elig[tpl].get("reason", "template failed interface preflight")})
                continue
            rows.append({"run_id": run_id, "model_key": model_key, "dataset_id": dataset_id, "module": module,
                         "locus_id": LOCI[spec.locus], "fit_seed": seed, "template_id": tpl, "concept": concept,
                         "expected_rows": exp, "actual_unique_rows": ok, "failed_rows": failed, "execution_status": status,
                         "baseline_module": spec.baseline_module or "", "baseline_fit_seed": 0 if spec.baseline_module else "",
                         "reason": "" if status in ("COMPLETE", "NOT_STARTED") else f"{ok}/{exp} ok, {failed} failed"})
    return rows


def prompts_json(model_key: str, dataset_id: str, settings: dict, primary: str = "IY") -> dict:
    """Every (concept, template) question with its rendered prompt: the model's chat template around the question text
    of that template (example_prompt_IY is the wrapper rendered around the placeholder question). `is_primary` marks the
    block's primary template, the one CORE/CALIBRATION/DOSE/REFIT/LOCUS/LOCUS_CALIBRATION scored."""
    cands = settings["candidate_tokens"]
    out = {}
    for concept in CONCEPTS[dataset_id]:
        for t in TEMPLATE_ORDER:
            q = render_question(dataset_id, concept, t)
            out[f"{dataset_id}|{concept}|{t}"] = {
                "dataset_id": dataset_id, "concept": concept, "template_id": t, "is_primary": t == primary, "question": q,
                "rendered_prompt": settings["example_prompt_IY"].replace("Is there X in this image? Answer yes or no.", q),
                "generation_prompt": "chat template with add_generation_prompt=True; answer read at the final input position",
                "sequence_index_rule": "logits[:, -1] with left padding; no generated tokens",
                "positive_token_ids": cands[t]["positive_ids"], "negative_token_ids": cands[t]["negative_ids"],
                "candidate_detail": cands[t]["detail"],
                "raw_ab": {"A_ids": cands.get("IA", {}).get("positive_ids", []), "B_ids": cands.get("IA", {}).get("negative_ids", [])} if t in ("IA", "IB", "WA", "WB") else None,
                "score_aggregation": "semantic_margin = max(positive) - max(negative); p_present = sigmoid(margin); lse_margin = logsumexp(positive) - logsumexp(negative); raw_ab_margin = max(A) - max(B)",
            }
    return out


def build(model_key: str, dataset_id: str, status: str = "RUNNING") -> dict:
    rd = run_dir(model_key, dataset_id)
    rd.mkdir(parents=True, exist_ok=True)
    primary = primary_template(rd)
    cov, merged_stats, completed, ineligible = [], {}, [], []
    for module in MODULES:
        merged, stats = merge_module(model_key, dataset_id, module)
        merged_stats[module] = stats
        rows = coverage_rows(model_key, dataset_id, module, merged, primary)
        cov += rows
        def terminal(r):   # INELIGIBLE template cells are a recorded interface disposition, not pending work
            return r["execution_status"] in ("COMPLETE", "NOT_REQUESTED") or \
                (r["execution_status"] == "NOT_STARTED" and r["reason"].startswith("INELIGIBLE"))
        if rows and all(terminal(r) for r in rows) and any(r["execution_status"] == "COMPLETE" for r in rows):
            completed.append(module)
        elif rows and all(terminal(r) for r in rows) and any(r["reason"].startswith("INELIGIBLE") for r in rows):
            ineligible.append(module)      # every requested cell of the module is an INELIGIBLE template: terminal, no outcomes
    requested = requested_modules(dataset_id, rd)
    if status == "RUNNING" and requested and all(m in completed or m in ineligible for m in requested):
        status = "COMPLETE"      # derived from coverage, so a block never stays RUNNING once every cell is terminal
    with (rd / "coverage.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COVERAGE_COLS); w.writeheader(); w.writerows(cov)
    # manifests copy
    (rd / "manifests").mkdir(exist_ok=True)
    for name in ("cohort.csv", "labels.csv"):
        shutil.copy(DATA_ROOT / dataset_id / "manifests" / name, rd / "manifests" / name)
    # environment
    try:
        env = subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=freeze"], text=True)
    except Exception as e:
        env = f"pip list failed: {e}"
    (rd / "environment.txt").write_text(env)
    # metadata from preflight / feature meta / runner meta; merge per-locus preflight files into preflight.json
    pre = {}
    for p in rd.glob("preflight.*.json"):
        if p.name != "preflight.json":
            pre[p.stem.split(".", 1)[1]] = json.loads(p.read_text())
    if pre:
        (rd / "preflight.json").write_text(json.dumps(pre, indent=1))
    deviations = []
    for lid, r in pre.items():
        for k in r.get("tolerance_deviations", []):
            deviations.append({"kind": "batch_vs_single_tolerance", "locus_id": lid, "check": k,
                               "value": r["checks"][k].get("max_abs_candidate_logit_diff"), "declared_tolerance": 0.25,
                               "affected_modules": "none (fixed batch composition per question); recorded for transparency"})
        if not r.get("pass"):
            deviations.append({"kind": "preflight_not_passed", "locus_id": lid,
                               "failed_checks": [k for k, v in r["checks"].items() if isinstance(v, dict) and v.get("pass") is False]})
    # single probe_scores.parquet from the per-locus files
    ps = sorted(rd.glob("probe_scores.*.parquet"))
    if ps:
        pq.write_table(pa.concat_tables([pq.read_table(x) for x in ps]), rd / "probe_scores.parquet", compression="zstd")
    settings = None
    if (rd / "processing_settings.json").exists():
        settings = json.loads((rd / "processing_settings.json").read_text())
    metas = [json.loads(p.read_text()) for p in (rd / "outcomes").glob("*/meta-*.json")] if (rd / "outcomes").exists() else []
    gpu_seconds = sum(m["seconds"] * m.get("gpu_count", 1) for m in metas)
    feat_meta = json.loads((rd / "features" / "features_meta.json").read_text()) if (rd / "features" / "features_meta.json").exists() else {}
    import torch, transformers, sklearn
    run = {
        "protocol_id": PROTOCOL_ID, "run_id": run_id_for(model_key, dataset_id), "model_key": model_key,
        "model_id": MODELS[model_key]["model_id"],
        "model_revision": (pre.get("vis.last") or {}).get("revision"), "processor_revision": (pre.get("vis.last") or {}).get("revision"),
        "dataset_id": dataset_id, "dataset_release": {"nih": "ChestX-ray14 (NIH Box release, images_001..012.zip); labels Data_Entry_2017_v2020.csv",
                                                       "coco": "COCO 2017 train2017/val2017 + instances annotations (2017-09-01)",
                                                       "chexpert": "see manifests/release.json"}[dataset_id],
        "source_commit": git_commit(), "adapter_path": "src/cftransfer/adapters/",
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
        "weight_dtype": "bfloat16", "activation_dtype": "bfloat16 (hook delta computed in float32, cast back)",
        "logit_dtype": "float32", "attention_backend": (settings or {}).get("attn_implementation", "sdpa"),
        "device_map": sorted({str(m.get("device_map", "cuda:0")) for m in metas}) or ["cuda:0"],
        "gpu_models": sorted({g for m in metas for g in m.get("gpu_models", [])}),
        "gpu_count": max([m.get("gpu_count", 1) for m in metas] or [1]),
        "batch_size": sorted({m.get("batch") for m in metas if m.get("batch")}),
        "started_utc": min([m.get("ended_utc", "") for m in metas] or [""]), "ended_utc": max([m.get("ended_utc", "") for m in metas] or [""]),
        "wall_seconds": round(sum(m["seconds"] for m in metas), 1), "gpu_hours": round(gpu_seconds / 3600, 3),
        "peak_gpu_memory_bytes": max([m.get("peak_gpu_memory_bytes", 0) for m in metas] or [0]),
        "processing_settings": settings, "loci": feat_meta.get("loci"),
        "fit_seeds": [0, 1, 2], "random_seed": 0, "projection_seed": 0, "cohort_file": "manifests/cohort.csv",
        "primary_template": primary, "requested_modules": requested,
        "completed_modules": completed, "ineligible_modules": ineligible, "status": status, "deviations": deviations,
        "feature_extraction_seconds": feat_meta.get("seconds"),
        "notes": "gpu_hours sums wall time x GPUs over runner shards recorded in outcomes/*/meta-*.json; CPU fitting time is in fits/*/summary.json",
        "merged_outcomes": merged_stats,
    }
    (rd / "run.json").write_text(json.dumps(run, indent=1))
    # summary.json (written by cftransfer.analysis) carries the same top-level primary_template for the table/figure code
    sp = rd / "summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text())
        if summary.get("primary_template") != primary:
            summary["primary_template"] = primary
            sp.write_text(json.dumps(summary, indent=1))
    if settings:
        (rd / "prompts.json").write_text(json.dumps(prompts_json(model_key, dataset_id, settings, primary), indent=1))
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--status", default="RUNNING")
    a = ap.parse_args()
    r = build(a.model_key, a.dataset, a.status)
    print(json.dumps({k: r[k] for k in ("run_id", "completed_modules", "gpu_hours", "merged_outcomes")}, indent=1))
