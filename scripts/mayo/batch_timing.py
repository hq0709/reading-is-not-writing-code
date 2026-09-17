"""Seconds per forward at the RUNNER's batch composition, and the GPU-hours CORE + CALIBRATION imply.

Preflight check F times 256 image-conditions in its own composition (a batch-1 baseline plus one group of 15
steered conditions), which is not what a block costs: `runner.run_block` scores every condition of a question
at ONE fixed composition -- `batch` replicas of the same image and question, one clean replicated baseline
batch with the hook passive, then ceil((C-1)/batch) steered batches padded to `batch`. Clean-only modules
(CALIBRATION) force batch 1. This script measures exactly those two forwards at a few batch sizes on one
preflight image and reports the implied cost, so the registered batch is chosen on a measurement.

    python scripts/mayo/batch_timing.py --model-key huatuo-7 --dataset nih --batches 1,8,16,32
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from cftransfer.adapters import get_adapter
from cftransfer.fit import load_fit
from cftransfer.hooks import LocusHook
from cftransfer.images import image_path, load_cohort, open_rgb
from cftransfer.protocol import (CONCEPTS, LOCI, MODULES, PRIMARY_ALPHA, conditions_for, question_list,
                                 render_question)
from cftransfer.runpaths import run_dir


def module_forwards(module: str, dataset_id: str, batch: int) -> dict:
    """Forwards a whole module costs at `batch`, split into clean-baseline and steered batches (runner.run_block)."""
    spec = MODULES[module]
    n_rows = {"test": 600, "calibration": 400}[spec.role]
    b = 1 if spec.directions == "clean" else batch
    clean = steered = 0
    for concept, _t in question_list(dataset_id, module):
        conds = conditions_for(module, dataset_id, concept)
        n_clean = sum(1 for c in conds if c[0] == "baseline")
        n_steer = len(conds) - n_clean
        clean += n_rows * (1 if n_clean else 0)
        steered += n_rows * ((n_steer + b - 1) // b)
    return {"batch": b, "clean_batches": clean, "steered_batches": steered}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--locus", default="primary", choices=["primary", "connector"])
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--batches", default="1,8,16,32")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    locus_id = LOCI[a.locus]
    rows = load_cohort(a.dataset, ("preflight",))
    image = open_rgb(image_path(a.dataset, rows[0]))
    ad = get_adapter(a.model_key).load(device_map=a.device_map)
    locus = ad.loci()[locus_id]
    fit = load_fit(a.model_key, a.dataset, locus_id, 0)
    v0 = torch.from_numpy(np.asarray(fit["clinical_vectors"][0:1]))
    q = render_question(a.dataset, CONCEPTS[a.dataset][0], "IY")
    hook = LocusHook(ad.module(locus.module_path), locus_id)
    out = {"model_key": a.model_key, "dataset_id": a.dataset, "locus_id": locus_id, "revision": ad.revision,
           "reps": a.reps, "per_batch": {}, "gpu_models": sorted({torch.cuda.get_device_name(i)
                                                                  for i in range(torch.cuda.device_count())})}
    with hook:
        for B in [int(x) for x in a.batches.split(",")]:
            enc = ad.expand(ad.encode([image], [q]), B)
            lay = ad.layouts(enc, [image] * B)[locus_id]
            timings = {}
            for kind in ("clean", "steered"):
                def one():
                    if kind == "clean":
                        hook.arm(lay)
                    else:
                        hook.arm(lay, v0.repeat(B, 1), torch.full((B,), PRIMARY_ALPHA))
                    ad.forward_last_logits(enc)
                for _ in range(2):
                    one()
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(a.reps):
                    one()
                torch.cuda.synchronize()
                timings[kind] = (time.time() - t0) / a.reps
            out["per_batch"][str(B)] = {
                "seconds_per_forward_clean": round(timings["clean"], 4),
                "seconds_per_forward_steered": round(timings["steered"], 4),
                "seconds_per_condition_steered": round(timings["steered"] / B, 5),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "input_token_count": int(enc["attention_mask"][0].sum()),
                "valid_token_count": lay.counts()[0]}
            print(json.dumps({str(B): out["per_batch"][str(B)]}), flush=True)

    # implied cost of CORE + CALIBRATION on this dataset, at each measured batch
    cal = module_forwards("CALIBRATION", a.dataset, 1)
    t1 = out["per_batch"].get("1")
    for B, rec in out["per_batch"].items():
        core = module_forwards("CORE", a.dataset, int(B))
        core_s = core["clean_batches"] * rec["seconds_per_forward_clean"] + core["steered_batches"] * rec["seconds_per_forward_steered"]
        cal_s = cal["clean_batches"] * (t1 or rec)["seconds_per_forward_clean"]
        rec["cost"] = {"CORE": {**core, "seconds": round(core_s, 1), "gpu_hours": round(core_s / 3600, 2)},
                       "CALIBRATION": {**cal, "seconds": round(cal_s, 1), "gpu_hours": round(cal_s / 3600, 2),
                                       "timed_at_batch_1": t1 is not None},
                       "CORE_plus_CALIBRATION_gpu_hours": round((core_s + cal_s) / 3600, 2)}
    rd = run_dir(a.model_key, a.dataset)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / f"batch_timing.{locus_id}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)
    print("BATCH_TIMING_DONE", flush=True)


if __name__ == "__main__":
    main()
