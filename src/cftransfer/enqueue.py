"""Generate queue tasks for (model, dataset): preparation, then every module shard, with file-based dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import (ANSDIR_N_TRAIN, MODULE_SETTINGS, MODULES, PROJSEED_SEEDS, REPLAY_SOURCE, TOWERSWAP_PAIRS,
                       expected_rows)
from .runpaths import fits_dir, run_dir, valid_features_dir
from .worker import QUEUE

SRC = "/rodata/azradonc_dev/m253405/concept-flow/src"
PREP = "/rodata/azradonc_dev/m253405/concept-flow/scripts/mayo/prep_model.sh"
LANE = {"q25-3": "gpu1", "q25-7": "gpu1", "q3-4": "gpu1", "q3-8": "gpu1", "iv35-8": "gpu1", "iv35-14": "gpu1",
        "gemma3-4": "gpu1", "gemma3-12": "gpu1", "medgemma-4": "gpu1", "llama32-11": "gpu1", "llava15-7": "gpu1",
        "llava15-13": "gpu1", "lingshu-7": "gpu1", "llavamed-7": "gpu1", "chexagent-3": "gpu1",
        "huatuo-7": "gpu1", "chexagent-8": "gpu1", "llavarad-7": "gpu1", "maira2-7": "gpu1",
        "q25-32": "gpu2", "q3-32": "gpu2", "iv35-38": "gpu2", "gemma3-27": "gpu2", "medgemma-27": "gpu2", "lingshu-32": "gpu2",
        "q25-72": "gpu3", "llama32-90": "gpu3"}
BATCH = {"gpu1": 32, "gpu2": 16, "gpu3": 8, "gpu4": 8}
# per-model overrides: Mllama computes all 4 tile slots per image (3 zero tiles), so its vision cost is ~4x.
# chexagent-3 is small (3B) but expensive per row: 1,024 visual tokens at 512 px, eager attention (its remote
# code offers no SDPA path) over a 1,077-token prompt, and its forward takes no `logits_to_keep`, so every
# forward materialises (B, 1077, 51,200) float32 logits. Measured on one A100-80GB, steered forwards of the
# runner's replicated-image composition: 0.361 s at batch 4 (8.1 GB peak), 0.682 at 8 (9.7 GB), 1.323 at 16
# (12.9 GB) -- per condition 0.0903 / 0.0852 / 0.0827, i.e. already flat. Per-condition throughput is not what
# decides the batch here: the runner spends one FULL batch forward on each clean baseline condition, so a
# larger batch makes the baselines and the whole CALIBRATION module proportionally more expensive. NIH
# CORE + CALIBRATION costs 12.55 / 12.81 / 14.26 GPU-h at batch 4 / 8 / 16. 8 is the flat part of that curve
# with the smaller memory footprint.
# huatuo-7 is LLaVA-1.5 (576 visual tokens, 605-token prompt) over a Qwen2-7B reader. Measured on one
# A100-80GB (scripts/mayo/batch_timing.py, runs/huatuo-7/nih/batch_timing.vis.last.json), steered forwards of
# the runner's replicated-image composition: 0.0596 s at batch 1 (18.2 GB peak), 0.332 at 8 (18.8), 0.643 at
# 16 (19.5), 1.265 at 32 (21.0) -- per condition 0.0596 / 0.0415 / 0.0402 / 0.0395, i.e. flat from 8 upward.
# NIH CORE + CALIBRATION costs 7.67 / 5.75 / 5.89 / 6.43 GPU-h at batch 1 / 8 / 16 / 32: the runner spends one
# FULL batch forward on every clean baseline condition, so the curve turns back up above 16. 16 is on the flat
# part, 2.4% above the measured minimum, and 12% cheaper than the lane default of 32.
# chexagent-8 is BLIP-2: a 40-layer ViT at 448 px over 1,025 tokens with the remote code's own eager attention,
# a 12-layer Q-Former and Mistral-7B, but only a 30-token prompt and 128 visual embeddings reaching the reader,
# so the cost is almost entirely the tower and it amortises well. Measured on one A100-80GB
# (runs/chexagent-8/nih/batch_timing.vis.last.json): 0.0722 s at batch 1 (17.4 GB peak), 0.162 at 4 (17.6),
# 0.303 at 8 (18.0), 0.568 at 16 (18.8) -- per condition 0.0722 / 0.0406 / 0.0379 / 0.0355. NIH
# CORE + CALIBRATION costs 9.30 / 5.49 / 5.28 / 5.24 GPU-h at batch 1 / 4 / 8 / 16; 16 is the measured minimum
# and the largest composition measured.
# llavarad-7 is the most expensive 7B block of the grid per row: a DINOv2 ViT-B/14 at 518 px emits 1,369
# consumed visual tokens (2.4x LLaVA-1.5's 576), so the language model reads a 1,430-token prompt on every
# condition. Measured on one A100-80GB (runs/llavarad-7/nih/batch_timing.vis.last.json): 0.162 s at batch 1
# (14.4 GB peak), 0.582 at 4 (14.9), 1.158 at 8 (15.5), 2.175 at 16 (16.7) -- per condition 0.162 / 0.146 /
# 0.145 / 0.136. NIH CORE + CALIBRATION costs 20.85 / 19.49 / 19.95 / 19.88 GPU-h at batch 1 / 4 / 8 / 16,
# i.e. flat within 2% from 4 upward; 8 sits in that band with a 15.5 GB footprint.
# maira2-7 has llavarad-7's shape (DINOv2 ViT-B/14 at 518 px, 1,369 consumed tokens, a 1,421-token prompt) with
# a deeper projector. Measured on one A100-80GB (runs/maira2-7/nih/batch_timing.vis.last.json): 0.158 s at batch
# 1 (14.5 GB peak), 0.543 at 4 (14.9), 1.061 at 8 (15.5), 2.115 at 16 (16.8) -- per condition 0.158 / 0.136 /
# 0.133 / 0.132. NIH CORE + CALIBRATION costs 20.32 / 18.18 / 18.36 / 19.32 GPU-h at batch 1 / 4 / 8 / 16, flat
# within 1% from 4 to 8. NOTE the block is NOT eligible for the yes/no grid: all six templates failed preflight
# check E (see runs/maira2-7/nih/template_eligibility.json), so every module would close with no outcomes.
BATCH_MODEL = {"llama32-11": 16, "llama32-90": 4, "chexagent-3": 8, "huatuo-7": 16, "chexagent-8": 16,
               "llavarad-7": 8, "maira2-7": 8}


def batch_for(model_key: str, lane: str) -> int:
    return BATCH_MODEL.get(model_key, BATCH[lane])
# shards per module for a 600-row test block (row budget relative to CORE); calibration modules are single tasks
SHARDS = {"CORE": 4, "LOCUS": 4, "PROMPT": 7, "DOSE": 2, "REFIT": 1, "CALIBRATION": 1, "LOCUS_CALIBRATION": 1, "ALTDIR": 1,
          "EXTCOMP": 1, "TOKENW": 1, "PRECISION": 1, "ANSDIR": 1, "ALTDIRD": 1, "ATTR": 1, "ANSDIRT": 2,
          "VALID": 1, "ATTRRAND": 2, "VALIDFIT": 1, "ATTRQ": 1,
          "PROJSEED": 1, "TOWERSWAP": 1, "TOWERSWAPD": 1, "REPLAY": 1, "SEMEND": 4}   # PRECISION: one shard per setting; VALID: 152,400 outcomes on 200 rows, one third of a CORE shard
# budget; ATTRRAND: 214,200 outcomes, two shards keep a shard near CORE's 114,300; VALIDFIT: 21,600 outcomes, one shard;
# ATTRQ: 3,600 clean forwards on the calibration rows (3 attributes x 3 phrasings x 400), one shard, no prep of its own --
# it writes nothing, so it needs no directions beyond the standard fit the prep already produces;
# PROJSEED: one shard PER PROJECTION SEED (MODULE_SEED_TASKS), i.e. 453,600 outcomes per task, and the module is meant
# for a handful of representative blocks rather than the whole grid.
# TOWERSWAP / REPLAY: 152,400 outcomes on 200 rows, one shard each (VALID's budget); SEMEND: 475,200 outcomes over 600
# rows, four shards keep a shard at 118,800, near CORE's 114,300.
# Modules whose fit seeds are independent grids and are therefore emitted as one task per seed (runner --fit-seed).
MODULE_SEED_TASKS = {"PROJSEED": PROJSEED_SEEDS}
# The addendum modules are not in the default order: they are enqueued explicitly (--modules ...). ALTDIR / EXTCOMP need a
# CPU prep run by hand (python -m cftransfer.altdir / cftransfer.extcomp) whose file the task requires; ANSDIR's prep is a
# GPU task (python -m cftransfer.ansdir) that enqueue emits itself, with the module task requiring its file.
# One value per module: the prep file(s) under fits/vis.last/ the module task must wait for (a tuple when a module has
# more than one, e.g. PROJSEED's file per projection seed).
PREP_FILE = {"ALTDIR": "altdir_seed0.npz", "EXTCOMP": "extcomp_seed0.npz", "ANSDIR": "ansdir_seed0.npz",
             "ALTDIRD": "altdir_seed0.npz", "ATTR": "attr_seed0.npz", "ANSDIRT": "ansdir_seed0.npz",
             "ATTRRAND": "attr_seed0.npz", "VALIDFIT": "validfit_seed0.npz",
             "PROJSEED": tuple(f"projseed_seed{k}.npz" for k in PROJSEED_SEEDS),
             "SEMEND": "semend_seed0.npz"}
PREP_TASK = {"ANSDIR": ["python", "-m", "cftransfer.ansdir", "--n-train", str(ANSDIR_N_TRAIN)],
             "ANSDIRT": ["python", "-m", "cftransfer.ansdir", "--n-train", str(ANSDIR_N_TRAIN)],
             "VALIDFIT": ["python", "-m", "cftransfer.validfit"], "PROJSEED": ["python", "-m", "cftransfer.projseed"],
             "SEMEND": ["python", "-m", "cftransfer.semend"]}
# Preps that never touch a GPU: the task is emitted without --device-map and may run on any lane.
PREP_TASK_CPU = {"VALIDFIT", "PROJSEED", "SEMEND"}
# Modules that read a file of ANOTHER block: TOWERSWAP writes the PARTNER checkpoint's tower into this block's reader
# and therefore writes the PARTNER's seed-0 directions; REPLAY writes the shared-tower group's SOURCE block's stored
# consumed-block tensors with that block's seed-0 directions. Both dependencies are cross-block, so they cannot live in
# PREP_FILE (which is always this block's fits/vis.last/).
MODULE_DATA_FN = {
    "TOWERSWAP": lambda mk, ds: [str(fits_dir(TOWERSWAP_PAIRS[mk], ds, "vis.last") / "seed0.npz")],
    "TOWERSWAPD": lambda mk, ds: [str(fits_dir(mk, ds, "vis.last") / "seed0.npz")],
    "REPLAY": lambda mk, ds: [str(run_dir(REPLAY_SOURCE[mk], ds) / "replay" / "vis.last.npz"),
                              str(fits_dir(REPLAY_SOURCE[mk], ds, "vis.last") / "seed0.npz")]}
# Models a module can be enqueued for at all (the pair / group tables of the protocol).
MODULE_MODELS = {"TOWERSWAP": set(TOWERSWAP_PAIRS), "TOWERSWAPD": set(TOWERSWAP_PAIRS), "REPLAY": set(REPLAY_SOURCE)}
# FGOBJ (fine-grained COCO objects): CORE's grid on six new questions, 457,200 outcomes over 600 rows, so CORE's four
# shards. Its CPU prep (python -m cftransfer.fgobj) fits the six probes and draws the family's own 119 random
# directions; the module task waits for fgobj_seed0.npz. FGOBJ_CALIBRATION is 2,400 clean forwards on the calibration
# rows and needs no directions at all, so it has no prep file and can run before the prep.
# The six concepts must already be appended to the COCO label manifest (python -m cftransfer.fgobj --build-labels,
# once per data root) -- the prep fails with that command when they are not.
SHARDS["FGOBJ"] = 4
SHARDS["FGOBJ_CALIBRATION"] = 1
PREP_FILE["FGOBJ"] = "fgobj_seed0.npz"
PREP_TASK["FGOBJ"] = ["python", "-m", "cftransfer.fgobj"]
PREP_TASK_CPU.add("FGOBJ")


def prep_files(mod: str) -> list[str]:
    """The prep file names of a module (PREP_FILE takes a string or a tuple)."""
    v = PREP_FILE.get(mod)
    return [] if v is None else ([v] if isinstance(v, str) else list(v))
# VALID scores the images of the `valid` role (scripts/mayo/extract_chexpert_valid.py writes the marker) and needs no prep of
# its own (the seed-0 fit is CORE's). enqueue also emits a GPU task extracting the valid rows' features into
# features/valid/ (python -m cftransfer.features --roles valid) so analysis.valid can grade probe readability against the
# radiologist labels; the module task does not wait for it (the analysis runs without it, readability "not available").
VALID_IMAGES = "/rodata/azradonc_dev/m253405/cf-transfer/data/chexpert/images/valid/.complete"
MODULE_DATA = {"VALID": [VALID_IMAGES]}
# VALIDFIT refits the six directions on the radiologist-labelled valid rows, so its CPU prep needs the valid-role
# features the VALID feature task writes (features/valid/<locus>.npz); the module task itself waits only for the prep file.
# SEMEND's prep reads the block's own CORE write matrix (to freeze each question's strongest competitor) and needs the
# answer directions, because a_q is one of its 25 conditions; so it waits for the merged CORE parquet and for ANSDIR.
PREP_DATA = {"VALIDFIT": lambda mk, ds: [str(valid_features_dir(mk, ds) / "vis.last.npz")],
             "SEMEND": lambda mk, ds: [str(fits_dir(mk, ds, "vis.last") / "ansdir_seed0.npz"),
                                       str(run_dir(mk, ds) / "outcomes" / "CORE.parquet")]}
MODULE_ORDER = ["CALIBRATION", "CORE", "LOCUS_CALIBRATION", "DOSE", "REFIT", "LOCUS", "PROMPT"]
MODEL_PRIORITY = ["q25-7", "llava15-7", "lingshu-7", "llavamed-7", "q3-8", "iv35-8", "medgemma-4", "q25-3", "q3-4", "iv35-14",
                  "llava15-13", "gemma3-4", "gemma3-12", "llama32-11", "q25-32", "q3-32", "lingshu-32", "medgemma-27",
                  "gemma3-27", "iv35-38", "q25-72", "llama32-90",
                  "chexagent-3", "huatuo-7", "chexagent-8", "llavarad-7", "maira2-7"]   # addendum checkpoints, appended so no existing block's task names change


def enqueue(model_key: str, dataset_id: str, modules: list[str] | None = None, prep: bool = True, prefix: str = "") -> list[str]:
    rd = run_dir(model_key, dataset_id)
    lane = LANE[model_key]
    dm = "cuda:0" if lane == "gpu1" else "auto"
    prio_m = MODEL_PRIORITY.index(model_key) if model_key in MODEL_PRIORITY else 99
    prio_d = {"nih": 0, "coco": 1, "chexpert": 2}[dataset_id]
    names = []
    env = {"HF_HOME": "/rodata/azradonc_dev/m253405/cache", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
    prep_out = [str(rd / "preflight.vis.last.json"), str(rd / "preflight.connector.json"), str(rd / "fits" / "connector" / "seed2.npz")]
    data_ready = ["/rodata/azradonc_dev/m253405/cf-transfer/data/chexpert/images/.complete"] if dataset_id == "chexpert" else []
    if prep:
        name = f"{prefix}{prio_m:02d}{prio_d}-00-prep-{model_key}-{dataset_id}"
        (QUEUE / "pending" / f"{name}.json").write_text(json.dumps({
            "name": name, "lane": lane, "cmd": ["bash", PREP, model_key, dataset_id, str(batch_for(model_key, lane)), dm],
            "requires": data_ready, "produces": prep_out, "env": env}, indent=1))
        names.append(name)
    for mi, mod in enumerate(modules or MODULE_ORDER):
        spec = MODULES[mod]
        if dataset_id not in spec.datasets or expected_rows(mod, dataset_id) == 0:
            continue
        if mod in MODULE_MODELS and model_key not in MODULE_MODELS[mod]:
            raise SystemExit(f"{mod} is only defined for {sorted(MODULE_MODELS[mod])}; {model_key} is not one of them")
        n = SHARDS[mod]
        requires = (prep_out + [str(rd / "fits" / "vis.last" / f) for f in prep_files(mod)] + MODULE_DATA.get(mod, [])
                    + (MODULE_DATA_FN[mod](model_key, dataset_id) if mod in MODULE_DATA_FN else []))
        if mod == "REPLAY":
            # one tower pass per SOURCE block and dataset, shared by every reader of the group: the task name is the
            # same whichever reader enqueues it, so the second enqueue overwrites the first pending file
            src = REPLAY_SOURCE[model_key]
            sprio = MODEL_PRIORITY.index(src) if src in MODEL_PRIORITY else 99
            name = f"{prefix}{sprio:02d}{prio_d}-00-REPLAY-tensors-{src}-{dataset_id}"
            slane = LANE[src]
            (QUEUE / "pending" / f"{name}.json").write_text(json.dumps({
                "name": name, "lane": slane,
                "cmd": ["python", "-m", "cftransfer.replay", "--model-key", src, "--dataset", dataset_id,
                        "--batch-size", str(batch_for(src, slane)),
                        "--device-map", "cuda:0" if slane == "gpu1" else "auto"],
                "requires": data_ready + [str(run_dir(src, dataset_id) / "preflight.vis.last.json")],
                "produces": [str(run_dir(src, dataset_id) / "replay" / "vis.last.npz")], "env": env}, indent=1))
            if name not in names:
                names.append(name)
        if mod == "VALID":
            name = f"{prefix}{prio_m:02d}{prio_d}-{mi + 1:02d}-{mod}-features-{model_key}-{dataset_id}"
            vf = valid_features_dir(model_key, dataset_id)
            (QUEUE / "pending" / f"{name}.json").write_text(json.dumps({
                "name": name, "lane": lane,
                "cmd": ["python", "-m", "cftransfer.features", "--model-key", model_key, "--dataset", dataset_id, "--roles", "valid",
                        "--batch-size", str(batch_for(model_key, lane)), "--device-map", dm, "--out-dir", str(vf)],
                "requires": requires, "produces": [str(vf / "vis.last.npz"), str(vf / "connector.npz")], "env": env}, indent=1))
            names.append(name)
        if mod in PREP_TASK:                      # the module's own prep as a queue task; the module task waits for its file
            name = f"{prefix}{prio_m:02d}{prio_d}-{mi + 1:02d}-{mod}-prep-{model_key}-{dataset_id}"
            cmd = PREP_TASK[mod] + ["--model-key", model_key, "--dataset", dataset_id]
            if mod not in PREP_TASK_CPU:
                cmd += ["--device-map", dm]
            (QUEUE / "pending" / f"{name}.json").write_text(json.dumps({
                "name": name, "lane": lane, "cmd": cmd,
                "requires": prep_out + (PREP_DATA[mod](model_key, dataset_id) if mod in PREP_DATA else []),
                "produces": [str(rd / "fits" / "vis.last" / f) for f in prep_files(mod)], "env": env}, indent=1))
            names.append(name)
        # per-setting tasks (PRECISION: numerics) and per-seed tasks (PROJSEED: one projection seed each)
        for setting in MODULE_SETTINGS.get(mod, (None,)):
            for seed in MODULE_SEED_TASKS.get(mod, (None,)):
                label = mod if setting is None else f"{mod}-{setting}"
                label = label if seed is None else f"{label}-s{seed}"
                for s in range(n):
                    tag = f"{s:03d}of{n:03d}"
                    meta_tag = tag if setting is None else f"{setting}-{tag}"
                    meta_tag = meta_tag if seed is None else f"s{seed}-{meta_tag}"
                    name = f"{prefix}{prio_m:02d}{prio_d}-{mi + 1:02d}-{label}-{model_key}-{dataset_id}-{tag}"
                    cmd = ["python", "-m", "cftransfer.runner", "--model-key", model_key, "--dataset", dataset_id, "--module", mod,
                           "--shard", str(s), "--n-shards", str(n), "--batch", str(batch_for(model_key, lane)), "--device-map", dm]
                    if setting is not None:
                        cmd += ["--numerics", setting]
                    if seed is not None:
                        cmd += ["--fit-seed", str(seed)]
                    (QUEUE / "pending" / f"{name}.json").write_text(json.dumps({
                        "name": name, "lane": lane, "cmd": cmd,
                        "requires": requires, "produces": [str(rd / "outcomes" / mod / f"meta-{meta_tag}-*.json")], "env": env}, indent=1))
                    names.append(name)
    return names


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma list or 'all'")
    ap.add_argument("--datasets", default="nih,coco")
    ap.add_argument("--modules", default=None)
    ap.add_argument("--no-prep", action="store_true")
    ap.add_argument("--prefix", default="", help="task-name prefix; names sort as priorities, so '3' runs after every current task")
    a = ap.parse_args()
    models = MODEL_PRIORITY if a.models == "all" else a.models.split(",")
    total = 0
    for m in models:
        for d in a.datasets.split(","):
            total += len(enqueue(m, d, a.modules.split(",") if a.modules else None, prep=not a.no_prep, prefix=a.prefix))
    print("enqueued", total, "tasks; pending now", len(list((QUEUE / "pending").glob("*.json"))))
