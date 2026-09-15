"""Generate queue tasks for (model, dataset): preparation, then every module shard, with file-based dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import ANSDIR_N_TRAIN, MODULE_SETTINGS, MODULES, expected_rows
from .runpaths import run_dir
from .worker import QUEUE

SRC = "/rodata/azradonc_dev/m253405/concept-flow/src"
PREP = "/rodata/azradonc_dev/m253405/concept-flow/scripts/mayo/prep_model.sh"
LANE = {"q25-3": "gpu1", "q25-7": "gpu1", "q3-4": "gpu1", "q3-8": "gpu1", "iv35-8": "gpu1", "iv35-14": "gpu1",
        "gemma3-4": "gpu1", "gemma3-12": "gpu1", "medgemma-4": "gpu1", "llama32-11": "gpu1", "llava15-7": "gpu1",
        "llava15-13": "gpu1", "lingshu-7": "gpu1", "llavamed-7": "gpu1",
        "q25-32": "gpu2", "q3-32": "gpu2", "iv35-38": "gpu2", "gemma3-27": "gpu2", "medgemma-27": "gpu2", "lingshu-32": "gpu2",
        "q25-72": "gpu3", "llama32-90": "gpu3"}
BATCH = {"gpu1": 32, "gpu2": 16, "gpu3": 8, "gpu4": 8}
# per-model overrides: Mllama computes all 4 tile slots per image (3 zero tiles), so its vision cost is ~4x
BATCH_MODEL = {"llama32-11": 16, "llama32-90": 4}


def batch_for(model_key: str, lane: str) -> int:
    return BATCH_MODEL.get(model_key, BATCH[lane])
# shards per module for a 600-row test block (row budget relative to CORE); calibration modules are single tasks
SHARDS = {"CORE": 4, "LOCUS": 4, "PROMPT": 7, "DOSE": 2, "REFIT": 1, "CALIBRATION": 1, "LOCUS_CALIBRATION": 1, "ALTDIR": 1,
          "EXTCOMP": 1, "TOKENW": 1, "PRECISION": 1, "ANSDIR": 1}   # PRECISION: one shard per numerics setting (two tasks)
# The addendum modules are not in the default order: they are enqueued explicitly (--modules ...). ALTDIR / EXTCOMP need a
# CPU prep run by hand (python -m cftransfer.altdir / cftransfer.extcomp) whose file the task requires; ANSDIR's prep is a
# GPU task (python -m cftransfer.ansdir) that enqueue emits itself, with the module task requiring its file.
PREP_FILE = {"ALTDIR": "altdir_seed0.npz", "EXTCOMP": "extcomp_seed0.npz", "ANSDIR": "ansdir_seed0.npz"}
PREP_TASK = {"ANSDIR": ["python", "-m", "cftransfer.ansdir", "--n-train", str(ANSDIR_N_TRAIN)]}
MODULE_ORDER = ["CALIBRATION", "CORE", "LOCUS_CALIBRATION", "DOSE", "REFIT", "LOCUS", "PROMPT"]
MODEL_PRIORITY = ["q25-7", "llava15-7", "lingshu-7", "llavamed-7", "q3-8", "iv35-8", "medgemma-4", "q25-3", "q3-4", "iv35-14",
                  "llava15-13", "gemma3-4", "gemma3-12", "llama32-11", "q25-32", "q3-32", "lingshu-32", "medgemma-27",
                  "gemma3-27", "iv35-38", "q25-72", "llama32-90"]


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
        n = SHARDS[mod]
        requires = prep_out + ([str(rd / "fits" / "vis.last" / PREP_FILE[mod])] if mod in PREP_FILE else [])
        if mod in PREP_TASK:                      # the module's own prep as a queue task; the module task waits for its file
            name = f"{prefix}{prio_m:02d}{prio_d}-{mi + 1:02d}-{mod}-prep-{model_key}-{dataset_id}"
            (QUEUE / "pending" / f"{name}.json").write_text(json.dumps({
                "name": name, "lane": lane,
                "cmd": PREP_TASK[mod] + ["--model-key", model_key, "--dataset", dataset_id, "--device-map", dm],
                "requires": prep_out, "produces": [str(rd / "fits" / "vis.last" / PREP_FILE[mod])], "env": env}, indent=1))
            names.append(name)
        for setting in MODULE_SETTINGS.get(mod, (None,)):
            label = mod if setting is None else f"{mod}-{setting}"
            for s in range(n):
                tag = f"{s:03d}of{n:03d}"
                meta_tag = tag if setting is None else f"{setting}-{tag}"
                name = f"{prefix}{prio_m:02d}{prio_d}-{mi + 1:02d}-{label}-{model_key}-{dataset_id}-{tag}"
                cmd = ["python", "-m", "cftransfer.runner", "--model-key", model_key, "--dataset", dataset_id, "--module", mod,
                       "--shard", str(s), "--n-shards", str(n), "--batch", str(batch_for(model_key, lane)), "--device-map", dm]
                if setting is not None:
                    cmd += ["--numerics", setting]
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
