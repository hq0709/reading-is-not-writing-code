"""Multi-GPU activation extraction. One model replica per GPU, images sharded across them.

This is the workhorse: every experiment in the paper reads the arrays it writes. A 7B model in bf16 is
about 16 GB, so a replica fits on one card with room for activations, and data parallelism is the right
axis. Four cards give close to four times the throughput because the work is a single forward pass per
image with no cross-device communication.

    python src/extract.py --manifest data/manifest.csv --model Qwen/Qwen2.5-VL-7B-Instruct \
        --gpus 4,5,6 --out runs/act_qwen7b --batch-size 8

Output, one file per shard, plus a merged index:

    runs/<name>/shard{k}.npz     acts[locus] -> (n, d) float16, plus row ids
    runs/<name>/meta.json        model, loci, pooling, prompt, git sha, timings

The manifest is a CSV with at least `image_path` and `row_id`; every other column is carried through to
the metadata so probes can join labels without re-reading the source dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from multiprocessing import Process

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read_manifest(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"manifest {path} is empty")
    for need in ("image_path", "row_id"):
        if need not in rows[0]:
            raise SystemExit(f"manifest needs a {need!r} column, has {list(rows[0])}")
    return rows


def select_loci(loci, requested):
    """Select named loci without changing their registered order."""
    names = [name.strip() for name in requested.split(",") if name.strip()]
    if not names:
        return loci
    known = {locus.name for locus in loci}
    missing = sorted(set(names) - known)
    if missing:
        raise ValueError(f"unknown loci: {missing}; known: {sorted(known)}")
    wanted = set(names)
    return [locus for locus in loci if locus.name in wanted]


def worker(shard_id, gpu, rows, args, out_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    from loci import ActivationCache, find_image_token_id, loci_for
    from registry import REGISTRY

    arch = REGISTRY[args.arch]
    dev = "cuda:0"
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(arch.hf_id)
    proc = AutoProcessor.from_pretrained(arch.hf_id, **arch.processor_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(arch.hf_id, dtype=torch.bfloat16, device_map=dev)
    model.eval()
    load_s = time.time() - t0

    loci = select_loci(loci_for(arch), args.loci)
    img_tok = arch.image_token_id or find_image_token_id(proc, cfg)

    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    store: dict[str, list] = {}
    ids, failed = [], []
    t1 = time.time()
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i + args.batch_size]
        imgs, keep = [], []
        for r in chunk:
            try:
                imgs.append(Image.open(r["image_path"]).convert("RGB"))
                keep.append(r)
            except Exception as e:
                failed.append({"row_id": r["row_id"], "error": f"{type(e).__name__}: {e}"})
        if not imgs:
            continue
        try:
            inputs = proc(text=[text] * len(imgs), images=imgs, return_tensors="pt",
                          padding=True).to(dev)
            cache = ActivationCache(model, loci, image_token_id=img_tok)
            cache.set_visual_mask(inputs["input_ids"])
            with cache, torch.inference_mode():
                model(**inputs)
            acts = cache.pop()
        except Exception as e:
            failed.extend({"row_id": r["row_id"], "error": f"{type(e).__name__}: {e}"} for r in keep)
            continue
        for name, a in acts.items():
            store.setdefault(name, []).append(a.to(torch.float16).numpy())
        ids.extend(r["row_id"] for r in keep)
        if (i // args.batch_size) % 20 == 0:
            done = i + len(chunk)
            rate = done / max(time.time() - t1, 1e-6)
            print(f"[shard{shard_id} gpu{gpu}] {done}/{len(rows)} {rate:.1f} img/s", flush=True)

    out = {k: np.concatenate(v, axis=0) for k, v in store.items()}
    np.savez_compressed(os.path.join(out_dir, f"shard{shard_id}.npz"),
                        row_id=np.array(ids, dtype=object), **out)
    json.dump({"shard": shard_id, "gpu": gpu, "n_in": len(rows), "n_out": len(ids),
               "failed": failed, "load_s": round(load_s, 1),
               "extract_s": round(time.time() - t1, 1),
               "img_per_s": round(len(ids) / max(time.time() - t1, 1e-6), 2)},
              open(os.path.join(out_dir, f"shard{shard_id}.json"), "w"), indent=1)
    print(f"[shard{shard_id} gpu{gpu}] done {len(ids)}/{len(rows)} in "
          f"{time.time() - t1:.0f}s, {len(failed)} failed", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--arch", required=True, help="key in src/registry.py")
    ap.add_argument("--gpus", default="4,5,6", help="comma separated, one replica each")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--loci", default="",
                    help="optional comma-separated registered locus names; empty captures every locus")
    ap.add_argument("--limit", type=int, default=0, help="cap rows, for a quick pass")
    ap.add_argument("--prompt", default="Is there a pleural effusion in this chest radiograph? Answer yes or no.")
    args = ap.parse_args()

    from loci import loci_for
    from registry import REGISTRY
    if args.arch not in REGISTRY:
        raise SystemExit(f"unknown arch {args.arch!r}. known: {sorted(REGISTRY)}")
    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    os.makedirs(args.out, exist_ok=True)

    sha = os.environ.get("SOURCE_COMMIT", "")
    if not sha:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            )
            sha = result.stdout.strip() if result.returncode == 0 else "unknown"
        except OSError:
            sha = "unknown"
    selected = select_loci(loci_for(REGISTRY[args.arch]), args.loci)
    json.dump({"arch": args.arch, "model": REGISTRY[args.arch].hf_id, "prompt": args.prompt, "n_rows": len(rows), "gpus": gpus,
               "batch_size": args.batch_size, "git_sha": sha,
               "loci": [locus.name for locus in selected],
               "pooling": "mean over selected positions, fixed in src/loci.py",
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(os.path.join(args.out, "meta.json"), "w"), indent=1)

    shards = [rows[k::len(gpus)] for k in range(len(gpus))]
    print(f"{args.arch} ({REGISTRY[args.arch].hf_id}): {len(rows)} rows over {len(gpus)} gpus "
          f"-> {[len(s) for s in shards]}")

    t0 = time.time()
    procs = [Process(target=worker, args=(k, gpus[k], shards[k], args, args.out))
             for k in range(len(gpus))]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    failed_workers = [(k, p.exitcode) for k, p in enumerate(procs) if p.exitcode != 0]
    if failed_workers:
        raise SystemExit(f"activation workers failed: {failed_workers}")
    print(f"all shards done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
