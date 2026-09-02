"""The control that decides whether a probe is reading the model or reading the labels.

A probe fitted on 18,212 labelled radiographs will find something in almost any high-dimensional image
representation, including one produced by a network that was never trained. Without that floor, "a probe
reads pleural effusion from the activations at AUROC 0.80" does not distinguish

    the trained model represents the finding          from      a linear map plus 18k labels can separate
                                                                these images at all

and the whole decodable-versus-used comparison inherits the ambiguity. The survey behind this project
found a randomly initialised encoder floor in 5 of 121 applicable studies.

This builds the same architecture from config with freshly initialised weights, extracts at the same loci
with the same pooling, and writes shards in the same format, so src/probe.py runs on it unchanged.

    python src/extract_random_init.py --arch llavamed7b --gpus 4,5 --out runs/rndact_llavamed7b
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Process

try:
    from .gpu_env import bind_gpu
except ImportError:
    from gpu_env import bind_gpu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def worker(shard_id, gpu, rows, args, out_dir):
    bind_gpu(gpu)
    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    from loci import ActivationCache, find_image_token_id, loci_for
    from registry import REGISTRY

    arch = REGISTRY[args.arch]
    dev = "cuda:0"
    cfg = AutoConfig.from_pretrained(arch.hf_id)
    proc = AutoProcessor.from_pretrained(arch.hf_id, **arch.processor_kwargs)
    # from_config, not from_pretrained: identical shapes and identical hook paths, untrained weights.
    torch.manual_seed(args.seed + shard_id * 0)      # same weights on every shard, or the shards disagree
    model = AutoModelForImageTextToText.from_config(cfg).to(dtype=torch.bfloat16, device=dev).eval()
    loci = loci_for(arch)
    img_tok = arch.image_token_id or find_image_token_id(proc, cfg)

    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    store, ids, failed = {}, [], []
    t1 = time.time()
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i + args.batch_size]
        imgs, keep = [], []
        for r in chunk:
            try:
                imgs.append(Image.open(r["image_path"]).convert("RGB")); keep.append(r)
            except Exception as e:
                failed.append({"row_id": r["row_id"], "error": f"{type(e).__name__}: {e}"})
        if not imgs:
            continue
        try:
            inputs = proc(text=[text] * len(imgs), images=imgs, return_tensors="pt",
                          padding=True).to(dev)
            cache = ActivationCache(model, loci, image_token_id=img_tok)
            cache.set_visual_mask(inputs["input_ids"])
            cache.batch_size = len(imgs)
            with cache, torch.inference_mode():
                model(**inputs)
            acts = cache.pop()
        except Exception as e:
            failed.extend({"row_id": r["row_id"], "error": f"{type(e).__name__}: {e}"} for r in keep)
            continue
        for name, a in acts.items():
            store.setdefault(name, []).append(a.to(torch.float16).numpy())
        ids.extend(r["row_id"] for r in keep)
        if (i // args.batch_size) % 25 == 0:
            print(f"[rnd shard{shard_id} gpu{gpu}] {i + len(chunk)}/{len(rows)} "
                  f"{(i + len(chunk)) / max(time.time() - t1, 1e-9):.1f} img/s", flush=True)

    out = {k: np.concatenate(v, axis=0) for k, v in store.items()}
    np.savez_compressed(os.path.join(out_dir, f"shard{shard_id}.npz"),
                        row_id=np.array(ids, dtype=object), **out)
    json.dump({"shard": shard_id, "gpu": gpu, "n_in": len(rows), "n_out": len(ids), "failed": failed,
               "extract_s": round(time.time() - t1, 1)},
              open(os.path.join(out_dir, f"shard{shard_id}.json"), "w"), indent=1)
    print(f"[rnd shard{shard_id}] done {len(ids)}/{len(rows)}, {len(failed)} failed", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--arch", required=True)
    ap.add_argument("--gpus", default="4,5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt",
                    default="Is there a pleural effusion in this chest radiograph? Answer yes or no.")
    args = ap.parse_args()

    from registry import REGISTRY
    rows = list(csv.DictReader(open(args.manifest)))
    if args.limit:
        rows = rows[:args.limit]
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    os.makedirs(args.out, exist_ok=True)
    json.dump({"arch": args.arch, "model": REGISTRY[args.arch].hf_id + " [RANDOMLY INITIALISED]",
               "prompt": args.prompt, "n_rows": len(rows), "gpus": gpus, "seed": args.seed,
               "pooling": "mean over selected positions, identical to src/extract.py",
               "purpose": "floor for the probe: what a linear map plus the training labels can extract "
                          "from an untrained network of the same shape",
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(os.path.join(args.out, "meta.json"), "w"), indent=1)

    shards = [rows[k::len(gpus)] for k in range(len(gpus))]
    print(f"{args.arch} RANDOM INIT: {len(rows)} rows over {len(gpus)} gpus -> {[len(s) for s in shards]}")
    t0 = time.time()
    ps = [Process(target=worker, args=(k, gpus[k], shards[k], args, args.out)) for k in range(len(gpus))]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print(f"all shards done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
