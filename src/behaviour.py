"""D0: what the representation carries, against what the answer carries.

Every other experiment here compares directions or loci. This one compares two numbers that the field
routinely conflates, on the same images and the same labels:

    probe AUROC        how well concept c can be decoded from the activations
    behavioural AUROC  how well the model's OWN ANSWER to "is there a c?" separates the same labels

The first is what a probing paper reports. The second is what the model does. The gap between them is the
paper's thesis reduced to a single number per model and concept, and it needs no intervention, no direction
and no assumption about linearity to compute.

The observation that forced this experiment: LLaVA-Med answers "yes, there is a pleural effusion" with
median probability 0.967 on label-normal chest radiographs, and not one of 64 screened test radiographs
produced a probability between 0.05 and 0.95. Its answer is close to constant. Whatever its activations
encode about effusion, its behaviour cannot be using much of it.

    python src/behaviour.py --arch lingshu7b --gpu 0 --out runs/behav_lingshu7b
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

try:
    from .gpu_env import bind_gpu
except ImportError:
    from gpu_env import bind_gpu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from intervene import yes_margin_batch, yes_prob_batch
from loci import find_image_token_id
from registry import REGISTRY

CONCEPTS = ["Effusion", "Cardiomegaly", "Pneumothorax", "Atelectasis", "Consolidation",
            "Edema", "Infiltration", "Mass", "Nodule"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--arch", required=True)
    ap.add_argument("--concepts", default=",".join(CONCEPTS))
    ap.add_argument("--prompt", default="Is there a {c} in this chest radiograph? Answer yes or no.")
    ap.add_argument("--n-images", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    bind_gpu(args.gpu)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    dev = "cuda:0"

    man = [r for r in csv.DictReader(open(args.manifest)) if r["split"] == "test"]
    idx = rng.permutation(len(man))[:args.n_images]
    rows = [man[i] for i in idx]
    print(f"{len(man)} test rows, scoring {len(rows)}")

    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    arch = REGISTRY[args.arch]
    cfg = AutoConfig.from_pretrained(arch.hf_id)
    proc = AutoProcessor.from_pretrained(arch.hf_id, **arch.processor_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(arch.hf_id, dtype=torch.bfloat16,
                                                        device_map=dev).eval()
    tok = getattr(proc, "tokenizer", proc)
    _ = arch.image_token_id or find_image_token_id(proc, cfg)

    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    out, per_image = [], {}
    for c in concepts:
        if c not in rows[0]:
            print(f"skip {c}: not a manifest column"); continue
        y = np.array([int(r[c]) for r in rows])
        if len(np.unique(y)) < 2:
            print(f"skip {c}: one class only in this sample"); continue
        text = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"},
                                          {"type": "text", "text": args.prompt.format(c=c.lower())}]}],
            tokenize=False, add_generation_prompt=True)
        p, m = [], []
        t0 = time.time()
        for b0 in range(0, len(rows), args.batch_size):
            chunk = rows[b0:b0 + args.batch_size]
            imgs = [Image.open(r["image_path"]).convert("RGB") for r in chunk]
            inp = proc(text=[text] * len(imgs), images=imgs, return_tensors="pt", padding=True).to(dev)
            with torch.inference_mode():
                lg = model(**inp).logits
            p.extend(yes_prob_batch(lg, tok))
            m.extend(yes_margin_batch(lg, tok))
        p, m = np.array(p), np.array(m)
        per_image[c] = {"p_yes": p.tolist(), "y": y.tolist()}
        auc = float(roc_auc_score(y, p))
        # How often the answer is effectively pinned. A model whose answer never crosses the middle is
        # not weakly informative, it is constant, and no intervention can be measured in that coordinate.
        pinned = float(np.mean((p > 0.95) | (p < 0.05)))
        out.append({
            "arch": args.arch, "domain": arch.domain, "concept": c, "n": len(rows),
            "prevalence": round(float(y.mean()), 4),
            "behavioural_auroc": round(auc, 4),
            "mean_p_yes": round(float(p.mean()), 4), "sd_p_yes": round(float(p.std()), 4),
            "median_p_yes": round(float(np.median(p)), 4),
            "frac_pinned": round(pinned, 4),
            "mean_margin": round(float(m.mean()), 4), "sd_margin": round(float(m.std()), 4),
            "yes_rate": round(float((p > 0.5).mean()), 4),
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"  {c:15s} behavioural AUROC {auc:.4f}  P(yes) {p.mean():.3f}+-{p.std():.3f}  "
              f"pinned {pinned*100:.0f}%  yes-rate {(p > 0.5).mean()*100:.0f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(args.out, "behaviour_auroc.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    json.dump({"arch": args.arch, "n_images": len(rows), "concepts": concepts,
               "prompt": args.prompt, "row_ids": [r["row_id"] for r in rows]},
              open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(args.out, "per_image.npz"),
                        **{f"{c}_p": np.array(v["p_yes"]) for c, v in per_image.items()},
                        **{f"{c}_y": np.array(v["y"]) for c, v in per_image.items()})
    with open(os.path.join(args.out, "DONE"), "w") as fh:
        fh.write(json.dumps({"rows": len(out), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    print(f"\nwrote {len(out)} rows to {args.out}/behaviour_auroc.csv")


if __name__ == "__main__":
    main()
