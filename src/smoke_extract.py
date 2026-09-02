"""End-to-end check of the activation instrument, on real GPU, on a real model.

Loads Qwen2.5-VL-7B, registers hooks at every locus, runs a forward pass on a synthetic radiograph-like
image, and reports the shape and norm at each locus. If this prints a full locus table, the instrument
that the whole paper depends on works.

    python src/smoke_extract.py --gpu 6
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image

from loci import ActivationCache, find_image_token_id, qwen_loci

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def synthetic_cxr(size=448, with_finding=False, seed=0):
    """A grey field with a lung-like structure. Stands in for a radiograph until the real data lands.
    `with_finding` adds a bright blob in the lower right, which is what a planted effusion looks like."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    img = 40 + 30 * np.exp(-(((x - size * 0.5) / (size * 0.45)) ** 2 + ((y - size * 0.5) / (size * 0.55)) ** 2))
    for cx in (size * 0.32, size * 0.68):
        img -= 22 * np.exp(-(((x - cx) / (size * 0.16)) ** 2 + ((y - size * 0.52) / (size * 0.30)) ** 2))
    img += 18 * np.exp(-(((x - size * 0.5) / (size * 0.09)) ** 2 + ((y - size * 0.62) / (size * 0.22)) ** 2))
    if with_finding:
        img += 55 * np.exp(-(((x - size * 0.70) / (size * 0.10)) ** 2 + ((y - size * 0.78) / (size * 0.07)) ** 2))
    img += rng.normal(0, 3, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--prompt", default="Is there a pleural effusion in this chest radiograph? Answer yes or no.")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    dev = "cuda:0"

    from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText

    t0 = time.time()
    cfg = AutoConfig.from_pretrained(args.model)
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dev)
    model.eval()
    print(f"loaded {args.model} in {time.time() - t0:.1f}s  "
          f"({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")

    tcfg = getattr(cfg, "text_config", cfg)
    vcfg = getattr(cfg, "vision_config", None)
    n_llm = tcfg.num_hidden_layers
    n_vis = getattr(vcfg, "depth", None) or getattr(vcfg, "num_hidden_layers", 32)
    print(f"llm layers {n_llm} hidden {tcfg.hidden_size} | vision blocks {n_vis}")

    img_tok = find_image_token_id(proc, cfg)
    print(f"image token id: {img_tok}")

    loci = qwen_loci(n_vis, n_llm)
    print(f"{len(loci)} loci defined\n")

    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    for tag, finding in (("normal", False), ("planted", True)):
        img = synthetic_cxr(with_finding=finding, seed=0)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(dev)
        cache = ActivationCache(model, loci, image_token_id=img_tok)
        cache.set_visual_mask(inputs["input_ids"])
        t1 = time.time()
        with cache, torch.inference_mode():
            out = model(**inputs)
        acts = cache.pop()
        dt = time.time() - t1
        nvis = int((inputs["input_ids"] == img_tok).sum()) if img_tok is not None else -1
        print(f"[{tag}] forward {dt:.2f}s | seq {inputs['input_ids'].shape[1]} | visual tokens {nvis} "
              f"| captured {len(acts)}/{len(loci)} loci")
        if tag == "normal":
            for lo in loci:
                if lo.name in acts and (lo.name.startswith("vis.") or lo.name == "connector"
                                        or lo.name in ("llm.L0.vis", "llm.L0.ans",
                                                       f"llm.L{n_llm // 2}.vis", f"llm.L{n_llm - 1}.ans")):
                    a = acts[lo.name]
                    print(f"    {lo.name:16s} {str(tuple(a.shape)):12s} norm {a.norm():8.2f}  [{lo.stage}]")
        globals()[f"acts_{tag}"] = acts

    a, b = globals()["acts_normal"], globals()["acts_planted"]
    common = [lo.name for lo in loci if lo.name in a and lo.name in b]
    print(f"\nplanted-vs-normal cosine per locus (lower means the planted finding changed it more):")
    for n in common:
        if n.startswith("vis.block") and n not in ("vis.last",):
            continue
        cos = torch.nn.functional.cosine_similarity(a[n], b[n], dim=-1).mean().item()
        if n in ("vis.last", "connector") or n.endswith((".L0.vis", f".L{n_llm // 2}.vis", f".L{n_llm - 1}.ans")):
            print(f"    {n:16s} cos {cos:.4f}")

    missing = [lo.name for lo in loci if lo.name not in a]
    print(f"\nmissing loci: {missing if missing else 'none'}")
    print("SMOKE OK" if not missing else "SMOKE INCOMPLETE")


if __name__ == "__main__":
    main()
