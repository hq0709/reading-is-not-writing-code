"""Load every registered model and check that the registry tells the truth about it.

The registry's module paths decide where every activation in the paper is read from. A wrong path does
not crash, it silently reads a different tensor, so it has to be checked against a loaded model rather
than trusted. This script loads each one, asserts every claimed path exists, counts LLM layers, resolves
the image token, runs a forward pass, and confirms that hooks at all three stages actually fire.

    python src/verify_arch.py                 # all enabled models, auto-picks a free GPU
    python src/verify_arch.py --only huatuo7b --gpu 7
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loci import as_hidden
try:
    from .gpu_env import bind_gpu
except ImportError:
    from gpu_env import bind_gpu
from registry import REGISTRY, enabled_archs


def free_gpu(min_free_gb=40):
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    best, best_free = None, 0
    for line in q.strip().splitlines():
        i, used, total = [int(x) for x in line.split(",")]
        f = (total - used) / 1024
        if f > best_free:
            best, best_free = i, f
    if best is None or best_free < min_free_gb:
        raise SystemExit(f"no GPU with {min_free_gb} GB free (best was gpu{best} at {best_free:.0f} GB)")
    print(f"using gpu{best}, {best_free:.0f} GB free")
    return best


def verify(arch, gpu, synth):
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    rep = {"key": arch.key, "hf_id": arch.hf_id, "ok": False, "problems": [], "fixes": {}}
    t0 = time.time()
    try:
        cfg = AutoConfig.from_pretrained(arch.hf_id, trust_remote_code=False)
        proc = AutoProcessor.from_pretrained(arch.hf_id, **arch.processor_kwargs)
        model = AutoModelForImageTextToText.from_pretrained(
            arch.hf_id, dtype=torch.bfloat16, device_map="cuda:0").eval()
    except Exception as e:
        rep["problems"].append(f"LOAD FAILED: {type(e).__name__}: {str(e)[:220]}")
        return rep
    rep["load_s"] = round(time.time() - t0, 1)
    rep["params_b"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 2)
    if abs(rep["params_b"] - arch.params_b) > 0.2:
        rep["problems"].append(f"params {rep['params_b']}B, registry says {arch.params_b}B")
        rep["fixes"]["params_b"] = rep["params_b"]

    mods = dict(model.named_modules())

    # every claimed path must exist
    for label, path in (("vision_root", arch.vision_root), ("connector", arch.connector)):
        if path not in mods:
            near = [m for m in mods if m.startswith(path.split(".")[0])][:6]
            rep["problems"].append(f"{label} path {path!r} not found. nearby: {near}")

    for label, fmt, n in (("vision_block", arch.vision_block_fmt, arch.n_vision_blocks),
                          ("llm_layer", arch.llm_layer_fmt, arch.n_llm_layers)):
        present = sum(1 for i in range(n) if fmt.format(i=i) in mods)
        extra = 0
        while fmt.format(i=n + extra) in mods:
            extra += 1
        if present != n:
            rep["problems"].append(f"{label}: {present}/{n} of the claimed indices exist")
        if extra:
            rep["problems"].append(f"{label}: registry says {n} but at least {n + extra} exist")
            rep["fixes"][f"n_{label}s"] = n + extra

    # image token
    tok = getattr(proc, "tokenizer", proc)
    tid = None
    for attr in ("image_token_id", "image_token_index"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int):
            tid = v
            break
    if tid is None and arch.image_token:
        try:
            tid = tok.convert_tokens_to_ids(arch.image_token)
        except Exception:
            tid = None
    rep["image_token_id"] = tid
    if arch.image_token_id is not None and tid is not None and tid != arch.image_token_id:
        rep["problems"].append(f"image token id {tid}, registry says {arch.image_token_id}")
        rep["fixes"]["image_token_id"] = tid
    elif arch.image_token_id is None and tid is not None:
        rep["fixes"]["image_token_id"] = tid

    # hooks must fire at all three stages during a real forward pass
    fired = {}
    handles = []
    targets = {"vision_root": arch.vision_root, "connector": arch.connector,
               "llm_first": arch.llm_layer_fmt.format(i=0),
               "llm_last": arch.llm_layer_fmt.format(i=arch.n_llm_layers - 1)}
    for name, path in targets.items():
        if path not in mods:
            continue
        def mk(name=name):
            def h(_m, _i, out):
                t = as_hidden(out)
                if t is not None:
                    fired[name] = tuple(t.shape)
            return h
        handles.append(mods[path].register_forward_hook(mk()))

    try:
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": "Is there a pleural effusion? Answer yes or no."}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[synth], return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            model(**inputs)
        rep["seq_len"] = int(inputs["input_ids"].shape[1])
        if tid is not None:
            rep["visual_tokens"] = int((inputs["input_ids"] == tid).sum())
            if arch.visual_tokens and abs(rep["visual_tokens"] - arch.visual_tokens) > 0:
                rep["problems"].append(f"visual tokens {rep['visual_tokens']}, registry says {arch.visual_tokens}")
                rep["fixes"]["visual_tokens"] = rep["visual_tokens"]
    except Exception as e:
        rep["problems"].append(f"FORWARD FAILED: {type(e).__name__}: {str(e)[:220]}")
    finally:
        for h in handles:
            h.remove()

    rep["hooks_fired"] = fired
    missing = [k for k in targets if k not in fired and targets[k] in mods]
    if missing:
        rep["problems"].append(f"hooks did not fire at: {missing}")

    rep["ok"] = not rep["problems"]
    del model
    torch.cuda.empty_cache()
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--out", default="runs/arch_verification.json")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    gpu = args.gpu if args.gpu >= 0 else free_gpu()
    bind_gpu(gpu)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from smoke_extract import synthetic_cxr
    synth = synthetic_cxr()

    archs = ([REGISTRY[k.strip()] for k in args.only.split(",") if k.strip()]
             if args.only else enabled_archs())
    reports = []
    for a in archs:
        print(f"\n{'=' * 70}\n{a.key}  ({a.hf_id})")
        r = verify(a, gpu, synth)
        reports.append(r)
        if r["ok"]:
            print(f"  OK  {r.get('params_b')}B, loaded in {r.get('load_s')}s, "
                  f"seq {r.get('seq_len')}, visual tokens {r.get('visual_tokens')}")
            for k, v in r.get("hooks_fired", {}).items():
                print(f"      hook {k:12s} {v}")
        else:
            for p in r["problems"]:
                print(f"  PROBLEM  {p}")
            if r["fixes"]:
                print(f"  SUGGESTED REGISTRY FIXES: {json.dumps(r['fixes'])}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(reports, open(args.out, "w"), indent=1)
    ok = sum(1 for r in reports if r["ok"])
    print(f"\n{'=' * 70}\n{ok}/{len(reports)} models verified. Report at {args.out}")
    for r in reports:
        if not r["ok"]:
            print(f"  {r['key']}: {len(r['problems'])} problem(s)")


if __name__ == "__main__":
    main()
