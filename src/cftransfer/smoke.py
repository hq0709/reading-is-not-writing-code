"""Family interface smoke on the 16 preflight images, before any fit exists (used when a new family lands).

Reports candidate sets, feature shapes/token counts at both loci, a reach check with a random unit direction
(determinism, alpha-0 no-op, consumer change, CLS/non-consumed rows untouched), batch-vs-single agreement,
fp32-vs-model logit agreement and the rendered prompt. Writes <run_dir>/smoke.json.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from .adapters import get_adapter
from .features import extract
from .hooks import CaptureHook, LocusHook
from .images import image_path, load_cohort, open_rgb
from .protocol import CONCEPTS, render_question
from .runpaths import run_dir
from .scoring import score_logits


def main(model_key: str, dataset_id: str, device_map: str):
    t0 = time.time()
    out = run_dir(model_key, dataset_id); out.mkdir(parents=True, exist_ok=True)
    meta = extract(model_key, dataset_id, ("preflight",), 16, device_map, out / "features_smoke")
    ad = get_adapter(model_key).load(device_map=device_map)
    loci = ad.loci()
    rows = load_cohort(dataset_id, ("preflight",))
    images = [open_rgb(image_path(dataset_id, r)) for r in rows]
    q = render_question(dataset_id, CONCEPTS[dataset_id][0], "IY")
    rep = {"model_key": model_key, "dataset_id": dataset_id, "revision": ad.revision, "features": {k: v for k, v in meta.items() if k != "loci"},
           "loci": {k: vars(v) for k, v in loci.items()},
           "candidates": {t: {"positive": list(c.positive_ids), "negative": list(c.negative_ids), "detail": c.detail}
                          for t, c in ad.candidates.items() if t in ("IY", "IA")}}
    D = loci["vis.last"].hidden_dim
    v = torch.from_numpy(np.random.default_rng(0).standard_normal((1, D)).astype(np.float32)); v = v / v.norm()
    hook = LocusHook(ad.module(loci["vis.last"].module_path), "vis.last")
    cons = CaptureHook(ad.module(loci["connector"].module_path), "connector")
    loc = CaptureHook(ad.module(loci["vis.last"].module_path), "locus")
    sub = images[:4]
    enc = ad.encode(sub, [q] * 4); lay = ad.layouts(enc, sub)["vis.last"]
    with hook, cons, loc:
        hook.arm(lay); c1, m1 = ad.forward_last_logits(enc, return_model_logits=True); k1 = cons.value.clone(); l1 = loc.value.clone()
        hook.arm(lay); c2 = ad.forward_last_logits(enc)
        hook.arm(lay, torch.zeros(4, D), torch.zeros(4)); z = ad.forward_last_logits(enc)
        hook.arm(lay, v.repeat(4, 1), torch.full((4,), 0.25)); st = ad.forward_last_logits(enc); k2 = cons.value.clone(); l2 = loc.value.clone()
    cid = list(ad.candidates["IY"].positive_ids) + list(ad.candidates["IY"].negative_ids)
    diff = (l2 - l1).abs()
    outside = 0.0
    if not lay.flat:
        om = torch.stack([~m for m in lay.masks])
        outside = float(diff[om].max()) if om.any() else 0.0
    rep["reach"] = {"determinism": float((c1[:, cid] - c2[:, cid]).abs().max()), "alpha0": float((z[:, cid] - c1[:, cid]).abs().max()),
                    "connector_change": float((k2 - k1).abs().max()), "logit_change": float((st[:, cid] - c1[:, cid]).abs().max()),
                    "locus_change_inside": float(diff.max()), "locus_change_outside_consumed": outside,
                    "margin_clean": score_logits(c1, ad.candidates["IY"]).semantic_margin.tolist(),
                    "margin_steered": score_logits(st, ad.candidates["IY"]).semantic_margin.tolist(),
                    "fp32_vs_model_max_abs": float((c1[:, cid] - m1[:, cid]).abs().max()), "hook_stats": hook.stats}
    singles = []
    with hook:
        for im in images:
            e = ad.encode([im], [q]); hook.arm(ad.layouts(e, [im])["vis.last"]); singles.append(ad.forward_last_logits(e)[:, cid])
        eb = ad.encode(images, [q] * 16); hook.arm(ad.layouts(eb, images)["vis.last"]); batched = ad.forward_last_logits(eb)[:, cid]
        e1 = ad.encode([images[0]], [q]); hook.arm(ad.layouts(e1, [images[0]])["vis.last"]); s1 = ad.forward_last_logits(e1)[:, cid]
        er = ad.encode([images[0]] * 16, [q] * 16); hook.arm(ad.layouts(er, [images[0]] * 16)["vis.last"]); sr = ad.forward_last_logits(er)[:, cid]
    single = torch.cat(singles)
    rep["batch_vs_single"] = {"max_abs_logit_diff": float((single - batched).abs().max()),
                              "replicated_max_abs_logit_diff": float((sr - s1).abs().max())}
    ex = ad.encode([images[0]], [q])
    rep["prompt"] = {"text": ad.prompt_text(q), "input_token_count": int(ex["attention_mask"].sum()),
                     "valid_tokens": {k: ad.layouts(ex, [images[0]])[k].counts()[0] for k in loci}}
    rep["seconds"] = round(time.time() - t0, 1)
    rep["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    ok = rep["reach"]["determinism"] == 0 and rep["reach"]["alpha0"] == 0 and rep["reach"]["connector_change"] > 0 \
        and rep["reach"]["logit_change"] > 0 and outside == 0 and rep["batch_vs_single"]["max_abs_logit_diff"] <= 0.25
    rep["pass"] = bool(ok)
    (out / "smoke.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("candidates", "reach", "batch_vs_single", "prompt", "seconds", "pass")}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True); ap.add_argument("--dataset", default="nih"); ap.add_argument("--device-map", default="cuda:0")
    a = ap.parse_args(); main(a.model_key, a.dataset, a.device_map)
