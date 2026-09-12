"""Interface verification on the 16 fixed preflight images (README 5.1, return-format preflight.json).

Checks, for one (model, dataset, locus):
  A. determinism: two clean forwards give identical candidate logits;
  B. alpha-0 no-op: a zero-vector, alpha-0 hook pass equals the clean forward bitwise;
  C. reach: alpha=+0.25 along the first concept direction changes the connector output and the answer logits,
     and only the consumed tokens of the hooked tensor change (CLS / non-consumed entries untouched);
  D. batch vs single: the 16 images scored one at a time and as one padded batch agree within the declared
     tolerance (max |delta candidate logit| <= 0.25, semantic-margin sign identical); D2 repeats the comparison
     for the real run composition (one image replicated 16x, clean) against its single-example score;
  E. semantic direction: four generic present/absent statements x six templates, image-free, must give
     correctly signed margins (present > 0 > absent);
  F. throughput: 256 image-conditions (16 x [baseline + 6 concepts + 9 random]) timed end to end including
     processor, forward, scoring and parquet write; peak memory and GPU count recorded.
A failed check means the measurement is not ready for this family; nothing scientific is inferred from it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .adapters import get_adapter
from .fit import load_fit
from .hooks import CaptureHook, LocusHook
from .images import image_path, load_cohort, open_rgb
from .protocol import CONCEPTS, LOCI, PRIMARY_ALPHA, TEMPLATE_ORDER, TEMPLATES, IMAGE_PHRASE, render_question
from .runpaths import run_dir
from .scoring import score_logits

STATEMENTS = (("The finding is present.", 1), ("The finding is absent.", 0),
              ("The finding is explicitly reported as present.", 1), ("The finding is explicitly reported as absent.", 0))
LOGIT_TOL = 0.25   # two bf16 ulps at |logit| in [16, 32); batch composition changes kernel paths


def cand_logits(ad, logits, template_id):
    c = ad.candidates[template_id]
    return logits[:, list(c.positive_ids) + list(c.negative_ids)]


def run_preflight(model_key: str, dataset_id: str, locus_key: str, device_map: str, batch: int = 16,
                  revision: str | None = None) -> dict:
    locus_id = LOCI[locus_key]
    rows = load_cohort(dataset_id, ("preflight",))
    assert len(rows) == 16, len(rows)
    images = [open_rgb(image_path(dataset_id, r)) for r in rows]
    ad = get_adapter(model_key, revision).load(device_map=device_map)
    loci = ad.loci()
    locus = loci[locus_id]
    fit = load_fit(model_key, dataset_id, locus_id, 0)
    concepts = CONCEPTS[dataset_id]
    q = render_question(dataset_id, concepts[0], "IY")
    report = {"model_key": model_key, "dataset_id": dataset_id, "locus_id": locus_id, "locus": vars(locus),
              "revision": ad.revision, "rows": [r["row_id"] for r in rows], "checks": {}}
    hook = LocusHook(ad.module(locus.module_path), locus_id)
    consumer = CaptureHook(ad.module(loci["connector"].module_path if locus_id == "vis.last" else "lm_head"), "consumer")
    locus_cap = CaptureHook(ad.module(locus.module_path), "locus")
    v0 = torch.from_numpy(fit["clinical_vectors"][0:1])

    # ---- A/B/C on the first 4 images as a batch, plus each singly
    sub = images[:4]
    enc = ad.encode(sub, [q] * 4)
    lay = ad.layouts(enc, sub)[locus_id]
    with hook, consumer, locus_cap:
        hook.arm(lay)
        clean1 = ad.forward_last_logits(enc); cons_clean = consumer.value.clone(); loc_clean = locus_cap.value.clone()
        hook.arm(lay)
        clean2 = ad.forward_last_logits(enc)
        hook.arm(lay, torch.zeros(4, lay_dim := int(fit["clinical_vectors"].shape[1])), torch.zeros(4))
        zero = ad.forward_last_logits(enc)
        hook.arm(lay, v0.repeat(4, 1), torch.full((4,), PRIMARY_ALPHA))
        steered = ad.forward_last_logits(enc); cons_st = consumer.value.clone(); loc_st = locus_cap.value.clone()
    cl = cand_logits(ad, clean1, "IY")
    report["checks"]["A_determinism_max_abs_diff"] = float((cand_logits(ad, clean2, "IY") - cl).abs().max())
    report["checks"]["B_alpha0_max_abs_diff"] = float((cand_logits(ad, zero, "IY") - cl).abs().max())
    report["checks"]["C_reach"] = {
        "consumer_module": consumer.name if consumer.name != "consumer" else (loci["connector"].module_path if locus_id == "vis.last" else "lm_head"),
        "consumer_max_abs_change": float((cons_st - cons_clean).abs().max()),
        "answer_logit_max_abs_change": float((cand_logits(ad, steered, "IY") - cl).abs().max()),
        "mean_semantic_margin_change": float((score_logits(steered, ad.candidates["IY"]).semantic_margin
                                              - score_logits(clean1, ad.candidates["IY"]).semantic_margin).mean()),
        "hook_stats_steered": {k: [round(x, 4) for x in v] for k, v in hook.stats.items()},
    }
    # only consumed tokens changed at the locus tensor
    diff = (loc_st - loc_clean).abs()
    if lay.flat:
        changed_outside = 0.0
    else:
        outside = torch.stack([~m for m in lay.masks])
        changed_outside = float(diff[outside].max()) if outside.any() else 0.0
    report["checks"]["C_reach"]["locus_change_outside_consumed_tokens"] = changed_outside
    report["checks"]["C_reach"]["locus_change_inside_consumed_tokens"] = float(diff.max())

    # ---- D batch vs single on all 16 images (clean)
    singles = []
    with hook:
        for im in images:
            e1 = ad.encode([im], [q]); hook.arm(ad.layouts(e1, [im])[locus_id])
            singles.append(cand_logits(ad, ad.forward_last_logits(e1), "IY"))
        eb = ad.encode(images, [q] * len(images)); hook.arm(ad.layouts(eb, images)[locus_id])
        batched = cand_logits(ad, ad.forward_last_logits(eb), "IY")
    single = torch.cat(singles)
    npos = len(ad.candidates["IY"].positive_ids)
    m_single = single[:, :npos].max(1).values - single[:, npos:].max(1).values
    m_batch = batched[:, :npos].max(1).values - batched[:, npos:].max(1).values
    report["checks"]["D_batch_vs_single"] = {
        "max_abs_candidate_logit_diff": float((single - batched).abs().max()),
        "max_abs_margin_diff": float((m_single - m_batch).abs().max()),
        "margin_sign_agreement": bool(torch.all(torch.sign(m_single) == torch.sign(m_batch))),
        "declared_tolerance_logit": LOGIT_TOL,
        "pass": bool((single - batched).abs().max() <= LOGIT_TOL and torch.all(torch.sign(m_single) == torch.sign(m_batch))),
    }

    # ---- G float32 answer logits vs the model's own bf16 logits (must agree within bf16 resolution)
    with hook:
        eg = ad.encode(images[:4], [q] * 4); hook.arm(ad.layouts(eg, images[:4])[locus_id])
        lg32, lgm = ad.forward_last_logits(eg, return_model_logits=True)
    c32, cm = cand_logits(ad, lg32, "IY"), cand_logits(ad, lgm, "IY")
    ulp = torch.maximum(torch.tensor(0.0625), (2.0 ** torch.floor(torch.log2(cm.abs().clamp(min=1e-6)))) / 128)
    report["checks"]["G_fp32_vs_model_logits"] = {"max_abs_diff": float((c32 - cm).abs().max()),
                                                 "max_diff_in_ulps": float(((c32 - cm).abs() / ulp).max()),
                                                 "pass": bool(((c32 - cm).abs() <= 2 * ulp).all())}

    # ---- D2 replicated batch (the runner's composition) vs single, clean
    with hook:
        e1 = ad.encode([images[0]], [q]); hook.arm(ad.layouts(e1, [images[0]])[locus_id])
        s1 = cand_logits(ad, ad.forward_last_logits(e1), "IY")
        eb = ad.encode([images[0]] * 16, [q] * 16); hook.arm(ad.layouts(eb, [images[0]] * 16)[locus_id])
        sb = cand_logits(ad, ad.forward_last_logits(eb), "IY")
    report["checks"]["D2_replicated_batch_vs_single"] = {
        "max_abs_candidate_logit_diff": float((sb - s1).abs().max()),
        "within_batch_max_spread": float((sb - sb[0:1]).abs().max()),
        "pass": bool((sb - s1).abs().max() <= LOGIT_TOL)}

    # ---- E semantic direction, image-free: statement + "Is the finding present?" + the template's answer instruction
    # (the wording stems reference an image that is absent here, so only the answer mapping is verified; the
    # image-phrased variant is kept as a descriptive record)
    sem, sem_wording = {}, {}
    all_ok = True
    for t in TEMPLATE_ORDER:
        c = ad.candidates[t]
        suffix = TEMPLATES[t].split("? ", 1)[1]
        cases, wcases = [], []
        for stmt, lab in STATEMENTS:
            for variant, text in (("mapping", f"{stmt} Is the finding present? {suffix}"),
                                  ("wording", f"The target finding is {concepts[0].lower()}. {stmt} "
                                              + TEMPLATES[t].format(finding="the finding", image_phrase=IMAGE_PHRASE[dataset_id]))):
                e = ad.encode(None, [text])
                s = score_logits(ad.forward_last_logits(e), c)
                ok = (s.semantic_margin[0] > 0) if lab == 1 else (s.semantic_margin[0] < 0)
                rec = {"statement": stmt, "expected": lab, "semantic_margin": float(s.semantic_margin[0]),
                       "lse_margin": float(s.lse_margin[0]), "raw_ab_margin": float(s.raw_ab_margin[0]), "ok": bool(ok), "text": text}
                if variant == "mapping":
                    all_ok &= bool(ok); cases.append(rec)
                else:
                    wcases.append(rec)
        sem[t] = cases; sem_wording[t] = wcases
    report["checks"]["E_semantic_direction"] = {"cases": sem, "pass": all_ok,
                                                "descriptive_image_phrased_variant": sem_wording}

    # ---- F throughput: 256 image-conditions incl. processor, forward, scoring, parquet write
    conds = [("baseline", 0.0)] + [(f"concept:{c}", PRIMARY_ALPHA) for c in concepts] + [(f"random:{i:03d}", PRIMARY_ALPHA) for i in range(9)]
    vec = {f"concept:{c}": fit["clinical_vectors"][i] for i, c in enumerate(concepts)}
    vec.update({f"random:{i:03d}": fit["random_vectors"][i] for i in range(9)})
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time(); rows_out = []
    with hook:
        for im, r in zip(images, rows):
            e1 = ad.encode([im], [q]); hook.arm(ad.layouts(e1, [im])[locus_id])
            lg = ad.forward_last_logits(e1); s = score_logits(lg, ad.candidates["IY"])
            rows_out.append({"row_id": r["row_id"], "direction_id": "baseline", "semantic_margin": float(s.semantic_margin[0])})
            st = conds[1:]
            for i in range(0, len(st), batch):
                grp = st[i:i + batch]
                eb = ad.encode([im] * len(grp), [q] * len(grp))
                hook.arm(ad.layouts(eb, [im] * len(grp))[locus_id], torch.from_numpy(np.stack([vec[d] for d, _ in grp])),
                         torch.tensor([a for _, a in grp]))
                s = score_logits(ad.forward_last_logits(eb), ad.candidates["IY"])
                rows_out += [{"row_id": r["row_id"], "direction_id": d, "semantic_margin": float(s.semantic_margin[j])}
                             for j, (d, _) in enumerate(grp)]
    out = run_dir(model_key, dataset_id); out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows_out), out / f"preflight_timing.{locus_id}.parquet")
    torch.cuda.synchronize(); el = time.time() - t0
    g = torch.cuda.device_count()
    report["checks"]["F_throughput"] = {"image_conditions": len(rows_out), "seconds": round(el, 2),
                                        "rate_per_s": round(len(rows_out) / el, 2), "gpu_count": g,
                                        "batch": batch, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                                        "gpu_models": sorted({torch.cuda.get_device_name(i) for i in range(g)})}
    ex = ad.encode([images[0]], [q])
    report["input_example"] = {"prompt": ad.prompt_text(q), "input_token_count": int(ex["attention_mask"].sum()),
                               "valid_token_count": ad.layouts(ex, [images[0]])[locus_id].counts()[0],
                               "processor_kwargs": ad.processor_kwargs}
    report["batch_policy"] = ("runner scores every condition of a question at one fixed batch composition (replicated image, "
                              "clean replicated baseline batch with the hook passive), so single-vs-batch differences never enter W_qd; "
                              "D/D2 are reported against the declared tolerance and recorded as a deviation when exceeded")
    report["checks"]["D2_replicated_batch_vs_single"]["within_batch_spread_pass"] = report["checks"]["D2_replicated_batch_vs_single"]["within_batch_max_spread"] == 0.0
    report["pass"] = bool(report["checks"]["A_determinism_max_abs_diff"] == 0.0 and report["checks"]["B_alpha0_max_abs_diff"] == 0.0
                          and report["checks"]["C_reach"]["consumer_max_abs_change"] > 0 and report["checks"]["C_reach"]["answer_logit_max_abs_change"] > 0
                          and report["checks"]["C_reach"]["locus_change_outside_consumed_tokens"] == 0.0
                          and report["checks"]["D_batch_vs_single"]["margin_sign_agreement"]
                          and report["checks"]["D2_replicated_batch_vs_single"]["within_batch_spread_pass"]
                          and report["checks"]["G_fp32_vs_model_logits"]["pass"] and all_ok)
    report["tolerance_deviations"] = [k for k in ("D_batch_vs_single", "D2_replicated_batch_vs_single") if not report["checks"][k]["pass"]]
    (out / f"preflight.{locus_id}.json").write_text(json.dumps(report, indent=1))
    elig = {t: all(x["ok"] for x in cases) for t, cases in sem.items()}
    elig_path = out / "template_eligibility.json"
    prev = json.loads(elig_path.read_text()) if elig_path.exists() else {}
    prev.update({t: {"eligible": ok, "reason": "" if ok else "semantic mapping failed image-free preflight (check E)",
                     "source": f"preflight.{locus_id}.json"} for t, ok in elig.items()})
    elig_path.write_text(json.dumps(prev, indent=1))
    print(json.dumps({k: report["checks"][k] for k in report["checks"] if k != "E_semantic_direction"}, indent=1))
    print("E_semantic_direction pass:", all_ok, "| overall pass:", report["pass"], flush=True)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--locus", default="primary", choices=["primary", "connector"])
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    run_preflight(a.model_key, a.dataset, a.locus, a.device_map, a.batch)
