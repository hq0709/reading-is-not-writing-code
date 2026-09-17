"""Why a block fails preflight check E: is the answer mass absent, or is the mapping inverted?

For one (model, dataset) it reports, on the 16 preflight images and the primary yes/no question:
  - the softmax mass of the template's candidate tokens at the answer position (candidate_mass),
  - P(yes) against P(no) among the candidates,
  - the five most probable tokens at the answer position, with probabilities,
and the same top-token read-out for the four image-free statements check E scores, per template.

    python answer_interface_probe.py --model-key maira2-7 --dataset nih
"""
from __future__ import annotations

import argparse
import json

import torch

from cftransfer.adapters import get_adapter
from cftransfer.images import image_path, load_cohort, open_rgb
from cftransfer.protocol import CONCEPTS, IMAGE_PHRASE, TEMPLATE_ORDER, TEMPLATES, render_question
from cftransfer.runpaths import run_dir
from cftransfer.scoring import score_logits

STATEMENTS = (("The finding is present.", 1), ("The finding is absent.", 0),
              ("The finding is explicitly reported as present.", 1),
              ("The finding is explicitly reported as absent.", 0))


def top_tokens(tok, logits, k=5):
    p = torch.softmax(logits.float(), dim=-1)
    v, i = p.topk(k)
    return [{"token": tok.convert_ids_to_tokens(int(j)), "id": int(j), "p": round(float(x), 5)}
            for x, j in zip(v, i)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device-map", default="cuda:0")
    a = ap.parse_args()

    rows = load_cohort(a.dataset, ("preflight",))
    images = [open_rgb(image_path(a.dataset, r)) for r in rows]
    ad = get_adapter(a.model_key).load(device_map=a.device_map)
    tok = ad.tokenizer
    concept = CONCEPTS[a.dataset][0]
    q = render_question(a.dataset, concept, "IY")
    out = {"model_key": a.model_key, "dataset_id": a.dataset, "revision": ad.revision, "question_IY": q,
           "candidates": {t: {"positive_ids": list(c.positive_ids), "negative_ids": list(c.negative_ids),
                              "positive_tokens": tok.convert_ids_to_tokens(list(c.positive_ids)),
                              "negative_tokens": tok.convert_ids_to_tokens(list(c.negative_ids))}
                          for t, c in ad.candidates.items()}}

    with_image = []
    for im, r in zip(images, rows):
        enc = ad.encode([im], [q])
        lg = ad.forward_last_logits(enc)
        s = score_logits(lg, ad.candidates["IY"])
        with_image.append({"row_id": r["row_id"], "semantic_margin": round(float(s.semantic_margin[0]), 4),
                           "p_present": round(float(s.p_present[0]), 5),
                           "candidate_mass": round(float(s.candidate_mass[0]), 6),
                           "top_tokens": top_tokens(tok, lg[0])})
    out["with_image_IY"] = with_image
    out["with_image_IY_summary"] = {
        "candidate_mass_min": round(min(x["candidate_mass"] for x in with_image), 6),
        "candidate_mass_max": round(max(x["candidate_mass"] for x in with_image), 6),
        "candidate_mass_mean": round(sum(x["candidate_mass"] for x in with_image) / len(with_image), 6),
        "p_present_min": round(min(x["p_present"] for x in with_image), 5),
        "p_present_max": round(max(x["p_present"] for x in with_image), 5),
        "distinct_top1_tokens": sorted({x["top_tokens"][0]["token"] for x in with_image})}

    image_free = {}
    for t in TEMPLATE_ORDER:
        suffix = TEMPLATES[t].split("? ", 1)[1]
        cases = []
        for stmt, lab in STATEMENTS:
            text = f"{stmt} Is the finding present? {suffix}"
            enc = ad.encode(None, [text])
            lg = ad.forward_last_logits(enc)
            s = score_logits(lg, ad.candidates[t])
            cases.append({"statement": stmt, "expected": lab,
                          "semantic_margin": round(float(s.semantic_margin[0]), 4),
                          "candidate_mass": round(float(s.candidate_mass[0]), 6),
                          "top_tokens": top_tokens(tok, lg[0])})
        image_free[t] = cases
    out["image_free_by_template"] = image_free

    # the image-phrased variant preflight keeps as a descriptive record
    wording = {}
    for t in TEMPLATE_ORDER:
        cases = []
        for stmt, lab in STATEMENTS:
            text = (f"The target finding is {concept.lower()}. {stmt} "
                    + TEMPLATES[t].format(finding="the finding", image_phrase=IMAGE_PHRASE[a.dataset]))
            enc = ad.encode(None, [text])
            lg = ad.forward_last_logits(enc)
            s = score_logits(lg, ad.candidates[t])
            cases.append({"statement": stmt, "expected": lab,
                          "semantic_margin": round(float(s.semantic_margin[0]), 4),
                          "candidate_mass": round(float(s.candidate_mass[0]), 6),
                          "top_tokens": top_tokens(tok, lg[0])})
        wording[t] = cases
    out["image_free_image_phrased_by_template"] = wording

    path = run_dir(a.model_key, a.dataset) / "answer_interface_probe.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({"with_image_IY_summary": out["with_image_IY_summary"],
                      "IY_image_free": [{k: c[k] for k in ("statement", "expected", "semantic_margin",
                                                           "candidate_mass")} | {"top1": c["top_tokens"][0]}
                                        for c in image_free["IY"]]}, indent=1))
    print("written", path, flush=True)


if __name__ == "__main__":
    main()
