"""Is the gap between what a probe reads and what the model answers a matter of ELICITATION?

The objection this experiment exists to settle: a probe is fitted on 18,212 labelled radiographs, while
the model is asked one zero-shot question. A probe winning is then unremarkable, and "the model knows but
cannot say" is not established. Two outcomes separate the hypotheses:

    the answer improves a lot under better elicitation   the information was accessible and the readout
                                                          was the bottleneck. An elicitation gap.
    the answer does not move                              the probe's advantage came from its labels, and
                                                          the claim does not survive.

One thing this cannot be. AUROC is invariant to any strictly monotone transform of the score, so a constant
tendency to answer yes cannot lower it. LLaVA-Med answering yes on 100% of radiographs is a statement about
its decision threshold; its AUROC of 0.556 for effusion is a statement about its ranking, and no
recalibration can change the second. Elicitation methods are therefore only interesting here if they change
the ORDER the model puts the images in, which is why a content-free prior subtraction is not among them.

Four readouts, on identical images and labels:

  yesno       the original question, P(yes) against P(no) at the answer position
  yesno_alt   the same content, different surface form. Tests sensitivity to phrasing alone
  statement   the mean log-probability of "Yes, there is a <finding>." against "No, there is no
              <finding>." scored by teacher forcing. A yes/no token may be a poor readout even when the
              underlying distinction is present
  negation    P(yes | "Is there a <finding>?") minus P(yes | "Is this radiograph free of <finding>?").
              A model with a fixed response bias contributes to both terms and cancels; a model that
              tracks the finding does not

    python src/elicit.py --arch lingshu7b --gpu 5 --out runs/elicit_lingshu7b
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from intervene import yes_margin_batch
from registry import REGISTRY

CONCEPTS = ["Effusion", "Cardiomegaly", "Pneumothorax", "Atelectasis", "Consolidation",
            "Edema", "Infiltration", "Mass", "Nodule"]
PHRASE = {
    "Effusion": "pleural effusion", "Cardiomegaly": "cardiomegaly", "Pneumothorax": "pneumothorax",
    "Atelectasis": "atelectasis", "Consolidation": "consolidation", "Edema": "pulmonary edema",
    "Infiltration": "infiltration", "Mass": "mass", "Nodule": "nodule",
}


def chat(proc, text):
    return proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
        tokenize=False, add_generation_prompt=True)


def margin_scores(model, proc, tok, imgs, prompt_text, batch_size, dev):
    """P(yes)-vs-P(no) margin at the answer position, for a list of PIL images."""
    out = []
    t = chat(proc, prompt_text)
    for b0 in range(0, len(imgs), batch_size):
        chunk = imgs[b0:b0 + batch_size]
        inp = proc(text=[t] * len(chunk), images=chunk, return_tensors="pt", padding=True).to(dev)
        with torch.inference_mode():
            lg = model(**inp).logits
        out.extend(yes_margin_batch(lg, tok))
    return np.array(out)


def continuation_logprob(model, proc, tok, imgs, prompt_text, answer, batch_size, dev):
    """Mean log-probability of `answer` as a continuation of the prompt, by teacher forcing."""
    t = chat(proc, prompt_text)
    ans_ids = tok.encode(answer, add_special_tokens=False)
    n_ans = len(ans_ids)
    out = []
    for b0 in range(0, len(imgs), batch_size):
        chunk = imgs[b0:b0 + batch_size]
        inp = proc(text=[t + answer] * len(chunk), images=chunk, return_tensors="pt",
                   padding=True).to(dev)
        with torch.inference_mode():
            lg = model(**inp).logits.float()
        # positions predicting the answer tokens are the n_ans logits ending one before the last
        lp = torch.log_softmax(lg[:, -n_ans - 1:-1, :], dim=-1)
        tgt = torch.tensor(ans_ids, device=lp.device).unsqueeze(0).expand(lp.shape[0], -1)
        out.extend(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean(dim=1).tolist())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--arch", required=True)
    ap.add_argument("--concepts", default=",".join(CONCEPTS))
    ap.add_argument("--n-images", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="must match src/behaviour.py so the rows match")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    dev = "cuda:0"

    man = [r for r in csv.DictReader(open(args.manifest)) if r["split"] == "test"]
    idx = rng.permutation(len(man))[:args.n_images]
    rows = [man[i] for i in idx]
    print(f"{len(man)} test rows, scoring {len(rows)} (same seed and sample as src/behaviour.py)")

    from transformers import AutoModelForImageTextToText, AutoProcessor
    arch = REGISTRY[args.arch]
    proc = AutoProcessor.from_pretrained(arch.hf_id, **arch.processor_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(arch.hf_id, dtype=torch.bfloat16,
                                                        device_map=dev).eval()
    tok = getattr(proc, "tokenizer", proc)

    imgs = [Image.open(r["image_path"]).convert("RGB") for r in rows]
    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    out = []
    for c in concepts:
        if c not in rows[0]:
            continue
        y = np.array([int(r[c]) for r in rows])
        if len(np.unique(y)) < 2:
            continue
        ph = PHRASE.get(c, c.lower())
        t0 = time.time()
        scores = {}
        scores["yesno"] = margin_scores(
            model, proc, tok, imgs, f"Is there a {ph} in this chest radiograph? Answer yes or no.",
            args.batch_size, dev)
        scores["yesno_alt"] = margin_scores(
            model, proc, tok, imgs, f"Does this chest radiograph show {ph}? Answer yes or no.",
            args.batch_size, dev)
        q = f"Is there a {ph} in this chest radiograph?"
        pos = continuation_logprob(model, proc, tok, imgs, q, f"Yes, there is a {ph}.",
                                   args.batch_size, dev)
        neg = continuation_logprob(model, proc, tok, imgs, q, f"No, there is no {ph}.",
                                   args.batch_size, dev)
        scores["statement"] = pos - neg
        free = margin_scores(
            model, proc, tok, imgs, f"Is this chest radiograph free of {ph}? Answer yes or no.",
            args.batch_size, dev)
        scores["negation"] = scores["yesno"] - free

        row = {"arch": args.arch, "domain": arch.domain, "concept": c, "n": len(rows),
               "prevalence": round(float(y.mean()), 4)}
        for k, v in scores.items():
            row[f"auroc_{k}"] = round(float(roc_auc_score(y, v)), 4)
        best = max(scores, key=lambda k: row[f"auroc_{k}"])
        row["best_method"] = best
        row["auroc_best"] = row[f"auroc_{best}"]
        row["gain_over_yesno"] = round(row["auroc_best"] - row["auroc_yesno"], 4)
        row["elapsed_s"] = round(time.time() - t0, 1)
        out.append(row)
        print(f"  {c:15s} yesno {row['auroc_yesno']:.4f}  alt {row['auroc_yesno_alt']:.4f}  "
              f"statement {row['auroc_statement']:.4f}  negation {row['auroc_negation']:.4f}  "
              f"-> best {best} {row['auroc_best']:.4f} (+{row['gain_over_yesno']:.4f})  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        np.savez_compressed(os.path.join(args.out, f"scores_{c}.npz"), y=y, **scores)
        with open(os.path.join(args.out, "elicitation.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

    json.dump({"arch": args.arch, "n_images": len(rows), "seed": args.seed,
               "row_ids": [r["row_id"] for r in rows],
               "note": "AUROC is invariant to monotone rescaling, so a constant answer bias cannot lower "
                       "it. Only methods that change the ORDER of the images can close the gap."},
              open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    with open(os.path.join(args.out, "DONE"), "w") as fh:
        fh.write(json.dumps({"rows": len(out), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    print(f"\nwrote {len(out)} rows to {args.out}/elicitation.csv")


if __name__ == "__main__":
    main()
