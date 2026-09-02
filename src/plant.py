"""Experiment D3: plant a finding of known strength and see which direction the representation moves.

Every other experiment here compares directions against each other. None of them has a ground truth, so
none can say which direction is *right*. This one manufactures one. Take radiographs the model reads as
normal, add a synthetic lesion of controlled intensity, and record two things: how the answer moves, and
how the activation at each locus moves.

The activation displacement caused by the planted lesion,

    dh(beta) = h(image + beta * lesion) - h(image)

is the direction the concept actually travels in when the concept is actually added to the input. It is
the reference the field has never had. Everything else can then be scored against it:

    cos(dh, v_probe)    is the direction that best READS the concept the direction the image moves along?
    cos(dh, v_cad)      is the direction that best WRITES it?
    cos(dh, random)     the floor, which in 3584 dimensions sits near zero

Two controls decide whether the planted lesion means anything at all.

  location   the identical blob placed outside the thorax. Same pixels changed, same intensity, no
             anatomy. A model that answers "yes" to a bright patch anywhere is reporting pixel statistics,
             and its dose-response carries no information about the finding
  concept    the answer to the other findings' questions. A lesion planted as a nodule that raises P(yes)
             for effusion equally is not being read as a nodule

The lesion model is deliberately simple: a smooth blob, of the sign and rough location the finding takes.
It is not a clinically realistic lesion and the paper must not imply that it is. What it has to be is
*controlled*, so that beta is a known quantity and the two controls above are available. The gate is stated
in the plan: unless the behavioural dose-response at the anatomical site is monotone and separates from the
location control, nothing downstream of it is interpretable.

    python src/plant.py --arch lingshu7b --concept Effusion --gpu 3 --out runs/plant_lingshu_effusion
"""
from __future__ import annotations

import argparse
import csv
import glob
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

from intervene import fit_direction, load_shards
from loci import ActivationCache, find_image_token_id, loci_for
from registry import REGISTRY

# Where each finding is planted, in fractions of the image box, and with what sign. Chest radiographs in
# this dataset are roughly aligned: the mediastinum runs down the middle, the diaphragm sits near 0.72 of
# the height. These are coarse anatomical priors, not a segmentation, which is why the location control
# exists to absorb the error.
# Only findings a radially symmetric opacity can honestly stand in for. Cardiomegaly is deliberately
# absent: it is a change in the SHAPE of the cardiac silhouette, and a bright disc at the heart position is
# not a small version of it. Run as one anyway, it produced exactly what an invalid lesion model produces:
# a response that was erratic across models, that moved the other findings' questions as much as its own
# (concept ratio 0.2 to 2.0), and that on LLaVA-1.5 moved the answer while the location control moved it
# the other way. The finding has to be one the perturbation actually resembles, or the dose axis means
# nothing.
SITES = {
    # name          centres (x, y)                       radius  sign  what it imitates
    "Effusion":     ([(0.24, 0.74), (0.76, 0.74)],       0.085,  +1),   # basal opacity, blunted angles
    "Nodule":       ([(0.34, 0.42)],                     0.035,  +1),   # a solitary round opacity
    "Mass":         ([(0.66, 0.45)],                     0.065,  +1),   # a larger round opacity
    "Consolidation":([(0.30, 0.55)],                     0.090,  +1),   # a segmental airspace opacity
    "Pneumothorax": ([(0.28, 0.22)],                     0.075,  -1),   # apical lucency
}
# The location control: the same blob, in a corner that carries no thoracic anatomy.
CONTROL_SITE = [(0.09, 0.13)]

YES = ("yes", "Yes", "YES")
NO = ("no", "No", "NO")


def blob(h, w, centres, radius, softness=0.55):
    """A smooth radially-decaying mask, summed over centres, peaking at 1."""
    yy, xx = np.mgrid[0:h, 0:w]
    m = np.zeros((h, w), dtype=np.float32)
    r = radius * min(h, w)
    for cx, cy in centres:
        d2 = (xx - cx * w) ** 2 + (yy - cy * h) ** 2
        m = np.maximum(m, np.exp(-d2 / (2 * (r * softness) ** 2)))
    return m


def plant(img: Image.Image, centres, radius, sign, beta):
    """Add `sign * beta * blob` in normalised intensity, then clip. beta is a fraction of full range."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    m = blob(a.shape[0], a.shape[1], centres, radius)[..., None]
    out = np.clip(a + sign * beta * m, 0.0, 1.0)
    return Image.fromarray((out * 255).astype(np.uint8))


def answer_ids(tok):
    def ids_of(words):
        out = []
        for w in words:
            for form in (w, " " + w):
                i = tok.encode(form, add_special_tokens=False)
                if len(i) == 1:
                    out.append(i[0])
        return sorted(set(out))
    return ids_of(YES), ids_of(NO)


def p_yes_batch(logits, yi, ni):
    lp = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
    return torch.sigmoid(lp[:, yi].max(-1).values - lp[:, ni].max(-1).values).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--acts", default="", help="activation dir, for the probe direction. defaults by arch")
    ap.add_argument("--arch", required=True)
    ap.add_argument("--concept", default="Effusion")
    ap.add_argument("--others", default="Cardiomegaly,Pneumothorax")
    ap.add_argument("--betas", default="0,0.05,0.1,0.2,0.3,0.45,0.6")
    ap.add_argument("--loci", default="", help="default: the D2 locus set for this arch")
    ap.add_argument("--cad-dir", default="", help="runs/cad_<arch>_<concept>, to score v_cad as well")
    ap.add_argument("--n-images", type=int, default=96)
    ap.add_argument("--n-screen", type=int, default=192,
                    help="normal radiographs scored before selection, to find ones with headroom")
    ap.add_argument("--max-base-p", type=float, default=0.5,
                    help="keep screened images whose baseline P(yes) is at or below this")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    bind_gpu(args.gpu)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    dev = "cuda:0"
    if args.concept not in SITES:
        raise SystemExit(f"no lesion model for {args.concept}. known: {sorted(SITES)}")
    centres, radius, sign = SITES[args.concept]
    betas = [float(b) for b in args.betas.split(",")]

    man = list(csv.DictReader(open(args.manifest)))
    # Only radiographs the dataset calls normal AND that lie in the test split, so the images that shaped
    # the probe direction are not the images the planted lesion is measured on.
    pool = [r for r in man if r.get("no_finding") == "1" and r["split"] == "test"]
    idx = rng.permutation(len(pool))[:args.n_screen]
    screen_rows = [pool[i] for i in idx]
    print(f"{len(pool)} normal test radiographs available, screening {len(screen_rows)}")

    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    arch = REGISTRY[args.arch]
    cfg = AutoConfig.from_pretrained(arch.hf_id)
    proc = AutoProcessor.from_pretrained(arch.hf_id, **arch.processor_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(arch.hf_id, dtype=torch.bfloat16,
                                                        device_map=dev).eval()
    tok = getattr(proc, "tokenizer", proc)
    yi, ni = answer_ids(tok)
    img_tok = arch.image_token_id or find_image_token_id(proc, cfg)

    n = arch.n_llm_layers
    locus_names = ([s.strip() for s in args.loci.split(",") if s.strip()] or
                   ["vis.last", "connector", f"llm.L{n//4}.vis", f"llm.L{n//2}.vis", f"llm.L{(3*n)//4}.vis"])
    all_loci = {lo.name: lo for lo in loci_for(arch)}
    loci = [all_loci[k] for k in locus_names if k in all_loci]
    print(f"loci: {[l.name for l in loci]}")

    questions = [args.concept] + [c.strip() for c in args.others.split(",") if c.strip() != args.concept]
    texts = []
    for c in questions:
        m = [{"role": "user", "content": [{"type": "image"},
              {"type": "text", "text": f"Is there a {c.lower()} in this chest radiograph? Answer yes or no."}]}]
        texts.append(proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True))

    def run(images, text):
        """One pass over the planted images: P(yes) per image and the pooled activation per locus."""
        ps, acc = [], {l.name: [] for l in loci}
        for b0 in range(0, len(images), args.batch_size):
            chunk = images[b0:b0 + args.batch_size]
            inp = proc(text=[text] * len(chunk), images=chunk, return_tensors="pt", padding=True).to(dev)
            cache = ActivationCache(model, loci, image_token_id=img_tok)
            cache.set_visual_mask(inp["input_ids"])
            cache.batch_size = len(chunk)
            with cache, torch.inference_mode():
                out = model(**inp)
            a = cache.pop()
            ps.extend(p_yes_batch(out.logits, yi, ni))
            for k, v in a.items():
                acc[k].append(v.float().numpy())
        return np.array(ps), {k: np.concatenate(v, 0) for k, v in acc.items() if v}

    # Plant into radiographs the MODEL reads as normal, not merely ones the dataset labels normal. The two
    # are not the same set and the difference is not small: LLaVA-Med answers "yes, effusion" on 97% of
    # label-normal radiographs, so a planted lesion had no room left to move the answer and the cell failed
    # the dose-response gate for a reason that had nothing to do with the lesion. Screening on the model's
    # own baseline turns a ceiling artefact into a measurable range, and the screen is reported so the
    # selection is auditable rather than implicit.
    screen_imgs = [Image.open(r["image_path"]).convert("RGB") for r in screen_rows]
    base_p, _ = run(screen_imgs, texts[0])
    order = np.argsort(base_p)                       # lowest baseline P(yes) first: the most room to rise
    take = [i for i in order if base_p[i] <= args.max_base_p][:args.n_images]
    if len(take) < args.n_images // 2:
        take = list(order[:args.n_images])
        print(f"WARNING: only {int((base_p <= args.max_base_p).sum())} of {len(base_p)} screened images "
              f"have baseline P(yes) <= {args.max_base_p}; falling back to the {len(take)} lowest. "
              f"This model is saturated on this question and its dose-response has limited headroom.")
    rows_img = [screen_rows[i] for i in take]
    print(f"screened baseline P(yes): median {np.median(base_p):.3f}, "
          f"{int((base_p <= args.max_base_p).sum())}/{len(base_p)} at or below {args.max_base_p}. "
          f"Using {len(rows_img)} images, baseline median {np.median(base_p[take]):.3f}")

    originals = [Image.open(r["image_path"]).convert("RGB") for r in rows_img]
    behav, geom = [], []
    base_act = {}

    for placement, cs in (("anatomical", centres), ("control_location", CONTROL_SITE)):
        for beta in betas:
            imgs = originals if beta == 0 else [plant(im, cs, radius, sign, beta) for im in originals]
            for qi, (q, text) in enumerate(zip(questions, texts)):
                t0 = time.time()
                ps, acts = run(imgs, text)
                behav.append({"placement": placement, "beta": beta, "question": q,
                              "p_yes": round(float(ps.mean()), 5), "sd": round(float(ps.std()), 5),
                              "n": len(ps)})
                if qi == 0:                      # geometry is read off the concept's own question only
                    key = (placement, beta)
                    if beta == 0:
                        base_act[placement] = acts
                    else:
                        for lname, A in acts.items():
                            dh = A - base_act[placement][lname]
                            geom.append({"placement": placement, "beta": beta, "locus": lname,
                                         "dh_norm": round(float(np.linalg.norm(dh, axis=1).mean()), 4),
                                         "_dh": dh.mean(0)})
                print(f"  {placement:17s} beta={beta:.2f} {q:14s} P(yes)={ps.mean():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    # ---- score the candidate directions against the displacement the image actually produced ----
    act_dir = args.acts or f"runs/act_{args.arch}"
    ids, A = load_shards(act_dir)
    mman = {r["row_id"]: r for r in man}
    keep = [i for i, r in enumerate(ids) if r in mman]
    A = {k: v[keep] for k, v in A.items()}
    meta = [mman[ids[i]] for i in keep]
    tr = np.array([m["split"] for m in meta]) == "train"
    y = np.array([int(m[args.concept]) for m in meta])

    cad_dirs = {}
    if args.cad_dir:
        for f in glob.glob(os.path.join(args.cad_dir, "dirs_*.npz")):
            lname = os.path.basename(f)[5:-4].replace("_", ".")
            z = np.load(f)
            cad_dirs[lname] = {k: z[k] for k in z.files}

    # The probe normal depends only on the locus, not on the placement or the dose, so it is fitted once
    # per locus and reused. Fitting it inside the loop refitted the same logistic regression on 18k rows at
    # full width twelve times per locus, which turned a five-minute step into hours and pinned twenty cores
    # per job doing arithmetic it had already done.
    probe_cache: dict[str, np.ndarray] = {}
    rand_cache: dict[int, np.ndarray] = {}

    align = []
    for g in geom:
        lname = g["locus"]
        if lname not in A:
            continue
        dh = g.pop("_dh")
        dh_u = dh / (np.linalg.norm(dh) + 1e-9)
        if lname not in probe_cache:
            probe_cache[lname] = fit_direction(A[lname][tr].astype(np.float32), y[tr], args.seed)
        v_probe = probe_cache[lname]
        d = len(dh_u)
        if d not in rand_cache:
            rr = rng.standard_normal(d).astype(np.float32)
            rand_cache[d] = rr / np.linalg.norm(rr)
        r = rand_cache[d]
        row = {"placement": g["placement"], "beta": g["beta"], "locus": lname,
               "dh_norm": g["dh_norm"],
               "cos_probe": round(float(dh_u @ v_probe), 4),
               "cos_random": round(float(dh_u @ r), 4)}
        for tag, v in (cad_dirs.get(lname) or {}).items():
            row[f"cos_{tag}"] = round(float(dh_u @ (v / (np.linalg.norm(v) + 1e-9))), 4)
        align.append(row)

    for name, data in (("behaviour.csv", behav), ("alignment.csv", align)):
        if not data:
            continue
        cols = sorted({k for d in data for k in d}, key=lambda k: (k.startswith("cos"), k))
        with open(os.path.join(args.out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, restval="")
            w.writeheader(); w.writerows(data)
    json.dump({"arch": args.arch, "concept": args.concept, "betas": betas,
               "centres": centres, "radius": radius, "sign": sign,
               "control_site": CONTROL_SITE, "n_images": len(rows_img),
               "loci": [l.name for l in loci], "questions": questions,
               "lesion_model": "smooth radial blob added in normalised intensity, clipped to [0,1]. A "
                               "controlled perturbation with a known magnitude, not a clinically realistic "
                               "lesion.",
               "gate": "the anatomical dose-response must be monotone and must separate from the location "
                       "control before any alignment number is interpretable"},
              open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    with open(os.path.join(args.out, "DONE"), "w") as fh:
        fh.write(json.dumps({"behaviour": len(behav), "alignment": len(align),
                             "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    print(f"\nwrote {len(behav)} behaviour rows and {len(align)} alignment rows to {args.out}")


if __name__ == "__main__":
    main()
