"""Experiment D2: is the direction that best reads a concept the direction that best writes it?

A probe normal is fitted to *decode* a concept: it is the direction along which the labels are most
separable. The field then adds that same vector to the residual stream and reads the behavioural change as
evidence about what the model uses. That step assumes the read direction and the write direction are the
same object. Nothing guarantees it, and D1 already shows the two disagree about *where*: the locus of peak
decodability is almost never the locus of peak steerability.

This asks the sharper question at a single locus. Alongside the probe normal `v_probe`, fit a second unit
vector `v_cad` by gradient descent directly on the behaviour:

    maximise   mean_i [ m_c(x_i; v) - m_c(x_i; 0) ]                     move the answer to question c
    minus  L * mean_{c' != c} mean_i | m_{c'}(x_i; v) - m_{c'}(x_i; 0) |   without moving the others

where m is the yes-minus-no logit margin. The objective uses no labels at all. It only knows which question
it is steering, so `v_cad` is the direction that writes the concept into the model's answer, fitted without
ever being told which images contain the finding. That is what makes the comparison clean: `v_probe` is
built from labels and never sees behaviour, `v_cad` is built from behaviour and never sees labels.

The specificity penalty is not optional. Without it the optimiser converges on a generic "say yes"
direction, or simply degrades the representation until the model answers yes to everything, and either
would move question c strongly while meaning nothing about c. The survey found 5 of 8 targeted medical
interventions reported no concept-specificity control, which is exactly the gap that lets such a direction
be mistaken for a concept direction. `--lambda-spec 0` reproduces the uncontrolled version as an ablation,
and the two are reported side by side.

Reported per (model, concept, locus), all on held-out test images from patients disjoint from the fit:

    effect        |change in P(yes)| for question c
    spec_cost     mean |change in P(yes)| for the other questions
    auroc         AUROC of the 1-D projection x.v, i.e. how well the direction READS the concept
    cos_probe     cosine between the direction and the probe normal
    frontier      the same quantities along the interpolation between v_probe and v_cad

If the field's assumption held, `v_cad` would land near `v_probe`, decode about as well, and the frontier
would be flat. The prediction from D1 is that it does not: that writing and reading are different
directions, and that gaining one costs the other.

    python src/cad.py --acts runs/act_lingshu7b --arch lingshu7b --concept Effusion \
        --loci connector,llm.L27.vis --gpu 4 --out runs/cad_lingshu7b_effusion
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
from sklearn.metrics import roc_auc_score

from intervene import NormProbe, Steerer, fit_direction, load_shards
from loci import find_image_token_id, loci_for
from registry import REGISTRY

YES = ("yes", "Yes", "YES")
NO = ("no", "No", "NO")


def answer_token_ids(tok):
    def ids_of(words):
        out = []
        for w in words:
            for form in (w, " " + w):
                i = tok.encode(form, add_special_tokens=False)
                if len(i) == 1:
                    out.append(i[0])
        return sorted(set(out))
    yi, ni = ids_of(YES), ids_of(NO)
    if not yi or not ni:
        raise SystemExit("could not find single-token yes/no forms for this tokenizer")
    return yi, ni


def margin(logits, yi, ni):
    """Yes-minus-no logit margin at the answer position, per row. Differentiable."""
    lp = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
    return lp[:, yi].max(dim=-1).values - lp[:, ni].max(dim=-1).values


def p_yes(margins):
    return torch.sigmoid(margins)


def build_batches(proc, texts, meta, idx, dev, img_tok, batch_size):
    """One entry per (batch, question). Images are shared, so they are opened once per batch."""
    out = []
    for b0 in range(0, len(idx), batch_size):
        chunk = idx[b0:b0 + batch_size]
        imgs = [Image.open(meta[i]["image_path"]).convert("RGB") for i in chunk]
        per_q = []
        for t in texts:
            inp = proc(text=[t] * len(imgs), images=imgs, return_tensors="pt", padding=True).to(dev)
            mask = (inp["input_ids"] == img_tok) if img_tok is not None else None
            per_q.append((inp, mask))
        out.append(per_q)
    return out


def measure(model, steer, batches, yi, ni, base=None, want_margin=False):
    """P(yes), and optionally the raw margin, for every question over every batch."""
    n_q = len(batches[0])
    acc = [[] for _ in range(n_q)]
    accm = [[] for _ in range(n_q)]
    for per_q in batches:
        for q, (inp, mask) in enumerate(per_q):
            steer.mask = mask
            with torch.inference_mode():
                out = model(**inp)
            m = margin(out.logits, yi, ni).float().cpu()
            accm[q].append(m)
            acc[q].append(p_yes(m))
    p = [torch.cat(a).numpy() for a in acc]
    mg = [torch.cat(a).numpy() for a in accm]
    if base is None:
        return (p, mg) if want_margin else p
    eff = [float(np.mean(np.abs(p[q] - base[q]))) for q in range(n_q)]
    return (p, eff, mg) if want_margin else (p, eff)


def fit_cad(model, lo, batches, yi, ni, base_m, dim, dev, alpha, steps, lam, lr, seed,
            accum=4, log_every=10):
    """Gradient descent on a unit vector, maximising the effect on question 0 minus the effect on the rest.

    The objective is written in P(yes), not in the logit margin, and that is a correction rather than a
    presentation choice. On the margin the optimiser has an unbounded axis to run along: a first attempt
    reached a mean margin shift of 60 logits at `llm.L27.ans`, which is not a measurement of anything, it is
    the answer distribution being destroyed. P(yes) is bounded in [0, 1], it is the quantity the paper
    reports, and its gradient dies exactly where the reading stops being informative.

    The specificity weight matters as much as the objective. At lambda = 1 the optimiser converged on a
    direction that raised *every* question's margin almost equally (gain 35.5, specificity cost 35.0),
    which is a generic "answer yes" direction and says nothing about the concept. It survived because
    gain minus cost is near zero for such a direction rather than strongly negative. A weight above one
    makes non-specific directions actively costly.
    """
    # The step size has to be read against the norm of what is being optimised. `u` is a unit vector, and
    # Adam's update is roughly lr per coordinate, so its norm is about lr * sqrt(D): at lr = 0.05 in 3584
    # dimensions that is 3.0, three times the length of the vector itself. In that regime the update is
    # sign(grad) and nothing else survives, and it showed: runs with the specificity weight at 3.0 and at
    # 0.0 followed byte-identical trajectories, because the penalty never flipped enough coordinate signs
    # to change the step. lr is therefore expressed as a target update norm and divided down by sqrt(D).
    g = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.randn(dim, generator=g).to(dev).float()
    u = (u / u.norm()).requires_grad_(True)
    step_lr = lr / np.sqrt(dim)
    opt = torch.optim.Adam([u], lr=step_lr)
    steer = Steerer(model, lo.module, positions=lo.positions, mode="reltoken")
    steer.alpha = alpha
    hist = []
    order = list(range(len(batches)))
    with steer:
        for step in range(steps):
            # Accumulate over several batches per step. With a single batch of two images the gradient was
            # noisy enough that the objective moved non-monotonically over forty steps, which is not a
            # signal about the loss surface, only about the sample.
            sel = [order[(step * accum + k) % len(order)] for k in range(min(accum, len(order)))]
            steer.vec = u / u.norm()
            gains, costs = [], []
            for bi in sel:
                per_q = batches[bi]
                for q, (inp, mask) in enumerate(per_q):
                    steer.mask = mask
                    out = model(**inp)
                    m = p_yes(margin(out.logits, yi, ni))
                    d = m - base_m[q][bi].to(m.device)
                    (gains if q == 0 else costs).append(d.mean() if q == 0 else d.abs().mean())
            gain = torch.stack(gains).mean()
            cost_t = torch.stack(costs).mean() if costs else None
            loss = -gain + (lam * cost_t if cost_t is not None else 0.0)

            # How aligned are the two objectives? If pushing the concept question necessarily pushes the
            # others, no specificity weight can separate them, and that is a property of the model rather
            # than a failure of the fit. Measuring it costs one extra backward every log_every steps.
            align = float("nan")
            if cost_t is not None and (step % log_every == 0 or step == steps - 1):
                gg = torch.autograd.grad(gain, u, retain_graph=True)[0]
                gc = torch.autograd.grad(cost_t, u, retain_graph=True)[0]
                align = float(torch.nn.functional.cosine_similarity(gg, gc, dim=0))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(u.grad.norm())
            opt.step()
            with torch.no_grad():
                u.div_(u.norm() + 1e-8)
            hist.append({"step": step, "gain": float(gain.detach()),
                         "cost": float(cost_t.detach()) if cost_t is not None else 0.0,
                         "loss": float(loss.detach()), "grad_norm": gnorm, "grad_align": align})
            if step % log_every == 0 or step == steps - 1:
                h = hist[-1]
                print(f"    step {step:4d}  gain {h['gain']:+.4f}  spec_cost {h['cost']:.4f}  "
                      f"loss {h['loss']:+.4f}  |g| {gnorm:.2e}  cos(grad_gain,grad_cost) {align:+.3f}",
                      flush=True)
    v = (u.detach() / u.detach().norm()).float().cpu().numpy()
    return v, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--arch", required=True)
    ap.add_argument("--concept", default="Effusion")
    ap.add_argument("--loci", default="connector,llm.L27.ans")
    ap.add_argument("--others", default="Cardiomegaly,Pneumothorax",
                    help="questions the direction must leave alone")
    ap.add_argument("--alpha", type=float, default=0.25,
                    help="fraction of the per-token activation norm. The answer position is far "
                         "more sensitive than the visual positions, so 0.5 there already saturates")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--accum", type=int, default=4, help="batches accumulated per update")
    ap.add_argument("--lr", type=float, default=0.08,
                    help="target NORM of each update, not a per-coordinate rate; divided by "
                         "sqrt(D) internally so it is comparable across loci of different width")
    ap.add_argument("--lambda-spec", type=float, default=3.0,
                    help="weight on the specificity penalty. Must exceed 1: at 1.0 a direction "
                         "that moves every question equally costs nothing")
    ap.add_argument("--n-fit", type=int, default=96, help="train images the direction is fitted on")
    ap.add_argument("--n-eval", type=int, default=96, help="held-out test images it is scored on")
    ap.add_argument("--screen", type=int, default=1,
                    help="select evaluation images by the model's own baseline, so a saturated answer "
                         "does not read as an unsteerable representation")
    ap.add_argument("--n-screen", type=int, default=192)
    ap.add_argument("--screen-band", type=float, default=0.45,
                    help="keep images whose baseline P(yes) is within this of 0.5")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--frontier", type=int, default=5, help="interpolation points between probe and cad")
    ap.add_argument("--nospec-loci", default="first", choices=["first", "all"],
                    help="where to also fit the unpenalised ablation arm")
    ap.add_argument("--checkpoint", type=int, default=1, help="gradient checkpointing, 1 or 0")
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    bind_gpu(args.gpu)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    dev = "cuda:0"

    ids, acts = load_shards(args.acts)
    man = {r["row_id"]: r for r in csv.DictReader(open(args.manifest))}
    keep = [i for i, r in enumerate(ids) if r in man]
    ids = [ids[i] for i in keep]
    acts = {k: v[keep] for k, v in acts.items()}
    meta = [man[r] for r in ids]
    split = np.array([m["split"] for m in meta])
    y = np.array([int(m[args.concept]) for m in meta])
    tr = split == "train"

    tr_idx = np.where(tr)[0]; rng.shuffle(tr_idx); tr_idx = tr_idx[:args.n_fit]
    te_pool = np.where(split == "test")[0]; rng.shuffle(te_pool)
    te_idx = te_pool[:args.n_eval]                 # replaced below by a baseline-screened selection
    print(f"{tr.sum()} train rows for the probe | {len(tr_idx)} images to fit cad | "
          f"{len(te_idx)} held-out images to score")

    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    arch = REGISTRY[args.arch]
    cfg = AutoConfig.from_pretrained(arch.hf_id)
    proc = AutoProcessor.from_pretrained(arch.hf_id, **arch.processor_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(arch.hf_id, dtype=torch.bfloat16,
                                                        device_map=dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # Backward from an early locus traverses the whole language model, and on a shared card there is not
    # room to keep every intermediate. Checkpointing trades about a third more compute for a large drop in
    # activation memory. use_reentrant=False is required here because the only tensor that needs a gradient
    # enters partway through the network, through a forward hook, rather than at the inputs.
    if args.checkpoint:
        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            # Transformers guards every checkpointed block with `if self.gradient_checkpointing and
            # self.training`, so calling enable() on a model in eval mode silently does nothing: the flag
            # is set, the message prints, and the full activation graph is kept anyway. That is how three
            # jobs died of out-of-memory while reporting that checkpointing was on. The model therefore has
            # to be in train mode, with every dropout put back into eval by hand so the forward pass stays
            # deterministic. These architectures carry no batch norm, so nothing else changes with the mode.
            model.train()
            n_drop = 0
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.eval(); n_drop += 1
            print(f"gradient checkpointing on (train mode, {n_drop} dropout modules held in eval)")
        except Exception as e:
            print(f"gradient checkpointing unavailable: {type(e).__name__}: {e}")
        args._verify_determinism = True
    tok = getattr(proc, "tokenizer", proc)
    yi, ni = answer_token_ids(tok)
    img_tok = arch.image_token_id or find_image_token_id(proc, cfg)
    loci_all = {lo.name: lo for lo in loci_for(arch)}

    others = [c.strip() for c in args.others.split(",") if c.strip() and c.strip() != args.concept]
    questions = [args.concept] + others
    texts = []
    for c in questions:
        m = [{"role": "user", "content": [{"type": "image"},
              {"type": "text", "text": f"Is there a {c.lower()} in this chest radiograph? Answer yes or no."}]}]
        texts.append(proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True))
    print(f"steering question: {questions[0]!r} | held fixed: {others}")

    # Putting the model in train mode to make checkpointing take effect is only safe if the forward pass is
    # still deterministic. Rather than trust that these architectures carry no stochastic layer, check it:
    # two identical passes must give identical logits, or every measurement below is contaminated by noise
    # that would be read as an effect.
    if getattr(args, "_verify_determinism", False):
        _m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "test"}]}]
        _t = proc.apply_chat_template(_m, tokenize=False, add_generation_prompt=True)
        _im = Image.open(man[ids[int(np.where(tr)[0][0])]]["image_path"]).convert("RGB")
        _in = proc(text=[_t], images=[_im], return_tensors="pt", padding=True).to(dev)
        with torch.inference_mode():
            _a = model(**_in).logits.float().clone()
            _b = model(**_in).logits.float()
        _d = float((_a - _b).abs().max())
        print(f"determinism check: max abs logit difference over two identical passes = {_d:.3e}")
        if _d > 1e-4:
            raise SystemExit(f"forward pass is not deterministic in train mode (max diff {_d:.3e}). "
                             f"Rerun with --checkpoint 0 and a larger card.")

    # A model whose answer is already pinned has no room for an intervention to show up in P(yes), and
    # measuring it there reports a ceiling as a property of the representation. LLaVA-Med answers "yes,
    # effusion" with median probability 0.967 on label-normal radiographs, leaving 0.033 of headroom, while
    # Qwen2.5-VL sits at 0.181. The evaluation set is therefore screened on the model's own baseline, the
    # same way src/plant.py screens, and the screen is printed so the selection is visible.
    if args.screen:
        scr = te_pool[:args.n_screen]
        scr_batches = build_batches(proc, texts[:1], meta, scr, dev, img_tok, args.batch_size)
        bp = []
        for per_q in scr_batches:
            inp, _ = per_q[0]
            with torch.inference_mode():
                bp.append(p_yes(margin(model(**inp).logits, yi, ni)).float().cpu())
        bp = torch.cat(bp).numpy()
        order = np.argsort(np.abs(bp - 0.5))        # closest to an undecided answer first
        keep_i = [int(i) for i in order if abs(bp[i] - 0.5) <= args.screen_band]
        chosen = keep_i[:args.n_eval] if len(keep_i) >= args.n_eval // 2 else [int(i) for i in order[:args.n_eval]]
        te_idx = scr[chosen]
        print(f"baseline screen: median P(yes) {np.median(bp):.3f}, "
              f"{int((np.abs(bp - 0.5) <= args.screen_band).sum())}/{len(bp)} within "
              f"{args.screen_band} of 0.5; using {len(te_idx)}, baseline median {np.median(bp[chosen]):.3f}")

    rows, frontier_rows, dose_rows, hists = [], [], [], {}
    locus_list = [s.strip() for s in args.loci.split(",") if s.strip()]
    for locus_name in locus_list:
        if locus_name not in acts or locus_name not in loci_all:
            print(f"skip {locus_name}: absent from activations or locus set")
            continue
        lo = loci_all[locus_name]
        X = acts[locus_name].astype(np.float32)
        D = X.shape[1]
        v_probe = fit_direction(X[tr], y[tr], args.seed)

        fit_batches = build_batches(proc, texts, meta, tr_idx, dev, img_tok, args.batch_size)
        eval_batches = build_batches(proc, texts, meta, te_idx, dev, img_tok, args.batch_size)

        # Unsteered reference, per batch, kept on device-side cpu so the fit can subtract it cheaply.
        steer0 = Steerer(model, lo.module, positions=lo.positions, mode="reltoken")
        steer0.vec = None
        base_fit = [[None] * len(questions) for _ in fit_batches]
        with steer0:
            for b, per_q in enumerate(fit_batches):
                for q, (inp, mask) in enumerate(per_q):
                    steer0.mask = mask
                    with torch.inference_mode():
                        base_fit[b][q] = p_yes(margin(model(**inp).logits, yi, ni)).float().cpu()
            base_eval, base_margin = measure(model, steer0, eval_batches, yi, ni, want_margin=True)
        base_m = [[base_fit[b][q] for b in range(len(fit_batches))] for q in range(len(questions))]

        probe_n = NormProbe(model, lo.module, positions=lo.positions)
        with probe_n:
            for per_q in eval_batches[:2]:
                probe_n.mask = per_q[0][1]
                with torch.inference_mode():
                    model(**per_q[0][0])
        nstat = probe_n.stats()
        print(f"\n{locus_name}: D={D}  per-token ||h|| med {nstat['median']:.1f} "
              f"mean {nstat['mean']:.1f}  alpha={args.alpha} of it")

        # ---- candidate directions ----
        cands = {"probe": v_probe}
        t0 = time.time()
        # The unpenalised arm is an ablation: it exists to show what the same optimiser finds when nobody
        # checks specificity. One locus demonstrates that, and running it at every locus doubled the cost
        # of the whole experiment for a point already made.
        fits = [("cad", args.lambda_spec)]
        if args.nospec_loci in ("all", locus_name) or (
                args.nospec_loci == "first" and locus_name == locus_list[0]):
            fits.append(("cad_nospec", 0.0))
        for tag, lam in fits:
            print(f"  fitting {tag} (lambda_spec={lam})")
            v, hist = fit_cad(model, lo, fit_batches, yi, ni, base_m, D, dev,
                              args.alpha, args.steps, lam, args.lr, args.seed, accum=args.accum)
            cands[tag] = v
            hists[f"{locus_name}/{tag}"] = hist
        r = rng.standard_normal(D).astype(np.float32); cands["random"] = r / np.linalg.norm(r)
        sh = v_probe[rng.permutation(D)]; cands["sham"] = sh / np.linalg.norm(sh)
        print(f"  directions ready in {time.time() - t0:.0f}s")

        # ---- score every candidate on held-out images, and on decoding ----
        te_mask = split == "test"
        Xte, yte = X[te_mask], y[te_mask]
        steer = Steerer(model, lo.module, positions=lo.positions, mode="reltoken")
        steer.alpha = args.alpha
        with steer:
            for tag, v in cands.items():
                steer.vec = v
                p, eff, mg = measure(model, steer, eval_batches, yi, ni, base_eval, want_margin=True)
                proj = Xte @ v
                auroc = float(roc_auc_score(yte, proj)) if len(np.unique(yte)) > 1 else float("nan")
                row = {"locus": locus_name, "direction": tag, "alpha": args.alpha,
                       "effect": round(eff[0], 5),
                       "spec_cost": round(float(np.mean(eff[1:])) if len(eff) > 1 else 0.0, 5),
                       "auroc": round(max(auroc, 1 - auroc), 4),
                       "auroc_signed": round(auroc, 4),
                       "cos_probe": round(float(np.dot(v, v_probe)), 4),
                       "p_yes_base": round(float(np.mean(base_eval[0])), 5),
                       "p_yes_steered": round(float(np.mean(p[0])), 5),
                       "margin_effect": round(float(np.mean(np.abs(mg[0] - base_margin[0]))), 5),
                       "margin_base": round(float(np.mean(base_margin[0])), 5),
                       "margin_spec_cost": round(float(np.mean([np.mean(np.abs(mg[q] - base_margin[q]))
                                                                for q in range(1, len(mg))])), 5)
                                           if len(mg) > 1 else 0.0,
                       "token_norm": round(nstat["median"], 2), "dim": D, "n_eval": len(te_idx)}
                if tag == "cad":
                    al = [h["grad_align"] for h in hists[f"{locus_name}/cad"]
                          if h["grad_align"] == h["grad_align"]]
                    # The SIGN carries the meaning and must not be averaged away. +1 means pushing the
                    # target finding also pushes the others, a conflict that no specificity weight can
                    # resolve. -1 means pushing the target RELIEVES the others, which is synergy, the
                    # opposite situation. Reporting the mean of |cos| collapsed the two: at the final
                    # answer position the per-step sequence is [-1, +1, -1, -1, +1], flipping every step,
                    # and averaging magnitudes turned it into a flat 0.9999 that was then read as
                    # "inseparable". What that locus actually shows is a ONE-DIMENSIONAL causal axis:
                    # the alignment is +-1 in magnitude because the two gradients are collinear, but
                    # whether they conflict or cooperate changes from step to step.
                    a = np.asarray(al, dtype=float)
                    row["grad_align_signed_mean"] = round(float(a.mean()), 4) if al else float("nan")
                    row["grad_collinearity"] = round(float(np.abs(a).mean()), 4) if al else float("nan")
                    row["grad_align_sign_flips"] = int((np.diff(np.sign(a)) != 0).sum()) if len(a) > 1 else 0
                    row["grad_align_frac_positive"] = round(float((a > 0).mean()), 4) if al else float("nan")
                    row["grad_align_mean"] = row["grad_collinearity"]   # kept so old readers do not break
                rows.append(row)
                print(f"    {tag:11s} effect {row['effect']:.4f}  spec_cost {row['spec_cost']:.4f}  "
                      f"read AUROC {row['auroc']:.3f}  cos(probe) {row['cos_probe']:+.3f}")

            # ---- dose response, so a ceiling shows up as a ceiling ----
            for tag in ("probe", "cad"):
                for a in (0.1, 0.25, 0.5):
                    steer.vec = cands[tag]; steer.alpha = a
                    p, eff = measure(model, steer, eval_batches, yi, ni, base_eval)
                    dose_rows.append({"locus": locus_name, "direction": tag, "alpha": a,
                                      "effect": round(eff[0], 5),
                                      "spec_cost": round(float(np.mean(eff[1:])) if len(eff) > 1 else 0.0, 5),
                                      "p_yes_steered": round(float(np.mean(p[0])), 5)})
            steer.alpha = args.alpha

            # ---- the frontier between reading and writing ----
            if args.frontier > 1:
                print("  frontier probe -> cad")
                for k in range(args.frontier):
                    t = k / (args.frontier - 1)
                    v = (1 - t) * v_probe + t * cands["cad"]
                    v = v / (np.linalg.norm(v) + 1e-8)
                    steer.vec = v
                    p, eff = measure(model, steer, eval_batches, yi, ni, base_eval)
                    a = float(roc_auc_score(yte, Xte @ v)) if len(np.unique(yte)) > 1 else float("nan")
                    frontier_rows.append({"locus": locus_name, "t": round(t, 3),
                                          "effect": round(eff[0], 5),
                                          "spec_cost": round(float(np.mean(eff[1:])) if len(eff) > 1 else 0.0, 5),
                                          "auroc": round(max(a, 1 - a), 4),
                                          "cos_probe": round(float(np.dot(v, v_probe)), 4)})
                    print(f"    t={t:.2f}  effect {eff[0]:.4f}  read AUROC {max(a, 1-a):.3f}")

        for name, data in (("cad_results.csv", rows), ("cad_frontier.csv", frontier_rows),
                           ("cad_dose.csv", dose_rows)):
            if not data:
                continue
            # Union of keys, not the first row's: only the cad rows carry the gradient-alignment
            # columns, so keying off row zero silently dropped the measurement and then raised.
            cols, seen = [], set()
            for d in data:
                for k in d:
                    if k not in seen:
                        seen.add(k); cols.append(k)
            tmp = os.path.join(args.out, name + ".tmp")
            with open(tmp, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols, restval="")
                w.writeheader(); w.writerows(data)
                fh.flush(); os.fsync(fh.fileno())
            os.replace(tmp, os.path.join(args.out, name))
        np.savez_compressed(os.path.join(args.out, f"dirs_{locus_name.replace('.', '_')}.npz"), **cands)

    json.dump({"arch": args.arch, "concept": args.concept, "others": others, "alpha": args.alpha,
               "steps": args.steps, "lr": args.lr, "lambda_spec": args.lambda_spec,
               "n_fit": len(tr_idx), "n_eval": len(te_idx), "seed": args.seed,
               "objective": "maximise the change in P(yes) for the concept question, penalised by "
                            "lambda times the mean absolute change in P(yes) for the other questions. "
                            "P(yes) rather than the logit margin, because the margin is unbounded and the "
                            "optimiser ran to 60 logits, which destroys the answer distribution instead of "
                            "measuring it. No labels enter the objective.",
               "history": hists},
              open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    with open(os.path.join(args.out, "DONE"), "w") as fh:
        fh.write(json.dumps({"rows": len(rows), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    print(f"\nwrote {len(rows)} rows to {args.out}/cad_results.csv")


if __name__ == "__main__":
    main()
