"""Experiment D1: does the model use what it encodes?

Take the probe normal `v_c` for a concept at a locus, add `alpha * v_c` to the residual stream at that
locus during generation, and measure whether the model's answer moves. A change on its own means
nothing, because a random direction of the same norm also moves the output. The result is only
interpretable against matched controls, and the survey found only 3 of 8 targeted medical interventions
reported one, so the control battery here is not optional.

Controls, all at the same norm as `v_c`:
  random        `n_random` isotropic directions. The null band
  unrelated     probe normals for other concepts. Tests concept specificity, not just direction
                specificity
  sham          a coordinate permutation of `v_c`. Same norm, same marginal distribution, no structure
  sign          -alpha as well as +alpha. A direction that only ever increases the answer regardless of
                sign is measuring a norm effect

Magnitudes are fractions of the median per-token activation norm at the steered module, measured by a
calibration pass rather than inferred from the pooled activations, so that a given alpha means the same
relative perturbation at every locus.

The statistic: `v_c` counts as selective at a locus only if its effect exceeds the 95th percentile of
the random-direction effects, on held-out images disjoint from those that fitted the probe.

    python src/intervene.py --acts runs/act_lingshu7b --manifest data/manifest.csv \
        --model lingshu-medical-mllm/Lingshu-7B --concept Effusion \
        --loci connector,llm.L13.vis,llm.L20.vis --gpu 6 --out runs/int_lingshu_effusion
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from loci import as_hidden, find_image_token_id, loci_for
from registry import REGISTRY

YES_WORDS = ("yes", "Yes", "YES")
NO_WORDS = ("no", "No", "NO")
REGISTERED_UNRELATED = ("Atelectasis", "Pneumothorax", "Cardiomegaly", "Mass", "Nodule")
REGISTERED_PROMPT = "Is there a pleural effusion in this chest radiograph? Answer yes or no."


def load_shards(act_dir):
    ids, per = [], {}
    for f in sorted(glob.glob(os.path.join(act_dir, "shard*.npz"))):
        z = np.load(f, allow_pickle=True)
        ids.extend(str(x) for x in z["row_id"])
        for k in z.files:
            if k != "row_id":
                per.setdefault(k, []).append(z[k])
    return ids, {k: np.concatenate(v, 0) for k, v in per.items()}


def fit_direction(X, y, seed=0):
    """`v_c` is the probe normal in the standardised space, mapped back and unit-normalised."""
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(sc.transform(X), y)
    v = clf.coef_[0] / np.maximum(sc.scale_, 1e-8)
    return v / (np.linalg.norm(v) + 1e-8)


def load_registered_directions(path, expected_dim, concept):
    """Load the capacity-matched probe normals produced by the registered first-gate probe."""
    with np.load(path, allow_pickle=False) as bundle:
        required = {
            "names", "vectors", "projection", "scale", "coefficients", "locus", "raw_dim",
            "projection_dim", "projection_seed", "C", "probe_seed",
        }
        if set(bundle.files) != required:
            raise ValueError(f"direction bundle fields mismatch: {sorted(bundle.files)}")
        names = [str(value) for value in bundle["names"]]
        vectors = np.asarray(bundle["vectors"], dtype=np.float32)
        projection = np.asarray(bundle["projection"], dtype=np.float64)
        scale = np.asarray(bundle["scale"], dtype=np.float64)
        coefficients = np.asarray(bundle["coefficients"], dtype=np.float64)
        expected_names = [concept, *REGISTERED_UNRELATED]
        if names != expected_names or str(bundle["locus"]) != "vis.last":
            raise ValueError("direction bundle identity mismatch")
        if int(bundle["raw_dim"]) != expected_dim or vectors.shape != (len(names), expected_dim):
            raise ValueError("direction bundle activation dimension mismatch")
        if projection.shape != (expected_dim, 512) or scale.shape != (512,):
            raise ValueError("direction bundle projection or scale shape mismatch")
        if coefficients.shape != (len(names), 512):
            raise ValueError("direction bundle coefficient shape mismatch")
        if int(bundle["projection_dim"]) != 512 or int(bundle["projection_seed"]) != 0:
            raise ValueError("direction bundle projection contract mismatch")
        if float(bundle["C"]) != 1.0 or int(bundle["probe_seed"]) != 0:
            raise ValueError("direction bundle readout contract mismatch")
    reconstructed = projection @ (coefficients / np.maximum(scale, 1e-8)).T
    reconstructed /= np.linalg.norm(reconstructed, axis=0, keepdims=True)
    reconstructed = reconstructed.T.astype(np.float32)
    if not np.allclose(vectors, reconstructed, rtol=1e-5, atol=1e-6):
        raise ValueError("direction vectors do not match the registered readout artifacts")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(vectors)) or not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("direction bundle vectors are not finite unit normals")
    with open(path, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(zip(names, vectors)), digest


def yes_margin_batch(logits, tok):
    """The yes-minus-no log-probability margin at the last prompt position, per row.

    Reported alongside P(yes) because the two are not interchangeable across models. P(yes) is a squashed
    coordinate, so the same causal push reads as a large or a tiny number depending on where the model
    started. Measured over 96 label-normal radiographs, the median baseline P(yes) for effusion is 0.181 on
    Qwen2.5-VL and 0.967 on LLaVA-Med: the second has 0.033 of room to move upward and the first has 0.819.
    Comparing their raw effect sizes in P(yes) would report a ceiling as a property of the representation.
    The margin has no ceiling, so it is the coordinate any cross-model comparison has to use.
    """
    lp = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
    def ids_of(words):
        out = []
        for w in words:
            for form in (w, " " + w):
                i = tok.encode(form, add_special_tokens=False)
                if len(i) == 1:
                    out.append(i[0])
        return sorted(set(out))
    yi, ni = ids_of(YES_WORDS), ids_of(NO_WORDS)
    if not yi or not ni:
        return [float("nan")] * lp.shape[0]
    return (lp[:, yi].max(dim=-1).values - lp[:, ni].max(dim=-1).values).tolist()


def yes_prob_batch(logits, tok):
    """P(yes) against P(no) at the last prompt position, for every row in the batch."""
    lp = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
    def ids_of(words):
        out = []
        for w in words:
            for form in (w, " " + w):
                i = tok.encode(form, add_special_tokens=False)
                if len(i) == 1:
                    out.append(i[0])
        return sorted(set(out))
    yi, ni = ids_of(YES_WORDS), ids_of(NO_WORDS)
    if not yi or not ni:
        return [float("nan")] * lp.shape[0]
    y = lp[:, yi].max(dim=-1).values
    n = lp[:, ni].max(dim=-1).values
    return torch.sigmoid(y - n).tolist()


def yes_prob(logits, tok):
    """P(yes) against P(no) at the first generated position, which is the behavioural end-point."""
    lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
    def best(words):
        vals = []
        for w in words:
            for form in (w, " " + w):
                ids = tok.encode(form, add_special_tokens=False)
                if len(ids) == 1:
                    vals.append(lp[ids[0]].item())
        return max(vals) if vals else float("-inf")
    y, n = best(YES_WORDS), best(NO_WORDS)
    m = max(y, n)
    return float(np.exp(y - m) / (np.exp(y - m) + np.exp(n - m)))


class Steerer:
    """Add `alpha * ||h_t|| * v` to one module's output, token by token.

    The gain is set per token rather than once per locus, and that choice is forced by the data. A single
    scalar magnitude assumes the tokens at a locus have comparable norms. In the vision tower they do not:
    on Qwen2.5-VL the median patch norm at the last block is 2171 while the mean is 5261, because a small
    number of very high norm patches carries most of the mass. Inside the language model the two agree to
    within a few percent. A fixed-norm perturbation therefore lands as a large relative change on a typical
    patch and a negligible one on the high-norm patches, and how badly it is mismatched varies from stage to
    stage. Since the paper's claim is a comparison across stages, that mismatch would sit directly on the
    axis of interest.

    Scaling each token by its own norm makes `alpha` mean the same relative perturbation at every token and
    every locus. The perturbation stays rank one along `v`; only its gain adapts. Every control direction is
    treated identically, so the comparison against the random band is unaffected.

    `relnorm` keeps the older behaviour, a single median-based magnitude, as a robustness check.
    """

    def __init__(self, model, module_path, positions="visual", mode="reltoken", global_scale=1.0):
        self.mod = dict(model.named_modules())[module_path]
        self.positions = positions
        self.mode = mode
        self.global_scale = global_scale
        self.vec = None                                        # unit direction, or None for no steering
        self.alpha = 0.0
        self.mask = None
        self.handle = None

    def _delta(self, h):
        """The perturbation tensor, broadcastable to h."""
        v = torch.as_tensor(self.vec, dtype=h.dtype, device=h.device)
        if self.mode == "reltoken":
            # No detach. `h` is the steered module's own output, so it does not depend on the vector being
            # added to it and detaching changes no gradient. It did change something else: calling detach
            # inside a block wrapped by non-reentrant gradient checkpointing made the forward and the
            # recomputation pack a different number of saved tensors, and every D2 job died with
            # "A different number of tensors was saved during the original forward and recomputation,
            # forward: 33, recomputation: 32" the moment it moved from the connector, which sits outside
            # the checkpointed region, to a decoder layer, which sits inside it.
            gain = self.alpha * h.norm(dim=-1, keepdim=True)
        else:
            gain = torch.as_tensor(self.alpha * self.global_scale, dtype=h.dtype, device=h.device)
        return gain * v

    def __enter__(self):
        def hook(_m, _i, out):
            if self.vec is None or self.alpha == 0:
                return out
            tup = isinstance(out, tuple)
            h = as_hidden(out)
            if h is None:
                return out
            if h.dim() == 2:                                   # merger output, (T, D)
                h = h + self._delta(h)
            elif self.positions == "answer":
                h = h.clone()
                h[:, -1, :] = h[:, -1, :] + self._delta(h[:, -1, :])
            elif self.positions == "visual" and self.mask is not None and self.mask.shape[1] == h.shape[1]:
                h = h + self._delta(h) * self.mask.to(h.device).unsqueeze(-1).to(h.dtype)
            else:
                h = h + self._delta(h)
            return (h,) + out[1:] if tup else h
        # Transformers 5.x installs a persistent hidden-state recorder on CLIP blocks the first time
        # output_hidden_states is requested. LLaVA selects vision_feature_layer from that recorder, so
        # a later ordinary hook changes the next block but not the tensor sent to the connector. Prepend
        # the intervention so both the recorder and the downstream block observe the steered output.
        self.handle = self.mod.register_forward_hook(hook, prepend=True)
        return self

    def __exit__(self, *exc):
        if self.handle:
            self.handle.remove()
        return False


class NormProbe:
    """Measure the norm of the activation the steering vector is actually added to.

    The magnitude of an intervention only means something relative to what it perturbs. The obvious
    reference, the norm of the pooled activation stored by extract.py, is the wrong one: pooling averages
    over visual tokens, and the amount a mean shrinks a norm depends on how correlated the summands are.
    Token correlation rises with depth, so the pooled norm understates the per-token norm by a factor that
    itself varies systematically along the very axis the paper compares across. Scaling alpha by it would
    make deep loci receive relatively weaker perturbations than shallow ones and leave the depth profile of
    the effect uninterpretable.

    This hooks the same module the Steerer hooks, over the same positions, and reports the median norm of
    the tensor the vector is added to. The ratio to the pooled norm is recorded per locus, both to document
    the correction and because that ratio is a direct read-out of how token-redundant a stage is.
    """

    def __init__(self, model, module_path, positions="visual"):
        self.mod = dict(model.named_modules())[module_path]
        self.positions = positions
        self.mask = None
        self.norms = []
        self.handle = None

    def __enter__(self):
        def hook(_m, _i, out):
            h = as_hidden(out)
            if h is None:
                return out
            h = h.detach().float()
            if h.dim() == 2:                                    # merger output, already (T, D)
                n = h.norm(dim=-1)
            elif self.positions == "answer":
                n = h[:, -1, :].norm(dim=-1)
            elif self.positions == "visual" and self.mask is not None and self.mask.shape[1] == h.shape[1]:
                n = h[self.mask.to(h.device)].norm(dim=-1)
            else:
                n = h.norm(dim=-1)
            self.norms.append(n.flatten().cpu())
            return out
        self.handle = self.mod.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self.handle:
            self.handle.remove()
        return False

    def stats(self):
        import torch as _t
        if not self.norms:
            return {"median": float("nan"), "mean": float("nan"), "p95": float("nan"), "n": 0}
        a = _t.cat(self.norms)
        return {"median": float(a.median()), "mean": float(a.mean()),
                "p95": float(a.quantile(0.95)), "n": int(a.numel())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--arch", required=True, help="key in src/registry.py")
    ap.add_argument("--concept", default="Effusion")
    ap.add_argument("--loci", default="connector,llm.L6.vis,llm.L13.vis,llm.L20.vis,llm.L27.vis")
    ap.add_argument("--alphas", default="-1.0,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1.0",
                    help="perturbation norm as a FRACTION of the locus median activation norm, both signs")
    ap.add_argument("--alpha-mode", default="reltoken", choices=["reltoken", "relnorm"],
                    help="reltoken: each token is perturbed by alpha times its own norm. relnorm: one "
                         "magnitude per locus, alpha times the median per-token norm")
    ap.add_argument("--control-alphas", default="-1.0,-0.5,-0.25,0.25,0.5,1.0",
                    help="magnitudes at which the CONTROL directions are run. The concept direction gets "
                         "the full sweep because its dose-response is a result; the controls only have to "
                         "define the null band at the magnitudes where a comparison is made, and running "
                         "25 of them at every magnitude spends most of the budget on points no statistic "
                         "reads")
    ap.add_argument("--n-random", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=200, help="held-out images per locus")
    ap.add_argument("--eval-split", default="test", choices=["validation", "test"],
                    help="evaluation split; the registered decisive run uses test, smoke runs use validation")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--model-path", default="",
                    help="verified local model snapshot; required by immutable registered runners")
    ap.add_argument("--directions", default="",
                    help="capacity-matched direction bundle emitted by the registered probe")
    ap.add_argument("--prompt", default="",
                    help="fixed behavior prompt; the registered runner supplies the preregistered text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
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

    # Directions are fitted on train. Evaluation images come from test, so the images that shaped the
    # direction never appear in the measurement.
    tr = split == "train"
    eval_name = "val" if args.eval_split == "validation" else "test"
    te_idx = np.where(split == eval_name)[0]
    rng.shuffle(te_idx)
    te_idx = te_idx[:args.n_eval]
    if len(te_idx) != args.n_eval:
        raise SystemExit(f"requested {args.n_eval} {eval_name} images, found {len(te_idx)}")
    eval_row_ids = [ids[i] for i in te_idx]
    print(f"{tr.sum()} train rows for directions | {len(te_idx)} held-out {eval_name} images")

    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    arch = REGISTRY[args.arch]
    model_source = args.model_path or arch.hf_id
    local_only = bool(args.model_path)
    cfg = AutoConfig.from_pretrained(model_source, local_files_only=local_only)
    proc = AutoProcessor.from_pretrained(
        model_source, local_files_only=local_only, **arch.processor_kwargs
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_source, dtype=torch.bfloat16, device_map=dev, local_files_only=local_only
    ).eval()
    tok = getattr(proc, "tokenizer", proc)
    img_tok = arch.image_token_id or find_image_token_id(proc, cfg)
    loci_all = {lo.name: lo for lo in loci_for(arch)}

    prompt = args.prompt or (
        REGISTERED_PROMPT if args.concept == "Effusion"
        else f"Is there {args.concept.lower()} in this chest radiograph? Answer yes or no."
    )
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    other_concepts = [c for c in REGISTERED_UNRELATED if c != args.concept]
    alphas = [float(a) for a in args.alphas.split(",")]
    ctrl_alphas = sorted({float(a) for a in args.control_alphas.split(",") if a.strip()} & set(alphas))
    if not ctrl_alphas:
        raise SystemExit("control alphas must be a subset of --alphas, otherwise nothing is comparable")
    rows = []

    for locus_name in [s.strip() for s in args.loci.split(",") if s.strip()]:
        if locus_name not in acts or locus_name not in loci_all:
            print(f"skip {locus_name}: not in activations or locus set")
            continue
        lo = loci_all[locus_name]
        X = acts[locus_name].astype(np.float32)
        pooled_norm = float(np.median(np.linalg.norm(X, axis=1)))
        bundle_sha256 = ""
        if args.directions:
            if locus_name != "vis.last":
                raise ValueError("registered direction bundle is only valid for vis.last")
            registered, bundle_sha256 = load_registered_directions(
                args.directions, X.shape[1], args.concept
            )
            v_c = registered[args.concept]
        else:
            registered = None
            v_c = fit_direction(X[tr], y[tr], args.seed)

        directions = {"concept": v_c}
        for j in range(args.n_random):
            r = rng.standard_normal(X.shape[1]).astype(np.float32)
            directions[f"random{j}"] = r / np.linalg.norm(r)
        for oc in other_concepts:
            if registered is not None:
                directions[f"unrelated_{oc}"] = registered[oc]
            else:
                yo = np.array([int(m[oc]) for m in meta])
                if yo[tr].sum() <= 50:
                    continue
                directions[f"unrelated_{oc}"] = fit_direction(X[tr], yo[tr], args.seed)
        perm = v_c[rng.permutation(len(v_c))]
        directions["sham"] = perm / np.linalg.norm(perm)


        # Tokenise and move to device once. The same batches are reused for every direction and every
        # magnitude, so the cost of preprocessing is paid once rather than 27 x 9 times.
        batches, masks = [], []
        for b0 in range(0, len(te_idx), args.batch_size):
            chunk = te_idx[b0:b0 + args.batch_size]
            imgs = [Image.open(meta[i]["image_path"]).convert("RGB") for i in chunk]
            inp = proc(text=[text] * len(imgs), images=imgs, return_tensors="pt", padding=True).to(dev)
            batches.append(inp)
            masks.append((inp["input_ids"] == img_tok) if img_tok is not None else None)
        print(f"  {len(batches)} batches of up to {args.batch_size} prepared")

        # Calibration: one clean forward pass to read the norm of the tensor the vector is added to.
        probe_n = NormProbe(model, lo.module, positions=lo.positions)
        with probe_n:
            for b0 in range(min(2, len(batches))):
                probe_n.mask = masks[b0]
                with torch.inference_mode():
                    model(**batches[b0])
        nstat = probe_n.stats()
        token_norm = nstat["median"]
        if not np.isfinite(token_norm) or token_norm <= 0:
            raise RuntimeError(f"locus {locus_name}: could not measure a per-token activation norm "
                               f"(got {token_norm}). Refusing to fall back to the pooled norm, which is "
                               f"the confound this calibration exists to remove.")
        skew = nstat["mean"] / max(token_norm, 1e-9)
        print(f"\n{locus_name}: D={X.shape[1]}  per-token ||h|| med {token_norm:.1f} "
              f"mean {nstat['mean']:.1f} p95 {nstat['p95']:.1f} (mean/med {skew:.2f})  "
              f"pooled ||X|| {pooled_norm:.1f}  n_tok {nstat['n']}")
        n_cfg = len(alphas) + (len(directions) - 1) * len(ctrl_alphas)
        print(f"  mode={args.alpha_mode}  {len(directions)} directions, {n_cfg} configurations "
              f"(concept at {len(alphas)} magnitudes, controls at {len(ctrl_alphas)}), "
              f"{min(abs(a) for a in alphas if a) * 100:.0f}%..{max(abs(a) for a in alphas) * 100:.0f}% "
              f"of ||h||, {len(te_idx)} images")

        steer = Steerer(model, lo.module, positions=lo.positions,
                        mode=args.alpha_mode, global_scale=token_norm)
        t0 = time.time()
        with steer:
            for name, vec in directions.items():
                for alpha in (alphas if name == "concept" else ctrl_alphas):
                    # v is unit norm and alpha is a fraction of the activation norm, per token in
                    # reltoken mode and of the locus median in relnorm mode. An earlier parameterisation
                    # divided by sqrt(D), which put the entire sweep below 14% of ||h||, so its null
                    # result measured nothing; those runs are archived under runs/_legacy_alpha.
                    steer.vec = None if alpha == 0 else vec
                    steer.alpha = float(alpha)
                    probs, margins = [], []
                    for b0 in range(0, len(batches), 1):
                        inputs = batches[b0]
                        steer.mask = masks[b0]
                        with torch.inference_mode():
                            out = model(**inputs)
                        probs.extend(yes_prob_batch(out.logits, tok))
                        margins.extend(yes_margin_batch(out.logits, tok))
                    rows.append({"locus": locus_name, "direction": name, "alpha": alpha,
                                 "rel_norm": abs(float(alpha)),
                                 "token_norm": round(token_norm, 3),
                                 "token_norm_mean": round(nstat["mean"], 3),
                                 "pooled_norm": round(pooled_norm, 3),
                                 "norm_ratio": round(token_norm / max(pooled_norm, 1e-9), 4),
                                 "mean_p_yes": float(np.mean(probs)), "sd": float(np.std(probs)),
                                 "mean_margin": float(np.mean(margins)),
                                 "sd_margin": float(np.std(margins)),
                                 "n": len(probs)})
                    if name == "concept" or alpha == max(alphas):
                        print(f"    {name:22s} a={alpha:+5.1f}  P(yes)={np.mean(probs):.4f}")
        print(f"  locus done in {time.time() - t0:.0f}s")

        # Written atomically. The analysis reads this file while the job is still running, and a plain
        # truncate-and-rewrite lets a reader see a torn file: a locus whose last directions or last
        # magnitudes are simply absent. That reads as a real but weaker null band rather than as an
        # incomplete one. Writing to a sibling temp file and renaming means a reader sees either the
        # previous complete flush or the new one, never a half of either.
        csv_path = os.path.join(args.out, "intervention.csv")
        tmp_path = csv_path + ".tmp"
        with open(tmp_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, csv_path)

    json.dump({"arch": args.arch, "concept": args.concept, "alphas": alphas,
               "alpha_mode": args.alpha_mode,
               "scale_reference": "alpha is a fraction of the per-token activation norm at the steered "
                                  "module. Earlier runs scaled by the norm of the POOLED activation, which "
                                  "stands in a depth-dependent ratio to the per-token norm and so made "
                                  "cross-locus magnitudes incomparable.",
               "control_alphas": ctrl_alphas,
               "n_random": args.n_random, "n_eval": len(te_idx), "seed": args.seed,
               "eval_split": eval_name, "eval_row_ids": eval_row_ids,
               "model_source": os.path.realpath(model_source) if local_only else model_source,
               "model_local_only": local_only,
               "directions_path": os.path.realpath(args.directions) if args.directions else "",
               "directions_sha256": bundle_sha256,
               "prompt": prompt,
               "source_commit": os.environ.get("SOURCE_COMMIT", "unknown"),
               "statistic": "concept effect must exceed the 95th percentile of random-direction effects"},
              open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    # A completion marker. The CSV is written after every locus so that a crash keeps partial results,
    # which means its existence says nothing about whether the job finished. Only this file does, and the
    # supervisor keys off it.
    done_loci = sorted({r["locus"] for r in rows})
    requested_loci = sorted({name.strip() for name in args.loci.split(",") if name.strip()})
    if done_loci != requested_loci:
        raise RuntimeError(f"intervention incomplete: requested {requested_loci}, completed {done_loci}")
    with open(os.path.join(args.out, "DONE"), "w") as fh:
        fh.write(json.dumps({"loci": done_loci, "rows": len(rows),
                             "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    print(f"\nwrote {len(rows)} rows over {len(done_loci)} loci to {args.out}/intervention.csv")


if __name__ == "__main__":
    main()
