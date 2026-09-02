"""Patient-level bootstrap confidence intervals for the D2 probe AUROCs.

runs/probe_<arch>/probe_results.csv reports point estimates. A point estimate cannot support a
difference claim, so "lingshu selectivity +0.145 against qwen +0.075" is currently a number, not an
observation. This module turns it into one.

Method, and the three things it gets right that a naive bootstrap gets wrong:

1. RESAMPLE THE TEST SET, DO NOT REFIT. The probe is fitted exactly once per (model, concept, locus),
   its decision scores on the held-out rows are kept, and the 2000 resamples recompute AUROC from those
   stored scores. Refitting 2000 times per cell would be a different estimator (it would fold training
   variance into the interval) and would cost weeks.

2. RESAMPLE PATIENTS, NOT IMAGES. NIH ChestX-ray14 has several studies per patient: 5343 test rows come
   from 1659 test patients. An image bootstrap treats those as independent and returns an interval that
   is too narrow. Here a bootstrap replicate draws 1659 patients with replacement and takes every row
   belonging to a drawn patient, as many times as the patient was drawn.

3. ONE PATIENT RESAMPLE DRIVES REAL AND CONTROL TOGETHER. Selectivity is real minus control on the same
   rows, so the two AUROCs are correlated. Bootstrapping them separately and subtracting the endpoints
   would give an interval for selectivity that is far too wide. Each replicate here computes both from
   the same drawn patients and subtracts within the replicate.

   The same shared resample is reused across models, which is legitimate because every model in the
   study was scored on the identical test rows. That makes the medical-against-general difference a
   PAIRED statistic and gives it a much tighter, and correct, interval.

The probe being bootstrapped is the same probe probe.py fitted: the same random projection to 512
dimensions (the RNG stream is replayed in the shard's locus order so the projection matrices are
identical), the same C, the same standardiser, the same control-task label assignment. The refit AUROCs
are compared against the stored ones and the deviation is reported in runs/probe_ci_meta.json.

REPRODUCTION IS EXACT, AND IT IS A CHECK, NOT A COURTESY. The refit must equal
runs/probe_<model>/probe_results.csv to the last bit. The projection is a float32-by-float64 GEMM over
26229 rows, and its summation order depends on how many threads OpenBLAS uses, which OpenBLAS fixes at
import time from the environment. Run this module under the same BLAS thread count as the probe job
(on this machine that is the unconstrained default, 64) and every one of the 2376 compared values
matches exactly. Run it with OMP_NUM_THREADS or OPENBLAS_NUM_THREADS set lower and the AUROCs drift:
measured on llava15_7b/Edema/connector, 64 threads gives the stored 0.799552337, 32 gives 0.800395086,
4 gives 0.800826738, 1 gives 0.801608904. That is 2e-3 of AUROC out of nothing but thread count, and it
is largest on the rare-positive concepts (Edema has 179 test positives out of 5343). An earlier run of
this module was made in a shell with a reduced thread count and published intervals whose centres did
not match probe_results.csv; the meta now records the threading it ran under and shouts if the
reproduction is not exact, so that cannot pass silently again.

Cells are the ones the paper needs, not all 45 x 66: per (model, concept) the peak-real and
peak-selectivity loci, plus the connector, the vision tower output and the last LLM layer at both
position sets. For a matched pair the union of the two models' loci is measured on both, so the pair is
always compared at the same place.

    python src/bootstrap_probe.py --n-boot 2000
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe import build_types, control_labels  # noqa: E402  same type scheme, same assignment
from registry import REGISTRY  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODELS = ["lingshu7b", "qwen7b", "llavamed7b", "llava15_7b", "internvl3_8b"]
PAIRS = [("lingshu7b", "qwen7b"), ("llavamed7b", "llava15_7b")]

# What these intervals do not cover, kept next to the numbers because a CI invites the reader to think
# it covers everything. Each line was checked against the raw activations or the manifest, not assumed.
CAVEATS = [
    "The intervals cover test-set sampling only. They do not include the random-projection draw (one "
    "seed), the control-task label assignment (one draw per concept), or the train split. A cell whose "
    "selectivity CI just clears zero is not established against those other sources of variation.",
    "No nuisance-probe intervals. view_AP and sex_M are outside this schema and were not bootstrapped. "
    "Measured over every locus of all five models in runs/probe_<model>/probe_results.csv, view_AP "
    "decodes at 0.9964 to 0.9994; sex_M runs 0.769 to 0.989 with a per-model median of 0.934 to 0.970 "
    "and clears 0.98 at only 6 to 15 percent of loci. Comparing a clinical concept against the view "
    "ceiling compares an interval to a point estimate.",
    "The pair report makes 351 comparisons at a nominal 95%. Read gap_bh_sig (Benjamini-Hochberg within "
    "the pair-and-kind family named in bh_family), not paired_excludes_zero on its own.",
    "llavamed7b and llava15_7b share a numerically identical frozen CLIP ViT-L/14-336: at vis.last the "
    "mean absolute activation difference is 0.00600 against a mean magnitude of 0.39462, cosine "
    "0.999912. Every pre-connector comparison in that pair is vacuous by construction, and its gaps "
    "there are ~0.005 wide, so tiny differences reach nominal significance while meaning nothing. "
    "Restrict that pair to the connector and beyond. Past the connector the models genuinely diverge "
    "(cosine -0.006115 at connector, 0.012993 at llm.L31.ans).",
    "A peak-selectivity locus for a concept the model cannot decode is the least negative of a set of "
    "negatives, not a peak. Nodule is negative at all 88 of its cells on all five models. Read "
    "sel_a_positive and sel_b_positive before quoting any own_peak row.",
    "own_peak rows rank by the refit selectivity. stored_peak_a / stored_peak_b carry the peak recorded "
    "in probe_results.csv and own_peak_matches_stored says whether they agree.",
]


# --------------------------------------------------------------------------- environment

def blas_info():
    """What BLAS this process is using and with how many threads.

    The refit AUROCs only reproduce probe_results.csv bit for bit when this matches the probe job's
    configuration, so it is recorded next to the reproduction diagnostic rather than left to memory.
    """
    info = {"env": {k: os.environ.get(k) for k in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
            "cpu_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    try:
        from threadpoolctl import threadpool_info
        info["threadpools"] = [{k: d.get(k) for k in
                                ("user_api", "internal_api", "num_threads", "threading_layer")}
                               for d in threadpool_info()]
    except Exception as exc:
        info["threadpools"] = f"unavailable: {exc!r}"
    return info


# --------------------------------------------------------------------------- bootstrap AUROC

def boot_auroc(scores, labels, counts, chunk=400):
    """AUROC of `scores` against `labels` under each row-multiplicity vector in `counts`.

    counts is (B, n) non-negative integers: how many times each test row appears in that replicate.
    Ties in `scores` are given the 0.5 credit the Mann-Whitney statistic gives them, which matters
    because a bootstrap replicate contains duplicated rows with exactly equal scores.

    Returns (B,) AUROCs, nan for a replicate that drew only one class.
    """
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    pos = labels[order].astype(bool)
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])   # first index of each tie group
    out = np.empty(counts.shape[0], dtype=np.float64)
    for a in range(0, counts.shape[0], chunk):
        c = counts[a:a + chunk][:, order].astype(np.float64)
        Pg = np.add.reduceat(np.where(pos, c, 0.0), starts, axis=1)
        Ng = np.add.reduceat(np.where(pos, 0.0, c), starts, axis=1)
        cumN = np.cumsum(Ng, axis=1) - Ng                    # negatives strictly ranked below
        num = (Pg * (cumN + 0.5 * Ng)).sum(axis=1)
        Pt, Nt = Pg.sum(1), Ng.sum(1)
        den = Pt * Nt
        out[a:a + chunk] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return out


def pct_ci(draws, alpha=0.05):
    d = draws[np.isfinite(draws)]
    if d.size < 2:
        return float("nan"), float("nan")
    return float(np.percentile(d, 100 * alpha / 2)), float(np.percentile(d, 100 * (1 - alpha / 2)))


def boot_p(draws):
    """Two-sided bootstrap p-value for "this difference is zero", with the usual +1 so p is never 0."""
    d = draws[np.isfinite(draws)]
    if d.size < 2:
        return float("nan")
    n = d.size
    lo = (1.0 + np.sum(d <= 0)) / (n + 1.0)
    hi = (1.0 + np.sum(d >= 0)) / (n + 1.0)
    return float(min(1.0, 2.0 * min(lo, hi)))


def bh_reject(pvals, q=0.05):
    """Benjamini-Hochberg at level q. Returns a boolean per input p, nan p's never reject.

    The pair report makes hundreds of comparisons at a nominal 95%, so about one in twenty of the null
    ones is expected to clear zero by itself. Without this the reader cannot tell a real gap from the
    expected haul of false positives, and a 0.006 AUROC difference between two models that share a
    frozen vision tower gets printed next to a 0.16 one.
    """
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.zeros(p.shape, dtype=bool)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx], kind="mergesort")]
    m = order.size
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    if passed.any():
        out[order[:np.flatnonzero(passed)[-1] + 1]] = True
    return out


# --------------------------------------------------------------------------- activations

def shard_files(model):
    return sorted(glob.glob(os.path.join(ROOT, "runs", f"act_{model}", "shard*.npz")))


def locus_order_and_dims(files):
    z = np.load(files[0], allow_pickle=True)
    order = [k for k in z.files if k != "row_id"]
    return order, {k: int(z[k].shape[1]) for k in order}


def replay_projections(order, dims, wanted, proj_dim, seed):
    """Reproduce probe.py's projection matrices.

    probe.py draws them from one default_rng(seed) stream, iterating the activation dict in shard key
    order, so a matrix depends on every locus drawn before it. The whole stream is replayed and only the
    wanted matrices are kept; skipping a draw would silently shift every later projection.
    """
    rng = np.random.default_rng(seed)
    keep = {}
    for k in order:
        d = dims[k]
        p = (rng.standard_normal((d, proj_dim)) / np.sqrt(proj_dim) if d > proj_dim
             else np.eye(d)[:, :min(d, proj_dim)])
        if k in wanted:
            keep[k] = p
    return keep


def load_locus(files, locus):
    parts, ids = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if locus not in z.files:
            return None, None
        parts.append(z[locus])
        ids.extend(str(x) for x in z["row_id"])
    return np.concatenate(parts, axis=0), ids


# --------------------------------------------------------------------------- cell selection

def read_probe_csv(model):
    path = os.path.join(ROOT, "runs", f"probe_{model}", "probe_results.csv")
    if not os.path.exists(path):
        return None, None, None
    rows = list(csv.DictReader(open(path)))
    concepts = list(dict.fromkeys(r["concept"] for r in rows))
    return rows, concepts, path


def focus_loci(rows):
    """Per concept the peak-real and peak-selectivity locus, plus the four fixed anchors."""
    sel, why = set(), {}

    def add(lo, tag):
        sel.add(lo)
        why.setdefault(lo, []).append(tag)

    for c in dict.fromkeys(r["concept"] for r in rows):
        rr = [r for r in rows if r["concept"] == c]
        fin = [r for r in rr if np.isfinite(float(r["real"]))]
        if fin:
            add(max(fin, key=lambda r: float(r["real"]))["locus"], f"peak_real:{c}")
        fin = [r for r in rr if np.isfinite(float(r["selectivity"]))]
        if fin:
            add(max(fin, key=lambda r: float(r["selectivity"]))["locus"], f"peak_sel:{c}")
    loci = {r["locus"] for r in rows}
    if "connector" in loci:
        add("connector", "connector")
    if "vis.last" in loci:
        add("vis.last", "vision_tower_out")
    depths = [int(l.split(".L")[1].split(".")[0]) for l in loci if ".L" in l]
    if depths:
        last = max(depths)
        for suf in ("vis", "ans"):
            if f"llm.L{last}.{suf}" in loci:
                add(f"llm.L{last}.{suf}", f"last_llm_layer_{suf}")
    return sel, why


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "manifest.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "probe_ci.csv"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=20260827)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--proj-dim", type=int, default=512)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0, help="must match the probe run's --seed")
    ap.add_argument("--type-cols", default="view_AP,sex_M,age",
                    help="must match the probe run: the columns crossed to define a control-task type. "
                         "chest radiographs view_AP,sex_M,age | dermoscopy site,sex_M,age")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    man = {r["row_id"]: r for r in csv.DictReader(open(args.manifest))}
    t_start = time.time()

    # -- the focus set, with matched pairs measured at the union of their loci -------------------
    csv_rows, concepts, focus, why = {}, {}, {}, {}
    for m in models:
        rows, cons, path = read_probe_csv(m)
        if rows is None:
            print(f"[skip] {m}: no runs/probe_{m}/probe_results.csv")
            continue
        csv_rows[m], concepts[m] = rows, cons
        focus[m], why[m] = focus_loci(rows)
    models = [m for m in models if m in csv_rows]
    if not models:
        raise SystemExit("no probe_results.csv found for any requested model")

    for a, b in PAIRS:
        if a in focus and b in focus:
            have_a = {r["locus"] for r in csv_rows[a]}
            have_b = {r["locus"] for r in csv_rows[b]}
            union = (focus[a] | focus[b])
            for lo in union - focus[a]:
                if lo in have_a:
                    focus[a].add(lo)
                    why[a].setdefault(lo, []).append(f"pair_with:{b}")
            for lo in union - focus[b]:
                if lo in have_b:
                    focus[b].add(lo)
                    why[b].setdefault(lo, []).append(f"pair_with:{a}")

    n_cells = sum(len(focus[m]) * len(concepts[m]) for m in models)
    print(f"focus set: {sum(len(focus[m]) for m in models)} (model, locus) reads, {n_cells} cells, "
          f"{2 * n_cells} probe fits, {args.n_boot} bootstrap resamples each")
    for m in models:
        print(f"  {m:14s} {len(focus[m]):2d} loci x {len(concepts[m])} concepts: {sorted(focus[m])}")

    # -- one shared patient bootstrap, valid for every model because the test rows are identical --
    test_ids = sorted(rid for rid, r in man.items() if r["split"] == "test")
    pat_of = {rid: man[rid]["patient_id"] for rid in test_ids}
    patients = sorted(set(pat_of.values()))
    pidx = {p: i for i, p in enumerate(patients)}
    row_pat = np.array([pidx[pat_of[r]] for r in test_ids], dtype=np.int64)
    P = len(patients)
    brng = np.random.default_rng(args.boot_seed)
    pat_counts = brng.multinomial(P, np.full(P, 1.0 / P), size=args.n_boot).astype(np.int32)
    counts_canon = pat_counts[:, row_pat]          # (B, n_test) multiplicity per canonical test row
    canon_pos = {rid: i for i, rid in enumerate(test_ids)}
    print(f"bootstrap: {P} test patients, {len(test_ids)} test rows, "
          f"{args.n_boot} replicates, seed {args.boot_seed}")

    out_rows, diag, draws_store = [], [], {}

    for m in models:
        files = shard_files(m)
        if not files:
            print(f"[skip] {m}: no activation shards in runs/act_{m}")
            continue
        order, dims = locus_order_and_dims(files)
        proj = replay_projections(order, dims, focus[m], args.proj_dim, args.seed) if args.proj_dim else {}
        stored = {(r["concept"], r["locus"]): r for r in csv_rows[m]}

        for lo in sorted(focus[m]):
            t0 = time.time()
            Xraw, ids = load_locus(files, lo)
            if Xraw is None:
                print(f"[skip] {m}/{lo}: locus absent from the shards")
                continue
            keep = [i for i, r in enumerate(ids) if r in man]
            Xraw = Xraw[keep].astype(np.float32)
            meta = [man[ids[i]] for i in keep]
            split = np.array([r["split"] for r in meta])
            tr, te = split == "train", split == "test"
            # canonical order of this model's test rows, so the shared count matrix lines up
            te_pos = np.array([canon_pos[ids[keep[i]]] for i in np.flatnonzero(te)], dtype=np.int64)
            counts = counts_canon[:, te_pos]

            X = Xraw @ proj[lo] if lo in proj else Xraw.astype(np.float64)
            del Xraw
            sc = StandardScaler().fit(X[tr])
            Atr, Ate = sc.transform(X[tr]), sc.transform(X[te])
            del X
            types = build_types(meta, [c.strip() for c in args.type_cols.split(",") if c.strip()])

            for ci, concept in enumerate(concepts[m]):
                y = np.array([int(r[concept]) for r in meta])
                yc, _, ok = control_labels(types, np.random.default_rng(args.seed + ci))
                rec = {"model": m, "concept": concept, "locus": lo,
                       "n_test_patients": int(np.unique(row_pat[te_pos]).size)}
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                    print(f"[nan] {m}/{lo}/{concept}: one class in train or test")
                    for k in ("real", "real_lo", "real_hi", "control", "control_lo", "control_hi",
                              "selectivity", "sel_lo", "sel_hi"):
                        rec[k] = float("nan")
                    out_rows.append(rec)
                    continue

                s_real = LogisticRegression(C=args.C, max_iter=2000, random_state=args.seed) \
                    .fit(Atr, y[tr]).predict_proba(Ate)[:, 1]
                d_real = boot_auroc(s_real, y[te], counts)
                rec["real"] = float(roc_auc_score(y[te], s_real))
                rec["real_lo"], rec["real_hi"] = pct_ci(d_real, args.alpha)

                if ok and len(np.unique(yc[tr])) > 1 and len(np.unique(yc[te])) > 1:
                    s_ctrl = LogisticRegression(C=args.C, max_iter=2000, random_state=args.seed) \
                        .fit(Atr, yc[tr]).predict_proba(Ate)[:, 1]
                    d_ctrl = boot_auroc(s_ctrl, yc[te], counts)
                    rec["control"] = float(roc_auc_score(yc[te], s_ctrl))
                    rec["control_lo"], rec["control_hi"] = pct_ci(d_ctrl, args.alpha)
                    d_sel = d_real - d_ctrl                 # same replicate, correlation preserved
                    rec["selectivity"] = rec["real"] - rec["control"]
                    rec["sel_lo"], rec["sel_hi"] = pct_ci(d_sel, args.alpha)
                    draws_store[(m, concept, lo)] = d_sel.astype(np.float32)
                else:
                    for k in ("control", "control_lo", "control_hi",
                              "selectivity", "sel_lo", "sel_hi"):
                        rec[k] = float("nan")
                out_rows.append(rec)

                st = stored.get((concept, lo))
                if st:
                    for k in ("real", "control", "selectivity"):
                        a, b = rec[k], float(st[k])
                        if np.isfinite(a) and np.isfinite(b):
                            diag.append({"model": m, "concept": concept, "locus": lo,
                                         "field": k, "refit": a, "stored": b, "abs_diff": abs(a - b)})
            print(f"  {m:14s} {lo:14s} {len(concepts[m])} concepts in {time.time() - t0:5.1f}s")

    cols = ["model", "concept", "locus", "real", "real_lo", "real_hi",
            "control", "control_lo", "control_hi", "selectivity", "sel_lo", "sel_hi",
            "n_test_patients"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\nwrote {len(out_rows)} rows to {args.out}")

    # -- matched pair: does the selectivity gap survive? -----------------------------------------
    pair_rows = []
    idx = {(r["model"], r["concept"], r["locus"]): r for r in out_rows}
    stored_peak = {}
    for m in models:
        for c in concepts[m]:
            fin = [r for r in csv_rows[m]
                   if r["concept"] == c and np.isfinite(float(r["selectivity"]))]
            if fin:
                stored_peak[(m, c)] = max(fin, key=lambda r: float(r["selectivity"]))["locus"]
    try:
      for a, b in PAIRS:
          if a not in models or b not in models:
              continue
          common = sorted({l for (m, c, l) in draws_store if m == a} &
                          {l for (m, c, l) in draws_store if m == b})
          for concept in concepts[a]:
              # each model's own peak-selectivity locus, and every locus both models were measured at
              cands = [("own_peak", None)] + [("common", l) for l in common]
              for kind, lo in cands:
                  if kind == "own_peak":
                      ka = max((k for k in draws_store if k[0] == a and k[1] == concept),
                               key=lambda k: idx[k]["selectivity"], default=None)
                      kb = max((k for k in draws_store if k[0] == b and k[1] == concept),
                               key=lambda k: idx[k]["selectivity"], default=None)
                  else:
                      ka, kb = (a, concept, lo), (b, concept, lo)
                      if ka not in draws_store or kb not in draws_store:
                          continue
                  if ka is None or kb is None:
                      continue
                  ra, rb = idx[ka], idx[kb]
                  gap = ra["selectivity"] - rb["selectivity"]
                  wa = ra["sel_hi"] - ra["sel_lo"]
                  wb = rb["sel_hi"] - rb["sel_lo"]
                  dd = (draws_store[ka] - draws_store[kb]).astype(np.float64)  # same patients, same replicate
                  dlo, dhi = pct_ci(dd, args.alpha)
                  pair_rows.append({
                      "pair": f"{a}_vs_{b}", "concept": concept, "kind": kind,
                      "locus_a": ka[2], "locus_b": kb[2],
                      "sel_a": ra["selectivity"], "sel_a_lo": ra["sel_lo"], "sel_a_hi": ra["sel_hi"],
                      "sel_b": rb["selectivity"], "sel_b_lo": rb["sel_lo"], "sel_b_hi": rb["sel_hi"],
                      "gap": gap, "width_a": wa, "width_b": wb,
                      "gap_gt_widths": bool(abs(gap) > max(wa, wb)),
                      "intervals_disjoint": bool(ra["sel_lo"] > rb["sel_hi"] or rb["sel_lo"] > ra["sel_hi"]),
                      "gap_lo": dlo, "gap_hi": dhi,
                      "paired_excludes_zero": bool(np.isfinite(dlo) and (dlo > 0 or dhi < 0)),
                      "gap_p_boot": boot_p(dd),
                      # a "peak" for a concept the model cannot decode is the least negative of a set of
                      # negatives. Carry the sign so no table can read as though a peak had been found.
                      "sel_a_positive": bool(ra["selectivity"] > 0),
                      "sel_b_positive": bool(rb["selectivity"] > 0),
                      # the focus set is chosen from the STORED peak; this row ranks by the refit. On a
                      # near-tie the two can differ, and the reader has to be able to see that.
                      "stored_peak_a": stored_peak.get((a, concept), ""),
                      "stored_peak_b": stored_peak.get((b, concept), ""),
                      "own_peak_matches_stored": (
                          bool(ka[2] == stored_peak.get((a, concept))
                               and kb[2] == stored_peak.get((b, concept)))
                          if kind == "own_peak" else ""),
                  })
    except Exception as exc:                        # never lose probe_ci.csv to a reporting bug
        print(f"[warn] matched-pair report failed: {exc!r}")
    # Benjamini-Hochberg within each (pair, kind) family. 351 comparisons at a nominal 95% will hand
    # back roughly a dozen null gaps that clear zero on their own; this says which survive the family.
    fam = {}
    for i, r in enumerate(pair_rows):
        fam.setdefault((r["pair"], r["kind"]), []).append(i)
    for key, members in fam.items():
        rej = bh_reject([pair_rows[i]["gap_p_boot"] for i in members], args.alpha)
        for i, keep_it in zip(members, rej):
            pair_rows[i]["gap_bh_sig"] = bool(keep_it)
            pair_rows[i]["bh_family"] = f"{key[0]}:{key[1]}(n={len(members)})"

    pair_out = os.path.join(os.path.dirname(args.out), "probe_ci_pairs.csv")
    if pair_rows:
        with open(pair_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pair_rows[0]))
            w.writeheader()
            w.writerows(pair_rows)
        print(f"wrote {len(pair_rows)} matched-pair comparisons to {pair_out}")

    md = max((d["abs_diff"] for d in diag), default=float("nan"))
    exact = bool(diag) and md == 0.0
    if not exact:
        print("\n" + "!" * 100)
        print(f"REPRODUCTION IS NOT EXACT: {len(diag)} values compared against "
              f"runs/probe_<model>/probe_results.csv, max abs diff {md:.3e}.")
        print("The refit must equal the stored probe AUROCs bit for bit. It does when this module runs")
        print("under the same BLAS thread count as the probe job (the machine default, no OMP_NUM_THREADS")
        print("or OPENBLAS_NUM_THREADS set). A non-zero diff means these intervals are centred on")
        print("numbers the rest of the pipeline does not have. Re-run with the threading unconstrained")
        print("before quoting anything from probe_ci.csv.")
        print("!" * 100)
    json.dump({
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": models, "n_boot": args.n_boot, "boot_seed": args.boot_seed,
        "alpha": args.alpha, "proj_dim": args.proj_dim, "C": args.C, "probe_seed": args.seed,
        "type_cols": args.type_cols,
        "n_test_rows": len(test_ids), "n_test_patients": P,
        "resample_unit": "patient", "estimator": "single fit per cell, test scores resampled",
        "cells": len(out_rows), "elapsed_s": round(time.time() - t_start, 1),
        "focus_loci": {m: {lo: sorted(set(why[m][lo])) for lo in sorted(focus[m])} for m in models},
        "blas": blas_info(),
        "caveats": CAVEATS,
        "repro_vs_stored": {
            "exact": exact,
            "n_compared": len(diag), "max_abs_diff": md,
            "median_abs_diff": float(np.median([d["abs_diff"] for d in diag])) if diag else float("nan"),
            "worst": sorted(diag, key=lambda d: -d["abs_diff"])[:5],
            "note": "refit AUROCs are compared against runs/probe_<model>/probe_results.csv. The "
                    "reproduction is exact when this module runs under the same BLAS thread count as "
                    "the probe job (this machine's unconstrained default, 64 OpenBLAS threads). Any "
                    "non-zero max_abs_diff means the thread count differed: the float32-by-float64 "
                    "projection GEMM changes summation order with thread count, which moves AUROC by "
                    "up to 2e-3 on the rare-positive concepts. Intervals whose centres do not match "
                    "probe_results.csv must not be quoted.",
        },
    }, open(os.path.join(os.path.dirname(args.out), "probe_ci_meta.json"), "w"), indent=1)
    print(f"reproduction of stored point estimates: {len(diag)} compared, max abs diff {md:.2e}"
          f" ({'EXACT' if exact else 'NOT EXACT, see the warning above'})")
    print(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
