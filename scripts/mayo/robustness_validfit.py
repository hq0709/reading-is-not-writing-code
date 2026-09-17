#!/usr/bin/env python
"""VALIDFIT arms: separating the LABEL SOURCE of a direction from the SAMPLE SIZE it was estimated on.

The campaign's expert-label refit fits each CheXpert concept direction on the 200 radiologist-labelled `valid` films.
The shipped direction is fitted on the block's whole training cohort (about 20,000 rows) from report-derived labeler
labels. Comparing those two mixes two things at once: which labels were used, and how many rows were available. This
script adds the arms that hold one of them fixed. Everything is CPU and reads only what is already on disk (the block's
seed-0 fit, its cached features, the frozen cohort manifests and label tables); no GPU work and no refitting of any
scaler.

Five arms, all evaluated on THE SAME 200 valid films and scored by a direction that never saw the film it scores:

  expert_valid200        radiologist labels, 200 valid films, VALIDFIT_FOLDS-fold cross-fit over patients
  report_valid200        report-derived labels of THOSE SAME 200 films, the same folds, the same probe settings. The
                         labeler leaves many of those films blank, so this arm fits FEWER rows than the one above: it
                         holds the films fixed, not the estimation sample
  report_train_sub       report-derived labels, 200-row subsamples of the TRAINING cohort drawn by patient, several
                         draws; each draw is split into the same number of folds and each fold's fit scores the
                         matching fold of valid films. It holds the COHORT sample fixed, and for a sparsely labelled
                         concept that is again fewer usable rows
  report_train_sub_known report-derived labels, subsamples drawn so that 200 rows with a KNOWN label for the concept
                         enter. This is THE SIZE MATCH: it fits the same number of rows per concept per fold as
                         expert_valid200, so the only thing that differs from it is the label source
  report_train_full      report-derived labels, every training row: the shipped direction, the un-matched arm

The contrasts the arms exist to separate are `label_source_at_matched_size` (expert_valid200 against
report_train_sub_known) and `sample_size_at_one_label_source` (report_train_full against report_train_sub_known).

Every arm keeps the block's own projection, its stored TRAIN-ONLY scaler (never refitted), the protocol's probe
settings and the same cross-fitting. Per arm and concept it reports the cross-fitted AUROC against the radiologist
labels, the cross-fitted AUROC against the report-derived labels of the same films, and the raw model-space and
covariance-whitened cosine to the shipped direction.

Writes <RUN_ROOT>/robustness/validfit.json and validfit.md. Minutes, CPU only.

Usage (from the concept-flow repo):
  PYTHONPATH=src python scripts/mayo/robustness_validfit.py [--out <dir>] [--draws N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
from cftransfer.altdir import scaled_training_features                                          # noqa: E402
from cftransfer.fit import direction_from_projected, fit_lr, load_fit                           # noqa: E402
from cftransfer.images import load_cohort                                                       # noqa: E402
from cftransfer.manifest import block_included                                                  # noqa: E402
from cftransfer.protocol import CONCEPTS, VALIDFIT_CV_SEED, VALIDFIT_FOLDS                      # noqa: E402
from cftransfer.runpaths import RUN_ROOT, valid_features_dir                                    # noqa: E402
from cftransfer.validfit import auroc_safe, expert_labels, fold_assignment, report_labels, whitened_cosine  # noqa: E402

OUT_DIR = RUN_ROOT / "robustness"
DATASET = "chexpert"
SUBSAMPLE_DRAWS = 20
SUBSAMPLE_SEED = 2026091703
ARMS = ("expert_valid200", "report_valid200", "report_train_sub", "report_train_sub_known", "report_train_full")
CONTRASTS = ("label_source_at_matched_size", "label_source_on_the_same_films", "sample_size_at_one_label_source",
             "expert_refit_against_the_shipped_direction")
ARM_LABEL = {
    "expert_valid200": ("radiologist", "200 valid films"),
    "report_valid200": ("report-derived", "the same 200 valid films"),
    "report_train_sub": ("report-derived", "200 training rows, drawn by patient"),
    "report_train_sub_known": ("report-derived", "200 training rows with a known label for the concept"),
    "report_train_full": ("report-derived", "every training row")}


def med(xs):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def discover(run_root: Path) -> list[tuple[str, str]]:
    """Included CheXpert blocks whose fits directory carries the VALIDFIT refit."""
    keys = []
    for d in sorted(run_root.glob(f"*/{DATASET}")):
        run = d / "run.json"
        if not run.exists():
            continue
        r = json.loads(run.read_text())
        if not block_included(r):
            continue
        if "VALIDFIT" not in set(r.get("completed_modules") or []):
            continue
        keys.append((d.parent.name, DATASET))
    return keys


def fit_directions(Z: np.ndarray, Y: np.ndarray, n_concepts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(coefficients, fitted flag, rows used) of one probe per concept on the rows with a known label."""
    coef = np.zeros((n_concepts, Z.shape[1]))
    ok = np.zeros(n_concepts, bool)
    used = np.zeros(n_concepts, int)
    for ci in range(n_concepts):
        known = Y[ci] >= 0
        used[ci] = int(known.sum())
        if used[ci] == 0 or Y[ci][known].min() == Y[ci][known].max():
            continue
        coef[ci] = fit_lr(Z[known], Y[ci][known]).coef_[0]
        ok[ci] = True
    return coef, ok, used


def crossfit(Zfit: np.ndarray, Yfit: np.ndarray, folds_fit: np.ndarray, Zeval: np.ndarray, folds_eval: np.ndarray,
             n_concepts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Held-out logits for every eval row: fold f of the eval rows is scored by a probe fitted on the fit rows OUTSIDE
    fold f. When the fit rows are the eval rows this is the campaign's own cross-fitting; when they are training rows
    the eval rows are out of sample already and the split only holds the number of fitting rows fixed."""
    held = np.full((n_concepts, Zeval.shape[0]), np.nan)
    folds_used = np.zeros(n_concepts, int)
    rows_used = np.zeros(n_concepts, int)
    for ci in range(n_concepts):
        known = Yfit[ci] >= 0
        n = []
        for f in range(VALIDFIT_FOLDS):
            tr = known & (folds_fit != f)
            te = folds_eval == f
            if te.sum() == 0 or tr.sum() == 0 or Yfit[ci][tr].min() == Yfit[ci][tr].max():
                continue
            clf = fit_lr(Zfit[tr], Yfit[ci][tr])
            held[ci, te] = Zeval[te] @ clf.coef_[0] + clf.intercept_[0]
            folds_used[ci] += 1
            n.append(int(tr.sum()))
        rows_used[ci] = int(np.median(n)) if n else 0
    return held, folds_used, rows_used


def arm_rows(concepts: list, held: np.ndarray, Y_exp: np.ndarray, Y_rep: np.ndarray, coef: np.ndarray,
             ok: np.ndarray, ref_coef: np.ndarray, ref_vec: np.ndarray, P: np.ndarray, s: np.ndarray,
             Sigma: np.ndarray, folds_used: np.ndarray, rows_used: np.ndarray) -> list[dict]:
    """One row per concept: the two cross-fitted AUROCs and the two cosines to the shipped direction."""
    out = []
    for ci, c in enumerate(concepts):
        cos_model = cos_wh = None
        if ok[ci]:
            v = direction_from_projected(P, s, coef[ci]).astype(np.float64)
            r = ref_vec[ci].astype(np.float64)
            den = np.linalg.norm(v) * np.linalg.norm(r)
            cos_model = float(v @ r / den) if den > 1e-30 else None
            cos_wh = whitened_cosine(coef[ci].astype(np.float64), ref_coef[ci].astype(np.float64), Sigma)
        out.append({"concept": c, "fitted": bool(ok[ci]), "folds_used": int(folds_used[ci]),
                    "median_fit_rows": int(rows_used[ci]),
                    "auroc_expert_labels": auroc_safe(Y_exp[ci], held[ci]),
                    "auroc_report_labels": auroc_safe(Y_rep[ci], held[ci]),
                    "cos_model": cos_model, "cos_whitened": cos_wh})
    return out


def subsample_indices(units: np.ndarray, rng: np.random.Generator, target: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Row indices of whole patients drawn without replacement until `target` rows are collected. `mask` restricts the
    count (and the rows kept) to the rows the concept's probe can actually use."""
    keep = np.ones(len(units), bool) if mask is None else mask
    uniq = np.array(sorted(set(units[keep].tolist())))
    take = []
    for u in uniq[rng.permutation(len(uniq))]:
        take.extend(np.where(keep & (units == u))[0].tolist())
        if len(take) >= target:
            break
    return np.array(take[:target], int)


def block_arms(mk: str, ds: str, locus: str, draws: int) -> dict:
    concepts = list(CONCEPTS[ds])
    K = len(concepts)
    fit = load_fit(mk, ds, locus, 0)
    if list(fit["concept_names"].astype(str)) != concepts:
        raise RuntimeError(f"{mk}/{ds}: seed0.npz concept order differs from the protocol")
    P, s = fit["projection"], fit["scaler_scale"]
    f = np.load(valid_features_dir(mk, ds) / f"{locus}.npz")
    pos = {r: i for i, r in enumerate(f["row_id"].astype(str))}
    rows = load_cohort(ds, ("valid",))
    X = f["x"][[pos[r["row_id"]] for r in rows]]
    Zs = ((X.astype(np.float32) @ P - fit["scaler_mean"]) / np.maximum(s, 1e-8)).astype(np.float32)
    Y_exp = expert_labels(ds, rows, concepts)
    Y_rep, rep_note = report_labels(ds, rows, concepts)
    folds = fold_assignment([r["unit_id"] for r in rows])

    Z_tr, Y_tr = scaled_training_features(mk, ds, locus, fit)
    units_tr = np.array([r["unit_id"] for r in load_cohort(ds, ("train",))])
    Sigma = np.cov(Z_tr.astype(np.float64), rowvar=False)
    ref_coef = fit["coefficients"].astype(np.float64)
    ref_vec = fit["clinical_vectors"].astype(np.float64)

    arms = {}
    # -- the two arms fitted on the 200 valid films, differing only in the label source
    for name, Yfit in (("expert_valid200", Y_exp), ("report_valid200", Y_rep)):
        held, fu, ru = crossfit(Zs, Yfit, folds, Zs, folds, K)
        coef, ok, _ = fit_directions(Zs, Yfit, K)
        arms[name] = {"per_concept": arm_rows(concepts, held, Y_exp, Y_rep, coef, ok, ref_coef, ref_vec, P, s,
                                              Sigma, fu, ru), "n_draws": 1}
    # -- the shipped direction: report labels, the whole training cohort
    logit = np.stack([Zs @ ref_coef[ci] + fit["intercepts"][ci] for ci in range(K)])
    n_known = np.array([int((Y_tr[ci] >= 0).sum()) for ci in range(K)])
    arms["report_train_full"] = {
        "per_concept": [{"concept": c, "fitted": True, "folds_used": 0, "median_fit_rows": int(n_known[ci]),
                         "auroc_expert_labels": auroc_safe(Y_exp[ci], logit[ci]),
                         "auroc_report_labels": auroc_safe(Y_rep[ci], logit[ci]),
                         "cos_model": 1.0, "cos_whitened": 1.0} for ci, c in enumerate(concepts)],
        "n_draws": 1, "note": "the shipped direction; its cosine to itself is 1 by definition"}
    # -- the size-matched training arms
    rng = np.random.Generator(np.random.PCG64(SUBSAMPLE_SEED))
    for name, per_concept_known in (("report_train_sub", False), ("report_train_sub_known", True)):
        per_draw = []
        for d in range(draws):
            held = np.full((K, Zs.shape[0]), np.nan)
            coef = np.zeros((K, Z_tr.shape[1]))
            ok = np.zeros(K, bool)
            fu = np.zeros(K, int)
            ru = np.zeros(K, int)
            idx_common = None if per_concept_known else subsample_indices(units_tr, rng, 200)
            for ci in range(K):
                idx = idx_common if idx_common is not None else subsample_indices(units_tr, rng, 200, Y_tr[ci] >= 0)
                sub_folds = fold_assignment(units_tr[idx].tolist(), VALIDFIT_FOLDS, VALIDFIT_CV_SEED + d)
                h, fu1, ru1 = crossfit(Z_tr[idx], Y_tr[:, idx], sub_folds, Zs, folds, K)
                held[ci] = h[ci]
                fu[ci], ru[ci] = fu1[ci], ru1[ci]
                c1, ok1, _ = fit_directions(Z_tr[idx], Y_tr[:, idx], K)
                coef[ci], ok[ci] = c1[ci], ok1[ci]
            per_draw.append(arm_rows(concepts, held, Y_exp, Y_rep, coef, ok, ref_coef, ref_vec, P, s, Sigma, fu, ru))
        pooled = []
        for ci, c in enumerate(concepts):
            xs = [dw[ci] for dw in per_draw]
            pooled.append({"concept": c, "fitted": any(x["fitted"] for x in xs),
                           "folds_used": int(np.median([x["folds_used"] for x in xs])),
                           "median_fit_rows": int(np.median([x["median_fit_rows"] for x in xs])),
                           "auroc_expert_labels": med([x["auroc_expert_labels"] for x in xs]),
                           "auroc_report_labels": med([x["auroc_report_labels"] for x in xs]),
                           "cos_model": med([x["cos_model"] for x in xs]),
                           "cos_whitened": med([x["cos_whitened"] for x in xs]),
                           "auroc_expert_labels_draw_min": min((x["auroc_expert_labels"] for x in xs
                                                                if x["auroc_expert_labels"] is not None), default=None),
                           "auroc_expert_labels_draw_max": max((x["auroc_expert_labels"] for x in xs
                                                                if x["auroc_expert_labels"] is not None), default=None)})
        arms[name] = {"per_concept": pooled, "per_draw": per_draw, "n_draws": draws,
                      "note": "median over draws; a draw that cannot fit a concept contributes nothing to that concept"}

    return {"block": f"{mk}/{ds}", "model": mk, "dataset": ds, "locus": locus, "n_valid_rows": int(Zs.shape[0]),
            "n_train_rows": int(Z_tr.shape[0]), "n_folds": VALIDFIT_FOLDS, "report_label_source": rep_note,
            "expert_support": {c: {"n_pos": int((Y_exp[ci] == 1).sum()), "n_neg": int((Y_exp[ci] == 0).sum())}
                               for ci, c in enumerate(concepts)},
            "report_support_valid": {c: {"n_pos": int((Y_rep[ci] == 1).sum()), "n_neg": int((Y_rep[ci] == 0).sum())}
                                     for ci, c in enumerate(concepts)},
            "report_support_train": {c: {"n_pos": int((Y_tr[ci] == 1).sum()), "n_neg": int((Y_tr[ci] == 0).sum())}
                                     for ci, c in enumerate(concepts)},
            "arms": arms}


def aggregate(blocks: list) -> dict:
    """Median over every (block, concept) cell of each arm, and the two contrasts the comparison exists to separate."""
    per_arm = {}
    for arm in ARMS:
        cells = [c for b in blocks for c in b["arms"][arm]["per_concept"]]
        per_arm[arm] = {
            "label_source": ARM_LABEL[arm][0], "fit_sample": ARM_LABEL[arm][1],
            "cells": len(cells), "cells_fitted": sum(bool(c["fitted"]) for c in cells),
            "median_fit_rows": med([c["median_fit_rows"] for c in cells]),
            "auroc_expert_labels": med([c["auroc_expert_labels"] for c in cells]),
            "auroc_report_labels": med([c["auroc_report_labels"] for c in cells]),
            "cos_model": med([c["cos_model"] for c in cells]),
            "cos_whitened": med([c["cos_whitened"] for c in cells]),
            "n_auroc_expert": sum(c["auroc_expert_labels"] is not None for c in cells),
            "n_auroc_report": sum(c["auroc_report_labels"] is not None for c in cells)}

    def diff(a, b, key):
        x, y = per_arm[a][key], per_arm[b][key]
        return None if (x is None or y is None) else float(x - y)
    # The arm that matches the expert refit's ESTIMATION SAMPLE is `report_train_sub_known`: both fit a probe on the
    # same number of labelled rows per concept. `report_valid200` uses the same films but the labeler leaves many of
    # them blank, so it fits fewer rows and is a secondary arm, reported with its own row count rather than as the
    # size match.
    matched = "report_train_sub_known"
    rows_ok = (per_arm["expert_valid200"]["median_fit_rows"] is not None
               and per_arm[matched]["median_fit_rows"] == per_arm["expert_valid200"]["median_fit_rows"])
    return {"per_arm": per_arm, "size_matched_arm": matched, "size_match_exact": bool(rows_ok),
            "label_source_at_matched_size": {
                "description": "radiologist labels minus report-derived labels, both fitted on the same number of rows "
                               f"({per_arm['expert_valid200']['median_fit_rows']:.0f} per concept)",
                "delta_auroc_expert_labels": diff("expert_valid200", matched, "auroc_expert_labels")},
            "label_source_on_the_same_films": {
                "description": "radiologist labels minus report-derived labels of the SAME 200 films; the labeler "
                               "leaves many of those films blank, so this arm fits fewer rows and is not the size match",
                "delta_auroc_expert_labels": diff("expert_valid200", "report_valid200", "auroc_expert_labels")},
            "sample_size_at_one_label_source": {
                "description": "report-derived labels on every training row minus report-derived labels on the "
                               "size-matched training subsample",
                "delta_auroc_expert_labels": diff("report_train_full", matched, "auroc_expert_labels")},
            "expert_refit_against_the_shipped_direction": {
                "description": "the comparison the paper reported before the size-matched arms existed; it mixes both",
                "delta_auroc_expert_labels": diff("expert_valid200", "report_train_full", "auroc_expert_labels")}}


def write_md(R: dict, path: Path) -> None:
    f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
    m, A = R["meta"], R["aggregate"]
    L = ["# VALIDFIT arms: label source against sample size", "",
         f"Generated {m['generated_utc']} from `{m['run_root']}`; {m['n_blocks']} CheXpert blocks with VALIDFIT, "
         f"{m['draws']} subsample draws per training arm. Every arm keeps the block's own projection, its stored "
         "train-only scaler, the protocol's probe settings and the same cross-fitting, and every arm is scored on the "
         "same 200 radiologist-labelled films.", "",
         "| arm | labels | fitted on | median fit rows | AUROC vs radiologist | AUROC vs report | cos to shipped | whitened cos |",
         "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        d = A["per_arm"][arm]
        L.append(f"| `{arm}` | {d['label_source']} | {d['fit_sample']} | {d['median_fit_rows']:.0f} | "
                 f"{f3(d['auroc_expert_labels'])} | {f3(d['auroc_report_labels'])} | {f3(d['cos_model'])} | "
                 f"{f3(d['cos_whitened'])} |")
    L += ["", "What each contrast isolates (median cross-fitted AUROC against the radiologist labels):", ""]
    for k in CONTRASTS:
        d = A[k]
        L.append(f"- **{k}**: {d['description']} = {f3(d['delta_auroc_expert_labels'])}")
    L += ["", "## Per block and concept", ""]
    for b in R["blocks"]:
        L += [f"### {b['block']}", "",
              f"{b['n_valid_rows']} valid films, {b['n_train_rows']} training rows, {b['n_folds']} folds. "
              f"Report labels of the valid films: {b['report_label_source']}", "",
              "| concept | arm | fit rows | AUROC vs radiologist | AUROC vs report | cos | whitened cos |",
              "|---|---|---|---|---|---|---|"]
        for ci, c in enumerate(b["arms"]["expert_valid200"]["per_concept"]):
            for arm in ARMS:
                r = b["arms"][arm]["per_concept"][ci]
                L.append(f"| {r['concept']} | `{arm}` | {r['median_fit_rows']} | {f3(r['auroc_expert_labels'])} | "
                         f"{f3(r['auroc_report_labels'])} | {f3(r['cos_model'])} | {f3(r['cos_whitened'])} |")
        L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=str(RUN_ROOT))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--locus", default="vis.last")
    ap.add_argument("--draws", type=int, default=SUBSAMPLE_DRAWS)
    a = ap.parse_args()
    t0 = time.time()
    keys = discover(Path(a.run_root))
    blocks = []
    for mk, ds in keys:
        blocks.append(block_arms(mk, ds, a.locus, a.draws))
        print(f"  {mk}/{ds} done [{time.time() - t0:.0f}s]", flush=True)
    R = {"meta": {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_root": a.run_root,
                  "n_blocks": len(blocks), "blocks": [b["block"] for b in blocks], "locus": a.locus,
                  "draws": a.draws, "subsample_seed": SUBSAMPLE_SEED, "folds": VALIDFIT_FOLDS,
                  "arms": {arm: {"label_source": ARM_LABEL[arm][0], "fitted_on": ARM_LABEL[arm][1]} for arm in ARMS},
                  "rule": "included CheXpert blocks with VALIDFIT in run.json completed_modules",
                  "script": str(Path(__file__).resolve())},
         "blocks": blocks, "aggregate": aggregate(blocks) if blocks else {}}
    R["meta"]["seconds"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validfit.json").write_text(json.dumps(clean(R), indent=1), encoding="utf-8")
    write_md(R, out / "validfit.md")
    for arm in ARMS:
        d = R["aggregate"]["per_arm"][arm]
        print(f"{arm:24s} {d['label_source']:14s} {d['fit_sample']:48s} rows {d['median_fit_rows']:8.0f} "
              f"AUROC(expert) {d['auroc_expert_labels']} AUROC(report) {d['auroc_report_labels']} "
              f"cos {d['cos_model']} whitened {d['cos_whitened']}")
    print(f"  size-matched arm {R['aggregate']['size_matched_arm']}, exact = {R['aggregate']['size_match_exact']}")
    for k in CONTRASTS:
        print(f"  {k}: {R['aggregate'][k]['delta_auroc_expert_labels']}")
    print(f"wrote {out / 'validfit.json'} and validfit.md in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
