"""Expert-label direction refits for the VALIDFIT module (reviewer objection: the radiologist-labelled cohort so far
only RE-EVALUATES directions fitted on report-derived labeler labels, so it cannot say whether the directions themselves
depend on the label source).

The six CheXpert concept directions are refitted on the 200 radiologist-labelled `valid` rows (role `valid`,
manifests/release.json label_source radiologist; features/valid/<locus>.npz, written by the VALID feature task) with the
protocol's OWN probe settings and nothing else changed:

  projection   R = the block's seed-0 projection (fits/<locus>/seed0.npz `projection`), not a new draw
  scaler       the block's TRAIN-ONLY mu / s (`scaler_mean` / `scaler_scale`) -- NEVER refitted on 200 rows; the valid
               rows are standardised with the stored training scaler, exactly as every evaluation row is
  probe        LogisticRegression(C=1, lbfgs, 2000 iterations, random_state 0) on the known-label valid rows
  lift         v = normalize(R @ (w / max(s, 1e-8)))                       (fit.direction_from_projected)

Two fits per concept:
  full        every known-label valid row; these six directions are what the GPU part writes (direction ids vfit:<c>)
  cross-fit   VALIDFIT_FOLDS folds over PATIENTS (KFold shuffled, seed VALIDFIT_CV_SEED, over the sorted unit ids), so
              every valid row is scored by a direction that was not fitted on it

CPU report (no GPU anywhere in this module's prep):
  cos_model / cos_projected     raw cosine between each expert-label direction and the corresponding report-label
                                direction, in model space (unit vectors) and in the projected 512-d coefficient space
  cos_whitened                  the same pair under the training covariance Sigma of the projected, train-scaled
                                training features: <a, b>_Sigma / sqrt(<a, a>_Sigma <b, b>_Sigma), i.e. the cosine
                                between Sigma^{1/2} a and Sigma^{1/2} b (the metric in which a coefficient vector's
                                induced pattern lives; Sigma is the one altdir's Haufe family uses)
  auroc_expert_heldout          held-out cross-fitted probe AUROC against the EXPERT labels
  auroc_report_labels_heldout   the same held-out scores against the REPORT-DERIVED (labeler) labels of the same rows
  report_direction_auroc_*      the block's seed-0 report-label direction on the same valid rows, against both label
                                sets, as the reference the expert refit is compared with

Writes fits/<locus>/validfit_seed0.npz. CPU only; seconds.

    python -m cftransfer.validfit --model-key M --dataset chexpert
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from .altdir import scaled_training_features
from .fit import direction_from_projected, fit_lr, load_fit
from . import images                                     # DATA_ROOT is read through the module, so a test root applies
from .images import load_cohort, load_labels
from .protocol import CONCEPTS, VALIDFIT_CV_SEED, VALIDFIT_FOLDS
from .runpaths import fits_dir, valid_features_dir

VALIDFIT_FILE = "validfit_seed0.npz"
# report-derived labels of the valid rows: the CheXpert Plus labeler output (one JSON object per line, keyed by
# path_to_image) that labels every OTHER role of this dataset. It is raw input, not a manifest, so it may be absent on a
# machine that only received the frozen manifests; the AUROCs against it are then NaN and their support counts zero.
LABELER_FILE = {"chexpert": "impression_fixed.json"}
CHEXPERT_RAW = {"Effusion": "Pleural Effusion", "Atelectasis": "Atelectasis", "Pneumothorax": "Pneumothorax",
                "Cardiomegaly": "Cardiomegaly", "Edema": "Edema", "Consolidation": "Consolidation"}


def expert_labels(dataset_id: str, rows: list[dict], concepts: list[str]) -> np.ndarray:
    """(n_concepts, n_rows) radiologist labels of the `valid` rows from manifests/labels.csv, -1 where unknown."""
    labels = load_labels(dataset_id)
    Y = np.full((len(concepts), len(rows)), -1, np.int8)
    for ci, c in enumerate(concepts):
        for j, r in enumerate(rows):
            lab = labels.get((r["row_id"], c))
            if lab is not None and lab["label_known"] == "true":
                Y[ci, j] = int(lab["label"])
    return Y


def report_labels(dataset_id: str, rows: list[dict], concepts: list[str]) -> tuple[np.ndarray, str]:
    """(n_concepts, n_rows) REPORT-DERIVED (labeler) labels of the same rows and the source note. -1 where the labeler
    left the observation blank or the file is not on this machine."""
    Y = np.full((len(concepts), len(rows)), -1, np.int8)
    name = LABELER_FILE.get(dataset_id)
    path = (images.DATA_ROOT / dataset_id / name) if name else None
    if path is None or not path.exists():
        return Y, f"not available ({path if path else dataset_id} missing); report-label AUROCs are NaN"
    table = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                table[rec["path_to_image"]] = rec
    hit = 0
    for j, r in enumerate(rows):
        rec = table.get(r["relative_image_path"].rsplit(".", 1)[0] + ".jpg")
        if rec is None:
            continue
        hit += 1
        for ci, c in enumerate(concepts):
            v = rec.get(CHEXPERT_RAW.get(c, c))
            if v in (0.0, 1.0, 0, 1):
                Y[ci, j] = int(v)
    return Y, f"{name} (CheXpert labeler output, the label source of every other role); {hit}/{len(rows)} rows matched"


def fold_assignment(units: list[str], folds: int = VALIDFIT_FOLDS, seed: int = VALIDFIT_CV_SEED) -> np.ndarray:
    """Fold index per row from a KFold(shuffle, seed) split over the SORTED unique unit ids (patients), so a patient's
    rows never straddle the split and the assignment does not depend on cohort order."""
    uniq = sorted(set(units))
    if len(uniq) < folds:
        raise RuntimeError(f"{len(uniq)} patients cannot be split into {folds} folds")
    of = {}
    for fi, (_tr, te) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(np.arange(len(uniq)))):
        for i in te:
            of[uniq[i]] = fi
    return np.array([of[u] for u in units], np.int32)


def whitened_cosine(a: np.ndarray, b: np.ndarray, Sigma: np.ndarray) -> float:
    """cos(Sigma^{1/2} a, Sigma^{1/2} b) = a^T Sigma b / sqrt(a^T Sigma a * b^T Sigma b)."""
    a, b, S = np.asarray(a, float), np.asarray(b, float), np.asarray(Sigma, float)
    num = float(a @ S @ b)
    den = float(np.sqrt(max(a @ S @ a, 0.0) * max(b @ S @ b, 0.0)))
    return num / den if den > 1e-30 else float("nan")


def auroc_safe(y: np.ndarray, s: np.ndarray) -> float:
    m = (y >= 0) & np.isfinite(s)
    if m.sum() == 0 or y[m].min() == y[m].max():
        return float("nan")
    return float(roc_auc_score(y[m], s[m]))


def validfit_arrays(fit: dict, Zs: np.ndarray, Y_exp: np.ndarray, Y_rep: np.ndarray, folds_of_row: np.ndarray,
                    concepts: list[str], Sigma: np.ndarray) -> dict:
    """Every CPU quantity of the module from the valid rows already standardised with the fit's TRAIN scaler."""
    P, s = fit["projection"], fit["scaler_scale"]
    coef_rep, vec_rep = fit["coefficients"].astype(np.float32), fit["clinical_vectors"]
    K, dim = len(concepts), Zs.shape[1]
    coef = np.zeros((K, dim), np.float32); inter = np.zeros(K, np.float32)
    held = np.full((K, Zs.shape[0]), np.nan)
    n_pos = np.zeros(K, np.int64); n_neg = np.zeros(K, np.int64)
    n_pos_rep = np.zeros(K, np.int64); n_neg_rep = np.zeros(K, np.int64)
    folds_used = np.zeros(K, np.int64); n_heldout = np.zeros(K, np.int64)
    auc_in = np.full(K, np.nan)
    for ci in range(K):
        known = Y_exp[ci] >= 0
        y = Y_exp[ci]
        n_pos[ci], n_neg[ci] = int((y == 1).sum()), int((y == 0).sum())
        n_pos_rep[ci], n_neg_rep[ci] = int((Y_rep[ci] == 1).sum()), int((Y_rep[ci] == 0).sum())
        if n_pos[ci] == 0 or n_neg[ci] == 0:
            raise RuntimeError(f"{concepts[ci]}: {n_pos[ci]} positive / {n_neg[ci]} negative expert-labelled valid rows")
        clf = fit_lr(Zs[known], y[known])
        coef[ci], inter[ci] = clf.coef_[0], clf.intercept_[0]
        auc_in[ci] = auroc_safe(y, Zs @ coef[ci] + inter[ci])
        for f in sorted(set(folds_of_row.tolist())):
            tr, te = known & (folds_of_row != f), known & (folds_of_row == f)
            if te.sum() == 0 or tr.sum() == 0 or y[tr].min() == y[tr].max():
                continue
            cf = fit_lr(Zs[tr], y[tr])
            held[ci, te] = Zs[te] @ cf.coef_[0] + cf.intercept_[0]
            folds_used[ci] += 1
        n_heldout[ci] = int(np.isfinite(held[ci]).sum())
    vec = np.stack([direction_from_projected(P, s, coef[ci]) for ci in range(K)])
    c64, r64 = coef.astype(np.float64), coef_rep.astype(np.float64)
    v64, vr64 = vec.astype(np.float64), vec_rep.astype(np.float64)
    cos_model_matrix = (v64 @ vr64.T) / (np.linalg.norm(v64, axis=1)[:, None] * np.linalg.norm(vr64, axis=1)[None, :])
    cos_proj_matrix = (c64 @ r64.T) / (np.linalg.norm(c64, axis=1)[:, None] * np.linalg.norm(r64, axis=1)[None, :])
    rep_logit = np.stack([Zs @ coef_rep[ci] + fit["intercepts"][ci] for ci in range(K)])
    return {"concept_names": np.array(concepts), "fit_seed": np.int64(0), "label_source": np.array("radiologist"),
            "expert_coefficients": coef, "expert_intercepts": inter, "expert_vectors": vec,
            "report_coefficients": coef_rep, "report_vectors": vec_rep,
            "cos_model": np.diagonal(cos_model_matrix).copy(), "cos_projected": np.diagonal(cos_proj_matrix).copy(),
            "cos_whitened": np.array([whitened_cosine(c64[ci], r64[ci], Sigma) for ci in range(K)]),
            "cos_model_matrix": cos_model_matrix, "cos_projected_matrix": cos_proj_matrix,
            "auroc_expert_heldout": np.array([auroc_safe(Y_exp[ci], held[ci]) for ci in range(K)]),
            "auroc_report_labels_heldout": np.array([auroc_safe(Y_rep[ci], held[ci]) for ci in range(K)]),
            "auroc_expert_in_sample": auc_in,
            "report_direction_auroc_expert": np.array([auroc_safe(Y_exp[ci], rep_logit[ci]) for ci in range(K)]),
            "report_direction_auroc_report_labels": np.array([auroc_safe(Y_rep[ci], rep_logit[ci]) for ci in range(K)]),
            "heldout_logits": held.astype(np.float32), "fold_of_row": folds_of_row, "n_folds": np.int64(VALIDFIT_FOLDS),
            "cv_seed": np.int64(VALIDFIT_CV_SEED), "folds_used": folds_used, "n_heldout_rows": n_heldout,
            "n_pos": n_pos, "n_neg": n_neg, "n_pos_report": n_pos_rep, "n_neg_report": n_neg_rep,
            "n_valid_rows": np.int64(Zs.shape[0]), "expert_labels": Y_exp, "report_derived_labels": Y_rep}


def build_validfit(model_key: str, dataset_id: str, locus_id: str = "vis.last", out_dir: Path | None = None) -> tuple[Path, dict]:
    t0 = time.time()
    if dataset_id != "chexpert":
        raise SystemExit(f"VALIDFIT is NOT_REQUESTED for {dataset_id} (only CheXpert has a radiologist-labelled cohort)")
    concepts = CONCEPTS[dataset_id]
    fit = load_fit(model_key, dataset_id, locus_id, 0)
    if list(fit["concept_names"].astype(str)) != list(concepts):
        raise RuntimeError("seed0.npz concept order differs from the protocol")
    path_v = valid_features_dir(model_key, dataset_id) / f"{locus_id}.npz"
    if not path_v.exists():
        raise FileNotFoundError(f"VALIDFIT needs the valid-role features {path_v}; run the VALID feature task first "
                                f"(python -m cftransfer.features --model-key {model_key} --dataset {dataset_id} --roles valid "
                                f"--out-dir {valid_features_dir(model_key, dataset_id)})")
    f = np.load(path_v)
    pos = {r: i for i, r in enumerate(f["row_id"].astype(str))}
    rows = load_cohort(dataset_id, ("valid",))
    missing = [r["row_id"] for r in rows if r["row_id"] not in pos]
    if missing:
        raise RuntimeError(f"{len(missing)} valid rows have no features in {path_v.name}, e.g. {missing[:3]}")
    X = f["x"][[pos[r["row_id"]] for r in rows]]
    # the block's own projection and its TRAIN-ONLY scaler; the 200 valid rows never refit a scaler
    Zs = ((X.astype(np.float32) @ fit["projection"] - fit["scaler_mean"]) / np.maximum(fit["scaler_scale"], 1e-8)).astype(np.float32)
    Y_exp = expert_labels(dataset_id, rows, concepts)
    Y_rep, rep_note = report_labels(dataset_id, rows, concepts)
    folds_of_row = fold_assignment([r["unit_id"] for r in rows])
    Zs_train, _Y_train = scaled_training_features(model_key, dataset_id, locus_id, fit)
    Sigma = np.cov(Zs_train.astype(np.float64), rowvar=False)
    arrays = validfit_arrays(fit, Zs, Y_exp, Y_rep, folds_of_row, list(concepts), Sigma)
    arrays.update(locus_id=np.array(locus_id), report_label_source=np.array(rep_note),
                  valid_row_ids=np.array([r["row_id"] for r in rows]), n_train_rows=np.int64(Zs_train.shape[0]))
    out_dir = Path(out_dir) if out_dir else fits_dir(model_key, dataset_id, locus_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / VALIDFIT_FILE
    np.savez(path, **arrays)
    print(f"[{model_key}/{dataset_id}/{locus_id}] validfit written to {path} in {time.time() - t0:.1f}s", flush=True)
    return path, arrays


def load_validfit(model_key: str, dataset_id: str, locus_id: str) -> dict:
    path = fits_dir(model_key, dataset_id, locus_id) / VALIDFIT_FILE
    if not path.exists():
        raise FileNotFoundError(f"VALIDFIT needs {path}; compute it first (CPU, seconds): "
                                f"python -m cftransfer.validfit --model-key {model_key} --dataset {dataset_id}")
    return dict(np.load(path, allow_pickle=False))


def report_table(arrays: dict) -> list[dict]:
    """One printable row per concept: supports, cosines to the report-label direction and the four AUROCs."""
    out = []
    for ci, c in enumerate(arrays["concept_names"].astype(str)):
        out.append({"concept": c, "n_pos": int(arrays["n_pos"][ci]), "n_neg": int(arrays["n_neg"][ci]),
                    "n_pos_report": int(arrays["n_pos_report"][ci]), "n_neg_report": int(arrays["n_neg_report"][ci]),
                    "folds_used": int(arrays["folds_used"][ci]), "n_heldout_rows": int(arrays["n_heldout_rows"][ci]),
                    "cos_model": float(arrays["cos_model"][ci]), "cos_projected": float(arrays["cos_projected"][ci]),
                    "cos_whitened": float(arrays["cos_whitened"][ci]),
                    "auroc_expert_heldout": float(arrays["auroc_expert_heldout"][ci]),
                    "auroc_report_labels_heldout": float(arrays["auroc_report_labels_heldout"][ci]),
                    "auroc_expert_in_sample": float(arrays["auroc_expert_in_sample"][ci]),
                    "report_direction_auroc_expert": float(arrays["report_direction_auroc_expert"][ci]),
                    "report_direction_auroc_report_labels": float(arrays["report_direction_auroc_report_labels"][ci])})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--locus", default="vis.last")
    ap.add_argument("--out-dir", default=None, help="write validfit_seed0.npz here instead of fits/<locus>/ (verification runs)")
    a = ap.parse_args()
    path, arr = build_validfit(a.model_key, a.dataset, a.locus, a.out_dir)
    for r in report_table(arr):
        print(f"  {r['concept']:14s} valid pos/neg={r['n_pos']:3d}/{r['n_neg']:3d} report pos/neg={r['n_pos_report']:3d}/{r['n_neg_report']:3d} "
              f"folds={r['folds_used']} cos_model={r['cos_model']:+.4f} cos_proj={r['cos_projected']:+.4f} "
              f"cos_whitened={r['cos_whitened']:+.4f} auroc_expert_heldout={r['auroc_expert_heldout']:.4f} "
              f"auroc_report_heldout={r['auroc_report_labels_heldout']:.4f} "
              f"report_dir_auroc_expert={r['report_direction_auroc_expert']:.4f}")
    print(json.dumps({"path": str(path), "n_valid_rows": int(arr["n_valid_rows"]), "n_train_rows": int(arr["n_train_rows"]),
                      "report_label_source": str(arr["report_label_source"])}))
