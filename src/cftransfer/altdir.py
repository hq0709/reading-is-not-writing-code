"""Alternative direction estimators for the ALTDIR module (reviewer objection: low ownership may be a property of the
logistic-normal estimator after the 512-d projection, not of the model).

Every family is computed from the SAME projected (P, seed 0) and train-only-scaled (mu, s) 512-d training features the
seed-0 logistic normals were fitted on (fits/<locus>/seed0.npz + features/<locus>.npz train rows), and lifted to model
space with the SAME rule as the logistic normal, v = normalize(P @ (u / max(s, 1e-8))) (fit.direction_from_projected).
So the families differ only in how the 512-d vector u_c is estimated; projection, scaling, lift, dose, locus, rows,
template and batch composition are the CORE ones. Per concept c (w_c = seed-0 logistic normal, Zs = scaled train rows):

  dom      u_c = mean(Zs[y_c = 1]) - mean(Zs[y_c = 0])          known-label training rows of c (real_train_mask)
  pattern  u_c = Sigma @ w_c, Sigma = cov(Zs, ddof=1)             Haufe pattern over all training rows
  orth     u_c = w_c - proj_{span(w_d, d != c)} w_c               logistic normal with the other five normals' span removed
  resid    u_c = logistic normal of c refitted (same C, iterations, seed) on Zs residualised column-wise against the
           other five binary labels (intercept + five labels; training rows with all six labels known)

Writes fits/<locus>/altdir_seed0.npz: <family>_vectors (6 x D, model space, unit), <family>_projected (6 x 512),
logistic_vectors / logistic_projected (copies of seed 0), cos_model / cos_projected (6 x 5 x 5 cosine matrices over
FAMILY_ORDER), orth_norm_removed_fraction, resid_cos_to_logistic_projected, resid_n_rows, and per-concept counts.
CPU only; seconds.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .fit import direction_from_projected, fit_lr, load_features, load_fit
from .images import load_labels
from .protocol import CONCEPTS, PROJECTION_DIM
from .runpaths import fits_dir

FAMILIES = ("dom", "pattern", "orth", "resid")          # direction kinds the ALTDIR module scores
FAMILY_ORDER = ("logistic",) + FAMILIES                  # order of the cosine matrices
ALTDIR_FILE = "altdir_seed0.npz"


# ----------------------------------------------------------------------------------------- estimators
def dom_directions(Zs: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Difference of class means per concept. Y: (n_concepts, n_rows) int with -1 = label unknown."""
    out = np.zeros((Y.shape[0], Zs.shape[1]), np.float64)
    n_pos = np.zeros(Y.shape[0], np.int64); n_neg = np.zeros(Y.shape[0], np.int64)
    for ci in range(Y.shape[0]):
        pos, neg = Y[ci] == 1, Y[ci] == 0
        if pos.sum() == 0 or neg.sum() == 0:
            raise RuntimeError(f"concept {ci}: single-class training labels")
        out[ci] = Zs[pos].mean(axis=0) - Zs[neg].mean(axis=0)
        n_pos[ci], n_neg[ci] = pos.sum(), neg.sum()
    return out, n_pos, n_neg


def pattern_directions(Zs: np.ndarray, coef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Haufe pattern a_c = Sigma w_c with Sigma the covariance (ddof=1) of ALL training rows."""
    Sigma = np.cov(Zs.astype(np.float64), rowvar=False)
    return coef.astype(np.float64) @ Sigma.T, Sigma                    # (Sigma @ w)^T rows; Sigma symmetric


def orth_directions(coef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """w_c with its component in span{w_d : d != c} removed (orthonormal basis of the span via QR). Returns the
    orthogonalised vectors and the fraction of the norm removed, 1 - |u_c| / |w_c|."""
    W = coef.astype(np.float64)
    out = np.zeros_like(W); removed = np.zeros(W.shape[0])
    for ci in range(W.shape[0]):
        others = np.delete(W, ci, axis=0)                              # (5, 512)
        Q, _ = np.linalg.qr(others.T)                                  # (512, 5) orthonormal columns
        u = W[ci] - Q @ (Q.T @ W[ci])
        out[ci] = u
        removed[ci] = 1.0 - np.linalg.norm(u) / np.linalg.norm(W[ci])
    return out, removed


def residualise(Zs: np.ndarray, labels_other: np.ndarray) -> np.ndarray:
    """Residuals of every feature column after least-squares regression on [1, labels_other] (rows x k)."""
    A = np.column_stack([np.ones(len(Zs)), labels_other.astype(np.float64)])
    beta, *_ = np.linalg.lstsq(A, Zs.astype(np.float64), rcond=None)
    return Zs.astype(np.float64) - A @ beta


def resid_directions(Zs: np.ndarray, Y: np.ndarray, coef: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Refit the logistic probe of every concept on features residualised against the other concepts' labels, using the
    training rows where all labels are known. Returns the refit normals, their projected-space cosine to the original
    normals, and the number of rows used."""
    all_known = (Y >= 0).all(axis=0)
    n_rows = int(all_known.sum())
    if n_rows == 0:
        raise RuntimeError("no training row has every concept label known")
    Zk, Yk = Zs[all_known], Y[:, all_known]
    out = np.zeros((Y.shape[0], Zs.shape[1]), np.float64); cos = np.zeros(Y.shape[0])
    for ci in range(Y.shape[0]):
        others = np.delete(Yk, ci, axis=0).T                           # (n_rows, 5)
        R = residualise(Zk, others)
        yy = Yk[ci]
        if yy.min() == yy.max():
            raise RuntimeError(f"concept {ci}: single-class labels on the all-known rows")
        w = fit_lr(R.astype(np.float32), yy).coef_[0].astype(np.float64)
        out[ci] = w
        cos[ci] = float(w @ coef[ci] / (np.linalg.norm(w) * np.linalg.norm(coef[ci])))
    return out, cos, n_rows


def cosine_matrix(vectors: dict[str, np.ndarray]) -> np.ndarray:
    """(n_concepts, n_families, n_families) cosines between the families of FAMILY_ORDER."""
    fams = [vectors[f] for f in FAMILY_ORDER]
    n = fams[0].shape[0]
    out = np.zeros((n, len(fams), len(fams)))
    for ci in range(n):
        for a, va in enumerate(fams):
            for b, vb in enumerate(fams):
                x, y = va[ci].astype(np.float64), vb[ci].astype(np.float64)
                out[ci, a, b] = x @ y / (np.linalg.norm(x) * np.linalg.norm(y))
    return out


def altdir_arrays(fit: dict, Zs: np.ndarray, Y: np.ndarray) -> dict:
    """All four families from a seed-0 fit dict and its scaled training features. Zs: (n_train, 512) in the fit's
    projected, scaled space, in the fit's training-row order; Y: (n_concepts, n_train) labels, -1 = unknown."""
    P, s, coef = fit["projection"], fit["scaler_scale"], fit["coefficients"]
    concepts = list(fit["concept_names"].astype(str))
    if Zs.shape != (len(fit["train_row_ids"]), PROJECTION_DIM):
        raise ValueError(f"Zs {Zs.shape} does not match the fit's {len(fit['train_row_ids'])} training rows x {PROJECTION_DIM}")
    known = Y >= 0
    if not np.array_equal(known, fit["real_train_mask"]):
        raise RuntimeError("known-label mask differs from the fit's real_train_mask: labels or features changed since the fit")
    proj = {"logistic": coef.astype(np.float64)}
    proj["dom"], n_pos, n_neg = dom_directions(Zs, Y)
    proj["pattern"], _Sigma = pattern_directions(Zs, coef)
    proj["orth"], removed = orth_directions(coef)
    proj["resid"], resid_cos, resid_n = resid_directions(Zs, Y, coef)
    for f, u in proj.items():
        norms = np.linalg.norm(u, axis=1)
        if not np.all(norms > 1e-12):
            raise RuntimeError(f"{f}: zero-norm direction for concept(s) {[concepts[i] for i in np.where(norms <= 1e-12)[0]]}")
    vec = {f: np.stack([direction_from_projected(P, s, proj[f][ci].astype(np.float32)) for ci in range(len(concepts))])
           for f in FAMILY_ORDER}
    if not np.allclose(vec["logistic"], fit["clinical_vectors"], atol=1e-6):
        raise RuntimeError("re-lifted logistic normals differ from clinical_vectors: mapping mismatch")
    arrays = {"concept_names": np.array(concepts), "family_order": np.array(FAMILY_ORDER), "fit_seed": np.int64(0),
              "locus_id": np.array(str(fit.get("locus_id", ""))), "n_train_rows": np.int64(Zs.shape[0]),
              "dom_n_pos": n_pos, "dom_n_neg": n_neg, "orth_norm_removed_fraction": removed,
              "resid_cos_to_logistic_projected": resid_cos, "resid_n_rows": np.int64(resid_n),
              "cos_model": cosine_matrix(vec), "cos_projected": cosine_matrix(proj)}
    for f in FAMILY_ORDER:
        arrays[f"{f}_projected"] = proj[f].astype(np.float32)
        arrays[f"{f}_vectors"] = vec[f]
    return arrays


# ------------------------------------------------------------------------------------------- pipeline
def scaled_training_features(model_key: str, dataset_id: str, locus_id: str, fit: dict) -> tuple[np.ndarray, np.ndarray]:
    """(Zs_train, Y_train): the fit's training rows projected with its P and scaled with its train-only mu/s, plus the
    concept labels of those rows. Verifies the rows are the ones the fit recorded and that a fresh train-only scaler
    reproduces the saved one (the features file has not changed since the fit)."""
    ids, X, _vtc, roles = load_features(model_key, dataset_id, locus_id)
    train_idx = np.array([i for i, r in enumerate(roles) if r == "train"])
    if not np.array_equal(ids[train_idx], fit["train_row_ids"].astype(str)):
        raise RuntimeError("training rows of features/<locus>.npz differ from the fit's train_row_ids")
    Z = X[train_idx].astype(np.float32) @ fit["projection"]
    mu, s = fit["scaler_mean"], fit["scaler_scale"]
    if not (np.allclose(Z.mean(axis=0), mu, rtol=1e-3, atol=1e-3) and np.allclose(Z.std(axis=0), s, rtol=1e-3, atol=1e-3)):
        raise RuntimeError("train-only scaler recomputed from the features does not reproduce the fit's scaler")
    Zs = ((Z - mu) / np.maximum(s, 1e-8)).astype(np.float32)
    labels = load_labels(dataset_id)
    concepts = CONCEPTS[dataset_id]
    Y = np.full((len(concepts), len(train_idx)), -1, np.int8)
    for ci, c in enumerate(concepts):
        for j, rid in enumerate(ids[train_idx]):
            lab = labels[(rid, c)]
            if lab["label_known"] == "true":
                Y[ci, j] = int(lab["label"])
    return Zs, Y


def build_altdir(model_key: str, dataset_id: str, locus_id: str = "vis.last", out_dir: Path | None = None) -> tuple[Path, dict]:
    t0 = time.time()
    fit = load_fit(model_key, dataset_id, locus_id, 0)
    fit["locus_id"] = locus_id
    Zs, Y = scaled_training_features(model_key, dataset_id, locus_id, fit)
    arrays = altdir_arrays(fit, Zs, Y)
    out_dir = Path(out_dir) if out_dir else fits_dir(model_key, dataset_id, locus_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ALTDIR_FILE
    np.savez(path, **arrays)
    print(f"[{model_key}/{dataset_id}/{locus_id}] altdir written to {path} in {time.time() - t0:.1f}s", flush=True)
    return path, arrays


def load_altdir(model_key: str, dataset_id: str, locus_id: str) -> dict:
    path = fits_dir(model_key, dataset_id, locus_id) / ALTDIR_FILE
    if not path.exists():
        raise FileNotFoundError(f"ALTDIR needs {path}; compute it first (CPU, seconds): "
                                f"python -m cftransfer.altdir --model-key {model_key} --dataset {dataset_id}")
    return dict(np.load(path, allow_pickle=False))


def cosine_table(arrays: dict) -> list[dict]:
    """Per-concept model-space cosines to the logistic direction and between families, as printable rows."""
    concepts = arrays["concept_names"].astype(str)
    cm, cp = arrays["cos_model"], arrays["cos_projected"]
    fo = list(arrays["family_order"].astype(str))
    rows = []
    for ci, c in enumerate(concepts):
        r = {"concept": c}
        for f in fo[1:]:
            r[f"cos_model_{f}_logistic"] = float(cm[ci, 0, fo.index(f)])
            r[f"cos_projected_{f}_logistic"] = float(cp[ci, 0, fo.index(f)])
        r["cos_model_dom_pattern"] = float(cm[ci, fo.index("dom"), fo.index("pattern")])
        r["cos_model_orth_resid"] = float(cm[ci, fo.index("orth"), fo.index("resid")])
        r["orth_norm_removed_fraction"] = float(arrays["orth_norm_removed_fraction"][ci])
        r["resid_cos_to_logistic_projected"] = float(arrays["resid_cos_to_logistic_projected"][ci])
        rows.append(r)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--locus", default="vis.last")
    ap.add_argument("--out-dir", default=None, help="write altdir_seed0.npz here instead of fits/<locus>/ (verification runs)")
    a = ap.parse_args()
    path, arr = build_altdir(a.model_key, a.dataset, a.locus, a.out_dir)
    for r in cosine_table(arr):
        print("  " + " ".join(f"{k}={v:+.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    print(json.dumps({"path": str(path), "n_train_rows": int(arr["n_train_rows"]), "resid_n_rows": int(arr["resid_n_rows"])}))
