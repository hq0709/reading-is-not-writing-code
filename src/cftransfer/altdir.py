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
           other concepts' binary labels (intercept + labels). Rows: c's label and every used covariate label known;
           covariates are dropped one at a time, worst first, while one has fewer than RESID_MIN_CLASS_ROWS rows in a
           class on those rows or the rows fall below RESID_RETENTION_FRACTION of c's own known rows (CheXpert:
           uncertain labels are unknown, so the all-known intersection is 8 rows); if c itself is single-class on the
           usable rows, resid falls back to the logistic normal.
           resid_covariates_used / resid_rows_used / resid_fallback record this per concept.

  dom_disp / pattern_disp (ALTDIRD): the same u as dom / pattern, lifted as a DISPLACEMENT instead of a coefficient:
           a coefficient beta of the standardised space lifts as R diag(1/s) beta (the rule above), whereas a
           displacement u of that space is diag(s) u in the projected space and its minimum-norm preimage in model
           space is x = R (R^T R)^{-1} diag(s) u (displacement_lift; R = P). Unit-normalised like the others. The
           eigenvalue spectrum of R^T R / D is stored (disp_gram_eigenvalues, min / max).

Writes fits/<locus>/altdir_seed0.npz: <family>_vectors (6 x D, model space, unit), <family>_projected (6 x 512),
logistic_vectors / logistic_projected (copies of seed 0), cos_model / cos_projected (6 x 5 x 5 cosine matrices over
FAMILY_ORDER), orth_norm_removed_fraction, resid_cos_to_logistic_projected, resid_covariates_used (6 x 6 bool),
resid_rows_used, resid_fallback, per-concept counts, and the ALTDIRD extension: dom_disp_* / pattern_disp_*,
disp_gram_eigenvalues / disp_gram_eig_min / disp_gram_eig_max, family_order_ext, cos_model_ext / cos_projected_ext
(6 x 7 x 7 over FAMILY_ORDER_EXT). Re-running the CLI on an existing file appends the extension keys and verifies that
every key already present is reproduced bit for bit (existing arrays are never changed).
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
DISP_FAMILIES = ("dom_disp", "pattern_disp")             # ALTDIRD: displacement-lifted dom / pattern
RESID_MIN_CLASS_ROWS = 5                                 # a resid covariate needs at least this many rows in each class
RESID_RETENTION_FRACTION = 0.25                          # the usable rows must keep this fraction of the concept's known rows
FAMILY_ORDER = ("logistic",) + FAMILIES                  # order of the cosine matrices
FAMILY_ORDER_EXT = FAMILY_ORDER + DISP_FAMILIES          # order of the extended cosine matrices (cos_model_ext)
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


def resid_covariate_rows(Y: np.ndarray, ci: int, min_class_rows: int = RESID_MIN_CLASS_ROWS,
                         retention: float = RESID_RETENTION_FRACTION) -> tuple[np.ndarray, np.ndarray]:
    """Usable rows and covariate set for concept ci: rows where ci's label and every used covariate label are known.
    Covariates are dropped one at a time, worst first, until both hold: every covariate has at least `min_class_rows`
    rows in each class on the usable rows (else the one with the smallest minor class goes), and the usable rows keep
    at least `retention` of ci's own known rows (else the covariate whose known-mask costs the most rows goes).
    Dropping only enlarges the row set, so at most n_concepts - 1 steps. Returns (rows mask, covariate mask)."""
    known = Y >= 0
    own = int(known[ci].sum())
    cov = np.ones(Y.shape[0], bool); cov[ci] = False
    while True:
        rows = known[ci] & known[cov].all(axis=0)
        used = np.where(cov)[0]
        minor = {d: min(int((Y[d][rows] == 0).sum()), int((Y[d][rows] == 1).sum())) for d in used}
        short = [d for d, m in minor.items() if m < min_class_rows]
        if short:
            cov[min(short, key=lambda d: (minor[d], d))] = False
            continue
        if len(used) and rows.sum() < retention * own:
            gain = {d: int((known[ci] & known[cov & (np.arange(Y.shape[0]) != d)].all(axis=0)).sum()) for d in used}
            cov[max(used, key=lambda d: (gain[d], -d))] = False
            continue
        return rows, cov


def resid_directions(Zs: np.ndarray, Y: np.ndarray, coef: np.ndarray, min_class_rows: int = RESID_MIN_CLASS_ROWS,
                     retention: float = RESID_RETENTION_FRACTION) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refit the logistic probe of every concept on features residualised against the usable covariate labels
    (resid_covariate_rows). A concept that is single-class on its usable rows keeps the logistic normal (fallback).
    Returns the refit normals, their projected-space cosine to the original normals, the (n_concepts, n_concepts)
    covariate mask, the rows used per concept and the fallback flags."""
    n_c = Y.shape[0]
    out = np.zeros((n_c, Zs.shape[1]), np.float64); cos = np.zeros(n_c)
    cov_used = np.zeros((n_c, n_c), bool); rows_used = np.zeros(n_c, np.int64); fallback = np.zeros(n_c, bool)
    for ci in range(n_c):
        rows, cov = resid_covariate_rows(Y, ci, min_class_rows, retention)
        cov_used[ci], rows_used[ci] = cov, int(rows.sum())
        yy = Y[ci][rows]
        if rows.sum() == 0 or yy.min() == yy.max():
            out[ci], cos[ci], fallback[ci] = coef[ci].astype(np.float64), 1.0, True
            continue
        R = residualise(Zs[rows], Y[cov][:, rows].T) if cov.any() else Zs[rows].astype(np.float64) - Zs[rows].mean(axis=0)
        w = fit_lr(R.astype(np.float32), yy).coef_[0].astype(np.float64)
        out[ci] = w
        cos[ci] = float(w @ coef[ci] / (np.linalg.norm(w) * np.linalg.norm(coef[ci])))
    return out, cos, cov_used, rows_used, fallback


def gram_inverse(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R^T R)^{-1} of the projection R = P (512 x 512, float64) and the eigenvalues of R^T R / D (ascending). Every
    campaign locus has D >= 1024 > 512, so R^T R is full rank and this is the plain inverse; the Moore-Penrose
    pseudoinverse is used so that a rank-deficient R (D < 512, synthetic runs) still yields the least-squares
    minimum-norm preimage instead of noise."""
    R = P.astype(np.float64)
    G = R.T @ R
    return np.linalg.pinv(G, hermitian=True), np.linalg.eigvalsh(G / R.shape[0])


def gram_spectrum_summary(eig: np.ndarray) -> dict:
    """min / max / median / rank / condition of the R^T R / D spectrum; the condition number is over the eigenvalues
    above rank tolerance (max * 512 * eps), so a rank-deficient gram reports its rank instead of a meaningless ratio."""
    eig = np.asarray(eig, float)
    tol = eig.max() * len(eig) * np.finfo(float).eps
    pos = eig[eig > tol]
    return {"min": float(eig.min()), "max": float(eig.max()), "median": float(np.median(eig)), "n": int(len(eig)),
            "rank": int(len(pos)), "full_rank": bool(len(pos) == len(eig)), "condition": float(eig.max() / pos.min())}


def displacement_lift(P: np.ndarray, s: np.ndarray, u: np.ndarray, G_inv: np.ndarray) -> np.ndarray:
    """Lift a DISPLACEMENT u of the projected, train-scaled space to a unit model-space direction through its
    minimum-norm preimage: x = R (R^T R)^{-1} diag(s) u  (R^T x = diag(s) u exactly). Contrast with
    fit.direction_from_projected, the coefficient lift R diag(1/s) beta."""
    x = P.astype(np.float64) @ (G_inv @ (np.maximum(s, 1e-8).astype(np.float64) * u.astype(np.float64)))
    return (x / np.linalg.norm(x)).astype(np.float32)


def cosine_matrix(vectors: dict[str, np.ndarray], order: tuple[str, ...] = FAMILY_ORDER) -> np.ndarray:
    """(n_concepts, n_families, n_families) cosines between the families of `order`."""
    fams = [vectors[f] for f in order]
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
    proj["resid"], resid_cos, resid_cov, resid_rows, resid_fb = resid_directions(Zs, Y, coef)
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
              "resid_cos_to_logistic_projected": resid_cos, "resid_covariates_used": resid_cov,
              "resid_rows_used": resid_rows, "resid_fallback": resid_fb, "resid_min_class_rows": np.int64(RESID_MIN_CLASS_ROWS),
              "resid_retention_fraction": np.float64(RESID_RETENTION_FRACTION),
              "cos_model": cosine_matrix(vec), "cos_projected": cosine_matrix(proj)}
    for f in FAMILY_ORDER:
        arrays[f"{f}_projected"] = proj[f].astype(np.float32)
        arrays[f"{f}_vectors"] = vec[f]
    arrays.update(displacement_extension(P, s, proj, vec))
    return arrays


def displacement_extension(P: np.ndarray, s: np.ndarray, proj: dict, vec: dict) -> dict:
    """ALTDIRD extension arrays: the dom / pattern projected vectors of `proj` lifted as displacements, the gram spectrum,
    and the extended cosine tables. `proj`/`vec` hold the four base families (projected / model space); they are not
    modified. Used both by a fresh build and by the extension of an existing altdir file."""
    proj, vec = dict(proj), dict(vec)
    n = next(iter(proj.values())).shape[0]
    G_inv, eig = gram_inverse(P)
    out = {}
    for f, src in (("dom_disp", "dom"), ("pattern_disp", "pattern")):
        proj[f] = np.asarray(proj[src], np.float64)
        vec[f] = np.stack([displacement_lift(P, s, proj[src][ci], G_inv) for ci in range(n)])
        out[f"{f}_projected"] = proj[f].astype(np.float32)
        out[f"{f}_vectors"] = vec[f]
    gs = gram_spectrum_summary(eig)
    out.update(disp_gram_eigenvalues=eig, disp_gram_eig_min=np.float64(eig.min()), disp_gram_eig_max=np.float64(eig.max()),
               disp_gram_rank=np.int64(gs["rank"]), disp_gram_condition=np.float64(gs["condition"]),
               family_order_ext=np.array(FAMILY_ORDER_EXT), cos_model_ext=cosine_matrix(vec, FAMILY_ORDER_EXT),
               cos_projected_ext=cosine_matrix(proj, FAMILY_ORDER_EXT))
    return out


def extend_altdir_file(path: Path, fit: dict) -> dict:
    """Append the ALTDIRD arrays to an existing altdir file from ITS OWN stored dom / pattern vectors (no refit), so the
    displacement families lift exactly the coefficient vectors the ALTDIR module already scored. Existing keys are kept
    byte for byte; a file that already carries the extension is returned unchanged. The file must belong to the fit
    (its logistic vectors must equal the fit's clinical vectors)."""
    old = dict(np.load(path, allow_pickle=False))
    if "dom_disp_vectors" in old and "cos_model_ext" in old:
        return old
    if not np.allclose(old["logistic_vectors"], fit["clinical_vectors"], atol=1e-6):
        raise RuntimeError(f"{path} does not belong to this fit (logistic_vectors != clinical_vectors); refusing to extend")
    proj = {f: old[f"{f}_projected"].astype(np.float64) for f in FAMILY_ORDER}
    vec = {f: old[f"{f}_vectors"] for f in FAMILY_ORDER}
    ext = displacement_extension(fit["projection"], fit["scaler_scale"], proj, vec)
    merged = {**old, **{k: v for k, v in ext.items() if k not in old}}
    np.savez(path, **merged)
    return merged


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
    out_dir = Path(out_dir) if out_dir else fits_dir(model_key, dataset_id, locus_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ALTDIR_FILE
    if path.exists():
        # append-only: an existing file is never rebuilt (its vectors are the ones ALTDIR scored); the ALTDIRD arrays
        # are derived from the file's own stored vectors and appended
        arrays = extend_altdir_file(path, fit)
        print(f"[{model_key}/{dataset_id}/{locus_id}] altdir extended in place at {path} in {time.time() - t0:.1f}s", flush=True)
        return path, arrays
    Zs, Y = scaled_training_features(model_key, dataset_id, locus_id, fit)
    arrays = altdir_arrays(fit, Zs, Y)
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
        if "resid_rows_used" in arrays:      # files built before the covariate-drop amendment lack these keys
            r["resid_rows_used"] = int(arrays["resid_rows_used"][ci])
            r["resid_fallback"] = bool(arrays["resid_fallback"][ci])
            r["resid_dropped"] = ",".join(d for di, d in enumerate(concepts) if di != ci and not arrays["resid_covariates_used"][ci, di]) or "-"
        if "cos_model_ext" in arrays:
            fe = list(arrays["family_order_ext"].astype(str)); ce = arrays["cos_model_ext"]
            for f, src in (("dom_disp", "dom"), ("pattern_disp", "pattern")):
                r[f"cos_model_{f}_logistic"] = float(ce[ci, 0, fe.index(f)])
                r[f"cos_model_{f}_{src}"] = float(ce[ci, fe.index(src), fe.index(f)])
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
    gs = gram_spectrum_summary(arr["disp_gram_eigenvalues"])
    print(f"  R^T R / D spectrum: min={gs['min']:.6f} max={gs['max']:.6f} median={gs['median']:.6f} rank={gs['rank']}/{gs['n']} "
          f"condition={gs['condition']:.3f} (D={arr['logistic_vectors'].shape[1]})")
    print(json.dumps({"path": str(path), "n_train_rows": int(arr["n_train_rows"]),
                      "resid_rows_used": arr["resid_rows_used"].tolist() if "resid_rows_used" in arr else None,
                      "resid_fallback": arr["resid_fallback"].tolist() if "resid_fallback" in arr else None}))
