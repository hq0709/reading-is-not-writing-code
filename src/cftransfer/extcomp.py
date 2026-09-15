"""Extended competitor directions for the EXTCOMP module: one logistic direction per extra dataset label, fitted
exactly like the six protocol normals (same seed-0 projection P, train-only scaler mu/s, LogisticRegression C=1,
lbfgs, 2000 iterations, random_state 0) on the known-label training rows of that label, lifted with
fit.direction_from_projected. The frozen label lists (protocol.EXTCOMP_LABELS) are the manifest labels with at least
EXTCOMP_MIN_SUPPORT known positives and negatives in the training rows; the prep recomputes the counts and refuses to
write when a frozen label falls short or an excluded label qualifies without a recorded reason.

Writes fits/<locus>/extcomp_seed0.npz: extra_names, extra_coefficients (K x 512), extra_intercepts, extra_vectors
(K x D, unit, model space), n_pos / n_neg, auroc_calibration / auroc_test (known-label evaluation rows), cos_model /
cos_projected (K x 6 cosines to the six protocol normals), concept_names. CPU only; seconds.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .fit import direction_from_projected, fit_lr, load_features, load_fit
from .images import load_labels
from .protocol import EXTCOMP_EXCLUDED, EXTCOMP_LABELS, EXTCOMP_MIN_SUPPORT
from .runpaths import fits_dir

EXTCOMP_FILE = "extcomp_seed0.npz"


def label_matrix(dataset_id: str, ids: np.ndarray, names: list[str]) -> np.ndarray:
    """(len(names), len(ids)) int8 labels from the manifest, -1 where the label is unknown or absent."""
    labels = load_labels(dataset_id)
    Y = np.full((len(names), len(ids)), -1, np.int8)
    for ni, name in enumerate(names):
        for j, rid in enumerate(ids):
            lab = labels.get((rid, name))
            if lab is not None and lab["label_known"] == "true":
                Y[ni, j] = int(lab["label"])
    return Y


def support_counts(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (Y == 1).sum(axis=1), (Y == 0).sum(axis=1)


def extcomp_arrays(fit: dict, Zs_train: np.ndarray, Y_train: np.ndarray, names: list[str], Zs_eval: np.ndarray | None = None,
                   Y_eval: np.ndarray | None = None, eval_roles: np.ndarray | None = None,
                   min_support: int = EXTCOMP_MIN_SUPPORT) -> dict:
    """Fit the extra directions from the fit's projected, scaled training features. Y_train: (K, n_train), -1 unknown."""
    P, s, coef6, vec6 = fit["projection"], fit["scaler_scale"], fit["coefficients"], fit["clinical_vectors"]
    n_pos, n_neg = support_counts(Y_train)
    short = [f"{k} ({p} pos / {n} neg)" for k, p, n in zip(names, n_pos, n_neg) if p < min_support or n < min_support]
    if short:
        raise RuntimeError(f"extra labels below the support rule (>= {min_support} positives and negatives): {short}")
    K = len(names)
    coef = np.zeros((K, Zs_train.shape[1]), np.float32); inter = np.zeros(K, np.float32)
    auc_cal = np.full(K, np.nan); auc_test = np.full(K, np.nan)
    for ki in range(K):
        m = Y_train[ki] >= 0
        clf = fit_lr(Zs_train[m], Y_train[ki][m])
        coef[ki], inter[ki] = clf.coef_[0], clf.intercept_[0]
        if Zs_eval is not None:
            for role, store in (("calibration", auc_cal), ("test", auc_test)):
                e = (eval_roles == role) & (Y_eval[ki] >= 0)
                if e.sum() and Y_eval[ki][e].min() != Y_eval[ki][e].max():
                    store[ki] = roc_auc_score(Y_eval[ki][e], Zs_eval[e] @ coef[ki] + inter[ki])
    vec = np.stack([direction_from_projected(P, s, coef[ki]) for ki in range(K)])
    cos_model = vec.astype(np.float64) @ vec6.astype(np.float64).T                                   # unit vectors
    c64, c6 = coef.astype(np.float64), coef6.astype(np.float64)
    cos_proj = (c64 @ c6.T) / (np.linalg.norm(c64, axis=1)[:, None] * np.linalg.norm(c6, axis=1)[None, :])
    return {"extra_names": np.array(names), "concept_names": fit["concept_names"], "extra_coefficients": coef,
            "extra_intercepts": inter, "extra_vectors": vec, "n_pos": n_pos.astype(np.int64), "n_neg": n_neg.astype(np.int64),
            "auroc_calibration": auc_cal, "auroc_test": auc_test, "cos_model": cos_model, "cos_projected": cos_proj,
            "fit_seed": np.int64(0), "min_support": np.int64(min_support), "n_train_rows": np.int64(Zs_train.shape[0])}


def build_extcomp(model_key: str, dataset_id: str, locus_id: str = "vis.last", out_dir: Path | None = None,
                  min_support: int = EXTCOMP_MIN_SUPPORT) -> tuple[Path, dict]:
    t0 = time.time()
    if dataset_id not in EXTCOMP_LABELS:
        raise SystemExit(f"EXTCOMP is NOT_REQUESTED for {dataset_id} (no extra labels)")
    names = list(EXTCOMP_LABELS[dataset_id])
    fit = load_fit(model_key, dataset_id, locus_id, 0)
    ids, X, _vtc, roles = load_features(model_key, dataset_id, locus_id)
    train_idx = np.array([i for i, r in enumerate(roles) if r == "train"])
    eval_idx = np.array([i for i, r in enumerate(roles) if r in ("calibration", "test")])
    if not np.array_equal(ids[train_idx], fit["train_row_ids"].astype(str)):
        raise RuntimeError("training rows of features/<locus>.npz differ from the fit's train_row_ids")
    mu, s = fit["scaler_mean"], fit["scaler_scale"]
    Zs = ((X.astype(np.float32) @ fit["projection"] - mu) / np.maximum(s, 1e-8)).astype(np.float32)
    Y_all = label_matrix(dataset_id, ids, names)
    # excluded labels: verify the recorded reason still holds (a no-finding label, or support below the rule)
    excl = EXTCOMP_EXCLUDED.get(dataset_id, {})
    Y_ex = label_matrix(dataset_id, ids[train_idx], list(excl))
    for name, (p, n) in zip(excl, zip(*support_counts(Y_ex))):
        if "no-finding" not in excl[name] and p >= min_support and n >= min_support:
            raise RuntimeError(f"{name} is excluded as '{excl[name]}' but has {p} pos / {n} neg training rows; update EXTCOMP_LABELS")
    arrays = extcomp_arrays(fit, Zs[train_idx], Y_all[:, train_idx], names, Zs[eval_idx], Y_all[:, eval_idx], roles[eval_idx],
                            min_support)
    out_dir = Path(out_dir) if out_dir else fits_dir(model_key, dataset_id, locus_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / EXTCOMP_FILE
    np.savez(path, **arrays)
    print(f"[{model_key}/{dataset_id}/{locus_id}] extcomp written to {path} in {time.time() - t0:.1f}s", flush=True)
    return path, arrays


def load_extcomp(model_key: str, dataset_id: str, locus_id: str) -> dict:
    path = fits_dir(model_key, dataset_id, locus_id) / EXTCOMP_FILE
    if not path.exists():
        raise FileNotFoundError(f"EXTCOMP needs {path}; compute it first (CPU, seconds): "
                                f"python -m cftransfer.extcomp --model-key {model_key} --dataset {dataset_id}")
    return dict(np.load(path, allow_pickle=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--locus", default="vis.last")
    ap.add_argument("--out-dir", default=None, help="write extcomp_seed0.npz here instead of fits/<locus>/ (verification runs)")
    a = ap.parse_args()
    path, arr = build_extcomp(a.model_key, a.dataset, a.locus, a.out_dir)
    six = arr["concept_names"].astype(str)
    for ki, k in enumerate(arr["extra_names"].astype(str)):
        cos = " ".join(f"{c}={arr['cos_model'][ki, ci]:+.3f}" for ci, c in enumerate(six))
        print(f"  {k:28s} pos={int(arr['n_pos'][ki]):6d} neg={int(arr['n_neg'][ki]):6d} auroc_cal={arr['auroc_calibration'][ki]:.3f} "
              f"auroc_test={arr['auroc_test'][ki]:.3f}  cos_model: {cos}")
    print(json.dumps({"path": str(path), "extra_names": arr["extra_names"].tolist()}))
