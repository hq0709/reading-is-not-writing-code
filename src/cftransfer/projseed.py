"""Projection-seed refits for the PROJSEED module (reviewer objection: every direction in the campaign is estimated
after ONE fixed 512-dimensional random projection, the PCG64(0) draw of fit.fit_locus, so an ownership grade might be a
property of that single projection rather than of the model).

For each k in protocol.PROJSEED_SEEDS the whole probe pipeline is repeated with the projection redrawn and NOTHING else
changed:

  projection   R_k = PCG64(k).standard_normal((D, 512)) / sqrt(512)        (fit.fit_locus with PROJECTION_SEED -> k)
  scaler       train-only StandardScaler recomputed in THAT projected space (a scaler of the seed-0 space is meaningless
               under R_k), on the same training rows the seed-0 fit recorded (train_row_ids)
  probes       LogisticRegression(C=1, lbfgs, 2000 iterations, random_state 0) on the SAME rows and the SAME known-label
               masks as the seed-0 fit (real_train_mask is verified against the manifest labels)
  lift         v = normalize(R_k @ (w / max(s_k, 1e-8)))                   (fit.direction_from_projected)
  random/sham  the same construction re-drawn in that projection: PCG64(k) -> 119 x D standard normals, row-normalised,
               then one coordinate permutation per concept applied to that concept's direction, exactly the order in
               which the seed-0 file draws its family from PCG64(RANDOM_SEED)

So seed k yields a complete, self-contained ownership grade: six directions, six questions, its own 119-direction random
family and its own per-question sham. Model-space cosines to the seed-0 directions are stored (the 512-d coefficient
spaces of two projections are different bases and are NOT comparable, so only model-space cosines are reported).

Writes fits/<locus>/projseed_seed<k>.npz, one file per seed. CPU only; seconds.

    python -m cftransfer.projseed --model-key M --dataset D [--seeds 1,2]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .fit import direction_from_projected, fit_lr, load_features, load_fit
from .images import load_labels
from .protocol import CONCEPTS, N_RANDOM, PROJECTION_DIM, PROJSEED_SEEDS
from .runpaths import fits_dir

PROJSEED_FILE = "projseed_seed{seed}.npz"


def projection_for_seed(seed: int, D: int, dim: int = PROJECTION_DIM) -> np.ndarray:
    """The protocol's projection construction with the seed replaced: PCG64(seed) standard normals / sqrt(dim)."""
    return (np.random.default_rng(seed).standard_normal((D, dim)) / np.sqrt(dim)).astype(np.float32)


def random_family_for_seed(seed: int, D: int, n_concepts: int, n_random: int = N_RANDOM) -> tuple[np.ndarray, np.ndarray]:
    """(random vectors, per-concept coordinate permutations) drawn from PCG64(seed) in the order fit.fit_locus uses:
    the n_random x D normals first (row-normalised), then one permutation per concept."""
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n_random, D))
    R = (R / np.linalg.norm(R, axis=1, keepdims=True)).astype(np.float32)
    perms = np.stack([rng.permutation(D) for _ in range(n_concepts)]).astype(np.int32)
    return R, perms


def concept_labels(dataset_id: str, ids: np.ndarray, concepts: list[str]) -> np.ndarray:
    """(n_concepts, n) manifest labels, -1 unknown."""
    labels = load_labels(dataset_id)
    Y = np.full((len(concepts), len(ids)), -1, np.int8)
    for ci, c in enumerate(concepts):
        for j, rid in enumerate(ids):
            lab = labels[(rid, c)]
            if lab["label_known"] == "true":
                Y[ci, j] = int(lab["label"])
    return Y


def projseed_arrays(fit: dict, X: np.ndarray, Y: np.ndarray, train_idx: np.ndarray, roles: np.ndarray, seed: int,
                    concepts: list[str]) -> dict:
    """Every array of one projection seed from the raw pooled features (X) and the manifest labels (Y)."""
    D = X.shape[1]
    K = len(concepts)
    P = projection_for_seed(seed, D)
    Z = X.astype(np.float32) @ P
    scaler = StandardScaler().fit(Z[train_idx])
    mu, s = scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)
    Zs = ((Z - mu) / np.maximum(s, 1e-8)).astype(np.float32)
    known = Y[:, train_idx] >= 0
    if not np.array_equal(known, fit["real_train_mask"]):
        raise RuntimeError("known-label mask of the training rows differs from the seed-0 fit's real_train_mask: "
                           "labels or features changed since that fit")
    coef = np.zeros((K, PROJECTION_DIM), np.float32); inter = np.zeros(K, np.float32)
    auc_cal = np.full(K, np.nan); auc_test = np.full(K, np.nan)
    n_pos = np.zeros(K, np.int64); n_neg = np.zeros(K, np.int64)
    for ci in range(K):
        m = known[ci]
        y = Y[ci, train_idx][m]
        n_pos[ci], n_neg[ci] = int((y == 1).sum()), int((y == 0).sum())
        if y.min() == y.max():
            raise RuntimeError(f"{concepts[ci]}: single-class training labels")
        clf = fit_lr(Zs[train_idx][m], y)
        coef[ci], inter[ci] = clf.coef_[0], clf.intercept_[0]
        for role, store in (("calibration", auc_cal), ("test", auc_test)):
            ev = roles == role
            kn = ev & (Y[ci] >= 0)
            if kn.sum() and Y[ci][kn].min() != Y[ci][kn].max():
                store[ci] = float(roc_auc_score(Y[ci][kn], Zs[kn] @ coef[ci] + inter[ci]))
    vec = np.stack([direction_from_projected(P, s, coef[ci]) for ci in range(K)])
    R, perms = random_family_for_seed(seed, D, K)
    shams = np.stack([vec[ci][perms[ci]] for ci in range(K)]).astype(np.float32)
    v64, s64 = vec.astype(np.float64), fit["clinical_vectors"].astype(np.float64)
    cos_matrix = v64 @ s64.T                                                  # both families are unit vectors
    return {"concept_names": np.array(concepts), "projection_seed": np.int64(seed), "fit_seed": np.int64(0),
            "projection": P, "scaler_mean": mu, "scaler_scale": s, "coefficients": coef, "intercepts": inter,
            "clinical_vectors": vec, "random_vectors": R, "sham_vectors": shams, "sham_permutations": perms,
            "random_seed": np.int64(seed), "n_random": np.int64(N_RANDOM),
            "cos_model_to_seed0": np.diagonal(cos_matrix).copy(), "cos_model_matrix": cos_matrix,
            "auroc_calibration": auc_cal, "auroc_test": auc_test, "n_train_pos": n_pos, "n_train_neg": n_neg,
            "n_train_rows": np.int64(len(train_idx)), "train_row_ids": fit["train_row_ids"]}


def build_projseed(model_key: str, dataset_id: str, locus_id: str = "vis.last", seeds: tuple[int, ...] = PROJSEED_SEEDS,
                   out_dir: Path | None = None) -> dict[int, tuple[Path, dict]]:
    t0 = time.time()
    concepts = list(CONCEPTS[dataset_id])
    fit = load_fit(model_key, dataset_id, locus_id, 0)
    if list(fit["concept_names"].astype(str)) != concepts:
        raise RuntimeError("seed0.npz concept order differs from the protocol")
    ids, X, _vtc, roles = load_features(model_key, dataset_id, locus_id)
    train_idx = np.array([i for i, r in enumerate(roles) if r == "train"])
    if not np.array_equal(ids[train_idx], fit["train_row_ids"].astype(str)):
        raise RuntimeError("training rows of features/<locus>.npz differ from the fit's train_row_ids")
    Y = concept_labels(dataset_id, ids, concepts)
    out_dir = Path(out_dir) if out_dir else fits_dir(model_key, dataset_id, locus_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for seed in seeds:
        if seed == 0:
            raise ValueError("PROJSEED refits the FURTHER projection seeds; seed 0 is the campaign fit itself")
        arrays = projseed_arrays(fit, X, Y, train_idx, roles, seed, concepts)
        arrays["locus_id"] = np.array(locus_id)
        path = out_dir / PROJSEED_FILE.format(seed=seed)
        np.savez(path, **arrays)
        print(f"[{model_key}/{dataset_id}/{locus_id}] projseed seed {seed} written to {path} in {time.time() - t0:.1f}s", flush=True)
        out[seed] = (path, arrays)
    return out


def load_projseed(model_key: str, dataset_id: str, locus_id: str, seed: int) -> dict:
    path = fits_dir(model_key, dataset_id, locus_id) / PROJSEED_FILE.format(seed=seed)
    if not path.exists():
        raise FileNotFoundError(f"PROJSEED needs {path}; compute it first (CPU, seconds): "
                                f"python -m cftransfer.projseed --model-key {model_key} --dataset {dataset_id}")
    return dict(np.load(path, allow_pickle=False))


def cosine_table(arrays: dict) -> list[dict]:
    """Per concept: the model-space cosine to the seed-0 direction, its largest off-diagonal cosine, and the AUROCs."""
    concepts = list(arrays["concept_names"].astype(str))
    cm = arrays["cos_model_matrix"]
    rows = []
    for ci, c in enumerate(concepts):
        off = {d: float(cm[ci, di]) for di, d in enumerate(concepts) if di != ci}
        rows.append({"concept": c, "cos_model_to_seed0": float(arrays["cos_model_to_seed0"][ci]),
                     "max_abs_cos_other": max(abs(v) for v in off.values()),
                     "argmax_other": max(off, key=lambda d: abs(off[d])),
                     "auroc_calibration": float(arrays["auroc_calibration"][ci]),
                     "auroc_test": float(arrays["auroc_test"][ci]),
                     "n_train_pos": int(arrays["n_train_pos"][ci]), "n_train_neg": int(arrays["n_train_neg"][ci])})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--locus", default="vis.last")
    ap.add_argument("--seeds", default=",".join(str(s) for s in PROJSEED_SEEDS))
    ap.add_argument("--out-dir", default=None, help="write projseed_seed<k>.npz here instead of fits/<locus>/ (verification runs)")
    a = ap.parse_args()
    built = build_projseed(a.model_key, a.dataset, a.locus, tuple(int(x) for x in a.seeds.split(",")), a.out_dir)
    for seed, (path, arr) in built.items():
        for r in cosine_table(arr):
            print(f"  seed{seed} {r['concept']:14s} cos_to_seed0={r['cos_model_to_seed0']:+.4f} "
                  f"max|cos| to another concept={r['max_abs_cos_other']:.4f} ({r['argmax_other']}) "
                  f"auroc_cal={r['auroc_calibration']:.4f} auroc_test={r['auroc_test']:.4f}")
    print(json.dumps({str(s): {"path": str(p), "n_train_rows": int(arr["n_train_rows"]),
                               "cos_model_to_seed0": [round(float(x), 4) for x in arr["cos_model_to_seed0"]]}
                      for s, (p, arr) in built.items()}))
