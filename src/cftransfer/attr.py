"""Non-clinical radiographic attribute directions for the ATTR module: view_AP (1 = AP/portable, 0 = PA), sex_F
(1 = female, 0 = male) and age_60 (1 = age >= 60), read from manifests/labels.csv and fitted exactly like the six
protocol normals (seed-0 projection P, train-only scaler mu/s, LogisticRegression C=1, lbfgs, 2000 iterations,
random_state 0, on the known-attribute training rows; >= ATTR_MIN_CLASS_ROWS rows per class), lifted with
fit.direction_from_projected. Shams are coordinate permutations from PCG64(ATTR_SHAM_SEED), one per attribute.

For the calibration-style readability of the attributes the prep also stores, on the calibration rows, the real
probe logits and the 20 type->random-label control logits (the seed-0 control assignments over the same type strata,
view x sex x age-decade, so an attribute label is one particular labelling of the types and the controls are random
labellings of the same types), plus the attribute labels; the analysis applies the campaign's readability rule to them.

Writes fits/<locus>/attr_seed0.npz: attr_names, attr_coefficients (3 x 512), attr_intercepts, attr_vectors (3 x D),
attr_sham_vectors, attr_sham_permutations, n_pos / n_neg, cos_model / cos_projected (3 x 6 to the protocol normals),
auroc_calibration / auroc_test, cal_row_ids, cal_labels (3 x n_cal), cal_real_logits (3 x n_cal),
cal_control_logits (3 x 20 x n_cal), cal_control_labels (20 x n_cal), controls_eligible. CPU only; seconds.
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
from .protocol import ATTR_CONCEPTS, ATTR_MIN_CLASS_ROWS, ATTR_SHAM_SEED, CONCEPTS, CONTROL_SEEDS
from .runpaths import fits_dir

ATTR_FILE = "attr_seed0.npz"


def attribute_value(name: str, lab: dict) -> int:
    """Attribute label of one manifest row (-1 unknown): view_AP from `view` (AP / PA), sex_F from `sex` (F / M),
    age_60 from numeric `age`."""
    if name == "view_AP":
        return {"AP": 1, "PA": 0}.get(lab["view"], -1)
    if name == "sex_F":
        return {"F": 1, "M": 0}.get(lab["sex"], -1)
    if name == "age_60":
        try:
            return int(float(lab["age"]) >= 60)
        except (TypeError, ValueError):
            return -1
    raise KeyError(name)


def attribute_labels(dataset_id: str, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(len(ATTR_CONCEPTS), n) attribute labels (-1 unknown) and the manifest type_id per row, from labels.csv."""
    labels = load_labels(dataset_id)
    c0 = CONCEPTS[dataset_id][0]
    Y = np.full((len(ATTR_CONCEPTS), len(ids)), -1, np.int8)
    types = np.empty(len(ids), dtype=object)
    for j, rid in enumerate(ids):
        lab = labels[(rid, c0)]
        types[j] = lab["type_id"]
        for ai, a in enumerate(ATTR_CONCEPTS):
            Y[ai, j] = attribute_value(a, lab)
    return Y, types.astype(str)


def build_attr(model_key: str, dataset_id: str, locus_id: str = "vis.last", out_dir: Path | None = None,
               min_class_rows: int = ATTR_MIN_CLASS_ROWS) -> tuple[Path, dict]:
    t0 = time.time()
    if dataset_id not in ("nih", "chexpert"):
        raise SystemExit(f"ATTR is NOT_REQUESTED for {dataset_id} (no radiographic attributes)")
    fit = load_fit(model_key, dataset_id, locus_id, 0)
    ids, X, _vtc, roles = load_features(model_key, dataset_id, locus_id)
    train_idx = np.array([i for i, r in enumerate(roles) if r == "train"])
    cal_idx = np.array([i for i, r in enumerate(roles) if r == "calibration"])
    test_idx = np.array([i for i, r in enumerate(roles) if r == "test"])
    if not np.array_equal(ids[train_idx], fit["train_row_ids"].astype(str)):
        raise RuntimeError("training rows of features/<locus>.npz differ from the fit's train_row_ids")
    P, mu, s = fit["projection"], fit["scaler_mean"], fit["scaler_scale"]
    Zs = ((X.astype(np.float32) @ P - mu) / np.maximum(s, 1e-8)).astype(np.float32)
    Y, types = attribute_labels(dataset_id, ids)
    type_names = list(fit["type_names"].astype(str))
    type_ix = np.array([type_names.index(t) if t in type_names else -1 for t in types])
    ctrl_labels = fit["control_type_labels"]                                            # (20, T)
    controls_eligible = bool(fit["controls_eligible"])
    K = len(ATTR_CONCEPTS)
    coef = np.zeros((K, Zs.shape[1]), np.float32); inter = np.zeros(K, np.float32)
    n_pos = np.zeros(K, np.int64); n_neg = np.zeros(K, np.int64)
    auc_cal = np.full(K, np.nan); auc_test = np.full(K, np.nan)
    cal_real = np.full((K, len(cal_idx)), np.nan, np.float32)
    cal_ctrl = np.full((K, len(CONTROL_SEEDS), len(cal_idx)), np.nan, np.float32)
    for ai, a in enumerate(ATTR_CONCEPTS):
        m = Y[ai][train_idx] >= 0
        yy = Y[ai][train_idx][m]
        n_pos[ai], n_neg[ai] = int(yy.sum()), int((1 - yy).sum())
        if n_pos[ai] < min_class_rows or n_neg[ai] < min_class_rows:
            raise RuntimeError(f"{a}: {n_pos[ai]} pos / {n_neg[ai]} neg known training rows (< {min_class_rows} per class)")
        clf = fit_lr(Zs[train_idx][m], yy)
        coef[ai], inter[ai] = clf.coef_[0], clf.intercept_[0]
        for idx, store in ((cal_idx, auc_cal), (test_idx, auc_test)):
            kn = Y[ai][idx] >= 0
            if kn.sum() and Y[ai][idx][kn].min() != Y[ai][idx][kn].max():
                store[ai] = roc_auc_score(Y[ai][idx][kn], Zs[idx][kn] @ coef[ai] + inter[ai])
        cal_real[ai] = Zs[cal_idx] @ coef[ai] + inter[ai]
        if controls_eligible:
            tr_types = type_ix[train_idx][m]
            for k in range(len(CONTROL_SEEDS)):
                yk = ctrl_labels[k][tr_types]
                if yk.min() == yk.max():
                    continue
                ck = fit_lr(Zs[train_idx][m], yk)
                cal_ctrl[ai, k] = Zs[cal_idx] @ ck.coef_[0] + ck.intercept_[0]
    vec = np.stack([direction_from_projected(P, s, coef[ai]) for ai in range(K)])
    rng = np.random.default_rng(ATTR_SHAM_SEED)
    perms = np.stack([rng.permutation(vec.shape[1]) for _ in ATTR_CONCEPTS]).astype(np.int32)
    shams = np.stack([vec[ai][perms[ai]] for ai in range(K)]).astype(np.float32)
    cos_model = vec.astype(np.float64) @ fit["clinical_vectors"].astype(np.float64).T
    c64, c6 = coef.astype(np.float64), fit["coefficients"].astype(np.float64)
    cos_proj = (c64 @ c6.T) / (np.linalg.norm(c64, axis=1)[:, None] * np.linalg.norm(c6, axis=1)[None, :])
    cal_ctrl_labels = np.where(type_ix[cal_idx][None, :] >= 0, ctrl_labels[:, np.maximum(type_ix[cal_idx], 0)], -1).astype(np.int8)
    arrays = {"attr_names": np.array(ATTR_CONCEPTS), "concept_names": fit["concept_names"], "attr_coefficients": coef,
              "attr_intercepts": inter, "attr_vectors": vec, "attr_sham_vectors": shams, "attr_sham_permutations": perms,
              "attr_sham_seed": np.int64(ATTR_SHAM_SEED), "n_pos": n_pos, "n_neg": n_neg, "auroc_calibration": auc_cal,
              "auroc_test": auc_test, "cos_model": cos_model, "cos_projected": cos_proj, "fit_seed": np.int64(0),
              "min_class_rows": np.int64(min_class_rows), "controls_eligible": np.bool_(controls_eligible),
              "cal_row_ids": ids[cal_idx], "cal_labels": Y[:, cal_idx], "cal_real_logits": cal_real, "cal_control_logits": cal_ctrl,
              "cal_control_labels": cal_ctrl_labels, "test_row_ids": ids[test_idx], "test_labels": Y[:, test_idx]}
    out_dir = Path(out_dir) if out_dir else fits_dir(model_key, dataset_id, locus_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ATTR_FILE
    np.savez(path, **arrays)
    print(f"[{model_key}/{dataset_id}/{locus_id}] attr written to {path} in {time.time() - t0:.1f}s", flush=True)
    return path, arrays


def load_attr(model_key: str, dataset_id: str, locus_id: str) -> dict:
    path = fits_dir(model_key, dataset_id, locus_id) / ATTR_FILE
    if not path.exists():
        raise FileNotFoundError(f"ATTR needs {path}; compute it first (CPU, seconds): "
                                f"python -m cftransfer.attr --model-key {model_key} --dataset {dataset_id}")
    return dict(np.load(path, allow_pickle=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--locus", default="vis.last")
    ap.add_argument("--out-dir", default=None, help="write attr_seed0.npz here instead of fits/<locus>/ (verification runs)")
    a = ap.parse_args()
    path, arr = build_attr(a.model_key, a.dataset, a.locus, a.out_dir)
    six = arr["concept_names"].astype(str)
    for ai, k in enumerate(arr["attr_names"].astype(str)):
        cos = " ".join(f"{c}={arr['cos_model'][ai, ci]:+.3f}" for ci, c in enumerate(six))
        print(f"  {k:8s} pos={int(arr['n_pos'][ai]):6d} neg={int(arr['n_neg'][ai]):6d} auroc_cal={arr['auroc_calibration'][ai]:.3f} "
              f"auroc_test={arr['auroc_test'][ai]:.3f}  cos_model: {cos}")
    print(json.dumps({"path": str(path), "controls_eligible": bool(arr["controls_eligible"])}))
