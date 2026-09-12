"""Fit the fixed-capacity readouts and build every intervention direction (README 5.2, return-format fits/).

Per (model, dataset, locus, fit_seed):
  projection P = default_rng(0).standard_normal((D, 512)) / sqrt(512)              (PCG64, float32)
  scaler     train-only StandardScaler on X @ P
  probes     LogisticRegression(C=1, max_iter=2000, lbfgs, class_weight=None, random_state=0), one per concept
  controls   20 type->random-label probes per concept (seeds 0..19; type = view x sex x age-decade or COCO buckets)
  nuisance   view_AP, sex_M where available
  direction  v_c = normalize(P @ (w_c / max(s, 1e-8)))
  random     119 x D standard normal from default_rng(0), row-normalised, then one coordinate permutation per
             concept (in concept order) applied to v_c -> sham            (seed-0 file only; shared by all seeds)
  REFIT      seeds 1,2 resample training units with replacement, keep every row of a sampled unit with its
             multiplicity, refit scaler + six probes with the same P; controls and random family stay seed 0.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .images import load_cohort, load_labels
from .protocol import (CONCEPTS, CONTROL_SEEDS, LOGREG, N_RANDOM, PROJECTION_DIM, PROJECTION_SEED, PROTOCOL_ID,
                       RANDOM_SEED, REFIT_SEEDS)
from .runpaths import features_dir, fits_dir, run_dir

MIN_TYPES = 8


def run_id_for(model_key: str, dataset_id: str) -> str:
    return f"{PROTOCOL_ID}:{model_key}:{dataset_id}:mayo"


def control_assignment(type_names: list[str], seed: int) -> np.ndarray:
    """Fixed random label per type, as src/probe.py:control_labels (rng.integers(0,2) per sorted type)."""
    rng = np.random.default_rng(seed)
    assign = np.array([int(rng.integers(0, 2)) for _ in type_names], dtype=np.int8)
    if len(set(assign.tolist())) < 2:
        assign[0] = 1 - assign[0]
    return assign


def fit_lr(z: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(**LOGREG).fit(z, y)


def load_features(model_key: str, dataset_id: str, locus_id: str):
    f = np.load(features_dir(model_key, dataset_id) / f"{locus_id}.npz")
    return f["row_id"].astype(str), f["x"], f["valid_token_count"], f["fit_role"].astype(str)


def fit_locus(model_key: str, dataset_id: str, locus_id: str, seeds: tuple[int, ...] = (0, *REFIT_SEEDS),
              write_scores: bool = True) -> dict:
    t0 = time.time()
    ids, X, vtc, roles = load_features(model_key, dataset_id, locus_id)
    cohort = {r["row_id"]: r for r in load_cohort(dataset_id)}
    labels = load_labels(dataset_id)
    concepts = CONCEPTS[dataset_id]
    D = X.shape[1]
    idx_of = {rid: i for i, rid in enumerate(ids)}
    train_idx = np.array([i for i, r in enumerate(roles) if r == "train"])
    eval_idx = np.array([i for i, r in enumerate(roles) if r in ("preflight", "calibration", "test")])
    if len(train_idx) == 0:
        raise RuntimeError("no train rows in features")

    P = (np.random.default_rng(PROJECTION_SEED).standard_normal((D, PROJECTION_DIM)) / np.sqrt(PROJECTION_DIM)).astype(np.float32)
    Z = X.astype(np.float32) @ P                                        # (N, 512) float32

    # labels and types per row
    y = {c: np.array([int(labels[(rid, c)]["label"] or -1) for rid in ids], dtype=np.int8) for c in concepts}
    known = {c: np.array([labels[(rid, c)]["label_known"] == "true" for rid in ids]) for c in concepts}
    types = np.array([labels[(rid, concepts[0])]["type_id"] for rid in ids])
    type_names = sorted(set(types[train_idx].tolist()))
    type_index = {t: i for i, t in enumerate(type_names)}
    units = np.array([cohort[rid]["unit_id"] for rid in ids])
    nuis_cols = {}
    if dataset_id in ("nih", "chexpert"):
        nuis_cols["view_AP"] = np.array([{"AP": 1, "PA": 0}.get(labels[(rid, concepts[0])]["view"], -1) for rid in ids], dtype=np.int8)
        nuis_cols["sex_M"] = np.array([{"M": 1, "F": 0}.get(labels[(rid, concepts[0])]["sex"], -1) for rid in ids], dtype=np.int8)

    controls_eligible = len(type_names) >= MIN_TYPES
    ctrl_labels = np.stack([control_assignment(type_names, s) for s in CONTROL_SEEDS])   # (20, T)
    type_ix = np.array([type_index.get(t, -1) for t in types])

    # random family (seed-0 file) then per-concept shams, in that order
    rng = np.random.default_rng(RANDOM_SEED)
    R = rng.standard_normal((N_RANDOM, D))
    R = (R / np.linalg.norm(R, axis=1, keepdims=True)).astype(np.float32)
    perms = np.stack([rng.permutation(D) for _ in concepts]).astype(np.int32)      # (6, D)

    out_dir = fits_dir(model_key, dataset_id, locus_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_rows = []
    summary = {"model_key": model_key, "dataset_id": dataset_id, "locus_id": locus_id, "D": int(D),
               "n_train_rows": int(len(train_idx)), "n_types": len(type_names), "controls_eligible": controls_eligible,
               "seeds": {}}
    run_id = run_id_for(model_key, dataset_id)

    for seed in seeds:
        # ---- training rows for this seed (multiplicity-weighted resample of independent units for REFIT)
        if seed == 0:
            tr = train_idx
            unit_ids, mult = np.unique(units[train_idx], return_counts=False), None
            train_units, train_mult = np.array(sorted(set(units[train_idx].tolist()))), np.ones(len(set(units[train_idx].tolist())), dtype=np.int32)
        else:
            srng = np.random.default_rng(seed)
            base_units = np.array(sorted(set(units[train_idx].tolist())))
            drawn = srng.choice(base_units, size=len(base_units), replace=True)
            cnt = Counter(drawn.tolist())
            train_units = np.array(sorted(cnt))
            train_mult = np.array([cnt[u] for u in train_units], dtype=np.int32)
            rows_by_unit = {}
            for i in train_idx:
                rows_by_unit.setdefault(units[i], []).append(i)
            tr = np.array([i for u in train_units for _ in range(cnt[u]) for i in rows_by_unit[u]])
        scaler = StandardScaler().fit(Z[tr])
        mu, s = scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)
        Zs = ((Z - mu) / np.maximum(s, 1e-8)).astype(np.float32)

        coef = np.zeros((len(concepts), PROJECTION_DIM), np.float32)
        intercept = np.zeros(len(concepts), np.float32)
        real_mask = np.zeros((len(concepts), len(tr)), bool)
        vectors = np.zeros((len(concepts), D), np.float32)
        ctrl_coef = np.zeros((len(concepts), len(CONTROL_SEEDS), PROJECTION_DIM), np.float32)
        ctrl_int = np.zeros((len(concepts), len(CONTROL_SEEDS)), np.float32)
        ctrl_mask = np.zeros((len(concepts), len(tr)), bool)
        seed_summary = {}
        for ci, c in enumerate(concepts):
            m = known[c][tr]
            real_mask[ci] = m
            yy = y[c][tr][m]
            if yy.min() == yy.max():
                raise RuntimeError(f"{c}: single-class training labels")
            clf = fit_lr(Zs[tr][m], yy)
            coef[ci], intercept[ci] = clf.coef_[0], clf.intercept_[0]
            raw = P @ (coef[ci] / np.maximum(s, 1e-8))
            vectors[ci] = (raw / np.linalg.norm(raw)).astype(np.float32)
            cell = {"n_train_pos": int(yy.sum()), "n_train_neg": int((1 - yy).sum())}
            # controls: same known-label subset, type-based random labels (fit for seed 0 only; reused by REFIT)
            if seed == 0 and controls_eligible:
                ctrl_mask[ci] = m
                for k, cs in enumerate(CONTROL_SEEDS):
                    yk = ctrl_labels[k][type_ix[tr][m]]
                    if yk.min() == yk.max():
                        raise RuntimeError(f"{c}: control seed {cs} single-class")
                    ck = fit_lr(Zs[tr][m], yk)
                    ctrl_coef[ci, k], ctrl_int[ci, k] = ck.coef_[0], ck.intercept_[0]
            # evaluation scores
            for role in ("calibration", "test"):
                ev = np.array([i for i in eval_idx if roles[i] == role])
                lg = Zs[ev] @ coef[ci] + intercept[ci]
                kn = known[c][ev]
                if kn.sum() and y[c][ev][kn].min() != y[c][ev][kn].max():
                    cell[f"auroc_{role}"] = float(roc_auc_score(y[c][ev][kn], lg[kn]))
                if seed == 0 and controls_eligible:
                    cl = [roc_auc_score(ctrl_labels[k][type_ix[ev]], Zs[ev] @ ctrl_coef[ci, k] + ctrl_int[ci, k])
                          for k in range(len(CONTROL_SEEDS)) if len(set(ctrl_labels[k][type_ix[ev]].tolist())) > 1]
                    if cl:
                        cell[f"control_mean_{role}"] = float(np.mean(cl))
                        if f"auroc_{role}" in cell:
                            cell[f"selectivity_{role}"] = cell[f"auroc_{role}"] - cell[f"control_mean_{role}"]
            seed_summary[c] = cell
            if write_scores:
                for i in eval_idx:
                    zi = Zs[i]
                    lg = float(zi @ coef[ci] + intercept[ci])
                    score_rows.append((run_id, dataset_id, ids[i], units[i], roles[i], locus_id, seed, c, "real", None,
                                       int(y[c][i]) if known[c][i] else None, bool(known[c][i]), lg, float(1 / (1 + np.exp(-lg)))))
                    if seed == 0 and controls_eligible:
                        for k, cs in enumerate(CONTROL_SEEDS):
                            lgk = float(zi @ ctrl_coef[ci, k] + ctrl_int[ci, k])
                            tl = int(ctrl_labels[k][type_ix[i]]) if type_ix[i] >= 0 else None
                            score_rows.append((run_id, dataset_id, ids[i], units[i], roles[i], locus_id, seed, c, "control", cs,
                                               tl, tl is not None, lgk, float(1 / (1 + np.exp(-lgk)))))
        shams = np.stack([vectors[ci][perms[ci]] for ci in range(len(concepts))]).astype(np.float32)

        # nuisance probes (seed 0)
        nuis = {}
        if seed == 0:
            for name, col in nuis_cols.items():
                m = col[tr] >= 0
                if m.sum() and col[tr][m].min() != col[tr][m].max():
                    clf = fit_lr(Zs[tr][m], col[tr][m])
                    nuis[name] = (clf.coef_[0].astype(np.float32), float(clf.intercept_[0]))
                    ev = np.array([i for i in eval_idx if roles[i] == "calibration"])
                    kn = col[ev] >= 0
                    seed_summary[f"nuisance_{name}_auroc_calibration"] = float(roc_auc_score(col[ev][kn], Zs[ev][kn] @ clf.coef_[0]))
                    if write_scores:
                        for i in eval_idx:
                            lg = float(Zs[i] @ clf.coef_[0] + clf.intercept_[0])
                            score_rows.append((run_id, dataset_id, ids[i], units[i], roles[i], locus_id, seed, name, "nuisance",
                                               None, int(col[i]) if col[i] >= 0 else None, bool(col[i] >= 0), lg, float(1 / (1 + np.exp(-lg)))))
        arrays = dict(
            projection=P, scaler_mean=mu, scaler_scale=s, concept_names=np.array(concepts), coefficients=coef,
            intercepts=intercept, clinical_vectors=vectors, sham_vectors=shams, sham_permutations=perms,
            type_names=np.array(type_names), control_type_labels=ctrl_labels, real_train_mask=real_mask,
            train_row_ids=ids[tr], train_unit_ids=train_units, train_unit_multiplicities=train_mult,
            projection_seed=np.int64(PROJECTION_SEED), fit_seed=np.int64(seed), C=np.float64(LOGREG["C"]),
            random_seed=np.int64(RANDOM_SEED), controls_eligible=np.bool_(controls_eligible),
        )
        if seed == 0:
            arrays.update(random_vectors=R, control_coefficients=ctrl_coef, control_intercepts=ctrl_int,
                          control_train_mask=ctrl_mask)
            for name, (w, b) in nuis.items():
                arrays[f"nuisance_{name}_coefficient"] = w
                arrays[f"nuisance_{name}_intercept"] = np.float32(b)
        np.savez(out_dir / f"seed{seed}.npz", **arrays)
        summary["seeds"][str(seed)] = seed_summary
        print(f"[{model_key}/{dataset_id}/{locus_id}] seed {seed} fitted in {time.time() - t0:.0f}s", flush=True)

    if write_scores:
        cols = ["run_id", "dataset_id", "row_id", "unit_id", "role", "locus_id", "fit_seed", "concept", "probe_kind",
                "control_seed", "target_label", "target_known", "logit", "probability"]
        table = pa.table({c: [r[i] for r in score_rows] for i, c in enumerate(cols)},
                         schema=pa.schema([("run_id", pa.string()), ("dataset_id", pa.string()), ("row_id", pa.string()),
                                           ("unit_id", pa.string()), ("role", pa.string()), ("locus_id", pa.string()),
                                           ("fit_seed", pa.int32()), ("concept", pa.string()), ("probe_kind", pa.string()),
                                           ("control_seed", pa.int32()), ("target_label", pa.int8()), ("target_known", pa.bool_()),
                                           ("logit", pa.float32()), ("probability", pa.float64())]))
        pq.write_table(table, run_dir(model_key, dataset_id) / f"probe_scores.{locus_id}.parquet")
    summary["seconds"] = round(time.time() - t0, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def load_fit(model_key: str, dataset_id: str, locus_id: str, seed: int) -> dict:
    return dict(np.load(fits_dir(model_key, dataset_id, locus_id) / f"seed{seed}.npz", allow_pickle=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--locus", default="vis.last,connector")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    for lid in a.locus.split(","):
        s = fit_locus(a.model_key, a.dataset, lid, tuple(int(x) for x in a.seeds.split(",")))
        print(json.dumps({k: v for k, v in s.items() if k != "seeds"}))
        for seed, cells in s["seeds"].items():
            for c, cell in cells.items():
                if isinstance(cell, dict):
                    print(f"  seed{seed} {c:14s} " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in cell.items()))
