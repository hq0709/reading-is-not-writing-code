"""Answer-direction oracle for the ANSDIR module: the direction the reader actually responds to.

Prep (GPU unless --fit-only):
  1. clean forwards (hook passive, no write, batch of one) of the six primary-template questions on the first
     --n-train training-cohort rows (role "train", manifest order); the fp32 answer margin ell_yes - ell_no and
     p = sigmoid(margin) per (row, question) go to outcomes/ANSDIR_prep.parquet (row_id, concept, template_id,
     semantic_margin, p_present);
  2. per question q a ridge regression of the margin on the projected, train-scaled 512-d features of those rows
     (the pooled vis.last features of features/<locus>.npz, projected with the seed-0 P and scaled with the seed-0
     train-only mu / s), alpha chosen by ANSDIR_CV_FOLDS-fold CV (KFold shuffled, seed 0) over ANSDIR_ALPHAS;
  3. the coefficient vector is lifted with fit.direction_from_projected and normalised -> a_q; the coordinate
     permutation of the seed-0 sham (sham_permutations[q]) applied to a_q is the a-sham.
Writes fits/<locus>/ansdir_seed0.npz: answer_vectors (6 x D), answer_sham_vectors (6 x D), answer_coefficients
(6 x 512), answer_intercepts, ridge_alpha, cv_r2, cos_to_logistic (cos(a_q, w_q)), cos_model / cos_projected (6 x 6
between a_q and the logistic normals w_d), train_row_ids, n_train_rows.
--fit-only refits from an existing ANSDIR_prep.parquet without loading the model (CPU, seconds).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

from .adapters import get_adapter
from .fit import direction_from_projected, load_features, load_fit, run_id_for
from .hooks import LocusHook
from .images import image_path, load_cohort, open_rgb
from .protocol import ANSDIR_ALPHAS, ANSDIR_CV_FOLDS, ANSDIR_N_TRAIN, CONCEPTS, LOCI, primary_template, render_question
from .runpaths import fits_dir, outcomes_dir, run_dir
from .scoring import score_logits

ANSDIR_FILE = "ansdir_seed0.npz"
PREP_PARQUET = "ANSDIR_prep.parquet"
PREP_SCHEMA = pa.schema([("run_id", pa.string()), ("model_key", pa.string()), ("dataset_id", pa.string()), ("row_id", pa.string()),
                         ("concept", pa.string()), ("template_id", pa.string()), ("semantic_margin", pa.float32()),
                         ("p_present", pa.float64()), ("sample_status", pa.string()), ("error_reason", pa.string())])


def train_rows(dataset_id: str, n_train: int) -> list[dict]:
    """The first n_train training-cohort rows in manifest order."""
    return load_cohort(dataset_id, ("train",))[:n_train]


def clean_margins(model_key: str, dataset_id: str, rows: list[dict], device_map: str = "cuda:0", locus_id: str = "vis.last",
                  revision: str | None = None) -> pa.Table:
    """Clean answer margins of the six primary-template questions for `rows`; one forward per (row, question), hook
    passive, fp32 answer logits, as the runner's baseline rows (batch of one, consistent across rows)."""
    primary = primary_template(run_dir(model_key, dataset_id))
    concepts = CONCEPTS[dataset_id]
    ad = get_adapter(model_key, revision).load(device_map=device_map)
    locus = ad.loci()[locus_id]
    hook = LocusHook(ad.module(locus.module_path), locus_id)
    run_id = run_id_for(model_key, dataset_id)
    out, t0 = [], time.time()
    with hook:
        for ri, row in enumerate(rows):
            image = open_rgb(image_path(dataset_id, row))
            for concept in concepts:
                rec = {"run_id": run_id, "model_key": model_key, "dataset_id": dataset_id, "row_id": row["row_id"], "concept": concept,
                       "template_id": primary, "semantic_margin": None, "p_present": None, "sample_status": "OK", "error_reason": ""}
                try:
                    enc = ad.encode([image], [render_question(dataset_id, concept, primary)])
                    hook.arm(ad.layouts(enc, [image])[locus_id])
                    sc = score_logits(ad.forward_last_logits(enc), ad.candidates[primary])
                    rec.update(semantic_margin=float(sc.semantic_margin[0]), p_present=float(sc.p_present[0]))
                except Exception as e:                                             # keep the key, record the reason
                    rec.update(sample_status="FAILED", error_reason=f"{type(e).__name__}: {str(e)[:300]}")
                out.append(rec)
            if (ri + 1) % 100 == 0:
                print(f"[{model_key}/{dataset_id}/ansdir] {ri + 1}/{len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    return pa.Table.from_pylist(out, schema=PREP_SCHEMA)


def fit_answer_directions(fit: dict, Zs: np.ndarray, margins: np.ndarray, alphas=ANSDIR_ALPHAS, folds: int = ANSDIR_CV_FOLDS) -> dict:
    """Per question a ridge regression of the margin on the scaled features; alpha by K-fold CV R^2 (KFold shuffled,
    seed 0), refit on every row with the chosen alpha. margins: (6, n) with NaN for failed rows (dropped per question)."""
    P, s, coef6, vec6, perms = fit["projection"], fit["scaler_scale"], fit["coefficients"], fit["clinical_vectors"], fit["sham_permutations"]
    n_c = margins.shape[0]
    coef = np.zeros((n_c, Zs.shape[1]), np.float32); inter = np.zeros(n_c, np.float32)
    alpha_best = np.zeros(n_c); r2 = np.zeros(n_c); n_used = np.zeros(n_c, np.int64)
    for qi in range(n_c):
        ok = np.isfinite(margins[qi])
        Z, y = Zs[ok].astype(np.float64), margins[qi][ok].astype(np.float64)
        n_used[qi] = int(ok.sum())
        if n_used[qi] < folds:
            raise RuntimeError(f"question {qi}: only {n_used[qi]} scored rows")
        kf = KFold(folds, shuffle=True, random_state=0)
        scores = {a: float(cross_val_score(Ridge(alpha=a), Z, y, cv=kf, scoring="r2").mean()) for a in alphas}
        alpha_best[qi] = max(alphas, key=lambda a: (scores[a], -a))
        r2[qi] = scores[alpha_best[qi]]
        m = Ridge(alpha=alpha_best[qi]).fit(Z, y)
        coef[qi], inter[qi] = m.coef_, m.intercept_
    vec = np.stack([direction_from_projected(P, s, coef[qi]) for qi in range(n_c)])
    sham = np.stack([vec[qi][perms[qi]] for qi in range(n_c)]).astype(np.float32)
    cos_model = vec.astype(np.float64) @ vec6.astype(np.float64).T
    c64, c6 = coef.astype(np.float64), coef6.astype(np.float64)
    cos_proj = (c64 @ c6.T) / (np.linalg.norm(c64, axis=1)[:, None] * np.linalg.norm(c6, axis=1)[None, :])
    return {"concept_names": fit["concept_names"], "answer_vectors": vec, "answer_sham_vectors": sham, "answer_coefficients": coef,
            "answer_intercepts": inter, "ridge_alpha": alpha_best, "cv_r2": r2, "cv_alphas": np.array(alphas), "cv_folds": np.int64(folds),
            "cos_to_logistic": np.diagonal(cos_model).copy(), "cos_model": cos_model, "cos_projected": cos_proj,
            "n_rows_used": n_used, "fit_seed": np.int64(0)}


def build_ansdir(model_key: str, dataset_id: str, n_train: int = ANSDIR_N_TRAIN, device_map: str = "cuda:0", locus_id: str = "vis.last",
                 out_dir: Path | None = None, fit_only: bool = False, revision: str | None = None) -> tuple[Path, dict]:
    t0 = time.time()
    rows = train_rows(dataset_id, n_train)
    prep_path = outcomes_dir(model_key, dataset_id) / PREP_PARQUET
    fit = load_fit(model_key, dataset_id, locus_id, 0)
    if fit_only:
        if not prep_path.exists():
            raise FileNotFoundError(f"--fit-only needs {prep_path}")
        table = pq.read_table(prep_path)
    else:
        table = clean_margins(model_key, dataset_id, rows, device_map, locus_id, revision)
        prep_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, prep_path, compression="zstd")
    df = table.to_pandas()
    df = df[df.sample_status == "OK"]
    concepts = list(fit["concept_names"].astype(str))
    ids_all, X, _vtc, roles = load_features(model_key, dataset_id, locus_id)
    pos = {rid: i for i, rid in enumerate(ids_all)}
    row_ids = [r["row_id"] for r in rows]
    missing = [r for r in row_ids if r not in pos]
    if missing:
        raise RuntimeError(f"{len(missing)} prep rows have no pooled features in features/{locus_id}.npz, e.g. {missing[:3]}")
    Zs = ((X[[pos[r] for r in row_ids]].astype(np.float32) @ fit["projection"] - fit["scaler_mean"])
          / np.maximum(fit["scaler_scale"], 1e-8)).astype(np.float32)
    margins = np.full((len(concepts), len(row_ids)), np.nan)
    ridx = {r: i for i, r in enumerate(row_ids)}
    for c, rid, m in zip(df.concept, df.row_id, df.semantic_margin):
        if rid in ridx and c in concepts:
            margins[concepts.index(c), ridx[rid]] = m
    arrays = fit_answer_directions(fit, Zs, margins)
    arrays.update(train_row_ids=np.array(row_ids), n_train_rows=np.int64(len(row_ids)), locus_id=np.array(locus_id))
    out_dir = Path(out_dir) if out_dir else fits_dir(model_key, dataset_id, locus_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ANSDIR_FILE
    np.savez(path, **arrays)
    print(f"[{model_key}/{dataset_id}/{locus_id}] ansdir written to {path} in {time.time() - t0:.0f}s", flush=True)
    return path, arrays


def load_ansdir(model_key: str, dataset_id: str, locus_id: str) -> dict:
    path = fits_dir(model_key, dataset_id, locus_id) / ANSDIR_FILE
    if not path.exists():
        raise FileNotFoundError(f"ANSDIR needs {path}; run its prep first (GPU): "
                                f"python -m cftransfer.ansdir --model-key {model_key} --dataset {dataset_id} --n-train {ANSDIR_N_TRAIN}")
    return dict(np.load(path, allow_pickle=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-train", type=int, default=ANSDIR_N_TRAIN)
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--locus", default="vis.last")
    ap.add_argument("--out-dir", default=None, help="write ansdir_seed0.npz here instead of fits/<locus>/ (verification runs)")
    ap.add_argument("--fit-only", action="store_true", help="refit from an existing outcomes/ANSDIR_prep.parquet (CPU)")
    a = ap.parse_args()
    path, arr = build_ansdir(a.model_key, a.dataset, a.n_train, a.device_map, a.locus, a.out_dir, a.fit_only)
    for qi, q in enumerate(arr["concept_names"].astype(str)):
        print(f"  {q:14s} alpha={arr['ridge_alpha'][qi]:6.1f} cv_r2={arr['cv_r2'][qi]:+.3f} cos(a_q,w_q)={arr['cos_to_logistic'][qi]:+.3f} "
              f"rows={int(arr['n_rows_used'][qi])}")
    print(json.dumps({"path": str(path), "n_train_rows": int(arr["n_train_rows"])}))
