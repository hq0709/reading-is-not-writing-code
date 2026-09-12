"""Fit pipeline on a synthetic run directory: array contract, direction reconstruction, sham/random rules, REFIT."""
import csv
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from cftransfer import fit as F, images as I, runpaths as R, protocol as P   # noqa: E402


def make_synthetic(tmp: Path, D=16, n_units=30, rows_per_unit=2, n_eval=12):
    rng = np.random.default_rng(1)
    data = tmp / "data"; runs = tmp / "runs"
    man = data / "nih" / "manifests"; man.mkdir(parents=True)
    cohort, labels, ids, roles, units = [], [], [], [], []
    views, sexes, ages = ["AP", "PA"], ["M", "F"], [20, 35, 48, 61, 75]
    k = 0
    for u in range(n_units):
        for j in range(rows_per_unit):
            rid = f"{u:08d}_{j:03d}"; ids.append(rid); roles.append("train"); units.append(str(u))
            cohort.append(dict(dataset_id="nih", row_id=rid, unit_id=str(u), role="train", order=k, relative_image_path="x", original_split="train")); k += 1
    for role in ("preflight", "calibration", "test"):
        for j in range(n_eval):
            rid = f"{role[:3]}_{j:03d}"; ids.append(rid); roles.append(role); units.append(f"{role}{j}")
            cohort.append(dict(dataset_id="nih", row_id=rid, unit_id=f"{role}{j}", role=role, order=j, relative_image_path="x", original_split="val"))
    for i, rid in enumerate(ids):
        v, s, a = views[i % 2], sexes[(i // 2) % 2], ages[(i // 4) % 5]
        tid = f"{1 if v == 'AP' else 0}_{1 if s == 'M' else 0}_{a // 10}"
        for c in P.CONCEPTS["nih"] + ["Consolidation", "Edema", "Infiltration", "no_finding"]:
            lab = int(rng.integers(0, 2))
            labels.append(dict(dataset_id="nih", row_id=rid, concept=c, label=lab, label_known="true", label_raw=lab, view=v, sex=s,
                               age=a, width=1024, height=1024, type_id=tid))
    with (man / "cohort.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cohort[0])); w.writeheader(); w.writerows(cohort)
    with (man / "labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(labels[0])); w.writeheader(); w.writerows(labels)
    feat = runs / "m" / "nih" / "features"; feat.mkdir(parents=True)
    X = rng.standard_normal((len(ids), D)).astype(np.float16)
    for lid in ("vis.last", "connector"):
        np.savez(feat / f"{lid}.npz", row_id=np.array(ids), x=X, valid_token_count=np.full(len(ids), 9), fit_role=np.array(roles))
    return data, runs


def test_fit_contract(tmp_path, monkeypatch):
    data, runs = make_synthetic(tmp_path)
    monkeypatch.setattr(I, "DATA_ROOT", data)
    monkeypatch.setattr(R, "RUN_ROOT", runs)
    s = F.fit_locus("m", "nih", "vis.last", seeds=(0, 1))
    f0 = F.load_fit("m", "nih", "vis.last", 0)
    D = 16
    for name, shape in {"projection": (D, 512), "scaler_mean": (512,), "scaler_scale": (512,), "coefficients": (6, 512),
                        "intercepts": (6,), "clinical_vectors": (6, D), "random_vectors": (119, D), "sham_vectors": (6, D),
                        "sham_permutations": (6, D), "control_coefficients": (6, 20, 512), "control_intercepts": (6, 20)}.items():
        assert f0[name].shape == shape, (name, f0[name].shape)
    # direction reconstruction rule
    for ci in range(6):
        raw = f0["projection"] @ (f0["coefficients"][ci] / np.maximum(f0["scaler_scale"], 1e-8))
        assert np.allclose(raw / np.linalg.norm(raw), f0["clinical_vectors"][ci], atol=1e-6)
        assert np.allclose(f0["clinical_vectors"][ci][f0["sham_permutations"][ci]], f0["sham_vectors"][ci])
    assert np.allclose(np.linalg.norm(f0["random_vectors"], axis=1), 1.0, atol=1e-6)
    # random family is the first draw of PCG64(seed 0), row-normalised; permutations follow it
    rng = np.random.default_rng(0)
    Rr = rng.standard_normal((119, D)); Rr /= np.linalg.norm(Rr, axis=1, keepdims=True)
    assert np.allclose(Rr.astype(np.float32), f0["random_vectors"], atol=1e-6)
    assert np.array_equal(rng.permutation(D), f0["sham_permutations"][0])
    # projection is PCG64(0) standard normal / sqrt(512)
    assert np.allclose(np.random.default_rng(0).standard_normal((D, 512)) / np.sqrt(512), f0["projection"], atol=1e-6)
    assert f0["fit_seed"] == 0 and f0["projection_seed"] == 0 and float(f0["C"]) == 1.0
    assert f0["type_names"].shape[0] >= 8 and bool(f0["controls_eligible"])
    # REFIT: resampled units with multiplicities, same projection, refit scaler/probes, sham via seed-0 permutation
    f1 = F.load_fit("m", "nih", "vis.last", 1)
    assert int(f1["train_unit_multiplicities"].sum()) == 30
    assert np.array_equal(f1["projection"], f0["projection"]) and not np.allclose(f1["scaler_mean"], f0["scaler_mean"])
    assert "random_vectors" not in f1 and np.array_equal(f1["sham_permutations"], f0["sham_permutations"])
    assert np.allclose(f1["clinical_vectors"][2][f1["sham_permutations"][2]], f1["sham_vectors"][2])
    # probe scores: eval rows x (6 real + 6x20 control) for seed 0, + 6 real for seed 1, + 2 nuisance
    t = pq.read_table(runs / "m" / "nih" / "probe_scores.vis.last.parquet").to_pandas()
    n_eval = 36
    assert (t.probe_kind == "real").sum() == n_eval * 6 * 2
    assert (t.probe_kind == "control").sum() == n_eval * 6 * 20
    assert (t.probe_kind == "nuisance").sum() == n_eval * 2
    assert set(t[t.probe_kind == "control"].control_seed.unique()) == set(range(20))
    assert s["seeds"]["0"]["Effusion"]["n_train_pos"] + s["seeds"]["0"]["Effusion"]["n_train_neg"] == 60
