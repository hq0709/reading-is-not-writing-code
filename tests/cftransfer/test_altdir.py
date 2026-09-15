"""ALTDIR: direction constructions on synthetic data, the protocol grid, the runner's steered-only batches, the
package bookkeeping, the analysis statistics and the leaderboard column."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from sklearn.preprocessing import StandardScaler

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from cftransfer import altdir as AD, analysis as A, enqueue as E, fit as F, leaderboard as LB, package as PK, protocol as P, \
    runner as RN  # noqa: E402
from test_runner_fake import _setup  # noqa: E402

CONCEPTS = P.CONCEPTS["nih"]


def _synthetic_fit(rng, n=240, D=16, others_independent=False):
    """A seed-0-style fit dict with its scaled training features and labels. With others_independent the feature
    columns are made exactly orthogonal to [1, y_2..y_6] in-sample (the labels of concepts 1..5) while y_1 is then
    read off the features, so residualising concept 0's features against the other five labels is a no-op. Concepts
    1..5 then carry no linear signal at all (the L2 logistic optimum is exactly w = 0, which no real fit produces), so
    they get random stand-in normals; nothing asserted about concept 0 depends on them."""
    P_ = (rng.standard_normal((D, P.PROJECTION_DIM)) / np.sqrt(P.PROJECTION_DIM)).astype(np.float32)
    Y = rng.integers(0, 2, size=(len(CONCEPTS), n)).astype(np.int8)
    X = rng.standard_normal((n, D)).astype(np.float32)
    Z = X @ P_
    if others_independent:
        Aq = np.column_stack([np.ones(n), Y[1:].T.astype(float)])
        Z = (Z - Aq @ np.linalg.lstsq(Aq, Z.astype(float), rcond=None)[0]).astype(np.float32)
        Y[0] = (Z[:, :8].sum(axis=1) + 0.3 * rng.standard_normal(n) > 0).astype(np.int8)
    sc = StandardScaler().fit(Z)
    mu, s = sc.mean_.astype(np.float32), sc.scale_.astype(np.float32)
    Zs = ((Z - mu) / np.maximum(s, 1e-8)).astype(np.float32)
    coef = np.stack([F.fit_lr(Zs, Y[ci]).coef_[0] for ci in range(len(CONCEPTS))]).astype(np.float32)
    if others_independent:
        # concepts 1..5 have no linear signal (their fits are zero up to float32 rounding); use random stand-in normals
        assert np.linalg.norm(coef[0]) > 1e-2 and all(np.linalg.norm(coef[ci]) < 1e-3 for ci in range(1, len(CONCEPTS)))
        coef[1:] = rng.standard_normal((len(CONCEPTS) - 1, P.PROJECTION_DIM)).astype(np.float32)
    fit = {"projection": P_, "scaler_mean": mu, "scaler_scale": s, "coefficients": coef,
           "concept_names": np.array(CONCEPTS), "train_row_ids": np.array([f"r{i}" for i in range(n)]),
           "real_train_mask": np.ones((len(CONCEPTS), n), bool),
           "clinical_vectors": np.stack([F.direction_from_projected(P_, s, coef[ci]) for ci in range(len(CONCEPTS))])}
    return fit, Zs, Y


def test_direction_constructions_synthetic():
    rng = np.random.default_rng(3)
    fit, Zs, Y = _synthetic_fit(rng)
    arr = AD.altdir_arrays(fit, Zs, Y)
    P_, s, coef = fit["projection"], fit["scaler_scale"], fit["coefficients"]
    Sigma = np.cov(Zs.astype(np.float64), rowvar=False)
    for ci in range(len(CONCEPTS)):
        # dom is the difference of class means; pattern is Sigma w
        assert np.allclose(arr["dom_projected"][ci], Zs[Y[ci] == 1].mean(0) - Zs[Y[ci] == 0].mean(0), atol=1e-5)
        assert np.allclose(arr["pattern_projected"][ci], Sigma @ coef[ci].astype(np.float64), rtol=1e-4, atol=1e-5)
        # orth: orthogonal to every other logistic normal, removed part in their span, norm fraction recorded
        u = arr["orth_projected"][ci].astype(np.float64)
        others = np.delete(coef.astype(np.float64), ci, axis=0)
        assert np.all(np.abs(others @ u) < 1e-4 * np.linalg.norm(others, axis=1) * np.linalg.norm(u))
        removed = coef[ci].astype(np.float64) - u
        resid = removed - others.T @ np.linalg.lstsq(others.T, removed, rcond=None)[0]
        assert np.linalg.norm(resid) < 1e-4 * np.linalg.norm(coef[ci])
        assert abs(arr["orth_norm_removed_fraction"][ci] - (1 - np.linalg.norm(u) / np.linalg.norm(coef[ci]))) < 1e-6
        assert 0 < arr["orth_norm_removed_fraction"][ci] < 1
        # every family is lifted with the logistic rule, unit-normalised
        for fam in AD.FAMILY_ORDER:
            u_f = arr[f"{fam}_projected"][ci]
            raw = P_ @ (u_f / np.maximum(s, 1e-8))
            assert np.allclose(arr[f"{fam}_vectors"][ci], raw / np.linalg.norm(raw), atol=1e-6)
            assert np.allclose(arr[f"{fam}_vectors"][ci], F.direction_from_projected(P_, s, u_f), atol=1e-7)
            assert abs(np.linalg.norm(arr[f"{fam}_vectors"][ci]) - 1) < 1e-5
        assert np.allclose(arr["logistic_vectors"][ci], fit["clinical_vectors"][ci], atol=1e-6)
    # cosine matrices: symmetric with unit diagonal, over FAMILY_ORDER
    for key in ("cos_model", "cos_projected"):
        cm = arr[key]
        assert cm.shape == (6, 5, 5) and np.allclose(cm, cm.transpose(0, 2, 1)) and np.allclose(np.diagonal(cm, axis1=1, axis2=2), 1)
    assert list(arr["family_order"].astype(str)) == ["logistic", "dom", "pattern", "orth", "resid"]
    # the mask contract is enforced
    bad = dict(fit); bad["real_train_mask"] = np.zeros_like(fit["real_train_mask"])
    with pytest.raises(RuntimeError):
        AD.altdir_arrays(bad, Zs, Y)


def test_resid_on_independent_labels_equals_original():
    rng = np.random.default_rng(5)
    fit, Zs, Y = _synthetic_fit(rng, others_independent=True)
    arr = AD.altdir_arrays(fit, Zs, Y)
    # concept 0: the other five labels are orthogonal to the features, so residualising is a no-op and the refit
    # reproduces the original normal up to solver tolerance
    assert np.allclose(AD.residualise(Zs, Y[1:].T), Zs, atol=1e-4)
    assert arr["resid_cos_to_logistic_projected"][0] > 0.9999 and arr["cos_model"][0, 0, 4] > 0.9999
    assert np.allclose(arr["resid_projected"][0], fit["coefficients"][0], rtol=1e-3, atol=1e-4)
    assert (arr["resid_rows_used"] == Zs.shape[0]).all() and not arr["resid_fallback"].any()
    assert np.array_equal(arr["resid_covariates_used"], ~np.eye(6, dtype=bool))
    # a competitor label the features do carry changes concept 0's residualised probe
    assert not np.allclose(AD.residualise(Zs, np.delete(Y, 1, axis=0).T), Zs, atol=1e-3)
    Y2 = Y.copy(); Y2[1] = (Zs[:, 0] > 0).astype(np.int8)
    assert AD.altdir_arrays(fit, Zs, Y2)["resid_cos_to_logistic_projected"][0] < 0.9999
    # unknown labels (CheXpert) restrict resid to the all-known rows and the other families to their own known rows
    Yk = Y.copy(); Yk[2, :40] = -1
    fitk = dict(fit); fitk["real_train_mask"] = Yk >= 0            # normals kept: only the row masks matter here
    arrk = AD.altdir_arrays(fitk, Zs, Yk)
    assert (arrk["resid_rows_used"] == Zs.shape[0] - 40).all() and not arrk["resid_fallback"].any()   # covariate 2 kept, its rows required
    assert int(arrk["dom_n_pos"][2] + arrk["dom_n_neg"][2]) == Zs.shape[0] - 40


def test_resid_single_class_covariate_dropped_and_fallback():
    """CheXpert pattern: concept 3 has three known negatives, and concept 2 is unknown on exactly those rows, so on any
    all-known row set concept 3 is single-class. It must be dropped as a covariate (with the rows recomputed without it)
    and, as a question, fall back to the logistic normal; dom/pattern/orth are unchanged."""
    rng = np.random.default_rng(11)
    fit, Zs, Y = _synthetic_fit(rng)
    n = Zs.shape[0]
    Y = Y.copy(); Y[3] = 1; Y[3, :3] = 0; Y[2, :3] = -1
    fit = dict(fit); fit["real_train_mask"] = Y >= 0
    fit["coefficients"] = np.stack([F.fit_lr(Zs[Y[ci] >= 0], Y[ci][Y[ci] >= 0]).coef_[0] for ci in range(6)]).astype(np.float32)
    fit["clinical_vectors"] = np.stack([F.direction_from_projected(fit["projection"], fit["scaler_scale"], fit["coefficients"][ci]) for ci in range(6)])
    arr = AD.altdir_arrays(fit, Zs, Y)
    used = arr["resid_covariates_used"]
    for ci in (0, 1, 2, 4, 5):
        assert not used[ci, 3] and not used[ci, ci] and used[ci].sum() == 4          # covariate 3 dropped, the rest kept
        assert arr["resid_rows_used"][ci] == n - 3 and not arr["resid_fallback"][ci]  # rows: concept 2 known
        rows = Y[2] >= 0
        cov = [d for d in range(6) if d not in (ci, 3)]
        w = F.fit_lr(AD.residualise(Zs[rows], Y[cov][:, rows].T).astype(np.float32), Y[ci][rows]).coef_[0]
        assert np.allclose(arr["resid_projected"][ci], w, rtol=1e-4, atol=1e-6)
    assert arr["resid_fallback"][3] and arr["resid_rows_used"][3] == n - 3 and used[3].sum() == 5
    assert np.array_equal(arr["resid_projected"][3], fit["coefficients"][3]) and arr["resid_cos_to_logistic_projected"][3] == 1.0
    assert np.allclose(arr["resid_vectors"][3], arr["logistic_vectors"][3])
    # the direct rule: rows and covariates for concept 0
    rows, cov = AD.resid_covariate_rows(Y, 0)
    assert rows.sum() == n - 3 and cov.tolist() == [False, True, True, False, True, True]
    # dom uses each concept's own known rows; pattern and orth do not depend on labels
    assert int(arr["dom_n_pos"][3] + arr["dom_n_neg"][3]) == n and int(arr["dom_n_neg"][3]) == 3
    ref = AD.altdir_arrays({**fit, "real_train_mask": np.ones((6, n), bool)}, Zs, np.where(Y < 0, 0, Y))
    for fam in ("pattern", "orth"):
        assert np.array_equal(arr[f"{fam}_projected"], ref[f"{fam}_projected"])


def test_resid_row_retention():
    """A covariate known on too few rows (below 25% of the concept's own known rows) is dropped even though it is
    two-class there; the class-count rule is applied first when both fail; nothing else changes."""
    rng = np.random.default_rng(13)
    fit, Zs, Y = _synthetic_fit(rng)
    n = Zs.shape[0]
    Y = Y.copy(); Y[5, 50:] = -1                                  # covariate 5 known on 50 rows < 0.25 * 240
    fit = dict(fit); fit["real_train_mask"] = Y >= 0
    fit["coefficients"] = np.stack([F.fit_lr(Zs[Y[ci] >= 0], Y[ci][Y[ci] >= 0]).coef_[0] for ci in range(6)]).astype(np.float32)
    fit["clinical_vectors"] = np.stack([F.direction_from_projected(fit["projection"], fit["scaler_scale"], fit["coefficients"][ci]) for ci in range(6)])
    arr = AD.altdir_arrays(fit, Zs, Y)
    assert float(arr["resid_retention_fraction"]) == AD.RESID_RETENTION_FRACTION == 0.25
    for ci in range(5):
        assert arr["resid_covariates_used"][ci].tolist() == [d not in (ci, 5) for d in range(6)]
        assert arr["resid_rows_used"][ci] == n and not arr["resid_fallback"][ci]
    assert arr["resid_covariates_used"][5].sum() == 5 and arr["resid_rows_used"][5] == 50    # its own rows keep every covariate
    rows, cov = AD.resid_covariate_rows(Y, 0)
    assert rows.sum() == n and cov.tolist() == [False, True, True, True, True, False]
    rows_lo, cov_lo = AD.resid_covariate_rows(Y, 0, retention=0.2)                 # 50 >= 0.2 * 240: covariate 5 kept
    assert cov_lo[5] and cov_lo.sum() == 5 and rows_lo.sum() == 50
    # class-count shortfall is handled before retention: covariate 3 single-class on the 50 rows goes first, then 5
    Y2 = Y.copy(); Y2[3, :50] = 1
    rows2, cov2 = AD.resid_covariate_rows(Y2, 0)
    assert cov2.tolist() == [False, True, True, False, True, False] and rows2.sum() == n
    assert AD.resid_covariate_rows(Y2, 0, retention=0.0)[1].tolist() == [False, True, True, False, True, True]


def test_altdir_protocol_grid():
    conds = P.conditions_for("ALTDIR", "nih", "Mass")
    assert len(conds) == 24 and all(a == P.PRIMARY_ALPHA for _, a in conds) and "baseline" not in {d for d, _ in conds}
    assert [d for d, _ in conds] == [f"{f}:{c}" for f in ("dom", "pattern", "orth", "resid") for c in CONCEPTS]
    assert {P.direction_kind(d) for d, _ in conds} == {"dom", "pattern", "orth", "resid"}
    assert P.direction_kind("dom:Effusion") == "dom" and P.direction_kind("pattern:person") == "pattern"
    spec = P.MODULES["ALTDIR"]
    assert spec.role == "test" and spec.row_limit is None and spec.templates == ("IY",) and spec.fit_seeds == (0,)
    assert spec.locus == "primary" and spec.baseline_module == "CORE" and spec.datasets == ("nih", "chexpert", "coco")
    for ds in P.DATASETS:
        assert P.expected_rows("ALTDIR", ds) == 6 * 24 * 600 == 86_400
        assert "ALTDIR" in P.MODULES_ADDED_LATER[ds]
        assert P.question_list(ds, "ALTDIR", "IB") == [(c, "IB") for c in P.CONCEPTS[ds]]
    assert P.PROTOCOL["modules"]["ALTDIR"]["expected_rows"] == 86_400
    assert E.SHARDS["ALTDIR"] == 1 and "ALTDIR" not in E.MODULE_ORDER
    assert P.ALTDIR_FAMILIES == ("dom", "pattern", "orth", "resid")


def test_runner_altdir_steered_only_and_package(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    rd = runs / "m" / "nih"
    # the prep file is required before the model is loaded, with the prep command in the message
    with pytest.raises(FileNotFoundError, match="cftransfer.altdir"):
        RN.run_block("m", "nih", "ALTDIR", 0, 1, batch=8, device_map="cpu", rows_per_part=2, limit_rows=2)
    assert not (rd / "outcomes" / "ALTDIR").exists()                 # failed before touching disk
    assert "ALTDIR" not in PK.requested_modules("nih", rd)          # not yet started: packaged blocks stay COMPLETE
    path, arr = AD.build_altdir("m", "nih", "vis.last")
    assert path == rd / "fits" / "vis.last" / "altdir_seed0.npz" and path.exists()
    f0 = F.load_fit("m", "nih", "vis.last", 0)
    assert np.allclose(arr["logistic_vectors"], f0["clinical_vectors"], atol=1e-6)
    meta = RN.run_block("m", "nih", "ALTDIR", 0, 1, batch=8, device_map="cpu", rows_per_part=2, limit_rows=2)
    assert meta["outcomes"] == 2 * 6 * 24
    df = pq.read_table(next((rd / "outcomes" / "ALTDIR").glob("part-*.parquet"))).to_pandas()
    assert len(df) == 2 * 6 * 24 and "baseline" not in set(df.direction_id) and (df.sample_status == "OK").all()
    assert set(df.direction_kind) == {"dom", "pattern", "orth", "resid"} and (df.delta_norm_mean > 0).all()
    assert set(df.fit_seed) == {0} and set(df.alpha) == {0.25} and set(df.template_id) == {"IY"}
    for (rid, q), g in df.groupby(["row_id", "concept"]):
        assert sorted(g.direction_id) == sorted(d for d, _ in P.conditions_for("ALTDIR", "nih", q))
    # the four families steer differently for the same question
    q = df[(df.row_id == df.row_id.iloc[0]) & (df.concept == "Effusion")].set_index("direction_id").semantic_margin
    assert len({round(float(q[f"{fam}:Effusion"]), 6) for fam in ("dom", "pattern", "orth", "resid")}) == 4
    # package: coverage rows with the CORE baseline, merged parquet, requested once started
    run = PK.build("m", "nih")
    cov = [r for r in csv.DictReader((rd / "coverage.csv").open()) if r["module"] == "ALTDIR"]
    assert len(cov) == 6 and all(r["baseline_module"] == "CORE" and r["baseline_fit_seed"] == "0" for r in cov)
    assert all(r["execution_status"] == "RUNNING" and r["expected_rows"] == str(24 * 600) for r in cov)   # protocol rows
    assert (rd / "outcomes" / "ALTDIR.parquet").exists() and "ALTDIR" in run["requested_modules"]
    assert run["merged_outcomes"]["ALTDIR"]["rows_unique"] == 2 * 6 * 24


def test_analysis_altdir_statistics(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    rd = runs / "m" / "nih"
    AD.build_altdir("m", "nih", "vis.last")
    RN.run_block("m", "nih", "CORE", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    RN.run_block("m", "nih", "ALTDIR", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    PK.build("m", "nih")
    res = A.altdir("m", "nih", draws=150)
    assert res["families"] == ["dom", "pattern", "orth", "resid"] and res["draws"] == 150 and res["template_id"] == "IY"
    core = pq.read_table(rd / "outcomes" / "CORE.parquet").to_pandas()
    alt = pq.read_table(rd / "outcomes" / "ALTDIR.parquet").to_pandas()
    base = core[core.direction_id == "baseline"].set_index(["concept", "row_id"]).p_present
    for fam in res["families"]:
        r = res[fam]
        assert set(r["W"]) == set(CONCEPTS) and all(set(v) == set(CONCEPTS) for v in r["W"].values())
        assert len(r["contrasts"]) == 30 and r["n_scored_rows_min"] == 12
        for q in CONCEPTS:
            cell = r["per_question"][q]
            g = alt[(alt.concept == q) & (alt.direction_id == f"{fam}:{q}")]
            w_manual = float(np.mean(g.p_present.values - base.loc[[(q, x) for x in g.row_id]].values))
            assert abs(cell["W_qq"] - w_manual) < 1e-9 and abs(r["W"][q][q] - w_manual) < 1e-9
            assert abs(cell["O_q"] - (cell["W_qq"] - max(r["W"][q][d] for d in CONCEPTS if d != q))) < 1e-12
            assert cell["verdict"] in ("stronger_competitor", "fixed_family_advantage", "unresolved")
            lo, hi = cell["O_q_ci95_percentile"]; assert lo <= hi
            ref = res["logistic_reference"][q]
            assert abs(cell["own_minus_max_logistic_competitor"] - (cell["W_qq"] - ref["max_other_logistic"])) < 1e-12
            assert len(cell["own_minus_max_logistic_competitor_ci95_percentile"]) == 2
            assert -1 <= cell["cos_to_logistic_model"] <= 1 and isinstance(cell["steering_reference"], bool)
            if fam == "orth":
                assert 0 < cell["norm_removed_fraction"] < 1
    # the logistic reference matches analysis.core on the same block
    c = A.core("m", "nih", n_boot=50)
    for q in CONCEPTS:
        assert abs(res["logistic_reference"][q]["W_logistic_qq"] - c["per_question"][q]["W_qq"]) < 1e-9
        assert abs(res["logistic_reference"][q]["random_p95"] - c["per_question"][q]["random_p95"]) < 1e-9
    assert set(res["cosines"]) == set(CONCEPTS) and set(res["cosines"]["Mass"]["model"]) == set(AD.FAMILY_ORDER)
    # summary.json round trip through the CLI writer path and the leaderboard column
    summary = {"core": c, "altdir": res, "calibration": {}}
    (rd / "summary.json").write_text(json.dumps(summary, default=float))
    row = LB.grade_block(json.loads((rd / "summary.json").read_text()), {"status": "RUNNING"}, None)
    assert LB.ALTDIR_COL in row and len(row[LB.ALTDIR_COL].split("/")) == 4
    assert LB.ALTDIR_COL not in LB.grade_block({"core": c, "calibration": {}}, None, None)
    monkeypatch.setattr(LB, "MODELS", {"m": {}})
    LB.main(runs)
    header = (runs / "leaderboard.csv").read_text().splitlines()[0]
    assert header.endswith("," + LB.ALTDIR_COL)
