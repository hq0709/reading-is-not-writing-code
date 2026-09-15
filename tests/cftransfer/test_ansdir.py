"""ANSDIR: protocol grid and enqueue prep task, the prep on the fake model (clean margins -> ridge -> lift), the
runner's seven steered conditions per question, and the analysis."""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from cftransfer import analysis as A, ansdir as AN, enqueue as E, fit as F, package as PK, protocol as P, runner as RN  # noqa: E402
from test_runner_fake import _setup                                                                                  # noqa: E402

CONCEPTS = P.CONCEPTS["nih"]


def test_ansdir_protocol_and_enqueue(tmp_path, monkeypatch):
    conds = P.conditions_for("ANSDIR", "nih", "Mass")
    assert [d for d, _ in conds] == [f"ans:{c}" for c in CONCEPTS] + ["anssham:Mass"] and {a for _, a in conds} == {0.25}
    assert {P.direction_kind(d) for d, _ in conds} == {"ans", "anssham"}
    assert all(P.expected_rows("ANSDIR", d) == 6 * 7 * 600 == 25_200 for d in P.DATASETS)
    assert P.MODULES["ANSDIR"].baseline_module == "CORE" and all("ANSDIR" in P.MODULES_ADDED_LATER[d] for d in P.DATASETS)
    assert "ANSDIR" in P.ADDENDUM_MODULES and E.SHARDS["ANSDIR"] == 1 and "ANSDIR" not in E.MODULE_ORDER
    assert P.ANSDIR_N_TRAIN == 3000 and P.ANSDIR_ALPHAS == (0.1, 1.0, 10.0, 100.0) and P.ANSDIR_CV_FOLDS == 5
    # enqueue emits the GPU prep task first, producing the npz the module task requires
    q = tmp_path / "queue"; (q / "pending").mkdir(parents=True)
    monkeypatch.setattr(E, "QUEUE", q)
    names = E.enqueue("q25-7", "nih", ["ANSDIR"], prep=False, prefix="1")
    assert len(names) == 2 and names[0].endswith("-ANSDIR-prep-q25-7-nih") and names[1].endswith("-ANSDIR-q25-7-nih-000of001")
    prep = json.loads((q / "pending" / f"{names[0]}.json").read_text())
    mod = json.loads((q / "pending" / f"{names[1]}.json").read_text())
    npz = str(P.REPO.parent / "cf-transfer" / "runs" / "q25-7" / "nih" / "fits" / "vis.last" / "ansdir_seed0.npz")
    assert prep["cmd"][:3] == ["python", "-m", "cftransfer.ansdir"] and "--n-train" in prep["cmd"] and "3000" in prep["cmd"]
    assert prep["produces"] == [npz] and npz not in prep["requires"] and npz in mod["requires"]
    assert mod["cmd"][2] == "cftransfer.runner" and "ANSDIR" in mod["cmd"] and "--numerics" not in mod["cmd"]
    # PRECISION enqueues one task per setting with the flag
    names = E.enqueue("q25-7", "coco", ["PRECISION"], prep=False, prefix="1")
    assert [n.split("-")[2:4] for n in names] == [["PRECISION", "fp32"], ["PRECISION", "batch1"]]
    t = json.loads((q / "pending" / f"{names[1]}.json").read_text())
    assert t["cmd"][-2:] == ["--numerics", "batch1"] and t["produces"][0].endswith("meta-batch1-000of001-*.json")


def test_ansdir_prep_runner_and_analysis(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(AN, "get_adapter", lambda key, rev=None: ad)
    monkeypatch.setattr(AN, "image_path", lambda ds, row: row["row_id"])
    monkeypatch.setattr(AN, "open_rgb", lambda p: p)
    rd = runs / "m" / "nih"
    with pytest.raises(FileNotFoundError, match="cftransfer.ansdir"):
        RN.run_block("m", "nih", "ANSDIR", 0, 1, batch=8, device_map="cpu", rows_per_part=2, limit_rows=2)
    assert not (rd / "outcomes" / "ANSDIR").exists()
    with pytest.raises(FileNotFoundError, match="fit-only"):
        AN.build_ansdir("m", "nih", n_train=20, device_map="cpu", fit_only=True)
    path, arr = AN.build_ansdir("m", "nih", n_train=20, device_map="cpu")
    assert path == rd / "fits" / "vis.last" / "ansdir_seed0.npz" and path.exists()
    # prep margins: the first 20 training rows in manifest order, six questions each, fp32 margins
    prep = pq.read_table(rd / "outcomes" / "ANSDIR_prep.parquet").to_pandas()
    rows = [r["row_id"] for r in AN.train_rows("nih", 20)]
    from cftransfer.images import load_cohort
    assert rows == [r["row_id"] for r in load_cohort("nih", ("train",))[:20]]          # manifest order, role train
    assert len(prep) == 20 * 6 and list(prep.row_id.unique()) == rows and set(prep.template_id) == {"IY"} and (prep.sample_status == "OK").all()
    assert np.allclose(prep.p_present, 1 / (1 + np.exp(-prep.semantic_margin.astype(float))))
    # ridge fit: alpha by 5-fold CV over the protocol grid on the seed-0 projected, scaled features; lifted like the normals
    f0 = F.load_fit("m", "nih", "vis.last", 0)
    ids, X, _v, _r = F.load_features("m", "nih", "vis.last")
    pos = {r: i for i, r in enumerate(ids)}
    Zs = ((X[[pos[r] for r in rows]].astype(np.float32) @ f0["projection"] - f0["scaler_mean"]) / np.maximum(f0["scaler_scale"], 1e-8))
    assert arr["answer_vectors"].shape == (6, 16) and np.allclose(np.linalg.norm(arr["answer_vectors"], axis=1), 1, atol=1e-5)
    for qi, q in enumerate(CONCEPTS):
        y = prep[prep.concept == q].set_index("row_id").semantic_margin.loc[rows].values.astype(float)
        kf = KFold(5, shuffle=True, random_state=0)
        sc = {a: cross_val_score(Ridge(alpha=a), Zs.astype(float), y, cv=kf, scoring="r2").mean() for a in P.ANSDIR_ALPHAS}
        best = max(P.ANSDIR_ALPHAS, key=lambda a: (sc[a], -a))
        assert arr["ridge_alpha"][qi] == best and abs(arr["cv_r2"][qi] - sc[best]) < 1e-9
        w = Ridge(alpha=best).fit(Zs.astype(float), y).coef_
        assert np.allclose(arr["answer_coefficients"][qi], w, atol=1e-5)
        assert np.allclose(arr["answer_vectors"][qi], F.direction_from_projected(f0["projection"], f0["scaler_scale"], w.astype(np.float32)), atol=1e-5)
        assert np.allclose(arr["answer_sham_vectors"][qi], arr["answer_vectors"][qi][f0["sham_permutations"][qi]])
        assert abs(arr["cos_to_logistic"][qi] - float(arr["answer_vectors"][qi] @ f0["clinical_vectors"][qi])) < 1e-6
    assert arr["cos_model"].shape == (6, 6) and int(arr["n_train_rows"]) == 20 and (arr["n_rows_used"] == 20).all()
    # --fit-only reproduces the file from the stored margins without the model
    monkeypatch.setattr(AN, "get_adapter", lambda key, rev=None: (_ for _ in ()).throw(AssertionError("model loaded")))
    _p2, arr2 = AN.build_ansdir("m", "nih", n_train=20, device_map="cpu", fit_only=True)
    assert np.array_equal(arr2["answer_vectors"], arr["answer_vectors"]) and np.array_equal(arr2["ridge_alpha"], arr["ridge_alpha"])
    # runner: seven steered conditions per question, no baseline
    meta = RN.run_block("m", "nih", "ANSDIR", 0, 1, batch=8, device_map="cpu", rows_per_part=4)
    assert meta["outcomes"] == 12 * 6 * 7
    df = pq.read_table(next((rd / "outcomes" / "ANSDIR").glob("part-*.parquet"))).to_pandas()
    assert set(df.direction_kind) == {"ans", "anssham"} and "baseline" not in set(df.direction_id) and (df.delta_norm_mean > 0).all()
    for (rid, q), g in df.groupby(["row_id", "concept"]):
        assert sorted(g.direction_id) == sorted(d for d, _ in P.conditions_for("ANSDIR", "nih", q))
    # analysis against CORE
    RN.run_block("m", "nih", "CORE", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    PK.build("m", "nih")
    res = A.ansdir("m", "nih", draws=100)
    core = A.core("m", "nih", n_boot=50)
    ans = pq.read_table(rd / "outcomes" / "ANSDIR.parquet").to_pandas()
    cdf = pq.read_table(rd / "outcomes" / "CORE.parquet").to_pandas()
    base = cdf[cdf.direction_id == "baseline"].set_index(["concept", "row_id"]).p_present
    assert res["n_train_rows"] == 20 and len(res["contrasts"]) == 30 and set(res["cos_model"]) == set(CONCEPTS)
    for q in CONCEPTS:
        cell = res["per_question"][q]
        g = ans[(ans.concept == q) & (ans.direction_id == f"ans:{q}")]
        assert abs(cell["W_qq"] - float(np.mean(g.p_present.values - base.loc[[(q, r) for r in g.row_id]].values))) < 1e-9
        s = ans[(ans.concept == q) & (ans.direction_id == f"anssham:{q}")]
        assert abs(cell["abs_sham"] - abs(float(np.mean(s.p_present.values - base.loc[[(q, r) for r in s.row_id]].values)))) < 1e-9
        assert abs(cell["random_p95"] - core["per_question"][q]["random_p95"]) < 1e-9
        assert abs(cell["own_minus_max_logistic_competitor"] - (cell["W_qq"] - core["per_question"][q]["max_other_clinical"])) < 1e-9
        assert cell["steering_reference"] == (cell["W_qq"] > 0 and cell["W_qq"] > cell["random_p95"] and cell["W_qq"] > cell["abs_sham"])
        assert cell["verdict"] in ("stronger_competitor", "fixed_family_advantage", "unresolved") and len(cell["O_q_ci95_percentile"]) == 2
        assert -1 <= cell["cos_to_logistic_model"] <= 1 and cell["ridge_alpha"] in P.ANSDIR_ALPHAS and cell["n_rows_used"] == 20
    assert "ANSDIR" in PK.requested_modules("nih", rd) and (rd / "outcomes" / "ANSDIR.parquet").exists()
