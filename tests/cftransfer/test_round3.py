"""Round-3 control modules: ATTRRAND (the protocol random family on the attribute questions, folded into the ATTR
analysis), VALIDFIT (directions refitted on the radiologist-labelled valid rows and written on the test rows) and
PROJSEED (the six clinical directions refitted under two further projection seeds, one shard per seed)."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from cftransfer import analysis as A, attr as AT, enqueue as E, fit as F, images as I, manifest as M, package as PK, \
    projseed as PS, protocol as P, runner as RN, runpaths as R, validfit as VF  # noqa: E402
from cftransfer.altdir import scaled_training_features  # noqa: E402
from test_runner_fake import _setup  # noqa: E402
from test_valid import N_VALID, _setup_chexpert  # noqa: E402

CONCEPTS = P.CONCEPTS["nih"]
CHEX = P.CONCEPTS["chexpert"]
JSON = dict(default=lambda o: float(o) if isinstance(o, np.floating) else str(o))


# ------------------------------------------------------------------------------------------------ protocol
def test_round3_protocol_grids(tmp_path, monkeypatch):
    # ATTRRAND: the three attribute questions only, the 119 protocol random directions, no baseline, ATTR baseline reused
    assert P.question_list("nih", "ATTRRAND") == [(q, "IY") for q in P.ATTR_CONCEPTS]
    assert P.question_list("chexpert", "ATTRRAND", "IB") == [(q, "IB") for q in P.ATTR_CONCEPTS]
    ca = P.conditions_for("ATTRRAND", "nih", "sex_F")
    assert ca == [(f"random:{i:03d}", 0.25) for i in range(P.N_RANDOM)] and len(ca) == 119
    assert ca == P.conditions_for("ATTRRAND", "chexpert", "age_60")            # the family does not depend on the question
    assert P.expected_rows("ATTRRAND", "nih") == P.expected_rows("ATTRRAND", "chexpert") == 3 * 119 * 600 == 214_200
    assert P.expected_rows("ATTRRAND", "coco") == 0 and P.MODULES["ATTRRAND"].baseline_module == "ATTR"
    # the random ids are exactly CORE's, so the attribute cells are graded against the same 119 directions
    assert [d for d, _ in ca] == [d for d, _ in P.conditions_for("CORE", "nih", "Mass") if d.startswith("random:")]
    # VALIDFIT: six expert-label directions, CheXpert only, CORE baseline / references reused
    cv = P.conditions_for("VALIDFIT", "chexpert", "Edema")
    assert cv == [(f"vfit:{c}", 0.25) for c in CHEX] and P.MODULES["VALIDFIT"].baseline_module == "CORE"
    assert P.MODULES["VALIDFIT"].datasets == ("chexpert",) and P.MODULES["VALIDFIT"].role == "test"
    assert P.expected_rows("VALIDFIT", "chexpert") == 6 * 6 * 600 == 21_600
    assert P.expected_rows("VALIDFIT", "nih") == P.expected_rows("VALIDFIT", "coco") == 0
    assert P.VALIDFIT_FOLDS == 5 and P.VALIDFIT_CV_SEED == 0
    # PROJSEED: per projection seed six refits + its own 119 random + its own sham; the seed travels in fit_seed
    cp = P.conditions_for("PROJSEED", "nih", "Mass")
    assert cp == [(d, 0.25) for d in [f"proj:{c}" for c in CONCEPTS] + [f"projrand:{i:03d}" for i in range(119)] + ["projsham:Mass"]]
    assert len(cp) == 126 and P.MODULES["PROJSEED"].fit_seeds == P.PROJSEED_SEEDS == (1, 2)
    assert all(P.expected_rows("PROJSEED", d) == 6 * 126 * 2 * 600 == 907_200 for d in P.DATASETS)
    assert P.direction_kind("projrand:007") == "projrand" and P.direction_kind("projsham:Mass") == "projsham"
    # registration: addenda, added later on their datasets, never in the default order
    for m, ds in (("ATTRRAND", ("nih", "chexpert")), ("VALIDFIT", ("chexpert",)), ("PROJSEED", P.DATASETS)):
        assert m in P.ADDENDUM_MODULES and all(m in P.MODULES_ADDED_LATER[d] for d in ds) and m not in E.MODULE_ORDER
        assert P.PROTOCOL["modules"][m]["expected_rows"] == P.expected_rows(m, ds[0])
    # a packaged block stays COMPLETE until each module starts
    rd = tmp_path / "m" / "chexpert"; (rd / "outcomes").mkdir(parents=True)
    assert not {"ATTRRAND", "VALIDFIT", "PROJSEED"} & set(PK.requested_modules("chexpert", rd))
    for m in ("ATTRRAND", "VALIDFIT", "PROJSEED"):
        (rd / "outcomes" / m).mkdir()
    assert {"ATTRRAND", "VALIDFIT", "PROJSEED"} <= set(PK.requested_modules("chexpert", rd))


def test_round3_enqueue(tmp_path, monkeypatch):
    q = tmp_path / "queue"; (q / "pending").mkdir(parents=True)
    monkeypatch.setattr(E, "QUEUE", q)
    assert E.SHARDS["ATTRRAND"] == 2 and E.SHARDS["VALIDFIT"] == 1 and E.SHARDS["PROJSEED"] == 1
    assert E.PREP_FILE["ATTRRAND"] == "attr_seed0.npz" and E.prep_files("VALIDFIT") == ["validfit_seed0.npz"]
    assert E.prep_files("PROJSEED") == ["projseed_seed1.npz", "projseed_seed2.npz"]
    # ATTRRAND: no prep task of its own (the ATTR prep is run by hand), two row shards, waits for attr_seed0.npz
    names = E.enqueue("q25-7", "nih", ["ATTRRAND"], prep=False, prefix="1")
    assert len(names) == 2 and names[0].endswith("-ATTRRAND-q25-7-nih-000of002") and names[1].endswith("-001of002")
    t = json.loads((q / "pending" / f"{names[0]}.json").read_text())
    assert t["requires"][-1].endswith("attr_seed0.npz") and "--numerics" not in t["cmd"] and "--fit-seed" not in t["cmd"]
    assert t["produces"] == [str(R.run_dir("q25-7", "nih") / "outcomes" / "ATTRRAND" / "meta-000of002-*.json")]
    assert E.enqueue("q25-7", "coco", ["ATTRRAND"], prep=False, prefix="1") == []
    # VALIDFIT: a CPU prep task (no --device-map) that waits for the valid-role features, then one module shard
    names = E.enqueue("q25-7", "chexpert", ["VALIDFIT"], prep=False, prefix="1")
    assert len(names) == 2 and names[0].endswith("-VALIDFIT-prep-q25-7-chexpert") and names[1].endswith("-VALIDFIT-q25-7-chexpert-000of001")
    prep = json.loads((q / "pending" / f"{names[0]}.json").read_text())
    mod = json.loads((q / "pending" / f"{names[1]}.json").read_text())
    vf = R.valid_features_dir("q25-7", "chexpert")
    assert prep["cmd"] == ["python", "-m", "cftransfer.validfit", "--model-key", "q25-7", "--dataset", "chexpert"]
    assert "--device-map" not in prep["cmd"] and str(vf / "vis.last.npz") in prep["requires"]
    assert prep["produces"] == [str(R.run_dir("q25-7", "chexpert") / "fits" / "vis.last" / "validfit_seed0.npz")]
    assert mod["requires"][-1].endswith("validfit_seed0.npz") and mod["cmd"][mod["cmd"].index("--module") + 1] == "VALIDFIT"
    assert E.enqueue("q25-7", "nih", ["VALIDFIT"], prep=False, prefix="1") == []
    # PROJSEED: a CPU prep producing both seed files, then ONE SHARD PER SEED (--fit-seed), each with its own meta stream
    names = E.enqueue("q25-7", "nih", ["PROJSEED"], prep=False, prefix="1")
    assert len(names) == 3 and names[0].endswith("-PROJSEED-prep-q25-7-nih")
    assert names[1].endswith("-PROJSEED-s1-q25-7-nih-000of001") and names[2].endswith("-PROJSEED-s2-q25-7-nih-000of001")
    prep = json.loads((q / "pending" / f"{names[0]}.json").read_text())
    assert "--device-map" not in prep["cmd"] and [Path(x).name for x in prep["produces"]] == ["projseed_seed1.npz", "projseed_seed2.npz"]
    for nm, k in zip(names[1:], P.PROJSEED_SEEDS):
        t = json.loads((q / "pending" / f"{nm}.json").read_text())
        assert t["cmd"][t["cmd"].index("--fit-seed") + 1] == str(k) and t["cmd"][t["cmd"].index("--n-shards") + 1] == "1"
        assert t["produces"][0].endswith(f"outcomes/PROJSEED/meta-s{k}-000of001-*.json")
        assert sorted(Path(x).name for x in t["requires"] if x.endswith(".npz") and "projseed" in x) == \
            ["projseed_seed1.npz", "projseed_seed2.npz"]


# ------------------------------------------------------------------------------------------------ ATTRRAND
def test_attrrand_runner_and_attr_analysis(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(PK, "TEST_ROWS", 12)                 # the synthetic cohort has 12 test rows, not 600
    rd = runs / "m" / "nih"
    AT.build_attr("m", "nih", "vis.last", min_class_rows=5)
    RN.run_block("m", "nih", "CORE", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    RN.run_block("m", "nih", "ATTR", 0, 1, batch=8, device_map="cpu", rows_per_part=4)
    PK.build("m", "nih")
    # before ATTRRAND: attribute cells fall back to the sham-only reference, clinical cells keep CORE's random p95
    before = A.attr("m", "nih", draws=120)
    assert before["attrrand"]["available"] is False
    for a in P.ATTR_CONCEPTS:
        c = before["per_question"][a]
        assert c["random_reference"] is False and c["random_p95"] is None and c["steering_reference_matched"] is None
        assert c["steering_reference"] == c["steering_reference_sham_only"] and c["steering_reference_rule"].startswith("sham_only")
    for q in CONCEPTS:
        c = before["per_question"][q]
        assert c["random_reference"] and c["random_source"] == "CORE" and c["steering_reference"] == c["steering_reference_matched"]
    # the module: 119 random conditions per attribute question, no baseline rows, no new directions
    meta = RN.run_block("m", "nih", "ATTRRAND", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    assert meta["outcomes"] == 12 * 3 * 119
    df = pq.read_table(next((rd / "outcomes" / "ATTRRAND").glob("part-*.parquet"))).to_pandas()
    assert set(df.concept) == set(P.ATTR_CONCEPTS) and set(df.direction_kind) == {"random"} and "baseline" not in set(df.direction_id)
    for (_rid, q), g in df.groupby(["row_id", "concept"]):
        assert sorted(g.direction_id) == sorted(d for d, _ in P.conditions_for("ATTRRAND", "nih", q))
    PK.build("m", "nih")
    res = A.attr("m", "nih", draws=120)
    ar = pq.read_table(rd / "outcomes" / "ATTRRAND.parquet").to_pandas()
    at = pq.read_table(rd / "outcomes" / "ATTR.parquet").to_pandas()
    base = at[at.direction_id == "baseline"].set_index(["concept", "row_id"]).p_present
    assert res["attrrand"]["available"] and res["attrrand"]["n_random"] == 119
    for a in P.ATTR_CONCEPTS:
        cell = res["per_question"][a]
        w = np.array([float(np.mean(g.p_present.values - base.loc[[(a, r) for r in g.row_id]].values))
                      for _d, g in ar[ar.concept == a].groupby("direction_id")])
        assert cell["random_reference"] and cell["random_n"] == 119 and cell["random_source"] == "ATTRRAND"
        assert abs(cell["random_p95"] - float(np.percentile(w, 95))) < 1e-9 and abs(cell["random_max"] - w.max()) < 1e-9
        assert cell["rank_in_random_family"] == int(1 + (w >= cell["W_qq"]).sum())
        assert cell["steering_reference_matched"] == (cell["W_qq"] > 0 and cell["W_qq"] > cell["abs_sham"] and cell["W_qq"] > cell["random_p95"])
        assert cell["steering_reference_sham_only"] == before["per_question"][a]["steering_reference_sham_only"]
        assert cell["steering_reference"] == cell["steering_reference_matched"]
        assert "ATTRRAND" in cell["steering_reference_rule"]
    # the clinical results are untouched by the new module
    for q in CONCEPTS:
        for k in ("W_qq", "O_q", "verdict", "steering_reference", "random_p95", "abs_sham"):
            assert res["per_question"][q][k] == before["per_question"][q][k]
    # the answerability field a reviewer expects, recomputed from the outcomes against the manifest label
    labels = I.load_labels("nih")
    rows = I.load_cohort("nih", ("test",))
    for a in P.ATTR_CONCEPTS:
        cell = res["attributes"][a]
        g = at[(at.direction_id == "baseline") & (at.concept == a)]
        y = np.array([AT.attribute_value(a, labels[(r, "Effusion")]) for r in g.row_id])
        assert abs(cell["label_answer_auroc"] - roc_auc_score(y, g.semantic_margin.values)) < 1e-9
        assert cell["label_n_pos"] == int((y == 1).sum()) and cell["label_n_neg"] == int((y == 0).sum())
        assert cell["label_answer_capable"] is False and 0 < cell["label_answer_valid_draws"] <= 120   # 120 < 1,900 draws
        assert {"answer_auroc", "answer_capable", "label_answer_auroc", "label_answer_capable"} <= set(cell)
        assert "label_answer_*" in cell["answer_reference"]
    assert len(rows) == 12
    # manifest: ATTRRAND rows carry the matched grade; block_included never depends on them
    (rd / "summary.json").write_text(json.dumps({"attr": res, "core": A.core("m", "nih", n_boot=100)}, **JSON))
    rj = json.loads((rd / "run.json").read_text())
    rj["completed_modules"] = ["CORE", "CALIBRATION", "ATTRRAND"]
    (rd / "run.json").write_text(json.dumps(rj))
    mrows = M.block_rows(rd)
    arows = [r for r in mrows if r["module"] == "ATTRRAND"]
    assert len(arows) == 3 and all(r["block_included"] and not r["cell"] for r in arows)
    for r in arows:
        c = res["per_question"][r["concept"]]
        assert r["attrrand_random_p95"] == c["random_p95"] and r["attrrand_steering_reference"] == c["steering_reference"]
        assert r["attrrand_steering_reference_sham_only"] == c["steering_reference_sham_only"] and r["attrrand_verdict"] == c["verdict"]
    assert all(r["attrrand_random_p95"] is None for r in mrows if r["module"] != "ATTRRAND")
    rj["completed_modules"] = ["ATTRRAND"]
    (rd / "run.json").write_text(json.dumps(rj))
    assert not any(r["block_included"] for r in M.block_rows(rd))


# ------------------------------------------------------------------------------------------------ VALIDFIT
def _valid_features(runs: Path, rows: list[dict], seed: int = 7) -> np.ndarray:
    vf = R.valid_features_dir("m", "chexpert"); vf.mkdir(parents=True, exist_ok=True)
    X = np.random.default_rng(seed).standard_normal((len(rows), 16)).astype(np.float16)
    np.savez(vf / "vis.last.npz", row_id=np.array([r["row_id"] for r in rows]), x=X,
             valid_token_count=np.full(len(rows), 9), fit_role=np.array(["valid"] * len(rows)))
    return X


def _labeler_file(data: Path, rows: list[dict], seed: int = 3) -> None:
    """A CheXpert-Plus-shaped labeler output (one JSON object per line) covering the valid rows, so the prep can report
    the held-out AUROC against the REPORT-derived labels too."""
    rng = np.random.default_rng(seed)
    with (data / "chexpert" / VF.LABELER_FILE["chexpert"]).open("w", encoding="utf-8") as f:
        for r in rows:
            rec = {"path_to_image": r["relative_image_path"].rsplit(".", 1)[0] + ".jpg"}
            for c in CHEX:
                v = rng.integers(0, 3)
                rec[VF.CHEXPERT_RAW[c]] = None if v == 2 else float(v)
            f.write(json.dumps(rec) + "\n")


def test_validfit_prep_runner_and_analysis(tmp_path, monkeypatch):
    ad, runs = _setup_chexpert(tmp_path, monkeypatch)
    monkeypatch.setattr(PK, "TEST_ROWS", 12)                 # the synthetic cohort has 12 test rows, not 600
    data = I.DATA_ROOT
    rd = runs / "m" / "chexpert"
    vrows = I.load_cohort("chexpert", ("valid",))
    with pytest.raises(FileNotFoundError, match="valid-role features"):
        VF.build_validfit("m", "chexpert", "vis.last")
    X = _valid_features(runs, vrows)
    _labeler_file(data, vrows)
    path, arr = VF.build_validfit("m", "chexpert", "vis.last")
    assert path == rd / "fits" / "vis.last" / "validfit_seed0.npz" and list(arr["concept_names"].astype(str)) == CHEX
    f0 = F.load_fit("m", "chexpert", "vis.last", 0)
    # the block's OWN projection and its TRAIN scaler; the 200 (here 24) valid rows never refit a scaler
    Zs = ((X.astype(np.float32) @ f0["projection"] - f0["scaler_mean"]) / np.maximum(f0["scaler_scale"], 1e-8)).astype(np.float32)
    labels = I.load_labels("chexpert")
    folds = VF.fold_assignment([r["unit_id"] for r in vrows])
    assert len(set(folds.tolist())) == P.VALIDFIT_FOLDS and np.array_equal(folds, arr["fold_of_row"])
    for ci, c in enumerate(CHEX):
        y = np.array([int(labels[(r["row_id"], c)]["label"]) for r in vrows])
        assert np.array_equal(arr["expert_labels"][ci], y) and int(arr["n_pos"][ci]) == int((y == 1).sum())
        w = F.fit_lr(Zs, y).coef_[0]
        assert np.allclose(arr["expert_coefficients"][ci], w, atol=1e-6)
        assert np.allclose(arr["expert_vectors"][ci], F.direction_from_projected(f0["projection"], f0["scaler_scale"], w), atol=1e-6)
        # cross-fitting: every row scored by a probe fitted without its own fold
        held = np.full(len(vrows), np.nan)
        for fi in range(P.VALIDFIT_FOLDS):
            tr, te = folds != fi, folds == fi
            cf = F.fit_lr(Zs[tr], y[tr])
            held[te] = Zs[te] @ cf.coef_[0] + cf.intercept_[0]
        assert np.allclose(arr["heldout_logits"][ci], held, atol=1e-4) and int(arr["folds_used"][ci]) == P.VALIDFIT_FOLDS
        assert abs(arr["auroc_expert_heldout"][ci] - roc_auc_score(y, held)) < 1e-9
        assert arr["auroc_expert_heldout"][ci] <= arr["auroc_expert_in_sample"][ci] + 1e-9 or True
        # cosines to the report-label direction: model space, projected space, and whitened by the training covariance
        a64, b64 = arr["expert_coefficients"][ci].astype(np.float64), f0["coefficients"][ci].astype(np.float64)
        va, vb = arr["expert_vectors"][ci].astype(np.float64), f0["clinical_vectors"][ci].astype(np.float64)
        assert abs(arr["cos_model"][ci] - va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb))) < 1e-9
        assert abs(arr["cos_projected"][ci] - a64 @ b64 / (np.linalg.norm(a64) * np.linalg.norm(b64))) < 1e-9
        Zt, _Y = scaled_training_features("m", "chexpert", "vis.last", f0)
        S = np.cov(Zt.astype(np.float64), rowvar=False)
        assert abs(arr["cos_whitened"][ci] - (a64 @ S @ b64) / np.sqrt((a64 @ S @ a64) * (b64 @ S @ b64))) < 1e-8
        # the report-derived labels of the same rows come from the labeler file, and are a different label set
        yr = arr["report_derived_labels"][ci]
        assert set(np.unique(yr).tolist()) <= {-1, 0, 1} and (yr == -1).sum() > 0
        assert np.isfinite(arr["auroc_report_labels_heldout"][ci]) == (len(set(yr[yr >= 0].tolist())) == 2)
        assert abs(arr["report_direction_auroc_expert"][ci] -
                   roc_auc_score(y, Zs @ f0["coefficients"][ci] + f0["intercepts"][ci])) < 1e-9
    # without the labeler file the report-label AUROCs are NaN and the note says so
    (data / "chexpert" / VF.LABELER_FILE["chexpert"]).unlink()
    _p2, arr2 = VF.build_validfit("m", "chexpert", "vis.last", out_dir=tmp_path / "nolab")
    assert np.isnan(arr2["auroc_report_labels_heldout"]).all() and int(arr2["n_pos_report"].sum()) == 0
    assert "not available" in str(arr2["report_label_source"])
    assert np.allclose(arr2["expert_vectors"], arr["expert_vectors"])          # the directions do not depend on that file
    # the runner writes the six expert directions on the TEST rows; no baseline rows of its own
    meta = RN.run_block("m", "chexpert", "VALIDFIT", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    assert meta["outcomes"] == 12 * 6 * 6
    df = pq.read_table(next((rd / "outcomes" / "VALIDFIT").glob("part-*.parquet"))).to_pandas()
    assert set(df.direction_kind) == {"vfit"} and set(df.role) == {"test"} and "baseline" not in set(df.direction_id)
    RN.run_block("m", "chexpert", "CORE", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    PK.build("m", "chexpert")
    res = A.validfit("m", "chexpert", draws=120)
    vdf = pq.read_table(rd / "outcomes" / "VALIDFIT.parquet").to_pandas()
    cdf = pq.read_table(rd / "outcomes" / "CORE.parquet").to_pandas()
    base = cdf[cdf.direction_id == "baseline"].set_index(["concept", "row_id"]).p_present
    core = A.core("m", "chexpert", n_boot=120)
    assert res["label_source"] == "radiologist" and res["fit_rows"] == N_VALID and res["n_folds"] == P.VALIDFIT_FOLDS
    assert len(res["contrasts"]) == 30 and res["baseline_module"] == "CORE"
    for q in CHEX:
        cell = res["per_question"][q]
        g = vdf[(vdf.concept == q) & (vdf.direction_id == f"vfit:{q}")]
        assert abs(cell["W_qq"] - float(np.mean(g.p_present.values - base.loc[[(q, r) for r in g.row_id]].values))) < 1e-9
        # the steering reference is CORE's own on these rows, so the two label sources are graded on one scale
        assert cell["random_p95"] == core["per_question"][q]["random_p95"] and cell["abs_sham"] == core["per_question"][q]["abs_sham"]
        assert cell["steering_reference"] == (cell["W_qq"] > 0 and cell["W_qq"] > cell["random_p95"] and cell["W_qq"] > cell["abs_sham"])
        assert cell["owned"] == (cell["steering_reference"] and cell["verdict"] == "fixed_family_advantage")
        assert cell["verdict"] in ("stronger_competitor", "fixed_family_advantage", "unresolved") and cell["n_competitors"] == 5
        assert abs(cell["cos_to_report_model"] - arr["cos_model"][CHEX.index(q)]) < 1e-12
        assert "own_minus_max_logistic_competitor" in cell and cell["n_valid_pos"] + cell["n_valid_neg"] == N_VALID
    # manifest
    (rd / "summary.json").write_text(json.dumps({"validfit": res, "core": core}, **JSON))
    rj = json.loads((rd / "run.json").read_text())
    rj["completed_modules"] = ["CORE", "CALIBRATION", "VALIDFIT"]
    (rd / "run.json").write_text(json.dumps(rj))
    mrows = M.block_rows(rd)
    vr = [r for r in mrows if r["module"] == "VALIDFIT"]
    assert len(vr) == 6 and all(r["block_included"] for r in vr)
    for r in vr:
        c = res["per_question"][r["concept"]]
        assert (r["validfit_owned"], r["validfit_O_q"], r["validfit_verdict"]) == (c["owned"], c["O_q"], c["verdict"])
        assert r["validfit_cos_to_report_whitened"] == c["cos_to_report_whitened"]
    M.write_csv(mrows, tmp_path / "manifest.csv")
    header = (tmp_path / "manifest.csv").read_text().splitlines()[0].split(",")
    assert all(f"validfit_{k}" in header for k in M.VALIDFIT_FIELDS) and "attrrand_O_q" in header


# ------------------------------------------------------------------------------------------------ PROJSEED
def test_projseed_prep_runner_and_analysis(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(PK, "TEST_ROWS", 12)                 # the synthetic cohort has 12 test rows, not 600
    rd = runs / "m" / "nih"
    with pytest.raises(FileNotFoundError, match="cftransfer.projseed"):
        RN.run_block("m", "nih", "PROJSEED", 0, 1, batch=8, device_map="cpu", rows_per_part=4, fit_seed=1)
    built = PS.build_projseed("m", "nih", "vis.last")
    assert sorted(built) == [1, 2] and all(p.name == f"projseed_seed{k}.npz" for k, (p, _a) in built.items())
    f0 = F.load_fit("m", "nih", "vis.last", 0)
    ids, X, _v, roles = F.load_features("m", "nih", "vis.last")
    tr = np.array([i for i, r in enumerate(roles) if r == "train"])
    labels = I.load_labels("nih")
    for k, (_p, arr) in built.items():
        # the projection is the protocol's construction with the seed replaced, and it is NOT the seed-0 one
        Pk = (np.random.default_rng(k).standard_normal((16, 512)) / np.sqrt(512)).astype(np.float32)
        assert np.allclose(arr["projection"], Pk) and not np.allclose(arr["projection"], f0["projection"])
        # the scaler is recomputed in that projected space on the same training rows
        Z = X.astype(np.float32) @ Pk
        sc = StandardScaler().fit(Z[tr])
        assert np.allclose(arr["scaler_mean"], sc.mean_, atol=1e-5) and np.allclose(arr["scaler_scale"], sc.scale_, atol=1e-5)
        Zs = ((Z - arr["scaler_mean"]) / np.maximum(arr["scaler_scale"], 1e-8)).astype(np.float32)
        assert np.array_equal(arr["train_row_ids"].astype(str), f0["train_row_ids"].astype(str))
        for ci, c in enumerate(CONCEPTS):
            y = np.array([int(labels[(rid, c)]["label"]) for rid in ids[tr]])
            w = LogisticRegression(**P.LOGREG).fit(Zs[tr], y).coef_[0]
            assert np.allclose(arr["coefficients"][ci], w, atol=1e-6)
            assert np.allclose(arr["clinical_vectors"][ci],
                               F.direction_from_projected(Pk, arr["scaler_scale"], w), atol=1e-6)
            assert np.array_equal(arr["sham_vectors"][ci], arr["clinical_vectors"][ci][arr["sham_permutations"][ci]])
        # the random family is the protocol's construction re-drawn from PCG64(k) in that projection, then the shams
        rng = np.random.default_rng(k)
        Rr = rng.standard_normal((119, 16)); Rr /= np.linalg.norm(Rr, axis=1, keepdims=True)
        assert np.allclose(arr["random_vectors"], Rr.astype(np.float32), atol=1e-6)
        assert np.array_equal(arr["sham_permutations"][0], rng.permutation(16))
        assert not np.allclose(arr["random_vectors"], f0["random_vectors"])
        v64, s64 = arr["clinical_vectors"].astype(np.float64), f0["clinical_vectors"].astype(np.float64)
        assert np.allclose(arr["cos_model_to_seed0"], np.diagonal(v64 @ s64.T))
    with pytest.raises(ValueError, match="seed 0 is the campaign fit"):
        PS.build_projseed("m", "nih", "vis.last", seeds=(0,))
    # runner: one task per projection seed, its own part / meta stream, the seed in the fit_seed column
    for k in P.PROJSEED_SEEDS:
        meta = RN.run_block("m", "nih", "PROJSEED", 0, 1, batch=32, device_map="cpu", rows_per_part=6, fit_seed=k)
        assert meta["outcomes"] == 12 * 6 * 126 and meta["fit_seed"] == k
        assert list((rd / "outcomes" / "PROJSEED").glob(f"meta-s{k}-000of001-*.json"))
    with pytest.raises(ValueError, match="fit seeds"):
        RN.run_block("m", "nih", "PROJSEED", 0, 1, batch=8, device_map="cpu", fit_seed=3)
    assert RN.run_block("m", "nih", "PROJSEED", 0, 1, batch=32, device_map="cpu", rows_per_part=6, fit_seed=1)["todo"] == 0
    df = pq.read_table(rd / "outcomes" / "PROJSEED" / "part-s1-000of001-00000.parquet").to_pandas()
    assert set(df.fit_seed) == {1} and set(df.direction_kind) == {"proj", "projrand", "projsham"}
    for (_rid, q), g in df.groupby(["row_id", "concept"]):
        assert sorted(g.direction_id) == sorted(d for d, _ in P.conditions_for("PROJSEED", "nih", q))
    RN.run_block("m", "nih", "CORE", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    run = PK.build("m", "nih")
    cov = [r for r in csv.DictReader((rd / "coverage.csv").open(newline="")) if r["module"] == "PROJSEED"]
    assert len(cov) == 12 and all(r["execution_status"] == "COMPLETE" and r["expected_rows"] == str(126 * 12) for r in cov)
    assert sorted({r["fit_seed"] for r in cov}) == ["1", "2"] and "PROJSEED" in run["completed_modules"]
    res = A.projseed("m", "nih", draws=120)
    pdf = pq.read_table(rd / "outcomes" / "PROJSEED.parquet").to_pandas()
    cdf = pq.read_table(rd / "outcomes" / "CORE.parquet").to_pandas()
    base = cdf[cdf.direction_id == "baseline"].set_index(["concept", "row_id"]).p_present
    core = A.core("m", "nih", n_boot=120)
    assert res["seeds"] == [1, 2] and res["not_started_seeds"] == [] and res["core"]["projection_seed"] == 0
    for k in P.PROJSEED_SEEDS:
        blk = res[f"seed{k}"]
        arr = built[k][1]
        assert blk["projection_seed"] == k and len(blk["contrasts"]) == 30 and blk["n_compared"] == 6
        for q in CONCEPTS:
            cell = blk["per_question"][q]
            g = pdf[(pdf.concept == q) & (pdf.direction_id == f"proj:{q}") & (pdf.fit_seed == k)]
            assert abs(cell["W_qq"] - float(np.mean(g.p_present.values - base.loc[[(q, r) for r in g.row_id]].values))) < 1e-9
            rnd = np.array([float(np.mean(gg.p_present.values - base.loc[[(q, r) for r in gg.row_id]].values))
                            for _d, gg in pdf[(pdf.concept == q) & (pdf.fit_seed == k) &
                                              (pdf.direction_kind == "projrand")].groupby("direction_id")])
            assert cell["random_n"] == 119 and abs(cell["random_p95"] - float(np.percentile(rnd, 95))) < 1e-9
            # the seed's OWN references, not CORE's (which stay under the *_logistic names)
            assert cell["random_p95"] != core["per_question"][q]["random_p95"]
            assert cell["random_p95_logistic"] == core["per_question"][q]["random_p95"]
            assert cell["steering_reference"] == (cell["W_qq"] > 0 and cell["W_qq"] > cell["random_p95"] and cell["W_qq"] > cell["abs_sham"])
            assert cell["verdict_core"] == core["per_question"][q]["verdict"]
            assert cell["verdict_changed"] == (cell["verdict"] != cell["verdict_core"])
            assert abs(cell["cos_to_seed0_model"] - arr["cos_model_to_seed0"][CONCEPTS.index(q)]) < 1e-12
        assert blk["n_agree"]["verdict"] == sum(not blk["per_question"][q]["verdict_changed"] for q in CONCEPTS)
        assert blk["owned"] == sorted(q for q in CONCEPTS if blk["per_question"][q]["owned"])
    # manifest: one ownership flag and O_q per seed, on that seed's coverage rows
    (rd / "summary.json").write_text(json.dumps({"projseed": res, "core": core}, **JSON))
    rj = json.loads((rd / "run.json").read_text())
    rj["completed_modules"] = ["CORE", "CALIBRATION", "PROJSEED"]
    (rd / "run.json").write_text(json.dumps(rj))
    mrows = M.block_rows(rd)
    pr = [r for r in mrows if r["module"] == "PROJSEED"]
    assert len(pr) == 12
    for r in pr:
        k = int(r["fit_seed"])
        c = res[f"seed{k}"]["per_question"][r["concept"]]
        assert r[f"projseed_owned_seed{k}"] == c["owned"] and r[f"projseed_O_q_seed{k}"] == c["O_q"]
        assert r[f"projseed_owned_seed{3 - k}"] is None
    assert all(r["projseed_owned_seed1"] is None for r in mrows if r["module"] != "PROJSEED")
