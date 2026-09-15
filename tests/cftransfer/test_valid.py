"""VALID: the CORE grid on the radiologist-labelled CheXpert valid split. Cohort append (build_chexpert_valid: idempotent,
existing rows untouched, one frontal per patient, dense radiologist labels, provenance), spec / expected rows / enqueue,
the runner on the synthetic fixture, the analysis with the test-versus-valid comparison, and the manifest columns."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from cftransfer import analysis as A, enqueue as E, fit as F, images as I, manifest as M, manifests as MF, package as PK, \
    protocol as P, runner as RN, runpaths as R  # noqa: E402
from test_fit_synthetic import make_synthetic  # noqa: E402
from test_runner_fake import FakeAdapter  # noqa: E402

CONCEPTS = P.CONCEPTS["chexpert"]
RAW = list(MF.CHEXPERT_RAW_TO_CONCEPT) + MF.CHEXPERT_EXTRA          # the 14 CheXpert observations, as in the radiologist csv
N_VALID = 24                                                        # patients; labels alternate so every concept has 12 / 12
JSON = dict(default=lambda o: float(o) if isinstance(o, np.floating) else str(o))


def _valid_inputs(data: Path) -> dict:
    """Mini CheXpert Plus parquet and radiologist csv: N_VALID valid patients, the first with two frontals, the second with
    a lateral too, one train row; dense 0.0 / 1.0 labels, concept c of patient i = (i + c) mod 2 for every image."""
    d = data / "chexpert"
    meta, lab = [], []
    for i in range(N_VALID):
        pid = f"patient{64541 + i}"
        views = ["view1_frontal", "view2_frontal"] if i == 0 else (["view1_frontal", "view2_lateral"] if i == 1 else ["view1_frontal"])
        for v in views:
            path = f"valid/{pid}/study1/{v}.jpg"
            fl = "Lateral" if "lateral" in v else "Frontal"
            meta.append({"path_to_image": path, "frontal_lateral": fl, "ap_pa": "AP" if i % 3 else "PA", "deid_patient_id": pid,
                         "age": float(20 + 3 * i), "sex": "Male" if i % 2 else "Female", "split": "valid"})
            row = {"Path": path, "Sex": "Male" if i % 2 else "Female", "Age": str(20 + 3 * i), "Frontal/Lateral": fl, "AP/PA": "AP" if i % 3 else "PA"}
            row.update({k: f"{(i + ci) % 2:.1f}" for ci, k in enumerate(RAW)})
            lab.append(row)
    meta.append({"path_to_image": "train/patient00001/study1/view1_frontal.jpg", "frontal_lateral": "Frontal", "ap_pa": "AP",
                 "deid_patient_id": "patient00001", "age": 40.0, "sex": "Male", "split": "train"})
    pd.DataFrame(meta).to_parquet(d / "df_chexpert_plus_240401.parquet")
    csv_path = d / MF.VALID_LABELS_CSV
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"] + RAW); w.writeheader(); w.writerows(lab)
    Path(str(csv_path) + ".provenance.json").write_text(json.dumps({
        "source": "test", "hf_repo": "x/y", "hf_revision": "abc123", "hf_file": "data/validation.parquet",
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}))
    return {r["Path"]: r for r in lab}


def test_valid_cohort_append_idempotent(tmp_path, monkeypatch):
    data, runs = make_synthetic(tmp_path, dataset_id="chexpert")
    monkeypatch.setattr(I, "DATA_ROOT", data)
    man = data / "chexpert" / "manifests"
    lab = _valid_inputs(data)
    before = {n: (man / n).read_bytes() for n in ("cohort.csv", "labels.csv")}
    s = MF.build_chexpert_valid(data)
    assert s["rows"] == s["patients"] == N_VALID and s["appended"] and s["labels_pos_neg_unknown"] == {c: [12, 12, 0] for c in CONCEPTS}
    cohort = list(csv.DictReader((man / "cohort.csv").open(newline="")))
    v = [r for r in cohort if r["role"] == "valid"]
    assert len(v) == N_VALID and len({r["unit_id"] for r in v}) == N_VALID and [int(r["order"]) for r in v] == list(range(N_VALID))
    assert all(r["original_split"] == "valid" and r["relative_image_path"].startswith("valid/patient") and
               r["relative_image_path"].endswith("_frontal.png") and r["dataset_id"] == "chexpert" for r in v)
    assert cohort[:len(cohort) - N_VALID] == list(csv.DictReader(before["cohort.csv"].decode().splitlines()))
    # one frontal per patient by the row hash (the two-frontal patient), laterals out, patients in patient-hash order
    p0 = next(r for r in v if r["unit_id"] == "patient64541")
    best = min((f"valid/patient64541/study1/view{k}_frontal.jpg" for k in (1, 2)), key=lambda p: MF.sha("cf-transfer-v1-chexpert-row:", p))
    assert p0["relative_image_path"] == best.replace(".jpg", ".png") and p0["row_id"] == best.split("/", 1)[1].replace("/", "__")[:-4]
    assert [r["unit_id"] for r in v] == sorted({r["unit_id"] for r in v}, key=lambda pid: MF.sha("cf-transfer-v1-chexpert-patient:", pid))
    # existing rows byte-identical, new rows appended; labels dense, from the radiologist csv, every row known
    for n, b in before.items():
        assert (man / n).read_bytes().startswith(b)
    labels = I.load_labels("chexpert")
    for r in v:
        src = lab[r["relative_image_path"].replace(".png", ".jpg")]
        for raw, c in MF.CHEXPERT_RAW_TO_CONCEPT.items():
            L = labels[(r["row_id"], c)]
            assert L["label_known"] == "true" and L["label"] == ("1" if src[raw] == "1.0" else "0") and L["label_raw"] == src[raw]
            assert L["view"] == src["AP/PA"] and L["sex"] == src["Sex"][0] and L["age"] == src["Age"] and L["type_id"]
        assert all(labels[(r["row_id"], x)]["label_known"] == "true" for x in MF.CHEXPERT_EXTRA)
    rel = json.loads((man / "release.json").read_text())["valid"]
    prov = json.loads((data / "chexpert" / (MF.VALID_LABELS_CSV + ".provenance.json")).read_text())
    assert rel["label_source"] == "radiologist" and rel["rows"] == rel["patients"] == N_VALID and rel["frontal_valid_rows"] == N_VALID + 1
    assert rel["labels_sha256"] == prov["sha256"] and rel["hf_repo"] == "x/y" and rel["hf_revision"] == "abc123" and rel["hf_file"] == prov["hf_file"]
    # idempotent: a second build changes nothing
    after = {n: (man / n).read_bytes() for n in ("cohort.csv", "labels.csv", "release.json")}
    s2 = MF.build_chexpert_valid(data)
    assert not s2["appended"] and s2["rows"] == N_VALID and all((man / n).read_bytes() == b for n, b in after.items())
    # the cohort loader accepts the role, after test
    rows = I.load_cohort("chexpert", ("valid",))
    assert [r["row_id"] for r in rows] == [r["row_id"] for r in v] and I.load_cohort("chexpert")[-1]["role"] == "valid"
    assert [r["role"] for r in I.load_cohort("chexpert", ("test", "valid"))][:12] == ["test"] * 12


def test_valid_spec_expected_rows_and_enqueue(tmp_path, monkeypatch):
    spec = P.MODULES["VALID"]
    assert spec.role == "valid" and spec.datasets == ("chexpert",) and spec.directions == "core" and spec.baseline_module is None
    assert spec.fit_seeds == (0,) and spec.templates == ("IY",) and spec.alphas == (P.PRIMARY_ALPHA,) and spec.row_limit is None
    assert P.conditions_for("VALID", "chexpert", "Edema") == P.conditions_for("CORE", "chexpert", "Edema")
    assert P.question_list("chexpert", "VALID", "IB") == [(c, "IB") for c in CONCEPTS]
    assert P.VALID_ROWS == 200 and P.expected_rows("VALID", "chexpert") == 6 * 127 * 200 == 152_400
    assert P.expected_rows("VALID", "nih") == P.expected_rows("VALID", "coco") == 0
    assert "VALID" in P.ADDENDUM_MODULES and "VALID" in P.MODULES_ADDED_LATER["chexpert"] and "VALID" not in E.MODULE_ORDER
    assert E.SHARDS["VALID"] == 1 and "VALID" not in E.PREP_FILE and "VALID" not in E.PREP_TASK
    assert P.PROTOCOL["modules"]["VALID"]["expected_rows"] == 152_400 and P.PROTOCOL["modules"]["VALID"]["cohort"] == "valid"
    assert P.PROTOCOL["total_planned_outcomes"]["addenda"]["VALID"] == 152_400 * 22
    q = tmp_path / "queue"; (q / "pending").mkdir(parents=True)
    monkeypatch.setattr(E, "QUEUE", q)
    names = E.enqueue("q25-7", "chexpert", ["VALID"], prep=False, prefix="1")
    assert len(names) == 2 and names[0].endswith("-VALID-features-q25-7-chexpert") and names[1].endswith("-VALID-q25-7-chexpert-000of001")
    feat = json.loads((q / "pending" / f"{names[0]}.json").read_text())
    mod = json.loads((q / "pending" / f"{names[1]}.json").read_text())
    vf = R.valid_features_dir("q25-7", "chexpert")
    assert vf == R.features_dir("q25-7", "chexpert") / "valid"
    assert feat["cmd"][:3] == ["python", "-m", "cftransfer.features"] and feat["cmd"][feat["cmd"].index("--roles") + 1] == "valid"
    assert feat["cmd"][-2:] == ["--out-dir", str(vf)] and feat["produces"] == [str(vf / "vis.last.npz"), str(vf / "connector.npz")]
    assert E.VALID_IMAGES in feat["requires"] and E.VALID_IMAGES in mod["requires"] and E.VALID_IMAGES.endswith("images/valid/.complete")
    assert not any("features" in r for r in mod["requires"])                       # the module never waits for the features
    assert mod["cmd"][2] == "cftransfer.runner" and mod["cmd"][mod["cmd"].index("--module") + 1] == "VALID" and "--numerics" not in mod["cmd"]
    assert mod["produces"][0].endswith("outcomes/VALID/meta-000of001-*.json") and mod["lane"] == feat["lane"]
    assert E.enqueue("q25-7", "nih", ["VALID"], prep=False, prefix="1") == []
    # a CheXpert block packaged before VALID existed stays COMPLETE until VALID starts
    rd = tmp_path / "m" / "chexpert"; (rd / "outcomes").mkdir(parents=True)
    assert "VALID" not in PK.requested_modules("chexpert", rd)
    (rd / "outcomes" / "VALID").mkdir()
    assert "VALID" in PK.requested_modules("chexpert", rd)


def _setup_chexpert(tmp_path, monkeypatch):
    data, runs = make_synthetic(tmp_path, dataset_id="chexpert")
    monkeypatch.setattr(I, "DATA_ROOT", data)
    monkeypatch.setattr(R, "RUN_ROOT", runs)
    for mod in (I, RN):
        monkeypatch.setattr(mod, "image_path", lambda ds, row: row["row_id"])
        monkeypatch.setattr(mod, "open_rgb", lambda p: p)
    _valid_inputs(data)
    MF.build_chexpert_valid(data)
    F.fit_locus("m", "chexpert", "vis.last", seeds=(0,), write_scores=False)
    ad = FakeAdapter("m", "fake/m", "rev")
    monkeypatch.setattr(RN, "get_adapter", lambda key, rev=None: ad)
    monkeypatch.setattr(PK, "MODELS", {**PK.MODELS, "m": {"model_id": "fake/m"}})
    return ad, runs


def test_valid_runner_analysis_and_manifest(tmp_path, monkeypatch):
    ad, runs = _setup_chexpert(tmp_path, monkeypatch)
    rd = runs / "m" / "chexpert"
    vrows = I.load_cohort("chexpert", ("valid",))
    # runner: the full CORE grid on every valid row, resumable
    meta = RN.run_block("m", "chexpert", "VALID", 0, 1, batch=32, device_map="cpu", rows_per_part=8)
    assert meta["rows"] == N_VALID and meta["outcomes"] == N_VALID * 6 * 127 and meta["primary_template"] == "IY"
    parts = sorted((rd / "outcomes" / "VALID").glob("part-*.parquet"))
    df = pd.concat([pq.read_table(p).to_pandas() for p in parts])
    assert set(df.role) == {"valid"} and set(df.row_id) == {r["row_id"] for r in vrows} and set(df.sample_status) == {"OK"}
    for (_rid, q), g in df.groupby(["row_id", "concept"]):
        assert sorted(g.direction_id) == sorted(d for d, _ in P.conditions_for("VALID", "chexpert", q))
    assert RN.run_block("m", "chexpert", "VALID", 0, 1, batch=32, device_map="cpu", rows_per_part=8)["todo"] == 0
    # package: VALID coverage against the valid role size; block completeness never depends on VALID
    RN.run_block("m", "chexpert", "CORE", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    monkeypatch.setattr(PK, "VALID_ROWS", N_VALID)
    run = PK.build("m", "chexpert")
    cov = [r for r in csv.DictReader((rd / "coverage.csv").open(newline="")) if r["module"] == "VALID"]
    assert [r["concept"] for r in cov] == CONCEPTS and all(r["execution_status"] == "COMPLETE" and r["expected_rows"] == str(127 * N_VALID) for r in cov)
    assert "VALID" in run["completed_modules"] and "VALID" in run["requested_modules"] and "CORE" not in run["completed_modules"]
    # analysis without the valid features: the CORE grade on the valid rows, answers against the radiologist labels
    cal = {q: {"readable": q in CONCEPTS[:3], "n_pos": 20, "n_neg": 20, "selectivity": 0.1, "answer_capable": q in CONCEPTS[2:]} for q in CONCEPTS}
    (rd / "summary.json").write_text(json.dumps({"calibration": cal}))
    res = A.valid("m", "chexpert", draws=120)
    vdf = pq.read_table(rd / "outcomes" / "VALID.parquet").to_pandas()
    base = vdf[vdf.direction_id == "baseline"].set_index(["concept", "row_id"]).p_present
    labels = I.load_labels("chexpert")
    c_valid = A.core("m", "chexpert", "VALID", n_boot=120)
    c_test = A.core("m", "chexpert", "CORE", n_boot=120)
    assert res["n_rows"] == c_valid["n_rows"] == N_VALID and c_test["n_rows"] == 12 and res["draws"] == 120
    assert res["label_source"] == "radiologist" and res["features_available"] is False and len(res["contrasts"]) == 30
    for q in CONCEPTS:
        cell = res["per_question"][q]
        g = vdf[(vdf.concept == q) & (vdf.direction_id == f"concept:{q}")]
        assert abs(cell["W_qq"] - float(np.mean(g.p_present.values - base.loc[[(q, r) for r in g.row_id]].values))) < 1e-9
        assert cell["O_q"] == c_valid["per_question"][q]["O_q"] and cell["verdict"] == c_valid["per_question"][q]["verdict"]
        assert cell["verdict"] in ("stronger_competitor", "fixed_family_advantage", "unresolved") and isinstance(cell["steering_reference"], bool)
        assert np.isfinite(cell["random_p95"]) and np.isfinite(cell["abs_sham"]) and len(cell["O_q_ci95_percentile"]) == 2
        assert cell["owned"] == (cell["steering_reference"] and cell["verdict"] == "fixed_family_advantage") and cell["label_source"] == "radiologist"
        assert cell["readable"] is None and cell["readable_status"] == "features_not_available"
        gb = vdf[(vdf.concept == q) & (vdf.direction_id == "baseline")]
        y = [int(labels[(r, q)]["label"]) for r in gb.row_id]
        assert cell["n_pos"] == cell["n_neg"] == 12 and abs(cell["answer_auroc"] - roc_auc_score(y, gb.semantic_margin.values)) < 1e-9
        assert cell["answer_valid_draws"] == 120 and cell["answer_capable"] is False                     # 120 < 1,900 draws
    # the paired comparison with CORE on the test rows
    cmp = res["comparison"]
    assert cmp["test_rows"] == 12 and cmp["valid_rows"] == N_VALID
    for q in CONCEPTS:
        t, v = cmp["per_question"][q]["test"], cmp["per_question"][q]["valid"]
        assert t["O_q"] == c_test["per_question"][q]["O_q"] and t["verdict"] == c_test["per_question"][q]["verdict"] and t["label_source"] == "labeler"
        assert t["readable"] == cal[q]["readable"] and t["answer_capable"] == cal[q]["answer_capable"]
        assert t["owned"] == (t["steering_reference"] and t["verdict"] == "fixed_family_advantage")
        assert v == {k: res["per_question"][q][k] for k in v} and v["readable"] is None and v["label_source"] == "radiologist"
        assert abs(cmp["per_question"][q]["dO_q"] - (v["O_q"] - t["O_q"])) < 1e-12
    assert cmp["n_compared"] == {"verdict": 6, "owned": 6, "readable": 0, "answer_capable": 6}
    assert cmp["n_agree"]["verdict"] == sum(cmp["per_question"][q]["test"]["verdict"] == cmp["per_question"][q]["valid"]["verdict"] for q in CONCEPTS)
    assert cmp["n_agree"]["answer_capable"] == sum(cal[q]["answer_capable"] is False for q in CONCEPTS)
    # with the valid features extracted, readability is the seed-0 probe against the 20 controls on the valid rows
    vf = R.valid_features_dir("m", "chexpert"); vf.mkdir(parents=True)
    X = np.random.default_rng(5).standard_normal((N_VALID, 16)).astype(np.float16)
    np.savez(vf / "vis.last.npz", row_id=np.array([r["row_id"] for r in vrows]), x=X, valid_token_count=np.full(N_VALID, 9),
             fit_role=np.array(["valid"] * N_VALID))
    res2 = A.valid("m", "chexpert", draws=120)
    f0 = F.load_fit("m", "chexpert", "vis.last", 0)
    Zs = (X.astype(np.float32) @ f0["projection"] - f0["scaler_mean"]) / np.maximum(f0["scaler_scale"], 1e-8)
    for ci, q in enumerate(CONCEPTS):
        cell = res2["per_question"][q]
        y = np.array([int(labels[(r["row_id"], q)]["label"]) for r in vrows])
        assert abs(cell["auroc_real"] - roc_auc_score(y, Zs @ f0["coefficients"][ci] + f0["intercepts"][ci])) < 1e-6
        assert np.isfinite(cell["control_mean"]) and cell["readable"] is False and cell["readable_status"] == "insufficient_draws"
        assert cell["O_q"] == res["per_question"][q]["O_q"]
    assert res2["features_available"] and res2["comparison"]["n_compared"]["readable"] == 6
    assert res2["comparison"]["n_agree"]["readable"] == sum(cal[q]["readable"] is False for q in CONCEPTS)
    # manifest: VALID rows carry the valid grade for included blocks; block_included ignores VALID
    (rd / "summary.json").write_text(json.dumps({"calibration": cal, "valid": res2, "core": c_test}, **JSON))
    rj = json.loads((rd / "run.json").read_text())
    rj["completed_modules"] = ["CORE", "CALIBRATION", "VALID"]
    (rd / "run.json").write_text(json.dumps(rj))
    rows = M.block_rows(rd)
    vr = [r for r in rows if r["module"] == "VALID"]
    assert len(vr) == 6 and all(r["block_included"] and not r["cell"] for r in vr)
    for r in vr:
        c = res2["per_question"][r["concept"]]
        assert (r["valid_owned"], r["valid_O_q"], r["valid_verdict"], r["valid_readable"], r["valid_answer_capable"]) == \
            (c["owned"], c["O_q"], c["verdict"], False, False)
    assert len([r for r in rows if r["cell"]]) == 6 and all(r["valid_owned"] is None for r in rows if r["cell"])
    rj["completed_modules"] = ["CORE", "CALIBRATION"]
    (rd / "run.json").write_text(json.dumps(rj))
    rows = M.block_rows(rd)
    assert all(r["block_included"] for r in rows) and all(r["valid_owned"] is None for r in rows)
    rj["completed_modules"] = ["VALID"]
    (rd / "run.json").write_text(json.dumps(rj))
    assert not any(r["block_included"] for r in M.block_rows(rd))
    M.write_csv(rows, tmp_path / "manifest.csv")
    header = (tmp_path / "manifest.csv").read_text().splitlines()[0].split(",")
    assert all(f"valid_{k}" in header for k in M.VALID_FIELDS) and M.summarize(rows)["chexpert"]["write_cells"] == 6
