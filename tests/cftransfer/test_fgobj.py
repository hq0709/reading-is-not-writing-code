"""FGOBJ: fine-grained COCO object concepts, the control that separates "clinical concept" from "hard question".

Covered here: the prespecified selection rule, the append-only / idempotent label build, the direction prep against the
protocol construction refitted by hand, the runner's 127-condition grid (and the clean calibration companion), the
analysis on synthetic outcomes, and the difficulty-matched comparison of scripts/mayo/robustness_round2.py including
the case where a chest cell finds no partner.
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from sklearn.linear_model import LogisticRegression

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from cftransfer import analysis as A, enqueue as E, fgobj as FG, fit as F, images as I, package as PK, protocol as P, \
    runner as RN, runpaths as R  # noqa: E402
from test_runner_fake import FakeAdapter  # noqa: E402

EASY = P.CONCEPTS["coco"]
FINE = P.FGOBJ_CONCEPTS
N_TRAIN, N_EVAL = 60, 40          # 12 eval rows is too few for the 10/10 support rule the probe grades use
D = 16
JSON = dict(default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
# aspect x area buckets of manifests.coco_type_id; nine of them, as the real COCO cohort has (>= fit.MIN_TYPES)
SIZES = [(200, 300), (300, 200), (256, 256), (700, 900), (900, 700), (800, 800), (1400, 1800), (1800, 1400), (1600, 1600)]


def r2():
    """scripts/mayo/robustness_round2.py loaded by path (it is a script, not a package module)."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "mayo" / "robustness_round2.py"
    spec = importlib.util.spec_from_file_location("robustness_round2_fgobj", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------ synthetic COCO fixture
def make_coco(tmp: Path, n_train: int = N_TRAIN, n_eval: int = N_EVAL, dim: int = D):
    """A miniature COCO data root: numeric row ids, the frozen role split, instance annotations in both splits, and
    labels.csv for the SIX PROTOCOL CONCEPTS ONLY (the fine-grained six are what the append adds). Presence is drawn
    once and then written into BOTH the annotations and labels.csv, so build_coco's rule and the fixture agree."""
    rng = np.random.default_rng(7)
    data, runs = tmp / "data", tmp / "runs"
    man = data / "coco" / "manifests"; man.mkdir(parents=True)
    ann = data / "coco" / "annotations"; ann.mkdir(parents=True)
    all_ids = {c: P.COCO_CATEGORY_IDS[c] for c in EASY} | {c: P.FGOBJ_CATEGORY_IDS[c] for c in FINE}
    roles = [("train", n_train), ("preflight", 4), ("calibration", n_eval), ("test", n_eval)]
    cohort, images, present = [], {}, {}
    k = 0
    for role, n in roles:
        for j in range(n):
            k += 1
            rid = f"{k:012d}"
            w, h = SIZES[k % len(SIZES)]
            split = "val" if role == "test" else "train"
            cohort.append(dict(dataset_id="coco", row_id=rid, unit_id=rid, role=role, order=j,
                               relative_image_path=f"{split}2017/{rid}.jpg", original_split=f"{split}2017"))
            images[rid] = {"id": k, "width": w, "height": h, "file_name": f"{rid}.jpg", "split": split}
            present[rid] = {c for c in all_ids if rng.random() < 0.5}
    for split in ("train", "val"):
        ims = [{"id": v["id"], "width": v["width"], "height": v["height"], "file_name": v["file_name"]}
               for v in images.values() if v["split"] == split]
        anns, aid = [], 0
        for rid, v in images.items():
            if v["split"] != split:
                continue
            for c in sorted(present[rid]):
                aid += 1
                # one instance per present category; every third is a crowd region, which still counts as present
                anns.append({"id": aid, "image_id": v["id"], "category_id": all_ids[c], "iscrowd": int(aid % 3 == 0),
                             "area": float(v["width"] * v["height"]) * (0.01 if c in FINE else 0.1)})
        (ann / f"instances_{split}2017.json").write_text(json.dumps(
            {"images": ims, "annotations": anns,
             "categories": [{"id": i, "name": n} for n, i in sorted(all_ids.items(), key=lambda kv: kv[1])]}))
    labels = []
    for r in cohort:
        rid = r["row_id"]
        v = images[rid]
        tid = FG.coco_type_id(v["width"], v["height"])
        for c in EASY:
            lab = "1" if c in present[rid] else "0"
            labels.append(dict(dataset_id="coco", row_id=rid, concept=c, label=lab, label_known="true", label_raw=lab,
                               view="", sex="", age="", width=v["width"], height=v["height"], type_id=tid))
    with (man / "cohort.csv").open("w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(cohort[0])); w_.writeheader(); w_.writerows(cohort)
    with (man / "labels.csv").open("w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=FG.LABEL_COLS); w_.writeheader(); w_.writerows(labels)
    feat = runs / "m" / "coco" / "features"; feat.mkdir(parents=True)
    ids = [r["row_id"] for r in cohort]
    X = rng.standard_normal((len(ids), dim)).astype(np.float16)
    for lid in ("vis.last", "connector"):
        np.savez(feat / f"{lid}.npz", row_id=np.array(ids), x=X,
                 valid_token_count=np.full(len(ids), 9), fit_role=np.array([r["role"] for r in cohort]))
    return data, runs, present


def _setup_coco(tmp_path, monkeypatch, build_labels: bool = True):
    data, runs, present = make_coco(tmp_path)
    monkeypatch.setattr(I, "DATA_ROOT", data)
    monkeypatch.setattr(FG, "DATA_ROOT", data)
    monkeypatch.setattr(R, "RUN_ROOT", runs)
    monkeypatch.setattr(I, "image_path", lambda ds, row: row["row_id"])
    monkeypatch.setattr(I, "open_rgb", lambda p: p)
    monkeypatch.setattr(RN, "image_path", lambda ds, row: row["row_id"])
    monkeypatch.setattr(RN, "open_rgb", lambda p: p)
    monkeypatch.setattr(PK, "TEST_ROWS", N_EVAL)
    monkeypatch.setattr(PK, "CALIBRATION_ROWS", N_EVAL)
    if build_labels:
        FG.build_fgobj_labels(data)
    F.fit_locus("m", "coco", "vis.last", seeds=(0,), write_scores=True)
    ad = FakeAdapter("m", "fake/m", "rev")
    monkeypatch.setattr(RN, "get_adapter", lambda key, rev=None: ad)
    monkeypatch.setattr(PK, "MODELS", {**PK.MODELS, "m": {"model_id": "fake/m"}})
    return data, runs, present, ad


# ------------------------------------------------------------------------------------------------ protocol / enqueue
def test_fgobj_protocol_grid_and_enqueue(tmp_path, monkeypatch):
    # six fine-grained concepts, COCO only, CORE's grid shape with the family's OWN random directions and sham
    assert P.MODULES["FGOBJ"].datasets == ("coco",) and P.MODULES["FGOBJ"].role == "test"
    assert P.MODULES["FGOBJ"].row_limit is None and P.MODULES["FGOBJ"].baseline_module is None
    assert P.MODULES["FGOBJ"].fit_seeds == (0,) and P.MODULES["FGOBJ"].locus == "primary"
    assert P.question_list("coco", "FGOBJ") == [(c, "IY") for c in FINE]
    assert P.question_list("coco", "FGOBJ", "IB") == [(c, "IB") for c in FINE]
    cf = P.conditions_for("FGOBJ", "coco", "knife")
    assert cf == [("baseline", 0.0)] + [(f"fgobj:{c}", 0.25) for c in FINE] \
        + [(f"fgobjrand:{i:03d}", 0.25) for i in range(P.N_RANDOM)] + [("fgobjsham:knife", 0.25)]
    assert len(cf) == len(P.conditions_for("CORE", "coco", "person")) == 127        # exactly CORE's grid shape
    assert P.expected_rows("FGOBJ", "coco") == 6 * 127 * 600 == 457_200
    assert P.expected_rows("FGOBJ", "nih") == P.expected_rows("FGOBJ", "chexpert") == 0
    assert P.direction_kind("fgobjrand:007") == "fgobjrand" and P.direction_kind("fgobjsham:book") == "fgobjsham"
    # the clean companion on the calibration rows: one condition per question, CALIBRATION's shape
    cc = P.conditions_for("FGOBJ_CALIBRATION", "coco", "book")
    assert cc == [("baseline", 0.0)] == P.conditions_for("CALIBRATION", "coco", "person")
    assert P.MODULES["FGOBJ_CALIBRATION"].role == "calibration" and P.MODULES["FGOBJ_CALIBRATION"].datasets == ("coco",)
    assert P.expected_rows("FGOBJ_CALIBRATION", "coco") == 6 * 400 == 2_400
    # the six protocol COCO concepts and their questions are untouched; the new ones render with the frozen templates
    assert P.CONCEPTS["coco"] == EASY and not set(FINE) & set(EASY)
    assert P.render_question("coco", "traffic light", "IY") == "Is there a traffic light in this photograph? Answer yes or no."
    assert P.render_question("coco", "cell phone", "WY") == "Does this photograph show a cell phone? Answer yes or no."
    assert P.render_question("coco", "person", "IY") == "Is there a person in this photograph? Answer yes or no."
    assert set(P.TEMPLATES) == set(P.TEMPLATE_ORDER)
    # the family's random seed is NOT the campaign's, so an FGOBJ cell is not graded against CORE's random family
    assert P.FGOBJ_RANDOM_SEED not in (P.RANDOM_SEED, *P.PROJSEED_SEEDS)
    # registration: addenda, added later on COCO, never in the default order; a packaged block stays COMPLETE until they start
    for m in ("FGOBJ", "FGOBJ_CALIBRATION"):
        assert m in P.ADDENDUM_MODULES and m in P.MODULES_ADDED_LATER["coco"] and m not in E.MODULE_ORDER
        assert m not in P.MODULES_ADDED_LATER["nih"] and m not in P.MODULES_ADDED_LATER["chexpert"]
        # the replication package carries the same grid and the same whole-grid row budget as the code
        assert P.PROTOCOL["modules"][m]["expected_rows"] == P.expected_rows(m, "coco")
        assert P.PROTOCOL["modules"][m]["datasets"] == ["coco"] and P.PROTOCOL["modules"][m]["alpha"] == [P.MODULES[m].alphas[0]]
        assert P.PROTOCOL["total_planned_outcomes"]["addenda"][m] == P.expected_rows(m, "coco") * 22
        assert m in P.PROTOCOL["total_planned_outcomes"]["addenda_notes"]
        assert m in P.PROTOCOL["total_planned_outcomes"]["ALTDIR_note"]        # addenda rows stay outside the planned total
    assert P.PROTOCOL["modules"]["FGOBJ"]["baseline"] == "own" and P.PROTOCOL["modules"]["FGOBJ"]["fit_seed"] == 0
    rd = tmp_path / "m" / "coco"; (rd / "outcomes").mkdir(parents=True)
    assert not {"FGOBJ", "FGOBJ_CALIBRATION"} & set(PK.requested_modules("coco", rd))
    for m in ("FGOBJ", "FGOBJ_CALIBRATION"):
        (rd / "outcomes" / m).mkdir()
    assert {"FGOBJ", "FGOBJ_CALIBRATION"} <= set(PK.requested_modules("coco", rd))
    # enqueue: a CPU prep producing fgobj_seed0.npz, four row shards that wait for it, and a prep-free clean shard
    q = tmp_path / "queue"; (q / "pending").mkdir(parents=True)
    monkeypatch.setattr(E, "QUEUE", q)
    assert E.SHARDS["FGOBJ"] == 4 and E.SHARDS["FGOBJ_CALIBRATION"] == 1
    assert E.prep_files("FGOBJ") == ["fgobj_seed0.npz"] and E.prep_files("FGOBJ_CALIBRATION") == []
    names = E.enqueue("q25-7", "coco", ["FGOBJ", "FGOBJ_CALIBRATION"], prep=False, prefix="3")
    assert len(names) == 6 and names[0].endswith("-FGOBJ-prep-q25-7-coco")
    prep = json.loads((q / "pending" / f"{names[0]}.json").read_text())
    assert prep["cmd"] == ["python", "-m", "cftransfer.fgobj", "--model-key", "q25-7", "--dataset", "coco"]
    assert "--device-map" not in prep["cmd"]                                    # the prep never touches a GPU
    assert [Path(x).name for x in prep["produces"]] == ["fgobj_seed0.npz"]
    mod = json.loads((q / "pending" / f"{names[1]}.json").read_text())
    assert mod["requires"][-1].endswith("fgobj_seed0.npz") and mod["cmd"][mod["cmd"].index("--n-shards") + 1] == "4"
    cal = json.loads((q / "pending" / f"{names[5]}.json").read_text())
    assert not any(x.endswith("fgobj_seed0.npz") for x in cal["requires"])       # the clean pass needs no directions
    assert E.enqueue("q25-7", "nih", ["FGOBJ", "FGOBJ_CALIBRATION"], prep=False, prefix="3") == []


# ------------------------------------------------------------------------------------------------ selection rule
def test_fgobj_selection_rule():
    """The frozen six are what the prespecified rule returns from the candidate pool on the real cohort's statistics
    (counted from the frozen COCO manifests and the 2017 instance annotations)."""
    measured = {   # name: (category id, train pos, calibration pos, test pos, mean relative instance area)
        "handbag": (31, 1156, 28, 32, 0.01097), "book": (84, 899, 19, 27, 0.01126),
        "backpack": (27, 938, 19, 26, 0.01323), "potted plant": (64, 763, 19, 16, 0.03425),
        "traffic light": (10, 692, 16, 25, 0.00519), "cell phone": (77, 777, 16, 27, 0.02047),
        "knife": (49, 771, 16, 23, 0.00754), "spoon": (50, 598, 15, 21, 0.00605),
        "remote": (75, 506, 13, 18, 0.01216), "bench": (15, 910, 19, 18, 0.04110),
        "tie": (32, 674, 15, 12, 0.01583), "vase": (86, 589, 11, 17, 0.03299)}
    stats = {k: {"category_id": v[0], "train_pos": v[1], "cal_pos": v[2], "test_pos": v[3], "mean_rel_area": v[4]}
             for k, v in measured.items()}
    assert set(stats) == set(P.FGOBJ_CANDIDATE_POOL)
    assert FG.select_fgobj_concepts(stats) == FINE == ["handbag", "book", "backpack", "traffic light", "knife", "cell phone"]
    assert P.FGOBJ_CATEGORY_IDS == {k: measured[k][0] for k in FINE}
    # each filter bites, and each for the stated reason
    assert all(stats[k]["mean_rel_area"] >= P.FGOBJ_MAX_MEAN_AREA for k in ("potted plant", "bench", "vase"))
    assert stats["remote"]["train_pos"] < P.FGOBJ_MIN_TRAIN_POS
    assert all(stats[k]["cal_pos"] >= P.FGOBJ_MIN_ROLE_POS and stats[k]["test_pos"] >= P.FGOBJ_MIN_ROLE_POS for k in FINE)
    # rank: calibration positives descending, ties by mean relative area ascending
    cal = [stats[k]["cal_pos"] for k in FINE]
    assert cal == sorted(cal, reverse=True) and cal == [28, 19, 19, 16, 16, 16]
    assert stats["book"]["mean_rel_area"] < stats["backpack"]["mean_rel_area"]
    assert stats["traffic light"]["mean_rel_area"] < stats["knife"]["mean_rel_area"] < stats["cell phone"]["mean_rel_area"]
    # the rule is a function of support and size only: permuting the pool order changes nothing
    assert FG.select_fgobj_concepts(stats, pool=tuple(reversed(P.FGOBJ_CANDIDATE_POOL))) == FINE
    # too few survivors is an explicit failure, never a silent short list
    starved = {k: {**v, "cal_pos": 1} for k, v in stats.items()}
    with pytest.raises(SystemExit, match="pass the FGOBJ support and size rule"):
        FG.select_fgobj_concepts(starved)


# ------------------------------------------------------------------------------------------------ labels
def test_fgobj_labels_append_only_and_idempotent(tmp_path, monkeypatch):
    data, runs, present = make_coco(tmp_path)
    monkeypatch.setattr(I, "DATA_ROOT", data)
    man = data / "coco" / "manifests"
    before = {n: (man / n).read_bytes() for n in ("cohort.csv", "labels.csv")}
    n_cohort = len(list(csv.DictReader((man / "cohort.csv").open(newline=""))))
    prov = FG.build_fgobj_labels(data)
    assert prov["appended"] and prov["rows_appended"] == n_cohort * len(FINE) and prov["concepts"] == list(FINE)
    # cohort.csv byte-identical; labels.csv only GREW, and every byte that was there is still there
    assert (man / "cohort.csv").read_bytes() == before["cohort.csv"]
    assert (man / "labels.csv").read_bytes().startswith(before["labels.csv"])
    rows = list(csv.DictReader((man / "labels.csv").open(newline="")))
    old = [r for r in rows if r["concept"] in EASY]
    assert old == list(csv.DictReader(before["labels.csv"].decode().splitlines()))
    assert len(list(csv.DictReader((man / "cohort.csv").open(newline="")))) == n_cohort
    # the appended labels are build_coco's rule on the same annotations, crowd instances included
    labels = I.load_labels("coco")
    for rid, cats in present.items():
        for c in FINE:
            L = labels[(rid, c)]
            assert L["label_known"] == "true" and L["label"] == ("1" if c in cats else "0") and L["label_raw"] == L["label"]
            assert L["view"] == "" and L["sex"] == "" and L["age"] == ""
            assert (L["width"], L["height"], L["type_id"]) == (labels[(rid, EASY[0])]["width"],
                                                               labels[(rid, EASY[0])]["height"],
                                                               labels[(rid, EASY[0])]["type_id"])
    crowd = json.loads((data / "coco" / "annotations" / "instances_train2017.json").read_text())
    assert any(a["iscrowd"] for a in crowd["annotations"])              # the fixture exercises the crowd rule
    # provenance: per-role counts, the digests of the annotation files, the rule
    assert set(prov["labels_pos_neg_by_role"]) == set(FINE)
    for c in FINE:
        for role in ("train", "preflight", "calibration", "test"):
            cell = prov["labels_pos_neg_by_role"][c][role]
            assert cell["pos"] + cell["neg"] == sum(1 for r in rows if r["concept"] == c
                                                    and r["row_id"] in {x["row_id"] for x in csv.DictReader((man / "cohort.csv").open(newline="")) if x["role"] == role})
    assert set(prov["annotations_sha256"]) == {"instances_train2017.json", "instances_val2017.json"}
    assert prov["selection_rule"]["min_train_positives"] == P.FGOBJ_MIN_TRAIN_POS
    # idempotent: a second call writes nothing and reports nothing appended
    size = (man / "labels.csv").stat().st_size
    again = FG.build_fgobj_labels(data)
    assert not again["appended"] and again["rows_appended"] == 0 and again["rows_present"] == n_cohort * len(FINE)
    assert (man / "labels.csv").stat().st_size == size
    # a label that disagrees with the annotations is refused, and nothing is written
    text = (man / "labels.csv").read_text().replace(f"coco,{list(present)[0]},handbag,0,", f"coco,{list(present)[0]},handbag,1,", 1)
    if text != (man / "labels.csv").read_text():
        (man / "labels.csv").write_text(text)
        size = (man / "labels.csv").stat().st_size
        with pytest.raises(SystemExit, match="differ from this build"):
            FG.build_fgobj_labels(data)
        assert (man / "labels.csv").stat().st_size == size


# ------------------------------------------------------------------------------------------------ directions
def test_fgobj_prep_reproduces_the_protocol_construction(tmp_path, monkeypatch):
    data, runs, present, ad = _setup_coco(tmp_path, monkeypatch)
    path, arr = FG.build_fgobj("m", "coco", "vis.last")
    assert path.name == "fgobj_seed0.npz"
    f0 = F.load_fit("m", "coco", "vis.last", 0)
    ids, X, _v, roles = F.load_features("m", "coco", "vis.last")
    tr = np.array([i for i, r in enumerate(roles) if r == "train"])
    cal = np.array([i for i, r in enumerate(roles) if r == "calibration"])
    labels = I.load_labels("coco")
    # the fit is the block's own seed-0 projection and TRAIN-ONLY scaler, never refitted
    Zs = ((X.astype(np.float32) @ f0["projection"] - f0["scaler_mean"]) / np.maximum(f0["scaler_scale"], 1e-8)).astype(np.float32)
    assert np.array_equal(arr["train_row_ids"].astype(str), f0["train_row_ids"].astype(str))
    assert list(arr["fgobj_names"].astype(str)) == list(FINE) and list(arr["concept_names"].astype(str)) == list(EASY)
    for ci, c in enumerate(FINE):
        y = np.array([int(labels[(rid, c)]["label"]) for rid in ids[tr]])
        w = LogisticRegression(**P.LOGREG).fit(Zs[tr], y).coef_[0]
        assert np.allclose(arr["coefficients"][ci], w, atol=1e-6)
        assert np.allclose(arr["fgobj_vectors"][ci],
                           F.direction_from_projected(f0["projection"], f0["scaler_scale"], w), atol=1e-6)
        assert np.array_equal(arr["sham_vectors"][ci], arr["fgobj_vectors"][ci][arr["sham_permutations"][ci]])
        assert int(arr["n_pos"][ci, 0]) == int(y.sum()) and int(arr["n_neg"][ci, 0]) == int((1 - y).sum())
    assert np.allclose(np.linalg.norm(arr["fgobj_vectors"], axis=1), 1.0, atol=1e-6)
    # the family's OWN random directions: fit_locus's construction re-drawn from PCG64(FGOBJ_RANDOM_SEED), then the perms
    rng = np.random.default_rng(P.FGOBJ_RANDOM_SEED)
    Rr = rng.standard_normal((P.N_RANDOM, D)); Rr /= np.linalg.norm(Rr, axis=1, keepdims=True)
    assert np.allclose(arr["random_vectors"], Rr.astype(np.float32), atol=1e-6)
    assert np.array_equal(arr["sham_permutations"][0], rng.permutation(D))
    assert not np.allclose(arr["random_vectors"], f0["random_vectors"])        # NOT the easy concepts' family
    assert int(arr["random_seed"]) == P.FGOBJ_RANDOM_SEED and int(arr["n_random"]) == P.N_RANDOM
    # calibration-row logits for the readability rule: the real probe, and the block's OWN stored control probes
    assert bool(arr["controls_eligible"]) and arr["cal_control_logits"].shape == (len(FINE), len(P.CONTROL_SEEDS), len(cal))
    assert all(s.startswith("seed0 control probes") for s in arr["controls_source"].astype(str))
    for ci in range(len(FINE)):
        assert np.allclose(arr["cal_real_logits"][ci], Zs[cal] @ arr["coefficients"][ci] + arr["intercepts"][ci], atol=1e-4)
        for k in range(len(P.CONTROL_SEEDS)):
            want = Zs[cal] @ f0["control_coefficients"][0, k] + f0["control_intercepts"][0, k]
            assert np.allclose(arr["cal_control_logits"][ci, k], want, atol=1e-4)
    assert np.array_equal(arr["cal_row_ids"].astype(str), ids[cal])
    # cosines to the six easy directions, and the CPU AUROCs
    assert arr["cos_model"].shape == (len(FINE), len(EASY)) and np.abs(arr["cos_model"]).max() <= 1.0 + 1e-6
    assert np.isfinite(arr["auroc_calibration"]).all() and np.isfinite(arr["auroc_test"]).all()
    assert FG.load_fgobj("m", "coco", "vis.last")["fgobj_vectors"].shape == (len(FINE), D)
    with pytest.raises(SystemExit, match="NOT_REQUESTED"):
        FG.build_fgobj("m", "nih", "vis.last")


def test_fgobj_prep_needs_the_appended_labels(tmp_path, monkeypatch):
    _setup_coco(tmp_path, monkeypatch, build_labels=False)
    with pytest.raises(FileNotFoundError, match="cftransfer.fgobj --build-labels"):
        FG.build_fgobj("m", "coco", "vis.last")


# ------------------------------------------------------------------------------------------------ runner
def test_fgobj_runner_grid_and_coverage(tmp_path, monkeypatch):
    data, runs, present, ad = _setup_coco(tmp_path, monkeypatch)
    rd = runs / "m" / "coco"
    # the missing prep fails before any model is loaded, and names the command
    with pytest.raises(FileNotFoundError, match="cftransfer.fgobj"):
        RN.run_block("m", "coco", "FGOBJ", 0, 1, batch=8, device_map="cpu", rows_per_part=4)
    FG.build_fgobj("m", "coco", "vis.last")
    meta = RN.run_block("m", "coco", "FGOBJ", 0, 1, batch=32, device_map="cpu", rows_per_part=10)
    assert meta["outcomes"] == N_EVAL * len(FINE) * 127
    df = pq.read_table(sorted((rd / "outcomes" / "FGOBJ").glob("part-*.parquet"))[0]).to_pandas()
    assert set(df.concept) <= set(FINE) and set(df.direction_kind) == {"baseline", "fgobj", "fgobjrand", "fgobjsham"}
    for (_rid, q), g in df.groupby(["row_id", "concept"]):
        assert sorted(g.direction_id) == sorted(d for d, _ in P.conditions_for("FGOBJ", "coco", q))
    one = df[(df.row_id == df.row_id.iloc[0]) & (df.concept == "knife")]
    base = one[one.direction_id == "baseline"]
    assert len(base) == 1 and float(base.alpha.iloc[0]) == 0.0 and float(base.delta_norm_mean.iloc[0]) == 0.0
    assert (one[one.direction_id != "baseline"].delta_norm_mean > 0).all()
    assert (one[one.direction_id != "baseline"].semantic_margin != float(base.semantic_margin.iloc[0])).all()
    assert RN.run_block("m", "coco", "FGOBJ", 0, 1, batch=32, device_map="cpu", rows_per_part=10)["todo"] == 0
    # the clean companion: one baseline forward per (calibration row, question), no directions at all
    mcal = RN.run_block("m", "coco", "FGOBJ_CALIBRATION", 0, 1, batch=8, device_map="cpu", rows_per_part=10)
    assert mcal["outcomes"] == N_EVAL * len(FINE)
    cdf = pq.read_table(sorted((rd / "outcomes" / "FGOBJ_CALIBRATION").glob("part-*.parquet"))[0]).to_pandas()
    assert set(cdf.direction_id) == {"baseline"} and set(cdf.role) == {"calibration"} and set(cdf.concept) == set(FINE)
    # CORE too, then package: both modules COMPLETE, the six protocol concepts' coverage untouched
    RN.run_block("m", "coco", "CORE", 0, 1, batch=32, device_map="cpu", rows_per_part=10)
    RN.run_block("m", "coco", "CALIBRATION", 0, 1, batch=8, device_map="cpu", rows_per_part=10)
    run = PK.build("m", "coco")
    cov = [r for r in csv.DictReader((rd / "coverage.csv").open(newline="")) if r["module"].startswith("FGOBJ")]
    assert len(cov) == 2 * len(FINE)
    assert all(r["execution_status"] == "COMPLETE" for r in cov)
    assert {r["expected_rows"] for r in cov if r["module"] == "FGOBJ"} == {str(127 * N_EVAL)}
    assert {r["expected_rows"] for r in cov if r["module"] == "FGOBJ_CALIBRATION"} == {str(N_EVAL)}
    assert "FGOBJ" in run["completed_modules"] and "FGOBJ_CALIBRATION" in run["completed_modules"]
    core_cov = [r for r in csv.DictReader((rd / "coverage.csv").open(newline="")) if r["module"] == "CORE"]
    assert len(core_cov) == len(EASY) and all(r["execution_status"] == "COMPLETE" for r in core_cov)


# ------------------------------------------------------------------------------------------------ analysis
def _run_everything(tmp_path, monkeypatch, with_calibration: bool = True):
    data, runs, present, ad = _setup_coco(tmp_path, monkeypatch)
    FG.build_fgobj("m", "coco", "vis.last")
    RN.run_block("m", "coco", "CORE", 0, 1, batch=32, device_map="cpu", rows_per_part=10)
    RN.run_block("m", "coco", "CALIBRATION", 0, 1, batch=8, device_map="cpu", rows_per_part=10)
    RN.run_block("m", "coco", "FGOBJ", 0, 1, batch=32, device_map="cpu", rows_per_part=10)
    if with_calibration:
        RN.run_block("m", "coco", "FGOBJ_CALIBRATION", 0, 1, batch=8, device_map="cpu", rows_per_part=10)
    PK.build("m", "coco")
    return data, runs


def test_fgobj_analysis_on_synthetic_outcomes(tmp_path, monkeypatch):
    data, runs = _run_everything(tmp_path, monkeypatch)
    rd = runs / "m" / "coco"
    res = A.fgobj("m", "coco", draws=120)
    fdf = pq.read_table(rd / "outcomes" / "FGOBJ.parquet").to_pandas()
    base = fdf[fdf.direction_id == "baseline"].set_index(["concept", "row_id"]).p_present
    arr = FG.load_fgobj("m", "coco", "vis.last")
    assert res["concepts"] == list(FINE) and res["n_rows"] == N_EVAL and res["baseline_module"] == "FGOBJ"
    assert res["random_seed"] == P.FGOBJ_RANDOM_SEED and res["n_random"] == P.N_RANDOM
    assert len(res["contrasts"]) == len(FINE) * (len(FINE) - 1) == 30
    for q in FINE:
        cell = res["per_question"][q]
        g = fdf[(fdf.concept == q) & (fdf.direction_id == f"fgobj:{q}")]
        want = float(np.mean(g.p_present.values - base.loc[[(q, r) for r in g.row_id]].values))
        assert abs(cell["W_qq"] - want) < 1e-9
        others = {d: res["W"][q][d] for d in FINE if d != q}
        assert abs(cell["O_q"] - (cell["W_qq"] - max(others.values()))) < 1e-12
        assert cell["argmax_other"] == max(others, key=others.get) and cell["n_competitors"] == len(FINE) - 1
        rnd = np.array([float(np.mean(gg.p_present.values - base.loc[[(q, r) for r in gg.row_id]].values))
                        for _d, gg in fdf[(fdf.concept == q) & (fdf.direction_kind == "fgobjrand")].groupby("direction_id")])
        assert cell["random_n"] == P.N_RANDOM and abs(cell["random_p95"] - float(np.percentile(rnd, 95))) < 1e-9
        assert cell["rank_in_random_family"] == int(1 + (rnd >= cell["W_qq"]).sum())
        sg = fdf[(fdf.concept == q) & (fdf.direction_id == f"fgobjsham:{q}")]
        sham = float(np.mean(sg.p_present.values - base.loc[[(q, r) for r in sg.row_id]].values))
        assert abs(cell["abs_sham"] - abs(sham)) < 1e-9
        # the campaign ownership rule, unchanged, on the family's OWN references
        assert cell["steering_reference"] == (cell["W_qq"] > 0 and cell["W_qq"] > cell["random_p95"]
                                              and cell["W_qq"] > cell["abs_sham"])
        assert cell["owned"] == (cell["steering_reference"] and cell["verdict"] == "fixed_family_advantage")
        assert cell["verdict"] in ("stronger_competitor", "fixed_family_advantage", "unresolved")
        assert cell["own_direction"] == f"fgobj:{q}"
        lo, hi = cell["O_q_ci95_percentile"]
        assert lo <= cell["O_q"] <= hi or np.isclose(lo, hi)
    assert res["owned"] == sorted(q for q in FINE if res["per_question"][q]["owned"])
    # probe grades: readability from the calibration rows, answerability from FGOBJ_CALIBRATION's clean margins
    assert res["probe_grade_source"]["answer_available"] and res["probe_grade_source"]["answer_missing_reason"] is None
    cal_df = pq.read_table(rd / "outcomes" / "FGOBJ_CALIBRATION.parquet").to_pandas()
    labels = I.load_labels("coco")
    cal_rows = I.load_cohort("coco", ("calibration",))
    for ci, q in enumerate(FINE):
        p = res["probes"][q]
        y = np.array([int(labels[(r["row_id"], q)]["label"]) for r in cal_rows])
        assert p["n_pos"] == int(y.sum()) and p["n_neg"] == int((1 - y).sum())
        g = cal_df[cal_df.concept == q].set_index("row_id").semantic_margin
        m = np.array([g.loc[r["row_id"]] for r in cal_rows], float)
        assert abs(p["answer_auroc"] - A.auroc_safe(y, m)) < 1e-9
        assert isinstance(p["readable"], bool) and isinstance(p["answer_capable"], bool)
        assert p["readable_status"] in ("readable", "not_readable", "insufficient_support", "insufficient_draws",
                                        "controls_ineligible")
        assert abs(p["auroc_test_prep"] - float(arr["auroc_test"][ci])) < 1e-12
        assert set(p["cos_model_to_easy"]) == set(EASY)
    # block comparison: the fine-grained family beside the block's own easy-COCO CORE cells, same rows and rule
    bc = res["block_comparison"]["families"]
    core = A.core("m", "coco", n_boot=120)
    cal = A.calibration("m", "coco")
    assert bc["fgobj"]["concepts"] == list(FINE) and bc["core"]["concepts"] == list(EASY)
    assert bc["core"]["owned"] == sorted(q for q in EASY if core["per_question"][q]["steering_reference"]
                                         and core["per_question"][q]["verdict"] == "fixed_family_advantage")
    assert abs(bc["core"]["median_answer_auroc"] - float(np.median([cal[q]["answer_auroc"] for q in EASY]))) < 1e-9
    assert abs(bc["core"]["median_selectivity"] - float(np.median([cal[q]["selectivity"] for q in EASY]))) < 1e-9
    assert bc["fgobj"]["owned_rate"] == len(bc["fgobj"]["owned"]) / len(FINE)
    assert res["block_comparison"]["owned_rate_difference_fgobj_minus_core"] == pytest.approx(
        bc["fgobj"]["owned_rate"] - bc["core"]["owned_rate"])
    # the CLI writes summary.json under the key `fgobj` and leaves the other keys alone
    import runpy
    (rd / "summary.json").write_text(json.dumps({"core": core, "calibration": cal}, **JSON))
    monkeypatch.setattr(sys, "argv", ["analysis", "--model-key", "m", "--dataset", "coco", "--what", "fgobj", "--n-boot", "120"])
    runpy.run_module("cftransfer.analysis", run_name="__main__")
    rep = json.loads((rd / "summary.json").read_text())
    assert set(rep["fgobj"]["concepts"]) == set(FINE) and "core" in rep and "calibration" in rep
    assert rep["fgobj"]["n_scored_rows_min"] == N_EVAL


def test_fgobj_analysis_without_the_calibration_companion(tmp_path, monkeypatch):
    """FGOBJ_CALIBRATION missing: the ownership grade still stands, and the cells are explicitly NOT answer-graded
    (so the matched comparison refuses to pool them rather than matching on a differently measured number)."""
    data, runs = _run_everything(tmp_path, monkeypatch, with_calibration=False)
    res = A.fgobj("m", "coco", draws=120)
    assert not res["probe_grade_source"]["answer_available"]
    assert "FGOBJ_CALIBRATION" in res["probe_grade_source"]["answer_missing_reason"]
    for q in FINE:
        assert "answer_auroc" not in res["probes"][q] and res["probes"][q]["readable"] in (True, False)
        assert res["per_question"][q]["owned"] in (True, False)
    assert res["block_comparison"]["families"]["fgobj"]["median_answer_auroc"] is None


# ------------------------------------------------------------------------------------- the matched comparison
def _cell(block, concept, group, owned, auroc, sel, model=None):
    return {"block": block, "model": model or block.split("/")[0], "dataset": block.split("/")[1], "concept": concept,
            "kind": "fgobj" if group == "coco_fine" else "core", "group": group, "owned": owned,
            "answer_auroc": auroc, "selectivity": sel, "readable": True, "answer_capable": True,
            "verdict": "fixed_family_advantage" if owned else "unresolved", "steering_reference": owned,
            "W_qq": 0.1, "O_q": 0.01, "n_pos": 20, "n_neg": 380}


def test_fgobj_matched_comparison():
    M2 = r2()
    assert (M2.FGOBJ_WINDOW_ANSWER_AUROC, M2.FGOBJ_WINDOW_SELECTIVITY) == (0.05, 0.05)
    chest = [_cell("a/nih", "Effusion", "chest", False, 0.70, 0.12), _cell("a/nih", "Mass", "chest", True, 0.72, 0.13),
             _cell("b/chexpert", "Edema", "chest", False, 0.99, 0.30)]      # the third is far from every partner
    # person is close to every chest cell on SELECTIVITY and far from all of them on ANSWER AUROC, so the three
    # matchings below genuinely differ: it enters the selectivity-only window and not the other two
    natural = [_cell("a/coco", "person", "coco_easy", True, 0.98, 0.10),
               _cell("a/coco", "handbag", "coco_fine", True, 0.71, 0.12),
               _cell("b/coco", "knife", "coco_fine", False, 0.73, 0.14)]
    # the window: both variables within 0.05, componentwise
    Mm = M2.match_matrix(chest, natural, 0.05, 0.05)
    assert Mm.tolist() == [[False, True, True], [False, True, True], [False, False, False]]
    assert M2.match_matrix(chest, natural, 0.05, None).tolist() == [[False, True, True], [False, True, True], [True, False, False]]
    assert M2.match_matrix(chest, natural, None, 0.05).tolist() == [[True, True, True], [True, True, True], [False, False, False]]
    res = M2.matched_comparison(chest, natural, 0.05, 0.05, "both", draws=200)
    assert res["n_chest_cells"] == 3 and res["n_matched_chest_cells"] == 2 and res["n_unmatched_chest_cells"] == 1
    assert res["unmatched_chest_cells"] == ["b/chexpert Edema"]
    assert res["n_natural_cells_used"] == 2 and res["partner_composition_used"] == {"coco_easy": 0, "coco_fine": 2}
    assert res["median_partners_per_matched_chest_cell"] == 2.0
    # matched difference: each matched chest cell against its OWN partner set (both cells see {owned, not owned} = 0.5)
    assert res["chest_owned_rate_matched"] == pytest.approx(0.5) and res["natural_owned_rate_used"] == pytest.approx(0.5)
    assert res["matched_ownership_difference"] == pytest.approx(0.0)
    assert res["pooled_ownership_difference"] == pytest.approx(0.0)
    lo, hi = res["matched_ownership_difference_ci95"]
    assert lo <= 0.0 <= hi and res["bootstrap_valid_draws"] > 0
    assert res["bootstrap_unit"].startswith("model")
    assert res["window_answer_auroc"] == 0.05 and res["window_selectivity"] == 0.05
    # a partner set that is entirely owned moves the difference to -1 for a chest cell that is not owned
    natural2 = [_cell("a/coco", "handbag", "coco_fine", True, 0.71, 0.12)]
    r1 = M2.matched_comparison(chest[:1], natural2, 0.05, 0.05, "both", draws=100)
    assert r1["n_matched_chest_cells"] == 1 and r1["matched_ownership_difference"] == pytest.approx(-1.0)
    assert r1["pooled_ownership_difference"] == pytest.approx(-1.0)
    # NO MATCH AT ALL: reported as such, with no estimate and no interval, never as a zero difference
    far = [_cell("z/coco", "person", "coco_easy", True, 0.10, 0.90)]
    r0 = M2.matched_comparison(chest, far, 0.05, 0.05, "both", draws=100)
    assert r0["n_matched_chest_cells"] == 0 and r0["n_unmatched_chest_cells"] == 3
    assert r0["matched_ownership_difference"] is None and r0["matched_ownership_difference_ci95"] is None
    assert r0["pooled_ownership_difference"] is None and r0["bootstrap_valid_draws"] == 0
    assert r0["n_natural_cells_used"] == 0
    # empty pools are a status, not a crash
    assert M2.matched_comparison([], natural, 0.05, 0.05, "both")["status"] == "NO_CELLS"
    assert M2.matched_comparison(chest, [], 0.05, 0.05, "both")["status"] == "NO_CELLS"
    # cells missing a matching variable never enter the matching
    assert len(M2._matchable(chest + [_cell("c/nih", "Nodule", "chest", False, None, 0.1)])) == len(chest)
    assert len(M2._matchable(chest + [_cell("c/nih", "Nodule", "chest", False, 0.7, float("nan"))])) == len(chest)


def test_fgobj_section_pools_every_group(tmp_path, monkeypatch):
    """fgobj_section over hand-built block summaries: chest, easy COCO and fine-grained COCO cells all enter the pool,
    a block whose FGOBJ_CALIBRATION is missing is skipped with the reason, and the matched comparison runs."""
    M2 = r2()

    def core_summary(concepts, owned):
        return {"per_question": {c: {"W_qq": 0.1, "O_q": 0.01, "verdict": "fixed_family_advantage" if c in owned else "unresolved",
                                     "steering_reference": c in owned} for c in concepts}}

    def cal_summary(concepts, auroc, sel):
        return {c: {"answer_auroc": auroc, "selectivity": sel, "readable": True, "answer_capable": True,
                    "n_pos": 20, "n_neg": 380} for c in concepts}

    def fgobj_summary(owned, auroc, sel, answer=True):
        return {"n_rows": 600, "n_scored_rows_min": 600, "concepts": list(FINE), "alpha": 0.25, "template_id": "IY",
                "random_seed": P.FGOBJ_RANDOM_SEED, "n_random": P.N_RANDOM, "median_abs_cos_to_easy": 0.1,
                "per_question": {c: {"owned": c in owned, "verdict": "fixed_family_advantage" if c in owned else "unresolved",
                                     "steering_reference": c in owned, "W_qq": 0.1, "O_q": 0.01} for c in FINE},
                "probes": {c: {"answer_auroc": auroc if answer else None, "selectivity": sel, "readable": True,
                               "answer_capable": True, "n_pos": 20, "n_neg": 380} for c in FINE},
                "probe_grade_source": {"answer_available": answer, "answer_missing_reason": None if answer else "no FGOBJ_CALIBRATION"}}

    done = ["CORE", "CALIBRATION", "FGOBJ", "FGOBJ_CALIBRATION"]
    blocks = {
        ("a", "nih"): {"run": {"completed_modules": ["CORE", "CALIBRATION"]},
                       "summary": {"core": core_summary(P.CONCEPTS["nih"], {"Mass"}),
                                   "calibration": cal_summary(P.CONCEPTS["nih"], 0.70, 0.12)}},
        ("a", "coco"): {"run": {"completed_modules": done},
                        "summary": {"core": core_summary(EASY, set(EASY)), "calibration": cal_summary(EASY, 0.98, 0.06),
                                    "fgobj": fgobj_summary({"handbag"}, 0.71, 0.12)}},
        ("b", "coco"): {"run": {"completed_modules": ["CORE", "CALIBRATION", "FGOBJ"]},
                        "summary": {"core": core_summary(EASY, set(EASY)), "calibration": cal_summary(EASY, 0.97, 0.05),
                                    "fgobj": fgobj_summary({"knife"}, None, 0.12, answer=False)}},
    }
    sec = M2.fgobj_section(blocks)
    assert sec["pool"]["chest"]["cells"] == 6 and sec["pool"]["chest"]["owned"] == 1
    assert sec["pool"]["coco_easy"]["cells"] == 12 and sec["pool"]["coco_easy"]["owned"] == 12
    assert sec["pool"]["coco_fine"]["cells"] == 6 and sec["pool"]["coco_fine"]["owned"] == 1     # only block a enters
    assert any("FGOBJ_CALIBRATION not scored" in s["reason"] for s in sec["skipped"])
    assert [r["block"] for r in sec["blocks"]] == ["a/coco"]
    assert sec["blocks"][0]["fine_owned"] == 1 and sec["blocks"][0]["easy_owned"] == len(EASY)
    m = sec["matched"]["both"]
    assert m["n_chest_cells"] == 6 and m["n_matched_chest_cells"] == 6           # every chest cell finds a fine partner
    assert m["partner_composition_used"] == {"coco_easy": 0, "coco_fine": 6}
    assert m["matched_ownership_difference"] == pytest.approx(1 / 6 - 1 / 6)
    # with only the easy cells as partners nothing matches: that is the starvation FGOBJ removes
    e = sec["matched"]["both_easy_partners_only"]
    assert e["n_matched_chest_cells"] == 0 and e["matched_ownership_difference"] is None
    assert sec["matched"]["answer_auroc"]["window_selectivity"] is None
    assert sec["matched"]["selectivity"]["window_answer_auroc"] is None
    assert "fixed before the FGOBJ grid was built" in sec["prespecification"]
