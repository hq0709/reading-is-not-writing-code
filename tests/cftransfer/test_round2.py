"""Round-2 modules: ALTDIRD (displacement lift, append-only npz), the PRECISION full grade, ATTR (attribute directions,
9-question module, readability / answerability) and ANSDIRT (answer directions under the other templates)."""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from cftransfer import altdir as AD, analysis as A, ansdir as AN, attr as AT, enqueue as E, fit as F, package as PK, protocol as P, \
    runner as RN  # noqa: E402
from cftransfer.images import load_labels  # noqa: E402
from test_altdir import _synthetic_fit  # noqa: E402
from test_primary_template import _elig  # noqa: E402
from test_runner_fake import _setup  # noqa: E402

CONCEPTS = P.CONCEPTS["nih"]


def test_round2_protocol_grids(tmp_path, monkeypatch):
    dd = P.conditions_for("ALTDIRD", "coco", "dog")
    assert [d for d, _ in dd] == [f"{f}:{c}" for f in ("dom_disp", "pattern_disp") for c in P.CONCEPTS["coco"]] and {a for _, a in dd} == {0.25}
    assert all(P.expected_rows("ALTDIRD", d) == 43_200 for d in P.DATASETS) and P.direction_kind("pattern_disp:Mass") == "pattern_disp"
    # ATTR: attribute questions first, own baseline only for them, sham per question, no random family
    assert P.question_list("nih", "ATTR") == [(q, "IY") for q in P.ATTR_CONCEPTS + CONCEPTS]
    assert P.question_list("chexpert", "ATTR", "IB") == [(q, "IB") for q in P.ATTR_CONCEPTS + P.CONCEPTS["chexpert"]]
    ca = P.conditions_for("ATTR", "nih", "sex_F"); cc = P.conditions_for("ATTR", "nih", "Mass")
    dirs = [f"attr:{a}" for a in P.ATTR_CONCEPTS] + [f"concept:{c}" for c in CONCEPTS]
    assert ca == [("baseline", 0.0)] + [(d, 0.25) for d in dirs] + [("attrsham:sex_F", 0.25)] and len(ca) == 11
    assert cc == [(d, 0.25) for d in dirs] + [("sham:Mass", 0.25)] and len(cc) == 10
    assert P.expected_rows("ATTR", "nih") == P.expected_rows("ATTR", "chexpert") == (3 * 11 + 6 * 10) * 600 == 55_800
    assert P.expected_rows("ATTR", "coco") == 0 and P.MODULES["ATTR"].concepts_rule == "attr"
    assert P.render_question("nih", "view_AP", "IY") == "Is there an anteroposterior (portable) projection in this chest radiograph? Answer yes or no."
    assert P.render_question("chexpert", "age_60", "WY") == "Does this chest radiograph show a patient aged sixty or older? Answer yes or no."
    assert P.render_question("nih", "sex_F", "IB").startswith("Is there a female patient in this chest radiograph? Answer B")
    assert P.CONCEPTS["nih"] == CONCEPTS and "view_AP" not in P.CONCEPTS["nih"]                  # protocol concept set untouched
    # ANSDIRT: five templates never substituted, own baseline, 8 conditions
    ct = P.conditions_for("ANSDIRT", "nih", "Mass")
    assert ct == [("baseline", 0.0)] + [(f"ans:{c}", 0.25) for c in CONCEPTS] + [("anssham:Mass", 0.25)]
    assert P.question_list("nih", "ANSDIRT", "IB") == [(c, t) for t in ("WY", "IA", "IB", "WA", "WB") for c in CONCEPTS]
    assert all(P.expected_rows("ANSDIRT", d) == 6 * 5 * 8 * 600 == 144_000 for d in P.DATASETS)
    for m, ds in (("ALTDIRD", P.DATASETS), ("ATTR", ("nih", "chexpert")), ("ANSDIRT", P.DATASETS)):
        assert m in P.ADDENDUM_MODULES and all(m in P.MODULES_ADDED_LATER[d] for d in ds) and m not in E.MODULE_ORDER
    assert E.SHARDS["ALTDIRD"] == 1 and E.SHARDS["ATTR"] == 1 and E.SHARDS["ANSDIRT"] == 2
    assert E.PREP_FILE["ALTDIRD"] == "altdir_seed0.npz" and E.PREP_FILE["ATTR"] == "attr_seed0.npz" and E.PREP_FILE["ANSDIRT"] == "ansdir_seed0.npz"
    q = tmp_path / "queue"; (q / "pending").mkdir(parents=True)
    monkeypatch.setattr(E, "QUEUE", q)
    names = E.enqueue("q25-7", "coco", ["ANSDIRT"], prep=False, prefix="1")
    assert len(names) == 3 and names[0].endswith("-ANSDIRT-prep-q25-7-coco") and names[2].endswith("-ANSDIRT-q25-7-coco-001of002")
    assert json.loads((q / "pending" / f"{names[1]}.json").read_text())["requires"][-1].endswith("ansdir_seed0.npz")
    names = E.enqueue("q25-7", "nih", ["ATTR", "ALTDIRD"], prep=False, prefix="1")
    reqs = [json.loads((q / "pending" / f"{nm}.json").read_text())["requires"][-1] for nm in names]
    assert reqs[0].endswith("attr_seed0.npz") and reqs[1].endswith("altdir_seed0.npz")


def test_displacement_lift_and_append_only_npz(tmp_path, monkeypatch):
    rng = np.random.default_rng(21)
    fit, Zs, Y = _synthetic_fit(rng)
    arr = AD.altdir_arrays(fit, Zs, Y)
    P_, s = fit["projection"].astype(np.float64), fit["scaler_scale"].astype(np.float64)
    G_inv, eig = AD.gram_inverse(fit["projection"])
    G = P_.T @ P_                                                       # rank 16 here (D = 16 < 512): pseudoinverse
    assert np.allclose(G @ G_inv @ G, G, atol=1e-8) and np.allclose(G_inv @ G @ G_inv, G_inv, atol=1e-6)
    assert np.allclose(np.sort(eig), np.linalg.eigvalsh(G / P_.shape[0])) and eig[0] == arr["disp_gram_eig_min"] and eig[-1] == arr["disp_gram_eig_max"]
    gs = AD.gram_spectrum_summary(eig)
    assert len(eig) == 512 and gs["rank"] == 16 and not gs["full_rank"] and gs["condition"] > 1 and int(arr["disp_gram_rank"]) == 16
    for fam, src in (("dom_disp", "dom"), ("pattern_disp", "pattern")):
        assert np.array_equal(arr[f"{fam}_projected"], arr[f"{src}_projected"])
        for ci in range(6):
            u = arr[f"{src}_projected"][ci].astype(np.float64)
            x = arr[f"{fam}_vectors"][ci].astype(np.float64)
            target = np.maximum(s, 1e-8) * u
            x_un = P_ @ (G_inv @ target)
            assert abs(np.linalg.norm(x) - 1) < 1e-5 and np.allclose(x, x_un / np.linalg.norm(x_un), atol=1e-6)
            # least-squares minimum-norm preimage: normal equations R R^T x = R z, and x in the column space of R
            assert np.allclose(P_ @ (P_.T @ x_un), P_ @ target, atol=1e-6 * np.linalg.norm(P_ @ target))
            coef, *_ = np.linalg.lstsq(P_, x, rcond=None)
            assert np.linalg.norm(P_ @ coef - x) < 1e-6
            assert np.allclose(x, AD.displacement_lift(fit["projection"], fit["scaler_scale"], u, G_inv), atol=1e-6)
            assert not np.allclose(x, arr[f"{src}_vectors"][ci], atol=1e-3)     # not the coefficient lift of the same u
    # full-rank case (every campaign locus: D >= 1024 > 512): exact preimage, R^T x proportional to diag(s) u
    Pf = (np.random.default_rng(0).standard_normal((640, 512)) / np.sqrt(512)).astype(np.float32)
    sf = np.random.default_rng(1).uniform(0.5, 2.0, 512).astype(np.float32)
    Gf_inv, eig_f = AD.gram_inverse(Pf)
    assert np.allclose(Gf_inv @ (Pf.astype(np.float64).T @ Pf.astype(np.float64)), np.eye(512), atol=1e-6) and AD.gram_spectrum_summary(eig_f)["full_rank"]
    u = np.random.default_rng(2).standard_normal(512)
    xf = AD.displacement_lift(Pf, sf, u, Gf_inv).astype(np.float64)
    back = Pf.astype(np.float64).T @ xf; target = sf.astype(np.float64) * u
    assert abs(back @ target / (np.linalg.norm(back) * np.linalg.norm(target)) - 1) < 1e-6
    fe = list(arr["family_order_ext"].astype(str))
    assert fe == ["logistic", "dom", "pattern", "orth", "resid", "dom_disp", "pattern_disp"] and arr["cos_model_ext"].shape == (6, 7, 7)
    assert np.allclose(arr["cos_model_ext"][:, :5, :5], arr["cos_model"]) and np.allclose(arr["cos_projected_ext"][:, :5, :5], arr["cos_projected"])
    assert np.allclose(arr["cos_projected_ext"][:, 1, 5], 1.0)                                    # same projected vector
    # append-only merge on the synthetic run: an older file without the extension gets the keys, existing arrays untouched
    ad, runs = _setup(tmp_path, monkeypatch)
    path, full = AD.build_altdir("m", "nih", "vis.last")
    old = {k: v for k, v in full.items() if "disp" not in k and not k.endswith("_ext")}
    np.savez(path, **old)
    path2, merged = AD.build_altdir("m", "nih", "vis.last")
    on_disk = dict(np.load(path2, allow_pickle=False))
    assert all(np.array_equal(on_disk[k], old[k]) for k in old) and "dom_disp_vectors" in on_disk and "cos_model_ext" in on_disk
    bad = dict(old); bad["dom_vectors"] = old["dom_vectors"] * -1
    np.savez(path, **bad)
    with pytest.raises(RuntimeError, match="not reproduced"):
        AD.build_altdir("m", "nih", "vis.last")


def test_altdird_runner_and_analysis(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    rd = runs / "m" / "nih"
    path, full = AD.build_altdir("m", "nih", "vis.last")
    np.savez(path, **{k: v for k, v in full.items() if "disp" not in k and not k.endswith("_ext")})     # pre-extension file
    with pytest.raises(FileNotFoundError, match="re-run"):
        RN.run_block("m", "nih", "ALTDIRD", 0, 1, batch=8, device_map="cpu", rows_per_part=4)
    AD.build_altdir("m", "nih", "vis.last")
    meta = RN.run_block("m", "nih", "ALTDIRD", 0, 1, batch=8, device_map="cpu", rows_per_part=4)
    assert meta["outcomes"] == 12 * 6 * 12
    df = pq.read_table(next((rd / "outcomes" / "ALTDIRD").glob("part-*.parquet"))).to_pandas()
    assert set(df.direction_kind) == {"dom_disp", "pattern_disp"} and "baseline" not in set(df.direction_id)
    RN.run_block("m", "nih", "CORE", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    PK.build("m", "nih")
    res = A.altdird("m", "nih", draws=100)
    alt = pq.read_table(rd / "outcomes" / "ALTDIRD.parquet").to_pandas()
    cdf = pq.read_table(rd / "outcomes" / "CORE.parquet").to_pandas()
    base = cdf[cdf.direction_id == "baseline"].set_index(["concept", "row_id"]).p_present
    assert res["families"] == ["dom_disp", "pattern_disp"] and res["gram_spectrum"]["n"] == 512 and res["gram_spectrum"]["rank"] == 16
    assert res["gram_spectrum"]["condition"] > 1 and not res["gram_spectrum"]["full_rank"]
    for fam, src in (("dom_disp", "dom"), ("pattern_disp", "pattern")):
        for q in CONCEPTS:
            cell = res[fam]["per_question"][q]
            g = alt[(alt.concept == q) & (alt.direction_id == f"{fam}:{q}")]
            assert abs(cell["W_qq"] - float(np.mean(g.p_present.values - base.loc[[(q, r) for r in g.row_id]].values))) < 1e-9
            assert f"cos_to_{src}_model" in cell and "own_minus_max_logistic_competitor" in cell and cell["verdict"] in ("stronger_competitor", "fixed_family_advantage", "unresolved")
            assert abs(cell["cos_to_logistic_model"] - res["cosines"][q]["logistic"][fam]) < 1e-12


def test_precision_full_grade(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    RN.run_block("m", "nih", "CORE", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    for st in P.PRECISION_SETTINGS:
        RN.run_block("m", "nih", "PRECISION", 0, 1, batch=32, device_map="cpu", rows_per_part=4, numerics=st)
    PK.build("m", "nih")
    pr = A.precision("m", "nih", draws=120)
    c200 = A.core("m", "nih", n_boot=120, row_limit=12)
    assert set(pr["core"]["grade"]) == set(CONCEPTS) and pr["draws"] == 120
    for q in CONCEPTS:
        assert pr["core"]["grade"][q]["verdict"] == c200["per_question"][q]["verdict"]
    for st in P.PRECISION_SETTINGS:
        g = pr[st]["grade"]
        assert g["n_verdict_changes"] == 0 and g["n_steering_reference_changes"] == 0        # the fake model is numerics-independent
        assert g["max_abs_dW_grid"] < 1e-5 and g["max_abs_dcontrast"] < 1e-5 and g["n_grid_cells_compared"] == 6 * 126
        for q in CONCEPTS:
            cell = g["per_question"][q]
            assert cell["verdict"] in ("stronger_competitor", "fixed_family_advantage", "unresolved") and not cell["verdict_changed"]
            assert isinstance(cell["steering_reference"], bool) and abs(cell["dO_q"]) < 1e-5 and len(cell["O_q_ci95_percentile"]) == 2
        assert pr[st]["n_agree_competitor"] == 6                                              # point-estimate agreements kept
    # core() on a row-limited CORE and on a setting keep only those rows
    g32 = A.core("m", "nih", "PRECISION", n_boot=50, numerics="fp32")
    assert g32["n_rows"] == 12 and set(g32["W"]["Mass"]) == set(c200["W"]["Mass"])
    with pytest.raises(ValueError, match="numerics"):
        A.core("m", "nih", "PRECISION", n_boot=50, numerics="nope")


def test_attr_prep_runner_and_analysis(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    rd = runs / "m" / "nih"
    with pytest.raises(FileNotFoundError, match="cftransfer.attr"):
        RN.run_block("m", "nih", "ATTR", 0, 1, batch=8, device_map="cpu", rows_per_part=4, limit_rows=2)
    with pytest.raises(RuntimeError, match="per class"):
        AT.build_attr("m", "nih", "vis.last", out_dir=tmp_path / "x", min_class_rows=100)      # 60 synthetic rows
    path, arr = AT.build_attr("m", "nih", "vis.last", min_class_rows=5)
    assert path == rd / "fits" / "vis.last" / "attr_seed0.npz" and list(arr["attr_names"].astype(str)) == P.ATTR_CONCEPTS
    # labels follow the manifest coding and the directions are the protocol fit on them
    f0 = F.load_fit("m", "nih", "vis.last", 0)
    ids, X, _v, roles = F.load_features("m", "nih", "vis.last")
    labels = load_labels("nih")
    tr = roles == "train"
    Zs = ((X.astype(np.float32) @ f0["projection"] - f0["scaler_mean"]) / np.maximum(f0["scaler_scale"], 1e-8)).astype(np.float32)
    Y, types = AT.attribute_labels("nih", ids)
    for j, rid in enumerate(ids[:8]):
        lab = labels[(rid, "Effusion")]
        assert Y[0, j] == (1 if lab["view"] == "AP" else 0) and Y[1, j] == (1 if lab["sex"] == "F" else 0) and Y[2, j] == int(float(lab["age"]) >= 60)
        assert types[j] == lab["type_id"]
    for ai in range(3):
        w = F.fit_lr(Zs[tr][Y[ai][tr] >= 0], Y[ai][tr][Y[ai][tr] >= 0]).coef_[0]
        assert np.allclose(arr["attr_coefficients"][ai], w, atol=1e-6)
        assert np.allclose(arr["attr_vectors"][ai], F.direction_from_projected(f0["projection"], f0["scaler_scale"], w), atol=1e-6)
        assert np.array_equal(arr["attr_sham_vectors"][ai], arr["attr_vectors"][ai][arr["attr_sham_permutations"][ai]])
    rng = np.random.default_rng(P.ATTR_SHAM_SEED)
    assert np.array_equal(arr["attr_sham_permutations"][0], rng.permutation(16))
    assert arr["cos_model"].shape == (3, 6) and arr["cal_real_logits"].shape == (3, 12) and arr["cal_control_logits"].shape == (3, 20, 12)
    assert arr["cal_labels"].shape == (3, 12) and arr["cal_control_labels"].shape == (20, 12) and bool(arr["controls_eligible"])
    # runner: nine questions; baseline rows only for the attribute questions
    meta = RN.run_block("m", "nih", "ATTR", 0, 1, batch=8, device_map="cpu", rows_per_part=4)
    assert meta["outcomes"] == 12 * (3 * 11 + 6 * 10)
    df = pq.read_table(next((rd / "outcomes" / "ATTR").glob("part-*.parquet"))).to_pandas()
    assert set(df.concept) == set(P.ATTR_CONCEPTS + CONCEPTS) and set(df[df.direction_id == "baseline"].concept) == set(P.ATTR_CONCEPTS)
    assert set(df.direction_kind) == {"baseline", "attr", "concept", "attrsham", "sham"}
    for (rid, q), g in df.groupby(["row_id", "concept"]):
        assert sorted(g.direction_id) == sorted(d for d, _ in P.conditions_for("ATTR", "nih", q))
    # analysis: 9x9 write matrix, ownership over the enlarged family, attribute readability / answerability
    RN.run_block("m", "nih", "CORE", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    PK.build("m", "nih")
    res = A.attr("m", "nih", draws=120)
    at = pq.read_table(rd / "outcomes" / "ATTR.parquet").to_pandas()
    cdf = pq.read_table(rd / "outcomes" / "CORE.parquet").to_pandas()
    base_core = cdf[cdf.direction_id == "baseline"].set_index(["concept", "row_id"]).p_present
    base_attr = at[at.direction_id == "baseline"].set_index(["concept", "row_id"]).p_present
    Q = P.ATTR_CONCEPTS + CONCEPTS
    assert res["questions"] == Q and len(res["directions"]) == 9 and len(res["contrasts"]) == 9 * 8
    for q in Q:
        assert set(res["W"][q]) == set(res["directions"])
        cell = res["per_question"][q]
        attribute = q in P.ATTR_CONCEPTS
        own = f"attr:{q}" if attribute else f"concept:{q}"
        g = at[(at.concept == q) & (at.direction_id == own)]
        b = base_attr if attribute else base_core
        assert abs(cell["W_qq"] - float(np.mean(g.p_present.values - b.loc[[(q, r) for r in g.row_id]].values))) < 1e-9
        assert cell["question_kind"] == ("attribute" if attribute else "clinical") and cell["n_competitors"] == 8
        assert cell["random_reference"] == (not attribute) and (cell["random_p95"] is None) == attribute
        assert cell["verdict"] in ("stronger_competitor", "fixed_family_advantage", "unresolved") and isinstance(cell["steering_reference"], bool)
    for a in P.ATTR_CONCEPTS:
        cell = res["attributes"][a]
        assert {"auroc_real", "control_mean", "selectivity", "readable", "readable_status", "answer_auroc", "answer_capable"} <= set(cell)
        assert cell["readable_status"] in ("readable", "insufficient_support", "controls_ineligible", "insufficient_draws", "not_readable")
        assert set(cell["cos_model_to_clinical"]) == set(CONCEPTS)


def test_ansdirt_runner_and_analysis(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(AN, "get_adapter", lambda key, rev=None: ad)
    monkeypatch.setattr(AN, "image_path", lambda ds, row: row["row_id"])
    monkeypatch.setattr(AN, "open_rgb", lambda p: p)
    rd = runs / "m" / "nih"
    _elig(rd, IY=True, WY=True, IA=True, IB=True, WA=False, WB=True)                            # WA INELIGIBLE by preflight
    with pytest.raises(FileNotFoundError, match="cftransfer.ansdir"):
        RN.run_block("m", "nih", "ANSDIRT", 0, 1, batch=8, device_map="cpu", rows_per_part=4)
    AN.build_ansdir("m", "nih", n_train=20, device_map="cpu")
    meta = RN.run_block("m", "nih", "ANSDIRT", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    assert meta["outcomes"] == 12 * 6 * 4 * 8 and meta["ineligible_templates"] == ["WA"] if "ineligible_templates" in meta else meta["outcomes"] == 12 * 6 * 4 * 8
    df = pq.read_table(next((rd / "outcomes" / "ANSDIRT").glob("part-*.parquet"))).to_pandas()
    assert set(df.template_id) == {"WY", "IA", "IB", "WB"} and set(df.direction_kind) == {"baseline", "ans", "anssham"}
    for (rid, q, t), g in df.groupby(["row_id", "concept", "template_id"]):
        assert sorted(g.direction_id) == sorted(d for d, _ in P.conditions_for("ANSDIRT", "nih", q))
    RN.run_block("m", "nih", "CORE", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    RN.run_block("m", "nih", "ANSDIR", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    RN.run_block("m", "nih", "PROMPT", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    PK.build("m", "nih")
    res = A.ansdirt("m", "nih", draws=100)
    iy = A.ansdir("m", "nih", draws=100)
    owned_iy = sorted(q for q, c in iy["per_question"].items() if c["steering_reference"] and c["verdict"] == "fixed_family_advantage")
    assert res["templates"] == ["WY", "IA", "IB", "WB"] and res["ineligible_templates"] == ["WA"] and res["iy_owned"] == owned_iy
    ans = pq.read_table(rd / "outcomes" / "ANSDIRT.parquet").to_pandas()
    for t in res["templates"]:
        blk = res["per_template"][t]
        base = ans[(ans.direction_id == "baseline") & (ans.template_id == t)].set_index(["concept", "row_id"]).p_present
        assert len(blk["contrasts"]) == 30 and blk["n_scored_rows_min"] == 12 and set(blk["owned"]) <= set(CONCEPTS)
        for q in CONCEPTS:
            cell = blk["per_question"][q]
            g = ans[(ans.concept == q) & (ans.direction_id == f"ans:{q}") & (ans.template_id == t)]
            assert abs(cell["W_qq"] - float(np.mean(g.p_present.values - base.loc[[(q, r) for r in g.row_id]].values))) < 1e-9
            if q in P.PROMPT_CONCEPTS["nih"]:
                assert cell["logistic_source"] == "PROMPT" and cell["random_reference"] and len(cell["own_minus_max_logistic_competitor_ci95_percentile"]) == 2
                assert abs(cell["own_minus_max_logistic_competitor"] - (cell["W_qq"] - cell["max_other_logistic"])) < 1e-12
            else:
                assert "logistic_source" not in cell and not cell["random_reference"] and cell["random_p95"] is None
            assert cell["steering_reference"] == (cell["W_qq"] > 0 and cell["W_qq"] > cell["abs_sham"] and
                                                  (not cell["random_reference"] or cell["W_qq"] > cell["random_p95"]))
        tr = res["transfer"][t]
        assert tr["n_owned_IY"] == len(owned_iy) and set(tr["stay"]) | set(tr["lost"]) == set(owned_iy) and set(tr["stay"]) <= set(blk["owned"])
