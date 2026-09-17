"""Round-6 analysis corrections, all recomputations over material already on disk:

  * the difficulty-matched comparison reports TWO estimators, each labelled and each with its own interval and its own
    effective sample, and the pooled one is exactly the difference of the two printed rates
    (scripts/mayo/robustness_round2.py matched_comparison);
  * the fine-grained object family is compared with the easy objects OF THE SAME BLOCKS, paired per block with an exact
    sign test (same_block_comparison, sign_test);
  * the incremental validity of ownership is tested OUT OF SAMPLE, with the penalty and the standardisation fitted
    inside each training fold and a concept-level permutation null (semend_frame, semend_design, _heldout_r2,
    semend_section);
  * the expert-label refit is compared with report-label refits at the SAME number of rows, so label source and
    estimation sample are separated (scripts/mayo/robustness_validfit.py).
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from cftransfer import protocol as P  # noqa: E402

EASY = P.CONCEPTS["coco"]
FINE = P.FGOBJ_CONCEPTS


def _script(name: str, mod_name: str):
    """A scripts/mayo/*.py module loaded by path (they are scripts, not package modules)."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "mayo" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def r2():
    return _script("robustness_round2", "robustness_round2_r6")


def vf():
    return _script("robustness_validfit", "robustness_validfit_r6")


def _cell(block, concept, group, owned, auroc, sel):
    return {"block": block, "model": block.split("/")[0], "dataset": block.split("/")[1], "concept": concept,
            "kind": "fgobj" if group == "coco_fine" else "core", "group": group, "owned": owned,
            "answer_auroc": auroc, "selectivity": sel, "readable": True, "answer_capable": True}


# ------------------------------------------------------------------- the two estimators of the matched comparison

def test_matched_and_pooled_are_different_estimators_and_both_are_reported():
    """The pooled difference is the difference of the two printed rates. The matched difference averages each chest
    cell against its OWN partner set, so an unequal number of partners per cell makes the two differ. Both are
    reported, each with its own interval and its own effective sample."""
    M2 = r2()
    # a: one chest cell, not owned, with ONE owned partner            -> its own gap is -1
    # b: one chest cell, owned, with TWO partners, one of them owned  -> its own gap is +0.5
    # pooled: chest rate 0.5 minus partner rate 2/3 = -1/6; matched: mean(-1, +0.5) = -0.25. They are not the same.
    chest = [_cell("a/nih", "Effusion", "chest", False, 0.70, 0.10), _cell("b/nih", "Mass", "chest", True, 0.90, 0.10)]
    natural = [_cell("a/coco", "handbag", "coco_fine", True, 0.70, 0.10),      # partner of the first only
               _cell("b/coco", "person", "coco_easy", True, 0.90, 0.10),       # partners of the second only
               _cell("b/coco", "knife", "coco_fine", False, 0.90, 0.10)]
    res = M2.matched_comparison(chest, natural, 0.05, 0.05, "both", draws=200)
    assert res["chest_owned_rate_matched"] == pytest.approx(0.5)
    assert res["natural_owned_rate_used"] == pytest.approx(2 / 3)
    assert res["pooled_ownership_difference"] == pytest.approx(0.5 - 2 / 3)
    assert res["matched_ownership_difference"] == pytest.approx(-0.25)
    assert res["matched_ownership_difference"] != pytest.approx(res["pooled_ownership_difference"])
    # the pooled estimator IS the difference of the two reported rates, and the file says so
    assert res["rate_difference"] == pytest.approx(res["pooled_ownership_difference"])
    assert res["pooled_equals_rate_difference"] is True
    assert res["matched_minus_rate_difference"] == pytest.approx(-0.25 - (0.5 - 2 / 3))
    # both estimands are written out, and neither wording is the other's
    assert set(res["estimands"]) == {"matched", "pooled"}
    assert "its own partner set" in res["estimands"]["matched"]
    assert "difference of the two reported rates" in res["estimands"]["pooled"]
    # both carry an interval over the same draws
    for key in ("matched_ownership_difference_ci95", "pooled_ownership_difference_ci95"):
        lo, hi = res[key]
        assert lo <= hi
    assert res["bootstrap_valid_draws"] > 0 and res["bootstrap_valid_draws_pooled"] > 0
    # the effective sample of each estimator: matched chest cells, and DISTINCT partners actually used
    es = res["effective_sample"]
    assert es == {"matched_chest_cells": 2, "distinct_partners_used": 3, "total_chest_cells": 2, "total_natural_cells": 3}
    assert (es["matched_chest_cells"], es["distinct_partners_used"]) == (res["n_matched_chest_cells"], res["n_natural_cells_used"])
    # a partner that matches nothing is not counted in the effective sample
    far = natural + [_cell("c/coco", "book", "coco_easy", True, 0.10, 0.90)]
    r2res = M2.matched_comparison(chest, far, 0.05, 0.05, "both", draws=100)
    assert r2res["effective_sample"] == {"matched_chest_cells": 2, "distinct_partners_used": 3,
                                         "total_chest_cells": 2, "total_natural_cells": 4}
    # with no match at all there is no estimate of either kind and no interval of either kind
    none = M2.matched_comparison(chest, [_cell("z/coco", "book", "coco_easy", True, 0.10, 0.90)], 0.05, 0.05, "both", draws=50)
    assert none["matched_ownership_difference"] is None and none["pooled_ownership_difference"] is None
    assert none["rate_difference"] is None and none["pooled_equals_rate_difference"] is False
    assert none["effective_sample"]["matched_chest_cells"] == 0


# ------------------------------------------------------- fine-grained against easy objects inside the SAME blocks

def test_sign_test_is_the_exact_two_sided_binomial():
    M2 = r2()
    assert M2.sign_test(0, 0) is None                                            # every pair tied
    assert M2.sign_test(0, 3) == pytest.approx(2 * 0.5 ** 3)                     # 3 of 3 one way
    assert M2.sign_test(3, 0) == pytest.approx(2 * 0.5 ** 3)                     # symmetric in its arguments
    assert M2.sign_test(1, 2) == pytest.approx(1.0)                              # 1 of 3 is the whole distribution
    assert M2.sign_test(0, 5) == pytest.approx(2 * 0.5 ** 5)
    assert M2.sign_test(1, 5) == pytest.approx(2 * (math.comb(6, 0) + math.comb(6, 1)) / 2 ** 6)
    for pos, neg in ((0, 4), (1, 7), (3, 9), (5, 5)):
        assert 0.0 < M2.sign_test(pos, neg) <= 1.0


def test_same_block_comparison_holds_the_checkpoint_set_fixed():
    """The easy counts that enter are the ones of the blocks the fine-grained cells come from, and the comparison is
    paired inside a block. It never pools blocks the fine-grained family was not run in."""
    M2 = r2()
    rows = [{"block": "a/coco", "fine_owned": 6, "fine_cells": 6, "easy_owned": 6, "easy_cells": 6},
            {"block": "b/coco", "fine_owned": 5, "fine_cells": 6, "easy_owned": 6, "easy_cells": 6},
            {"block": "c/coco", "fine_owned": 3, "fine_cells": 6, "easy_owned": 5, "easy_cells": 6}]
    sb = M2.same_block_comparison(rows)
    assert sb["n_blocks"] == 3 and sb["blocks"] == ["a/coco", "b/coco", "c/coco"]
    assert (sb["fine_owned"], sb["fine_cells"]) == (14, 18) and (sb["easy_owned"], sb["easy_cells"]) == (17, 18)
    assert sb["fine_rate"] == pytest.approx(14 / 18) and sb["easy_rate"] == pytest.approx(17 / 18)
    assert sb["rate_difference"] == pytest.approx(14 / 18 - 17 / 18)
    assert [p["difference"] for p in sb["per_block"]] == [0, -1, -2]
    assert (sb["paired_difference_min"], sb["paired_difference_max"]) == (-2, 0)
    assert sb["paired_difference_mean"] == pytest.approx(-1.0)
    assert (sb["n_blocks_fine_higher"], sb["n_blocks_easy_higher"], sb["n_blocks_tied"]) == (0, 2, 1)
    assert sb["sign_test_p"] == pytest.approx(M2.sign_test(0, 2))               # the tied block leaves the test
    assert "grid-wide easy rate" in sb["note"]


def test_fgobj_section_separates_the_grid_wide_pool_from_the_same_block_pool(tmp_path):
    """The easy-object pool of every COCO block and the easy-object pool of the FGOBJ blocks are two entries, and the
    same-block one spans exactly the blocks that carry fine-grained cells."""
    M2 = r2()

    def core_summary(concepts, owned):
        return {"per_question": {c: {"W_qq": 0.1, "O_q": 0.01,
                                     "verdict": "fixed_family_advantage" if c in owned else "unresolved",
                                     "steering_reference": c in owned} for c in concepts}}

    def cal_summary(concepts, auroc, sel):
        return {c: {"answer_auroc": auroc, "selectivity": sel, "readable": True, "answer_capable": True,
                    "n_pos": 20, "n_neg": 380} for c in concepts}

    def fg_summary(owned):
        return {"n_rows": 600, "n_scored_rows_min": 600, "concepts": list(FINE), "alpha": 0.25, "template_id": "IY",
                "random_seed": P.FGOBJ_RANDOM_SEED, "n_random": P.N_RANDOM, "median_abs_cos_to_easy": 0.1,
                "per_question": {c: {"owned": c in owned, "verdict": "fixed_family_advantage" if c in owned else "unresolved",
                                     "steering_reference": c in owned, "W_qq": 0.1, "O_q": 0.01} for c in FINE},
                "probes": {c: {"answer_auroc": 0.97, "selectivity": 0.06, "readable": True, "answer_capable": True,
                               "n_pos": 20, "n_neg": 380} for c in FINE},
                "probe_grade_source": {"answer_available": True, "answer_missing_reason": None}}

    done = ["CORE", "CALIBRATION", "FGOBJ", "FGOBJ_CALIBRATION"]
    blocks = {
        ("a", "coco"): {"run": {"completed_modules": done},
                        "summary": {"core": core_summary(EASY, set(EASY)), "calibration": cal_summary(EASY, 0.97, 0.06),
                                    "fgobj": fg_summary(set(FINE))}},
        # b has FGOBJ too, and one easy object of six is not owned there
        ("b", "coco"): {"run": {"completed_modules": done},
                        "summary": {"core": core_summary(EASY, set(EASY[:-1])), "calibration": cal_summary(EASY, 0.97, 0.06),
                                    "fgobj": fg_summary(set(FINE[:-2]))}},
        # c is a COCO block WITHOUT the fine-grained family: it belongs to the grid-wide pool only
        ("c", "coco"): {"run": {"completed_modules": ["CORE", "CALIBRATION"]},
                        "summary": {"core": core_summary(EASY, set()), "calibration": cal_summary(EASY, 0.97, 0.06)}},
    }
    sec = M2.fgobj_section(blocks)
    P_ = sec["pool"]
    assert P_["coco_easy"]["blocks"] == 3 and P_["coco_easy"]["cells"] == 18
    assert P_["coco_easy"]["owned"] == len(EASY) + len(EASY) - 1                        # c owns nothing
    assert P_["coco_easy_same_blocks"]["blocks"] == 2 and P_["coco_easy_same_blocks"]["cells"] == 12
    assert P_["coco_easy_same_blocks"]["owned"] == len(EASY) + len(EASY) - 1            # a and b only
    assert P_["coco_easy_same_blocks"]["block_list"] == ["a/coco", "b/coco"]
    assert P_["coco_easy"]["owned_rate"] != pytest.approx(P_["coco_easy_same_blocks"]["owned_rate"])
    assert "not the comparison for fine-grainedness" in P_["coco_easy_same_blocks"]["note"]
    sb = sec["same_block_comparison"]
    assert sb["n_blocks"] == 2 and sb["easy_cells"] == P_["coco_easy_same_blocks"]["cells"]
    assert sb["easy_owned"] == P_["coco_easy_same_blocks"]["owned"]
    assert (sb["fine_owned"], sb["fine_cells"]) == (P_["coco_fine"]["owned"], P_["coco_fine"]["cells"])
    assert M2.FGOBJ_POOLS == M2.FGOBJ_GROUPS + ("coco_easy_same_blocks",)


# --------------------------------------------------------- incremental validity of ownership, tested out of sample

def _semend_summary(n_ep=4, n_concepts=6, seed=0, signal=0.0):
    """A synthetic summary.json `semend` section: the target is a linear function of the base predictors plus
    `signal` times the ownership contrast, so a test can put a real effect in or leave one out."""
    rng = np.random.default_rng(seed)
    concepts = [f"C{i}" for i in range(n_concepts)]
    eps = [f"E{j}" for j in range(n_ep)]
    w = rng.normal(size=n_concepts)
    sel = rng.normal(size=n_concepts)
    auc = rng.uniform(0.5, 1.0, n_concepts)
    oq = rng.normal(size=n_concepts)
    own = (oq > 0).astype(float)
    out = {"endpoints": eps, "core": {c: {"W_qq": float(w[i]), "O_q": float(oq[i])} for i, c in enumerate(concepts)},
           "per_endpoint": {}, "incremental_validity": {}}
    for j, e in enumerate(eps):
        fams = {}
        for fam in ("label", "answer"):
            pq = {}
            for i, c in enumerate(concepts):
                y = 0.7 * w[i] + 0.3 * sel[i] - 0.2 * auc[i] + 0.05 * j + signal * oq[i] + 0.01 * rng.normal()
                pq[c] = {"selectivity": float(y), "W_qq_core": float(w[i]), "probe_selectivity": float(sel[i]),
                         "answer_auroc": float(auc[i]), "owned_core": bool(own[i]), "O_q_core": float(oq[i])}
            fams[fam] = {"per_question": pq}
        out["per_endpoint"][e] = {"families": fams}
    for fam in ("label", "answer"):
        out["incremental_validity"][fam] = {"base_model": {"r2": 0.5}, "full_model": {"r2": 0.6},
                                            "ownership_adds": {"delta_r2": 0.1}}
    return {"semend": out}


def test_semend_frame_and_design_rebuild_the_regression_from_the_summary_alone():
    M2 = r2()
    s = _semend_summary()
    eps, concepts, A, own, oq = M2.semend_frame(s, "label")
    assert eps == [f"E{j}" for j in range(4)] and concepts == [f"C{i}" for i in range(6)]
    assert A.shape == (24, 6)                                    # 6 concepts x 4 endpoints, six columns
    pq = s["semend"]["per_endpoint"]["E0"]["families"]["label"]["per_question"]
    assert own.tolist() == [1.0 if pq[c]["owned_core"] else 0.0 for c in concepts]
    assert oq == pytest.approx([pq[c]["O_q_core"] for c in concepts])
    Xb, Xf = M2.semend_design(A, len(eps), own, oq)
    # intercept, three endpoint dummies, three base predictors; the full model adds exactly the two ownership terms
    assert Xb.shape == (24, 1 + 3 + len(M2.SEMEND_BASE)) and Xf.shape == (24, Xb.shape[1] + len(M2.SEMEND_ADDED))
    assert np.array_equal(Xf[:, :Xb.shape[1]], Xb)
    assert Xb[:, 0].tolist() == [1.0] * 24
    assert Xf[:, -2].tolist() == [own[int(q)] for q in A[:, 0]]
    assert Xf[:, -1] == pytest.approx([oq[int(q)] for q in A[:, 0]])
    # the two families share the concept-level ownership, which is why one permutation per block serves both
    _, _, _, own_a, oq_a = M2.semend_frame(s, "answer")
    assert own_a.tolist() == own.tolist() and oq_a == pytest.approx(oq)


def test_heldout_r2_never_lets_a_fold_see_itself_and_can_go_negative():
    """A training R^2 cannot fall when predictors are added; a held-out one can, which is the whole point of the
    replacement. The fold's own rows enter neither the standardisation nor the penalty choice."""
    M2 = r2()
    rng = np.random.default_rng(3)
    y = rng.normal(size=30)                                       # pure noise: nothing is predictable out of sample
    X = np.column_stack([np.ones(30), rng.normal(size=(30, 4))])
    fold = np.repeat(np.arange(6), 5)
    r = M2._heldout_r2(y, X, fold, fold)
    assert r < 0.0                                                # worse than the mean, as it must be on noise
    # a fold's prediction does not depend on that fold's own rows: perturbing them leaves the prediction alone
    seen = {}
    for f in np.unique(fold):
        te = fold == f
        seen[f] = M2._fit_predict(X[~te], y[~te], X[te], fold[~te])
    y2 = y.copy()
    y2[fold == 0] += 100.0
    for f in np.unique(fold):
        if f == 0:
            continue
        te = fold == f
        tr = ~te
        assert not np.allclose(M2._fit_predict(X[tr], y2[tr], X[te], fold[tr]), seen[f])   # other folds do matter
    te0 = fold == 0
    assert M2._fit_predict(X[~te0], y2[~te0], X[te0], fold[~te0]) == pytest.approx(seen[0])
    # the ridge solution is the least-squares one at a vanishing penalty and shrinks to the intercept at a huge one
    b_small = M2._ridge(X, y, 1e-12)
    assert b_small == pytest.approx(np.linalg.lstsq(X, y, rcond=None)[0], abs=1e-6)
    b_big = M2._ridge(X, y, 1e12)
    assert abs(b_big[0] - y.mean()) < 1e-3 and np.allclose(b_big[1:], 0.0, atol=1e-6)


def test_semend_section_reports_the_null_and_finds_a_real_effect_when_there_is_one():
    M2 = r2()
    M2.SEMEND_CV_PERMUTATIONS = 200                                # the section reads the constant at call time
    run = {"completed_modules": ["CORE", "CALIBRATION", "SEMEND"]}
    blocks = {("a", "nih"): {"run": run, "summary": _semend_summary(seed=1)},
              ("b", "nih"): {"run": run, "summary": _semend_summary(seed=2)},
              ("c", "nih"): {"run": run, "summary": _semend_summary(seed=3)}}
    sec = M2.semend_section(blocks)
    assert sec["n_blocks"] == 3 and sec["permutations"] == 200
    assert len(sec["leave_one_concept_out"]) == 6 and len(sec["leave_one_block_out"]) == 2
    assert sec["base_predictors"] == list(M2.SEMEND_BASE) and sec["added_predictors"] == list(M2.SEMEND_ADDED)
    for r in sec["leave_one_concept_out"]:
        assert r["n_folds"] == 6 and r["n_cells"] == 24
        assert r["delta_heldout_r2"] == pytest.approx(r["heldout_r2_full"] - r["heldout_r2_base"])
        assert 0.0 < r["permutation_p"] <= 1.0
        assert "training_r2_base" in r and "training_delta_r2" in r       # the in-sample number is kept as a reference
    for r in sec["leave_one_block_out"]:
        assert r["n_folds"] == 3 and r["n_cells"] == 72
    assert sec["pooled_leave_one_concept_out"]["n_tests"] == 6
    assert sec["pooled_leave_one_block_out"]["n_tests"] == 2
    assert "permuted across the concepts of a block" in sec["permutation_null"]
    assert "cannot lower a training R^2" in sec["training_r2_note"]
    # ownership carries nothing here beyond the base predictors, and the section says so
    assert sec["ownership_adds_out_of_sample"] is False
    # with a real concept-level effect the same machinery finds it
    strong = {k: {"run": run, "summary": _semend_summary(seed=10 + i, signal=3.0)}
              for i, k in enumerate([("a", "nih"), ("b", "nih"), ("c", "nih")])}
    hit = M2.semend_section(strong)
    assert hit["pooled_leave_one_concept_out"]["mean_delta_heldout_r2"] > 0
    assert hit["pooled_leave_one_concept_out"]["permutation_p"] < 0.05
    assert hit["ownership_adds_out_of_sample"] is True
    # a block with SEMEND completed but no summary section is named, never silently dropped
    missing = dict(blocks)
    missing[("d", "nih")] = {"run": run, "summary": {}}
    assert any(s["block"] == "d/nih" for s in M2.semend_section(missing)["skipped"])


# ------------------------------------------------ the expert-label refit against report-label refits of the same size

def test_validfit_subsample_takes_whole_patients_up_to_the_target():
    VF = vf()
    units = np.array([f"p{i // 4}" for i in range(40)])            # ten patients, four rows each
    rng = np.random.default_rng(0)
    idx = VF.subsample_indices(units, rng, 20)
    assert len(idx) == 20 and len(set(idx.tolist())) == 20
    taken = units[idx]
    # every patient that contributes contributes all four of its rows (the target is a multiple of the patient size)
    assert all(c == 4 for c in np.unique(taken, return_counts=True)[1])
    # a mask restricts both the count and the rows kept
    mask = np.zeros(40, bool)
    mask[:12] = True
    idx2 = VF.subsample_indices(units, rng, 20, mask)
    assert len(idx2) == 12 and mask[idx2].all()


def test_validfit_arms_separate_the_label_source_from_the_estimation_sample():
    """Two probes fitted on the same rows from different label sets, and two fitted on different numbers of rows from
    one label set, are four distinct arms; the helpers that build them keep the evaluation rows and the scorer's
    ignorance of them fixed."""
    VF = vf()
    rng = np.random.default_rng(11)
    n, d, K = 200, 8, 3
    Z = rng.normal(size=(n, d))
    beta = rng.normal(size=(K, d))
    Y = np.stack([(Z @ beta[k] + 0.3 * rng.normal(size=n) > 0).astype(np.int8) for k in range(K)])
    folds = VF.fold_assignment([f"p{i // 4}" for i in range(n)])
    coef, ok, used = VF.fit_directions(Z, Y, K)
    assert ok.all() and used.tolist() == [n] * K and coef.shape == (K, d)
    held, fu, ru = VF.crossfit(Z, Y, folds, Z, folds, K)
    assert np.isfinite(held).all() and fu.tolist() == [VF.VALIDFIT_FOLDS] * K
    assert all(0 < r < n for r in ru)                              # each fit saw a strict subset of the rows
    # every row is scored by a probe that did not see it: dropping a fold's rows from the fit changes nothing for it
    f0 = folds == 0
    Zb = Z.copy()
    Zb[f0] = 0.0                                                   # corrupt the held-out fold's features for the FIT only
    held_b, _, _ = VF.crossfit(Zb, Y, folds, Z, folds, K)
    assert held_b[:, f0] == pytest.approx(held[:, f0], rel=1e-9, abs=1e-9)
    assert not np.allclose(held_b[:, ~f0], held[:, ~f0])
    # a concept with a single class in the fit rows is reported as unfitted, not as a silent zero direction
    Y1 = Y.copy()
    Y1[0] = 1
    coef1, ok1, _ = VF.fit_directions(Z, Y1, K)
    assert not ok1[0] and ok1[1:].all() and coef1[0].tolist() == [0.0] * d
    held1, fu1, _ = VF.crossfit(Z, Y1, folds, Z, folds, K)
    assert fu1[0] == 0 and np.isnan(held1[0]).all()
    assert VF.auroc_safe(Y1[0], held1[0]) != VF.auroc_safe(Y1[0], held1[0])          # NaN, never a number
    # fewer fitting rows is a real arm, not a relabelling of the same fit
    small = VF.fit_directions(Z[:40], Y[:, :40], K)[0]
    assert not np.allclose(small, coef)
    assert VF.ARMS == ("expert_valid200", "report_valid200", "report_train_sub", "report_train_sub_known",
                       "report_train_full")
    assert VF.ARM_LABEL["expert_valid200"][0] == "radiologist"
    assert all(VF.ARM_LABEL[a][0] == "report-derived" for a in VF.ARMS[1:])


def test_validfit_aggregate_names_which_contrast_each_delta_isolates():
    VF = vf()

    def block(aurocs):
        return {"block": "m/chexpert", "arms": {arm: {"per_concept": [
            {"concept": "C", "fitted": True, "folds_used": 5, "median_fit_rows": rows,
             "auroc_expert_labels": a, "auroc_report_labels": a + 0.02, "cos_model": 0.1, "cos_whitened": 0.5}]}
            for arm, (a, rows) in aurocs.items()}}

    # the fourth arm fits as many rows as the expert refit; the second and third fit fewer, because the labeler leaves
    # films blank, so only the fourth is the size match
    b = block({"expert_valid200": (0.70, 160), "report_valid200": (0.68, 44), "report_train_sub": (0.69, 42),
               "report_train_sub_known": (0.71, 160), "report_train_full": (0.80, 5000)})
    A = VF.aggregate([b])
    assert A["per_arm"]["expert_valid200"]["label_source"] == "radiologist"
    assert A["per_arm"]["report_train_full"]["fit_sample"] == "every training row"
    assert A["size_matched_arm"] == "report_train_sub_known" and A["size_match_exact"] is True
    # label source with the estimation sample held fixed: 0.70 - 0.71
    assert A["label_source_at_matched_size"]["delta_auroc_expert_labels"] == pytest.approx(-0.01)
    # the same films, but fewer usable rows, so this one is NOT the size match: 0.70 - 0.68
    assert A["label_source_on_the_same_films"]["delta_auroc_expert_labels"] == pytest.approx(0.02)
    # sample size, holding the label source fixed: 0.80 - 0.71
    assert A["sample_size_at_one_label_source"]["delta_auroc_expert_labels"] == pytest.approx(0.09)
    # the comparison the paper made before the size-matched arms existed mixes the two: 0.70 - 0.80
    assert A["expert_refit_against_the_shipped_direction"]["delta_auroc_expert_labels"] == pytest.approx(-0.10)
    for k in VF.CONTRASTS:
        assert A[k]["description"]
    # a size match that is not exact is reported as such rather than quietly named the size match
    b2 = block({"expert_valid200": (0.70, 160), "report_valid200": (0.68, 44), "report_train_sub": (0.69, 42),
                "report_train_sub_known": (0.71, 120), "report_train_full": (0.80, 5000)})
    assert VF.aggregate([b2])["size_match_exact"] is False
