"""Round-5 control work on the ATTR module: ATTRQ (three phrasings of each attribute question, clean, on the
calibration rows, with a per-block primary phrasing selected by the campaign's own rule) and the attribute-versus-finding
ownership comparison reported three ways (all cells, answer-capable cells only, answerability-matched pairs)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from cftransfer import analysis as A, attr as AT, enqueue as E, fit as F, images as I, package as PK, protocol as P, \
    runner as RN, runpaths as R  # noqa: E402
from test_fit_synthetic import make_synthetic  # noqa: E402
from test_runner_fake import FakeAdapter  # noqa: E402

CONCEPTS = P.CONCEPTS["nih"]
N_EVAL = 40                      # calibration / test rows of the synthetic cohort; 12 is too few for the 10/10 rule
JSON = dict(default=lambda o: float(o) if isinstance(o, np.floating) else str(o))


def _cli(monkeypatch, model_key, dataset_id, what, n_boot=50):
    """Run the analysis CLI in process, so the --what dispatch and the printing are covered too."""
    import runpy
    monkeypatch.setattr(sys, "argv", ["analysis", "--model-key", model_key, "--dataset", dataset_id,
                                      "--what", what, "--n-boot", str(n_boot)])
    runpy.run_module("cftransfer.analysis", run_name="__main__")


def r2():
    """scripts/mayo/robustness_round2.py loaded by path (it is a script, not a package module)."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "mayo" / "robustness_round2.py"
    spec = importlib.util.spec_from_file_location("robustness_round2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------- synthetic block with phrasings
class PhrasingAdapter(FakeAdapter):
    """FakeAdapter whose yes/no margin is 100 + signal(attribute, phrasing) * (2*label - 1) + noise(row).

    The plain fake model answers every question from the image alone, so all three phrasings would score identically
    and the selection rule would have nothing to choose between. With `signal` set the margin is a deterministic
    function of the row's attribute label and the phrasing, so the expected AUROC ordering is known in advance."""

    def load(self, device_map="cpu", dtype=None):
        # the runner calls load() itself, so the arming survives it (FakeAdapter.fail_row does the same)
        keep = {k: getattr(self, k, d) for k, d in (("signal", None), ("row_noise", {}), ("attr_labels", {}),
                                                    ("question_key", {}))}
        super().load(device_map, dtype)
        self.__dict__.update(keep)          # signal: {(attribute, phrasing): strength}; None -> plain FakeAdapter
        self._delta = None
        return self

    def encode(self, images, questions):
        self._delta = None
        if self.signal is not None:
            key = self.question_key.get(questions[0])
            if key is not None:
                a, ph = key
                y = self.attr_labels[images[0]][a]
                self._delta = self.row_noise[images[0]] + self.signal.get((a, ph), 0.0) * (1.0 if y == 1 else -1.0)
        return super().encode(images, questions)

    @torch.no_grad()
    def forward_last_logits(self, enc):
        out = super().forward_last_logits(enc)
        if self._delta is not None:
            out = out.clone()
            out[:, 1] = 100.0 + self._delta            # positive candidate, always the strongest
            out[:, 2] = 0.0
            out[:, 3] = 0.0                            # margin = max(pos) - max(neg) = 100 + delta
        return out


def _setup(tmp_path, monkeypatch, n_eval=N_EVAL):
    data, runs = make_synthetic(tmp_path, n_eval=n_eval)
    monkeypatch.setattr(I, "DATA_ROOT", data)
    monkeypatch.setattr(R, "RUN_ROOT", runs)
    for mod in (I, RN):
        monkeypatch.setattr(mod, "image_path", lambda ds, row: row["row_id"])
        monkeypatch.setattr(mod, "open_rgb", lambda p: p)
    F.fit_locus("m", "nih", "vis.last", seeds=(0, 1, 2), write_scores=False)
    ad = PhrasingAdapter("m", "fake/m", "rev")
    monkeypatch.setattr(RN, "get_adapter", lambda key, rev=None: ad)
    monkeypatch.setattr(PK, "MODELS", {**PK.MODELS, "m": {"model_id": "fake/m"}})
    return ad, runs


def _arm(ad, signal):
    """Point the adapter at the calibration rows: deterministic per-row noise, the attribute labels and the exact
    rendered text of every (attribute, phrasing)."""
    rows = I.load_cohort("nih", ("calibration",))
    labels = I.load_labels("nih")
    c0 = CONCEPTS[0]
    ad.row_noise = {r["row_id"]: i / len(rows) for i, r in enumerate(rows)}
    ad.attr_labels = {r["row_id"]: {a: AT.attribute_value(a, labels[(r["row_id"], c0)]) for a in P.ATTR_CONCEPTS}
                      for r in rows}
    ad.question_key = {P.render_attrq("nih", a, ph): (a, ph) for a in P.ATTR_CONCEPTS for ph in P.ATTRQ_PHRASINGS}
    ad.signal = signal
    return rows


# ------------------------------------------------------------------------------------------- protocol / enqueue
def test_attrq_protocol_grid_and_prompts(tmp_path, monkeypatch):
    spec = P.MODULES["ATTRQ"]
    assert spec.role == "calibration" and spec.directions == "clean" and spec.alphas == (0.0,)
    assert spec.templates == P.ATTRQ_PHRASINGS == ("AQ0", "AQ1", "AQ2") and spec.datasets == ("nih", "chexpert")
    assert spec.baseline_module is None and spec.locus == "primary" and spec.fit_seeds == (0,)
    # the three attribute questions under three phrasings, one clean condition each
    assert P.question_list("nih", "ATTRQ") == [(a, ph) for ph in P.ATTRQ_PHRASINGS for a in P.ATTR_CONCEPTS]
    assert P.conditions_for("ATTRQ", "nih", "sex_F") == [("baseline", 0.0)]
    assert P.expected_rows("ATTRQ", "nih") == P.expected_rows("ATTRQ", "chexpert") == 3 * 3 * 400 == 3600
    assert P.expected_rows("ATTRQ", "coco") == 0
    # AQ0 IS the question ATTR writes, byte for byte, under whichever template the block is primary on
    for ds in ("nih", "chexpert"):
        for a in P.ATTR_CONCEPTS:
            assert P.render_attrq(ds, a, "AQ0") == P.render_question(ds, a, "IY")
            assert P.render_attrq(ds, a, "AQ0", "IB") == P.render_question(ds, a, "IB")
            for ph in P.ATTRQ_PHRASINGS:                      # same yes/no form, texts from protocol.json only
                q = P.render_attrq(ds, a, ph, "IY")
                assert q.endswith("Answer yes or no.") and P.IMAGE_PHRASE[ds] in q
                assert q == P.ATTRQ["questions"][a][ph].get("text", "{image_phrase}").format(image_phrase=P.IMAGE_PHRASE[ds]) \
                    or ph == "AQ0"
    assert len({P.render_attrq("nih", a, ph) for a in P.ATTR_CONCEPTS for ph in P.ATTRQ_PHRASINGS}) == 9
    # candidate set / preflight eligibility: AQ0 follows the primary template, the new yes/no phrasings follow IY
    assert P.attrq_source_template("AQ0", "IB") == "IB" and P.attrq_source_template("AQ1", "IB") == "IY"
    assert all(P.attrq_source_template(ph) == "IY" for ph in P.ATTRQ_PHRASINGS)
    # registration: addendum, added later on both chest sets, never in the default order, no prep file, one shard
    assert "ATTRQ" in P.ADDENDUM_MODULES and "ATTRQ" not in E.MODULE_ORDER
    assert all("ATTRQ" in P.MODULES_ADDED_LATER[d] for d in ("nih", "chexpert"))
    assert P.PROTOCOL["modules"]["ATTRQ"]["expected_rows"] == P.expected_rows("ATTRQ", "nih")
    assert E.SHARDS["ATTRQ"] == 1 and E.prep_files("ATTRQ") == []
    q = tmp_path / "queue"
    (q / "pending").mkdir(parents=True)
    monkeypatch.setattr(E, "QUEUE", q)
    names = E.enqueue("q25-7", "nih", ["ATTRQ"], prep=False, prefix="4")
    assert len(names) == 1 and names[0].endswith("-ATTRQ-q25-7-nih-000of001")
    t = json.loads((q / "pending" / f"{names[0]}.json").read_text())
    assert t["cmd"][:8] == ["python", "-m", "cftransfer.runner", "--model-key", "q25-7", "--dataset", "nih", "--module"]
    assert "ATTRQ" in t["cmd"] and not any(r.endswith("attr_seed0.npz") for r in t["requires"])
    assert E.enqueue("q25-7", "coco", ["ATTRQ"], prep=False, prefix="4") == []


# ------------------------------------------------------------------------------------------- runner + selection rule
def test_attrq_runner_and_phrasing_selection(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    rows = _arm(ad, {("view_AP", "AQ0"): 0.0, ("view_AP", "AQ1"): 0.25, ("view_AP", "AQ2"): 0.6,
                     ("sex_F", "AQ0"): 0.6, ("sex_F", "AQ1"): 0.25, ("sex_F", "AQ2"): 0.0,
                     ("age_60", "AQ0"): -1.0, ("age_60", "AQ1"): -1.0, ("age_60", "AQ2"): -1.0})
    rd = runs / "m" / "nih"
    meta = RN.run_block("m", "nih", "ATTRQ", 0, 1, batch=8, device_map="cpu", rows_per_part=10)
    assert meta["outcomes"] == len(rows) * 9 and meta["batch"] == 1          # clean module: one forward per question
    PK.build("m", "nih")
    df = pq.read_table(rd / "outcomes" / "ATTRQ.parquet").to_pandas()
    assert set(df.template_id) == set(P.ATTRQ_PHRASINGS) and set(df.concept) == set(P.ATTR_CONCEPTS)
    assert set(df.direction_id) == {"baseline"} and (df.alpha == 0).all() and set(df.role) == {"calibration"}

    res = A.attrq("m", "nih", draws=2000)
    assert res["n_rows"] == len(rows) and res["role"] == "calibration" and res["written_phrasing"] == "AQ0"
    for a in P.ATTR_CONCEPTS:
        blk = res["per_attribute"][a]
        assert set(blk["phrasings"]) == set(P.ATTRQ_PHRASINGS)
        for ph, c in blk["phrasings"].items():
            assert c["question"] == P.render_attrq("nih", a, ph) and c["candidate_source_template"] == "IY"
            assert c["n_scored_rows"] == len(rows) and c["eligible"] == c["answer_capable"]
    view, sex, age = (res["per_attribute"][a] for a in ("view_AP", "sex_F", "age_60"))
    # view_AP: chance under the written phrasing, better under AQ1, perfect under AQ2 -> AQ2 selected
    aur = {ph: view["phrasings"][ph]["answer_auroc"] for ph in P.ATTRQ_PHRASINGS}
    assert aur["AQ0"] < 0.5 < aur["AQ1"] < aur["AQ2"] == 1.0
    assert view["eligible_phrasings"] == ["AQ1", "AQ2"] and view["selected"] == "AQ2"
    assert view["selection_status"] == "selected" and view["selection_changes_written_phrasing"] is True
    assert view["selected_answer_auroc_lower95_one_sided"] == view["phrasings"]["AQ2"]["answer_auroc_lower95_one_sided"]
    assert max(view["phrasings"][ph]["answer_auroc_lower95_one_sided"] for ph in view["eligible_phrasings"]) \
        == view["selected_answer_auroc_lower95_one_sided"]
    # sex_F: the written phrasing is the strongest -> selected, and the selection changes nothing
    assert sex["selected"] == "AQ0" and sex["selection_changes_written_phrasing"] is False
    assert sex["phrasings"]["AQ0"]["answer_auroc"] == 1.0
    # age_60: every phrasing answers the wrong way round -> no phrasing clears the rule, none selected
    assert age["eligible_phrasings"] == [] and age["selected"] is None
    assert age["selection_status"] == "no_eligible_phrasing" and "no phrasing clears" in age["selection_reason"]
    assert all(c["answer_auroc"] < 0.5 for c in age["phrasings"].values())
    # the rule is the campaign's own answer-capable rule, on the calibration rows
    for a in P.ATTR_CONCEPTS:
        for c in res["per_attribute"][a]["phrasings"].values():
            assert c["answer_capable"] == bool(c["n_pos"] >= 10 and c["n_neg"] >= 10 and c["answer_valid_draws"] >= 1900
                                               and c.get("answer_auroc_lower95_one_sided", 0) > 0.5)
    # the CLI dispatch and its printing, which is what the watcher runs
    _cli(monkeypatch, "m", "nih", "attrq", n_boot=2000)
    stored = json.loads((rd / "summary.json").read_text())["attrq"]
    assert stored["written_phrasing"] == "AQ0" and stored["per_attribute"]["view_AP"]["selected"] == "AQ2"
    assert stored["per_attribute"]["age_60"]["selection_status"] == "no_eligible_phrasing"


def test_attrq_selection_prefers_the_higher_lower_bound_not_the_order(tmp_path, monkeypatch):
    """Two eligible phrasings, the later one stronger: the rule takes the higher lower bound. With an exact tie the
    earlier phrasing wins, so a tie never silently rewrites the question the module already writes."""
    ad, runs = _setup(tmp_path, monkeypatch)
    _arm(ad, {("view_AP", "AQ0"): 0.3, ("view_AP", "AQ1"): 0.4, ("view_AP", "AQ2"): 0.6,
              ("sex_F", "AQ0"): 0.6, ("sex_F", "AQ1"): 0.6, ("sex_F", "AQ2"): 0.6,
              ("age_60", "AQ0"): -1.0, ("age_60", "AQ1"): -1.0, ("age_60", "AQ2"): -1.0})
    RN.run_block("m", "nih", "ATTRQ", 0, 1, batch=8, device_map="cpu", rows_per_part=10)
    PK.build("m", "nih")
    res = A.attrq("m", "nih", draws=2000)
    view = res["per_attribute"]["view_AP"]
    lo = {ph: view["phrasings"][ph]["answer_auroc_lower95_one_sided"] for ph in view["eligible_phrasings"]}
    assert len(lo) == 3 and view["selected"] == max(lo, key=lo.get) == "AQ2"
    sex = res["per_attribute"]["sex_F"]                                  # identical signal -> exact tie
    assert len(sex["eligible_phrasings"]) == 3 and sex["selected"] == "AQ0"
    assert len({sex["phrasings"][ph]["answer_auroc_lower95_one_sided"] for ph in P.ATTRQ_PHRASINGS}) == 1


# ------------------------------------------------------------------- attr(): answerability per cell, per-draw ownership
def _attr_block(tmp_path, monkeypatch):
    """A synthetic block with the ATTR grid scored, ready for A.attr()."""
    ad, runs = _setup(tmp_path, monkeypatch)
    rd = runs / "m" / "nih"
    AT.build_attr("m", "nih", "vis.last", min_class_rows=5)
    RN.run_block("m", "nih", "CORE", 0, 1, batch=32, device_map="cpu", rows_per_part=10)
    RN.run_block("m", "nih", "ATTR", 0, 1, batch=32, device_map="cpu", rows_per_part=10)
    PK.build("m", "nih")
    return ad, runs, rd


def test_attr_answerability_sources_and_per_draw_ownership(tmp_path, monkeypatch):
    ad, runs, rd = _attr_block(tmp_path, monkeypatch)
    Q = P.ATTR_CONCEPTS + CONCEPTS
    bare = A.attr("m", "nih", draws=200)                       # no summary.json yet
    assert bare["answerability"]["attrq_available"] is False and bare["answerability"]["same_rows"] is False
    for q in Q:
        ans = bare["per_question"][q]["answerability"]
        if q in P.ATTR_CONCEPTS:
            assert ans["rows"] == "test" and "ATTRQ not scored" in ans["source"]
            assert ans["answer_auroc"] == bare["attributes"][q]["label_answer_auroc"]
            assert ans["n_pos"] == bare["attributes"][q]["label_n_pos"]
        else:
            assert ans["rows"] is None and "no calibration section" in ans["source"]
        # per-draw ownership: one bit per draw, and the point grade is untouched by it
        flags = A.unpack_draw_flags(bare["per_question"][q]["owned_draws_hex"], bare["draws"])
        assert flags.shape == (bare["draws"],) and abs(flags.mean() - bare["per_question"][q]["owned_draw_rate"]) < 1e-12
        if not bare["per_question"][q]["steering_reference"]:
            assert not flags.any()

    # with a summary: the clinical answer number is READ from `calibration`, the attribute one from `attrq`
    cal = {c: {"answer_auroc": 0.70 + 0.01 * i, "answer_auroc_lower95_one_sided": 0.60, "answer_capable": i % 2 == 0,
               "answer_brier": 0.2, "answer_valid_draws": 2000, "n_pos": 100, "n_neg": 90} for i, c in enumerate(CONCEPTS)}
    aq = {"written_phrasing": "AQ0", "n_rows": 40,
          "per_attribute": {a: {"phrasings": {"AQ0": {"answer_auroc": 0.55, "answer_auroc_lower95_one_sided": 0.48,
                                                      "answer_capable": False, "answer_brier": 0.3,
                                                      "answer_valid_draws": 2000, "n_pos": 20, "n_neg": 20}}}
                            for a in P.ATTR_CONCEPTS}}
    (rd / "summary.json").write_text(json.dumps({"calibration": cal, "attrq": aq}, **JSON))
    full = A.attr("m", "nih", draws=200)
    assert full["answerability"]["attrq_available"] is True and full["answerability"]["same_rows"] is True
    for i, c in enumerate(CONCEPTS):
        ans = full["per_question"][c]["answerability"]
        assert ans["answer_auroc"] == cal[c]["answer_auroc"] and ans["answer_capable"] == (i % 2 == 0)
        assert ans["rows"] == "calibration" and "summary.json calibration" in ans["source"]
    for a in P.ATTR_CONCEPTS:
        ans = full["per_question"][a]["answerability"]
        assert ans["answer_auroc"] == 0.55 and ans["answer_capable"] is False and ans["rows"] == "calibration"
        assert "attrq" in ans["source"] and "AQ0" in ans["source"]
    # the grade itself is unchanged: answerability is reported beside the ownership numbers, never inside them
    keys = ("W_qq", "O_q", "max_other", "argmax_other", "verdict", "steering_reference", "steering_reference_matched",
            "random_p95", "abs_sham", "rank_in_random_family", "O_q_ci95_percentile", "owned_draws_hex")
    for q in Q:
        assert {k: bare["per_question"][q].get(k) for k in keys} == {k: full["per_question"][q].get(k) for k in keys}
    assert bare["W"] == full["W"] and bare["contrasts"] == full["contrasts"]
    for a in P.ATTR_CONCEPTS:                                   # the ATTR attribute block is untouched as well
        assert bare["attributes"][a] == full["attributes"][a]


# ------------------------------------------------------------------------------- the three attribute-vs-finding modes
def _cell(owned, auroc, capable, draws=None, flags=None):
    c = {"owned": owned, "verdict": "fixed_family_advantage" if owned else "unresolved", "steering_reference": owned,
         "W_qq": 0.1, "O_q": 0.01, "max_other": 0.09, "argmax_other": "x", "random_reference": True, "abs_sham": 0.0,
         "answerability": {"answer_auroc": auroc, "answer_capable": capable, "n_pos": 50, "n_neg": 50,
                           "rows": "calibration", "answer_auroc_lower95_one_sided": None, "source": "test"}}
    if draws:
        f = np.full(draws, bool(owned)) if flags is None else np.asarray(flags, bool)
        c["owned_draws_hex"] = A.pack_draw_flags(f)
        c["owned_draw_rate"] = float(f.mean())
    return c


def _row(block, attrs, clins, draws=None):
    return {"block": block, "model": block.split("/")[0], "dataset": block.split("/")[1], "draws": draws,
            "attributes": attrs, "clinical": clins}


def test_three_comparison_modes_and_the_no_match_case():
    M = r2()
    # one block: 3 attributes, 6 findings. Attribute answer AUROCs 0.52 / 0.60 / 0.91; findings 0.55 .. 0.95.
    attrs = {"view_AP": _cell(False, 0.52, False), "sex_F": _cell(True, 0.61, True), "age_60": _cell(True, 0.91, True)}
    clins = {"Effusion": _cell(True, 0.55, False), "Atelectasis": _cell(True, 0.58, True),
             "Pneumothorax": _cell(False, 0.62, True), "Cardiomegaly": _cell(True, 0.90, True),
             "Mass": _cell(True, 0.93, True), "Nodule": _cell(False, 0.95, False)}
    row = _row("m/nih", attrs, clins)
    res = M.attr_comparison([row])
    assert res["window"] == P.ATTR_MATCH_WINDOW == 0.05

    a = res["all_cells"]
    assert (a["attr_cells"], a["attr_owned"]) == (3, 2) and (a["clin_cells"], a["clin_owned"]) == (6, 4)
    assert abs(a["difference"] - (2 / 3 - 4 / 6)) < 1e-12 and a["bootstrap"]["available"] is False

    k = res["answer_capable_cells"]                       # attributes 2 of 3 capable, findings 4 of 6
    assert (k["attr_cells"], k["attr_owned"]) == (2, 2) and (k["clin_cells"], k["clin_owned"]) == (4, 3)
    assert k["attr_cells_excluded"] == 1 and k["clin_cells_excluded"] == 2
    assert k["attr_cells_without_answer_number"] == 0 and k["clin_cells_without_answer_number"] == 0

    m = res["answerability_matched"]
    # view_AP 0.52 -> Effusion 0.55 (gap 0.03) only; sex_F 0.61 -> Atelectasis 0.58 and Pneumothorax 0.62;
    # age_60 0.91 -> Cardiomegaly 0.90, Mass 0.93 and Nodule 0.95 (gap 0.04). One attribute cell may take several partners.
    got = {(p["attribute"], p["clinical"]) for p in m["unmatched"]} if m["unmatched"] else set()
    assert got == set() and m["n_attr_cells_unmatched"] == 0 and m["n_attr_cells_matched"] == 3
    pairs = M.match_block(row)[0]
    assert {(p["attribute"], p["clinical"]) for p in pairs} == {
        ("view_AP", "Effusion"), ("sex_F", "Atelectasis"), ("sex_F", "Pneumothorax"),
        ("age_60", "Cardiomegaly"), ("age_60", "Mass"), ("age_60", "Nodule")}
    assert m["n_pairs"] == 6 and m["attr_cells"] == m["clin_cells"] == 6
    assert m["attr_owned"] == 0 + 1 + 1 + 1 + 1 + 1 and m["clin_owned"] == 1 + 1 + 0 + 1 + 1 + 0
    assert max(p["abs_answer_auroc_gap"] for p in pairs) <= P.ATTR_MATCH_WINDOW + 1e-12
    assert m["n_distinct_clinical_cells"] == 6

    # no-match case: an attribute far outside every finding's answerability, and one with no number at all
    far = {"view_AP": _cell(True, 0.20, False), "sex_F": _cell(True, None, None), "age_60": _cell(True, 0.62, True)}
    res2 = M.attr_comparison([_row("m/nih", far, clins)])
    m2 = res2["answerability_matched"]
    # age_60 at 0.62 reaches Atelectasis (0.58) and Pneumothorax (0.62); the other two attribute cells match nothing
    assert m2["n_attr_cells_unmatched"] == 2 and m2["n_attr_cells_matched"] == 1 and m2["n_pairs"] == 2
    u = {x["attribute"]: x for x in m2["unmatched"]}
    assert set(u) == {"view_AP", "sex_F"}
    assert u["view_AP"]["nearest_clinical"]["concept"] == "Effusion" and abs(u["view_AP"]["nearest_clinical"]["gap"] - 0.35) < 1e-9
    assert "within the window" in u["view_AP"]["reason"]
    assert u["sex_F"]["answer_auroc"] is None and "no clean-answer AUROC" in u["sex_F"]["reason"]
    # a cell with no answerability at all is excluded from the capable mode and counted
    assert res2["answer_capable_cells"]["attr_cells_without_answer_number"] == 1


def test_comparison_patient_bootstrap_over_shared_draws():
    M = r2()
    B = 64
    d = np.arange(B)
    # every attribute cell owned in every draw; three findings never, three in a third of the draws. The per-draw
    # difference is then 1.0 or 0.5 and its interval excludes zero -- the point of the interval is that ownership is
    # re-decided inside each draw, not that the cells are independent.
    attrs = {a: _cell(True, 0.7, True, draws=B, flags=np.ones(B, bool)) for a in P.ATTR_CONCEPTS}
    clins = {c: _cell(i >= 3, 0.7, True, draws=B, flags=(np.zeros(B, bool) if i < 3 else (d % 3 == 0)))
             for i, c in enumerate(CONCEPTS)}
    res = M.attr_comparison([_row("m/nih", attrs, clins, draws=B)])
    bt = res["all_cells"]["bootstrap"]
    assert bt["available"] is True and bt["draws"] == B
    assert bt["attr_owned_share_per_draw"] == 1.0 and bt["attr_owned_share_ci95"] == [1.0, 1.0]
    assert bt["difference_excludes_zero"] is True and bt["difference_ci95_percentile"] == [0.5, 1.0]
    assert abs(bt["difference_per_draw"] - (1.0 - 0.5 * np.mean(d % 3 == 0))) < 1e-12
    assert res["all_cells"]["attr_owned_share"] == 1.0 and res["all_cells"]["clin_owned_share"] == 0.5
    # one cell without per-draw ownership disables the interval and says so, leaving the point estimate in place
    attrs2 = dict(attrs)
    attrs2["sex_F"] = _cell(True, 0.7, True)
    res2 = M.attr_comparison([_row("m/nih", attrs2, clins, draws=B)])
    assert res2["all_cells"]["bootstrap"]["available"] is False
    assert "per-draw ownership" in res2["all_cells"]["bootstrap"]["reason"]
    assert res2["all_cells"]["attr_owned_share"] == res["all_cells"]["attr_owned_share"]
    # blocks that disagree on the draw count are never pooled into one interval
    res3 = M.attr_comparison([_row("m/nih", attrs, clins, draws=B), _row("m/chexpert", attrs, clins, draws=B + 1)])
    assert res3["draws"] is None and "disagree" in res3["draws_note"]
    assert res3["all_cells"]["bootstrap"]["available"] is False
    # the block bootstrap needs no per-draw ownership: it resamples the (model, dataset) blocks
    assert res3["all_cells"]["block_bootstrap"]["available"] is True
    assert res3["all_cells"]["block_bootstrap"]["n_blocks"] == 2


def test_block_bootstrap_widens_when_blocks_disagree():
    """One block where the attributes win and one where the findings do: the block interval must straddle zero even
    though each block on its own is decisive. This is the variation a patient bootstrap inside a block cannot see."""
    M = r2()
    a_win = _row("m/nih", {a: _cell(True, 0.7, True) for a in P.ATTR_CONCEPTS},
                 {c: _cell(False, 0.7, True) for c in CONCEPTS})
    c_win = _row("n/nih", {a: _cell(False, 0.7, True) for a in P.ATTR_CONCEPTS},
                 {c: _cell(True, 0.7, True) for c in CONCEPTS})
    one = M.attr_comparison([a_win])["all_cells"]["block_bootstrap"]
    assert one["available"] is True and one["difference_ci95_percentile"] == [1.0, 1.0] and one["n_blocks"] == 1
    both = M.attr_comparison([a_win, c_win])["all_cells"]
    assert both["difference"] == 0.0
    lo, hi = both["block_bootstrap"]["difference_ci95_percentile"]
    assert lo <= -0.9 and hi >= 0.9 and both["block_bootstrap"]["difference_excludes_zero"] is False
    assert both["block_bootstrap"]["seed"] == P.BOOT_CORE_SEED


def test_attr_section_carries_the_comparisons_and_tolerates_old_summaries(tmp_path):
    """The round-2 attr section over a synthetic run root: the per-block rows carry each cell's answer AUROC, the
    per-dataset aggregates carry all three comparisons, and a summary written before these keys existed still yields
    exactly the ownership counts it used to."""
    M = r2()
    root = tmp_path / "runs"

    def block(mk, ds, attrs, clins, answerability=True, draws=64):
        rd = root / mk / ds
        rd.mkdir(parents=True)
        (rd / "run.json").write_text(json.dumps({"completed_modules": ["CORE", "CALIBRATION", "ATTR"]}))
        per, attributes = {}, {}
        for q, c in list(attrs.items()) + list(clins.items()):
            cell = dict(c)
            cell.update({"question_kind": "attribute" if q in attrs else "clinical",
                         "own_direction": f"attr:{q}" if q in attrs else f"concept:{q}"})
            if not answerability:
                cell.pop("answerability", None); cell.pop("owned_draws_hex", None); cell.pop("owned_draw_rate", None)
            per[q] = cell
        for a in attrs:
            attributes[a] = {"auroc_real": 0.8, "selectivity": 0.2, "selectivity_lower95_one_sided": 0.1,
                             "answer_auroc": 0.6, "answer_auroc_lower95_one_sided": 0.5, "readable": True,
                             "readable_status": "readable", "answer_capable": True, "n_pos": 30, "n_neg": 30,
                             "n_train_pos": 200, "n_train_neg": 200,
                             "cos_model_to_clinical": {c: 0.05 for c in P.CONCEPTS[ds]}}
        (rd / "summary.json").write_text(json.dumps({"attr": {
            "n_rows": 600, "n_scored_rows_min": 600, "alpha": 0.25, "template_id": "IY", "draws": draws,
            "questions": list(P.ATTR_CONCEPTS) + list(P.CONCEPTS[ds]), "attributes": attributes,
            "per_question": per}}, **JSON))

    attrs = {a: _cell(i > 0, 0.60 + 0.02 * i, True, draws=64) for i, a in enumerate(P.ATTR_CONCEPTS)}
    clins = {c: _cell(i < 4, 0.60 + 0.01 * i, i < 5, draws=64) for i, c in enumerate(CONCEPTS)}
    block("m", "nih", attrs, clins)
    block("n", "nih", attrs, clins, answerability=False)
    blocks, _ = M.discover(root)
    sec = M.attr_section(blocks)
    assert [r["block"] for r in sec["blocks"]] == ["m/nih", "n/nih"] and sec["match_window"] == P.ATTR_MATCH_WINDOW
    with_ans, without = sec["blocks"]
    # the numbers this section already reported are identical with and without the new keys
    same = ("attr_cells", "attr_owned", "clin_cells", "clin_owned", "attr_readable", "attr_answer_capable",
            "median_abs_W_qq_attr", "median_abs_W_qq_clinical", "median_abs_cos_attr_clinical")
    assert {k: with_ans[k] for k in same} == {k: without[k] for k in same} == {
        "attr_cells": 3, "attr_owned": 2, "clin_cells": 6, "clin_owned": 4, "attr_readable": 3,
        "attr_answer_capable": 3, "median_abs_W_qq_attr": 0.1, "median_abs_W_qq_clinical": 0.1,
        "median_abs_cos_attr_clinical": 0.05}
    # per-block rows carry each cell's answer AUROC, so the matching is auditable from the file alone
    assert with_ans["attributes"]["sex_F"]["answerability"]["answer_auroc"] == 0.62
    assert with_ans["clinical"]["Effusion"]["answerability"]["answer_auroc"] == 0.60
    assert without["attributes"]["sex_F"]["answerability"] == {}
    assert with_ans["comparison"]["all_cells"]["bootstrap"]["available"] is True
    assert without["comparison"]["all_cells"]["bootstrap"]["available"] is False
    assert without["comparison"]["answerability_matched"]["n_attr_cells_unmatched"] == 3      # no AUROC to match on
    # the pooled aggregates carry all three modes; the block without the keys degrades but never breaks
    for g in ("nih", "chest", "all"):
        cmp = sec["per_dataset"][g]["comparison"]
        assert set(cmp) >= {"all_cells", "answer_capable_cells", "answerability_matched"}
        assert cmp["all_cells"]["attr_cells"] == 6 and cmp["all_cells"]["clin_cells"] == 12
        assert cmp["answerability_matched"]["n_attr_cells"] == 6
        assert cmp["all_cells"]["bootstrap"]["available"] is False       # n/nih carries no per-draw ownership
    assert sec["per_dataset"]["coco"]["blocks"] == 0
