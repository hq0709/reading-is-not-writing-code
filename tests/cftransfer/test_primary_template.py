"""Primary template (IY unless IY failed preflight check E and IB is eligible) threaded through question_list, the
runner, the package and the analysis gating; and the CheXpert extension of DOSE/REFIT/LOCUS/LOCUS_CALIBRATION/PROMPT."""
import csv
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from cftransfer import analysis as A, package as PK, protocol as P, runner as RN, runpaths as R, tables as T   # noqa: E402
from test_runner_fake import _setup                                                                             # noqa: E402


def _elig(d: Path, **flags):
    d.mkdir(parents=True, exist_ok=True)
    (d / "template_eligibility.json").write_text(json.dumps(
        {t: {"eligible": ok, "reason": "" if ok else "semantic mapping failed image-free preflight (check E)"} for t, ok in flags.items()}))


def test_primary_template_resolution(tmp_path):
    assert P.PRIMARY_TEMPLATE_FALLBACK == ("IY", "IB")
    assert P.primary_template(tmp_path / "absent") == "IY"                       # no preflight yet: IY
    _elig(tmp_path / "a", IY=False, WY=False, IA=False, IB=True, WA=False, WB=True)
    assert P.primary_template(tmp_path / "a") == "IB"                             # the llama32-11 pattern
    _elig(tmp_path / "b", IY=True, IB=False)
    assert P.primary_template(tmp_path / "b") == "IY"
    _elig(tmp_path / "c", IY=False, WY=False, IA=False, IB=False, WA=False, WB=False)
    assert P.primary_template(tmp_path / "c") == "IY"                             # nothing eligible: IY, the runner closes the module
    _elig(tmp_path / "d", WY=False)                                               # a template absent from the file counts as eligible
    assert P.primary_template(tmp_path / "d") == "IY"


def test_question_list_substitutes_primary_for_iy():
    assert P.question_list("nih", "CORE", "IB") == [(c, "IB") for c in P.CONCEPTS["nih"]]
    assert P.question_list("nih", "CORE") == [(c, "IY") for c in P.CONCEPTS["nih"]]
    cal = P.question_list("nih", "CALIBRATION", "IB")
    assert cal[:6] == [(c, "IB") for c in P.CONCEPTS["nih"]]
    assert cal[6:] == [(c, t) for t in ("WY", "IA", "IB", "WA", "WB") for c in ("Effusion", "Mass")]    # extras untouched
    assert cal.count(("Effusion", "IB")) == 2                                                          # scored twice, deduplicated on merge
    for m in ("DOSE", "REFIT", "LOCUS", "LOCUS_CALIBRATION"):
        assert {t for _c, t in P.question_list("coco", m, "IB")} == {"IB"}
    assert P.question_list("nih", "PROMPT", "IB") == P.question_list("nih", "PROMPT")                  # PROMPT never substituted
    for m in P.MODULES:
        for d in P.DATASETS:
            assert len(P.question_list(d, m, "IB")) == len(P.question_list(d, m))


def test_chexpert_modules_requested():
    for m in ("PROMPT", "DOSE", "REFIT", "LOCUS", "LOCUS_CALIBRATION"):
        assert "chexpert" in P.MODULES[m].datasets
        assert P.expected_rows(m, "chexpert") == P.expected_rows(m, "nih") > 0
    assert P.expected_rows("CALIBRATION", "chexpert") == 2_400                    # unchanged: IY-only, no prompt-concept extras
    assert P.question_list("chexpert", "CALIBRATION") == [(c, "IY") for c in P.CONCEPTS["chexpert"]]
    assert P.CALIBRATION_PROMPT_DATASETS == ("nih", "coco")
    assert P.PROMPT_CONCEPTS["chexpert"] == ["Effusion", "Edema"]
    assert P.question_list("chexpert", "PROMPT") == [(c, t) for t in ("WY", "IA", "IB", "WA", "WB") for c in ("Effusion", "Edema")]
    # ALTDIR (2026-09-14) is added-later on every dataset and outside the planned campaign total
    assert set(P.MODULES_ADDED_LATER["chexpert"]) == {"PROMPT", "DOSE", "REFIT", "LOCUS", "LOCUS_CALIBRATION", "ALTDIR"}
    assert all("ALTDIR" in P.MODULES_ADDED_LATER[d] for d in P.DATASETS)
    per_model = sum(P.expected_rows(m, d) for d in P.DATASETS for m in P.MODULES if m != "ALTDIR")
    assert per_model * 22 == P.PROTOCOL["total_planned_outcomes"]["all_test_and_calibration_rows"] == 125_153_600


def test_requested_modules_keep_old_chexpert_blocks_complete(tmp_path):
    rd = tmp_path / "m" / "chexpert"; (rd / "outcomes").mkdir(parents=True)
    assert PK.requested_modules("chexpert", rd) == ["CORE", "CALIBRATION"]        # packaged before the extension: unchanged
    (rd / "outcomes" / "DOSE").mkdir()
    assert PK.requested_modules("chexpert", rd) == ["CORE", "CALIBRATION", "DOSE"]  # started: now required
    planned = [m for m in P.MODULES if m != "ALTDIR"]                              # ALTDIR is added-later everywhere
    assert PK.requested_modules("nih", tmp_path / "m" / "nih") == planned
    (tmp_path / "m" / "nih" / "outcomes" / "ALTDIR").mkdir(parents=True)
    assert PK.requested_modules("nih", tmp_path / "m" / "nih") == list(P.MODULES)   # started: now required


def test_prompt_contrast_gating():
    ok = {t: {"eligible": True} for t in P.TEMPLATE_ORDER}
    assert A.prompt_contrast_ineligible("IY", ok, "IY", "WY") is None
    assert A.prompt_contrast_ineligible("IY", {}, "IA", "IB") is None
    assert "not IY" in A.prompt_contrast_ineligible("IB", ok, "IY", "WY")
    bad = {**ok, "IB": {"eligible": False}}
    assert "IB" in A.prompt_contrast_ineligible("IY", bad, "IA", "IB")
    assert A.prompt_contrast_ineligible("IY", bad, "IY", "WY") is None


def test_runner_scores_primary_template(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    _elig(runs / "m" / "nih", IY=False, WY=False, IA=False, IB=True, WA=False, WB=True)
    meta = RN.run_block("m", "nih", "CORE", 0, 1, batch=16, device_map="cpu", rows_per_part=1, limit_rows=1)
    assert meta["primary_template"] == "IB" and meta["outcomes"] == 6 * 127
    df = pq.read_table(runs / "m" / "nih" / "outcomes" / "CORE" / "part-000of001-00000.parquet").to_pandas()
    assert set(df.template_id) == {"IB"} and set(df.concept) == set(P.CONCEPTS["nih"])
    meta = RN.run_block("m", "nih", "CALIBRATION", 0, 1, batch=8, device_map="cpu", rows_per_part=1, limit_rows=1)
    df = pq.read_table(runs / "m" / "nih" / "outcomes" / "CALIBRATION" / "part-000of001-00000.parquet").to_pandas()
    assert meta["outcomes"] == 10 and set(df.template_id) == {"IB", "WB"}            # 6 x IB, Effusion/Mass IB again, Effusion/Mass WB
    assert len(df[(df.concept == "Effusion") & (df.template_id == "IB")]) == 2
    (runs / "m" / "nih" / "processing_settings.json").write_text(json.dumps(ad.processing_settings()))
    run = PK.build("m", "nih")
    assert run["primary_template"] == "IB" and run["ineligible_modules"] == []
    assert run["requested_modules"] == [m for m in P.MODULES if m != "ALTDIR"]         # ALTDIR not started: not required
    cov = list(csv.DictReader((runs / "m" / "nih" / "coverage.csv").open()))
    core_rows = [r for r in cov if r["module"] == "CORE"]
    assert core_rows and {r["template_id"] for r in core_rows} == {"IB"}
    assert all(r["execution_status"] == "RUNNING" for r in core_rows)                 # 1 of 600 rows scored: pending work, not ineligible
    pj = json.loads((runs / "m" / "nih" / "prompts.json").read_text())
    assert pj["nih|Effusion|IB"]["is_primary"] and not pj["nih|Effusion|IY"]["is_primary"]
    assert "Answer B if the finding is present" in pj["nih|Effusion|IB"]["rendered_prompt"]
    merged = pq.read_table(runs / "m" / "nih" / "outcomes" / "CALIBRATION.parquet").to_pandas()
    assert len(merged) == 8                                                            # duplicate (Effusion/Mass, IB) rows collapsed
    (runs / "m" / "nih" / "summary.json").write_text(json.dumps({"calibration": {}}))
    PK.build("m", "nih")
    assert json.loads((runs / "m" / "nih" / "summary.json").read_text())["primary_template"] == "IB"


def test_runner_closes_module_when_nothing_eligible(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    _elig(runs / "m" / "nih", IY=False, WY=False, IA=False, IB=False, WA=False, WB=False)
    meta = RN.run_block("m", "nih", "CORE", 0, 1, batch=16, device_map="cpu", rows_per_part=1, limit_rows=1)
    assert meta["scored_rows"] == 0 and meta["primary_template"] == "IY" and meta["ineligible_templates"] == ["IY"]
    assert not list((runs / "m" / "nih" / "outcomes" / "CORE").glob("part-*.parquet"))
    run = PK.build("m", "nih")
    assert "CORE" in run["ineligible_modules"]


def test_tables_mark_prompt_contrasts_ineligible_on_non_iy_primary(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RUN_ROOT", tmp_path); monkeypatch.setattr(T, "RUN_ROOT", tmp_path)
    d = tmp_path / "q25-7" / "nih"; d.mkdir(parents=True)
    cal = {c: {"selectivity": 0.1, "selectivity_ci95": [0.0, 0.2], "readable_status": "readable", "readable": True, "answer_capable": False}
           for c in P.CONCEPTS["nih"]}
    t3 = {"Effusion|wording_IY_minus_WY_O": {"estimate": None, "ci_low": None, "ci_high": None, "status": "INELIGIBLE", "reason": "x"},
          "all|median_label_gap": {"estimate": 0.5, "ci_low": 0.4, "ci_high": 0.6, "status": "COMPLETE"}}
    (d / "summary.json").write_text(json.dumps({"calibration": cal, "t3": t3, "primary_template": "IB"}))
    rows = {(r["concept"], r["metric"]): r for r in csv.DictReader(T.fill().open()) if r["model_key"] == "q25-7" and r["dataset_id"] == "nih"}
    assert rows[("Effusion", "wording_IY_minus_WY_O")]["execution_status"] == "INELIGIBLE" and rows[("Effusion", "wording_IY_minus_WY_O")]["estimate"] == ""
    assert rows[("Mass", "mapping_IA_minus_IB_O")]["execution_status"] == "INELIGIBLE"      # absent key, non-IY primary
    assert rows[("all", "median_label_gap")]["execution_status"] == "COMPLETE" and rows[("all", "median_label_gap")]["estimate"] == "0.5000"
    assert rows[("all", "median_refit_O_sd")]["execution_status"] == "NOT_STARTED"
    assert rows[("Effusion", "calibration_selectivity")]["execution_status"] == "COMPLETE"
