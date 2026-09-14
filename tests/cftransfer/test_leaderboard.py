"""Leaderboard grading applies the protocol rules to packaged summaries and renders one row per block."""
import json

from cftransfer import leaderboard


def _summary(readable, capable, owned, unresolved=()):
    cal = {c: {"readable": c in readable, "answer_capable": c in capable} for c in ("A", "B", "C")}
    core = {c: {"steering_reference": c in owned or c in unresolved,
                "verdict": "fixed_family_advantage" if c in owned else ("unresolved" if c in unresolved else "stronger_competitor")}
            for c in ("A", "B", "C")}
    return {"calibration": cal, "core": {"per_question": core}}


def test_grade_block_rules():
    row = leaderboard.grade_block(_summary({"A", "B"}, {"A"}, {"A"}, unresolved={"B"}),
                                  {"status": "COMPLETE", "completed_modules": ["CORE"], "gpu_hours": 1.5},
                                  {"IY": {"eligible": True}, "IB": {"eligible": False}})
    assert (row["readable"], row["answer_capable"], row["owned"]) == (2, 1, 1)
    assert row["owned_concepts"] == "A" and row["eligible_templates"] == "IY"


def test_ineligible_core_block(tmp_path):
    row = leaderboard.grade_block(_summary({"A"}, set(), set()), {"status": "COMPLETE", "completed_modules": ["CALIBRATION"],
                                                                   "ineligible_modules": ["CORE"]}, None)
    assert row["owned"] is None and row["readable"] == 1
    d = tmp_path / "q25-7" / "nih"; d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(_summary({"A"}, {"A"}, {"A"})))
    (d / "run.json").write_text(json.dumps({"status": "COMPLETE", "completed_modules": ["CORE"], "gpu_hours": 2}))
    out = leaderboard.main(tmp_path)
    assert out.exists() and "q25-7" in (tmp_path / "leaderboard.md").read_text()
