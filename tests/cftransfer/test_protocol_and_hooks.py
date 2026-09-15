"""CPU tests that pin the protocol grid, the manifests, the steering rule and the scorer."""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from cftransfer import protocol as P                                   # noqa: E402
from cftransfer.hooks import LocusHook, TokenLayout                    # noqa: E402
from cftransfer.scoring import CandidateSet, score_logits              # noqa: E402


def test_expected_rows_match_protocol_table():
    assert P.expected_rows("CORE", "nih") == 457_200
    assert P.expected_rows("CALIBRATION", "nih") == 6_400
    assert P.expected_rows("CALIBRATION", "chexpert") == 2_400
    assert P.expected_rows("PROMPT", "coco") == 762_000
    assert P.expected_rows("DOSE", "nih") == 162_000
    assert P.expected_rows("REFIT", "coco") == 50_400
    assert P.expected_rows("LOCUS", "nih") + P.expected_rows("LOCUS_CALIBRATION", "nih") == 459_600
    # the planned campaign total covers the original seven modules; ALTDIR (added 2026-09-14) is a separate addendum
    per_model = sum(P.expected_rows(m, d) for d in P.DATASETS for m in P.MODULES if m != "ALTDIR")
    assert per_model * 22 == P.PROTOCOL["total_planned_outcomes"]["all_test_and_calibration_rows"]
    assert sum(P.expected_rows("ALTDIR", d) for d in P.DATASETS) * 22 == P.PROTOCOL["total_planned_outcomes"]["ALTDIR_rows_addendum"]
    core = sum(P.expected_rows("CORE", d) for d in P.DATASETS)
    assert core * 22 == P.PROTOCOL["total_planned_outcomes"]["primary_CORE_rows"]


def test_templates_render_exactly():
    assert P.render_question("nih", "Effusion", "IY") == "Is there a pleural effusion in this chest radiograph? Answer yes or no."
    assert P.render_question("coco", "person", "WB") == ("Does this photograph show a person? Answer B if the finding is present "
                                                        "and A if it is absent. Reply with A or B only.")
    assert P.render_question("chexpert", "Edema", "IA").startswith("Is there pulmonary edema in this chest radiograph?")


def test_condition_grid_shapes():
    core = P.conditions_for("CORE", "nih", "Mass")
    assert len(core) == 127 and core[0] == ("baseline", 0.0) and core[-1] == ("sham:Mass", 0.25)
    assert [d for d, _ in core[1:7]] == [f"concept:{c}" for c in P.CONCEPTS["nih"]]
    dose = P.conditions_for("DOSE", "coco", "dog")
    assert len(dose) == 5 * 27 and {a for _, a in dose} == {-0.5, -0.25, -0.1, 0.1, 0.5}
    assert "baseline" not in {d for d, _ in dose}
    assert len(P.conditions_for("REFIT", "nih", "Nodule")) == 7


def test_nih_cohorts_reproduce_from_manifest():
    rows = list(csv.DictReader((P.REPO / "data" / "manifest.csv").open()))
    H = lambda s: hashlib.sha256(s.encode()).hexdigest()
    by = {}
    for r in rows:
        by.setdefault(r["split"], {}).setdefault(r["patient_id"], []).append(r)
    def pick(split):
        chosen = {p: min(v, key=lambda r: H("cf-transfer-v1-row:" + r["row_id"])) for p, v in by[split].items()}
        return [chosen[p] for p in sorted(chosen, key=lambda p: H("cf-transfer-v1-patient:" + p))]
    val, test = pick("val"), pick("test")
    recon = [r["row_id"] for r in val[:416]] + [r["row_id"] for r in test[:600]]
    coh = [c["row_id"] for c in csv.DictReader((P.PKG / "cohorts.csv").open())]
    assert recon == coh


def test_reltoken_hook_math_flat_and_batched():
    torch.manual_seed(0)
    lin = torch.nn.Identity()
    D = 8
    h = torch.randn(6, D)                                        # two images of 3 tokens, flat
    v = torch.randn(2, D); v = v / v.norm(dim=1, keepdim=True)
    hook = LocusHook(lin, "t")
    with hook:
        hook.arm(TokenLayout(True, slices=[(0, 3), (3, 6)]), v, torch.tensor([0.25, -0.5]))
        out = lin(h.clone())
    exp = h.clone()
    exp[:3] += 0.25 * h[:3].norm(dim=1, keepdim=True) * v[0]
    exp[3:] += -0.5 * h[3:].norm(dim=1, keepdim=True) * v[1]
    assert torch.allclose(out, exp, atol=1e-6)
    assert hook.calls == 1 and abs(hook.stats["delta_norm_mean"][0] - 0.25 * float(h[:3].norm(dim=1).mean())) < 1e-5
    # batched layout with a CLS token that must not change
    hb = torch.randn(2, 4, D)
    mask = torch.tensor([False, True, True, True])
    with hook:
        hook.arm(TokenLayout(False, masks=[mask, mask]), v, torch.tensor([0.25, 0.25]), capture=True)
        outb = lin(hb.clone())
    assert torch.equal(outb[:, 0], hb[:, 0])
    assert torch.allclose(hook.pooled, hb[:, 1:].mean(dim=1), atol=1e-6)
    # passive (clean) pass returns the input untouched and still records norms
    with hook:
        hook.arm(TokenLayout(True, slices=[(0, 3), (3, 6)]))
        outc = lin(h.clone())
    assert torch.equal(outc, h) and hook.stats["delta_norm_mean"] == [0.0, 0.0]


def test_hook_refuses_mismatched_layout():
    lin = torch.nn.Identity()
    hook = LocusHook(lin, "t")
    with hook:
        hook.arm(TokenLayout(True, slices=[(0, 5)]))
        try:
            lin(torch.zeros(4, 3)); raised = False
        except RuntimeError:
            raised = True
    assert raised


def test_scorer_margins():
    cands = CandidateSet("IA", (5, 7), (9,), (5, 7), (9,), {})
    lg = torch.zeros(1, 12); lg[0, 5] = 1.0; lg[0, 7] = 3.0; lg[0, 9] = 2.0
    s = score_logits(lg, cands)
    assert abs(s.semantic_margin[0] - 1.0) < 1e-6 and abs(s.raw_ab_margin[0] - 1.0) < 1e-6
    assert abs(s.p_present[0] - 1 / (1 + np.exp(-1.0))) < 1e-9
    assert abs(s.lse_margin[0] - (np.logaddexp(1.0, 3.0) - 2.0)) < 1e-5
    assert abs(s.vocab_logsumexp[0] - torch.logsumexp(lg, -1).item()) < 1e-5
    cands_b = CandidateSet("IB", (9,), (5, 7), (5, 7), (9,), {})
    sb = score_logits(lg, cands_b)
    assert abs(sb.semantic_margin[0] + 1.0) < 1e-6 and abs(sb.raw_ab_margin[0] - 1.0) < 1e-6


def test_hook_vectorised_matches_loop():
    torch.manual_seed(1)
    lin = torch.nn.Identity()
    D, T, B = 8, 5, 3
    h = torch.randn(B * T, D)
    v = torch.randn(B, D); v = v / v.norm(dim=1, keepdim=True)
    al = torch.tensor([0.25, -0.1, 0.5])
    hook = LocusHook(lin, "t")
    with hook:
        hook.arm(TokenLayout(True, slices=[(i * T, (i + 1) * T) for i in range(B)]), v, al, capture=True)
        out = lin(h.clone())
    exp = h.clone().view(B, T, D)
    exp = exp + al[:, None, None] * exp.norm(dim=-1, keepdim=True) * v[:, None, :]
    assert torch.allclose(out, exp.view(B * T, D), atol=1e-6)
    assert torch.allclose(hook.pooled, h.view(B, T, D).mean(1), atol=1e-6)
    # unequal counts fall back to the loop and give the same rule
    h2 = torch.randn(7, D)
    with hook:
        hook.arm(TokenLayout(True, slices=[(0, 3), (3, 7)]), v[:2], al[:2])
        out2 = lin(h2.clone())
    e2 = h2.clone(); e2[:3] += 0.25 * h2[:3].norm(dim=1, keepdim=True) * v[0]; e2[3:] += -0.1 * h2[3:].norm(dim=1, keepdim=True) * v[1]
    assert torch.allclose(out2, e2, atol=1e-6)
