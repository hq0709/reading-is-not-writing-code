"""EXTCOMP, TOKENW and PRECISION: protocol grids, the per-token weighting in the hook, the extcomp prep, the runner's
steered-only / per-setting batches, the numerics-compatible package merge and the three analysis functions."""
import csv
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from cftransfer import analysis as A, enqueue as E, extcomp as EX, fit as F, package as PK, protocol as P, runner as RN  # noqa: E402
from cftransfer.hooks import MODE_SOFTMAX, MODE_TOPQ, MODE_UNIFORM, LocusHook, TokenLayout, token_weights            # noqa: E402
from test_runner_fake import _setup                                                                                    # noqa: E402

CONCEPTS = P.CONCEPTS["nih"]


def test_protocol_grids_ext():
    ext = P.conditions_for("EXTCOMP", "nih", "Mass")
    assert [d for d, _ in ext] == [f"extra:{k}" for k in ("Consolidation", "Edema", "Infiltration")] and {a for _, a in ext} == {0.25}
    assert len(P.conditions_for("EXTCOMP", "chexpert", "Edema")) == 6 and P.direction_kind("extra:Lung Opacity") == "extra"
    assert P.expected_rows("EXTCOMP", "nih") == 10_800 and P.expected_rows("EXTCOMP", "chexpert") == 21_600 and P.expected_rows("EXTCOMP", "coco") == 0
    tw = P.conditions_for("TOKENW", "coco", "dog")
    assert [d for d, _ in tw] == [f"{v}:{c}" for v in ("tokenw", "topq") for c in P.CONCEPTS["coco"]] and "baseline" not in {d for d, _ in tw}
    assert {P.direction_kind(d) for d, _ in tw} == {"tokenw", "topq"} and all(P.expected_rows("TOKENW", d) == 43_200 for d in P.DATASETS)
    assert P.conditions_for("PRECISION", "nih", "Mass") == P.conditions_for("CORE", "nih", "Mass")
    assert P.MODULES["PRECISION"].row_limit == 200 and P.MODULES["PRECISION"].baseline_module is None
    assert P.expected_rows("PRECISION", "nih") == 2 * 127 * 6 * 200 == 304_800 and P.expected_rows("PRECISION", "chexpert") == 0
    assert P.MODULE_SETTINGS == {"PRECISION": ("fp32", "batch1")} and P.NUMERICS_DEFAULT == "bf16-batched"
    for m, spec in (("EXTCOMP", ("nih", "chexpert")), ("TOKENW", P.DATASETS), ("PRECISION", ("nih", "coco"))):
        assert P.MODULES[m].datasets == tuple(spec) and E.SHARDS[m] == 1 and m not in E.MODULE_ORDER
        assert all(m in P.MODULES_ADDED_LATER[d] for d in spec)
    assert P.MODULES["EXTCOMP"].baseline_module == P.MODULES["TOKENW"].baseline_module == "CORE"
    assert "Pleural Other" in P.EXTCOMP_EXCLUDED["chexpert"] and "No Finding" in P.EXTCOMP_EXCLUDED["chexpert"]
    assert "numerics" in RN.KEY and RN.SCHEMA.field("numerics").type == pa.string()


def test_token_weights_math_and_hook():
    torch.manual_seed(0)
    B, T, D = 3, 8, 6
    scores = torch.randn(B, T)
    modes = torch.tensor([MODE_UNIFORM, MODE_SOFTMAX, MODE_TOPQ])
    w = token_weights(scores, modes)
    assert torch.allclose(w.mean(dim=1), torch.ones(B), atol=1e-6)                     # mean 1 in every mode
    assert torch.equal(w[0], torch.ones(T))
    assert torch.allclose(w[1], torch.softmax(scores[1], 0) * T, atol=1e-6)
    k = 2                                                                             # round(0.25 * 8)
    top = scores[2].topk(k).indices
    assert torch.allclose(w[2][top], torch.full((k,), T / k)) and float(w[2].sum()) == T and (w[2] == 0).sum() == T - k
    assert token_weights(torch.randn(1, 3), torch.tensor([MODE_TOPQ]))[0].max() == 3.0   # k = max(1, round(0.75)) = 1
    # hook: vectorised (flat, equal counts) and loop (unequal counts) paths match the formula
    lin = torch.nn.Identity()
    h = torch.randn(B * T, D)
    v = torch.randn(B, D); v = v / v.norm(dim=1, keepdim=True)
    sc = torch.randn(B, D)
    al = torch.tensor([0.25, 0.25, -0.5])
    hook = LocusHook(lin, "t")
    with hook:
        hook.arm(TokenLayout(True, slices=[(i * T, (i + 1) * T) for i in range(B)]), v, al, scorers=sc, modes=modes)
        out = lin(h.clone())
    hb = h.view(B, T, D)
    wt = token_weights(torch.einsum("btd,bd->bt", hb, sc), modes)
    exp = hb + (al[:, None] * hb.norm(dim=-1) * wt)[:, :, None] * v[:, None, :]
    assert torch.allclose(out, exp.view(B * T, D), atol=1e-6)
    assert hook.stats["token_weight_max"][0] == 1.0 and hook.stats["token_weight_max"][2] == T / k
    h2 = torch.randn(T + (T - 1) + T, D)
    sl = [(0, T), (T, 2 * T - 1), (2 * T - 1, 3 * T - 1)]
    with hook:
        hook.arm(TokenLayout(True, slices=sl), v, al, scorers=sc, modes=modes)
        out2 = lin(h2.clone())
    for b, (s0, e0) in enumerate(sl):
        hbb = h2[s0:e0]
        wb = token_weights((hbb @ sc[b]).unsqueeze(0), modes[b:b + 1])[0]
        assert torch.allclose(out2[s0:e0], hbb + (al[b] * hbb.norm(dim=-1) * wb)[:, None] * v[b][None, :], atol=1e-6)
    with pytest.raises(ValueError):
        hook.arm(TokenLayout(True, slices=sl), v, al, scorers=sc[:2], modes=modes)


def test_extcomp_build_and_runner(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    rd = runs / "m" / "nih"
    with pytest.raises(FileNotFoundError, match="cftransfer.extcomp"):
        RN.run_block("m", "nih", "EXTCOMP", 0, 1, batch=8, device_map="cpu", rows_per_part=2, limit_rows=2)
    assert not (rd / "outcomes" / "EXTCOMP").exists()
    with pytest.raises(RuntimeError, match="support rule"):
        EX.build_extcomp("m", "nih", "vis.last", out_dir=tmp_path / "x", min_support=100)   # 60 synthetic rows
    path, arr = EX.build_extcomp("m", "nih", "vis.last", min_support=5)
    assert path == rd / "fits" / "vis.last" / "extcomp_seed0.npz" and path.exists()
    names = list(arr["extra_names"].astype(str))
    assert names == P.EXTCOMP_LABELS["nih"] and arr["extra_vectors"].shape == (3, 16) and arr["cos_model"].shape == (3, 6)
    assert np.allclose(np.linalg.norm(arr["extra_vectors"], axis=1), 1, atol=1e-5)
    # the extra directions are the protocol fit applied to the extra labels
    f0 = F.load_fit("m", "nih", "vis.last", 0)
    ids, X, _v, roles = F.load_features("m", "nih", "vis.last")
    tr = roles == "train"
    Zs = ((X.astype(np.float32) @ f0["projection"] - f0["scaler_mean"]) / np.maximum(f0["scaler_scale"], 1e-8)).astype(np.float32)
    Y = EX.label_matrix("nih", ids[tr], names)
    for ki in range(3):
        w = F.fit_lr(Zs[tr][Y[ki] >= 0], Y[ki][Y[ki] >= 0]).coef_[0]
        assert np.allclose(arr["extra_coefficients"][ki], w, atol=1e-6)
        assert np.allclose(arr["extra_vectors"][ki], F.direction_from_projected(f0["projection"], f0["scaler_scale"], w), atol=1e-6)
        assert np.allclose(arr["cos_model"][ki], arr["extra_vectors"][ki] @ f0["clinical_vectors"].T, atol=1e-6)
    assert np.isfinite(arr["auroc_calibration"]).all() and int(arr["n_pos"][0] + arr["n_neg"][0]) == 60
    meta = RN.run_block("m", "nih", "EXTCOMP", 0, 1, batch=8, device_map="cpu", rows_per_part=2, limit_rows=2)
    assert meta["outcomes"] == 2 * 6 * 3 and meta["numerics"] == "bf16-batched"
    df = pq.read_table(next((rd / "outcomes" / "EXTCOMP").glob("part-*.parquet"))).to_pandas()
    assert set(df.direction_kind) == {"extra"} and "baseline" not in set(df.direction_id) and (df.delta_norm_mean > 0).all()
    assert set(df.numerics) == {"bf16-batched"} and (df.sample_status == "OK").all()
    for (rid, q), g in df.groupby(["row_id", "concept"]):
        assert sorted(g.direction_id) == sorted(d for d, _ in P.conditions_for("EXTCOMP", "nih", q))


def test_runner_tokenw(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    rd = runs / "m" / "nih"
    meta = RN.run_block("m", "nih", "TOKENW", 0, 1, batch=8, device_map="cpu", rows_per_part=2, limit_rows=2)
    assert meta["outcomes"] == 2 * 6 * 12
    df = pq.read_table(next((rd / "outcomes" / "TOKENW").glob("part-*.parquet"))).to_pandas()
    assert set(df.direction_kind) == {"tokenw", "topq"} and "baseline" not in set(df.direction_id) and (df.delta_norm_mean > 0).all()
    for (rid, q), g in df.groupby(["row_id", "concept"]):
        assert sorted(g.direction_id) == sorted(d for d, _ in P.conditions_for("TOKENW", "nih", q))
        m = g.set_index("direction_id").semantic_margin
        assert abs(float(m[f"tokenw:{q}"]) - float(m[f"topq:{q}"])) > 1e-9          # the two weightings differ
    # the scorer is the token's projected, scaled probe logit up to a constant
    bank = RN.DirectionBank("m", "nih", "vis.last", (0,), tokenw=True)
    f0 = F.load_fit("m", "nih", "vis.last", 0)
    sc = bank.scorer("tokenw:Mass", 0)
    ci = CONCEPTS.index("Mass")
    assert np.allclose(sc, f0["projection"] @ (f0["coefficients"][ci] / np.maximum(f0["scaler_scale"], 1e-8)), atol=1e-6)
    assert np.allclose(bank.vector("topq:Mass", 0), f0["clinical_vectors"][ci])


def test_runner_precision_settings_and_package(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    rd = runs / "m" / "nih"
    seen = []
    orig = ad.load
    ad.load = lambda device_map="cpu", dtype=None: (seen.append(dtype), orig(device_map, dtype))[1]
    with pytest.raises(ValueError, match="--numerics"):
        RN.run_block("m", "nih", "PRECISION", 0, 1, batch=16, device_map="cpu", rows_per_part=2, limit_rows=2)
    with pytest.raises(ValueError):
        RN.run_block("m", "nih", "CORE", 0, 1, batch=16, device_map="cpu", rows_per_part=2, limit_rows=2, numerics="fp32")
    m32 = RN.run_block("m", "nih", "PRECISION", 0, 1, batch=16, device_map="cpu", rows_per_part=2, limit_rows=2, numerics="fp32")
    assert seen[-1] == torch.float32 and m32["numerics"] == "fp32" and m32["batch"] == 16 and m32["outcomes"] == 2 * 6 * 127
    m1 = RN.run_block("m", "nih", "PRECISION", 0, 1, batch=16, device_map="cpu", rows_per_part=2, limit_rows=2, numerics="batch1")
    assert seen[-1] == torch.bfloat16 and m1["numerics"] == "batch1" and m1["batch"] == 1 and m1["outcomes"] == 2 * 6 * 127
    parts = sorted(p.name for p in (rd / "outcomes" / "PRECISION").glob("part-*.parquet"))
    assert parts == ["part-batch1-000of001-00000.parquet", "part-fp32-000of001-00000.parquet"]
    metas = sorted(p.name.rsplit("-", 1)[0] for p in (rd / "outcomes" / "PRECISION").glob("meta-*.json"))
    assert metas == ["meta-batch1-000of001", "meta-fp32-000of001"]
    for st in ("fp32", "batch1"):
        df = pq.read_table(rd / "outcomes" / "PRECISION" / f"part-{st}-000of001-00000.parquet").to_pandas()
        assert set(df.numerics) == {st} and (df.direction_id == "baseline").sum() == 2 * 6
        assert sorted(df[df.row_id == df.row_id.iloc[0]][df.concept == "Mass"].direction_id) == sorted(d for d, _ in P.conditions_for("CORE", "nih", "Mass"))
    # resume per setting: nothing left for fp32, the rows of batch1 are independent
    assert RN.run_block("m", "nih", "PRECISION", 0, 1, batch=16, device_map="cpu", rows_per_part=2, limit_rows=2, numerics="fp32")["todo"] == 0
    # an old part file without the numerics column (pre-2026-09-14 schema) still merges, with the default setting
    old_schema = pa.schema([f for f in RN.SCHEMA if f.name != "numerics"])
    row = {f.name: None for f in old_schema}
    row.update(protocol_id=P.PROTOCOL_ID, run_id=F.run_id_for("m", "nih"), model_key="m", dataset_id="nih", module="CORE", role="test",
               row_id="tes_000", unit_id="test0", concept="Mass", template_id="IY", locus_id="vis.last", fit_seed=0,
               direction_id="baseline", direction_kind="baseline", alpha=0.0, positive_token_ids=[1, 2], negative_token_ids=[3],
               sample_status="OK", error_reason="")
    (rd / "outcomes" / "CORE").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([row], schema=old_schema), rd / "outcomes" / "CORE" / "part-000of001-00000.parquet")
    run = PK.build("m", "nih")
    core = pq.read_table(rd / "outcomes" / "CORE.parquet").to_pandas()
    assert list(core.numerics) == ["bf16-batched"] and list(core.columns) == RN.SCHEMA.names
    assert run["merged_outcomes"]["PRECISION"]["rows_unique"] == 2 * 2 * 6 * 127          # both settings kept (numerics in KEY)
    cov = [r for r in csv.DictReader((rd / "coverage.csv").open()) if r["module"] == "PRECISION"]
    assert len(cov) == 6 and all(r["expected_rows"] == str(2 * 127 * 200) and r["execution_status"] == "RUNNING" for r in cov)
    assert all("fp32 254/25400 ok, batch1 254/25400 ok" in r["reason"] for r in cov)
    assert "PRECISION" in run["requested_modules"] and "EXTCOMP" not in run["requested_modules"]


def test_analysis_ext_modules(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    rd = runs / "m" / "nih"
    EX.build_extcomp("m", "nih", "vis.last", min_support=5)
    RN.run_block("m", "nih", "CORE", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    RN.run_block("m", "nih", "EXTCOMP", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    RN.run_block("m", "nih", "TOKENW", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    for st in P.PRECISION_SETTINGS:
        RN.run_block("m", "nih", "PRECISION", 0, 1, batch=32, device_map="cpu", rows_per_part=4, numerics=st)
    PK.build("m", "nih")
    core = A.core("m", "nih", n_boot=50)
    alt = pq.read_table(rd / "outcomes" / "CORE.parquet").to_pandas()
    base = alt[alt.direction_id == "baseline"].set_index(["concept", "row_id"]).p_present
    # ---- EXTCOMP: extended family of 5 + 3 competitors, own write and O_core from CORE
    ex = A.extcomp("m", "nih", draws=100)
    ext_df = pq.read_table(rd / "outcomes" / "EXTCOMP.parquet").to_pandas()
    assert ex["extra_directions"] == P.EXTCOMP_LABELS["nih"] and ex["n_competitors"] == 8 and len(ex["contrasts"]) == 6 * 8
    for q in CONCEPTS:
        cell = ex["per_question"][q]
        assert abs(cell["W_qq"] - core["per_question"][q]["W_qq"]) < 1e-9 and abs(cell["O_core_q"] - core["per_question"][q]["O_q"]) < 1e-9
        assert set(cell["W_extra"]) == set(P.EXTCOMP_LABELS["nih"]) and cell["n_competitors"] == 8
        g = ext_df[(ext_df.concept == q) & (ext_df.direction_id == "extra:Edema")]
        assert abs(cell["W_extra"]["Edema"] - float(np.mean(g.p_present.values - base.loc[[(q, r) for r in g.row_id]].values))) < 1e-9
        assert cell["O_ext_q"] == cell["O_q"] <= cell["O_core_q"] + 1e-12          # a larger family can only lower O
        assert cell["n_extra_beating_own"] == sum(v > cell["W_qq"] for v in cell["W_extra"].values())
        assert cell["verdict"] in ("stronger_competitor", "fixed_family_advantage", "unresolved") and len(cell["O_q_ci95_percentile"]) == 2
        assert set(ex["extra_info"]["Edema"]["cos_model_to_protocol"]) == set(CONCEPTS)
    # ---- TOKENW: two variants with the cross reference
    tw = A.tokenw("m", "nih", draws=100)
    tw_df = pq.read_table(rd / "outcomes" / "TOKENW.parquet").to_pandas()
    assert tw["variants"] == ["tokenw", "topq"]
    for v in tw["variants"]:
        assert len(tw[v]["contrasts"]) == 30 and tw[v]["n_scored_rows_min"] == 12
        for q in CONCEPTS:
            cell = tw[v]["per_question"][q]
            g = tw_df[(tw_df.concept == q) & (tw_df.direction_id == f"{v}:{q}")]
            assert abs(cell["W_qq"] - float(np.mean(g.p_present.values - base.loc[[(q, r) for r in g.row_id]].values))) < 1e-9
            assert abs(cell["own_minus_max_logistic_competitor"] - (cell["W_qq"] - core["per_question"][q]["max_other_clinical"])) < 1e-9
            assert isinstance(cell["steering_reference"], bool) and cell["verdict"] in ("stronger_competitor", "fixed_family_advantage", "unresolved")
    # ---- PRECISION: the fake model is dtype- and batch-independent, so both settings reproduce CORE on the same rows
    pr = A.precision("m", "nih")
    assert pr["settings"] == ["fp32", "batch1"] and pr["n_rows"] == 12
    for st in pr["settings"]:
        r = pr[st]
        assert r["status"] == "COMPLETE" and r["max_abs_dW"] < 1e-5 and r["max_abs_dO"] < 1e-5
        assert r["n_agree_competitor"] == 6 and r["n_agree_random_p95"] == 6
        for q in CONCEPTS:
            c = r["per_question"][q]
            assert abs(c["W_qq"] - core["per_question"][q]["W_qq"]) < 1e-5 and c["random_n"] == 119 and set(c["W"]) == set(CONCEPTS)
    assert abs(pr["core"]["per_question"]["Mass"]["O_q"] - core["per_question"]["Mass"]["O_q"]) < 1e-9
