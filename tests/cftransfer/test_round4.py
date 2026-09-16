"""Round-4 modules: TOWERSWAP (one reader, the other checkpoint's vision tower), REPLAY (byte-identical consumed-block
tensors into two readers) and SEMEND (three semantic endpoints outside the six templates)."""
import csv
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from cftransfer import analysis as A, ansdir as AN, enqueue as E, fit as F, manifest as M, package as PK, protocol as P, \
    replay as RP, runner as RN, semend as SM, towerswap as TS  # noqa: E402
from cftransfer.adapters.base import LocusInfo  # noqa: E402
from cftransfer.hooks import TokenLayout  # noqa: E402
from test_runner_fake import _setup  # noqa: E402

CONCEPTS = P.CONCEPTS["nih"]
JSON = dict(default=lambda o: float(o) if isinstance(o, np.floating) else str(o))


def _cli(monkeypatch, model_key, dataset_id, what, n_boot=50):
    """Run the analysis CLI in process, so the --what dispatch and the printing are covered too."""
    import runpy
    monkeypatch.setattr(sys, "argv", ["analysis", "--model-key", model_key, "--dataset", dataset_id,
                                      "--what", what, "--n-boot", str(n_boot)])
    runpy.run_module("cftransfer.analysis", run_name="__main__")


# --------------------------------------------------------------------------------------------- protocol / enqueue
def test_round4_protocol_grids(tmp_path, monkeypatch):
    # TOWERSWAP / REPLAY: the CORE grid with its own baseline on the first 200 test rows
    for mod in ("TOWERSWAP", "REPLAY"):
        spec = P.MODULES[mod]
        assert spec.role == "test" and spec.row_limit == 200 and spec.directions == "core" and spec.baseline_module is None
        assert spec.fit_seeds == (0,) and spec.locus == "primary" and spec.templates == ("IY",)
        assert P.conditions_for(mod, "nih", "Mass") == P.conditions_for("CORE", "nih", "Mass")
        assert all(P.expected_rows(mod, d) == 6 * 127 * 200 == 152_400 for d in P.DATASETS)
        assert mod in P.ADDENDUM_MODULES and all(mod in P.MODULES_ADDED_LATER[d] for d in P.DATASETS)
        assert mod not in E.MODULE_ORDER and E.SHARDS[mod] == 1
    # the pair / group tables the two modules are defined on
    assert P.TOWERSWAP_PAIRS == {"gemma3-4": "medgemma-4", "medgemma-4": "gemma3-4",
                                 "gemma3-27": "medgemma-27", "medgemma-27": "gemma3-27"}
    assert all(P.TOWERSWAP_PAIRS[P.TOWERSWAP_PAIRS[k]] == k for k in P.TOWERSWAP_PAIRS)
    assert set(P.REPLAY_SOURCE.values()) == {"gemma3-4", "medgemma-4", "llava15-7"}
    assert all(P.REPLAY_SOURCE[v] == v for v in set(P.REPLAY_SOURCE.values()))          # every source replays itself
    # SEMEND: four endpoints, 25 conditions, own baseline per (question, endpoint)
    sp = P.MODULES["SEMEND"]
    assert sp.templates == P.SEMEND_TEMPLATES == ("NY", "DA", "DB", "RF") and sp.row_limit is None and sp.baseline_module is None
    cc = P.conditions_for("SEMEND", "nih", "Mass")
    assert [d for d, _ in cc] == (["baseline"] + [f"concept:{c}" for c in CONCEPTS] + [f"ans:{c}" for c in CONCEPTS]
                                  + ["sham:Mass", "anssham:Mass"] + [f"random:{i:03d}" for i in range(18)])
    assert {a for _d, a in cc[1:]} == {0.25} and cc[0] == ("baseline", 0.0) and len(cc) == 33
    # 32 steered conditions fill exactly one batch-32 forward, so the full two-family grid costs no extra GPU time
    assert len(cc) - 1 == 32 and P.SEMEND_N_RANDOM == 18
    assert P.SEMEND_SIGN == {"NY": -1.0, "DA": 1.0, "DB": 1.0, "RF": 1.0}
    assert P.question_list("nih", "SEMEND") == [(c, t) for t in P.SEMEND_TEMPLATES for c in CONCEPTS]
    assert P.question_list("nih", "SEMEND", "IB") == P.question_list("nih", "SEMEND")     # endpoints are never substituted
    assert all(P.expected_rows("SEMEND", d) == 6 * 4 * 33 * 600 == 475_200 for d in P.DATASETS)
    assert "SEMEND" in P.ADDENDUM_MODULES and E.SHARDS["SEMEND"] == 4 and E.PREP_FILE["SEMEND"] == "semend_seed0.npz"
    # prompts come from protocol.json, never from the runner
    assert P.render_semend("nih", "Mass", "NY") == "Is there no evidence of a lung mass in this chest radiograph? Answer yes or no."
    assert P.render_semend("nih", "Mass", "DA", "Nodule").startswith("Which is present in this chest radiograph, a lung mass or a lung nodule?")
    assert P.render_semend("nih", "Mass", "DB", "Nodule").startswith("Which is present in this chest radiograph, a lung nodule or a lung mass?")
    assert P.render_semend("nih", "Mass", "RF").endswith("\nFindings:") and P.render_semend("coco", "dog", "RF").endswith("\nObjects:")
    assert P.semend_finding_word("nih", "Mass") == "mass" and P.semend_finding_word("coco", "bottle") == "bottle"
    with pytest.raises(ValueError, match="strongest competitor"):
        P.render_semend("nih", "Mass", "DA")
    with pytest.raises(KeyError):
        P.render_semend("nih", "Mass", "IY")
    assert set(P.TEMPLATES) == set(P.TEMPLATE_ORDER)                                     # the six frozen templates untouched


def test_round4_enqueue(tmp_path, monkeypatch):
    q = tmp_path / "queue"
    for sub in ("pending", "running", "done", "failed"):
        (q / sub).mkdir(parents=True)
    monkeypatch.setattr(E, "QUEUE", q)

    def task(name):
        return json.loads((q / "pending" / f"{name}.json").read_text())
    # SEMEND: a CPU prep that waits for the answer directions and the merged CORE parquet, then three shards
    names = E.enqueue("q25-7", "nih", ["SEMEND"], prep=False, prefix="4")
    assert len(names) == 5 and names[0].endswith("-SEMEND-prep-q25-7-nih") and names[-1].endswith("-SEMEND-q25-7-nih-003of004")
    prep = task(names[0])
    assert prep["cmd"][:3] == ["python", "-m", "cftransfer.semend"] and "--device-map" not in prep["cmd"]
    assert any(r.endswith("ansdir_seed0.npz") for r in prep["requires"]) and any(r.endswith("outcomes/CORE.parquet") for r in prep["requires"])
    assert prep["produces"] == [r for r in task(names[1])["requires"] if r.endswith("semend_seed0.npz")]
    # TOWERSWAP: one shard, and the PARTNER block's seed-0 fit is a hard dependency
    names = E.enqueue("gemma3-4", "nih", ["TOWERSWAP"], prep=False, prefix="4")
    assert len(names) == 1 and names[0].endswith("-TOWERSWAP-gemma3-4-nih-000of001")
    req = task(names[0])["requires"]
    assert any(r.endswith("/medgemma-4/nih/fits/vis.last/seed0.npz") for r in req)
    with pytest.raises(SystemExit, match="TOWERSWAP is only defined"):
        E.enqueue("q25-7", "nih", ["TOWERSWAP"], prep=False, prefix="4")
    # REPLAY: one shared tensor task per SOURCE block, named identically however the group member enqueues it
    n7 = E.enqueue("llava15-7", "nih", ["REPLAY"], prep=False, prefix="4")
    n13 = E.enqueue("llava15-13", "nih", ["REPLAY"], prep=False, prefix="4")
    tensors = [n for n in n7 if "REPLAY-tensors" in n]
    assert len(tensors) == 1 and tensors == [n for n in n13 if "REPLAY-tensors" in n]     # one task for the pair
    t = task(tensors[0])
    assert t["cmd"][:6] == ["python", "-m", "cftransfer.replay", "--model-key", "llava15-7", "--dataset"]
    assert t["produces"] == [str(RP.replay_dir("llava15-7", "nih") / "vis.last.npz")]
    req13 = task([n for n in n13 if n.endswith("REPLAY-llava15-13-nih-000of001")][0])["requires"]
    assert t["produces"][0] in req13 and any(r.endswith("/llava15-7/nih/fits/vis.last/seed0.npz") for r in req13)
    with pytest.raises(SystemExit, match="REPLAY is only defined"):
        E.enqueue("q25-7", "nih", ["REPLAY"], prep=False, prefix="4")


# ------------------------------------------------------------------------------------------- the weight swap itself
class _TinyModel(torch.nn.Module):
    """model.vision_tower.encoder.layers.{0,1} + post_layernorm, then the reader (projector + language model)."""

    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.model = torch.nn.Module()
        self.model.vision_tower = torch.nn.Module()
        self.model.vision_tower.encoder = torch.nn.Module()
        self.model.vision_tower.encoder.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])
        self.model.vision_tower.post_layernorm = torch.nn.LayerNorm(4)
        self.model.multi_modal_projector = torch.nn.Linear(4, 6)
        self.model.language_model = torch.nn.Linear(6, 6)


class _TinyAdapter:
    family = "tiny"

    def __init__(self, model, key="host", mid="fake/host", rev="r0"):
        self.model, self.model_key, self.model_id, self.revision = model, key, mid, rev
        self._modules = dict(model.named_modules())

    def module(self, path):
        return self._modules[path]

    def loci(self):
        return {"vis.last": LocusInfo("vis.last", "model.vision_tower.encoder.layers.1", "t", 4, "all", "m", "none", "p")}


def _tower_state(model):
    pref = "model.vision_tower."
    return {k[len(pref):]: v.detach().clone()
            for k, v in list(model.named_parameters()) + list(model.named_buffers()) if k.startswith(pref)}


def test_tower_swap_replaces_only_the_tower(tmp_path, monkeypatch):
    host, donor = _TinyModel(0), _TinyModel(1)
    ad = _TinyAdapter(host)
    assert TS.tower_root(ad) == "model.vision_tower"
    donor_t = _tower_state(donor)
    host_t = _tower_state(host)
    before_reader = {k: v.detach().clone() for k, v in host.named_parameters() if "vision_tower" not in k}
    n_differ = sum(1 for k, v in donor_t.items() if not torch.equal(v, host_t[k]))
    assert 0 < n_differ <= len(donor_t)
    monkeypatch.setitem(TS.MODELS, "donor", {"model_id": "fake/donor"})
    r = TS.apply_tower_swap(ad, "donor", revision="r1", donor_tensors=donor_t, donor_shas={k: "sha" for k in donor_t})
    assert r["tower_root"] == "model.vision_tower" and r["n_tensors_replaced"] == len(donor_t)
    assert r["verified_bitwise_equal_to_donor"] and r["verified_rest_unchanged"] and r["donor_model_key"] == "donor"
    assert r["n_tower_tensors_changed"] == n_differ and r["n_tensors_outside_tower_changed"] == 0
    assert r["n_tower_tensors"] == len(donor_t)
    # every replaced tensor IS the donor's, bit for bit; nothing outside the tower moved
    for k, v in _tower_state(host).items():
        assert torch.equal(v, donor_t[k])
    for k, v in host.named_parameters():
        if "vision_tower" not in k:
            assert torch.equal(v.detach(), before_reader[k])
    # a missing or mis-shaped donor tensor is refused, not silently ignored
    with pytest.raises(RuntimeError, match="without a donor tensor"):
        TS.apply_tower_swap(_TinyAdapter(_TinyModel(2)), "donor", donor_tensors={k: v for k, v in list(donor_t.items())[1:]})
    bad = dict(donor_t); k0 = sorted(bad)[0]; bad[k0] = torch.zeros(9)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        TS.apply_tower_swap(_TinyAdapter(_TinyModel(3)), "donor", donor_tensors=bad)
    # a family whose consumed block is not <tower>.encoder.layers.<k> is refused rather than guessed
    class _Odd(_TinyAdapter):
        def loci(self):
            return {"vis.last": LocusInfo("vis.last", "model.multi_modal_projector", "t", 4, "all", "m", "none", "p")}
    with pytest.raises(NotImplementedError, match="encoder layer"):
        TS.tower_root(_Odd(_TinyModel(4)))


def test_tower_key_normalisation_and_checkpoint_reading(tmp_path, monkeypatch):
    assert TS.normalise_tower_key("vision_tower.encoder.layers.0.mlp.fc1.weight") == "encoder.layers.0.mlp.fc1.weight"
    assert TS.normalise_tower_key("model.vision_tower.encoder.layers.0.mlp.fc1.weight") == "encoder.layers.0.mlp.fc1.weight"
    assert TS.normalise_tower_key("vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight") == "encoder.layers.0.mlp.fc1.weight"
    assert TS.normalise_tower_key("language_model.model.layers.0.mlp.fc1.weight") is None
    assert TS.normalise_tower_key("multi_modal_projector.linear_1.weight") is None
    # a staged snapshot read straight from safetensors, with the LLaVA spelling of the tower
    from safetensors.torch import save_file
    snap = tmp_path / "snap"; snap.mkdir()
    donor = _TinyModel(7)
    state = {f"vision_tower.vision_model.{k}": v.detach().clone() for k, v in _tower_state(donor).items()}
    state["language_model.model.layers.0.weight"] = torch.zeros(3, 3)
    save_file(state, str(snap / "model.safetensors"))
    (snap / "config.json").write_text("{}")
    monkeypatch.setattr(TS, "snapshot_dir", lambda key, revision=None: snap)
    idx = TS.safetensors_index(snap)
    assert len(idx) == len(state) and set(idx) == set(state)
    tensors, shas = TS.donor_tower_tensors("donor")
    assert set(tensors) == set(_tower_state(donor)) and len(shas) == len(tensors)
    assert all(torch.equal(tensors[k], _tower_state(donor)[k]) for k in tensors)
    host = _TinyModel(8)
    TS.apply_tower_swap(_TinyAdapter(host), "donor", donor_tensors=tensors, donor_shas=shas)
    assert all(torch.equal(v, tensors[k]) for k, v in _tower_state(host).items())


# ----------------------------------------------------------------------------------------- TOWERSWAP through the runner
def _second_block(runs: Path, key: str = "m2", dataset_id: str = "nih", seed: int = 11):
    """A partner block with its own features and its own seed-0 fit (the tower whose directions are swapped in)."""
    src = np.load(runs / "m" / dataset_id / "features" / "vis.last.npz")
    feat = runs / key / dataset_id / "features"
    feat.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    x = (src["x"].astype(np.float32) + 0.5 * rng.standard_normal(src["x"].shape)).astype(np.float16)
    for lid in ("vis.last", "connector"):
        np.savez(feat / f"{lid}.npz", row_id=src["row_id"], x=x, valid_token_count=src["valid_token_count"],
                 fit_role=src["fit_role"])
    F.fit_locus(key, dataset_id, "vis.last", seeds=(0,), write_scores=False)


def _stub_swap(gain: float = 1.05):
    """Stand-in for the weight swap on the fake family: changes the tower's output, records the same receipt shape."""
    def apply(ad, donor_key, revision=None):
        ad.model.vis.forward = lambda x, g=gain: x * g
        return {"host_model_key": ad.model_key, "donor_model_key": donor_key, "tower_root": "vis",
                "n_tensors_replaced": 1, "n_params_replaced": 1, "verified_bitwise_equal_to_donor": True,
                "verified_rest_unchanged": True, "n_tensors_outside_tower_changed": 0}
    return apply


def test_towerswap_runner_and_analysis(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    _second_block(runs)
    monkeypatch.setattr(PK, "MODELS", {**PK.MODELS, "m": {"model_id": "fake/m"}, "m2": {"model_id": "fake/m2"}})
    monkeypatch.setitem(P.TOWERSWAP_PAIRS, "m", "m2")
    monkeypatch.setitem(P.TOWERSWAP_PAIRS, "m2", "m")
    monkeypatch.setitem(P.MODULES, "TOWERSWAP", replace(P.MODULES["TOWERSWAP"], row_limit=12))
    monkeypatch.setattr(RN, "apply_tower_swap", _stub_swap())
    # the crossed arm: this block's reader, the partner's tower, the PARTNER's seed-0 directions
    meta = RN.run_block("m", "nih", "TOWERSWAP", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    assert meta["outcomes"] == 12 * 6 * 127 and meta["direction_block"] == "m2"
    assert meta["tower_swap"]["donor_model_key"] == "m2" and meta["replay"] is None
    rd = runs / "m" / "nih"
    receipts = sorted((rd / "outcomes" / "TOWERSWAP").glob("swap-*.json"))
    assert len(receipts) == 1 and json.loads(receipts[0].read_text())["donor_model_key"] == "m2"
    df = pq.read_table(next((rd / "outcomes" / "TOWERSWAP").glob("part-*.parquet"))).to_pandas()
    for (_rid, q), g in df.groupby(["row_id", "concept"]):
        assert sorted(g.direction_id) == sorted(d for d, _ in P.conditions_for("TOWERSWAP", "nih", q))
    assert set(df.fit_seed) == {0} and set(df.template_id) == {"IY"} and set(df.sample_status) == {"OK"}
    # the four combinations: both blocks' CORE (native) and both blocks' TOWERSWAP (crossed)
    for key in ("m", "m2"):
        RN.run_block(key, "nih", "CORE", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    RN.run_block("m2", "nih", "TOWERSWAP", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    for key in ("m", "m2"):
        PK.build(key, "nih")
    res = A.towerswap("m", "nih", draws=80)
    assert res["reader"] == "m" and res["partner"] == "m2" and res["n_rows"] == 12 and res["crossover_complete"]
    assert sorted(res["combinations"]) == sorted(["tower=m|reader=m", "tower=m2|reader=m", "tower=m2|reader=m2", "tower=m|reader=m2"])
    for name, c in res["combinations"].items():
        assert c["status"] == "COMPLETE" and c["directions_from"] == c["tower"]
        assert set(c["grade"]) == set(CONCEPTS) and set(c["W"]) == set(CONCEPTS)
        for q in CONCEPTS:
            cell = c["grade"][q]
            assert cell["verdict"] in ("stronger_competitor", "fixed_family_advantage", "unresolved")
            assert cell["owned"] == (cell["steering_reference"] and cell["verdict"] == "fixed_family_advantage")
    # the crossed arm really is a different write from the native one (different tower AND different directions)
    native, crossed = res["combinations"]["tower=m|reader=m"], res["combinations"]["tower=m2|reader=m"]
    assert any(abs(native["W"][q][f"concept:{q}"] - crossed["W"][q][f"concept:{q}"]) > 1e-9 for q in CONCEPTS)
    sm = res["swapped_minus_native"]
    for q in CONCEPTS:
        assert abs(sm[q]["estimate"] - (crossed["grade"][q]["W_qq"] - native["grade"][q]["W_qq"])) < 1e-9
        assert len(sm[q]["ci95_percentile"]) == 2 and isinstance(sm[q]["owned_changed"], bool)
    x = res["crossover"]
    assert set(x["ownership_by_combination"]) == set(res["combinations"]) and set(x["ownership_follows"]) == set(CONCEPTS)
    assert x["mean_abs_reader_effect"] >= 0 and x["mean_abs_tower_effect"] >= 0
    for k in ("reader_effect_at_own_tower", "tower_effect_at_own_reader"):
        assert x[k]["n_simultaneously_nonzero"] <= 6 and "max_t_critical" in x[k]
    # manifest: TOWERSWAP rows carry the crossed grade; block_included never depends on the module
    cg = A.core("m", "nih", n_boot=80)
    (rd / "summary.json").write_text(json.dumps({"core": cg, "towerswap": res}, **JSON))
    rj = json.loads((rd / "run.json").read_text())
    rj["completed_modules"] = ["CORE", "CALIBRATION", "TOWERSWAP"]
    (rd / "run.json").write_text(json.dumps(rj))
    rows = M.block_rows(rd)
    tr = [r for r in rows if r["module"] == "TOWERSWAP" and r["execution_status"] == "COMPLETE"]
    assert len(tr) == 6 and all(r["block_included"] and not r["cell"] for r in tr)
    for r in tr:
        g = crossed["grade"][r["concept"]]
        assert (r["towerswap_tower"], r["towerswap_reader"], r["towerswap_owned"], r["towerswap_verdict"]) == \
            ("m2", "m", g["owned"], g["verdict"])
    assert all(r["towerswap_owned"] is None for r in rows if r["cell"])
    rj["completed_modules"] = ["TOWERSWAP"]
    (rd / "run.json").write_text(json.dumps(rj))
    assert not any(r["block_included"] for r in M.block_rows(rd))
    M.write_csv(rows, tmp_path / "manifest.csv")
    header = (tmp_path / "manifest.csv").read_text().splitlines()[0].split(",")
    assert all(f"towerswap_{k}" in header for k in M.TOWERSWAP_FIELDS)
    _cli(monkeypatch, "m", "nih", "towerswap")
    assert "towerswap" in json.loads((rd / "summary.json").read_text())


# -------------------------------------------------------------------------------------------------------- REPLAY
def _fake_vision_features(ad):
    """The fake family has no separate tower pass; the capture hook fires on the same module either way."""
    ad.load(device_map="cpu")

    def vf(enc):
        return ad.model(enc["feats"])
    ad.vision_features = vf
    return ad


def test_replay_capture_and_hook(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    _fake_vision_features(ad)
    monkeypatch.setitem(P.REPLAY_SOURCE, "m", "m")
    path, meta = RP.build_replay("m", "nih", "vis.last", n_rows=12, batch_size=1, device_map="cpu", adapter=ad)
    assert path == runs / "m" / "nih" / "replay" / "vis.last.npz" and meta["rows"] == 12 and meta["shape"] == [12, 9, 16]
    assert meta["stored_dtype"] in ("float32", "float16", "bfloat16") and len(meta["sha256"]) == 64 and meta["capture_batch_size"] == 1
    bank = RP.ReplayBank.open("m", "nih", "vis.last")
    rows = RN.load_cohort("nih", ("test",))[:12]
    assert set(bank.tensors) == {r["row_id"] for r in rows} and bank.source_key == "m"
    # bit-preserving round trip through the npz
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        t = torch.randn(3, 5).to(dt)
        arr, name = RP.pack(t)
        assert torch.equal(RP.unpack(arr, name), t)
    # the hook overwrites the consumed block: replaying row A's tensor into row B's forward gives row A's logits
    ra, rb = rows[0]["row_id"], rows[1]["row_id"]
    enc_b = ad.expand(ad.encode([rb], ["q"]), 4)
    clean_b = ad.forward_last_logits(enc_b).clone()
    enc_a = ad.expand(ad.encode([ra], ["q"]), 4)
    clean_a = ad.forward_last_logits(enc_a).clone()
    assert not torch.allclose(clean_a, clean_b)
    hook = bank.hook(ad.module("vis"))
    with hook:
        hook.set_row(rb)
        same = ad.forward_last_logits(enc_b)
        assert torch.allclose(same, clean_b, atol=1e-5) and hook.calls == 1
        assert hook.drift[rb]["max_abs"] < 1e-6                       # this reader's own output IS the stored tensor
        hook.set_row(ra)
        swapped = ad.forward_last_logits(enc_b)
    assert torch.allclose(swapped, clean_a, atol=1e-5) and not torch.allclose(swapped, clean_b)
    assert hook.drift[ra]["max_abs"] > 0                              # measured against row b's own block output
    with pytest.raises(KeyError, match="no stored tensor"):
        hook.set_row("not-a-row")


def test_replay_module_and_analysis(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    _fake_vision_features(ad)
    monkeypatch.setitem(P.REPLAY_SOURCE, "m", "m")
    monkeypatch.setitem(P.MODULES, "REPLAY", replace(P.MODULES["REPLAY"], row_limit=12))
    rd = runs / "m" / "nih"
    with pytest.raises(FileNotFoundError, match="cftransfer.replay"):
        RN.run_block("m", "nih", "REPLAY", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    RP.build_replay("m", "nih", "vis.last", n_rows=12, batch_size=1, device_map="cpu", adapter=ad)
    meta = RN.run_block("m", "nih", "REPLAY", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    assert meta["outcomes"] == 12 * 6 * 127 and meta["direction_block"] == "m" and meta["tower_swap"] is None
    assert meta["replay"]["rows_replayed"] == 12 and meta["replay"]["drift_max_abs"] < 1e-5
    RN.run_block("m", "nih", "CORE", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    PK.build("m", "nih")
    # replaying this reader's OWN stored tensors reproduces its CORE grid exactly: the replacement is faithful
    rep = pq.read_table(rd / "outcomes" / "REPLAY.parquet").to_pandas().set_index(["row_id", "concept", "direction_id"])
    core = pq.read_table(rd / "outcomes" / "CORE.parquet").to_pandas().set_index(["row_id", "concept", "direction_id"])
    shared = rep.index.intersection(core.index)
    assert len(shared) == 12 * 6 * 127
    assert np.allclose(rep.loc[shared].p_present.values, core.loc[shared].p_present.values, atol=1e-6)
    # hook ORDER: the replacement runs BEFORE the steering hook, so the write still reaches the logits. If the order
    # were reversed the replayed tensor would overwrite the delta and every steered condition would equal the baseline.
    flat = pq.read_table(rd / "outcomes" / "REPLAY.parquet").to_pandas()
    one = flat[(flat.row_id == flat.row_id.iloc[0]) & (flat.concept == "Mass")]
    base = float(one[one.direction_id == "baseline"].p_present.iloc[0])
    assert (one[one.direction_id != "baseline"].p_present != base).all()
    assert (one[one.direction_id != "baseline"].delta_norm_mean > 0).all()
    res = A.replay("m", "nih", draws=80)
    assert res["reader"] == "m" and res["source_block"] == "m" and res["n_rows"] == 12 and res["group"] == ["m"]
    assert res["drift"]["rows_replayed"] == 12 and res["drift"]["max_abs"] < 1e-5
    for q in CONCEPTS:
        assert abs(res["replay_minus_core"][q]["estimate"]) < 1e-6
        assert res["replay"]["grade"][q]["verdict"] == res["core"]["grade"][q]["verdict"]
        assert res["replay_minus_core"][q]["owned_changed"] is False
    assert res["group_comparison"] == {}
    # manifest columns
    cg = A.core("m", "nih", n_boot=80)
    (rd / "summary.json").write_text(json.dumps({"core": cg, "replay": res}, **JSON))
    rj = json.loads((rd / "run.json").read_text())
    rj["completed_modules"] = ["CORE", "CALIBRATION", "REPLAY"]
    (rd / "run.json").write_text(json.dumps(rj))
    rows = M.block_rows(rd)
    rr = [r for r in rows if r["module"] == "REPLAY" and r["execution_status"] == "COMPLETE"]
    assert len(rr) == 6 and all(r["block_included"] and r["replay_source_block"] == "m" for r in rr)
    assert all(r["replay_owned"] == res["replay"]["grade"][r["concept"]]["owned"] for r in rr)
    assert all(r["replay_owned"] is None for r in rows if r["cell"])
    assert all(f"replay_{k}" in M.COLUMNS for k in M.REPLAY_FIELDS)
    _cli(monkeypatch, "m", "nih", "replay")
    assert "replay" in json.loads((rd / "summary.json").read_text())


# -------------------------------------------------------------------------------------------------------- SEMEND
class _FakeTok:
    """Deterministic ids inside the fake vocabulary (V = 40); finding words and the neutral word never collide."""

    def __init__(self, words):
        self.map = {}
        for i, w in enumerate(words):
            for j, s in enumerate(SM.word_spellings(w)):
                self.map[s] = [4 + 2 * i + (j >= 2)] + ([4 + 2 * i + (j >= 2), 33] if j % 2 else [])

    def encode(self, s, add_special_tokens=False):
        return list(self.map.get(s, [39]))


def _semend_setup(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    words = sorted({P.semend_finding_word("nih", c) for c in CONCEPTS} | {P.SEMEND_NEUTRAL_WORD["nih"]})
    ad.processor = types.SimpleNamespace(tokenizer=_FakeTok(words))
    monkeypatch.setattr(AN, "get_adapter", lambda key, rev=None: ad)
    monkeypatch.setattr(AN, "image_path", lambda ds, row: row["row_id"])
    monkeypatch.setattr(AN, "open_rgb", lambda p: p)
    return ad, runs


def test_semend_prep_runner_and_analysis(tmp_path, monkeypatch):
    ad, runs = _semend_setup(tmp_path, monkeypatch)
    rd = runs / "m" / "nih"
    with pytest.raises(FileNotFoundError, match="cftransfer.ansdir"):     # a_q is one of the 25 conditions
        RN.run_block("m", "nih", "SEMEND", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    RN.run_block("m", "nih", "CORE", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    PK.build("m", "nih")
    with pytest.raises(FileNotFoundError, match="ansdir_seed0"):
        SM.build_semend("m", "nih")
    AN.build_ansdir("m", "nih", n_train=20, device_map="cpu")
    with pytest.raises(FileNotFoundError, match="cftransfer.semend"):      # the competitor table is still missing
        RN.run_block("m", "nih", "SEMEND", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    path, arr = SM.build_semend("m", "nih", n_boot=80)
    assert path == rd / "fits" / "vis.last" / "semend_seed0.npz"
    # the frozen competitor is exactly CORE's strongest competitor per question
    cg = A.core("m", "nih", n_boot=80)
    comp = dict(zip(arr["concept_names"].astype(str), arr["competitor_names"].astype(str)))
    assert list(arr["concept_names"].astype(str)) == CONCEPTS
    assert all(comp[q] == cg["per_question"][q]["argmax_other"] and comp[q] != q for q in CONCEPTS)
    prompts = SM.load_prompts("m", "nih")
    for q in CONCEPTS:
        for t in P.SEMEND_TEMPLATES:
            assert prompts.render("nih", q, t) == P.render_semend("nih", q, t, comp[q])
    # the report endpoint scores the finding word against the neutral word, without a yes/no head
    cs = SM.report_candidates(ad.tokenizer, "nih", "Mass")
    assert cs.template_id == "RF" and cs.positive_ids and cs.negative_ids
    assert not set(cs.positive_ids) & set(cs.negative_ids) and cs.detail["finding_word"] == "mass"
    assert prompts.candidates(ad, "nih", "Mass", "NY") is ad.candidates["IY"]
    assert prompts.candidates(ad, "nih", "Mass", "DA") is ad.candidates["IA"]
    assert prompts.candidates(ad, "nih", "Mass", "DB") is ad.candidates["IB"]
    # runner: 24 questions x 25 conditions, own baseline per (question, endpoint)
    meta = RN.run_block("m", "nih", "SEMEND", 0, 1, batch=32, device_map="cpu", rows_per_part=4)
    assert meta["outcomes"] == 12 * 6 * 4 * 33 and meta["direction_block"] == "m"
    df = pq.read_table(next((rd / "outcomes" / "SEMEND").glob("part-*.parquet"))).to_pandas()
    assert set(df.template_id) == set(P.SEMEND_TEMPLATES) and set(df.fit_seed) == {0}
    assert set(df.direction_kind) == {"baseline", "concept", "sham", "ans", "anssham", "random"}
    for (_rid, q, t), g in df.groupby(["row_id", "concept", "template_id"]):
        assert sorted(g.direction_id) == sorted(d for d, _ in P.conditions_for("SEMEND", "nih", q))
        assert (g.direction_id == "baseline").sum() == 1
    rf = df[df.template_id == "RF"]
    assert {tuple(v) for v in rf[rf.concept == "Mass"].positive_token_ids} == {tuple(cs.positive_ids)}
    # the label and answer families are the block's own seed-0 / ANSDIR directions, written on every question
    f0 = F.load_fit("m", "nih", "vis.last", 0)
    an = AN.load_ansdir("m", "nih", "vis.last")
    bank = RN.DirectionBank("m", "nih", "vis.last", (0,), ansdir=True)
    for ci, q in enumerate(CONCEPTS):
        assert np.array_equal(bank.vector(f"concept:{q}", 0), f0["clinical_vectors"][ci])
        assert np.array_equal(bank.vector(f"ans:{q}", 0), an["answer_vectors"][ci])
    nd = df[(df.row_id == df.row_id.iloc[0]) & (df.concept == "Mass") & (df.template_id == "NY")]
    assert set(nd[nd.direction_kind == "concept"].direction_id) == {f"concept:{c}" for c in CONCEPTS}
    assert set(nd[nd.direction_kind == "ans"].direction_id) == {f"ans:{c}" for c in CONCEPTS}
    # analysis: per endpoint and per family a 6x6 write matrix, own reference, spillover and incremental validity
    PK.build("m", "nih")
    cal = {q: {"readable": True, "n_pos": 20, "n_neg": 20, "selectivity": 0.1 + 0.01 * i,
               "answer_capable": True, "answer_auroc": 0.5 + 0.02 * i} for i, q in enumerate(CONCEPTS)}
    (rd / "summary.json").write_text(json.dumps({"calibration": cal}))
    res = A.semend("m", "nih", draws=80)
    sem = pq.read_table(rd / "outcomes" / "SEMEND.parquet").to_pandas()
    assert res["endpoints"] == list(P.SEMEND_TEMPLATES) and res["n_random"] == 18 and res["competitor"] == comp
    assert res["families"] == ["label", "answer"] and len(res["cells"]) == 6 * 4 * 2
    for t in P.SEMEND_TEMPLATES:
        blk = res["per_endpoint"][t]
        assert blk["status"] in ("COMPLETE", "RUNNING") and blk["n_conditions"] == 33 and blk["sign"] == P.SEMEND_SIGN[t]
        assert blk["value"] == ("logp" if t == "RF" else "p_present") and set(blk["families"]) == {"label", "answer"}
        sub_t = sem[sem.template_id == t]
        base = sub_t[sub_t.direction_id == "baseline"].set_index(["concept", "row_id"])

        def eff(q, d):
            g = sub_t[(sub_t.concept == q) & (sub_t.direction_id == d)]
            b = base.loc[[(q, r) for r in g.row_id]]
            if t == "RF":
                v = g.positive_logits.map(lambda x: float(np.max(x))).values - g.vocab_logsumexp.values
                v0 = b.positive_logits.map(lambda x: float(np.max(x))).values - b.vocab_logsumexp.values
            else:
                v, v0 = g.p_present.values, b.p_present.values
            return P.SEMEND_SIGN[t] * float(np.mean(v - v0))
        for fam, prefix in (("label", "concept"), ("answer", "ans")):
            fb = blk["families"][fam]
            assert len(fb["contrasts"]) == 30                      # 6 questions x 5 competitors, as CORE
            for q in CONCEPTS:
                cell = fb["per_question"][q]
                assert abs(cell["W_qq"] - eff(q, f"{prefix}:{q}")) < 1e-6 and cell["family"] == fam
                assert cell["n_competitors"] == 5 and cell["competitor"] == comp[q]
                assert cell["random_n"] == 18 and 1 <= cell["rank_in_random_family"] <= 19
                assert cell["verdict"] in ("stronger_competitor", "fixed_family_advantage", "unresolved")
                assert cell["steering_reference"] == (cell["W_qq"] > 0 and cell["W_qq"] > cell["random_max"]
                                                      and cell["W_qq"] > abs(cell["E_sham"]))
                assert cell["owned"] == (cell["steering_reference"] and cell["verdict"] == "fixed_family_advantage")
                # spillover: the SAME direction written on the other five concepts' endpoints
                others = [eff(d, f"{prefix}:{q}") for d in CONCEPTS if d != q]
                assert abs(cell["intended"] - cell["W_qq"]) < 1e-9
                assert abs(cell["unintended_mean"] - float(np.mean(others))) < 1e-6
                assert abs(cell["unintended_max"] - max(others)) < 1e-6
                assert abs(cell["selectivity"] - (cell["intended"] - cell["unintended_mean"])) < 1e-6
                assert cell["owned_core"] == (res["core"][q]["steering_reference"]
                                              and res["core"][q]["verdict"] == "fixed_family_advantage")
        assert blk["per_question"] is blk["families"]["label"]["per_question"]
    # incremental validity: reported for both families, with the reference-met-not-owned cells kept as their own group
    for fam in ("label", "answer"):
        iv = res["incremental_validity"][fam]
        assert iv["available"] and iv["n_cells"] == 24 and iv["n_concepts"] == 6 and iv["endpoints"] == list(P.SEMEND_TEMPLATES)
        assert iv["base_predictors"] == ["W_qq_core", "probe_selectivity", "answer_auroc"]
        assert iv["added_predictors"] == ["owned_core", "O_q_core"]
        assert set(iv["groups"]) == {"owned", "reference_met_not_owned", "neither"}
        assert sum(g["n_cells"] for g in iv["groups"].values()) == 24
        add = iv["ownership_adds"]
        assert add["delta_r2"] >= -1e-9 and len(add["delta_r2_ci95"]) == 2         # nesting: full R2 >= base R2
        assert iv["full_model"]["r2"] + 1e-9 >= iv["base_model"]["r2"]
        for k in ("coef_owned_core", "coef_O_q_core"):
            assert np.isfinite(add[k]) and len(add[f"{k}_ci95"]) == 2 and isinstance(
                add[k.replace("coef_", "") + "_interval_excludes_zero"], bool)
        assert set(iv["full_model"]["coefficients"]) == {"intercept", "endpoint:DA", "endpoint:DB", "endpoint:RF",
                                                          "W_qq_core", "probe_selectivity", "answer_auroc",
                                                          "owned_core", "O_q_core"}
        d = iv["owned_minus_reference_met_not_owned"]
        assert "estimate" in d or d.get("available") is False
    # every cell carries its group, and the group labels come from the CORE grade
    for c in res["cells"]:
        assert c["group_core"] in ("owned", "reference_met_not_owned", "neither")
        assert (c["group_core"] == "owned") == c["owned_core"]
    # calibration is required for the regression; without it the analysis says so rather than guessing
    (rd / "summary.json").write_text(json.dumps({}))
    assert A.semend("m", "nih", draws=20)["incremental_validity"]["label"]["available"] is False
    (rd / "summary.json").write_text(json.dumps({"calibration": cal}))
    # the negated endpoint is reported on the raw scale as well, against CORE's affirmative effect
    ng = res["negation"]
    for q in CONCEPTS:
        assert abs(ng["raw_negated_effect"][q] + res["per_endpoint"]["NY"]["families"]["label"]["per_question"][q]["W_qq"]) < 1e-12
        assert ng["affirmative_effect_core"][q] == cg["per_question"][q]["W_qq"]
    assert 0 <= ng["n_opposite_sign"] <= 6 and -1 <= (ng["correlation_affirmative_vs_raw_negated"] or 0) <= 1
    fc = res["forced_choice"]
    for q in CONCEPTS:
        assert abs(fc[q]["mean"] - 0.5 * (fc[q]["DA"] + fc[q]["DB"])) < 1e-12 and fc[q]["competitor"] == comp[q]
        assert abs(fc[q]["order_gap"] - (fc[q]["DA"] - fc[q]["DB"])) < 1e-12
    assert set(res["report"]) == set(CONCEPTS) and res["report"]["Mass"]["finding_word"] == "mass"
    assert res["summary"]["n_cells"] == 24 and set(res["summary"]["n_owned"]) == {"label", "answer"}
    assert set(res["summary"]["reference_met_not_owned_core"]) <= set(CONCEPTS)
    # manifest: one row per (question, endpoint); block_included ignores SEMEND
    (rd / "summary.json").write_text(json.dumps({"core": cg, "semend": res}, **JSON))
    rj = json.loads((rd / "run.json").read_text())
    rj["completed_modules"] = ["CORE", "CALIBRATION", "SEMEND"]
    (rd / "run.json").write_text(json.dumps(rj))
    monkeypatch.setattr(PK, "MODULES", P.MODULES)
    rows = M.block_rows(rd)
    sr = [r for r in rows if r["module"] == "SEMEND"]
    assert len(sr) == 24 and {r["template"] for r in sr} == set(P.SEMEND_TEMPLATES)
    done = [r for r in sr if r["execution_status"] == "COMPLETE"]
    for r in done:
        c = res["per_endpoint"][r["template"]]["families"]["label"]["per_question"][r["concept"]]
        a = res["per_endpoint"][r["template"]]["families"]["answer"]["per_question"][r["concept"]]
        assert (r["semend_owned"], r["semend_verdict"], r["semend_competitor"]) == (c["owned"], c["verdict"], c["competitor"])
        assert (r["semend_intended"], r["semend_selectivity"]) == (c["intended"], c["selectivity"])
        assert (r["semend_answer_E_own"], r["semend_answer_owned"]) == (a["W_qq"], a["owned"])
    assert all(r["semend_owned"] is None for r in rows if r["cell"])
    rj["completed_modules"] = ["SEMEND"]
    (rd / "run.json").write_text(json.dumps(rj))
    assert not any(r["block_included"] for r in M.block_rows(rd))
    assert all(f"semend_{k}" in M.COLUMNS for k in M.SEMEND_FIELDS)
    _cli(monkeypatch, "m", "nih", "semend")
    assert "semend" in json.loads((rd / "summary.json").read_text())


def test_round4_coverage_and_completeness(tmp_path, monkeypatch):
    """Packaged blocks stay COMPLETE until one of the three modules starts, and count it once it has."""
    ad, runs = _setup(tmp_path, monkeypatch)
    rd = runs / "m" / "nih"
    for mod in ("TOWERSWAP", "REPLAY", "SEMEND"):
        assert mod in P.MODULES_ADDED_LATER["nih"] and mod not in PK.requested_modules("nih", rd)
    (rd / "outcomes" / "SEMEND").mkdir(parents=True)
    assert "SEMEND" in PK.requested_modules("nih", rd) and "REPLAY" not in PK.requested_modules("nih", rd)
    cov = PK.coverage_rows("m", "nih", "SEMEND", None, "IY")
    assert len(cov) == 24 and all(r["expected_rows"] == 33 * 600 for r in cov)
    assert all(r["execution_status"] == "NOT_STARTED" and r["baseline_module"] == "" for r in cov)
    cov = PK.coverage_rows("m", "nih", "TOWERSWAP", None, "IY")
    assert len(cov) == 6 and all(r["expected_rows"] == 127 * 200 for r in cov)
