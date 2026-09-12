"""Runner end to end on CPU with a fake family: grid completeness, hook steering reaches the logits,
resumability, failed-row handling, and package/coverage bookkeeping."""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from cftransfer import fit as F, images as I, runpaths as R, protocol as P, runner as RN, package as PK   # noqa: E402
from cftransfer.adapters.base import Adapter, LocusInfo                                                  # noqa: E402
from cftransfer.hooks import TokenLayout                                                                   # noqa: E402
from cftransfer.scoring import CandidateSet                                                                # noqa: E402
from test_fit_synthetic import make_synthetic                                                              # noqa: E402

D, V, T = 16, 40, 9


class Vis(torch.nn.Module):
    def forward(self, x):
        return x


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vis = Vis()
        self.W = torch.nn.Parameter(torch.randn(D, V, generator=torch.Generator().manual_seed(0)))

    def forward(self, feats):
        h = self.vis(feats)                                    # (B*T, D) flat, hook applies here
        pooled = h.view(-1, T, D).mean(1)
        return pooled @ self.W


class FakeAdapter(Adapter):
    family = "fake"

    def load(self, device_map="cpu", dtype=None):
        self.model = FakeModel()
        self._modules = dict(self.model.named_modules())
        self._cands = {t: CandidateSet(t, (1, 2), (3,), (1, 2), (3,), {}) for t in P.TEMPLATE_ORDER}
        self.fail_row = getattr(self, 'fail_row', None)
        return self

    def loci(self):
        return {"vis.last": LocusInfo("vis.last", "vis", "t", D, "all", "main", "none", "W"),
                "connector": LocusInfo("connector", "vis", "t", D, "all", "main", "none", "W")}

    def encode(self, images, questions):
        B = len(questions)
        g = torch.Generator().manual_seed(abs(hash(images[0])) % 1000)
        feats = torch.randn(T, D, generator=g).repeat(B, 1)
        if self.fail_row is not None and images[0] == self.fail_row:
            feats = torch.full_like(feats, float("nan"))
        return {"feats": feats, "attention_mask": torch.ones(B, 5, dtype=torch.long)}

    def expand(self, enc, B):
        return {"feats": enc["feats"].repeat(B, 1), "attention_mask": enc["attention_mask"].repeat(B, 1)}

    def layouts(self, enc, images):
        B = len(images)
        return {"vis.last": TokenLayout(True, slices=[(i * T, (i + 1) * T) for i in range(B)]),
                "connector": TokenLayout(True, slices=[(i * T, (i + 1) * T) for i in range(B)])}

    @torch.no_grad()
    def forward_last_logits(self, enc):
        out = self.model(enc["feats"])
        if torch.isnan(out).any():
            raise RuntimeError("nan features")
        return out.float()

    def processing_settings(self):
        return {"candidate_tokens": {t: {"positive_ids": [1, 2], "negative_ids": [3], "detail": {}} for t in P.TEMPLATE_ORDER},
                "example_prompt_IY": "Is there X in this image? Answer yes or no.", "attn_implementation": "cpu"}


def _setup(tmp_path, monkeypatch):
    data, runs = make_synthetic(tmp_path)
    monkeypatch.setattr(I, "DATA_ROOT", data)
    monkeypatch.setattr(R, "RUN_ROOT", runs)
    monkeypatch.setattr(I, "image_path", lambda ds, row: row["row_id"])
    monkeypatch.setattr(I, "open_rgb", lambda p: p)
    monkeypatch.setattr(RN, "image_path", lambda ds, row: row["row_id"])
    monkeypatch.setattr(RN, "open_rgb", lambda p: p)
    F.fit_locus("m", "nih", "vis.last", seeds=(0, 1, 2), write_scores=False)
    ad = FakeAdapter("m", "fake/m", "rev")
    monkeypatch.setattr(RN, "get_adapter", lambda key, rev=None: ad)
    monkeypatch.setattr(PK, "MODELS", {**PK.MODELS, "m": {"model_id": "fake/m"}})
    return ad, runs


def test_runner_calibration_and_core_grid(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    meta = RN.run_block("m", "nih", "CALIBRATION", 0, 1, batch=8, device_map="cpu", rows_per_part=5)
    t = pq.read_table(runs / "m" / "nih" / "outcomes" / "CALIBRATION" / "part-000of001-00000.parquet").to_pandas()
    assert meta["outcomes"] == 12 * 16 and len(t) == 5 * 16 and set(t.direction_id) == {"baseline"}
    # CORE on 3 test rows, two shards, batch 16: full 127-condition grid per question, steered rows differ from baseline
    m0 = RN.run_block("m", "nih", "CORE", 0, 2, batch=16, device_map="cpu", rows_per_part=1, limit_rows=3)
    m1 = RN.run_block("m", "nih", "CORE", 1, 2, batch=16, device_map="cpu", rows_per_part=1, limit_rows=3)
    assert m0["outcomes"] + m1["outcomes"] == 3 * 6 * 127
    parts = sorted((runs / "m" / "nih" / "outcomes" / "CORE").glob("part-*.parquet"))
    df = pq.read_table(parts[0]).to_pandas()
    q = df[(df.concept == "Effusion")]
    base = q[q.direction_id == "baseline"].semantic_margin.iloc[0]
    assert (q[q.direction_id != "baseline"].semantic_margin != base).all()
    assert (q[q.direction_id != "baseline"].delta_norm_mean > 0).all() and (q[q.direction_id == "baseline"].delta_norm_mean == 0).all()
    assert sorted(q.direction_id.unique()) == sorted(d for d, _ in P.conditions_for("CORE", "nih", "Effusion"))
    # resume: nothing left to do
    again = RN.run_block("m", "nih", "CORE", 0, 2, batch=16, device_map="cpu", rows_per_part=1, limit_rows=3)
    assert again["todo"] == 0


def test_runner_failed_rows_and_package(tmp_path, monkeypatch):
    ad, runs = _setup(tmp_path, monkeypatch)
    rows = I.load_cohort("nih", ("test",))
    ad.fail_row = rows[1]["row_id"]
    RN.run_block("m", "nih", "REFIT", 0, 1, batch=8, device_map="cpu", rows_per_part=2, limit_rows=2)
    df = pq.concat_tables([pq.read_table(p) for p in (runs / "m" / "nih" / "outcomes" / "REFIT").glob("part-*.parquet")]).to_pandas() \
        if False else pq.read_table(list((runs / "m" / "nih" / "outcomes" / "REFIT").glob("part-*.parquet"))[0]).to_pandas()
    failed = df[df.sample_status == "FAILED"]
    assert len(failed) == 6 * 7 * 2 and failed.p_present.isna().all() and failed.error_reason.str.contains("nan").all()
    assert len(df[df.sample_status == "OK"]) == 6 * 7 * 2
    run = PK.build("m", "nih")
    cov = (runs / "m" / "nih" / "coverage.csv").read_text().splitlines()
    assert any(",REFIT," in l and ",FAILED," in l for l in cov)
    assert any(",CORE," in l and ",NOT_STARTED," in l for l in cov)
    assert any(",chexpert," not in l for l in cov)
    assert run["completed_modules"] == [] and (runs / "m" / "nih" / "outcomes" / "REFIT.parquet").exists()
