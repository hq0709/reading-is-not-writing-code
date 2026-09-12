"""Answer-candidate token sets and margin scores (README 5.3, return-format outcomes columns).

Candidates: yes/no -> plain, capitalised and leading-space single-token spellings; A/B -> plain and
leading-space. Only true single-token candidates are kept, deduplicated by id. Positive and negative
sets must be non-empty and disjoint; otherwise the interface is not ready and scoring must not proceed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .protocol import AB_TEMPLATES, B_POSITIVE_TEMPLATES, YESNO_TEMPLATES

YES_STRINGS = ["yes", "Yes", " yes", " Yes"]
NO_STRINGS = ["no", "No", " no", " No"]
A_STRINGS = ["A", " A"]
B_STRINGS = ["B", " B"]


def single_token_ids(tokenizer, strings: list[str]) -> tuple[list[int], dict[str, int | None]]:
    ids, detail = [], {}
    for s in strings:
        enc = tokenizer.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            detail[s] = int(enc[0])
            if int(enc[0]) not in ids:
                ids.append(int(enc[0]))
        else:
            detail[s] = None
    return ids, detail


@dataclass(frozen=True)
class CandidateSet:
    template_id: str
    positive_ids: tuple[int, ...]
    negative_ids: tuple[int, ...]
    a_ids: tuple[int, ...]          # raw A set (empty for yes/no templates)
    b_ids: tuple[int, ...]
    detail: dict

    def check(self) -> None:
        if not self.positive_ids or not self.negative_ids:
            raise RuntimeError(f"{self.template_id}: empty candidate set {self.detail}")
        if set(self.positive_ids) & set(self.negative_ids):
            raise RuntimeError(f"{self.template_id}: candidate sets intersect {self.detail}")


def candidate_sets(tokenizer) -> dict[str, CandidateSet]:
    yes, dy = single_token_ids(tokenizer, YES_STRINGS)
    no, dn = single_token_ids(tokenizer, NO_STRINGS)
    a, da = single_token_ids(tokenizer, A_STRINGS)
    b, db = single_token_ids(tokenizer, B_STRINGS)
    out = {}
    for t in YESNO_TEMPLATES:
        out[t] = CandidateSet(t, tuple(yes), tuple(no), (), (), {"yes": dy, "no": dn})
    for t in AB_TEMPLATES:
        pos, neg = (b, a) if t in B_POSITIVE_TEMPLATES else (a, b)
        out[t] = CandidateSet(t, tuple(pos), tuple(neg), tuple(a), tuple(b), {"A": da, "B": db})
    for c in out.values():
        c.check()
    return out


@dataclass
class Scores:
    positive_logits: np.ndarray     # (B, n_pos) float32
    negative_logits: np.ndarray     # (B, n_neg)
    vocab_logsumexp: np.ndarray     # (B,) float32
    semantic_margin: np.ndarray     # (B,) float32: max pos - max neg
    p_present: np.ndarray           # (B,) float64: sigmoid(margin)
    lse_margin: np.ndarray          # (B,) float32
    raw_ab_margin: np.ndarray       # (B,) float32, NaN for yes/no templates
    candidate_mass: np.ndarray      # (B,) float32: total softmax mass of all candidates


def score_logits(last_logits: torch.Tensor, cands: CandidateSet) -> Scores:
    """`last_logits`: (B, V) logits at the fixed answer position, any dtype; computed in float32."""
    lg = last_logits.detach().float()
    pos = lg[:, list(cands.positive_ids)]
    neg = lg[:, list(cands.negative_ids)]
    lse_all = torch.logsumexp(lg, dim=-1)
    margin = pos.max(dim=-1).values - neg.max(dim=-1).values
    lse = torch.logsumexp(pos, dim=-1) - torch.logsumexp(neg, dim=-1)
    if cands.a_ids:
        raw = lg[:, list(cands.a_ids)].max(dim=-1).values - lg[:, list(cands.b_ids)].max(dim=-1).values
    else:
        raw = torch.full((lg.shape[0],), float("nan"), dtype=torch.float32)
    all_ids = list(cands.positive_ids) + list(cands.negative_ids)
    mass = torch.exp(torch.logsumexp(lg[:, all_ids], dim=-1) - lse_all)
    return Scores(
        positive_logits=pos.cpu().numpy().astype(np.float32),
        negative_logits=neg.cpu().numpy().astype(np.float32),
        vocab_logsumexp=lse_all.cpu().numpy().astype(np.float32),
        semantic_margin=margin.cpu().numpy().astype(np.float32),
        p_present=torch.sigmoid(margin.double()).cpu().numpy(),
        lse_margin=lse.cpu().numpy().astype(np.float32),
        raw_ab_margin=raw.cpu().numpy().astype(np.float32),
        candidate_mass=mass.cpu().numpy().astype(np.float32),
    )
