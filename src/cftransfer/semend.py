"""SEMEND: semantic endpoints that are not the six yes/no templates.

Ownership and the answer direction are today validated against the model's own yes/no (or A/B) answers under
the six protocol templates, so a reviewer cannot separate semantic control from template-specific sensitivity.
SEMEND writes the same directions at the same locus and dose and scores three endpoints that no protocol
template supplies, each with its own clean baseline and its own steering reference:

  NY  negated question   "Is there no evidence of <finding> in <image>? Answer yes or no."
                         A direction that carries the concept must move this answer in the OPPOSITE direction
                         to the affirmative question, so its endpoint is scored with sign -1 and reported
                         together with the correlation against the affirmative (CORE) effect.
  DA  forced choice, concept first   "Which is present in <image>, <q> or <d*>? Answer A ... B ..."
  DB  forced choice, concept second  (the same pair with the order and the A/B mapping swapped)
                         d* is the question's STRONGEST COMPETITOR in the block's own CORE write matrix
                         (argmax_{d != q} W_qd), frozen by this module's prep. The endpoint is the
                         probability of the CONCEPT option, so DA and DB counterbalance the order.
  RF  finding-word likelihood   "<report prefix>\\nFindings:" (COCO: a list prefix)
                         The endpoint is the model's log-probability of the finding word at the continuation
                         position, log p = max(positive logits) - vocab_logsumexp, both already stored by the
                         runner. It needs no yes/no head at all.

Conditions per (question, endpoint): clean baseline, the SIX label directions, the SIX answer directions of
the ANSDIR prep, the question's label sham, its answer sham and SEMEND_N_RANDOM protocol random directions
-- 33 conditions, of which 32 are steered and therefore fill exactly one batch-32 forward, so the full grid
costs the GPU time a grid half its size would. Both families thus have a complete 6x6 write matrix on every
endpoint: the campaign's ownership rule applies unchanged, and the spillover of one direction onto the OTHER
five concepts' endpoints is measured, which is what the incremental-validity analysis regresses.

Prep (CPU): `python -m cftransfer.semend --model-key K --dataset D` reads the block's CORE outcomes, freezes
the strongest competitor per question and the exact rendered prompt of every (question, endpoint), and writes
fits/vis.last/semend_seed0.npz. The prep needs fits/vis.last/ansdir_seed0.npz to exist (the answer direction
a_q is part of the grid), so SEMEND is enqueued with or after ANSDIR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .protocol import (CONCEPTS, LOCI, PRIMARY_ALPHA, SEMEND_AB_TEMPLATE_SOURCE, SEMEND_NEUTRAL_WORD,
                       SEMEND_TEMPLATES, SEMEND_YESNO_TEMPLATE_SOURCE, primary_template, render_semend,
                       semend_finding_word)
from .runpaths import fits_dir, run_dir
from .scoring import CandidateSet


def first_token_ids(tokenizer, strings: list[str]) -> tuple[list[int], dict]:
    """First token id of each spelling (deduplicated, order preserved).

    The report endpoint scores the word at ONE continuation position, so what matters is the first token the
    word would emit; unlike the yes/no candidates it is not required to be a whole single token.
    """
    ids, detail = [], {}
    for s in strings:
        enc = tokenizer.encode(s, add_special_tokens=False)
        if not enc:
            detail[s] = None
            continue
        detail[s] = [int(enc[0]), len(enc)]
        if int(enc[0]) not in ids:
            ids.append(int(enc[0]))
    return ids, detail


def word_spellings(word: str) -> list[str]:
    return [word, word[:1].upper() + word[1:], " " + word, " " + word[:1].upper() + word[1:]]


def report_candidates(tokenizer, dataset_id: str, concept: str) -> CandidateSet:
    """Finding word versus the dataset's neutral word, as first-token sets at the continuation position."""
    word = semend_finding_word(dataset_id, concept)
    pos, dp = first_token_ids(tokenizer, word_spellings(word))
    neg, dn = first_token_ids(tokenizer, word_spellings(SEMEND_NEUTRAL_WORD[dataset_id]))
    neg = [i for i in neg if i not in set(pos)]
    cs = CandidateSet("RF", tuple(pos), tuple(neg), (), (),
                      {"finding_word": word, "neutral_word": SEMEND_NEUTRAL_WORD[dataset_id],
                       "finding": dp, "neutral": dn,
                       "rule": "first token of each spelling; log p(finding word) = max(positive_logits) - vocab_logsumexp"})
    cs.check()
    return cs


class SemendPrompts:
    """The frozen prompts and competitors of one block (fits/<locus>/semend_seed0.npz)."""

    def __init__(self, arr: dict, model_key: str, dataset_id: str):
        self.model_key, self.dataset_id = model_key, dataset_id
        self.concepts = list(np.asarray(arr["concept_names"]).astype(str))
        self.competitor = dict(zip(self.concepts, np.asarray(arr["competitor_names"]).astype(str)))
        self.prompts = json.loads(str(arr["prompts_json"]))
        self.meta = json.loads(str(arr["meta_json"]))
        self._cands: dict[tuple[str, str], CandidateSet] = {}

    def render(self, dataset_id: str, concept: str, template_id: str) -> str:
        """The exact question text the prep froze (identical to protocol.render_semend on this block)."""
        key = f"{dataset_id}|{concept}|{template_id}"
        if key not in self.prompts:
            raise KeyError(f"semend_seed0.npz has no prompt for {key}")
        return self.prompts[key]["question"]

    def candidates(self, ad, dataset_id: str, concept: str, template_id: str) -> CandidateSet:
        """NY / DA / DB reuse the frozen yes-no and A/B candidate sets; RF scores the finding word."""
        if template_id in SEMEND_YESNO_TEMPLATE_SOURCE:
            return ad.candidates[SEMEND_YESNO_TEMPLATE_SOURCE[template_id]]
        if template_id in SEMEND_AB_TEMPLATE_SOURCE:
            return ad.candidates[SEMEND_AB_TEMPLATE_SOURCE[template_id]]
        if template_id != "RF":
            raise KeyError(f"{template_id}: not a SEMEND endpoint ({SEMEND_TEMPLATES})")
        if (dataset_id, concept) not in self._cands:
            self._cands[(dataset_id, concept)] = report_candidates(ad.tokenizer, dataset_id, concept)
        return self._cands[(dataset_id, concept)]


def semend_path(model_key: str, dataset_id: str, locus_id: str = "vis.last") -> Path:
    return fits_dir(model_key, dataset_id, locus_id) / "semend_seed0.npz"


def load_semend(model_key: str, dataset_id: str, locus_id: str = "vis.last") -> dict:
    p = semend_path(model_key, dataset_id, locus_id)
    if not p.exists():
        raise FileNotFoundError(f"SEMEND needs {p}; run python -m cftransfer.semend --model-key {model_key} "
                                f"--dataset {dataset_id}")
    return dict(np.load(p, allow_pickle=False))


def load_prompts(model_key: str, dataset_id: str, locus_id: str = "vis.last") -> SemendPrompts:
    return SemendPrompts(load_semend(model_key, dataset_id, locus_id), model_key, dataset_id)


def build_semend(model_key: str, dataset_id: str, locus_id: str = "vis.last", alpha: float = PRIMARY_ALPHA,
                 template_id: str | None = None, require_ansdir: bool = True, out_dir: Path | None = None,
                 n_boot: int = 200) -> tuple[Path, dict]:
    """Freeze the strongest competitor per question (from the block's own CORE write matrix) and the prompts."""
    from .analysis import core                      # local: analysis imports many preps, semend imports none
    rd = run_dir(model_key, dataset_id)
    if require_ansdir and not (fits_dir(model_key, dataset_id, locus_id) / "ansdir_seed0.npz").exists():
        raise FileNotFoundError(f"SEMEND writes the answer direction a_q as well, so it needs "
                                f"{fits_dir(model_key, dataset_id, locus_id) / 'ansdir_seed0.npz'}; run "
                                f"python -m cftransfer.ansdir --model-key {model_key} --dataset {dataset_id} first")
    primary = template_id or primary_template(rd)
    g = core(model_key, dataset_id, "CORE", template_id=primary, alpha=alpha, n_boot=n_boot)
    concepts = CONCEPTS[dataset_id]
    comp, w_qq, w_qc = [], [], []
    for q in concepts:
        cell = g["per_question"][q]
        d = cell["argmax_other"]
        if d is None or d not in concepts:
            raise RuntimeError(f"{model_key}/{dataset_id}: CORE has no strongest competitor for {q}")
        comp.append(d)
        w_qq.append(float(cell["W_qq"]))
        w_qc.append(float(g["W"][q][f"concept:{d}"]))
    prompts = {}
    for ci, q in enumerate(concepts):
        for t in SEMEND_TEMPLATES:
            prompts[f"{dataset_id}|{q}|{t}"] = {
                "dataset_id": dataset_id, "concept": q, "template_id": t, "competitor": comp[ci],
                "question": render_semend(dataset_id, q, t, comp[ci]),
                "endpoint": {"NY": "negated yes/no (sign -1: a concept direction must lower p(yes))",
                             "DA": "forced choice, concept is option A", "DB": "forced choice, concept is option B",
                             "RF": "log p(finding word) at the report continuation"}[t],
                "finding_word": semend_finding_word(dataset_id, q) if t == "RF" else None,
                "neutral_word": SEMEND_NEUTRAL_WORD[dataset_id] if t == "RF" else None}
    meta = {"model_key": model_key, "dataset_id": dataset_id, "locus_id": locus_id, "template_id": primary,
            "alpha": float(alpha), "competitor_source": "CORE argmax_{d != q} W_qd on the block's primary template",
            "core_n_rows": int(g["n_rows"]), "endpoints": list(SEMEND_TEMPLATES),
            "note": "prompts are frozen here so the scored text is auditable and identical across shards"}
    arr = {"concept_names": np.array(concepts), "competitor_names": np.array(comp),
           "W_qq_core": np.array(w_qq, dtype=np.float64), "W_qcomp_core": np.array(w_qc, dtype=np.float64),
           "O_q_core": np.array([float(g["per_question"][q]["O_q"]) for q in concepts], dtype=np.float64),
           "prompts_json": np.array(json.dumps(prompts)), "meta_json": np.array(json.dumps(meta))}
    out = Path(out_dir) / "semend_seed0.npz" if out_dir else semend_path(model_key, dataset_id, locus_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arr)
    print(json.dumps({**meta, "competitors": dict(zip(concepts, comp))}, indent=1), flush=True)
    return out, arr


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--locus", default=LOCI["primary"])
    ap.add_argument("--alpha", type=float, default=PRIMARY_ALPHA)
    ap.add_argument("--allow-missing-ansdir", action="store_true")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    build_semend(a.model_key, a.dataset, a.locus, a.alpha, require_ansdir=not a.allow_missing_ansdir,
                 out_dir=Path(a.out_dir) if a.out_dir else None)
