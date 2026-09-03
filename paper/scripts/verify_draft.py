#!/usr/bin/env python3
"""Static, evidence-aware validation for the evidence-locked paper draft."""

from __future__ import annotations

import re
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
MAIN = (PAPER / "main.tex").read_text(encoding="utf-8")
SECTION_FILES = sorted((PAPER / "sections").glob("*.tex"))
TABLE_FILES = sorted((PAPER / "tables").glob("*.tex"))
TEXT = "\n".join(path.read_text(encoding="utf-8") for path in SECTION_FILES + TABLE_FILES)
BIB = (PAPER / "refs.bib").read_text(encoding="utf-8")


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> None:
    inputs = set(re.findall(r"\\input\{([^}]+)\}", MAIN))
    expected_sections = {f"sections/{path.stem}" for path in SECTION_FILES}
    stale = sorted(expected_sections - inputs)
    if stale:
        fail(f"stale section files: {stale}")

    all_tex = MAIN + "\n" + TEXT
    labels = set(re.findall(r"\\label\{([^}]+)\}", all_tex))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", all_tex))
    if refs - labels:
        fail(f"undefined references: {sorted(refs - labels)}")

    cited: set[str] = set()
    for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", all_tex):
        cited.update(key.strip() for key in group.split(","))
    bib_keys = set(re.findall(r"@[A-Za-z]+\{([^,]+),", BIB))
    if cited - bib_keys:
        fail(f"missing bibliography entries: {sorted(cited - bib_keys)}")
    if bib_keys - cited:
        fail(f"uncited bibliography entries: {sorted(bib_keys - cited)}")

    abstract = (PAPER / "sections" / "0_abstract.tex").read_text(encoding="utf-8")
    word_count = len(re.findall(r"\b[\w'-]+\b", re.sub(r"\\[A-Za-z]+", " ", abstract)))
    if not 180 <= word_count <= 250:
        fail(f"abstract word count outside 180--250: {word_count}")

    forbidden = ["TODO", "FIXME", "XXX", "[VERIFY]", "delve", "pivotal", "tapestry", "underscore"]
    hits = [token for token in forbidden if token.lower() in all_tex.lower()]
    if hits:
        fail(f"draft markers or watch words remain: {hits}")

    required_numbers = {
        "llava_effusion_selectivity": "0.1139",
        "llava_effusion_ci_low": "0.0875",
        "llava_effusion_effect": "0.1470",
        "llava_effusion_sham": "0.1546",
        "llava_edema_selectivity": "0.1360",
        "llava_edema_paired_ci_low": "-0.0175",
        "llava_edema_effect": "0.0659",
        "qwen_selectivity": "0.1166",
        "qwen_paired_ci_low": "-0.0155",
        "qwen_effect": "0.2343",
        "qwen_random": "0.0963",
        "qwen_nodule_original": "0.2863",
        "supplement_effusion": "0.1945",
        "supplement_random": "0.0878",
        "supplement_sham": "0.0157",
        "supplement_nodule": "0.2519",
        "supplement_margin": "-0.0573",
        "supplement_one_sided": "-0.0678",
    }
    missing = [name for name, value in required_numbers.items() if value not in TEXT]
    if missing:
        fail(f"adjudicated values absent from draft: {missing}")

    if "Anonymous Authors" not in MAIN:
        fail("anonymous author block is missing")
    identity_pattern = r"wy-coliney|qingchan|University|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"
    if re.search(identity_pattern, all_tex, re.IGNORECASE):
        fail("potential identifying text in manuscript")

    print(f"sections={len(SECTION_FILES)}")
    print(f"tables={len(TABLE_FILES)}")
    print(f"citations={len(cited)}")
    print(f"labels={len(labels)}")
    print(f"abstract_words={word_count}")
    print(f"adjudicated_values={len(required_numbers)}")


if __name__ == "__main__":
    main()
