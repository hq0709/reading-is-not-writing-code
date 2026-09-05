"""Independent terminal replay for the LLaVA text semantic calibration."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

OBJECTS = ("red square", "blue circle")
PAIRS = (
    ("The scene contains a {object}.", "The scene contains no {object}."),
    ("A {object} is present.", "A {object} is absent."),
    ("It is false that the {object} is absent.", "It is false that the {object} is present."),
    (
        "The list is complete. Listed objects: {object}, green triangle.",
        "The list is complete. Listed objects: green triangle.",
    ),
)
CONDITIONS = (
    ("Y", "Y", False, "Answer yes or no."),
    (
        "incumbent-A",
        "A",
        False,
        "Answer A if the finding is present and B if it is absent. Reply with A or B only.",
    ),
    (
        "incumbent-B",
        "B",
        False,
        "Answer B if the finding is present and A if it is absent. Reply with A or B only.",
    ),
    (
        "candidate-A",
        "A",
        True,
        (
            "Choose the correct option.\nA. The finding is present.\n"
            "B. The finding is absent.\nReply with A or B only."
        ),
    ),
    (
        "candidate-B",
        "B",
        True,
        (
            "Choose the correct option.\nA. The finding is absent.\n"
            "B. The finding is present.\nReply with A or B only."
        ),
    ),
)
TOKENS = {
    "Y": ([3869, 4874, 22483], [694, 1939, 11698]),
    "A": ([319], [350]),
    "B": ([319], [350]),
}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        check(len(fields) == 2, "malformed dispatcher manifest")
        sha, name = fields
        name = name.strip().lstrip("*").removeprefix("./")
        check(re.fullmatch(r"[0-9a-f]{64}", sha) is not None, "malformed digest")
        check(name and name not in result, "duplicate dispatcher manifest member")
        result[name] = sha
    return result


def expected_cases():
    result = []
    for object_index, object_name in enumerate(OBJECTS):
        prefix = f"The target finding is a {object_name}."
        for pair_index, pair in enumerate(PAIRS, start=1):
            for present, sentence in ((True, pair[0]), (False, pair[1])):
                polarity = "positive" if present else "negative"
                description = f"{prefix} {sentence.format(object=object_name)}"
                for condition, encoding, candidate, suffix in CONDITIONS:
                    result.append(
                        {
                            "case_id": (
                                f"object-{object_index + 1}-pair-{pair_index}-{polarity}"
                            ),
                            "object": object_name,
                            "pair": pair_index,
                            "expected_present": present,
                            "expected_sign": 1 if present else -1,
                            "description": description,
                            "condition": condition,
                            "encoding": encoding,
                            "candidate": candidate,
                            "user_prompt": f"{description} Is the finding present? {suffix}",
                        }
                    )
    return result


def replay_record(record, identity):
    for key, value in identity.items():
        check(record.get(key) == value, "outcome identity or ordering")
    check(isinstance(record.get("rendered_prompt"), str) and record["rendered_prompt"], "rendered prompt")
    first, second = TOKENS[identity["encoding"]]
    ids = sorted(set(first + second))
    check(record.get("candidate_token_ids") == ids, "token identity")
    logits = np.asarray(record.get("candidate_logits"), dtype=np.float32)
    check(logits.shape == (len(ids),) and np.isfinite(logits).all(), "candidate logits")
    partition = record.get("log_partition")
    check(
        not isinstance(partition, bool)
        and isinstance(partition, (int, float))
        and math.isfinite(partition),
        "log partition",
    )
    check(partition >= float(logits.max()), "log partition bound")
    position = {token_id: i for i, token_id in enumerate(ids)}
    first_values = logits[[position[token_id] for token_id in first]]
    second_values = logits[[position[token_id] for token_id in second]]
    raw = float(first_values.max() - second_values.max())
    semantic = -raw if identity["encoding"] == "B" else raw
    oriented = semantic * identity["expected_sign"]
    lse = None
    if identity["encoding"] == "Y":
        lse = float(
            np.logaddexp.reduce(first_values.astype(np.float64))
            - np.logaddexp.reduce(second_values.astype(np.float64))
        )
    lse_oriented = None if lse is None else lse * identity["expected_sign"]
    candidate_lse = float(np.logaddexp.reduce(logits.astype(np.float64)))
    mass = float(np.exp(candidate_lse - partition))
    check(math.isfinite(mass) and 0 <= mass <= 1 + 1e-6, "answer-token mass")
    mass = min(mass, 1.0)
    expected = {
        "raw_margin": raw,
        "semantic_margin": semantic,
        "oriented_margin": oriented,
        "primary_sign_correct": oriented > 0,
        "lse_margin": lse,
        "lse_oriented_margin": lse_oriented,
        "lse_sign_correct": None if lse is None else lse_oriented > 0,
        "answer_token_mass": mass,
    }
    for key, value in expected.items():
        check(record.get(key) == value, "derived score " + key)
    check(
        set(record)
        == set(identity)
        | {
            "rendered_prompt",
            "candidate_token_ids",
            "candidate_logits",
            "log_partition",
            "answer_token_mass",
            "raw_margin",
            "semantic_margin",
            "oriented_margin",
            "primary_sign_correct",
            "lse_margin",
            "lse_oriented_margin",
            "lse_sign_correct",
        },
        "outcome schema",
    )


def terminal_route(outcomes):
    yes = [row for row in outcomes if row["condition"] == "Y"]
    candidate = [row for row in outcomes if row["candidate"]]
    if not all(row["primary_sign_correct"] and row["lse_sign_correct"] for row in yes):
        return "comprehension_unresolved"
    if all(row["primary_sign_correct"] for row in candidate):
        return "candidate_eligible"
    return "yes_no_route"


def summary_rows(outcomes):
    conditions = []
    paired = []
    for condition, encoding, _candidate, _suffix in CONDITIONS:
        selected = [row for row in outcomes if row["condition"] == condition]
        conditions.append(
            {
                "condition": condition,
                "primary_sign_errors": [
                    row["case_id"] for row in selected if not row["primary_sign_correct"]
                ],
                "lse_sign_errors": [
                    row["case_id"] for row in selected if row["lse_sign_correct"] is False
                ],
                "minimum_oriented_margin": min(row["oriented_margin"] for row in selected),
                "minimum_lse_oriented_margin": (
                    min(row["lse_oriented_margin"] for row in selected)
                    if encoding == "Y"
                    else None
                ),
            }
        )
        for object_name in OBJECTS:
            for pair_index in range(1, 5):
                positive = next(
                    row
                    for row in selected
                    if row["object"] == object_name
                    and row["pair"] == pair_index
                    and row["expected_present"]
                )
                negative = next(
                    row
                    for row in selected
                    if row["object"] == object_name
                    and row["pair"] == pair_index
                    and not row["expected_present"]
                )
                paired.append(
                    {
                        "condition": condition,
                        "object": object_name,
                        "pair": pair_index,
                        "positive_minus_negative_semantic_margin": (
                            positive["semantic_margin"] - negative["semantic_margin"]
                        ),
                    }
                )
    return conditions, paired


def verify(run_directory, expected_commit):
    run = Path(run_directory).resolve(strict=True)
    check(re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None, "full commit")
    manifest = inventory(run / "SHA256SUMS")
    required = (
        "metadata.env",
        "command_exit_status",
        "exit_status",
        "artifacts/llava-text-semantic-calibration-meta.json",
        "artifacts/llava-text-semantic-calibration-outcomes.json",
        "artifacts/llava-text-semantic-calibration-summary.json",
    )
    verified = {}
    for name in required:
        check(manifest.get(name) == digest(run / name), "dispatcher checksum " + name)
        verified[name] = manifest[name]
    metadata = dict(
        line.split("=", 1)
        for line in (run / "metadata.env").read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    check(metadata.get("SOURCE_COMMIT") == expected_commit, "source commit")
    check(
        metadata.get("COMMAND_STATUS") == metadata.get("DISPATCHER_STATUS") == "0"
        and metadata.get("CLEANUP_STATUS") == "0"
        and metadata.get("ABORT_SIGNAL") == "none"
        and metadata.get("GPU_COUNT") == "1",
        "terminal status",
    )
    check((run / "command_exit_status").read_text().strip() == "0", "command status file")
    check((run / "exit_status").read_text().strip() == "0", "dispatcher status file")
    meta = read(run / "artifacts/llava-text-semantic-calibration-meta.json")
    outcomes = read(run / "artifacts/llava-text-semantic-calibration-outcomes.json")
    summary = read(run / "artifacts/llava-text-semantic-calibration-summary.json")
    check(meta.get("source_commit") == expected_commit, "meta commit")
    check(meta.get("protocol", {}).get("n_outcomes") == 80, "protocol count")
    check(meta.get("protocol", {}).get("conditions") == [row[0] for row in CONDITIONS], "condition order")
    check("dataset" not in meta and "cohort" not in meta, "text-only metadata scope")
    source = meta.get("source", {})
    check(
        source.get("model_id") == "llava-hf/llava-1.5-7b-hf"
        and source.get("model_revision") == "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
        and source.get("asset_receipt", {}).get("revision")
        == "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
        and source.get("asset_receipt", {}).get("asset") == "llava15_7b",
        "accepted model identity",
    )
    check(
        set(source)
        == {"model_id", "model_revision", "model_source", "model_receipt", "asset_receipt"},
        "model-only source schema",
    )
    expected_tokens = {}
    for condition, encoding, _candidate, _suffix in CONDITIONS:
        first, second = TOKENS[encoding]
        present, absent = (second, first) if encoding == "B" else (first, second)
        expected_tokens[condition] = {
            "raw_first": list(first),
            "raw_second": list(second),
            "present": list(present),
            "absent": list(absent),
            "candidate_ids": sorted(set(first + second)),
        }
    check(meta.get("token_ids") == expected_tokens, "pinned token metadata")
    expected = expected_cases()
    check(len(outcomes) == len(expected) == 80, "eighty outcomes")
    for record, identity in zip(outcomes, expected, strict=True):
        replay_record(record, identity)
    route = terminal_route(outcomes)
    condition_rows, paired_rows = summary_rows(outcomes)
    check(summary.get("source_commit") == expected_commit, "summary commit")
    check(
        summary.get("status") == "OBSERVED"
        and summary.get("gate") == "llava-text-semantic-calibration"
        and summary.get("n_descriptions") == 16
        and summary.get("n_conditions") == 5
        and summary.get("n_outcomes") == 80
        and summary.get("route") == route,
        "summary decision",
    )
    check(summary.get("conditions") == condition_rows, "condition summaries")
    check(
        summary.get("paired_positive_minus_negative") == paired_rows,
        "paired positive-minus-negative summaries",
    )
    check(summary.get("protocol") == meta.get("protocol"), "summary protocol")
    check(summary.get("source") == source, "summary source")
    check(summary.get("token_ids") == expected_tokens, "summary token metadata")
    check(
        isinstance(summary.get("elapsed_seconds"), (int, float))
        and 0 < summary["elapsed_seconds"] <= 900,
        "resource envelope",
    )
    return {
        "status": "PASS",
        "run_id": run.name,
        "source_commit": expected_commit,
        "verified_new_files": verified,
        "n_outcomes": 80,
        "route": route,
    }


if __name__ == "__main__":
    print(json.dumps(verify(sys.argv[1], sys.argv[2]), sort_keys=True, allow_nan=False))
