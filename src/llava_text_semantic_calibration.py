"""Pure protocol and replay logic for the LLaVA text semantic calibration."""

from __future__ import annotations

import math

import numpy as np

OBJECTS = ("red square", "blue circle")
SENTENCE_PAIRS = (
    ("The scene contains a {object}.", "The scene contains no {object}."),
    ("A {object} is present.", "A {object} is absent."),
    (
        "It is false that the {object} is absent.",
        "It is false that the {object} is present.",
    ),
    (
        "The list is complete. Listed objects: {object}, green triangle.",
        "The list is complete. Listed objects: green triangle.",
    ),
)
SUFFIXES = {
    "Y": "Answer yes or no.",
    "incumbent-A": (
        "Answer A if the finding is present and B if it is absent. Reply with A or B only."
    ),
    "incumbent-B": (
        "Answer B if the finding is present and A if it is absent. Reply with A or B only."
    ),
    "candidate-A": (
        "Choose the correct option.\n"
        "A. The finding is present.\n"
        "B. The finding is absent.\n"
        "Reply with A or B only."
    ),
    "candidate-B": (
        "Choose the correct option.\n"
        "A. The finding is absent.\n"
        "B. The finding is present.\n"
        "Reply with A or B only."
    ),
}
CONDITIONS = (
    {"name": "Y", "encoding": "Y", "candidate": False, "suffix": SUFFIXES["Y"]},
    {
        "name": "incumbent-A",
        "encoding": "A",
        "candidate": False,
        "suffix": SUFFIXES["incumbent-A"],
    },
    {
        "name": "incumbent-B",
        "encoding": "B",
        "candidate": False,
        "suffix": SUFFIXES["incumbent-B"],
    },
    {
        "name": "candidate-A",
        "encoding": "A",
        "candidate": True,
        "suffix": SUFFIXES["candidate-A"],
    },
    {
        "name": "candidate-B",
        "encoding": "B",
        "candidate": True,
        "suffix": SUFFIXES["candidate-B"],
    },
)
PINNED_TOKEN_GROUPS = {
    "Y": ([3869, 4874, 22483], [694, 1939, 11698]),
    "A": ([319], [350]),
    "B": ([319], [350]),
}


def descriptions():
    """Return the sixteen fixed descriptions in registered order."""
    result = []
    for object_index, object_name in enumerate(OBJECTS):
        prefix = f"The target finding is a {object_name}."
        for pair_index, pair in enumerate(SENTENCE_PAIRS, start=1):
            for expected_present, sentence in ((True, pair[0]), (False, pair[1])):
                polarity = "positive" if expected_present else "negative"
                result.append(
                    {
                        "case_id": f"object-{object_index + 1}-pair-{pair_index}-{polarity}",
                        "object": object_name,
                        "pair": pair_index,
                        "expected_present": expected_present,
                        "expected_sign": 1 if expected_present else -1,
                        "description": f"{prefix} {sentence.format(object=object_name)}",
                    }
                )
    return result


def cases():
    """Return all eighty fixed case-condition identities in registered order."""
    result = []
    for case in descriptions():
        for condition in CONDITIONS:
            user_prompt = (
                f"{case['description']} Is the finding present? {condition['suffix']}"
            )
            result.append(
                {
                    **case,
                    "condition": condition["name"],
                    "encoding": condition["encoding"],
                    "candidate": condition["candidate"],
                    "user_prompt": user_prompt,
                }
            )
    return result


def token_metadata(encoding):
    """Return pinned raw and semantic token groups for one encoding."""
    first, second = PINNED_TOKEN_GROUPS[encoding]
    present, absent = (second, first) if encoding == "B" else (first, second)
    return {
        "raw_first": list(first),
        "raw_second": list(second),
        "present": list(present),
        "absent": list(absent),
        "candidate_ids": sorted(set(first + second)),
    }


def validate_token_metadata(tokens):
    expected = {condition["name"]: token_metadata(condition["encoding"]) for condition in CONDITIONS}
    if tokens != expected:
        raise ValueError("token groups differ from the pinned singleton identities")
    return True


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def derive_record(identity, rendered_prompt, candidate_logits, log_partition, token_ids):
    """Derive every decision score from retained final-position float32 logits."""
    if not isinstance(rendered_prompt, str) or not rendered_prompt:
        raise ValueError("rendered prompt must be preserved")
    metadata = token_metadata(identity["encoding"])
    if token_ids != metadata["candidate_ids"]:
        raise ValueError("candidate token identities differ from the pinned groups")
    logits = np.asarray(candidate_logits)
    if logits.dtype != np.float32 or logits.shape != (len(token_ids),) or not np.isfinite(logits).all():
        raise ValueError("candidate logits must be a complete finite float32 vector")
    partition = _finite_number(log_partition, "log partition")
    if partition < float(np.max(logits)):
        raise ValueError("log partition cannot be below a candidate logit")
    positions = {token_id: index for index, token_id in enumerate(token_ids)}
    first = np.asarray([logits[positions[token_id]] for token_id in metadata["raw_first"]])
    second = np.asarray([logits[positions[token_id]] for token_id in metadata["raw_second"]])
    raw = float(np.max(first) - np.max(second))
    semantic = -raw if identity["encoding"] == "B" else raw
    candidate_lse = float(np.logaddexp.reduce(logits.astype(np.float64)))
    answer_mass = float(np.exp(candidate_lse - partition))
    if not math.isfinite(answer_mass) or not 0 <= answer_mass <= 1 + 1e-6:
        raise ValueError("answer-token mass is invalid")
    lse = None
    if identity["encoding"] == "Y":
        lse = float(
            np.logaddexp.reduce(first.astype(np.float64))
            - np.logaddexp.reduce(second.astype(np.float64))
        )
    oriented = semantic * identity["expected_sign"]
    lse_oriented = None if lse is None else lse * identity["expected_sign"]
    return {
        **identity,
        "rendered_prompt": rendered_prompt,
        "candidate_token_ids": list(token_ids),
        "candidate_logits": logits.tolist(),
        "log_partition": partition,
        "answer_token_mass": min(answer_mass, 1.0),
        "raw_margin": raw,
        "semantic_margin": semantic,
        "oriented_margin": oriented,
        "primary_sign_correct": bool(oriented > 0),
        "lse_margin": lse,
        "lse_oriented_margin": lse_oriented,
        "lse_sign_correct": None if lse_oriented is None else bool(lse_oriented > 0),
    }


def validate_outcomes(outcomes):
    """Validate identity/order and replay every score from saved logits."""
    expected = cases()
    if not isinstance(outcomes, list) or len(outcomes) != 80:
        raise ValueError("exactly eighty outcomes are required")
    replayed = []
    for record, identity in zip(outcomes, expected, strict=True):
        identity_keys = set(identity)
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError("calibration outcome identity or order mismatch")
        derived = derive_record(
            identity,
            record.get("rendered_prompt"),
            np.asarray(record.get("candidate_logits"), dtype=np.float32),
            record.get("log_partition"),
            record.get("candidate_token_ids"),
        )
        if set(record) != set(derived):
            raise ValueError("calibration outcome schema mismatch")
        for key in set(derived) - identity_keys - {"candidate_logits"}:
            if record[key] != derived[key]:
                raise ValueError("calibration derived score mismatch: " + key)
        if not np.array_equal(
            np.asarray(record["candidate_logits"], dtype=np.float32),
            np.asarray(derived["candidate_logits"], dtype=np.float32),
        ):
            raise ValueError("saved candidate logits changed")
        replayed.append(derived)
    return replayed


def route(outcomes):
    """Apply the exhaustive three-way terminal route with strict signs."""
    outcomes = validate_outcomes(outcomes)
    yes = [record for record in outcomes if record["condition"] == "Y"]
    candidate = [record for record in outcomes if record["candidate"]]
    y_pass = all(record["primary_sign_correct"] and record["lse_sign_correct"] for record in yes)
    candidate_pass = all(record["primary_sign_correct"] for record in candidate)
    if not y_pass:
        return "comprehension_unresolved"
    if candidate_pass:
        return "candidate_eligible"
    return "yes_no_route"


def summarize(outcomes):
    outcomes = validate_outcomes(outcomes)
    conditions = []
    paired = []
    for condition in CONDITIONS:
        selected = [record for record in outcomes if record["condition"] == condition["name"]]
        errors = [record["case_id"] for record in selected if not record["primary_sign_correct"]]
        lse_errors = [
            record["case_id"]
            for record in selected
            if record["lse_sign_correct"] is False
        ]
        conditions.append(
            {
                "condition": condition["name"],
                "primary_sign_errors": errors,
                "lse_sign_errors": lse_errors,
                "minimum_oriented_margin": min(record["oriented_margin"] for record in selected),
                "minimum_lse_oriented_margin": (
                    min(record["lse_oriented_margin"] for record in selected)
                    if condition["encoding"] == "Y"
                    else None
                ),
            }
        )
        for object_name in OBJECTS:
            for pair_index in range(1, 5):
                positive = next(
                    record
                    for record in selected
                    if record["object"] == object_name
                    and record["pair"] == pair_index
                    and record["expected_present"]
                )
                negative = next(
                    record
                    for record in selected
                    if record["object"] == object_name
                    and record["pair"] == pair_index
                    and not record["expected_present"]
                )
                paired.append(
                    {
                        "condition": condition["name"],
                        "object": object_name,
                        "pair": pair_index,
                        "positive_minus_negative_semantic_margin": (
                            positive["semantic_margin"] - negative["semantic_margin"]
                        ),
                    }
                )
    return {
        "status": "OBSERVED",
        "gate": "llava-text-semantic-calibration",
        "n_descriptions": 16,
        "n_conditions": 5,
        "n_outcomes": 80,
        "conditions": conditions,
        "paired_positive_minus_negative": paired,
        "route": route(outcomes),
    }
