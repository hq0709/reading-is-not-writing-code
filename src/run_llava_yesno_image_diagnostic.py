"""Prepare, execute, and replay the registered LLaVA yes/no image diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import time
from pathlib import Path

import numpy as np

from src import llava_yesno_image_diagnostic as core
from src import run_llava_paired_opportunity as accepted
from src import run_llava_readout_diagnostic as parent
from src import run_llava_validation_opportunity as pilot
from src.qwen_answer_encoding import STATEMENTS

PARENT_RUN_ID = "20260905T211508Z-d24f2a7fe05e-llava-readout"
PARENT_COMMIT = "d24f2a7fe05eaddde8139dc9c3a64f7f8c2bc027"
PARENT_COHORT_MANIFEST_SHA256 = "2246b418971f9b85557960faf162e132bdaadce1b1d69b2d012d9c69cdcf46fe"
PARENT_INTERNAL_RECEIPT = (
    "state/llava-readout-diagnostic-preflight-internal-20260905T211744Z/receipt.json"
)
PARENT_REVIEW_RECEIPT = (
    "/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T211847245097Z.json"
)
ARIS_FULL_SHA = "94d8093ed21d20a790830318190095b9f5036ce8"
REVIEWER_WRAPPER_SHA256 = "2ebcbc89773d5913b8c00b7e0bf111d5ddaad0db4e178ef8405b90d150f0813d"
PILOT_RUN_ID = parent.PILOT_RUN_ID
PILOT_COMMIT = parent.PILOT_COMMIT
SUMMARY = "llava-yesno-image-diagnostic-summary"
PINNED_YES = [3869, 4874, 22483]
PINNED_NO = [694, 1939, 11698]
LSE_REPLAY_ATOL = 1e-5
TOKEN_MASS_REPLAY_ATOL = 3e-6


json_value = parent.json_value
write_json = parent.write_json
full_sha = parent.full_sha
paths = parent.paths


def protocol():
    return {
        "gate": "llava-yesno-image-diagnostic",
        "population": "retained validation index/donor allocation from the parent diagnostic",
        "allocation": [core.N_PAIRS, core.N_PAIRS, core.ALLOCATION_SEED],
        "conditions": list(core.CONDITIONS),
        "scoring": {
            "primary": "maximum yes singleton logit minus maximum no singleton logit",
            "secondary": "logsumexp(yes singleton logits) minus logsumexp(no singleton logits)",
            "probability": "sigmoid(semantic margin)",
        },
        "module": core.MODULE,
        "locus": "vis.last",
        "pooling": "mean over all 577 block-output tokens",
        "model_id": accepted.MODEL_ID,
        "model_revision": accepted.MODEL_REVISION,
        "batch_size": 16,
        "dtype": "bfloat16",
        "activation_storage": "float16",
        "bootstrap": [core.N_BOOTSTRAP, core.BOOTSTRAP_SEED],
        "simultaneous_interval": "unstudentized maximum absolute centered deviation, linear p95",
        "primary_names": list(core.PRIMARY_NAMES),
        "parent_run_id": PARENT_RUN_ID,
        "parent_commit": PARENT_COMMIT,
        "pilot_run_id": PILOT_RUN_ID,
        "pilot_commit": PILOT_COMMIT,
        "preprocessing": accepted.protocol_metadata()["preprocessing"],
    }


def _metadata(path):
    return dict(
        line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if "=" in line
    )


def _manifest(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        result[name.strip().lstrip("*").removeprefix("./")] = digest
    return result


def parent_receipt(root):
    run = root / "runs" / PARENT_RUN_ID
    meta = _metadata(run / "metadata.env")
    expected = {
        "RUN_ID": PARENT_RUN_ID,
        "SOURCE_COMMIT": PARENT_COMMIT,
        "COMMAND_STATUS": "1",
        "DISPATCHER_STATUS": "1",
        "CLEANUP_STATUS": "0",
        "ABORT_SIGNAL": "none",
    }
    internal = accepted.read_json(root / PARENT_INTERNAL_RECEIPT)
    review = accepted.read_json(PARENT_REVIEW_RECEIPT)
    manifest = _manifest(run / "SHA256SUMS")
    if (
        any(meta.get(key) != value for key, value in expected.items())
        or (run / "command_exit_status").read_text().strip() != "1"
        or (run / "exit_status").read_text().strip() != "1"
        or manifest.get("artifacts/cohort.json") != PARENT_COHORT_MANIFEST_SHA256
        or internal.get("status") != "VERIFIED_FAILED_PREFLIGHT"
        or internal.get("failure_class") != "MEASUREMENT"
        or internal.get("run_id") != PARENT_RUN_ID
        or internal.get("source_commit") != PARENT_COMMIT
        or internal.get("new_scientific_artifacts") != 0
        or internal.get("decision") != "BLOCKED"
        or review.get("valid") is not True
        or review.get("readOnly") is not True
        or review.get("modelsObserved") != ["claude-fable-5-1"]
        or review.get("effort") != "medium"
        or review.get("arisFullSha") != ARIS_FULL_SHA
        or review.get("wrapperSha256") != REVIEWER_WRAPPER_SHA256
        or review.get("checkoutBefore") != {"head": PARENT_COMMIT, "status": ""}
        or review.get("checkoutAfter") != {"head": PARENT_COMMIT, "status": ""}
    ):
        raise ValueError("accepted parent allocation receipt mismatch")
    return {
        "run_id": PARENT_RUN_ID,
        "source_commit": PARENT_COMMIT,
        "terminal": meta,
        "cohort_path": str(run / "artifacts/cohort.json"),
        "cohort_manifest_sha256": PARENT_COHORT_MANIFEST_SHA256,
        "internal_receipt": str(root / PARENT_INTERNAL_RECEIPT),
        "review_receipt": PARENT_REVIEW_RECEIPT,
    }


def load_sources(root):
    source, accepted_prepared, assignments, reconstructed = parent.load_sources(root)
    receipt = parent_receipt(root)
    retained = accepted.read_json(receipt["cohort_path"])
    core.validate_allocation(retained, check_files=True)
    if retained != reconstructed:
        raise ValueError("retained allocation differs from registered metadata reconstruction")
    return {**source, "parent_allocation": receipt}, accepted_prepared, assignments, retained


def prepare(data_root, out, source_commit):
    started = time.perf_counter()
    root, out = paths(data_root, out, source_commit)
    if any((out / name).exists() for name in ("cohort.json", "prepared.json", "prepared.npz")):
        raise ValueError("yes/no diagnostic preparation artifacts already exist")
    source, accepted_prepared, assignments, allocation = load_sources(root)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: accepted_prepared[key]
        for key in (
            "projection",
            "scale",
            "train_mean",
            "coefficients",
            "control_coefficients",
            "control_intercepts",
        )
    }
    arrays["bootstrap_indices"] = core.bootstrap_indices()
    np.savez_compressed(out / "prepared.npz", **arrays)
    meta = {
        "source_commit": source_commit,
        "protocol": protocol(),
        "sources": source,
        "control_assignments": assignments,
        "cohort": allocation,
        "preparation_seconds": time.perf_counter() - started,
    }
    write_json(out / "cohort.json", allocation)
    write_json(out / "prepared.json", meta)
    return meta


def load_prepared(root, out, source_commit):
    expected_source, accepted_prepared, assignments, allocation = load_sources(root)
    meta = accepted.read_json(out / "prepared.json")
    cohort = accepted.read_json(out / "cohort.json")
    if (
        meta.get("source_commit") != source_commit
        or meta.get("protocol") != protocol()
        or meta.get("sources") != expected_source
        or meta.get("control_assignments") != assignments
        or meta.get("cohort") != allocation
        or cohort != allocation
        or not np.isfinite(meta.get("preparation_seconds", np.nan))
        or meta["preparation_seconds"] <= 0
    ):
        raise ValueError("yes/no diagnostic preparation identity mismatch")
    with np.load(out / "prepared.npz", allow_pickle=False) as archive:
        prepared = {key: archive[key] for key in archive.files}
    expected_keys = {
        "projection",
        "scale",
        "train_mean",
        "coefficients",
        "control_coefficients",
        "control_intercepts",
        "bootstrap_indices",
    }
    if set(prepared) != expected_keys:
        raise ValueError("yes/no prepared array schema mismatch")
    for key in expected_keys - {"bootstrap_indices"}:
        if not np.array_equal(prepared[key], accepted_prepared[key]):
            raise ValueError("frozen reader parameter mismatch: " + key)
    core.validate_bootstrap(prepared["bootstrap_indices"])
    return meta, allocation, prepared


def token_metadata(parent_tokens):
    result = {spec["name"]: parent_tokens[spec["name"]] for spec in core.CONDITIONS}
    validate_token_metadata(result)
    return result


def validate_token_metadata(tokens):
    if set(tokens) != {spec["name"] for spec in core.CONDITIONS}:
        raise ValueError("token metadata must contain the two yes/no conditions")
    expected = {
        "raw_first": PINNED_YES,
        "raw_second": PINNED_NO,
        "present": PINNED_YES,
        "absent": PINNED_NO,
        "candidate_ids": sorted(PINNED_YES + PINNED_NO),
    }
    if any(tokens[spec["name"]] != expected for spec in core.CONDITIONS):
        raise ValueError("yes/no token groups differ from the pinned singleton IDs")
    return True


def expected_mapping_cases():
    return [
        {
            "case": case_index,
            "statement": statement,
            "prompt": f"{statement} Is the finding present? {core.SUFFIX}",
            "expected_present": expected,
        }
        for case_index, (statement, expected) in enumerate(STATEMENTS)
    ]


def consumed_prompts():
    result = []
    for condition, spec in enumerate(core.CONDITIONS):
        parent_index = condition * 3
        inherited = parent.core.CONDITIONS[parent_index]
        if inherited.get("prompt") != spec["prompt"] or inherited.get("wording") != spec["wording"]:
            raise ValueError("inherited scorer prompt differs from the registered yes/no prompt")
        result.append(
            {
                "condition": spec["name"],
                "parent_condition_index": parent_index,
                "wording": inherited["wording"],
                "prompt": inherited["prompt"],
            }
        )
    return result


def load_scorer(source, gpu):
    prompts = consumed_prompts()
    parent_score, parent_tokens, runtime = parent.load_scorer(source, gpu)
    tokens = token_metadata(parent_tokens)

    def score(condition, rows=(), capture=False, prompt_override=None):
        if condition not in (0, 1):
            raise ValueError("yes/no condition must be zero or one")
        return parent_score(condition * 3, rows, capture=capture, prompt_override=prompt_override)

    return score, tokens, runtime, prompts


def mapping_cases(score):
    cases = []
    for expected in expected_mapping_cases():
        values = score(0, prompt_override=expected["prompt"])
        cases.append(
            {
                **expected,
                "semantic_margin": float(values["semantic_margin"][0]),
                "lse_margin": float(values["lse_margin"][0]),
            }
        )
    return cases


def budget(pilot_seconds, elapsed_seconds, preparation_seconds):
    if (
        not np.isfinite([pilot_seconds, elapsed_seconds, preparation_seconds]).all()
        or not 0 < pilot_seconds <= elapsed_seconds
        or preparation_seconds <= 0
    ):
        raise ValueError("invalid yes/no diagnostic timing measurements")
    total_elapsed = preparation_seconds + elapsed_seconds
    projected = total_elapsed + 2802 * pilot_seconds / 512 + 600
    return {
        "pilot_equivalents": 512,
        "pilot_seconds": pilot_seconds,
        "gpu_phase_elapsed_seconds": elapsed_seconds,
        "preparation_seconds": preparation_seconds,
        "elapsed_preparation_loading_preflight_and_pilot": total_elapsed,
        "scientific_outcomes": 2802,
        "analysis_allowance_seconds": 600,
        "projected_seconds": projected,
        "limit_seconds": 2700,
        "passed": bool(projected <= 2700),
    }


def validate_preflight(flight, source_commit, model_source, preparation_seconds):
    if (
        flight.get("source_commit") != source_commit
        or flight.get("passed") is not True
        or flight.get("validation_row_ids") != accepted.VALIDATION_ROW_IDS
        or flight.get("module") != core.MODULE
        or flight.get("model_source") != model_source
        or flight.get("forwarded_no_cls_exact") is not True
        or flight.get("pooling") != "mean over all 577 block-output tokens"
        or flight.get("clean_repeat_exact") is not True
        or flight.get("consumed_prompts") != consumed_prompts()
    ):
        raise ValueError("yes/no preflight identity or consumed-locus check failed")
    validate_token_metadata(flight.get("token_ids", {}))
    expected = expected_mapping_cases()
    if len(flight.get("mapping_cases", [])) != 4:
        raise ValueError("four generic yes/no mapping cases are required")
    for case, identity in zip(flight["mapping_cases"], expected, strict=True):
        if set(case) != set(identity) | {"semantic_margin", "lse_margin"} or any(
            case.get(key) != value for key, value in identity.items()
        ):
            raise ValueError("registered yes/no semantic mapping identity mismatch")
        sign = 2 * identity["expected_present"] - 1
        if (
            not np.isfinite([case["semantic_margin"], case["lse_margin"]]).all()
            or case["semantic_margin"] * sign <= 0
            or case["lse_margin"] * sign <= 0
        ):
            raise ValueError("yes/no semantic orientation failed")
    timing = flight.get("throughput", {})
    expected_timing = budget(
        timing.get("pilot_seconds", np.nan),
        timing.get("gpu_phase_elapsed_seconds", np.nan),
        preparation_seconds,
    )
    if timing != expected_timing or not timing["passed"]:
        raise ValueError("registered 45-minute yes/no diagnostic projection failed")
    return True


def validate_text_only(records):
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("two text-only outcomes are required")
    for record, spec in zip(records, core.CONDITIONS, strict=True):
        expected_keys = {
            "condition",
            "wording",
            "prompt",
            "raw_margin",
            "semantic_margin",
            "probability",
            "answer_token_mass",
            "lse_margin",
            "candidate_logits",
            "log_partition",
            "constant_auroc_against_index_labels",
            "subtracting_constant_preserves_image_auroc",
        }
        if (
            set(record) != expected_keys
            or record["condition"] != spec["name"]
            or record["wording"] != spec["wording"]
            or record["prompt"] != spec["prompt"]
            or record["raw_margin"] != record["semantic_margin"]
            or record["constant_auroc_against_index_labels"] != 0.5
            or record["subtracting_constant_preserves_image_auroc"] is not True
        ):
            raise ValueError("text-only yes/no condition identity or schema mismatch")
        values = [
            record[key]
            for key in (
                "raw_margin",
                "semantic_margin",
                "probability",
                "answer_token_mass",
                "lse_margin",
                "log_partition",
            )
        ]
        logits = np.asarray(record["candidate_logits"], dtype=np.float32)
        candidate_ids = sorted(PINNED_YES + PINNED_NO)
        positions = {token_id: index for index, token_id in enumerate(candidate_ids)}
        yes = [positions[value] for value in PINNED_YES]
        no = [positions[value] for value in PINNED_NO]
        raw = float(logits[yes].max() - logits[no].max()) if logits.shape == (6,) else np.nan
        lse = (
            float(np.logaddexp.reduce(logits[yes]) - np.logaddexp.reduce(logits[no]))
            if logits.shape == (6,)
            else np.nan
        )
        mass = (
            float(np.exp(np.logaddexp.reduce(logits) - record["log_partition"]))
            if logits.shape == (6,)
            else np.nan
        )
        if (
            not np.isfinite(values).all()
            or not np.isfinite(logits).all()
            or not 0 <= record["answer_token_mass"] <= 1
            or not np.isclose(record["raw_margin"], raw, atol=1e-6, rtol=0)
            or not np.isclose(record["semantic_margin"], raw, atol=1e-6, rtol=0)
            or not np.isclose(record["lse_margin"], lse, atol=1e-6, rtol=0)
            or not np.isclose(record["answer_token_mass"], mass, atol=1e-6, rtol=1e-6)
            or not np.isclose(
                record["probability"],
                core.sigmoid(record["semantic_margin"]),
                atol=1e-12,
                rtol=1e-12,
            )
        ):
            raise ValueError("invalid text-only yes/no score")
    return True


def require_native_replay(observed, replayed, name, atol=1e-6, rtol=0):
    if not np.allclose(observed, replayed, atol=atol, rtol=rtol):
        raise ValueError("candidate logits do not reproduce " + name)


def run(data_root, out, source_commit, gpu, preflight_only=False):
    root, out = paths(data_root, out, source_commit)
    if any((out / name).exists() for name in ("preflight.json", "per-image.csv", "scores.npz")):
        raise ValueError("yes/no diagnostic GPU artifacts already exist")
    meta, allocation, _prepared = load_prepared(root, out, source_commit)
    started = time.perf_counter()
    flight = {
        "passed": False,
        "source_commit": source_commit,
        "validation_row_ids": accepted.VALIDATION_ROW_IDS,
    }
    try:
        score, tokens, runtime, prompts = load_scorer(meta["sources"]["model"], gpu)
        validation = pilot.load_sources(root)[3]["preflight"]
        clean = score(0, validation, capture=True)
        repeat = score(0, validation, capture=True)
        repeat_keys = (
            "raw_margin",
            "semantic_margin",
            "probability",
            "answer_token_mass",
            "candidate_logits",
            "log_partition",
            "lse_margin",
            "activation",
        )
        flight.update(
            {
                "model_source": meta["sources"]["model"]["model_source"],
                "module": core.MODULE,
                "pooling": "mean over all 577 block-output tokens",
                "runtime": runtime,
                "consumed_prompts": prompts,
                "token_ids": tokens,
                "mapping_cases": mapping_cases(score),
                "forwarded_no_cls_exact": clean["forwarded_no_cls_exact"]
                and repeat["forwarded_no_cls_exact"],
                "clean_repeat_exact": all(
                    np.array_equal(clean[key], repeat[key]) for key in repeat_keys
                ),
                "clean_semantic_margin": clean["semantic_margin"].tolist(),
            }
        )
        pilot_start = time.perf_counter()
        for index in range(32):
            score(index % 2, validation)
        flight["throughput"] = budget(
            time.perf_counter() - pilot_start,
            time.perf_counter() - started,
            meta["preparation_seconds"],
        )
        flight["passed"] = flight["throughput"]["passed"]
        validate_preflight(
            flight,
            source_commit,
            meta["sources"]["model"]["model_source"],
            meta["preparation_seconds"],
        )
    except Exception as error:
        flight.update(passed=False, error=f"{type(error).__name__}: {error}")
        write_json(out / "preflight.json", flight)
        raise
    write_json(out / "preflight.json", flight)
    if preflight_only:
        return flight

    temporary = out / "per-image.csv.tmp"
    stored, activation_batches = {}, []
    with temporary.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=core.FIELDS)
        writer.writeheader()
        for condition, spec in enumerate(core.CONDITIONS):
            for role in ("index", "donor"):
                logits, partitions = [], []
                rows = allocation[role]
                for start in range(0, core.N_PAIRS, 16):
                    chunk = rows[start : start + 16]
                    values = score(condition, chunk, capture=(condition == 0 and role == "index"))
                    logits.append(values["candidate_logits"])
                    partitions.append(values["log_partition"])
                    if condition == 0 and role == "index":
                        if not values["forwarded_no_cls_exact"]:
                            raise ValueError("scientific capture is not consumed by the connector")
                        activation_batches.append(values["activation"])
                    for offset, row in enumerate(chunk):
                        pair_index = start + offset
                        writer.writerow(
                            {
                                "source_commit": source_commit,
                                "condition": spec["name"],
                                "wording": spec["wording"],
                                "prompt": spec["prompt"],
                                "image_role": role,
                                "pair_index": pair_index,
                                "patient_id": row["patient_id"],
                                "row_id": row["row_id"],
                                "image_path": row["image_path"],
                                "index_label": allocation["index"][pair_index]["Effusion"],
                                "source_label": row["Effusion"],
                                "raw_margin": values["raw_margin"][offset],
                                "semantic_margin": values["semantic_margin"][offset],
                                "probability": values["probability"][offset],
                                "answer_token_mass": values["answer_token_mass"][offset],
                                "lse_margin": values["lse_margin"][offset],
                            }
                        )
                    stream.flush()
                key = spec["name"] + "__" + role
                stored["candidate_logits__" + key] = np.concatenate(logits).astype(np.float32)
                stored["log_partition__" + key] = np.concatenate(partitions).astype(np.float32)
    with temporary.open(newline="", encoding="utf-8") as stream:
        core.validate_grid(csv.DictReader(stream), allocation, source_commit)
    activation = np.concatenate(activation_batches)
    if activation.shape != (core.N_PAIRS, 1024) or activation.dtype != np.float16:
        raise ValueError("complete float16 index activation capture is required")
    stored["reader_raw_activation"] = activation
    with (out / "scores.npz.tmp").open("xb") as stream:
        np.savez_compressed(stream, **stored)
    (out / "scores.npz.tmp").replace(out / "scores.npz")
    temporary.replace(out / "per-image.csv")

    text_only = []
    for condition, spec in enumerate(core.CONDITIONS):
        values = score(condition)
        text_only.append(
            {
                "condition": spec["name"],
                "wording": spec["wording"],
                "prompt": spec["prompt"],
                **{
                    key: float(values[key][0])
                    for key in (
                        "raw_margin",
                        "semantic_margin",
                        "probability",
                        "answer_token_mass",
                        "lse_margin",
                    )
                },
                "candidate_logits": values["candidate_logits"][0].tolist(),
                "log_partition": float(values["log_partition"][0]),
                "constant_auroc_against_index_labels": 0.5,
                "subtracting_constant_preserves_image_auroc": True,
            }
        )
    write_json(out / "text-only.json", text_only)
    return {"n_image_outcomes": 2800, "n_text_only_outcomes": 2}


def _interval(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return (
        np.percentile(values, [2.5, 97.5], method="linear").tolist()
        if len(values)
        else [None, None]
    )


def summarize(data_root, out, source_commit, summary_out=None):
    root, out = paths(data_root, out, source_commit)
    meta, allocation, prepared = load_prepared(root, out, source_commit)
    flight = accepted.read_json(out / "preflight.json")
    validate_preflight(
        flight,
        source_commit,
        meta["sources"]["model"]["model_source"],
        meta["preparation_seconds"],
    )
    text_only = accepted.read_json(out / "text-only.json")
    validate_text_only(text_only)
    with (out / "per-image.csv").open(newline="", encoding="utf-8") as stream:
        scores = core.validate_grid(csv.DictReader(stream), allocation, source_commit)
    with np.load(out / "scores.npz", allow_pickle=False) as archive:
        stored = {key: archive[key] for key in archive.files}
    expected_keys = {"reader_raw_activation"}
    for spec in core.CONDITIONS:
        for role in ("index", "donor"):
            expected_keys.update(
                {
                    "candidate_logits__" + spec["name"] + "__" + role,
                    "log_partition__" + spec["name"] + "__" + role,
                }
            )
    if set(stored) != expected_keys:
        raise ValueError("yes/no candidate-logit archive schema mismatch")
    token = flight["token_ids"]["0Y"]
    positions = {token_id: i for i, token_id in enumerate(token["candidate_ids"])}
    yes = [positions[value] for value in PINNED_YES]
    no = [positions[value] for value in PINNED_NO]
    for condition, spec in enumerate(core.CONDITIONS):
        for role_index, role in enumerate(("index", "donor")):
            key = spec["name"] + "__" + role
            logits = core.finite(
                stored["candidate_logits__" + key],
                (core.N_PAIRS, len(token["candidate_ids"])),
                "candidate logits",
            )
            partition = core.finite(
                stored["log_partition__" + key], (core.N_PAIRS,), "log partition"
            )
            raw = logits[:, yes].max(1) - logits[:, no].max(1)
            lse = np.logaddexp.reduce(logits[:, yes], axis=1) - np.logaddexp.reduce(
                logits[:, no], axis=1
            )
            mass = np.exp(np.logaddexp.reduce(logits, axis=1) - partition)
            for observed, replayed, name, atol, rtol in (
                (scores["raw_margin"][condition, role_index], raw, "raw margin", 1e-6, 0),
                (scores["semantic_margin"][condition, role_index], raw, "semantic margin", 1e-6, 0),
                (
                    scores["lse_margin"][condition, role_index],
                    lse,
                    "lse margin",
                    LSE_REPLAY_ATOL,
                    0,
                ),
                (
                    scores["answer_token_mass"][condition, role_index],
                    mass,
                    "token mass",
                    TOKEN_MASS_REPLAY_ATOL,
                    0,
                ),
            ):
                require_native_replay(observed, replayed, name, atol, rtol)

    labels = np.asarray([row["Effusion"] for row in allocation["index"]], dtype=np.int8)
    donor_labels = np.asarray([row["Effusion"] for row in allocation["donor"]], dtype=np.int8)
    indices = prepared["bootstrap_indices"]
    answer_arrays, primary = core.answer_statistics(
        scores["semantic_margin"], labels, donor_labels, indices
    )
    reader_arrays, reader = core.reader_statistics(
        stored["reader_raw_activation"],
        prepared,
        meta["control_assignments"],
        allocation["index"],
        indices,
    )
    cells = []
    for condition, spec in enumerate(core.CONDITIONS):
        cells.append(
            {
                "condition": spec["name"],
                "prompt": spec["prompt"],
                "real_auroc": float(answer_arrays["auroc"][condition, 0]),
                "real_auroc_ci95": _interval(answer_arrays["bootstrap_auroc"][condition, 0]),
                "donor_auroc_against_index_label": float(answer_arrays["auroc"][condition, 1]),
                "donor_auroc_ci95": _interval(answer_arrays["bootstrap_auroc"][condition, 1]),
                "donor_auroc_against_donor_label": float(
                    answer_arrays["donor_own_label_auroc"][condition]
                ),
                "image_advantage": float(answer_arrays["image_advantage"][condition]),
                "image_advantage_ci95": _interval(
                    answer_arrays["bootstrap_image_advantage"][:, condition]
                ),
                "real_mean_margin": float(scores["semantic_margin"][condition, 0].mean()),
                "donor_mean_margin": float(scores["semantic_margin"][condition, 1].mean()),
                "real_brier": float(np.mean((scores["probability"][condition, 0] - labels) ** 2)),
                "donor_brier_against_index_label": float(
                    np.mean((scores["probability"][condition, 1] - labels) ** 2)
                ),
                "real_mean_answer_token_mass": float(
                    scores["answer_token_mass"][condition, 0].mean()
                ),
                "donor_mean_answer_token_mass": float(
                    scores["answer_token_mass"][condition, 1].mean()
                ),
            }
        )
    counts = core.multiplicities(indices)
    lse = []
    for condition, spec in enumerate(core.CONDITIONS):
        real, real_draw = core.auc_point_draws(scores["lse_margin"][condition, 0], labels, counts)
        donor, donor_draw = core.auc_point_draws(scores["lse_margin"][condition, 1], labels, counts)
        max_g = answer_arrays["image_advantage"][condition]
        difference_draw = (
            real_draw - donor_draw - answer_arrays["bootstrap_image_advantage"][:, condition]
        )
        lse.append(
            {
                "condition": spec["name"],
                "real_auroc": real,
                "donor_auroc_against_index_label": donor,
                "image_advantage": real - donor,
                "image_advantage_ci95": _interval(real_draw - donor_draw),
                "image_advantage_difference_from_max_variant": real - donor - max_g,
                "image_advantage_difference_from_max_variant_ci95": _interval(difference_draw),
            }
        )
    a1_minus_a0_draws = (
        answer_arrays["bootstrap_auroc"][1, 0] - answer_arrays["bootstrap_auroc"][0, 0]
    )
    result = {
        "source_commit": source_commit,
        "status": "OBSERVED",
        "gate": protocol()["gate"],
        "n_pairs": core.N_PAIRS,
        "n_image_outcomes": 2800,
        "n_text_only_outcomes": 2,
        "index_positive": int(labels.sum()),
        "donor_positive": int(donor_labels.sum()),
        "cells": cells,
        "a1_minus_a0": float(answer_arrays["auroc"][1, 0] - answer_arrays["auroc"][0, 0]),
        "a1_minus_a0_ci95": _interval(a1_minus_a0_draws),
        "primary_family": primary,
        "frozen_reader": reader,
        "same_logit_aggregation": lse,
        "text_only": text_only,
        "route": "evidence_synthesis",
        "protocol": protocol(),
        "cohort": allocation,
        "sources": meta["sources"],
        "preflight": flight,
    }
    arrays = {
        **prepared,
        **scores,
        **stored,
        **answer_arrays,
        **reader_arrays,
        "index_labels": labels,
        "donor_labels": donor_labels,
        "source_commit": np.asarray(source_commit),
        "route": np.asarray(result["route"]),
    }
    destination = out if summary_out is None else Path(summary_out).resolve()
    if destination == root or not destination.is_relative_to(root):
        raise ValueError("summary output must remain below the accepted data root")
    destination.mkdir(parents=True, exist_ok=True)
    for name in (SUMMARY + ".npz", SUMMARY + ".json"):
        if (destination / name).exists():
            raise ValueError("summary output already exists: " + name)
    np.savez_compressed(destination / (SUMMARY + ".npz"), **arrays)
    write_json(destination / (SUMMARY + ".json"), result)
    return result


def recover(data_root, out, source_commit, summary_out, validator_commit):
    root, out = paths(data_root, out, source_commit)
    destination = Path(summary_out).resolve()
    if destination.parent != root / "state" or destination.exists():
        raise ValueError("recovery output must be a fresh direct child of the state directory")
    validator_commit = full_sha(validator_commit)
    repository = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    upstream = subprocess.check_output(
        ["git", "rev-parse", "@{upstream}"], cwd=repository, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository, text=True
    )
    if head != validator_commit or upstream != validator_commit or status:
        raise ValueError("recovery validator commit must be clean and pushed")
    run = out.parent
    metadata = _metadata(run / "metadata.env")
    manifest = _manifest(run / "SHA256SUMS")
    native = (
        "cohort.json",
        "prepared.json",
        "prepared.npz",
        "preflight.json",
        "text-only.json",
        "per-image.csv",
        "scores.npz",
    )
    if (
        metadata.get("RUN_ID") != run.name
        or metadata.get("SOURCE_COMMIT") != source_commit
        or metadata.get("COMMAND_STATUS") != "1"
        or metadata.get("DISPATCHER_STATUS") != "1"
        or metadata.get("CLEANUP_STATUS") != "0"
        or metadata.get("ABORT_SIGNAL") != "none"
        or (run / "command_exit_status").read_text().strip() != "1"
        or (run / "exit_status").read_text().strip() != "1"
    ):
        raise ValueError("recovery source terminal receipt mismatch")
    for name in native:
        with (out / name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if manifest.get("artifacts/" + name) != digest:
            raise ValueError("recovery source manifest mismatch: " + name)
    result = summarize(data_root, out, source_commit, destination)
    receipt = {
        "status": "RECOVERED_SUMMARY",
        "source_run_id": run.name,
        "source_commit": source_commit,
        "validator_commit": validator_commit,
        "scientific_execution_reused": True,
        "new_scientific_outcomes": 0,
        "failure_class": "IMPLEMENTATION",
        "correction": "float32 cross-library reduction replay tolerances",
        "lse_replay_atol": LSE_REPLAY_ATOL,
        "token_mass_replay_atol": TOKEN_MASS_REPLAY_ATOL,
        "summary_status": result["status"],
    }
    write_json(destination / "recovery.json", receipt)
    with (destination / "SHA256SUMS").open("x", encoding="utf-8") as stream:
        for name in (SUMMARY + ".json", SUMMARY + ".npz", "recovery.json"):
            with (destination / name).open("rb") as artifact:
                digest = hashlib.file_digest(artifact, "sha256").hexdigest()
            stream.write(f"{digest}  {name}\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "summarize", "recover"):
        sub = commands.add_parser(command)
        sub.add_argument("--data-root", type=Path, required=True)
        sub.add_argument("--out", type=Path, required=True)
        sub.add_argument("--source-commit", type=full_sha, required=True)
        if command == "run":
            sub.add_argument("--gpu", type=int, required=True)
            sub.add_argument("--preflight-only", action="store_true")
        if command == "summarize":
            sub.add_argument("--summary-out", type=Path)
        if command == "recover":
            sub.add_argument("--summary-out", type=Path, required=True)
            sub.add_argument("--validator-commit", type=full_sha, required=True)
    args = vars(parser.parse_args(argv))
    return {"prepare": prepare, "run": run, "summarize": summarize, "recover": recover}[
        args.pop("command")
    ](**args)


if __name__ == "__main__":
    main()
