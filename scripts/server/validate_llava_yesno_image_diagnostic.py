"""Independent terminal replay for the LLaVA yes/no image diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.server import validate_llava_readout_diagnostic as base

N = 700
B = 10_000
LSE_REPLAY_ATOL = 1e-5
TOKEN_MASS_REPLAY_ATOL = 3e-6
SUMMARY = "llava-yesno-image-diagnostic-summary"
PARENT_RUN_ID = "20260905T211508Z-d24f2a7fe05e-llava-readout"
PARENT_COMMIT = "d24f2a7fe05eaddde8139dc9c3a64f7f8c2bc027"
PARENT_COHORT_SHA256 = "2246b418971f9b85557960faf162e132bdaadce1b1d69b2d012d9c69cdcf46fe"
CONDITIONS = ("0Y", "1Y")
WORDINGS = (
    "Is there a pleural effusion in this chest radiograph?",
    "Does this chest radiograph show a pleural effusion?",
)
SUFFIX = "Answer yes or no."
STATEMENTS = (
    ("The finding is present.", 1),
    ("The finding is absent.", 0),
    ("The finding is explicitly reported as present.", 1),
    ("The finding is explicitly reported as absent.", 0),
)
YES = [3869, 4874, 22483]
NO = [694, 1939, 11698]
VALIDATION_ROW_IDS = base.VALIDATION_ROW_IDS
PRIMARY = (
    "accepted_wording_discrimination_above_chance",
    "accepted_wording_image_advantage",
    "wording_dependent_image_advantage",
)
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MODEL_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
MODEL_SOURCE = (
    "/home/qingchan/data/concept-flow/models/huggingface/hub/"
    "models--llava-hf--llava-1.5-7b-hf/snapshots/"
    "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
)
ASSET = "llava15_7b"
ASSET_COMMIT = "ffd523c464c84417a93c5a6d0a34e5b74e55e76e"
ASSET_SOURCE_COMMIT = "a089458fab7cbfa5f46e66557b7b8cb51a8aab44"
SNAPSHOT_RELATIVE_PATH = (
    "hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
)
MODEL_RECEIPT = "/home/qingchan/data/concept-flow/models/huggingface/asset-receipt.json"
MODEL_RECEIPT_SHA256 = "1629f74a9cc03f6faa8dd8e3b6288baadf440172f60e8ba20f5980edf364caa9"
INPUT_VERIFICATION = (
    "/home/qingchan/data/concept-flow/runs/"
    "20260902T191411Z-ffd523c464c8-f99e2f39/artifacts/input-verification.json"
)
PILOT_RUN_ID = "20260905T093108Z-42a43207c848-llava-validation"
PILOT_COMMIT = "42a43207c848acfcadecf2f3e0bf866ea71d9dc7"
ARIS_FULL_SHA = "94d8093ed21d20a790830318190095b9f5036ce8"
REVIEWER_WRAPPER_SHA256 = "2ebcbc89773d5913b8c00b7e0bf111d5ddaad0db4e178ef8405b90d150f0813d"
PARENT_INTERNAL_RECEIPT = (
    "/home/qingchan/data/concept-flow/state/"
    "llava-readout-diagnostic-preflight-internal-20260905T211744Z/receipt.json"
)
PARENT_REVIEW_RECEIPT = (
    "/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T211847245097Z.json"
)


check = base.check
read = base.read
same = base.same
inventory = base.inventory
weights = base.weights
auc = base.auc


def read_path(path):
    return read(Path(path))


def validate_checkout(commit):
    commit = str(commit)
    check(len(commit) == 40, "terminal validator commit")
    repository = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    upstream = subprocess.check_output(
        ["git", "rev-parse", "@{upstream}"], cwd=repository, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository, text=True
    )
    check(head == commit and upstream == commit and not status, "terminal validator checkout")
    return commit


def expected_conditions():
    return [
        {"name": f"{index}Y", "wording": index, "prompt": f"{wording} {SUFFIX}"}
        for index, wording in enumerate(WORDINGS)
    ]


def expected_protocol():
    return {
        "gate": "llava-yesno-image-diagnostic",
        "population": "retained validation index/donor allocation from the parent diagnostic",
        "allocation": [700, 700, 20260916],
        "conditions": expected_conditions(),
        "scoring": {
            "primary": "maximum yes singleton logit minus maximum no singleton logit",
            "secondary": "logsumexp(yes singleton logits) minus logsumexp(no singleton logits)",
            "probability": "sigmoid(semantic margin)",
        },
        "module": "model.vision_tower.encoder.layers.22",
        "locus": "vis.last",
        "pooling": "mean over all 577 block-output tokens",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "batch_size": 16,
        "dtype": "bfloat16",
        "activation_storage": "float16",
        "bootstrap": [10_000, 20260917],
        "simultaneous_interval": "unstudentized maximum absolute centered deviation, linear p95",
        "primary_names": list(PRIMARY),
        "parent_run_id": PARENT_RUN_ID,
        "parent_commit": PARENT_COMMIT,
        "pilot_run_id": PILOT_RUN_ID,
        "pilot_commit": PILOT_COMMIT,
        "preprocessing": {
            "center_crop": [336, 336],
            "chat_template": "snapshot",
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_mode": "RGB",
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "patch_size": 14,
            "resample": 3,
            "rescale_factor": 0.00392156862745098,
            "shortest_edge": 336,
            "vision_feature_layer": -2,
            "vision_feature_select_strategy": "default",
        },
    }


def validate_protocol(protocol):
    check(protocol == expected_protocol(), "complete registered protocol identity")


def validate_model_source(model):
    check(
        model.get("model_id") == MODEL_ID
        and model.get("model_revision") == MODEL_REVISION
        and model.get("model_source") == MODEL_SOURCE,
        "accepted model identity",
    )
    asset = model.get("asset_receipt", {})
    inputs = model.get("input_verification", {})
    check(
        model.get("model_receipt") == MODEL_RECEIPT
        and model.get("input_verification_path") == INPUT_VERIFICATION
        and asset.get("asset") == ASSET
        and asset.get("repo_id") == MODEL_ID
        and asset.get("revision") == MODEL_REVISION
        and asset.get("snapshot_relative_path") == SNAPSHOT_RELATIVE_PATH
        and asset.get("source_commit") == ASSET_SOURCE_COMMIT
        and inputs.get("model_id") == MODEL_ID
        and inputs.get("model_revision") == MODEL_REVISION
        and inputs.get("source_commit") == ASSET_COMMIT
        and inputs.get("snapshot") == MODEL_SOURCE
        and inputs.get("model_receipt") == MODEL_RECEIPT
        and inputs.get("model_receipt_sha256") == MODEL_RECEIPT_SHA256
        and inputs.get("vision_feature_layer") == -2
        and inputs.get("verification_mode") == "accepted-asset-receipt",
        "accepted model receipt binding",
    )


def reader_scores(raw, prepared):
    """Replay the registered StandardScaler float32 arithmetic and each fit separately."""
    from sklearn.preprocessing import StandardScaler

    projected = raw.astype(np.float32) @ prepared["projection"].astype(np.float32)
    scaler = StandardScaler()
    scaler.mean_ = prepared["train_mean"]
    scaler.scale_ = prepared["scale"]
    scaler.n_features_in_ = 512
    features = scaler.transform(projected)
    clinical = features @ prepared["coefficients"][0]
    controls = np.stack(
        [
            features @ coefficient + intercept
            for coefficient, intercept in zip(
                prepared["control_coefficients"],
                prepared["control_intercepts"],
                strict=True,
            )
        ]
    )
    return projected, features, clinical, controls


def expected_mapping_cases():
    return [
        {
            "case": index,
            "statement": statement,
            "prompt": f"{statement} Is the finding present? {SUFFIX}",
            "expected_present": expected,
        }
        for index, (statement, expected) in enumerate(STATEMENTS)
    ]


def expected_consumed_prompts():
    return [
        {
            "condition": f"{index}Y",
            "parent_condition_index": index * 3,
            "wording": index,
            "prompt": f"{wording} {SUFFIX}",
        }
        for index, wording in enumerate(WORDINGS)
    ]


def validate_token_metadata(tokens):
    expected = {
        "raw_first": YES,
        "raw_second": NO,
        "present": YES,
        "absent": NO,
        "candidate_ids": sorted(YES + NO),
    }
    check(set(tokens) == set(CONDITIONS), "two-condition token identity")
    check(all(tokens[name] == expected for name in CONDITIONS), "pinned yes/no token identity")


def validate_preflight(preflight, preparation_seconds, commit, model_source):
    check(
        preflight.get("source_commit") == commit
        and preflight.get("model_source") == model_source
        and preflight.get("validation_row_ids") == list(VALIDATION_ROW_IDS)
        and preflight.get("module") == "model.vision_tower.encoder.layers.22"
        and preflight.get("pooling") == "mean over all 577 block-output tokens"
        and preflight.get("forwarded_no_cls_exact") is True
        and preflight.get("clean_repeat_exact") is True
        and preflight.get("consumed_prompts") == expected_consumed_prompts()
        and preflight.get("passed") is True,
        "preflight source and consumed-locus identity",
    )
    runtime = preflight.get("runtime", {})
    check(
        set(runtime)
        == {
            "gpu_name",
            "gpu_count",
            "chat_template",
            "module",
            "positions",
            "model_source",
        }
        and runtime["gpu_count"] == 1
        and "A100" in runtime["gpu_name"]
        and isinstance(runtime["chat_template"], str)
        and runtime["chat_template"]
        and runtime["module"] == "model.vision_tower.encoder.layers.22"
        and runtime["positions"] == "all"
        and runtime["model_source"] == model_source,
        "runtime model, device, template, and locus identity",
    )
    validate_token_metadata(preflight["token_ids"])
    expected = expected_mapping_cases()
    check(len(preflight["mapping_cases"]) == 4, "four generic mapping cases")
    for case, identity in zip(preflight["mapping_cases"], expected, strict=True):
        check(set(case) == set(identity) | {"semantic_margin", "lse_margin"}, "mapping schema")
        check(all(case.get(key) == value for key, value in identity.items()), "mapping identity")
        sign = 2 * identity["expected_present"] - 1
        check(
            np.isfinite([case["semantic_margin"], case["lse_margin"]]).all()
            and case["semantic_margin"] * sign > 0
            and case["lse_margin"] * sign > 0,
            "semantic and lse mapping signs",
        )
    timing = preflight["throughput"]
    total = preparation_seconds + timing["gpu_phase_elapsed_seconds"]
    projected = total + 2802 * timing["pilot_seconds"] / 512 + 600
    expected_timing = {
        "pilot_equivalents": 512,
        "pilot_seconds": timing["pilot_seconds"],
        "gpu_phase_elapsed_seconds": timing["gpu_phase_elapsed_seconds"],
        "preparation_seconds": preparation_seconds,
        "elapsed_preparation_loading_preflight_and_pilot": total,
        "scientific_outcomes": 2802,
        "analysis_allowance_seconds": 600,
        "projected_seconds": projected,
        "limit_seconds": 2700,
        "passed": projected <= 2700,
    }
    check(timing == expected_timing and timing["passed"] is True, "preparation-inclusive timing")


def validate_text_only(records):
    keys = {
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
    check(isinstance(records, list) and len(records) == 2, "two text-only outcomes")
    for record, spec in zip(records, expected_conditions(), strict=True):
        check(set(record) == keys, "text-only schema")
        check(
            record["condition"] == spec["name"]
            and record["wording"] == spec["wording"]
            and record["prompt"] == spec["prompt"]
            and record["raw_margin"] == record["semantic_margin"]
            and record["constant_auroc_against_index_labels"] == 0.5
            and record["subtracting_constant_preserves_image_auroc"] is True,
            "text-only identity",
        )
        values = np.asarray(
            [
                record[name]
                for name in (
                    "raw_margin",
                    "semantic_margin",
                    "probability",
                    "answer_token_mass",
                    "lse_margin",
                    "log_partition",
                )
            ]
        )
        logits = np.asarray(record["candidate_logits"], dtype=np.float32)
        candidate_ids = sorted(YES + NO)
        lookup = {token_id: index for index, token_id in enumerate(candidate_ids)}
        yes, no = [lookup[value] for value in YES], [lookup[value] for value in NO]
        raw = logits[yes].max() - logits[no].max() if logits.shape == (6,) else np.nan
        lse = (
            np.logaddexp.reduce(logits[yes]) - np.logaddexp.reduce(logits[no])
            if logits.shape == (6,)
            else np.nan
        )
        mass = (
            np.exp(np.logaddexp.reduce(logits) - record["log_partition"])
            if logits.shape == (6,)
            else np.nan
        )
        check(
            np.isfinite(values).all()
            and np.isfinite(logits).all()
            and 0 <= record["answer_token_mass"] <= 1,
            "finite text-only scores",
        )
        same(record["raw_margin"], raw, "text-only native raw", 1e-6)
        same(record["semantic_margin"], raw, "text-only native semantic", 1e-6)
        same(record["lse_margin"], lse, "text-only native lse", 1e-6)
        same(record["answer_token_mass"], mass, "text-only native mass", 1e-6)
        same(
            record["probability"],
            1 / (1 + np.exp(-record["semantic_margin"])),
            "text-only sigmoid",
            1e-12,
        )


def validate_parent_review(review):
    check(
        review.get("valid") is True
        and review.get("readOnly") is True
        and review.get("modelExpected") == "claude-fable-5-1"
        and review.get("modelsObserved") == ["claude-fable-5-1"]
        and review.get("effort") == "medium"
        and review.get("arisFullSha") == ARIS_FULL_SHA
        and review.get("wrapperSha256") == REVIEWER_WRAPPER_SHA256
        and review.get("checkoutBefore") == {"head": PARENT_COMMIT, "status": ""}
        and review.get("checkoutAfter") == {"head": PARENT_COMMIT, "status": ""},
        "pinned read-only parent review receipt",
    )


def parent_source(root, prepared, cohort):
    run = root / "runs" / PARENT_RUN_ID
    metadata = dict(
        line.split("=", 1)
        for line in (run / "metadata.env").read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    expected = {
        "RUN_ID": PARENT_RUN_ID,
        "SOURCE_COMMIT": PARENT_COMMIT,
        "COMMAND_STATUS": "1",
        "DISPATCHER_STATUS": "1",
        "CLEANUP_STATUS": "0",
        "ABORT_SIGNAL": "none",
    }
    check(all(metadata.get(key) == value for key, value in expected.items()), "parent terminal")
    check(
        (run / "command_exit_status").read_text().strip() == "1"
        and (run / "exit_status").read_text().strip() == "1",
        "parent statuses",
    )
    parent_manifest = inventory(run / "SHA256SUMS")
    check(
        parent_manifest.get("artifacts/cohort.json") == PARENT_COHORT_SHA256,
        "parent cohort manifest receipt",
    )
    parent_cohort = read(run / "artifacts/cohort.json")
    check(parent_cohort == cohort == base.replay_allocation(root), "retained allocation replay")
    internal = read_path(PARENT_INTERNAL_RECEIPT)
    check(
        internal.get("status") == "VERIFIED_FAILED_PREFLIGHT"
        and internal.get("failure_class") == "MEASUREMENT"
        and internal.get("run_id") == PARENT_RUN_ID
        and internal.get("source_commit") == PARENT_COMMIT
        and internal.get("new_scientific_artifacts") == 0,
        "accepted parent measurement receipt",
    )
    source = prepared["sources"]["parent_allocation"]
    validate_parent_review(read_path(PARENT_REVIEW_RECEIPT))
    check(
        source["run_id"] == PARENT_RUN_ID
        and source["source_commit"] == PARENT_COMMIT
        and source["terminal"] == metadata
        and source["cohort_path"] == str(run / "artifacts/cohort.json")
        and source["cohort_manifest_sha256"] == PARENT_COHORT_SHA256,
        "new parent-allocation provenance",
    )
    check(
        source["internal_receipt"] == PARENT_INTERNAL_RECEIPT
        and source["review_receipt"] == PARENT_REVIEW_RECEIPT,
        "stored parent receipt paths",
    )


def csv_scores(path, cohort, commit, prompts):
    arrays = {
        name: np.empty((2, 2, N))
        for name in (
            "raw_margin",
            "semantic_margin",
            "probability",
            "answer_token_mass",
            "lse_margin",
        )
    }
    with path.open(newline="", encoding="utf-8") as stream:
        records = iter(csv.DictReader(stream))
        for condition, name in enumerate(CONDITIONS):
            for role_index, role in enumerate(("index", "donor")):
                for index, row in enumerate(cohort[role]):
                    record = next(records, None)
                    check(record is not None, "complete 2,800-row score grid")
                    expected = {
                        "source_commit": commit,
                        "condition": name,
                        "wording": str(condition),
                        "prompt": prompts[condition]["prompt"],
                        "image_role": role,
                        "pair_index": str(index),
                        "patient_id": str(row["patient_id"]),
                        "row_id": row["row_id"],
                        "image_path": row["image_path"],
                        "index_label": str(cohort["index"][index]["Effusion"]),
                        "source_label": str(row["Effusion"]),
                    }
                    check(all(record.get(k) == v for k, v in expected.items()), "score identity")
                    for key, array in arrays.items():
                        array[condition, role_index, index] = float(record[key])
        check(next(records, None) is None, "no duplicate score rows")
    same(arrays["raw_margin"], arrays["semantic_margin"], "yes/no semantic orientation")
    same(
        arrays["probability"],
        1 / (1 + np.exp(-arrays["semantic_margin"])),
        "semantic sigmoid",
        1e-12,
    )
    check(
        np.isfinite(np.stack(list(arrays.values()))).all()
        and ((arrays["answer_token_mass"] >= 0) & (arrays["answer_token_mass"] <= 1)).all(),
        "finite score grid and token mass",
    )
    return arrays


def primary(values, advantages):
    return np.stack(
        (values[..., 0] - 0.5, advantages[..., 0], advantages[..., 1] - advantages[..., 0]),
        axis=-1,
    )


def interval(values):
    values = np.asarray(values)
    return np.percentile(values[np.isfinite(values)], [2.5, 97.5], method="linear")


def verify(run_id, commit, summary_dir=None, validator_commit=None, terminal_commit=None):
    root = Path("/home/qingchan/data/concept-flow")
    run, out = root / "runs" / run_id, root / "runs" / run_id / "artifacts"
    check(run.parent == root / "runs" and len(commit) == 40, "run identity")
    metadata = dict(
        line.split("=", 1)
        for line in (run / "metadata.env").read_text().splitlines()
        if "=" in line
    )
    recovery = summary_dir is not None
    expected_status = "1" if recovery else "0"
    expected_terminal = {
        "RUN_ID": run_id,
        "SOURCE_COMMIT": commit,
        "COMMAND_STATUS": expected_status,
        "DISPATCHER_STATUS": expected_status,
        "CLEANUP_STATUS": "0",
        "ABORT_SIGNAL": "none",
        "GPU_COUNT": "1",
    }
    check(all(metadata.get(k) == v for k, v in expected_terminal.items()), "terminal metadata")
    check(
        all(
            (run / name).read_text().strip() == expected_status
            for name in ("command_exit_status", "exit_status")
        ),
        "terminal statuses",
    )
    terminal_manifest = inventory(run / "SHA256SUMS")
    names = ["metadata.env", "command_exit_status", "exit_status"] + [
        "artifacts/" + name
        for name in (
            "cohort.json",
            "prepared.json",
            "prepared.npz",
            "preflight.json",
            "text-only.json",
            "per-image.csv",
            "scores.npz",
        )
    ]
    if not recovery:
        names.extend(("artifacts/" + SUMMARY + ".json", "artifacts/" + SUMMARY + ".npz"))
    digests = {}
    for name in names:
        with (run / name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        check(terminal_manifest.get(name) == digest, "decision-bearing terminal manifest " + name)
        digests[name] = digest

    summary_root = out
    if recovery:
        summary_root = Path(summary_dir).resolve(strict=True)
        check(
            summary_root.parent == root / "state"
            and validator_commit is not None
            and terminal_commit is not None,
            "recovery summary identity",
        )
        terminal_commit = validate_checkout(terminal_commit)
        recovery_manifest = inventory(summary_root / "SHA256SUMS")
        for name in (SUMMARY + ".json", SUMMARY + ".npz", "recovery.json"):
            with (summary_root / name).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            check(recovery_manifest.get(name) == digest, "recovery manifest " + name)
            digests["recovery/" + name] = digest
        recovery_receipt = read(summary_root / "recovery.json")
        check(
            recovery_receipt
            == {
                "status": "RECOVERED_SUMMARY",
                "source_run_id": run_id,
                "source_commit": commit,
                "validator_commit": validator_commit,
                "scientific_execution_reused": True,
                "new_scientific_outcomes": 0,
                "failure_class": "IMPLEMENTATION",
                "correction": "float32 cross-library reduction replay tolerances",
                "lse_replay_atol": LSE_REPLAY_ATOL,
                "token_mass_replay_atol": TOKEN_MASS_REPLAY_ATOL,
                "summary_status": "OBSERVED",
            },
            "recovery receipt",
        )

    cohort, prepared, preflight, text = [
        read(out / name)
        for name in ("cohort.json", "prepared.json", "preflight.json", "text-only.json")
    ]
    summary = read(summary_root / (SUMMARY + ".json"))
    parent_source(root, prepared, cohort)
    validate_protocol(prepared["protocol"])
    validate_model_source(prepared["sources"]["model"])
    check(prepared["cohort"] == cohort == summary["cohort"], "cohort copies")
    check(prepared["source_commit"] == summary["source_commit"] == commit, "commit copies")
    check(
        prepared["protocol"] == summary["protocol"]
        and prepared["sources"] == summary["sources"]
        and prepared["protocol"] == expected_protocol(),
        "protocol and source copies",
    )
    check(
        np.isfinite(prepared["preparation_seconds"]) and prepared["preparation_seconds"] > 0,
        "preparation timing",
    )
    validate_text_only(text)
    check(text == summary["text_only"], "text-only summary copy")
    check(preflight == summary["preflight"], "preflight summary copy")
    validate_preflight(
        preflight,
        prepared["preparation_seconds"],
        commit,
        prepared["sources"]["model"]["model_source"],
    )

    with np.load(out / "prepared.npz", allow_pickle=False) as archive:
        prep = {key: archive[key] for key in archive.files}
    base.verify_frozen_reader_source(root, prepared, prep)
    indices = np.random.default_rng(20260917).integers(0, N, size=(B, N), dtype=np.int64)
    check(
        prep["bootstrap_indices"].dtype == np.int64
        and np.array_equal(prep["bootstrap_indices"], indices),
        "bootstrap identity",
    )
    scores = csv_scores(out / "per-image.csv", cohort, commit, summary["protocol"]["conditions"])
    with np.load(out / "scores.npz", allow_pickle=False) as archive:
        stored = {key: archive[key] for key in archive.files}
    with np.load(summary_root / (SUMMARY + ".npz"), allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    check(
        saved["source_commit"].item() == commit and saved["route"].item() == "evidence_synthesis",
        "summary NPZ identity",
    )
    for key, value in prep.items():
        same(saved[key], value, "prepared array copy " + key)
    for key, value in stored.items():
        same(saved[key], value, "native array copy " + key)

    token = preflight["token_ids"]["0Y"]
    lookup = {token_id: index for index, token_id in enumerate(token["candidate_ids"])}
    yes, no = [lookup[value] for value in YES], [lookup[value] for value in NO]
    for condition, name in enumerate(CONDITIONS):
        for role_index, role in enumerate(("index", "donor")):
            key = name + "__" + role
            logits = stored["candidate_logits__" + key]
            partition = stored["log_partition__" + key]
            check(logits.shape == (N, 6) and logits.dtype == np.float32, "candidate logits")
            raw = logits[:, yes].max(1) - logits[:, no].max(1)
            lse = np.logaddexp.reduce(logits[:, yes], axis=1) - np.logaddexp.reduce(
                logits[:, no], axis=1
            )
            mass = np.exp(np.logaddexp.reduce(logits, axis=1) - partition)
            same(scores["raw_margin"][condition, role_index], raw, "candidate raw", 1e-6)
            same(scores["semantic_margin"][condition, role_index], raw, "candidate semantic", 1e-6)
            same(
                scores["lse_margin"][condition, role_index],
                lse,
                "candidate lse",
                LSE_REPLAY_ATOL,
            )
            same(
                scores["answer_token_mass"][condition, role_index],
                mass,
                "candidate mass",
                TOKEN_MASS_REPLAY_ATOL,
            )
    for key, value in scores.items():
        same(saved[key], value, "score array copy " + key)

    labels = np.asarray([row["Effusion"] for row in cohort["index"]])
    donor_labels = np.asarray([row["Effusion"] for row in cohort["donor"]])
    counts = np.vstack((np.ones(N, dtype=np.int64), weights(indices)))
    aurocs = np.empty((2, 2, B + 1))
    donor_own = np.empty(2)
    for condition in range(2):
        for role in range(2):
            aurocs[condition, role] = auc(
                scores["semantic_margin"][condition, role], labels, counts
            )
        donor_own[condition] = auc(
            scores["semantic_margin"][condition, 1], donor_labels, counts[:1]
        )[0]
    advantages = (aurocs[:, 0] - aurocs[:, 1]).T
    theta = primary(aurocs[:, 0, 0], advantages[0])
    theta_draws = primary(aurocs[:, 0, 1:].T, advantages[1:])
    valid = np.isfinite(aurocs[:, :, 1:]).all(axis=(0, 1)) & np.isfinite(theta_draws).all(1)
    valid_n = int(valid.sum())
    radius = np.percentile(np.max(np.abs(theta_draws[valid] - theta), axis=1), 95, method="linear")
    available = valid_n >= 9500 and radius > 0
    simultaneous = np.stack((theta - radius, theta + radius), axis=1)
    opportunity = bool(simultaneous[0, 0] > 0 and simultaneous[1, 0] > 0) if available else None
    wording = (
        (
            "positive"
            if simultaneous[2, 0] > 0
            else "negative"
            if simultaneous[2, 1] < 0
            else "unresolved"
        )
        if available
        else None
    )
    same(saved["auroc"], aurocs[:, :, 0], "saved AUROC")
    same(saved["bootstrap_auroc"], aurocs[:, :, 1:], "saved AUROC draws")
    same(saved["donor_own_label_auroc"], donor_own, "saved donor-own AUROC")
    same(saved["image_advantage"], advantages[0], "saved advantages")
    same(saved["bootstrap_image_advantage"], advantages[1:], "saved advantage draws")
    same(saved["primary"], theta, "saved primary")
    same(saved["bootstrap_primary"], theta_draws, "saved primary draws")
    check(np.array_equal(saved["common_valid"], valid), "saved valid mask")
    same(saved["simultaneous_radius"], radius, "saved radius")
    family = summary["primary_family"]
    check(
        family["common_valid_draws"] == valid_n
        and family["population_inference"] == ("AVAILABLE" if available else "UNAVAILABLE")
        and family["bootstrap_degenerate"] is bool(radius == 0)
        and family["opportunity_detected"] is opportunity
        and family["wording_dependence"] == wording,
        "primary decisions",
    )
    check([row["name"] for row in family["contrasts"]] == list(PRIMARY), "primary order")
    for index, row in enumerate(family["contrasts"]):
        same(row["estimate"], theta[index], "primary estimate")
        same(row["simultaneous_ci95"], simultaneous[index], "primary interval")

    check(len(summary["cells"]) == 2, "two wording reports")
    for condition, row in enumerate(summary["cells"]):
        expected_scalars = {
            "real_auroc": aurocs[condition, 0, 0],
            "donor_auroc_against_index_label": aurocs[condition, 1, 0],
            "donor_auroc_against_donor_label": donor_own[condition],
            "image_advantage": advantages[0, condition],
            "real_mean_margin": scores["semantic_margin"][condition, 0].mean(),
            "donor_mean_margin": scores["semantic_margin"][condition, 1].mean(),
            "real_brier": np.mean((scores["probability"][condition, 0] - labels) ** 2),
            "donor_brier_against_index_label": np.mean(
                (scores["probability"][condition, 1] - labels) ** 2
            ),
            "real_mean_answer_token_mass": scores["answer_token_mass"][condition, 0].mean(),
            "donor_mean_answer_token_mass": scores["answer_token_mass"][condition, 1].mean(),
        }
        check(row["condition"] == CONDITIONS[condition], "reported condition")
        for key, value in expected_scalars.items():
            same(row[key], value, "reported " + key)
        same(row["real_auroc_ci95"], interval(aurocs[condition, 0, 1:]), "real interval")
        same(row["donor_auroc_ci95"], interval(aurocs[condition, 1, 1:]), "donor interval")
        same(row["image_advantage_ci95"], interval(advantages[1:, condition]), "advantage interval")
    a_difference = aurocs[1, 0] - aurocs[0, 0]
    same(summary["a1_minus_a0"], a_difference[0], "A1-A0")
    same(summary["a1_minus_a0_ci95"], interval(a_difference[1:]), "A1-A0 interval")

    check(len(summary["same_logit_aggregation"]) == 2, "two scoring-rule reports")
    for condition, row in enumerate(summary["same_logit_aggregation"]):
        real = auc(scores["lse_margin"][condition, 0], labels, counts)
        donor = auc(scores["lse_margin"][condition, 1], labels, counts)
        lse_advantage = real - donor
        difference = lse_advantage - advantages[:, condition]
        same(row["real_auroc"], real[0], "lse real")
        same(row["donor_auroc_against_index_label"], donor[0], "lse donor")
        same(row["image_advantage"], lse_advantage[0], "lse advantage")
        same(row["image_advantage_ci95"], interval(lse_advantage[1:]), "lse interval")
        same(
            row["image_advantage_difference_from_max_variant"], difference[0], "scoring difference"
        )
        same(
            row["image_advantage_difference_from_max_variant_ci95"],
            interval(difference[1:]),
            "scoring difference interval",
        )

    raw = stored["reader_raw_activation"]
    check(raw.shape == (N, 1024) and raw.dtype == np.float16, "reader capture")
    projected, features, clinical, controls = reader_scores(raw, prep)
    types = [
        f"{row['view_AP']}_{row['sex_M']}_{min(max(int(row['age']) // 10, 0), 9)}"
        for row in cohort["index"]
    ]
    assignments = prepared["control_assignments"]
    check(
        len(assignments) == 20
        and all(all(kind in record["assignment"] for kind in types) for record in assignments),
        "twenty complete frozen controls",
    )
    control_labels = np.asarray(
        [[record["assignment"][kind] for kind in types] for record in assignments]
    )
    clinical_auc = auc(clinical, labels, counts)
    control_auc = np.stack(
        [auc(score, target, counts) for score, target in zip(controls, control_labels, strict=True)]
    )
    selectivity = clinical_auc - control_auc.mean(0)
    selectivity[~np.isfinite(control_auc).all(0)] = np.nan
    reader_valid = np.isfinite(selectivity[1:])
    reader_interval = interval(selectivity[1:])
    reader = summary["frozen_reader"]
    reader_available = min(labels.sum(), N - labels.sum()) >= 10 and reader_valid.sum() >= 9500
    reader_replicated = bool(reader_available and reader_interval[0] > 0)
    same(reader["clinical_auroc"], clinical_auc[0], "reader AUROC")
    same(reader["control_auroc"], control_auc[:, 0], "control AUROCs")
    same(reader["selectivity"], selectivity[0], "reader selectivity")
    same(reader["selectivity_ci95"], reader_interval, "reader interval")
    check(
        reader["valid_draws"] == int(reader_valid.sum())
        and reader["population_inference"] == ("AVAILABLE" if reader_available else "UNAVAILABLE")
        and reader["replicated"] is reader_replicated,
        "separate reader decision",
    )
    for key, value in {
        "reader_projected": projected,
        "reader_features": features,
        "reader_clinical_score": clinical,
        "reader_control_score": controls,
        "reader_control_labels": control_labels,
        "reader_clinical_auroc": clinical_auc[0],
        "reader_control_auroc": control_auc[:, 0],
        "bootstrap_reader_clinical_auroc": clinical_auc[1:],
        "bootstrap_reader_control_auroc": control_auc[:, 1:],
        "bootstrap_reader_selectivity": selectivity[1:],
    }.items():
        same(saved[key], value, "saved " + key)
    check(
        summary["status"] == "OBSERVED"
        and summary["n_image_outcomes"] == 2800
        and summary["n_text_only_outcomes"] == 2
        and summary["route"] == "evidence_synthesis",
        "summary status, counts, and route",
    )
    result = {
        "status": "PASS",
        "run_id": run_id,
        "source_commit": commit,
        "verified_new_files": digests,
        "n_image_outcomes": 2800,
        "answer_population_inference": family["population_inference"],
        "opportunity_detected": opportunity,
        "wording_dependence": wording,
        "reader_replicated": reader_replicated,
        "simultaneous_radius": float(radius),
        "max_numerical_discrepancy": base.MAX_ERROR,
    }
    if recovery:
        result.update(
            recovery_validator_commit=validator_commit,
            terminal_validator_commit=terminal_commit,
        )
    return result


if __name__ == "__main__":
    extra = sys.argv[3:]
    if len(extra) not in (0, 3):
        raise SystemExit(
            "usage: VALIDATOR RUN_ID SOURCE_COMMIT [SUMMARY_DIR RECOVERY_COMMIT TERMINAL_COMMIT]"
        )
    print(
        json.dumps(
            verify(sys.argv[1], sys.argv[2], *extra) if extra else verify(sys.argv[1], sys.argv[2]),
            allow_nan=False,
        )
    )
