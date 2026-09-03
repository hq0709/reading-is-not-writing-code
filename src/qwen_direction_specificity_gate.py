"""Prospectively registered Qwen Effusion direction-specificity supplement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

try:
    from .first_gate import MANIFEST_SHA256, sha256, write_json
    from .qwen_effusion_gate import (
        ARCH,
        CONCEPT,
        EXPECTED_LOCUS,
        LOCUS,
        MODEL_ID,
        MODEL_REVISION,
        REGISTERED_PROMPT,
        UNRELATED_CONCEPTS,
    )
except ImportError:
    from first_gate import MANIFEST_SHA256, sha256, write_json
    from qwen_effusion_gate import (
        ARCH,
        CONCEPT,
        EXPECTED_LOCUS,
        LOCUS,
        MODEL_ID,
        MODEL_REVISION,
        REGISTERED_PROMPT,
        UNRELATED_CONCEPTS,
    )


SOURCE_RUN_ID = "20260903T000321Z-caaae3ef346d-qwen-full"
SOURCE_COMMIT = "caaae3ef346d3ac01c76c53ba99b3b7237066639"
ALPHA = 0.25
EVAL_ROWS = 400
BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 20_260_903
REGISTERED_ROWS_SHA256 = "76be1da9659973d44a0898fa69c579afc4649f329ee227b19d1216efaa1c8bda"
SOURCE_SHA256 = {
    "./artifacts/activations/meta.json": (
        "151463dc0e59e7ed8c7750f76b2b049cdf7b1620053aa2b5ad81903444b5e6c5"
    ),
    "./artifacts/activations/shard0.json": (
        "633d6060ebadb7e6e0beb7c30c353f64fc30eb9f5867f5506af33aa0baac59ac"
    ),
    "./artifacts/activations/shard0.npz": (
        "6ec95da876840a8cf8abee2ba640004ba39568496b9fcf672427221228817147"
    ),
    "./artifacts/hook-verification.json": (
        "ffa18956d748aca9bf7c2e93d5ff871723ae9afb4e5b00e4eb16b4007b68a1db"
    ),
    "./artifacts/input-verification.json": (
        "178380343857575d084620c50c9b9a434059f8aabaee09c934c527741add7cad"
    ),
    "./artifacts/intervention-summary.json": (
        "47f8a1bfc3a46d69b10c0d07b23fbe64ad80c3d09f91cc10d3b06d18b19812f8"
    ),
    "./artifacts/probe/directions.npz": (
        "b1c00134fbb06c2ce0e392ff6eea5d6275837877412e0686d888f6d336dcde91"
    ),
}


def _tagged_hash(tag: str, value: str) -> str:
    return hashlib.sha256(f"{tag}\0{value}".encode()).hexdigest()


def _row_ids_hash(row_ids: list[str]) -> str:
    payload = json.dumps(row_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_registered_rows(
    manifest_rows: list[dict[str, str]], discovery_row_ids: list[str]
) -> tuple[list[dict[str, str]], str]:
    """Select one row per unseen test patient without consulting labels or images."""
    by_id = {row["row_id"]: row for row in manifest_rows}
    if len(by_id) != len(manifest_rows) or any(row_id not in by_id for row_id in discovery_row_ids):
        raise ValueError("manifest or discovery row identity is incomplete")
    discovery_patients = {by_id[row_id]["patient_id"] for row_id in discovery_row_ids}
    candidates: dict[str, list[dict[str, str]]] = {}
    for row in manifest_rows:
        if row["split"] == "test" and row["patient_id"] not in discovery_patients:
            candidates.setdefault(row["patient_id"], []).append(row)
    chosen = [
        min(
            rows,
            key=lambda row: (
                _tagged_hash("qwen-direction-specificity-row-v1", row["row_id"]),
                row["row_id"],
            ),
        )
        for rows in candidates.values()
    ]
    chosen.sort(
        key=lambda row: (
            _tagged_hash("qwen-direction-specificity-patient-v1", row["patient_id"]),
            row["patient_id"],
        )
    )
    selected = chosen[:EVAL_ROWS]
    if len(selected) != EVAL_ROWS or len({row["patient_id"] for row in selected}) != EVAL_ROWS:
        raise ValueError("insufficient unique unseen test patients")
    row_ids = [row["row_id"] for row in selected]
    digest = _row_ids_hash(row_ids)
    return selected, digest


def _metadata(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def _checksum_inventory(path: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        inventory[relative] = digest
    return inventory


def validate_source(source_run: Path, manifest: Path, reference_out: Path, rows_out: Path) -> None:
    """Bind the supplement to accepted Qwen evidence without rehashing its activation shard."""
    source_run = source_run.resolve(strict=True)
    if source_run.name != SOURCE_RUN_ID:
        raise ValueError("unexpected Qwen source run")
    metadata = _metadata(source_run / "metadata.env")
    expected_metadata = {
        "SOURCE_COMMIT": SOURCE_COMMIT,
        "COMMAND_STATUS": "0",
        "DISPATCHER_STATUS": "0",
        "CLEANUP_STATUS": "0",
        "ABORT_SIGNAL": "none",
        "GPU_COUNT": "1",
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("accepted Qwen source run is not terminal-successful")
    if (source_run / "command_exit_status").read_text(encoding="utf-8").strip() != "0":
        raise ValueError("accepted Qwen source command failed")
    if (source_run / "exit_status").read_text(encoding="utf-8").strip() != "0":
        raise ValueError("accepted Qwen source dispatcher failed")
    inventory = _checksum_inventory(source_run / "SHA256SUMS")
    for relative, digest in SOURCE_SHA256.items():
        if inventory.get(relative) != digest or not (source_run / relative).is_file():
            raise ValueError(f"accepted Qwen receipt mismatch for {relative}")

    if sha256(manifest) != MANIFEST_SHA256:
        raise ValueError("registered NIH manifest hash mismatch")
    manifest_rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    source_summary_path = source_run / "artifacts/intervention-summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    discovery_rows = source_summary.get("eval_row_ids")
    if (
        source_summary.get("arch") != ARCH
        or source_summary.get("concept") != CONCEPT
        or source_summary.get("availability_eligible") is not True
        or not isinstance(discovery_rows, list)
        or len(discovery_rows) != 200
        or len(set(discovery_rows)) != 200
    ):
        raise ValueError("accepted Qwen scientific source identity is incomplete")
    selected, digest = derive_registered_rows(manifest_rows, discovery_rows)
    if digest != REGISTERED_ROWS_SHA256:
        raise ValueError("prospectively registered evaluation rows changed")
    row_payload = {
        "selection": "one hash-ranked row from each of 400 hash-ranked unseen test patients",
        "selection_version": "qwen-direction-specificity-v1",
        "source_run_id": SOURCE_RUN_ID,
        "source_eval_rows": len(discovery_rows),
        "source_eval_patients": len(
            {
                {row["row_id"]: row for row in manifest_rows}[row_id]["patient_id"]
                for row_id in discovery_rows
            }
        ),
        "n_rows": EVAL_ROWS,
        "n_patients": EVAL_ROWS,
        "row_ids_sha256": digest,
        "row_ids": [row["row_id"] for row in selected],
        "patient_ids": [row["patient_id"] for row in selected],
    }
    write_json(rows_out, row_payload)

    activation_meta = json.loads(
        (source_run / "artifacts/activations/meta.json").read_text(encoding="utf-8")
    )
    expected_activation = {
        "arch": ARCH,
        "model": MODEL_ID,
        "n_rows": 26_229,
        "git_sha": SOURCE_COMMIT,
        "loci": [LOCUS],
        "prompt": REGISTERED_PROMPT,
    }
    if any(activation_meta.get(key) != value for key, value in expected_activation.items()):
        raise ValueError("accepted Qwen activation identity mismatch")
    input_verification = json.loads(
        (source_run / "artifacts/input-verification.json").read_text(encoding="utf-8")
    )
    write_json(
        reference_out,
        {
            "source_run_id": SOURCE_RUN_ID,
            "source_commit": SOURCE_COMMIT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_source": input_verification["snapshot"],
            "activations": os.fspath(source_run / "artifacts/activations"),
            "directions": os.fspath(source_run / "artifacts/probe/directions.npz"),
            "directions_sha256": SOURCE_SHA256["./artifacts/probe/directions.npz"],
            "registered_rows_sha256": digest,
            "accepted_sha256": SOURCE_SHA256,
            "verification_mode": "accepted-immutable-run-receipt",
        },
    )


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def summarize_direction_specificity(
    records: list[dict[str, str]],
    row_ids: list[str],
    patient_ids: list[str],
    bootstrap_out: Path,
) -> dict:
    if len(row_ids) != EVAL_ROWS or len(set(row_ids)) != EVAL_ROWS:
        raise ValueError("registered evaluation row count mismatch")
    if len(patient_ids) != EVAL_ROWS or len(set(patient_ids)) != EVAL_ROWS:
        raise ValueError("registered evaluation patients are not unique")
    expected_directions = [
        "concept",
        *(f"random{index}" for index in range(20)),
        "sham",
        *(f"unrelated_{name}" for name in UNRELATED_CONCEPTS),
    ]
    expected_conditions = {("concept", 0.0), *((name, ALPHA) for name in expected_directions)}
    row_position = {row_id: index for index, row_id in enumerate(row_ids)}
    patient_by_row = dict(zip(row_ids, patient_ids, strict=True))
    values: dict[tuple[str, float], np.ndarray] = {
        condition: np.full(EVAL_ROWS, np.nan, dtype=np.float64)
        for condition in expected_conditions
    }
    seen: set[tuple[str, str, float]] = set()
    for record in records:
        row_id = record.get("row_id", "")
        direction = record.get("direction", "")
        alpha = float(record.get("alpha", "nan"))
        condition = (direction, alpha)
        key = (row_id, direction, alpha)
        if (
            row_id not in row_position
            or record.get("patient_id") != patient_by_row[row_id]
            or record.get("locus") != LOCUS
            or condition not in expected_conditions
            or key in seen
        ):
            raise ValueError("per-image intervention grid identity mismatch")
        probability = float(record.get("p_yes", "nan"))
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("invalid per-image probability")
        values[condition][row_position[row_id]] = probability
        seen.add(key)
    if any(np.isnan(array).any() for array in values.values()):
        raise ValueError("per-image intervention grid is incomplete")
    if len(records) != EVAL_ROWS * len(expected_conditions):
        raise ValueError("unexpected per-image intervention row count")

    baseline = values[("concept", 0.0)]
    effects = {
        direction: values[(direction, ALPHA)] - baseline for direction in expected_directions
    }
    means = {direction: float(effect.mean()) for direction, effect in effects.items()}
    unrelated_names = [f"unrelated_{name}" for name in UNRELATED_CONCEPTS]
    concept_effect = means["concept"]
    maximum_unrelated = max(means[name] for name in unrelated_names)
    primary_margin = concept_effect - maximum_unrelated
    random_effects = np.asarray([means[f"random{index}"] for index in range(20)])
    random_p95 = _percentile(random_effects, 95.0)
    sham_effect = means["sham"]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    concept_bootstrap = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    unrelated_bootstrap = np.empty(
        (BOOTSTRAP_RESAMPLES, len(unrelated_names)), dtype=np.float64
    )
    margin_bootstrap = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        sampled = rng.integers(0, EVAL_ROWS, size=EVAL_ROWS)
        concept_draw = float(effects["concept"][sampled].mean())
        unrelated_draws = np.asarray(
            [effects[name][sampled].mean() for name in unrelated_names], dtype=np.float64
        )
        concept_bootstrap[index] = concept_draw
        unrelated_bootstrap[index] = unrelated_draws
        margin_bootstrap[index] = concept_draw - float(unrelated_draws.max())
    pairwise_bootstrap = concept_bootstrap[:, None] - unrelated_bootstrap

    bootstrap_out.parent.mkdir(parents=True, exist_ok=True)
    temp = bootstrap_out.with_suffix(bootstrap_out.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(
            stream,
            row_id=np.asarray(row_ids),
            patient_id=np.asarray(patient_ids),
            unrelated_names=np.asarray(unrelated_names),
            concept_effect_bootstrap=concept_bootstrap,
            unrelated_effect_bootstrap=unrelated_bootstrap,
            pairwise_difference_bootstrap=pairwise_bootstrap,
            primary_margin_bootstrap=margin_bootstrap,
        )
    os.replace(temp, bootstrap_out)

    one_sided_lower = _percentile(margin_bootstrap, 5.0)
    replicated_selectivity = bool(
        concept_effect > 0
        and concept_effect > random_p95
        and concept_effect > abs(sham_effect)
    )
    return {
        "arch": ARCH,
        "concept": CONCEPT,
        "locus": LOCUS,
        "module": EXPECTED_LOCUS,
        "alpha_mode": "reltoken",
        "alpha": ALPHA,
        "n_eval": EVAL_ROWS,
        "n_eval_patients": EVAL_ROWS,
        "bootstrap_unit": "patient",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "direction_effects": means,
        "random_effect_p95": random_p95,
        "absolute_sham_effect": abs(sham_effect),
        "replicated_random_sham_selectivity": replicated_selectivity,
        "primary": {
            "statistic": "Effusion effect minus maximum fixed unrelated-direction effect",
            "maximum_unrelated_direction": max(unrelated_names, key=means.__getitem__),
            "maximum_unrelated_effect": maximum_unrelated,
            "margin": primary_margin,
            "ci95": [
                _percentile(margin_bootstrap, 2.5),
                _percentile(margin_bootstrap, 97.5),
            ],
            "one_sided_lower_95": one_sided_lower,
            "multiplicity": (
                "the maximum over all five fixed unrelated directions is recomputed inside every "
                "patient-bootstrap replicate"
            ),
        },
        "pairwise_differences": {
            name.removeprefix("unrelated_"): {
                "estimate": concept_effect - means[name],
                "ci95": [
                    _percentile(pairwise_bootstrap[:, index], 2.5),
                    _percentile(pairwise_bootstrap[:, index], 97.5),
                ],
            }
            for index, name in enumerate(unrelated_names)
        },
        "direction_specific": bool(replicated_selectivity and one_sided_lower > 0),
    }


def summarize_run(
    per_image_path: Path,
    meta_path: Path,
    done_path: Path,
    source_reference_path: Path,
    registered_rows_path: Path,
    bootstrap_out: Path,
    out: Path,
) -> None:
    source = json.loads(source_reference_path.read_text(encoding="utf-8"))
    registered = json.loads(registered_rows_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_meta = {
        "arch": ARCH,
        "concept": CONCEPT,
        "alpha_mode": "reltoken",
        "alphas": [0.0, ALPHA],
        "control_alphas": [ALPHA],
        "n_random": 20,
        "n_eval": EVAL_ROWS,
        "seed": 0,
        "eval_split": "test",
        "model_source": source["model_source"],
        "model_local_only": True,
        "prompt": REGISTERED_PROMPT,
        "directions_sha256": source["directions_sha256"],
        "eval_row_ids_sha256": REGISTERED_ROWS_SHA256,
    }
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            raise ValueError(f"direction-specificity intervention mismatch for {key}")
    if meta.get("eval_row_ids") != registered.get("row_ids"):
        raise ValueError("intervention did not use the prospectively registered rows")
    if registered.get("row_ids_sha256") != REGISTERED_ROWS_SHA256:
        raise ValueError("registered row receipt hash mismatch")
    intervention_commit = os.environ.get("SOURCE_COMMIT", "unknown")
    if intervention_commit == "unknown" or meta.get("source_commit") != intervention_commit:
        raise ValueError("intervention artifacts do not match the immutable source commit")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("loci") != [LOCUS] or done.get("rows") != 28:
        raise ValueError("direction-specificity intervention is incomplete")
    records = list(csv.DictReader(per_image_path.open(newline="", encoding="utf-8")))
    summary = summarize_direction_specificity(
        records, registered["row_ids"], registered["patient_ids"], bootstrap_out
    )
    summary.update(
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "prompt": REGISTERED_PROMPT,
            "source_run_id": SOURCE_RUN_ID,
            "source_commit": SOURCE_COMMIT,
            "intervention_commit": intervention_commit,
            "registered_rows_sha256": REGISTERED_ROWS_SHA256,
            "evidence": {
                "source_reference_sha256": sha256(source_reference_path),
                "registered_rows_receipt_sha256": sha256(registered_rows_path),
                "intervention_meta_sha256": sha256(meta_path),
                "per_image_sha256": sha256(per_image_path),
                "completion_marker_sha256": sha256(done_path),
                "bootstrap_sha256": sha256(bootstrap_out),
            },
        }
    )
    write_json(out, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-source")
    validate.add_argument("--source-run", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--out", type=Path, required=True)
    validate.add_argument("--rows-out", type=Path, required=True)
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--per-image", type=Path, required=True)
    summarize.add_argument("--meta", type=Path, required=True)
    summarize.add_argument("--done", type=Path, required=True)
    summarize.add_argument("--source-reference", type=Path, required=True)
    summarize.add_argument("--registered-rows", type=Path, required=True)
    summarize.add_argument("--bootstrap", type=Path, required=True)
    summarize.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate-source":
        validate_source(args.source_run, args.manifest, args.out, args.rows_out)
    else:
        summarize_run(
            args.per_image,
            args.meta,
            args.done,
            args.source_reference,
            args.registered_rows,
            args.bootstrap,
            args.out,
        )


if __name__ == "__main__":
    main()
