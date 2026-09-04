"""Registered fresh-cohort Consolidation input-closure gate for Qwen ``vis.last``."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from .bootstrap_probe import boot_auroc, pct_ci
    from .first_gate import capacity_direction, load_single_locus, sha256, write_json
    from .probe import build_types, control_labels
    from .qwen_direction_specificity_gate import (
        SOURCE_COMMIT,
        SOURCE_RUN_ID,
        _checksum_inventory,
        _metadata,
        validate_source as validate_primary_source,
    )
    from .qwen_effusion_gate import ARCH, EXPECTED_LOCUS, LOCUS, MODEL_ID, MODEL_REVISION
except ImportError:
    from bootstrap_probe import boot_auroc, pct_ci
    from first_gate import capacity_direction, load_single_locus, sha256, write_json
    from probe import build_types, control_labels
    from qwen_direction_specificity_gate import (
        SOURCE_COMMIT,
        SOURCE_RUN_ID,
        _checksum_inventory,
        _metadata,
        validate_source as validate_primary_source,
    )
    from qwen_effusion_gate import ARCH, EXPECTED_LOCUS, LOCUS, MODEL_ID, MODEL_REVISION


CONCEPT = "Consolidation"
PROMPT = "Is there consolidation in this chest radiograph? Answer yes or no."
CLINICAL_CONTROLS = (
    "Effusion",
    "Atelectasis",
    "Pneumothorax",
    "Cardiomegaly",
    "Mass",
    "Nodule",
)
EVAL_PAIRS = 50
N_RANDOM = 119
MAX_ABS_ALPHA = 0.25
BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 20_260_905
PROBE_BOOTSTRAP_RESAMPLES = 2_000
PROBE_BOOTSTRAP_SEED = 20_260_827
REGISTERED_PAIRS_SHA256 = "20dcc8cc3aaa8c65a6f7f759a0ff6bf3b0cf4f1b87f4b65cdc6a15a46c6d85e2"

PREVIOUS_RUN_ID = "20260903T103508Z-456c81bad460-7a3edc4b"
OWNERSHIP_RUN_ID = "20260904T125710Z-a3bd883540eb-causal-ownership"
OWNERSHIP_COMMIT = "a3bd883540ebf81cc95bddd587d3439168f4c63d"
OWNERSHIP_SHA256 = {
    "./artifacts/causal-ownership-summary.json": (
        "13eb9aaa714d42c4e9a564ed53dd1c430560a1082798bddf45696f73ac3730e5"
    ),
    "./artifacts/mechanism-route.json": (
        "a6609d00e6f7b0fa329892461d429cb762200a4bcfd597458b6c43e168ba7b04"
    ),
    "./artifacts/registered-rows.json": (
        "b2657e6c246b7cdb7d16a984d72de62c50cade492a4b54ee3b0ab1ee64e5ddf2"
    ),
    "./artifacts/source-reference.json": (
        "ef7cf1b3a73d88d102e873e0208085fe52da705f55a64a343c1c554ef7b64186"
    ),
    "./command_exit_status": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "./exit_status": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "./metadata.env": "5e08764667a24d4a5b65c143963ef23624c7776f5945af50f63e7a0ccfecae79",
}


def _tagged_hash(tag: str, value: str) -> str:
    return hashlib.sha256(f"{tag}\0{value}".encode()).hexdigest()


def _pairs_hash(pairs: list[dict[str, str]]) -> str:
    payload = json.dumps(pairs, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def derive_registered_pairs(
    manifest_rows: list[dict[str, str]], prior_row_ids: list[str]
) -> tuple[list[dict[str, str]], str]:
    """Choose 50 unseen test patients with one positive/negative study each."""
    by_id = {row["row_id"]: row for row in manifest_rows}
    if len(by_id) != len(manifest_rows) or any(row_id not in by_id for row_id in prior_row_ids):
        raise ValueError("manifest or prior cohort identity is incomplete")
    excluded_patients = {by_id[row_id]["patient_id"] for row_id in prior_row_ids}
    by_patient: dict[str, list[dict[str, str]]] = {}
    for row in manifest_rows:
        if row["split"] == "test" and row["patient_id"] not in excluded_patients:
            by_patient.setdefault(row["patient_id"], []).append(row)

    pairs: list[dict[str, str]] = []
    for patient_id, rows in by_patient.items():
        positive = [row for row in rows if row[CONCEPT] == "1"]
        negative = [row for row in rows if row[CONCEPT] == "0"]
        if not positive or not negative:
            continue
        positive_row = min(
            positive,
            key=lambda row: (
                _tagged_hash("qwen-consolidation-input-closure-positive-v1", row["row_id"]),
                row["row_id"],
            ),
        )
        negative_row = min(
            negative,
            key=lambda row: (
                _tagged_hash("qwen-consolidation-input-closure-negative-v1", row["row_id"]),
                row["row_id"],
            ),
        )
        pairs.append(
            {
                "patient_id": patient_id,
                "positive_row_id": positive_row["row_id"],
                "negative_row_id": negative_row["row_id"],
            }
        )
    pairs.sort(
        key=lambda item: (
            _tagged_hash("qwen-consolidation-input-closure-patient-v1", item["patient_id"]),
            item["patient_id"],
        )
    )
    selected = pairs[:EVAL_PAIRS]
    if len(selected) != EVAL_PAIRS or len({item["patient_id"] for item in selected}) != EVAL_PAIRS:
        raise ValueError("insufficient unseen within-patient Consolidation transitions")
    return selected, _pairs_hash(selected)


def load_prior_row_ids(runs_root: Path) -> list[str]:
    source = json.loads(
        (runs_root / SOURCE_RUN_ID / "artifacts/intervention-summary.json").read_text(
            encoding="utf-8"
        )
    )
    previous = json.loads(
        (runs_root / PREVIOUS_RUN_ID / "artifacts/registered-rows.json").read_text(
            encoding="utf-8"
        )
    )
    ownership = json.loads(
        (runs_root / OWNERSHIP_RUN_ID / "artifacts/registered-rows.json").read_text(
            encoding="utf-8"
        )
    )
    row_ids = source.get("eval_row_ids", []) + previous.get("row_ids", []) + ownership.get(
        "row_ids", []
    )
    if len(row_ids) != 1_000 or len(set(row_ids)) != 1_000:
        raise ValueError("prior Qwen intervention cohorts are incomplete or overlap")
    return row_ids


def validate_source(
    source_run: Path,
    ownership_run: Path,
    manifest: Path,
    reference_out: Path,
    pairs_out: Path,
) -> None:
    """Bind the gate to accepted Qwen receipts without rehashing model or activation shards."""
    prior_rows_out = pairs_out.with_name("prior-direction-specificity-rows-derived.json")
    validate_primary_source(source_run, manifest, reference_out, prior_rows_out)
    ownership_run = ownership_run.resolve(strict=True)
    if ownership_run.name != OWNERSHIP_RUN_ID:
        raise ValueError("unexpected causal-ownership source run")
    metadata = _metadata(ownership_run / "metadata.env")
    expected = {
        "SOURCE_COMMIT": OWNERSHIP_COMMIT,
        "COMMAND_STATUS": "0",
        "DISPATCHER_STATUS": "0",
        "CLEANUP_STATUS": "0",
        "ABORT_SIGNAL": "none",
        "GPU_COUNT": "1",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("accepted causal-ownership run is not terminal-successful")
    inventory = _checksum_inventory(ownership_run / "SHA256SUMS")
    for relative, digest in OWNERSHIP_SHA256.items():
        if inventory.get(relative) != digest or not (ownership_run / relative).is_file():
            raise ValueError(f"accepted causal-ownership receipt mismatch for {relative}")
    ownership_summary = json.loads(
        (ownership_run / "artifacts/causal-ownership-summary.json").read_text(encoding="utf-8")
    )
    route = json.loads(
        (ownership_run / "artifacts/mechanism-route.json").read_text(encoding="utf-8")
    )
    if (
        ownership_summary.get("n_owned_concepts") != 0
        or ownership_summary.get("shared_alias", {}).get("detected") is not False
        or route.get("selected_route") != "consolidation_fallback"
        or route.get("selection", {}).get("concept") != CONCEPT
    ):
        raise ValueError("causal-ownership result does not authorize Consolidation fallback")

    manifest_rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    prior_row_ids = load_prior_row_ids(ownership_run.parent)
    selected, digest = derive_registered_pairs(manifest_rows, prior_row_ids)
    if digest != REGISTERED_PAIRS_SHA256:
        raise ValueError("prospectively registered Consolidation pairs changed")
    write_json(
        pairs_out,
        {
            "selection": (
                "one hash-selected positive and negative study from each of 50 hash-ranked "
                "test patients absent from all three earlier Qwen intervention cohorts"
            ),
            "selection_version": "qwen-consolidation-input-closure-v1",
            "source_run_id": SOURCE_RUN_ID,
            "ownership_run_id": OWNERSHIP_RUN_ID,
            "n_pairs": EVAL_PAIRS,
            "n_patients": EVAL_PAIRS,
            "n_rows": 2 * EVAL_PAIRS,
            "pairs_sha256": digest,
            "pairs": selected,
        },
    )
    reference = json.loads(reference_out.read_text(encoding="utf-8"))
    reference.update(
        {
            "ownership_run_id": OWNERSHIP_RUN_ID,
            "ownership_commit": OWNERSHIP_COMMIT,
            "registered_pairs_sha256": digest,
        }
    )
    write_json(reference_out, reference)


def run_probe(acts_dir: Path, manifest: Path, out_dir: Path) -> None:
    ids, raw = load_single_locus(acts_dir)
    manifest_rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    by_id = {row["row_id"]: row for row in manifest_rows}
    if len(ids) != len(by_id) or len(set(ids)) != len(ids) or set(ids) != set(by_id):
        raise ValueError("activation rows do not exactly cover the registered manifest")
    meta = [by_id[row_id] for row_id in ids]
    split = np.asarray([row["split"] for row in meta])
    train, test = split == "train", split == "test"
    labels = np.asarray([int(row[CONCEPT]) for row in meta], dtype=np.int8)

    rng = np.random.default_rng(0)
    projection = (rng.standard_normal((raw.shape[1], 512)) / np.sqrt(512)).astype(np.float32)
    projected = raw.astype(np.float32) @ projection
    scaler = StandardScaler().fit(projected[train])
    x_train = scaler.transform(projected[train])
    x_test = scaler.transform(projected[test])
    del projected, raw

    direction_names = [CONCEPT, *CLINICAL_CONTROLS]
    coefficients: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    real_probe = None
    for name in direction_names:
        target = np.asarray([int(row[name]) for row in meta], dtype=np.int8)
        if len(np.unique(target[train])) != 2:
            raise ValueError(f"registered direction {name} is unusable")
        readout = LogisticRegression(C=1.0, max_iter=2000, random_state=0).fit(
            x_train, target[train]
        )
        if name == CONCEPT:
            real_probe = readout
        coefficients.append(readout.coef_[0])
        vectors.append(capacity_direction(projection, scaler.scale_, readout.coef_[0]))
    assert real_probe is not None
    real_scores = real_probe.predict_proba(x_test)[:, 1]
    real_auroc = float(roc_auc_score(labels[test], real_scores))

    types = build_types(meta, ("view_AP", "sex_M", "age"))
    control_scores: list[np.ndarray] = []
    control_labels_test: list[np.ndarray] = []
    controls = []
    for seed in range(20):
        generated, assignment, usable = control_labels(types, np.random.default_rng(seed))
        if not usable or len(np.unique(generated[train])) != 2 or len(np.unique(generated[test])) != 2:
            raise ValueError(f"control seed {seed} is unusable")
        scores = (
            LogisticRegression(C=1.0, max_iter=2000, random_state=0)
            .fit(x_train, generated[train])
            .predict_proba(x_test)[:, 1]
        )
        auroc = float(roc_auc_score(generated[test], scores))
        control_scores.append(scores)
        control_labels_test.append(generated[test])
        controls.append({"seed": seed, "auroc": auroc, "assignment": assignment})

    test_meta = [meta[index] for index in np.flatnonzero(test)]
    patients = sorted({row["patient_id"] for row in test_meta})
    patient_index = {patient: index for index, patient in enumerate(patients)}
    row_patient = np.asarray([patient_index[row["patient_id"]] for row in test_meta])
    bootstrap_rng = np.random.default_rng(PROBE_BOOTSTRAP_SEED)
    patient_counts = bootstrap_rng.multinomial(
        len(patients),
        np.full(len(patients), 1 / len(patients)),
        size=PROBE_BOOTSTRAP_RESAMPLES,
    ).astype(np.int32)
    counts = patient_counts[:, row_patient]
    real_draws = boot_auroc(real_scores, labels[test], counts)
    control_draws = np.stack(
        [
            boot_auroc(scores, target, counts)
            for scores, target in zip(control_scores, control_labels_test, strict=True)
        ],
        axis=1,
    )
    mean_control_draws = np.nanmean(control_draws, axis=1)
    selectivity_draws = real_draws - mean_control_draws
    control_points = np.asarray([record["auroc"] for record in controls])
    mean_control = float(control_points.mean())

    out_dir.mkdir(parents=True, exist_ok=True)
    directions_path = out_dir / "directions.npz"
    np.savez_compressed(
        directions_path,
        names=np.asarray(direction_names),
        vectors=np.stack(vectors).astype(np.float32),
        projection=projection.astype(np.float32),
        scale=scaler.scale_.astype(np.float64),
        coefficients=np.stack(coefficients).astype(np.float64),
        locus=np.asarray(LOCUS),
        raw_dim=np.asarray(projection.shape[0], dtype=np.int64),
        projection_dim=np.asarray(512, dtype=np.int64),
        projection_seed=np.asarray(0, dtype=np.int64),
        C=np.asarray(1.0),
        probe_seed=np.asarray(0, dtype=np.int64),
    )
    bootstrap_path = out_dir / "probe_scores_and_bootstrap.npz"
    np.savez_compressed(
        bootstrap_path,
        test_row_id=np.asarray([ids[index] for index in np.flatnonzero(test)], dtype=object),
        consolidation_label=labels[test],
        real_scores=real_scores,
        control_scores=np.stack(control_scores),
        control_labels=np.stack(control_labels_test),
        real_bootstrap=real_draws,
        mean_control_bootstrap=mean_control_draws,
        selectivity_bootstrap=selectivity_draws,
    )
    real_ci = pct_ci(real_draws)
    control_ci = pct_ci(mean_control_draws)
    selectivity_ci = pct_ci(selectivity_draws)
    write_json(
        out_dir / "probe.json",
        {
            "arch": ARCH,
            "concept": CONCEPT,
            "locus": LOCUS,
            "module": EXPECTED_LOCUS,
            "projection_dim": 512,
            "projection_seed": 0,
            "C": 1.0,
            "probe_seed": 0,
            "direction_names": direction_names,
            "directions": os.fspath(directions_path),
            "directions_sha256": sha256(directions_path),
            "bootstrap_sha256": sha256(bootstrap_path),
            "control_seeds": list(range(20)),
            "bootstrap_resamples": PROBE_BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": PROBE_BOOTSTRAP_SEED,
            "bootstrap_unit": "patient",
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "n_test_patients": len(patients),
            "test_positive": int(labels[test].sum()),
            "real_auroc": real_auroc,
            "real_auroc_ci95": list(real_ci),
            "controls": controls,
            "control_auroc_mean": mean_control,
            "control_auroc_ci95_for_seed_mean": list(control_ci),
            "control_auroc_spread": {
                "min": float(control_points.min()),
                "max": float(control_points.max()),
                "sd": float(control_points.std(ddof=1)),
                "p05": float(np.percentile(control_points, 5)),
                "p95": float(np.percentile(control_points, 95)),
            },
            "real_minus_mean_control": real_auroc - mean_control,
            "selectivity_ci95": list(selectivity_ci),
            "availability_eligible": bool(selectivity_ci[0] > 0),
            "source_commit": os.environ.get("SOURCE_COMMIT", "unknown"),
        },
    )


def _direction_names() -> tuple[str, ...]:
    return (
        "concept",
        *(f"random{index}" for index in range(N_RANDOM)),
        "sham",
        *(f"unrelated_{concept}" for concept in CLINICAL_CONTROLS),
    )


def summarize_input_closure(
    input_displacement: np.ndarray,
    positive_probability: np.ndarray,
    negative_probability: np.ndarray,
    steered_probability: dict[str, np.ndarray],
    *,
    availability_eligible: bool,
    alpha: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Compute the preregistered paired input-closure statistic and controls."""
    expected = set(_direction_names())
    if set(steered_probability) != expected:
        raise ValueError("direction grid is incomplete or contains unregistered conditions")
    arrays = {
        "input_displacement": np.asarray(input_displacement, dtype=np.float64),
        "positive_probability": np.asarray(positive_probability, dtype=np.float64),
        "negative_probability": np.asarray(negative_probability, dtype=np.float64),
        "alpha": np.asarray(alpha, dtype=np.float64),
    }
    arrays.update(
        {name: np.asarray(values, dtype=np.float64) for name, values in steered_probability.items()}
    )
    if any(values.shape != (EVAL_PAIRS,) for values in arrays.values()):
        raise ValueError("input-closure arrays must contain exactly 50 paired patients")
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("input-closure arrays must be finite")
    probability_names = {"positive_probability", "negative_probability", *expected}
    if any(
        np.any((arrays[name] < 0) | (arrays[name] > 1)) for name in probability_names
    ):
        raise ValueError("input-closure probabilities must lie in [0, 1]")
    if np.any(np.abs(arrays["alpha"]) > MAX_ABS_ALPHA + 1e-12):
        raise ValueError("input-closure alpha exceeds the registered safety cap")

    baseline_gap = np.abs(arrays["positive_probability"] - arrays["negative_probability"])
    # Preserve the preregistered direction order in the bootstrap archive.  Iterating
    # over ``expected`` would retain the same values but make the NPZ member order
    # depend on Python's per-process hash seed, preventing byte-stable replay.
    gains = {
        name: baseline_gap - np.abs(arrays["positive_probability"] - arrays[name])
        for name in _direction_names()
    }
    means = {name: float(values.mean()) for name, values in gains.items()}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(0, EVAL_PAIRS, size=(BOOTSTRAP_RESAMPLES, EVAL_PAIRS))
    gain_bootstrap = {
        name: values[sampled].mean(axis=1) for name, values in gains.items()
    }
    unrelated_names = [f"unrelated_{concept}" for concept in CLINICAL_CONTROLS]
    unrelated_bootstrap = np.stack([gain_bootstrap[name] for name in unrelated_names], axis=1)
    clinical_margin_draws = gain_bootstrap["concept"] - unrelated_bootstrap.max(axis=1)
    clinical_point = max(unrelated_names, key=means.__getitem__)
    clinical_margin = means["concept"] - means[clinical_point]
    displacement_draws = arrays["input_displacement"][sampled].mean(axis=1)
    random_names = [f"random{index}" for index in range(N_RANDOM)]
    random_means = np.asarray([means[name] for name in random_names])
    random_p = float((1 + np.count_nonzero(random_means >= means["concept"])) / (N_RANDOM + 1))
    closure = bool(
        availability_eligible
        and float(np.percentile(displacement_draws, 5)) > 0
        and means["concept"] > 0
        and means["concept"] > float(random_means.max())
        and means["concept"] > means["sham"]
        and float(np.percentile(clinical_margin_draws, 5)) > 0
    )
    summary = {
        "metric": (
            "paired reduction in absolute P(yes) gap between a positive study and its "
            "same-patient negative study"
        ),
        "availability_eligible": bool(availability_eligible),
        "n_pairs": EVAL_PAIRS,
        "n_patients": EVAL_PAIRS,
        "n_random": N_RANDOM,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "patient pair",
        "max_abs_alpha": MAX_ABS_ALPHA,
        "alpha": {
            "min": float(arrays["alpha"].min()),
            "median": float(np.median(arrays["alpha"])),
            "max": float(arrays["alpha"].max()),
            "fraction_at_cap": float(np.mean(np.isclose(np.abs(arrays["alpha"]), MAX_ABS_ALPHA))),
        },
        "input_displacement": {
            "estimate": float(arrays["input_displacement"].mean()),
            "one_sided_lower_95": float(np.percentile(displacement_draws, 5)),
        },
        "baseline_absolute_probability_gap": float(baseline_gap.mean()),
        "direction_closure_gain": means,
        "concept_closure_gain": means["concept"],
        "random_closure_gain_max": float(random_means.max()),
        "exact_random_rank_p": random_p,
        "sham_closure_gain": means["sham"],
        "maximum_unrelated": {
            "concept": clinical_point.removeprefix("unrelated_"),
            "closure_gain": means[clinical_point],
        },
        "clinical_familywise_margin": {
            "estimate": clinical_margin,
            "one_sided_lower_95": float(np.percentile(clinical_margin_draws, 5)),
            "maximum_recomputed_inside_each_bootstrap": True,
        },
        "input_closure": closure,
    }
    output_arrays = {
        "patient_draw_indices": sampled,
        "input_displacement_bootstrap": displacement_draws,
        "clinical_familywise_margin_bootstrap": clinical_margin_draws,
        **{f"closure_gain_{name}": values for name, values in gains.items()},
        **{f"closure_gain_bootstrap_{name}": values for name, values in gain_bootstrap.items()},
    }
    return summary, output_arrays


def run_gpu_evaluation(
    acts_dir: Path,
    manifest: Path,
    source_reference: Path,
    pairs_path: Path,
    directions_path: Path,
    gpu: int,
    out_dir: Path,
) -> None:
    try:
        from .gpu_env import bind_gpu
        from .intervene import NormProbe, Steerer, load_registered_directions, yes_margin_batch, yes_prob_batch
        from .loci import loci_for
        from .registry import REGISTRY
    except ImportError:
        from gpu_env import bind_gpu
        from intervene import NormProbe, Steerer, load_registered_directions, yes_margin_batch, yes_prob_batch
        from loci import loci_for
        from registry import REGISTRY

    bind_gpu(gpu)
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    source = json.loads(source_reference.read_text(encoding="utf-8"))
    pair_receipt = json.loads(pairs_path.read_text(encoding="utf-8"))
    pairs = pair_receipt.get("pairs")
    if pair_receipt.get("pairs_sha256") != REGISTERED_PAIRS_SHA256 or not isinstance(pairs, list):
        raise ValueError("registered pair receipt mismatch")
    ids, activations = load_single_locus(acts_dir)
    positions = {row_id: index for index, row_id in enumerate(ids)}
    manifest_by_id = {
        row["row_id"]: row
        for row in csv.DictReader(manifest.open(newline="", encoding="utf-8"))
    }
    required_rows = {
        item[key]
        for item in pairs
        for key in ("positive_row_id", "negative_row_id")
    }
    if any(row_id not in positions or row_id not in manifest_by_id for row_id in required_rows):
        raise ValueError("registered pair is absent from accepted activations or manifest")

    registered, directions_sha256 = load_registered_directions(
        directions_path, activations.shape[1], CONCEPT
    )
    rng = np.random.default_rng(0)
    directions: dict[str, np.ndarray] = {"concept": registered[CONCEPT]}
    for index in range(N_RANDOM):
        vector = rng.standard_normal(activations.shape[1]).astype(np.float32)
        directions[f"random{index}"] = vector / np.linalg.norm(vector)
    sham = registered[CONCEPT][rng.permutation(activations.shape[1])]
    directions["sham"] = sham / np.linalg.norm(sham)
    directions.update(
        {f"unrelated_{name}": registered[name] for name in CLINICAL_CONTROLS}
    )

    arch = REGISTRY[ARCH]
    locus = next(item for item in loci_for(arch) if item.name == LOCUS)
    if locus.module != EXPECTED_LOCUS:
        raise ValueError("registered Qwen locus changed")
    snapshot = Path(source["model_source"])
    processor = AutoProcessor.from_pretrained(
        snapshot, local_files_only=True, **arch.processor_kwargs
    )
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True
    ).eval()
    tokenizer = getattr(processor, "tokenizer", processor)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    records: list[dict[str, object]] = []
    steerer = Steerer(model, locus.module, positions=locus.positions, mode="reltoken")
    with steerer:
        for pair_index, pair in enumerate(pairs):
            positive_row = manifest_by_id[pair["positive_row_id"]]
            negative_row = manifest_by_id[pair["negative_row_id"]]
            with Image.open(positive_row["image_path"]) as image:
                positive_inputs = processor(
                    text=[prompt_text], images=[image.convert("RGB")], return_tensors="pt"
                ).to("cuda:0")
            with Image.open(negative_row["image_path"]) as image:
                negative_inputs = processor(
                    text=[prompt_text], images=[image.convert("RGB")], return_tensors="pt"
                ).to("cuda:0")

            steerer.vec = None
            steerer.alpha = 0.0
            with torch.inference_mode():
                positive_logits = model(**positive_inputs).logits
            norm_probe = NormProbe(model, locus.module, positions=locus.positions)
            with norm_probe, torch.inference_mode():
                negative_logits = model(**negative_inputs).logits
            norm_stats = norm_probe.stats()
            token_norm_mean = float(norm_stats["mean"])
            if not np.isfinite(token_norm_mean) or token_norm_mean <= 0:
                raise ValueError("negative-study token norm is invalid")
            positive_probability = float(yes_prob_batch(positive_logits, tokenizer)[0])
            negative_probability = float(yes_prob_batch(negative_logits, tokenizer)[0])
            positive_margin = float(yes_margin_batch(positive_logits, tokenizer)[0])
            negative_margin = float(yes_margin_batch(negative_logits, tokenizer)[0])
            displacement = float(
                np.dot(
                    activations[positions[pair["positive_row_id"]]]
                    - activations[positions[pair["negative_row_id"]]],
                    registered[CONCEPT],
                )
            )
            alpha = float(np.clip(displacement / token_norm_mean, -MAX_ABS_ALPHA, MAX_ABS_ALPHA))

            for name, vector in directions.items():
                steerer.vec = vector
                steerer.alpha = alpha
                with torch.inference_mode():
                    logits = model(**negative_inputs).logits
                records.append(
                    {
                        "pair_index": pair_index,
                        "patient_id": pair["patient_id"],
                        "positive_row_id": pair["positive_row_id"],
                        "negative_row_id": pair["negative_row_id"],
                        "direction": name,
                        "alpha": alpha,
                        "input_displacement": displacement,
                        "token_norm_mean": token_norm_mean,
                        "positive_p_yes": positive_probability,
                        "negative_p_yes": negative_probability,
                        "steered_p_yes": float(yes_prob_batch(logits, tokenizer)[0]),
                        "positive_margin": positive_margin,
                        "negative_margin": negative_margin,
                        "steered_margin": float(yes_margin_batch(logits, tokenizer)[0]),
                    }
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per-pair.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, csv_path)
    write_json(
        out_dir / "meta.json",
        {
            "arch": ARCH,
            "concept": CONCEPT,
            "prompt": PROMPT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_source": os.fspath(snapshot),
            "model_local_only": True,
            "locus": LOCUS,
            "module": EXPECTED_LOCUS,
            "alpha_mode": "reltoken",
            "dose_rule": (
                "clip((positive-minus-negative pooled vis.last displacement along the "
                "Consolidation normal) / negative-study mean token norm, -0.25, 0.25)"
            ),
            "max_abs_alpha": MAX_ABS_ALPHA,
            "n_pairs": EVAL_PAIRS,
            "n_random": N_RANDOM,
            "directions_sha256": directions_sha256,
            "pairs_sha256": REGISTERED_PAIRS_SHA256,
            "source_commit": os.environ.get("SOURCE_COMMIT", "unknown"),
        },
    )
    write_json(
        out_dir / "DONE",
        {
            "pairs": EVAL_PAIRS,
            "directions_per_pair": len(_direction_names()),
            "intervention_rows": len(records),
            "image_condition_outcomes": EVAL_PAIRS * (len(_direction_names()) + 2),
        },
    )


def summarize_run(
    per_pair_path: Path,
    meta_path: Path,
    done_path: Path,
    probe_path: Path,
    pairs_path: Path,
    bootstrap_out: Path,
    out: Path,
) -> None:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    expected_meta = {
        "arch": ARCH,
        "concept": CONCEPT,
        "prompt": PROMPT,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "locus": LOCUS,
        "module": EXPECTED_LOCUS,
        "alpha_mode": "reltoken",
        "max_abs_alpha": MAX_ABS_ALPHA,
        "n_pairs": EVAL_PAIRS,
        "n_random": N_RANDOM,
        "pairs_sha256": REGISTERED_PAIRS_SHA256,
    }
    if any(meta.get(key) != value for key, value in expected_meta.items()):
        raise ValueError("input-closure GPU metadata mismatch")
    if (
        done.get("pairs") != EVAL_PAIRS
        or done.get("directions_per_pair") != len(_direction_names())
        or done.get("intervention_rows") != EVAL_PAIRS * len(_direction_names())
        or done.get("image_condition_outcomes") != EVAL_PAIRS * (len(_direction_names()) + 2)
    ):
        raise ValueError("input-closure completion marker mismatch")
    expected_probe = {
        "arch": ARCH,
        "concept": CONCEPT,
        "locus": LOCUS,
        "module": EXPECTED_LOCUS,
        "projection_dim": 512,
        "projection_seed": 0,
        "C": 1.0,
        "probe_seed": 0,
        "control_seeds": list(range(20)),
        "bootstrap_resamples": PROBE_BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": PROBE_BOOTSTRAP_SEED,
        "bootstrap_unit": "patient",
        "direction_names": [CONCEPT, *CLINICAL_CONTROLS],
    }
    if any(probe.get(key) != value for key, value in expected_probe.items()):
        raise ValueError("input-closure probe metadata mismatch")
    if pairs.get("pairs_sha256") != REGISTERED_PAIRS_SHA256:
        raise ValueError("input-closure pair receipt mismatch")

    records = list(csv.DictReader(per_pair_path.open(newline="", encoding="utf-8")))
    expected_directions = set(_direction_names())
    by_pair: dict[int, dict[str, dict[str, str]]] = {}
    for record in records:
        index = int(record["pair_index"])
        direction = record["direction"]
        if index not in range(EVAL_PAIRS) or direction not in expected_directions:
            raise ValueError("input-closure per-pair identity mismatch")
        if direction in by_pair.setdefault(index, {}):
            raise ValueError("duplicate input-closure direction")
        by_pair[index][direction] = record
    if set(by_pair) != set(range(EVAL_PAIRS)) or any(
        set(items) != expected_directions for items in by_pair.values()
    ):
        raise ValueError("input-closure per-pair grid is incomplete")

    displacement = np.empty(EVAL_PAIRS)
    positive = np.empty(EVAL_PAIRS)
    negative = np.empty(EVAL_PAIRS)
    positive_margin = np.empty(EVAL_PAIRS)
    negative_margin = np.empty(EVAL_PAIRS)
    alpha = np.empty(EVAL_PAIRS)
    steered = {name: np.empty(EVAL_PAIRS) for name in expected_directions}
    steered_margin = {name: np.empty(EVAL_PAIRS) for name in expected_directions}
    for index, pair in enumerate(pairs["pairs"]):
        items = by_pair[index]
        reference = items["concept"]
        if (
            reference["patient_id"] != pair["patient_id"]
            or reference["positive_row_id"] != pair["positive_row_id"]
            or reference["negative_row_id"] != pair["negative_row_id"]
        ):
            raise ValueError("input-closure pair order changed")
        displacement[index] = float(reference["input_displacement"])
        positive[index] = float(reference["positive_p_yes"])
        negative[index] = float(reference["negative_p_yes"])
        positive_margin[index] = float(reference["positive_margin"])
        negative_margin[index] = float(reference["negative_margin"])
        alpha[index] = float(reference["alpha"])
        for name, record in items.items():
            for field, value in (
                ("input_displacement", displacement[index]),
                ("positive_p_yes", positive[index]),
                ("negative_p_yes", negative[index]),
                ("alpha", alpha[index]),
            ):
                if not np.isclose(float(record[field]), value, rtol=0, atol=1e-12):
                    raise ValueError("input-closure repeated pair values disagree")
            steered[name][index] = float(record["steered_p_yes"])
            steered_margin[name][index] = float(record["steered_margin"])

    summary, arrays = summarize_input_closure(
        displacement,
        positive,
        negative,
        steered,
        availability_eligible=probe.get("availability_eligible") is True,
        alpha=alpha,
    )
    if not all(
        np.isfinite(values).all()
        for values in (positive_margin, negative_margin, *steered_margin.values())
    ):
        raise ValueError("input-closure margin values must be finite")
    baseline_margin_gap = np.abs(positive_margin - negative_margin)
    margin_gains = {
        name: baseline_margin_gap - np.abs(positive_margin - values)
        for name, values in steered_margin.items()
    }
    summary["secondary_margin_diagnostic"] = {
        "baseline_absolute_gap": float(baseline_margin_gap.mean()),
        "concept_closure_gain": float(margin_gains["concept"].mean()),
        "random_closure_gain_max": float(
            max(margin_gains[f"random{index}"].mean() for index in range(N_RANDOM))
        ),
        "sham_closure_gain": float(margin_gains["sham"].mean()),
        "maximum_unrelated_closure_gain": float(
            max(margin_gains[f"unrelated_{name}"].mean() for name in CLINICAL_CONTROLS)
        ),
        "threshold_role": "reported only",
    }
    bootstrap_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = bootstrap_out.with_suffix(bootstrap_out.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, bootstrap_out)
    summary.update(
        {
            "arch": ARCH,
            "concept": CONCEPT,
            "prompt": PROMPT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "locus": LOCUS,
            "module": EXPECTED_LOCUS,
            "alpha_mode": "reltoken",
            "source_run_id": SOURCE_RUN_ID,
            "source_commit": SOURCE_COMMIT,
            "ownership_run_id": OWNERSHIP_RUN_ID,
            "intervention_commit": os.environ.get("SOURCE_COMMIT", "unknown"),
            "registered_pairs_sha256": REGISTERED_PAIRS_SHA256,
            "probe": {
                "real_auroc": probe["real_auroc"],
                "selectivity": probe["real_minus_mean_control"],
                "selectivity_ci95": probe["selectivity_ci95"],
                "availability_eligible": probe["availability_eligible"],
            },
            "evidence": {
                "per_pair_sha256": sha256(per_pair_path),
                "meta_sha256": sha256(meta_path),
                "completion_marker_sha256": sha256(done_path),
                "probe_sha256": sha256(probe_path),
                "pairs_receipt_sha256": sha256(pairs_path),
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
    validate.add_argument("--ownership-run", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--out", type=Path, required=True)
    validate.add_argument("--pairs-out", type=Path, required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--acts", type=Path, required=True)
    probe.add_argument("--manifest", type=Path, required=True)
    probe.add_argument("--out", type=Path, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--acts", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--source-reference", type=Path, required=True)
    evaluate.add_argument("--pairs", type=Path, required=True)
    evaluate.add_argument("--directions", type=Path, required=True)
    evaluate.add_argument("--gpu", type=int, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--per-pair", type=Path, required=True)
    summarize.add_argument("--meta", type=Path, required=True)
    summarize.add_argument("--done", type=Path, required=True)
    summarize.add_argument("--probe", type=Path, required=True)
    summarize.add_argument("--pairs", type=Path, required=True)
    summarize.add_argument("--bootstrap", type=Path, required=True)
    summarize.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate-source":
        validate_source(args.source_run, args.ownership_run, args.manifest, args.out, args.pairs_out)
    elif args.command == "probe":
        run_probe(args.acts, args.manifest, args.out)
    elif args.command == "evaluate":
        run_gpu_evaluation(
            args.acts,
            args.manifest,
            args.source_reference,
            args.pairs,
            args.directions,
            args.gpu,
            args.out,
        )
    else:
        summarize_run(
            args.per_pair,
            args.meta,
            args.done,
            args.probe,
            args.pairs,
            args.bootstrap,
            args.out,
        )


if __name__ == "__main__":
    main()
