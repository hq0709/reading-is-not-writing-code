"""Pure-CPU protocol and inference for the LLaVA Effusion readout diagnostic."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.bootstrap_probe import boot_auroc
from src.build_manifest import CONCEPTS as NIH_CONCEPTS
from src.build_manifest import patient_split
from src.probe import build_types

N_PAIRS = 700
N_BOOTSTRAP = 10_000
ALLOCATION_SEED = 20260916
BOOTSTRAP_SEED = 20260917
MODULE = "model.vision_tower.encoder.layers.22"
WORDINGS = (
    "Is there a pleural effusion in this chest radiograph?",
    "Does this chest radiograph show a pleural effusion?",
)
ENCODINGS = ("Y", "A", "B")
SUFFIXES = {
    "Y": "Answer yes or no.",
    "A": "Answer A if the finding is present and B if it is absent. Reply with A or B only.",
    "B": "Answer B if the finding is present and A if it is absent. Reply with A or B only.",
}
CONDITIONS = tuple(
    {
        "name": f"{wording}{encoding}",
        "wording": wording,
        "encoding": encoding,
        "prompt": f"{WORDINGS[wording]} {SUFFIXES[encoding]}",
    }
    for wording in range(2)
    for encoding in ENCODINGS
)
PRIMARY_NAMES = (
    "accepted_condition_image_advantage",
    "wording_effect",
    "symbol_vs_yes_no_effect",
    "mapping_asymmetry",
    "wording_by_encoding",
    "wording_by_mapping",
)
FIELDS = (
    "source_commit",
    "condition",
    "wording",
    "encoding",
    "prompt",
    "image_role",
    "pair_index",
    "patient_id",
    "row_id",
    "image_path",
    "index_label",
    "source_label",
    "raw_margin",
    "semantic_margin",
    "probability",
    "answer_token_mass",
    "lse_margin",
)


def finite(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected finite numeric {shape}")
    return array


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    z = np.exp(-np.abs(value))
    return np.where(value >= 0, 1 / (1 + z), z / (1 + z))


def prompt(condition):
    return CONDITIONS[condition]["prompt"]


def _record(raw, image_directory):
    findings = set(raw["Finding Labels"].split("|"))
    try:
        age = int(raw["Patient Age"])
    except (TypeError, ValueError) as error:
        raise ValueError("invalid NIH patient age") from error
    filename = raw["Image Index"]
    return {
        "row_id": filename.removesuffix(".png"),
        "image_index": filename,
        "image_path": str(Path(image_directory) / filename),
        "patient_id": str(raw["Patient ID"]),
        "split": patient_split(str(raw["Patient ID"])),
        "age": age,
        "view_AP": int(raw["View Position"] == "AP"),
        "sex_M": int(raw["Patient Sex"] == "M"),
        "no_finding": int("No Finding" in findings),
        "finding_labels": raw["Finding Labels"],
        **{concept: int(concept in findings) for concept in NIH_CONCEPTS},
        "source_fields": dict(raw),
    }


def allocate(raw_rows, manifest_rows, image_directory):
    """Allocate identifiers before attaching labels or opening images."""
    manifest_patients = {str(row["patient_id"]) for row in manifest_rows}
    if len({row["row_id"] for row in manifest_rows}) != len(manifest_rows):
        raise ValueError("accepted manifest contains duplicate rows")
    raw_rows = list(raw_rows)
    required = {
        "Image Index",
        "Finding Labels",
        "Patient ID",
        "Patient Age",
        "Patient Sex",
        "View Position",
    }
    if not raw_rows or any(not required.issubset(row) for row in raw_rows):
        raise ValueError("NIH source metadata schema mismatch")
    if len({row["Image Index"] for row in raw_rows}) != len(raw_rows):
        raise ValueError("NIH source metadata contains duplicate images")

    # Candidate selection consults identifiers and the fixed split only. Labels are attached below.
    representatives = {}
    for raw in raw_rows:
        pid = str(raw["Patient ID"])
        if pid in manifest_patients or patient_split(pid) != "val":
            continue
        if pid not in representatives or raw["Image Index"] < representatives[pid]["Image Index"]:
            representatives[pid] = raw
    if len(representatives) < 2 * N_PAIRS:
        raise ValueError("UNAVAILABLE: fewer than 1,400 independent validation patients")
    order = np.random.default_rng(ALLOCATION_SEED).permutation(sorted(representatives)).tolist()
    selected = [_record(representatives[pid], image_directory) for pid in order[: 2 * N_PAIRS]]
    return {
        "index": selected[:N_PAIRS],
        "donor": selected[N_PAIRS:],
        "unused_patient_ids": order[2 * N_PAIRS :],
        "eligible_patient_ids": order,
        "selection_seed": ALLOCATION_SEED,
        "pairing": "allocation position",
    }


def validate_allocation(allocation, *, check_files=False):
    index, donor = allocation["index"], allocation["donor"]
    if (
        len(index) != N_PAIRS
        or len(donor) != N_PAIRS
        or allocation.get("selection_seed") != ALLOCATION_SEED
        or len(allocation.get("eligible_patient_ids", [])) != 2223
        or len(allocation.get("unused_patient_ids", [])) != 823
    ):
        raise ValueError("registered allocation size or seed mismatch")
    identities = [str(row["patient_id"]) for row in index + donor]
    if len(set(identities)) != 2 * N_PAIRS or any(row["split"] != "val" for row in index + donor):
        raise ValueError("index/donor patients must be distinct validation patients")
    if [sum(int(row["Effusion"]) for row in group) for group in (index, donor)] != [25, 22] or [
        (
            str(group[0]["patient_id"]),
            group[0]["image_index"],
            str(group[-1]["patient_id"]),
            group[-1]["image_index"],
        )
        for group in (index, donor)
    ] != [
        ("19570", "00019570_000.png", "28163", "00028163_000.png"),
        ("15575", "00015575_000.png", "12195", "00012195_000.png"),
    ]:
        raise ValueError("registered allocation anchors or Effusion support mismatch")
    if check_files and any(not Path(row["image_path"]).is_file() for row in index + donor):
        raise ValueError("UNAVAILABLE: a registered diagnostic image is missing")
    return True


def bootstrap_indices():
    return np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, N_PAIRS, size=(N_BOOTSTRAP, N_PAIRS), dtype=np.int64
    )


def validate_bootstrap(indices):
    indices = np.asarray(indices)
    if indices.dtype != np.int64 or not np.array_equal(indices, bootstrap_indices()):
        raise ValueError("registered pair-bootstrap identity mismatch")
    return indices


def multiplicities(indices):
    return np.stack([np.bincount(draw, minlength=N_PAIRS) for draw in indices])


def auc_point_draws(scores, labels, counts):
    scores = finite(scores, (N_PAIRS,), "AUROC scores")
    labels = finite(labels, (N_PAIRS,), "AUROC labels").astype(np.int8)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("AUROC labels must be binary")
    all_counts = np.vstack((np.ones(N_PAIRS, dtype=np.int64), counts))
    values = boot_auroc(scores, labels, all_counts)
    return float(values[0]), values[1:]


def primary_contrasts(advantages):
    values = np.asarray(advantages)
    if values.shape[-2:] != (2, 3):
        raise ValueError("image advantages must end in a 2x3 wording/encoding grid")
    s = (values[..., 1] + values[..., 2]) / 2
    return np.stack(
        (
            values[..., 0, 0],
            (values[..., 1, :] - values[..., 0, :]).mean(axis=-1),
            ((s - values[..., :, 0]).mean(axis=-1)),
            (values[..., :, 1] - values[..., :, 2]).mean(axis=-1),
            (s[..., 1] - values[..., 1, 0]) - (s[..., 0] - values[..., 0, 0]),
            (values[..., 1, 1] - values[..., 1, 2]) - (values[..., 0, 1] - values[..., 0, 2]),
        ),
        axis=-1,
    )


def answer_statistics(semantic_margin, index_labels, donor_labels, indices):
    semantic = finite(semantic_margin, (6, 2, N_PAIRS), "semantic margins")
    labels = finite(index_labels, (N_PAIRS,), "index labels").astype(np.int8)
    donors = finite(donor_labels, (N_PAIRS,), "donor labels").astype(np.int8)
    indices = validate_bootstrap(indices)
    counts = multiplicities(indices)
    point = np.empty((6, 2))
    draws = np.empty((6, 2, N_BOOTSTRAP))
    donor_own = np.empty(6)
    for c in range(6):
        for role in range(2):
            point[c, role], draws[c, role] = auc_point_draws(semantic[c, role], labels, counts)
        donor_own[c], _ = auc_point_draws(semantic[c, 1], donors, counts)
    advantage = (point[:, 0] - point[:, 1]).reshape(2, 3)
    advantage_draws = (draws[:, 0] - draws[:, 1]).T.reshape(N_BOOTSTRAP, 2, 3)
    theta = primary_contrasts(advantage)
    theta_draws = primary_contrasts(advantage_draws)
    valid = np.isfinite(theta_draws).all(axis=1)
    valid_n = int(valid.sum())
    if valid_n < 9500:
        raise ValueError("fewer than 9,500 common valid primary bootstrap draws")
    deviation = np.max(np.abs(theta_draws[valid] - theta), axis=1)
    radius = float(np.percentile(deviation, 95, method="linear"))
    return {
        "auroc": point,
        "bootstrap_auroc": draws,
        "donor_own_label_auroc": donor_own,
        "image_advantage": advantage,
        "bootstrap_image_advantage": advantage_draws,
        "primary": theta,
        "bootstrap_primary": theta_draws,
        "common_valid": valid,
        "simultaneous_radius": np.asarray(radius),
    }, {
        "common_valid_draws": valid_n,
        "simultaneous_radius": radius,
        "bootstrap_degenerate": bool(radius == 0),
        "contrasts": [
            {
                "name": name,
                "estimate": float(theta[i]),
                "simultaneous_ci95": [float(theta[i] - radius), float(theta[i] + radius)],
            }
            for i, name in enumerate(PRIMARY_NAMES)
        ],
    }


def frozen_control_labels(control_assignments, index_rows):
    """Apply the accepted type assignments without fitting or filling missing types."""
    if len(control_assignments) != 20:
        raise ValueError("twenty frozen control assignments are required")
    types = build_types(index_rows)
    control_labels = []
    for record in control_assignments:
        assignment = record.get("assignment", {})
        if any(kind not in assignment for kind in types):
            raise ValueError("frozen control assignment does not cover the diagnostic cohort")
        control_labels.append([assignment[kind] for kind in types])
    labels = np.asarray(control_labels, dtype=np.int8)
    if labels.shape != (20, N_PAIRS) or not np.isin(labels, (0, 1)).all():
        raise ValueError("frozen control labels must be a complete binary 20-by-cohort matrix")
    return labels


def reader_statistics(raw_activation, prepared, control_assignments, index_rows, indices):
    """Replay the frozen Effusion reader and all twenty accepted control fits."""
    from sklearn.preprocessing import StandardScaler

    raw = finite(raw_activation, (N_PAIRS, 1024), "captured vis.last activations")
    projection = finite(prepared["projection"], (1024, 512), "accepted projection")
    scale = finite(prepared["scale"], (512,), "accepted scale")
    train_mean = finite(prepared["train_mean"], (512,), "accepted train mean")
    coefficients = finite(prepared["coefficients"], (6, 512), "accepted clinical coefficients")
    controls = finite(prepared["control_coefficients"], (20, 512), "accepted control coefficients")
    intercepts = finite(prepared["control_intercepts"], (20,), "accepted control intercepts")
    projected = raw.astype(np.float32) @ projection.astype(np.float32)
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.n_features_in_ = train_mean, scale, 512
    features = scaler.transform(projected)
    clinical_score = features @ coefficients[0]
    control_score = np.stack(
        [
            features @ coefficient + intercept
            for coefficient, intercept in zip(controls, intercepts, strict=True)
        ]
    )
    control_labels = frozen_control_labels(control_assignments, index_rows)
    labels = np.asarray([int(row["Effusion"]) for row in index_rows], dtype=np.int8)
    counts = multiplicities(validate_bootstrap(indices))
    clinical_point, clinical_draws = auc_point_draws(clinical_score, labels, counts)
    control_point, control_draws = [], []
    for score, target in zip(control_score, control_labels, strict=True):
        point, draw = auc_point_draws(score, target, counts)
        control_point.append(point)
        control_draws.append(draw)
    control_point, control_draws = np.asarray(control_point), np.stack(control_draws)
    selectivity = clinical_point - float(control_point.mean())
    selectivity_draws = clinical_draws - control_draws.mean(axis=0)
    selectivity_draws[~np.isfinite(control_draws).all(axis=0) | ~np.isfinite(clinical_draws)] = (
        np.nan
    )
    valid = int(np.isfinite(selectivity_draws).sum())
    ci = np.percentile(
        selectivity_draws[np.isfinite(selectivity_draws)], [2.5, 97.5], method="linear"
    )
    positive = int(labels.sum())
    arrays = {
        "reader_raw_activation": raw,
        "reader_projected": projected,
        "reader_features": features,
        "reader_clinical_score": clinical_score,
        "reader_control_score": control_score,
        "reader_control_labels": control_labels,
        "reader_clinical_auroc": np.asarray(clinical_point),
        "reader_control_auroc": control_point,
        "bootstrap_reader_clinical_auroc": clinical_draws,
        "bootstrap_reader_control_auroc": control_draws,
        "bootstrap_reader_selectivity": selectivity_draws,
    }
    report = {
        "positive": positive,
        "negative": N_PAIRS - positive,
        "clinical_auroc": clinical_point,
        "control_auroc": control_point.tolist(),
        "selectivity": selectivity,
        "selectivity_ci95": ci.tolist(),
        "valid_draws": valid,
        "replicated": bool(min(positive, N_PAIRS - positive) >= 10 and valid >= 9500 and ci[0] > 0),
    }
    return arrays, report


def validate_grid(records, allocation, source_commit):
    """Validate the exact condition/role/pair ordering and recover score arrays."""
    records = iter(records)
    arrays = {
        name: np.empty((6, 2, N_PAIRS), dtype=np.float64)
        for name in (
            "raw_margin",
            "semantic_margin",
            "probability",
            "answer_token_mass",
            "lse_margin",
        )
    }
    for condition, spec in enumerate(CONDITIONS):
        for role_index, role in enumerate(("index", "donor")):
            rows = allocation[role]
            for pair_index, row in enumerate(rows):
                actual = next(records, None)
                if actual is None or set(actual) != set(FIELDS):
                    raise ValueError("missing or malformed diagnostic outcome")
                index = allocation["index"][pair_index]
                expected = {
                    "source_commit": source_commit,
                    "condition": spec["name"],
                    "wording": str(spec["wording"]),
                    "encoding": spec["encoding"],
                    "prompt": spec["prompt"],
                    "image_role": role,
                    "pair_index": str(pair_index),
                    "patient_id": str(row["patient_id"]),
                    "row_id": row["row_id"],
                    "image_path": row["image_path"],
                    "index_label": str(int(index["Effusion"])),
                    "source_label": str(int(row["Effusion"])),
                }
                if any(str(actual.get(key)) != value for key, value in expected.items()):
                    raise ValueError("diagnostic cohort, prompt, or ordering mismatch")
                for name in ("raw_margin", "semantic_margin", "probability", "answer_token_mass"):
                    value = float(actual[name])
                    if not np.isfinite(value):
                        raise ValueError("nonfinite diagnostic score")
                    arrays[name][condition, role_index, pair_index] = value
                lse = actual["lse_margin"]
                if spec["encoding"] == "Y":
                    value = float(lse)
                    if not np.isfinite(value):
                        raise ValueError("nonfinite yes/no log-sum-exp margin")
                    arrays["lse_margin"][condition, role_index, pair_index] = value
                else:
                    if lse not in ("", None):
                        raise ValueError("symbol conditions must not carry a yes/no lse margin")
                    arrays["lse_margin"][condition, role_index, pair_index] = np.nan
                if not np.isclose(
                    arrays["probability"][condition, role_index, pair_index],
                    sigmoid(arrays["semantic_margin"][condition, role_index, pair_index]),
                    atol=1e-12,
                    rtol=1e-12,
                ):
                    raise ValueError("persisted probability differs from semantic sigmoid")
                mass = arrays["answer_token_mass"][condition, role_index, pair_index]
                if not 0 <= mass <= 1:
                    raise ValueError("answer-token probability mass is outside [0,1]")
    if next(records, None) is not None:
        raise ValueError("extra or duplicate diagnostic outcome")
    return arrays
