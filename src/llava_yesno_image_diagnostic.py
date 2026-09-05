"""Pure-CPU protocol and inference for the LLaVA yes/no image diagnostic."""

from __future__ import annotations

import numpy as np

from src import llava_readout_diagnostic as allocation_source

N_PAIRS = 700
N_BOOTSTRAP = 10_000
ALLOCATION_SEED = 20260916
BOOTSTRAP_SEED = 20260917
MODULE = "model.vision_tower.encoder.layers.22"
WORDINGS = (
    "Is there a pleural effusion in this chest radiograph?",
    "Does this chest radiograph show a pleural effusion?",
)
SUFFIX = "Answer yes or no."
CONDITIONS = tuple(
    {
        "name": f"{wording}Y",
        "wording": wording,
        "prompt": f"{stem} {SUFFIX}",
    }
    for wording, stem in enumerate(WORDINGS)
)
PRIMARY_NAMES = (
    "accepted_wording_discrimination_above_chance",
    "accepted_wording_image_advantage",
    "wording_dependent_image_advantage",
)
FIELDS = (
    "source_commit",
    "condition",
    "wording",
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

# The allocation algorithm and frozen-reader contraction are inherited unchanged from
# the prospectively registered parent diagnostic. The new gate binds their outputs to
# its retained cohort and independently checks that binding in preparation/validation.
allocate = allocation_source.allocate
validate_allocation = allocation_source.validate_allocation
finite = allocation_source.finite
sigmoid = allocation_source.sigmoid
frozen_control_labels = allocation_source.frozen_control_labels
multiplicities = allocation_source.multiplicities
auc_point_draws = allocation_source.auc_point_draws


def bootstrap_indices():
    return np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, N_PAIRS, size=(N_BOOTSTRAP, N_PAIRS), dtype=np.int64
    )


def validate_bootstrap(indices):
    indices = np.asarray(indices)
    if indices.dtype != np.int64 or not np.array_equal(indices, bootstrap_indices()):
        raise ValueError("registered pair-bootstrap identity mismatch")
    return indices


def primary_contrasts(auroc, advantages):
    auroc = np.asarray(auroc)
    advantages = np.asarray(advantages)
    if auroc.shape[-1] != 2 or advantages.shape[-1] != 2:
        raise ValueError("answer summaries must end in the two registered wordings")
    return np.stack(
        (
            auroc[..., 0] - 0.5,
            advantages[..., 0],
            advantages[..., 1] - advantages[..., 0],
        ),
        axis=-1,
    )


def _interval(draws):
    draws = np.asarray(draws)
    valid = draws[np.isfinite(draws)]
    if not len(valid):
        return [None, None]
    return np.percentile(valid, [2.5, 97.5], method="linear").tolist()


def primary_decision(theta, theta_draws, common_valid):
    theta = finite(theta, (3,), "primary estimates")
    theta_draws = np.asarray(theta_draws, dtype=np.float64)
    common_valid = np.asarray(common_valid, dtype=bool)
    if (
        theta_draws.ndim != 2
        or theta_draws.shape[1] != 3
        or common_valid.shape != (len(theta_draws),)
    ):
        raise ValueError("primary bootstrap shape mismatch")
    common_valid &= np.isfinite(theta_draws).all(axis=1)
    valid_n = int(common_valid.sum())
    radius = None
    if valid_n:
        deviations = np.max(np.abs(theta_draws[common_valid] - theta), axis=1)
        radius = float(np.percentile(deviations, 95, method="linear"))
    degenerate = radius == 0 if radius is not None else False
    available = bool(valid_n >= 9500 and radius is not None and radius > 0)
    intervals = [
        [float(value - radius), float(value + radius)] if radius is not None else [None, None]
        for value in theta
    ]
    if available:
        opportunity = bool(intervals[0][0] > 0 and intervals[1][0] > 0)
        wording = (
            "positive"
            if intervals[2][0] > 0
            else "negative"
            if intervals[2][1] < 0
            else "unresolved"
        )
        unavailable_reason = None
    else:
        opportunity = None
        wording = None
        unavailable_reason = (
            "insufficient_common_bootstrap_draws" if valid_n < 9500 else "bootstrap_degenerate"
        )
    return {
        "common_valid_draws": valid_n,
        "population_inference": "AVAILABLE" if available else "UNAVAILABLE",
        "unavailable_reason": unavailable_reason,
        "simultaneous_radius": radius,
        "bootstrap_degenerate": degenerate,
        "opportunity_detected": opportunity,
        "wording_dependence": wording,
        "contrasts": [
            {
                "name": name,
                "estimate": float(theta[i]),
                "simultaneous_ci95": intervals[i],
            }
            for i, name in enumerate(PRIMARY_NAMES)
        ],
    }


def answer_statistics(semantic_margin, index_labels, donor_labels, indices):
    semantic = finite(semantic_margin, (2, 2, N_PAIRS), "semantic margins")
    labels = finite(index_labels, (N_PAIRS,), "index labels").astype(np.int8)
    donors = finite(donor_labels, (N_PAIRS,), "donor labels").astype(np.int8)
    if not np.isin(labels, (0, 1)).all() or not np.isin(donors, (0, 1)).all():
        raise ValueError("answer labels must be binary")
    counts = multiplicities(validate_bootstrap(indices))
    point = np.empty((2, 2))
    draws = np.empty((2, 2, N_BOOTSTRAP))
    donor_own = np.empty(2)
    for condition in range(2):
        for role in range(2):
            point[condition, role], draws[condition, role] = auc_point_draws(
                semantic[condition, role], labels, counts
            )
        donor_own[condition], _ = auc_point_draws(semantic[condition, 1], donors, counts)

    advantage = point[:, 0] - point[:, 1]
    advantage_draws = (draws[:, 0] - draws[:, 1]).T
    theta = primary_contrasts(point[:, 0], advantage)
    theta_draws = primary_contrasts(draws[:, 0].T, advantage_draws)
    common_valid = np.isfinite(draws).all(axis=(0, 1)) & np.isfinite(theta_draws).all(axis=1)
    report = primary_decision(theta, theta_draws, common_valid)
    radius = report["simultaneous_radius"]
    arrays = {
        "auroc": point,
        "bootstrap_auroc": draws,
        "donor_own_label_auroc": donor_own,
        "image_advantage": advantage,
        "bootstrap_image_advantage": advantage_draws,
        "primary": theta,
        "bootstrap_primary": theta_draws,
        "common_valid": common_valid,
        "simultaneous_radius": np.asarray(np.nan if radius is None else radius),
    }
    return arrays, report


def reader_decision(positive, negative, valid_draws, selectivity_ci95):
    support = min(positive, negative) >= 10
    enough_draws = valid_draws >= 9500
    available = bool(support and enough_draws)
    return {
        "population_inference": "AVAILABLE" if available else "UNAVAILABLE",
        "unavailable_reason": None
        if available
        else "insufficient_class_support"
        if not support
        else "insufficient_valid_bootstrap_draws",
        "replicated": bool(available and selectivity_ci95[0] > 0),
    }


def frozen_reader_scores(raw_activation, prepared):
    """Apply every accepted reader fit separately, preserving float32 GEMV order."""
    from sklearn.preprocessing import StandardScaler

    raw = np.asarray(raw_activation)
    if raw.ndim != 2 or raw.shape[1] != 1024 or not np.isfinite(raw).all():
        raise ValueError("captured vis.last activations must be finite N-by-1024")
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
    return projected, features, clinical_score, control_score


def reader_statistics(raw_activation, prepared, control_assignments, index_rows, indices):
    raw = finite(raw_activation, (N_PAIRS, 1024), "captured vis.last activations")
    projected, features, clinical_score, control_score = frozen_reader_scores(raw, prepared)
    control_labels = frozen_control_labels(control_assignments, index_rows)
    labels = np.asarray([int(row["Effusion"]) for row in index_rows], dtype=np.int8)
    counts = multiplicities(validate_bootstrap(indices))
    clinical_point, clinical_draws = auc_point_draws(clinical_score, labels, counts)
    control_point, control_draws = [], []
    for score, target in zip(control_score, control_labels, strict=True):
        point, draws = auc_point_draws(score, target, counts)
        control_point.append(point)
        control_draws.append(draws)
    control_point, control_draws = np.asarray(control_point), np.stack(control_draws)
    selectivity = clinical_point - float(control_point.mean())
    selectivity_draws = clinical_draws - control_draws.mean(axis=0)
    selectivity_draws[~np.isfinite(control_draws).all(axis=0) | ~np.isfinite(clinical_draws)] = (
        np.nan
    )
    valid = int(np.isfinite(selectivity_draws).sum())
    ci = _interval(selectivity_draws)
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
        "selectivity_ci95": ci,
        "valid_draws": valid,
    }
    report.update(
        reader_decision(report["positive"], report["negative"], report["valid_draws"], ci)
    )
    return arrays, report


def validate_grid(records, allocation, source_commit):
    records = iter(records)
    arrays = {
        name: np.empty((2, 2, N_PAIRS), dtype=np.float64)
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
            for pair_index, row in enumerate(allocation[role]):
                actual = next(records, None)
                if actual is None or set(actual) != set(FIELDS):
                    raise ValueError("missing or malformed yes/no image outcome")
                index = allocation["index"][pair_index]
                expected = {
                    "source_commit": source_commit,
                    "condition": spec["name"],
                    "wording": str(spec["wording"]),
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
                    raise ValueError("yes/no cohort, prompt, label, or ordering mismatch")
                for name, array in arrays.items():
                    value = float(actual[name])
                    if not np.isfinite(value):
                        raise ValueError("nonfinite yes/no image score")
                    array[condition, role_index, pair_index] = value
                if (
                    arrays["raw_margin"][condition, role_index, pair_index]
                    != arrays["semantic_margin"][condition, role_index, pair_index]
                ):
                    raise ValueError("yes/no raw and semantic margins must match")
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
        raise ValueError("extra or duplicate yes/no image outcome")
    return arrays
