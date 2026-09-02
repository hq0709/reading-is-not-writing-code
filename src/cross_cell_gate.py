"""Registered LLaVA Edema cross-cell gate using accepted ``vis.last`` activations."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from .bootstrap_probe import boot_auroc, pct_ci
    from .first_gate import (
        MANIFEST_SHA256,
        capacity_direction,
        load_single_locus,
        sha256,
        write_json,
    )
    from .probe import build_types, control_labels
except ImportError:
    from bootstrap_probe import boot_auroc, pct_ci
    from first_gate import (
        MANIFEST_SHA256,
        capacity_direction,
        load_single_locus,
        sha256,
        write_json,
    )
    from probe import build_types, control_labels


ARCH = "llava15_7b"
CONCEPT = "Edema"
LOCUS = "vis.last"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
PROMPT = "Is there pulmonary edema in this chest radiograph? Answer yes or no."
SOURCE_PROMPT = "Is there a pleural effusion in this chest radiograph? Answer yes or no."
SOURCE_RUN_ID = "20260902T191411Z-ffd523c464c8-f99e2f39"
SOURCE_COMMIT = "ffd523c464c84417a93c5a6d0a34e5b74e55e76e"
EXPECTED_ROWS = 26_229
ALPHAS = [-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0]
CONTROL_ALPHAS = [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]
UNRELATED_CONCEPTS = ("Effusion", "Atelectasis", "Pneumothorax", "Cardiomegaly", "Mass", "Nodule")
SOURCE_SHA256 = {
    "./artifacts/activations/meta.json": "c687ab18fbf136d573058d9739794d629000a1fedd7dfefdc1db73d941c5f6dd",
    "./artifacts/activations/shard0.json": "e3493e0a8debf9e0c6f25f71f2246298349ab4dde4dee5c26739c4895323c8fe",
    "./artifacts/activations/shard0.npz": "78ccffd7be7ef26e7e461622eca4c35fb18567bdbe9ff9cd23b1f1a6d9d7eef2",
    "./artifacts/hook-verification.json": "d2176d54871f6aa0531c1ad62a333dd973641556f40e92004931f0c9d30d91a0",
    "./artifacts/input-verification.json": "7df9ae79473c34bfaf74d90a87078209fb9a25e00f26451834d894014b392d72",
    "./artifacts/intervention-summary.json": "0ecbb655efb05312017167d21d5e0ef79aa5b94540909bcbaf2649e313c3d9ca",
    "./artifacts/probe/probe.json": "cafddb8aa037bbda66731aaff9b33cbcfef9642995322a4bfd7cbab947927788",
    "./artifacts/probe/probe_scores_and_bootstrap.npz": "49a6b267047e70aad65667a0b95b9760e922662ca3246c9024dacce1659fab6f",
}


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


def validate_activation_source(source_run: Path, out: Path) -> None:
    """Bind the new gate to the accepted source receipt without rehashing its activation shard."""
    source_run = source_run.resolve(strict=True)
    if source_run.name != SOURCE_RUN_ID:
        raise ValueError("unexpected activation source run")
    metadata = _metadata(source_run / "metadata.env")
    if metadata.get("SOURCE_COMMIT") != SOURCE_COMMIT:
        raise ValueError("activation source commit mismatch")
    if any(metadata.get(key) != "0" for key in ("COMMAND_STATUS", "DISPATCHER_STATUS")):
        raise ValueError("activation source run is not terminal-successful")
    if (source_run / "command_exit_status").read_text(encoding="utf-8").strip() != "0":
        raise ValueError("activation source command failed")
    if (source_run / "exit_status").read_text(encoding="utf-8").strip() != "0":
        raise ValueError("activation source dispatcher failed")

    inventory = _checksum_inventory(source_run / "SHA256SUMS")
    for relative, digest in SOURCE_SHA256.items():
        if inventory.get(relative) != digest or not (source_run / relative).is_file():
            raise ValueError(f"accepted source receipt mismatch for {relative}")

    acts = source_run / "artifacts/activations"
    activation_meta = json.loads((acts / "meta.json").read_text(encoding="utf-8"))
    expected_meta = {
        "arch": ARCH,
        "model": MODEL_ID,
        "prompt": SOURCE_PROMPT,
        "n_rows": EXPECTED_ROWS,
        "git_sha": SOURCE_COMMIT,
        "loci": [LOCUS],
    }
    if any(activation_meta.get(key) != value for key, value in expected_meta.items()):
        raise ValueError("accepted activation identity mismatch")
    shard = json.loads((acts / "shard0.json").read_text(encoding="utf-8"))
    if (
        shard.get("n_in") != EXPECTED_ROWS
        or shard.get("n_out") != EXPECTED_ROWS
        or shard.get("failed") != []
    ):
        raise ValueError("accepted activation shard is incomplete")
    summary = json.loads(
        (source_run / "artifacts/intervention-summary.json").read_text(encoding="utf-8")
    )
    eval_row_ids = summary.get("eval_row_ids")
    if (
        not isinstance(eval_row_ids, list)
        or len(eval_row_ids) != 200
        or len(set(eval_row_ids)) != 200
    ):
        raise ValueError("accepted source evaluation row identity is incomplete")
    model_source = activation_meta.get("model_source")
    if not isinstance(model_source, str) or not model_source:
        raise ValueError("accepted activation model source is missing")

    write_json(
        out,
        {
            "source_run_id": SOURCE_RUN_ID,
            "source_commit": SOURCE_COMMIT,
            "activations": os.fspath(acts),
            "source_probe": os.fspath(source_run / "artifacts/probe/probe.json"),
            "source_probe_bootstrap": os.fspath(
                source_run / "artifacts/probe/probe_scores_and_bootstrap.npz"
            ),
            "model_source": model_source,
            "locus": LOCUS,
            "rows": EXPECTED_ROWS,
            "eval_row_ids": eval_row_ids,
            "accepted_sha256": SOURCE_SHA256,
            "verification_mode": "accepted-immutable-run-receipt",
            "prompt_reuse_basis": "vis.last is upstream of text fusion",
        },
    )


def validate_manifest_identity(manifest: Path) -> None:
    if sha256(manifest) != MANIFEST_SHA256:
        raise ValueError("registered NIH manifest hash mismatch")


def paired_selectivity_difference(
    edema_point: float,
    edema_draws: np.ndarray,
    effusion_point: float,
    effusion_draws: np.ndarray,
) -> tuple[float, tuple[float, float], np.ndarray]:
    if edema_draws.shape != effusion_draws.shape:
        raise ValueError("paired selectivity bootstrap shapes differ")
    paired_draws = edema_draws - effusion_draws
    return edema_point - effusion_point, pct_ci(paired_draws), paired_draws


def load_source_probe_baseline(
    reuse: dict, test_row_ids: np.ndarray, n_boot: int
) -> tuple[float, np.ndarray]:
    """Load the accepted Effusion draws paired to the same test patients and resamples."""
    source_probe_path = Path(reuse.get("source_probe", ""))
    source_bootstrap_path = Path(reuse.get("source_probe_bootstrap", ""))
    source_probe = json.loads(source_probe_path.read_text(encoding="utf-8"))
    expected_probe = {
        "arch": ARCH,
        "concept": "Effusion",
        "locus": LOCUS,
        "projection_dim": 512,
        "projection_seed": 0,
        "C": 1.0,
        "probe_seed": 0,
        "control_seeds": list(range(20)),
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20260827,
        "bootstrap_unit": "patient",
        "n_test": 5343,
        "n_test_patients": 1659,
        "source_commit": SOURCE_COMMIT,
        "bootstrap_sha256": SOURCE_SHA256["./artifacts/probe/probe_scores_and_bootstrap.npz"],
    }
    for key, value in expected_probe.items():
        if source_probe.get(key) != value:
            raise ValueError(f"accepted Effusion probe contract mismatch for {key}")
    if n_boot != 2000:
        raise ValueError("paired Effusion comparison requires 2000 bootstrap resamples")
    with np.load(source_bootstrap_path, allow_pickle=True) as source_bootstrap:
        expected_arrays = {
            "test_row_id",
            "effusion_label",
            "real_scores",
            "control_scores",
            "control_labels",
            "real_bootstrap",
            "mean_control_bootstrap",
            "selectivity_bootstrap",
        }
        if set(source_bootstrap.files) != expected_arrays:
            raise ValueError("accepted Effusion bootstrap evidence inventory mismatch")
        source_ids = source_bootstrap["test_row_id"]
        if (
            source_ids.shape != (5343,)
            or len(set(source_ids.tolist())) != 5343
            or not np.array_equal(source_ids, test_row_ids)
        ):
            raise ValueError("Edema and Effusion probes do not use identical test row IDs")
        expected_shapes = {
            "effusion_label": (5343,),
            "real_scores": (5343,),
            "control_scores": (20, 5343),
            "control_labels": (20, 5343),
            "real_bootstrap": (2000,),
            "mean_control_bootstrap": (2000,),
            "selectivity_bootstrap": (2000,),
        }
        if any(source_bootstrap[name].shape != shape for name, shape in expected_shapes.items()):
            raise ValueError("accepted Effusion bootstrap evidence shape mismatch")
        source_draws = source_bootstrap["selectivity_bootstrap"].astype(np.float64).copy()
    point = source_probe.get("real_minus_mean_control")
    if not isinstance(point, (int, float)) or not np.isfinite(point):
        raise ValueError("accepted Effusion selectivity point estimate is invalid")
    return float(point), source_draws


def run_probe(
    acts_dir: Path,
    manifest: Path,
    reuse_receipt: Path,
    out_dir: Path,
    n_boot: int = 2000,
) -> None:
    reuse = json.loads(reuse_receipt.read_text(encoding="utf-8"))
    if reuse.get("source_run_id") != SOURCE_RUN_ID or reuse.get("locus") != LOCUS:
        raise ValueError("activation reuse receipt identity mismatch")
    if Path(reuse.get("activations", "")).resolve() != acts_dir.resolve():
        raise ValueError("probe activation path differs from accepted source")
    validate_manifest_identity(manifest)
    ids, raw = load_single_locus(acts_dir)
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    by_id = {row["row_id"]: row for row in rows}
    if len(ids) != EXPECTED_ROWS or len(set(ids)) != EXPECTED_ROWS or set(ids) != set(by_id):
        raise ValueError("activation rows do not exactly cover the registered manifest")
    if raw.shape[0] != len(ids):
        raise ValueError("activation row count does not match row identifiers")
    meta = [by_id[row_id] for row_id in ids]
    split = np.asarray([row["split"] for row in meta])
    train, test = split == "train", split == "test"
    labels = np.asarray([int(row[CONCEPT]) for row in meta], dtype=np.int8)

    projection_rng = np.random.default_rng(0)
    projection = (projection_rng.standard_normal((raw.shape[1], 512)) / np.sqrt(512)).astype(
        np.float32
    )
    projected = raw.astype(np.float32) @ projection
    scaler = StandardScaler().fit(projected[train])
    x_train = scaler.transform(projected[train])
    x_test = scaler.transform(projected[test])
    del projected, raw
    real_probe = LogisticRegression(C=1.0, max_iter=2000, random_state=0).fit(
        x_train, labels[train]
    )
    real_scores = real_probe.predict_proba(x_test)[:, 1]
    real = float(roc_auc_score(labels[test], real_scores))

    direction_names = [CONCEPT, *UNRELATED_CONCEPTS]
    coefficients = [real_probe.coef_[0]]
    vectors = [capacity_direction(projection, scaler.scale_, real_probe.coef_[0])]
    for concept in UNRELATED_CONCEPTS:
        target = np.asarray([int(row[concept]) for row in meta], dtype=np.int8)
        if len(np.unique(target[train])) != 2:
            raise ValueError(f"registered unrelated direction {concept} is unusable")
        readout = LogisticRegression(C=1.0, max_iter=2000, random_state=0).fit(
            x_train, target[train]
        )
        coefficients.append(readout.coef_[0])
        vectors.append(capacity_direction(projection, scaler.scale_, readout.coef_[0]))

    types = build_types(meta, ("view_AP", "sex_M", "age"))
    control_scores, control_test_labels, controls = [], [], []
    for seed in range(20):
        control, assignment, usable = control_labels(types, np.random.default_rng(seed))
        if not usable or len(np.unique(control[train])) < 2 or len(np.unique(control[test])) < 2:
            raise ValueError(f"control seed {seed} is unusable")
        scores = (
            LogisticRegression(C=1.0, max_iter=2000, random_state=0)
            .fit(x_train, control[train])
            .predict_proba(x_test)[:, 1]
        )
        controls.append(
            {
                "seed": seed,
                "auroc": float(roc_auc_score(control[test], scores)),
                "assignment": assignment,
            }
        )
        control_scores.append(scores)
        control_test_labels.append(control[test])

    test_meta = [meta[index] for index in np.flatnonzero(test)]
    patients = sorted({row["patient_id"] for row in test_meta})
    patient_index = {patient: index for index, patient in enumerate(patients)}
    row_patient = np.asarray(
        [patient_index[row["patient_id"]] for row in test_meta], dtype=np.int64
    )
    bootstrap_rng = np.random.default_rng(20260827)
    patient_counts = bootstrap_rng.multinomial(
        len(patients), np.full(len(patients), 1.0 / len(patients)), size=n_boot
    ).astype(np.int32)
    counts = patient_counts[:, row_patient]
    real_draws = boot_auroc(real_scores, labels[test], counts)
    control_draws = np.stack(
        [
            boot_auroc(scores, target, counts)
            for scores, target in zip(control_scores, control_test_labels)
        ],
        axis=1,
    )
    mean_control_draws = np.nanmean(control_draws, axis=1)
    selectivity_draws = real_draws - mean_control_draws
    control_points = np.asarray([record["auroc"] for record in controls])
    mean_control = float(control_points.mean())
    test_row_ids = np.asarray([ids[index] for index in np.flatnonzero(test)], dtype=object)
    effusion_selectivity, effusion_selectivity_draws = load_source_probe_baseline(
        reuse, test_row_ids, n_boot
    )
    paired_selectivity, paired_ci, paired_selectivity_draws = paired_selectivity_difference(
        real - mean_control,
        selectivity_draws,
        effusion_selectivity,
        effusion_selectivity_draws,
    )

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
        C=np.asarray(1.0, dtype=np.float64),
        probe_seed=np.asarray(0, dtype=np.int64),
    )
    directions_hash = sha256(directions_path)
    bootstrap_path = out_dir / "probe_scores_and_bootstrap.npz"
    np.savez_compressed(
        bootstrap_path,
        test_row_id=test_row_ids,
        target_label=labels[test],
        real_scores=real_scores,
        control_scores=np.stack(control_scores),
        control_labels=np.stack(control_test_labels),
        real_bootstrap=real_draws,
        mean_control_bootstrap=mean_control_draws,
        selectivity_bootstrap=selectivity_draws,
        edema_minus_effusion_selectivity_bootstrap=paired_selectivity_draws,
    )
    real_lo, real_hi = pct_ci(real_draws)
    ctrl_lo, ctrl_hi = pct_ci(mean_control_draws)
    sel_lo, sel_hi = pct_ci(selectivity_draws)
    write_json(
        out_dir / "probe.json",
        {
            "arch": ARCH,
            "concept": CONCEPT,
            "locus": LOCUS,
            "projection_dim": 512,
            "projection_seed": 0,
            "C": 1.0,
            "probe_seed": 0,
            "directions_sha256": directions_hash,
            "bootstrap_sha256": sha256(bootstrap_path),
            "direction_names": direction_names,
            "type_columns": ["view_AP", "sex_M", "age"],
            "control_seeds": list(range(20)),
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "n_test_patients": len(patients),
            "test_positive": int(labels[test].sum()),
            "bootstrap_resamples": n_boot,
            "bootstrap_seed": 20260827,
            "bootstrap_unit": "patient",
            "real_auroc": real,
            "real_auroc_ci95": [real_lo, real_hi],
            "controls": controls,
            "control_auroc_mean": mean_control,
            "control_auroc_ci95_for_seed_mean": [ctrl_lo, ctrl_hi],
            "control_auroc_spread": {
                "min": float(control_points.min()),
                "max": float(control_points.max()),
                "sd": float(control_points.std(ddof=1)),
                "p05": float(np.percentile(control_points, 5)),
                "p95": float(np.percentile(control_points, 95)),
            },
            "real_minus_mean_control": real - mean_control,
            "selectivity_ci95": [sel_lo, sel_hi],
            "effusion_real_minus_mean_control": effusion_selectivity,
            "edema_minus_effusion_selectivity": paired_selectivity,
            "edema_minus_effusion_selectivity_ci95": list(paired_ci),
            "paired_selectivity_reference": "accepted Effusion probe with identical test rows and patient bootstrap draws",
            "effusion_probe_sha256": SOURCE_SHA256["./artifacts/probe/probe.json"],
            "effusion_probe_bootstrap_sha256": SOURCE_SHA256[
                "./artifacts/probe/probe_scores_and_bootstrap.npz"
            ],
            "activation_source_run": SOURCE_RUN_ID,
            "activation_source_commit": SOURCE_COMMIT,
            "source_commit": os.environ.get("SOURCE_COMMIT", "unknown"),
        },
    )


def _rank(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values_array[order[stop]] == values_array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranks


def summarize_intervention_rows(rows: list[dict[str, str]]) -> dict:
    if {row["locus"] for row in rows} != {LOCUS}:
        raise ValueError("intervention output is not exactly the registered vis.last cell")
    lookup = {(row["direction"], float(row["alpha"])): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate intervention direction/alpha rows")
    expected_random = {f"random{index}" for index in range(20)}
    expected_unrelated = {f"unrelated_{concept}" for concept in UNRELATED_CONCEPTS}
    observed_random = {row["direction"] for row in rows if row["direction"].startswith("random")}
    observed_unrelated = {
        row["direction"] for row in rows if row["direction"].startswith("unrelated_")
    }
    if observed_random != expected_random:
        raise ValueError(
            "intervention output does not contain exactly the registered random directions"
        )
    if observed_unrelated != expected_unrelated:
        raise ValueError(
            "intervention output does not contain exactly the registered unrelated directions"
        )
    expected_keys = {("concept", alpha) for alpha in ALPHAS}
    for direction in expected_random | expected_unrelated | {"sham"}:
        expected_keys.update((direction, alpha) for alpha in CONTROL_ALPHAS)
    if set(lookup) != expected_keys:
        raise ValueError(
            "intervention direction/alpha grid is incomplete or contains unregistered rows"
        )
    for row in rows:
        if int(row["n"]) != 200 or not np.isfinite(float(row["mean_p_yes"])):
            raise ValueError("intervention row has wrong sample count or non-finite result")
    baseline = float(lookup[("concept", 0.0)]["mean_p_yes"])
    sham_abs = max(
        abs(float(lookup[("sham", alpha)]["mean_p_yes"]) - baseline) for alpha in CONTROL_ALPHAS
    )
    candidates = []
    for alpha in CONTROL_ALPHAS:
        sign = 1.0 if alpha > 0 else -1.0
        concept_delta = float(lookup[("concept", alpha)]["mean_p_yes"]) - baseline
        signed = sign * concept_delta
        random_signed = np.asarray(
            [
                sign * (float(lookup[(name, alpha)]["mean_p_yes"]) - baseline)
                for name in sorted(expected_random)
            ]
        )
        q05, q95 = np.percentile(random_signed, [5, 95])
        candidates.append(
            {
                "alpha": alpha,
                "mean_p_yes": float(lookup[("concept", alpha)]["mean_p_yes"]),
                "raw_change": concept_delta,
                "concept_consistent_change": signed,
                "random_p05": float(q05),
                "random_p95": float(q95),
                "random_effects": random_signed.tolist(),
                "sham_effect": sign * (float(lookup[("sham", alpha)]["mean_p_yes"]) - baseline),
                "unrelated_effects": {
                    name: sign * (float(lookup[(name, alpha)]["mean_p_yes"]) - baseline)
                    for name in sorted(expected_unrelated)
                },
                "outside_random_in_consistent_direction": bool(signed > q95),
                "exceeds_max_absolute_sham": bool(signed > sham_abs),
                "selective": bool(signed > q95 and signed > sham_abs),
            }
        )
    primary = max(candidates, key=lambda record: record["concept_consistent_change"])
    middle = [alpha for alpha in ALPHAS if -0.5 <= alpha <= 0.5]
    probabilities = [float(lookup[("concept", alpha)]["mean_p_yes"]) for alpha in middle]
    adjacent = np.diff(probabilities)
    return {
        "arch": ARCH,
        "concept": CONCEPT,
        "locus": LOCUS,
        "metric": "maximum concept-consistent change in mean P(yes)",
        "baseline_mean_p_yes": baseline,
        "eligible_alphas": CONTROL_ALPHAS,
        "n_random": 20,
        "n_eval": 200,
        "maximum_absolute_sham_effect": sham_abs,
        "candidates": candidates,
        "primary": primary,
        "selective_cell": primary["selective"],
        "monotonicity": {
            "range": [-0.5, 0.5],
            "alphas": middle,
            "mean_p_yes": probabilities,
            "spearman_rho": float(np.corrcoef(_rank(middle), _rank(probabilities))[0, 1]),
            "nondecreasing_adjacent_fraction": float(np.mean(adjacent >= 0)),
            "threshold_role": "reported only",
        },
    }


def summarize_intervention(
    csv_path: Path,
    meta_path: Path,
    reuse_path: Path,
    probe_path: Path,
    bootstrap_path: Path,
    directions_path: Path,
    done_path: Path,
    out: Path,
) -> None:
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    reuse = json.loads(reuse_path.read_text(encoding="utf-8"))
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    expected_meta = {
        "arch": ARCH,
        "concept": CONCEPT,
        "alpha_mode": "reltoken",
        "seed": 0,
        "eval_split": "test",
        "alphas": ALPHAS,
        "control_alphas": CONTROL_ALPHAS,
        "n_random": 20,
        "n_eval": 200,
        "model_source": reuse.get("model_source"),
        "model_local_only": True,
        "prompt": PROMPT,
    }
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            raise ValueError(f"intervention contract mismatch for {key}")
    if meta.get("eval_row_ids") != reuse.get("eval_row_ids"):
        raise ValueError("cross-cell intervention did not reuse the exact source evaluation rows")
    expected_probe = {
        "arch": ARCH,
        "concept": CONCEPT,
        "locus": LOCUS,
        "projection_dim": 512,
        "projection_seed": 0,
        "C": 1.0,
        "probe_seed": 0,
        "control_seeds": list(range(20)),
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20260827,
        "bootstrap_unit": "patient",
        "direction_names": [CONCEPT, *UNRELATED_CONCEPTS],
        "activation_source_run": SOURCE_RUN_ID,
        "activation_source_commit": SOURCE_COMMIT,
    }
    for key, value in expected_probe.items():
        if probe.get(key) != value:
            raise ValueError(f"probe contract mismatch for {key}")
    bootstrap_hash = sha256(bootstrap_path)
    directions_hash = sha256(directions_path)
    if probe.get("bootstrap_sha256") != bootstrap_hash:
        raise ValueError("probe bootstrap evidence hash mismatch")
    if (
        probe.get("directions_sha256") != directions_hash
        or meta.get("directions_sha256") != directions_hash
    ):
        raise ValueError("intervention did not use the registered Edema direction bundle")
    with np.load(bootstrap_path, allow_pickle=True) as bootstrap:
        expected_arrays = {
            "test_row_id",
            "target_label",
            "real_scores",
            "control_scores",
            "control_labels",
            "real_bootstrap",
            "mean_control_bootstrap",
            "selectivity_bootstrap",
            "edema_minus_effusion_selectivity_bootstrap",
        }
        if set(bootstrap.files) != expected_arrays:
            raise ValueError("probe bootstrap evidence inventory mismatch")
        n_test = probe.get("n_test")
        if (
            not isinstance(n_test, int)
            or n_test <= 0
            or bootstrap["test_row_id"].shape != (n_test,)
            or bootstrap["target_label"].shape != (n_test,)
            or bootstrap["control_scores"].shape != (20, n_test)
            or bootstrap["control_labels"].shape != (20, n_test)
            or bootstrap["real_bootstrap"].shape != (2000,)
            or bootstrap["mean_control_bootstrap"].shape != (2000,)
            or bootstrap["selectivity_bootstrap"].shape != (2000,)
            or bootstrap["edema_minus_effusion_selectivity_bootstrap"].shape != (2000,)
        ):
            raise ValueError("probe bootstrap evidence shape mismatch")
    for key in ("edema_minus_effusion_selectivity", "effusion_real_minus_mean_control"):
        if not isinstance(probe.get(key), (int, float)) or not np.isfinite(probe[key]):
            raise ValueError(f"probe paired comparison is missing {key}")
    paired_ci = probe.get("edema_minus_effusion_selectivity_ci95")
    if (
        not isinstance(paired_ci, list)
        or len(paired_ci) != 2
        or not all(isinstance(value, (int, float)) and np.isfinite(value) for value in paired_ci)
    ):
        raise ValueError("probe paired comparison interval is invalid")
    source_commit = os.environ.get("SOURCE_COMMIT", "unknown")
    if probe.get("source_commit") != source_commit or meta.get("source_commit") != source_commit:
        raise ValueError("new cross-cell artifacts do not share the immutable source commit")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("loci") != [LOCUS] or done.get("rows") != 171:
        raise ValueError("intervention completion marker is incomplete")
    summary = summarize_intervention_rows(rows)
    summary["eval_row_ids"] = meta["eval_row_ids"]
    summary["activation_source_run"] = SOURCE_RUN_ID
    summary["activation_source_commit"] = SOURCE_COMMIT
    summary["probe"] = {
        "real_auroc": probe["real_auroc"],
        "control_auroc_mean": probe["control_auroc_mean"],
        "real_minus_mean_control": probe["real_minus_mean_control"],
        "selectivity_ci95": probe["selectivity_ci95"],
        "effusion_real_minus_mean_control": probe["effusion_real_minus_mean_control"],
        "edema_minus_effusion_selectivity": probe["edema_minus_effusion_selectivity"],
        "edema_minus_effusion_selectivity_ci95": probe["edema_minus_effusion_selectivity_ci95"],
    }
    summary["evidence"] = {
        "activation_reuse_receipt_sha256": sha256(reuse_path),
        "probe_sha256": sha256(probe_path),
        "probe_bootstrap_sha256": bootstrap_hash,
        "directions_sha256": directions_hash,
        "intervention_csv_sha256": sha256(csv_path),
        "intervention_meta_sha256": sha256(meta_path),
        "completion_marker_sha256": sha256(done_path),
    }
    write_json(out, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-source")
    validate.add_argument("--source-run", type=Path, required=True)
    validate.add_argument("--out", type=Path, required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--acts", type=Path, required=True)
    probe.add_argument("--manifest", type=Path, required=True)
    probe.add_argument("--reuse-receipt", type=Path, required=True)
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--n-boot", type=int, default=2000)
    summary = sub.add_parser("summarize-intervention")
    summary.add_argument("--csv", type=Path, required=True)
    summary.add_argument("--meta", type=Path, required=True)
    summary.add_argument("--reuse-receipt", type=Path, required=True)
    summary.add_argument("--probe", type=Path, required=True)
    summary.add_argument("--bootstrap", type=Path, required=True)
    summary.add_argument("--directions", type=Path, required=True)
    summary.add_argument("--done", type=Path, required=True)
    summary.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate-source":
        validate_activation_source(args.source_run, args.out)
    elif args.command == "probe":
        run_probe(args.acts, args.manifest, args.reuse_receipt, args.out, args.n_boot)
    else:
        summarize_intervention(
            args.csv,
            args.meta,
            args.reuse_receipt,
            args.probe,
            args.bootstrap,
            args.directions,
            args.done,
            args.out,
        )


if __name__ == "__main__":
    main()
