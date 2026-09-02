"""Exact implementation and validation helpers for the first registered hard gate.

This module is deliberately specific to ``corrected-llava-effusion-vislast``.  It does not select a
model, concept, locus, metric, dose, or control after seeing results; those identities are fixed in
``docs/RESEARCH_PLAN.md``.
"""
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
    from .probe import build_types, control_labels
except ImportError:
    from bootstrap_probe import boot_auroc, pct_ci
    from probe import build_types, control_labels


MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MODEL_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
MODEL_HASHES = {
    "model-00001-of-00003.safetensors": "c11dbf016ee7d35ee130c19b67e20eb04873996006f424b7bcbd453b6517ee66",
    "model-00002-of-00003.safetensors": "46df6c6e5fad297fe7fbca4963dcf180cb1d0528cabc884d1f603311fed01328",
    "model-00003-of-00003.safetensors": "4f06177c37ca13944e73b57dd6d051767b950dffc12400c700dfa2509c24c697",
    "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
}
MANIFEST_SHA256 = "837ca37acce72cdf1e7c4a43e559a6d34d99edf2eb5c76a8f5e53295e99c6440"
EXPECTED_ROWS = 26_229
EXPECTED_LOCUS = "model.vision_tower.encoder.layers.22"
ALPHAS = [-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0]
CONTROL_ALPHAS = [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def verify_inputs(manifest: Path, dataset_receipt: Path, model_root: Path, out: Path) -> None:
    """Rehash the staged identities before the measurement starts."""
    dataset = json.loads(dataset_receipt.read_text(encoding="utf-8"))
    model = json.loads((model_root / "asset-receipt.json").read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest)
    if manifest_hash != MANIFEST_SHA256 or dataset.get("registered_manifest_sha256") != manifest_hash:
        raise ValueError("registered NIH manifest hash mismatch")
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    if len(rows) != EXPECTED_ROWS or dataset.get("registered_manifest_rows") != EXPECTED_ROWS:
        raise ValueError("registered NIH manifest row count mismatch")
    row_ids = [row["row_id"] for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("duplicate NIH row_id")
    patient_splits: dict[str, set[str]] = {}
    missing_images = []
    for row in rows:
        patient_splits.setdefault(row["patient_id"], set()).add(row["split"])
        path = Path(row["image_path"])
        if not path.is_file():
            missing_images.append(os.fspath(path))
    if missing_images:
        raise ValueError(f"missing registered images: {missing_images[:3]}")
    if any(len(splits) != 1 for splits in patient_splits.values()):
        raise ValueError("patient appears in multiple splits")

    if model.get("repo_id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise ValueError("staged model identity or revision mismatch")
    snapshot = (model_root / model["snapshot_relative_path"]).resolve(strict=True)
    if not snapshot.is_relative_to(model_root.resolve(strict=True)):
        raise ValueError("model snapshot escapes staged model root")
    observed = {}
    for name, expected in MODEL_HASHES.items():
        path = snapshot / name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"model hash mismatch for {name}")
        observed[name] = {"bytes": path.stat().st_size, "sha256": actual}
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if config.get("vision_feature_layer") != -2:
        raise ValueError("model no longer selects vision_feature_layer=-2")
    write_json(
        out,
        {
            "dataset_receipt": os.fspath(dataset_receipt),
            "dataset_receipt_sha256": sha256(dataset_receipt),
            "manifest": os.fspath(manifest),
            "manifest_sha256": manifest_hash,
            "manifest_rows": len(rows),
            "patients": len(patient_splits),
            "splits": {name: sum(row["split"] == name for row in rows) for name in ("train", "val", "test")},
            "model_receipt": os.fspath(model_root / "asset-receipt.json"),
            "model_receipt_sha256": sha256(model_root / "asset-receipt.json"),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "snapshot": os.fspath(snapshot),
            "verified_model_files": observed,
            "vision_feature_layer": -2,
            "source_commit": os.environ.get("SOURCE_COMMIT", "unknown"),
        },
    )


def verify_hook(model_root: Path, gpu: int, out: Path) -> None:
    """Prove that the registered block fires and changes its downstream connector on a synthetic image."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    try:
        from .intervene import Steerer
        from .loci import as_hidden, loci_for
        from .registry import REGISTRY
        from .smoke_extract import synthetic_cxr
    except ImportError:
        from intervene import Steerer
        from loci import as_hidden, loci_for
        from registry import REGISTRY
        from smoke_extract import synthetic_cxr

    receipt = json.loads((model_root / "asset-receipt.json").read_text(encoding="utf-8"))
    snapshot = model_root / receipt["snapshot_relative_path"]
    arch = REGISTRY["llava15_7b"]
    locus = next(locus for locus in loci_for(arch) if locus.name == "vis.last")
    if locus.module != EXPECTED_LOCUS:
        raise ValueError(f"vis.last resolved to {locus.module}, expected {EXPECTED_LOCUS}")
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True
    ).eval()
    modules = dict(model.named_modules())
    if locus.module not in modules or arch.connector not in modules:
        raise ValueError("registered target or connector module is absent")

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text":
                 "Is there a pleural effusion in this chest radiograph? Answer yes or no."}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[synthetic_cxr()], return_tensors="pt").to("cuda:0")
    target = {}
    connector = {}

    def target_hook(_module, _inputs, output):
        hidden = as_hidden(output)
        if hidden is not None:
            target["count"] = target.get("count", 0) + 1
            target["shape"] = list(hidden.shape)

    def connector_hook(_module, _inputs, output):
        hidden = as_hidden(output)
        if hidden is not None:
            connector["value"] = hidden.detach().float().cpu()

    target_handle = modules[locus.module].register_forward_hook(target_hook)
    connector_handle = modules[arch.connector].register_forward_hook(connector_hook)
    with torch.inference_mode():
        clean = model(**inputs).logits[:, -1, :].detach().float().cpu()
    clean_connector = connector["value"]
    target_handle.remove()
    connector_handle.remove()
    if target.get("count") != 1:
        raise ValueError(f"registered target hook fired {target.get('count', 0)} times")

    generator = np.random.default_rng(0)
    vector = generator.standard_normal(target["shape"][-1]).astype(np.float32)
    vector /= np.linalg.norm(vector)
    connector = {}
    connector_handle = modules[arch.connector].register_forward_hook(connector_hook)
    steerer = Steerer(model, locus.module, positions="all", mode="reltoken")
    steerer.vec = vector
    steerer.alpha = 0.1
    with steerer, torch.inference_mode():
        changed = model(**inputs).logits[:, -1, :].detach().float().cpu()
    changed_connector = connector["value"]
    connector_handle.remove()
    connector_delta = float((changed_connector - clean_connector).abs().max())
    logit_delta = float((changed - clean).abs().max())
    if connector_delta <= 0 or logit_delta <= 0:
        raise ValueError("vis.last perturbation did not change downstream connector and logits")
    write_json(
        out,
        {
            "arch": arch.key,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "config_vision_feature_layer": config.vision_feature_layer,
            "locus": "vis.last",
            "module": locus.module,
            "expected_module": EXPECTED_LOCUS,
            "hook_fires": target["count"],
            "hook_shape": target["shape"],
            "synthetic_only": True,
            "steering_mode": "reltoken",
            "steering_alpha": 0.1,
            "connector_max_abs_delta": connector_delta,
            "last_logit_max_abs_delta": logit_delta,
            "forward_path_proved": True,
        },
    )


def load_single_locus(acts_dir: Path) -> tuple[list[str], np.ndarray]:
    files = sorted(acts_dir.glob("shard*.npz"))
    if not files:
        raise ValueError(f"no activation shards in {acts_dir}")
    ids, parts = [], []
    for path in files:
        with np.load(path, allow_pickle=True) as shard:
            fields = set(shard.files)
            if fields != {"row_id", "vis.last"}:
                raise ValueError(f"unexpected activation fields in {path}: {sorted(fields)}")
            ids.extend(str(value) for value in shard["row_id"])
            parts.append(shard["vis.last"])
    return ids, np.concatenate(parts, axis=0)


def run_probe(acts_dir: Path, manifest: Path, out_dir: Path, n_boot: int = 2000) -> None:
    ids, raw = load_single_locus(acts_dir)
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    by_id = {row["row_id"]: row for row in rows}
    if len(ids) != EXPECTED_ROWS or len(set(ids)) != EXPECTED_ROWS or set(ids) != set(by_id):
        raise ValueError("activation rows do not exactly cover the registered manifest")
    if raw.shape[0] != len(ids):
        raise ValueError("activation row count does not match row identifiers")
    meta = [by_id[row_id] for row_id in ids]
    split = np.array([row["split"] for row in meta])
    train, test = split == "train", split == "test"
    labels = np.array([int(row["Effusion"]) for row in meta], dtype=np.int8)

    projection_rng = np.random.default_rng(0)
    projection = projection_rng.standard_normal((raw.shape[1], 512)) / np.sqrt(512)
    projected = raw.astype(np.float32) @ projection
    scaler = StandardScaler().fit(projected[train])
    x_train = scaler.transform(projected[train])
    x_test = scaler.transform(projected[test])
    del projected, raw
    real_scores = LogisticRegression(C=1.0, max_iter=2000, random_state=0).fit(
        x_train, labels[train]
    ).predict_proba(x_test)[:, 1]
    real = float(roc_auc_score(labels[test], real_scores))

    types = build_types(meta, ("view_AP", "sex_M", "age"))
    control_scores, control_test_labels, controls = [], [], []
    for seed in range(20):
        control, assignment, usable = control_labels(types, np.random.default_rng(seed))
        if not usable or len(np.unique(control[train])) < 2 or len(np.unique(control[test])) < 2:
            raise ValueError(f"control seed {seed} is unusable")
        scores = LogisticRegression(C=1.0, max_iter=2000, random_state=0).fit(
            x_train, control[train]
        ).predict_proba(x_test)[:, 1]
        auc = float(roc_auc_score(control[test], scores))
        control_scores.append(scores)
        control_test_labels.append(control[test])
        controls.append({"seed": seed, "auroc": auc, "assignment": assignment})

    test_meta = [meta[index] for index in np.flatnonzero(test)]
    patients = sorted({row["patient_id"] for row in test_meta})
    patient_index = {patient: index for index, patient in enumerate(patients)}
    row_patient = np.array([patient_index[row["patient_id"]] for row in test_meta], dtype=np.int64)
    bootstrap_rng = np.random.default_rng(20260827)
    patient_counts = bootstrap_rng.multinomial(
        len(patients), np.full(len(patients), 1.0 / len(patients)), size=n_boot
    ).astype(np.int32)
    counts = patient_counts[:, row_patient]
    real_draws = boot_auroc(real_scores, labels[test], counts)
    control_draws = np.stack(
        [boot_auroc(scores, target, counts) for scores, target in zip(control_scores, control_test_labels)],
        axis=1,
    )
    mean_control_draws = np.nanmean(control_draws, axis=1)
    selectivity_draws = real_draws - mean_control_draws
    control_points = np.array([record["auroc"] for record in controls])
    mean_control = float(control_points.mean())
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "probe_scores_and_bootstrap.npz",
        test_row_id=np.array([ids[index] for index in np.flatnonzero(test)], dtype=object),
        effusion_label=labels[test],
        real_scores=real_scores,
        control_scores=np.stack(control_scores),
        control_labels=np.stack(control_test_labels),
        real_bootstrap=real_draws,
        mean_control_bootstrap=mean_control_draws,
        selectivity_bootstrap=selectivity_draws,
    )
    real_lo, real_hi = pct_ci(real_draws)
    ctrl_lo, ctrl_hi = pct_ci(mean_control_draws)
    sel_lo, sel_hi = pct_ci(selectivity_draws)
    write_json(
        out_dir / "probe.json",
        {
            "arch": "llava15_7b",
            "concept": "Effusion",
            "locus": "vis.last",
            "module": EXPECTED_LOCUS,
            "projection_dim": 512,
            "projection_seed": 0,
            "C": 1.0,
            "probe_seed": 0,
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


def summarize_intervention_rows(
    rows: list[dict[str, str]], *, alphas: list[float], control_alphas: list[float],
    n_random: int, n_eval: int,
) -> dict:
    if {row["locus"] for row in rows} != {"vis.last"}:
        raise ValueError("intervention output is not exactly the registered vis.last cell")
    lookup = {(row["direction"], float(row["alpha"])): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate intervention direction/alpha rows")
    expected_random = {f"random{index}" for index in range(n_random)}
    observed_random = {row["direction"] for row in rows if row["direction"].startswith("random")}
    if observed_random != expected_random:
        raise ValueError("intervention output does not contain exactly the registered random directions")
    for alpha in alphas:
        if ("concept", alpha) not in lookup:
            raise ValueError(f"missing concept alpha {alpha}")
    baseline = float(lookup[("concept", 0.0)]["mean_p_yes"])
    if not np.isfinite(baseline):
        raise ValueError("non-finite intervention baseline")
    directions = {row["direction"] for row in rows}
    if "sham" not in directions or not any(name.startswith("unrelated_") for name in directions):
        raise ValueError("missing sham or unrelated intervention controls")
    for row in rows:
        if int(row["n"]) != n_eval or not np.isfinite(float(row["mean_p_yes"])):
            raise ValueError("intervention row has wrong sample count or non-finite result")

    sham_abs = max(abs(float(lookup[("sham", alpha)]["mean_p_yes"]) - baseline)
                   for alpha in control_alphas)
    candidates = []
    for alpha in control_alphas:
        sign = 1.0 if alpha > 0 else -1.0
        concept_delta = float(lookup[("concept", alpha)]["mean_p_yes"]) - baseline
        signed = sign * concept_delta
        random_signed = np.array([
            sign * (float(lookup[(name, alpha)]["mean_p_yes"]) - baseline)
            for name in sorted(expected_random)
        ])
        unrelated = {
            name: sign * (float(lookup[(name, alpha)]["mean_p_yes"]) - baseline)
            for name in sorted(directions) if name.startswith("unrelated_")
        }
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
                "unrelated_effects": unrelated,
                "outside_random_in_consistent_direction": bool(signed > q95),
                "exceeds_max_absolute_sham": bool(signed > sham_abs),
                "selective": bool(signed > q95 and signed > sham_abs),
            }
        )
    primary = max(candidates, key=lambda record: record["concept_consistent_change"])
    middle = [alpha for alpha in alphas if -0.5 <= alpha <= 0.5]
    probabilities = [float(lookup[("concept", alpha)]["mean_p_yes"]) for alpha in middle]
    rho = float(np.corrcoef(_rank(middle), _rank(probabilities))[0, 1])
    adjacent = np.diff(probabilities)
    return {
        "arch": "llava15_7b",
        "concept": "Effusion",
        "locus": "vis.last",
        "metric": "maximum concept-consistent change in mean P(yes)",
        "baseline_mean_p_yes": baseline,
        "eligible_alphas": control_alphas,
        "control_alpha_reason": "primary candidates require same-alpha random, sham, and unrelated controls",
        "n_random": n_random,
        "n_eval": n_eval,
        "maximum_absolute_sham_effect": sham_abs,
        "candidates": candidates,
        "primary": primary,
        "selective_cell": primary["selective"],
        "monotonicity": {
            "range": [-0.5, 0.5],
            "alphas": middle,
            "mean_p_yes": probabilities,
            "spearman_rho": rho,
            "nondecreasing_adjacent_fraction": float(np.mean(adjacent >= 0)),
            "threshold_role": "reported only",
        },
    }


def summarize_intervention(csv_path: Path, meta_path: Path, out: Path) -> None:
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("arch") != "llava15_7b" or meta.get("concept") != "Effusion":
        raise ValueError("intervention identity mismatch")
    if meta.get("alpha_mode") != "reltoken" or meta.get("seed") != 0:
        raise ValueError("intervention scale or seed mismatch")
    if meta.get("eval_split") != "test" or len(meta.get("eval_row_ids", [])) != 200:
        raise ValueError("intervention did not use exactly 200 held-out test rows")
    if meta.get("alphas") != ALPHAS or meta.get("control_alphas") != CONTROL_ALPHAS:
        raise ValueError("intervention alpha grid mismatch")
    if meta.get("n_random") != 20 or meta.get("n_eval") != 200:
        raise ValueError("intervention random-control or evaluation count mismatch")
    summary = summarize_intervention_rows(
        rows, alphas=ALPHAS, control_alphas=CONTROL_ALPHAS, n_random=20, n_eval=200
    )
    summary["eval_row_ids"] = meta["eval_row_ids"]
    write_json(out, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-inputs")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--dataset-receipt", type=Path, required=True)
    verify.add_argument("--model-root", type=Path, required=True)
    verify.add_argument("--out", type=Path, required=True)
    hook = sub.add_parser("verify-hook")
    hook.add_argument("--model-root", type=Path, required=True)
    hook.add_argument("--gpu", type=int, default=0)
    hook.add_argument("--out", type=Path, required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--acts", type=Path, required=True)
    probe.add_argument("--manifest", type=Path, required=True)
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--n-boot", type=int, default=2000)
    summary = sub.add_parser("summarize-intervention")
    summary.add_argument("--csv", type=Path, required=True)
    summary.add_argument("--meta", type=Path, required=True)
    summary.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify-inputs":
        verify_inputs(args.manifest, args.dataset_receipt, args.model_root, args.out)
    elif args.command == "verify-hook":
        verify_hook(args.model_root, args.gpu, args.out)
    elif args.command == "probe":
        run_probe(args.acts, args.manifest, args.out, args.n_boot)
    else:
        summarize_intervention(args.csv, args.meta, args.out)


if __name__ == "__main__":
    main()
