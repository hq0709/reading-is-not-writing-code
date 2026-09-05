"""LLaVA clean Consolidation opportunity on the accepted paired Qwen cohort.

Run or replay with ``python -B -m src.run_llava_paired_opportunity``.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import time

import numpy as np

from src import qwen_paired_opportunity as core
from src import run_qwen_paired_opportunity as qwen


COHORT_RUN_ID = qwen.CONSOLIDATION_COHORT_RUN_ID
COHORT_COMMIT = "9bc918b414ff14af804873aba3642ac4dcbb3ff3"
QWEN_RUN_ID = "20260905T053520Z-f0317eaf3f4e-paired-opportunity"
QWEN_COMMIT = "f0317eaf3f4e3d97a18c266eb3d84b0f1c09baec"
ASSET_RUN_ID = "20260902T191411Z-ffd523c464c8-f99e2f39"
ASSET_COMMIT = "ffd523c464c84417a93c5a6d0a34e5b74e55e76e"
ARCH = "llava15_7b"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MODEL_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
SNAPSHOT_RELATIVE_PATH = "hub/models--llava-hf--llava-1.5-7b-hf/snapshots/" + MODEL_REVISION
CONCEPT = "Consolidation"
PROMPT = core.PROMPTS[CONCEPT]["anchor"]
BATCH_SIZE = qwen.BATCH_SIZE
FIELDS = qwen.FIELDS
VALIDATION_ROW_IDS = [
    "00010613_005", "00007833_003", "00003974_003", "00009259_000",
    "00005977_010", "00009349_012", "00003459_018", "00003933_000",
    "00008451_011", "00010722_006", "00005496_000", "00005759_028",
    "00010294_050", "00000557_000", "00009107_007", "00003393_010",
]
read_json = qwen.read_json
write_json = qwen.write_json


def validate_indices(indices):
    indices = np.asarray(indices)
    if (indices.shape != (core.BOOTSTRAP_RESAMPLES, core.EVAL_PAIRS)
            or indices.dtype != np.dtype("int64") or np.any(indices < 0) or np.any(indices >= 100)):
        raise ValueError("accepted bootstrap indices must be int64 (5000,100) in [0,100)")
    return indices


def margin_arrays(positive, negative):
    positive, negative = (np.asarray(a, dtype=np.float64) for a in (positive, negative))
    if any(a.shape != (1, 100) or not np.isfinite(a).all() for a in (positive, negative)):
        raise ValueError("complete finite margin arrays must have shape (1,100)")
    with np.errstate(over="ignore", invalid="ignore"):
        gap = positive - negative
    if not np.isfinite(gap).all():
        raise ValueError("paired gaps must be finite")
    return {"positive_margin": positive, "negative_margin": negative, "margin_gap": gap,
            "probability_gap": core._sigmoid(positive) - core._sigmoid(negative),
            "pair_ranking": (positive > negative).astype(float) + .5 * (positive == negative)}


def summarize_arrays(positive, negative, qwen_positive, qwen_negative, indices, negative_no_finding):
    """Use the supplied joint patient draws for the within-model and direct contrast."""
    indices = validate_indices(indices)
    flags = np.asarray(negative_no_finding)
    if flags.shape != (100,) or flags.dtype.kind not in "biuf" or not np.isin(flags, [0, 1]).all():
        raise ValueError("negative No Finding metadata must contain 100 binary values")
    flags = flags.astype(bool)
    arrays = margin_arrays(positive, negative)
    reference = margin_arrays(qwen_positive, qwen_negative)
    with np.errstate(over="ignore", invalid="ignore"):
        bootstrap = arrays["margin_gap"][:, indices].mean(axis=2)
    if not np.isfinite(bootstrap).all():
        raise ValueError("bootstrap margin means must be finite")
    arrays["bootstrap_margin_gap"] = bootstrap
    delta = arrays["pair_ranking"] - reference["pair_ranking"]
    delta_bootstrap = delta[:, indices].mean(axis=2)
    arrays.update(bootstrap_indices=indices, concordance_difference=delta,
                  bootstrap_concordance_difference=delta_bootstrap, negative_no_finding=flags)
    lower = float(np.percentile(arrays["bootstrap_margin_gap"][0], 5))
    per_patient = {key: arrays[key][0].tolist() for key in (
        "positive_margin", "negative_margin", "margin_gap", "probability_gap", "pair_ranking")}
    per_patient.update(positive_probability=core._sigmoid(arrays["positive_margin"])[0].tolist(),
                       negative_probability=core._sigmoid(arrays["negative_margin"])[0].tolist())
    anchor = {"prompt": PROMPT, "mean_margin_gap": float(arrays["margin_gap"].mean()),
              "margin_gap_lower_95": lower,
              "margin_gap_ci95": np.percentile(arrays["bootstrap_margin_gap"][0], [2.5, 97.5]).tolist(),
              "mean_probability_gap": float(arrays["probability_gap"].mean()),
              "mean_pair_ranking": float(arrays["pair_ranking"].mean()),
              "ranking_endpoint": "within-patient concordance", "opportunity_available": lower > 0,
              "per_patient": per_patient,
              "negative_no_finding_strata": {
                  str(status).lower(): {"patients": int(np.sum(flags == status)),
                      "mean_margin_gap": float(arrays["margin_gap"][0, flags == status].mean())
                      if np.any(flags == status) else None} for status in (True, False)}}
    result = {"concept": CONCEPT, "n_pairs": 100, "primary_prompts": ["anchor"],
              "bootstrap_seed": core.BOOTSTRAP_SEED, "bootstrap_resamples": core.BOOTSTRAP_RESAMPLES,
              "prompts": {"anchor": anchor}, "opportunity_available": lower > 0,
              "cross_model": {"endpoint": "within-patient concordance difference",
                  "contrast": "LLaVA minus Qwen", "interval": "descriptive two-sided percentile 95%",
                  "mean_concordance_difference": float(delta.mean()),
                  "concordance_difference_ci95": np.percentile(delta_bootstrap[0], [2.5, 97.5]).tolist(),
                  "per_patient": {"concordance_difference": delta[0].tolist(),
                      "llava_pair_ranking": arrays["pair_ranking"][0].tolist(),
                      "qwen_pair_ranking": reference["pair_ranking"][0].tolist()}}}
    return result, arrays


def row_ids(receipt):
    return np.asarray([[p[side + "_row_id"] for side in ("positive", "negative")]
                       for p in receipt["pairs"]])


def validate_reference(reference, arrays):
    """Bind accepted Qwen metadata, ordered scores and stored patient statistics."""
    receipt, registered, meta = (reference[k] for k in ("cohort_receipt", "qwen_cohort_receipt", "qwen_meta"))
    qwen.validate_pairs(receipt)
    qwen.validate_pairs(registered)
    if (receipt["concept"] != CONCEPT or registered.get("source_commit") != QWEN_COMMIT
            or receipt.get("source_commit") != COHORT_COMMIT
            or any(registered.get(k) != receipt.get(k) for k in
                   ("concept", "selection_seed", "patient_ids", "pairs"))
            or reference.get("cohort_run_id") != COHORT_RUN_ID
            or reference.get("qwen_run_id") != QWEN_RUN_ID):
        raise ValueError("accepted reference cohort or source provenance mismatch")
    source = meta.get("source", {})
    if (any(meta.get(k) != v for k, v in qwen.protocol_metadata(CONCEPT).items())
            or meta.get("source_commit") != QWEN_COMMIT
            or meta.get("patient_ids") != receipt["patient_ids"]
            or meta.get("route") != registered.get("route")
            or meta.get("validation_row_ids") != VALIDATION_ROW_IDS
            or source.get("model_id") != qwen.MODEL_ID
            or source.get("model_revision") != qwen.MODEL_REVISION or not source.get("model_source")):
        raise ValueError("accepted Qwen metadata, validation order or model identity mismatch")
    route = meta["route"]
    if (route.get("run_id") != qwen.MASS_RUN_ID or route.get("source_commit") != qwen.MASS_COMMIT
            or route.get("concept") != CONCEPT or route.get("cross_wording_specificity") is not False
            or route.get("discovery_wording_replication") is not False):
        raise ValueError("accepted Qwen routing provenance mismatch")
    validate_indices(arrays["bootstrap_indices"])
    if (not np.array_equal(arrays["patient_ids"], receipt["patient_ids"])
            or not np.array_equal(arrays["row_ids"], row_ids(receipt))):
        raise ValueError("accepted Qwen patient or image order mismatch")
    grid = qwen.validate_grid(reference["qwen_records"], receipt)
    positive, negative = grid[:, :, 0, 1], grid[:, :, 1, 1]
    expected = {"positive_margin": positive, "negative_margin": negative,
                "pair_ranking": (positive > negative).astype(float) + .5 * (positive == negative),
                "raw_margin": grid[..., 0], "semantic_probability": grid[..., 2]}
    for key, value in expected.items():
        if key not in arrays or not np.array_equal(arrays[key], value):
            raise ValueError(f"accepted Qwen CSV/NPZ mismatch: {key}")
    summary = reference["qwen_summary"]
    if (summary.get("source_commit") != QWEN_COMMIT or summary.get("n_outcomes") != 200
            or summary.get("patient_ids") != receipt["patient_ids"]
            or summary.get("cohort_receipt") != registered
            or summary.get("opportunity_available") is not False):
        raise ValueError("accepted Qwen summary provenance or adjudication mismatch")
    return receipt


def load_reference(runs_root):
    runs_root = Path(runs_root).resolve(strict=True)
    accepted = runs_root / QWEN_RUN_ID
    terminal = dict(line.split("=", 1) for line in
                    (accepted / "metadata.env").read_text(encoding="utf-8").splitlines() if "=" in line)
    expected = {"RUN_ID": QWEN_RUN_ID, "SOURCE_COMMIT": QWEN_COMMIT, "COMMAND_STATUS": "0",
                "DISPATCHER_STATUS": "0", "CLEANUP_STATUS": "0", "ABORT_SIGNAL": "none", "GPU_COUNT": "1"}
    if any(terminal.get(k) != v for k, v in expected.items()) or any(
            (accepted / name).read_text(encoding="utf-8").strip() != "0"
            for name in ("command_exit_status", "exit_status")):
        raise ValueError("accepted Qwen terminal provenance mismatch")
    artifacts = accepted / "artifacts"
    paths = {"cohort_receipt": runs_root / COHORT_RUN_ID / "artifacts/registered-pairs.json",
             "qwen_cohort_receipt": artifacts / "registered-pairs.json",
             "qwen_meta": artifacts / "meta.json",
             "qwen_summary": artifacts / "paired-opportunity-summary.json"}
    reference = {k: read_json(path) for k, path in paths.items()}
    paths.update(qwen_records=artifacts / "per-image.csv", qwen_arrays=artifacts / "paired-opportunity-summary.npz")
    with paths["qwen_records"].open(newline="", encoding="utf-8") as stream:
        reference["qwen_records"] = list(csv.DictReader(stream))
    with np.load(paths["qwen_arrays"], allow_pickle=False) as bundle:
        arrays = {key: bundle[key] for key in bundle.files}
    reference.update(cohort_run_id=COHORT_RUN_ID, qwen_run_id=QWEN_RUN_ID,
                     paths={k: str(v) for k, v in paths.items()}, terminal_metadata=terminal)
    validate_reference(reference, arrays)
    return reference, arrays


def load_model_source(model_root, runs_root):
    model_root = Path(model_root).resolve(strict=True)
    receipt_path = model_root / "asset-receipt.json"
    input_path = Path(runs_root).resolve(strict=True) / ASSET_RUN_ID / "artifacts/input-verification.json"
    receipt, inputs = read_json(receipt_path), read_json(input_path)
    snapshot = (model_root / SNAPSHOT_RELATIVE_PATH).resolve(strict=True)
    source = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "model_source": str(snapshot),
              "model_receipt": str(receipt_path), "asset_receipt": receipt,
              "input_verification_path": str(input_path), "input_verification": inputs}
    validate_model_source(source)
    return source


def validate_model_source(source):
    receipt, inputs = source["asset_receipt"], source["input_verification"]
    if (source.get("model_id") != MODEL_ID or source.get("model_revision") != MODEL_REVISION
            or not Path(source["model_source"]).as_posix().endswith("/" + SNAPSHOT_RELATIVE_PATH)
            or any(receipt.get(k) != v for k, v in {"asset": ARCH, "repo_id": MODEL_ID,
            "revision": MODEL_REVISION, "snapshot_relative_path": SNAPSHOT_RELATIVE_PATH}.items())
            or inputs.get("model_id") != MODEL_ID or inputs.get("model_revision") != MODEL_REVISION
            or inputs.get("source_commit") != ASSET_COMMIT or inputs.get("vision_feature_layer") != -2
            or Path(inputs["snapshot"]) != Path(source["model_source"])
            or Path(inputs["model_receipt"]) != Path(source["model_receipt"])
            or not inputs.get("model_receipt_sha256")):
        raise ValueError("accepted LLaVA model identity or asset receipt mismatch")
    return True


def validation_rows(dataset_root, reference):
    receipt, meta = reference["cohort_receipt"], reference["qwen_meta"]
    manifest = Path(dataset_root) / "manifest.csv"
    if Path(receipt["sources"]["manifest"]).resolve() != manifest.resolve():
        raise ValueError("validation manifest source identity mismatch")
    wanted = meta["validation_row_ids"]
    selected = {}
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["row_id"] in wanted:
                if row["row_id"] in selected:
                    raise ValueError("duplicate validation row identity")
                selected[row["row_id"]] = row
    if set(selected) != set(wanted):
        raise ValueError("accepted validation rows missing from manifest")
    rows = [selected[key] for key in wanted]
    if any(row["split"] not in ("val", "validation") or row["patient_id"] in receipt["patient_ids"]
           or not row.get("image_path") for row in rows):
        raise ValueError("validation split or patient overlap mismatch")
    return rows


def protocol_metadata():
    return {**qwen.protocol_metadata(CONCEPT), "gate": "llava15_7b-paired-behavioral-opportunity",
            "arch": ARCH, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "dtype": "bfloat16", "padding_side": "left", "eval": True,
            "scoring": "yes_margin_batch", "probability": "sigmoid(raw_margin)",
            "bootstrap_source_run_id": QWEN_RUN_ID, "cohort_run_id": COHORT_RUN_ID,
            "preprocessing": {"image_mode": "RGB", "shortest_edge": 336, "center_crop": [336, 336],
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711], "rescale_factor": 1 / 255,
                "resample": 3, "patch_size": 14, "vision_feature_select_strategy": "default",
                "vision_feature_layer": -2, "chat_template": "snapshot"}}


def validate_processor(processor, config):
    image = processor.image_processor
    expected = {"do_resize": True, "size": {"shortest_edge": 336}, "do_center_crop": True,
                "crop_size": {"height": 336, "width": 336}, "do_convert_rgb": True,
                "do_normalize": True, "do_rescale": True, "resample": 3,
                "rescale_factor": 1 / 255}
    if (any(getattr(image, k, None) != v for k, v in expected.items())
            or not np.array_equal(image.image_mean, [0.48145466, 0.4578275, 0.40821073])
            or not np.array_equal(image.image_std, [0.26862954, 0.26130258, 0.27577711])
            or any(getattr(processor, k, None) != v for k, v in {
                "patch_size": 14, "vision_feature_select_strategy": "default",
                "num_additional_image_tokens": 1, "image_token": "<image>"}.items())
            or not processor.chat_template or config.vision_feature_layer != -2
            or config.vision_feature_select_strategy != "default"
            or config.vision_config.patch_size != 14 or config.vision_config.image_size != 336):
        raise ValueError("accepted LLaVA processor or visual feature configuration mismatch")


def load_scorer(source, gpu, started):
    from src.gpu_env import bind_gpu
    bind_gpu(gpu)
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from src.registry import REGISTRY
    from src.intervene import NO_WORDS, YES_WORDS, yes_margin_batch
    from src.qwen_answer_encoding import singleton_ids

    arch = REGISTRY[ARCH]
    if (arch.hf_id != MODEL_ID or arch.family != "llava" or arch.processor_kwargs != {}
            or arch.vision_feature_layer != -2 or torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name(0)):
        raise ValueError("opportunity requires the accepted LLaVA registry and one A100")
    model_path = source["model_source"]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, **arch.processor_kwargs)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True).eval()
    validate_processor(processor, model.config)
    token_ids = singleton_ids(processor.tokenizer, YES_WORDS, NO_WORDS)

    def score(prompt, rows=()):
        check_deadline(started)
        content = ([{"type": "image"}] if rows else []) + [{"type": "text", "text": prompt}]
        text = processor.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)
        images = []
        for row in rows:
            with Image.open(row["image_path"]) as image:
                images.append(image.convert("RGB"))
        inputs = processor(text=[text] * max(1, len(rows)), return_tensors="pt", padding=True,
                           **({"images": images} if images else {})).to("cuda:0")
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1:, :]
            margin = np.asarray(yes_margin_batch(logits, processor.tokenizer), dtype=np.float64)
        check_deadline(started)
        return qwen.validate_scores(np.stack((margin, margin, core._sigmoid(margin)), axis=-1),
                                    "anchor", max(1, len(rows)))

    return score, token_ids, {"gpu_name": torch.cuda.get_device_name(0), "gpu_count": 1,
                              "chat_template": processor.chat_template}


def check_deadline(started):
    if time.perf_counter() - started >= qwen.HARD_TIMEOUT_SECONDS:
        raise TimeoutError("paired opportunity reached its 900-second cap")


def validate_preflight(preflight):
    if preflight.get("validation_row_ids") != VALIDATION_ROW_IDS:
        raise ValueError("accepted validation row order mismatch")
    return qwen.validate_preflight(preflight, CONCEPT)


def summarize(per_image, out):
    per_image, out = Path(per_image), Path(out)
    root = per_image.parent
    reference = read_json(root / "qwen-reference.json")
    with np.load(root / "qwen-reference.npz", allow_pickle=False) as bundle:
        accepted_arrays = {key: bundle[key] for key in bundle.files}
    receipt = validate_reference(reference, accepted_arrays)
    meta = read_json(root / "meta.json")
    source = meta.get("source", {})
    validate_model_source(source)
    if (read_json(root / "registered-pairs.json") != receipt
            or any(meta.get(k) != v for k, v in protocol_metadata().items())
            or not re.fullmatch(r"[0-9a-f]{40}", meta.get("source_commit", ""))
            or meta.get("patient_ids") != receipt["patient_ids"]
            or meta.get("validation_row_ids") != VALIDATION_ROW_IDS
            or [r["row_id"] for r in meta.get("validation_rows", [])] != VALIDATION_ROW_IDS
            or meta.get("runtime", {}).get("gpu_count") != 1
            or "A100" not in meta.get("runtime", {}).get("gpu_name", "")
            or not meta.get("runtime", {}).get("chat_template")):
        raise ValueError("LLaVA summary cohort, protocol or model identity mismatch")
    preflight = read_json(root / "preflight.json")
    if meta.get("token_ids") != preflight.get("token_ids"):
        raise ValueError("LLaVA token metadata differs from preflight")
    validate_preflight(preflight)
    with per_image.open(newline="", encoding="utf-8") as stream:
        grid = qwen.validate_grid(csv.DictReader(stream), receipt)
    result, arrays = summarize_arrays(grid[:, :, 0, 1], grid[:, :, 1, 1],
        accepted_arrays["positive_margin"], accepted_arrays["negative_margin"],
        accepted_arrays["bootstrap_indices"], [p["negative_no_finding"] for p in receipt["pairs"]])
    arrays.update(raw_margin=grid[..., 0], semantic_probability=grid[..., 2],
                  patient_ids=np.asarray(receipt["patient_ids"]), row_ids=row_ids(receipt),
                  **{"qwen_" + key: value for key, value in accepted_arrays.items()})
    result.update(source_commit=meta["source_commit"], source=source, n_outcomes=200,
                  patient_ids=receipt["patient_ids"], cohort_receipt=receipt, metadata=meta,
                  qwen_reference=reference)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    write_json(out, result)
    return result


def run(dataset_root, runs_root, model_root, out, source_commit, gpu, preflight_only=False):
    started = time.perf_counter()
    out = Path(out)
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("a full source_commit is required")
    if any((out / name).exists() for name in ("meta.json", "per-image.csv", "per-image.csv.tmp")):
        raise ValueError("run output already exists; use CPU summarize for a completed grid")
    reference, arrays = load_reference(runs_root)
    receipt = reference["cohort_receipt"]
    source = load_model_source(model_root, runs_root)
    validation = validation_rows(dataset_root, reference)
    image_directory = Path(dataset_root) / "png/images"
    if Path(receipt["sources"]["image_directory"]).resolve() != image_directory.resolve():
        raise ValueError("paired image directory source identity mismatch")
    image_rows = [{"row_id": pair[side + "_row_id"], "patient_id": pair["patient_id"],
                   "pair_index": i, "label": int(side == "positive"),
                   "image_path": str(image_directory / pair[side + "_image_index"])}
                  for i, pair in enumerate(receipt["pairs"]) for side in ("positive", "negative")]
    meta = {**protocol_metadata(), "source": source, "source_commit": source_commit,
            "patient_ids": receipt["patient_ids"], "validation_row_ids": reference["qwen_meta"]["validation_row_ids"],
            "validation_rows": validation, "qwen_source_commit": QWEN_COMMIT,
            "reference_paths": reference["paths"]}
    write_json(out / "registered-pairs.json", receipt)
    write_json(out / "qwen-reference.json", reference)
    np.savez_compressed(out / "qwen-reference.npz", **arrays)
    write_json(out / "meta.json", meta)
    preflight = {"passed": False, "mapping_cases": [], "repeated_scores": {},
                 "validation_row_ids": meta["validation_row_ids"]}
    try:
        score, token_ids, runtime = load_scorer(source, gpu, started)
        meta.update(runtime=runtime, token_ids={"anchor": token_ids})
        write_json(out / "meta.json", meta)
        preflight["token_ids"] = {"anchor": token_ids}
        for case in qwen.mapping_cases(CONCEPT):
            preflight["mapping_cases"].append({**case, "scores": score(case["prompt"])[0].tolist()})
        preflight["repeated_scores"]["anchor"] = [score(PROMPT, validation[:1])[0].tolist() for _ in range(2)]
        pilot_start = time.perf_counter()
        for _ in range(qwen.PILOT_EQUIVALENTS // BATCH_SIZE):
            score(PROMPT, validation)
        preflight["throughput"] = qwen.budget(time.perf_counter() - pilot_start,
                                               time.perf_counter() - started, 200)
        preflight["passed"] = preflight["throughput"]["passed"]
        validate_preflight(preflight)
        check_deadline(started)
    except Exception as error:
        preflight.update(passed=False, error=f"{type(error).__name__}: {error}")
        write_json(out / "preflight.json", preflight)
        raise
    write_json(out / "preflight.json", preflight)
    if preflight_only:
        return True
    temporary = out / "per-image.csv.tmp"
    with temporary.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for start in range(0, 200, BATCH_SIZE):
            chunk = image_rows[start:start + BATCH_SIZE]
            values = qwen.validate_scores(score(PROMPT, chunk), "anchor", len(chunk))
            for row, scores in zip(chunk, values, strict=True):
                writer.writerow({**{key: row[key] for key in FIELDS[:3]}, "prompt": "anchor", "label": row["label"],
                                 **dict(zip(FIELDS[-3:], scores))})
            stream.flush()
    with temporary.open(newline="", encoding="utf-8") as stream:
        qwen.validate_grid(csv.DictReader(stream), receipt)
    temporary.replace(out / "per-image.csv")
    check_deadline(started)
    summarize(out / "per-image.csv", out / "paired-opportunity-summary.json")
    check_deadline(started)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("run")
    for name in ("dataset-root", "runs-root", "model-root", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--source-commit", default=os.environ.get("SOURCE_COMMIT"))
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--preflight-only", action="store_true")
    p = commands.add_parser("summarize")
    p.add_argument("--per-image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    {"run": run, "summarize": summarize}[args.pop("command")](**args)


if __name__ == "__main__":
    main()
