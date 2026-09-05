"""Prepare, execute, and replay the registered LLaVA Effusion readout diagnostic."""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from pathlib import Path

import numpy as np

from src import llava_readout_diagnostic as core
from src import run_llava_paired_opportunity as accepted
from src import run_llava_validation_opportunity as pilot
from src.qwen_answer_encoding import STATEMENTS, singleton_ids

PILOT_RUN_ID = "20260905T093108Z-42a43207c848-llava-validation"
PILOT_COMMIT = "42a43207c848acfcadecf2f3e0bf866ea71d9dc7"
SUMMARY = "llava-readout-diagnostic-summary"
PINNED_TOKEN_GROUPS = {
    "Y": ([3869, 4874, 22483], [694, 1939, 11698]),
    "A": ([319], [350]),
    "B": ([319], [350]),
}


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    accepted.write_json(path, json_value(value))


def full_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise argparse.ArgumentTypeError("source-commit must be a full lowercase SHA")
    return value


def paths(data_root, out, source_commit):
    full_sha(source_commit)
    if os.environ.get("SOURCE_COMMIT", source_commit) != source_commit:
        raise ValueError("source-commit differs from immutable dispatcher metadata")
    root, out = Path(data_root).resolve(strict=True), Path(out).resolve()
    if out == root or not out.is_relative_to(root):
        raise ValueError("output must remain below the accepted data root")
    return root, out


def protocol():
    return {
        "gate": "llava-effusion-readout-diagnostic",
        "population": "validation patients absent from the complete accepted manifest",
        "allocation": [core.N_PAIRS, core.N_PAIRS, core.ALLOCATION_SEED],
        "conditions": list(core.CONDITIONS),
        "scoring": {
            "Y": "maximum yes singleton logit minus maximum no singleton logit",
            "A": "maximum A singleton logit minus maximum B singleton logit",
            "B": "maximum B singleton logit minus maximum A singleton logit",
            "secondary_Y": "logsumexp(yes singleton logits) minus logsumexp(no singleton logits)",
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
        "pilot_run_id": PILOT_RUN_ID,
        "pilot_commit": PILOT_COMMIT,
        "preprocessing": accepted.protocol_metadata()["preprocessing"],
    }


def load_sources(root):
    run = root / "runs" / PILOT_RUN_ID
    terminal = pilot.terminal(run, PILOT_RUN_ID, PILOT_COMMIT)
    model = accepted.load_model_source(root / "models/huggingface", root / "runs")
    pilot_meta, _, prepared = pilot.load_prepared(root, run / "artifacts", PILOT_COMMIT)
    labels_path = (root / "datasets/nih-chestxray14/Data_Entry_2017_v2020.csv").resolve(strict=True)
    manifest_path = (root / "datasets/nih-chestxray14/manifest.csv").resolve(strict=True)
    image_directory = (root / "datasets/nih-chestxray14/png/images").resolve(strict=True)
    for path in (run, labels_path, manifest_path, image_directory, Path(model["model_source"])):
        if not path.is_relative_to(root):
            raise ValueError("accepted source path escapes data root")
    with labels_path.open(newline="", encoding="utf-8-sig") as stream:
        raw = list(csv.DictReader(stream))
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    if len(raw) != 112_120 or len(manifest) != 26_229:
        raise ValueError("accepted NIH source row count mismatch")
    allocation = core.allocate(raw, manifest, image_directory)
    core.validate_allocation(allocation, check_files=True)
    control_assignments = pilot_meta["control_assignments"]
    types = set(pilot.core.build_types(allocation["index"] + allocation["donor"]))
    if not types or any(not types.issubset(record["assignment"]) for record in control_assignments):
        raise ValueError("frozen control maps do not cover the registered diagnostic types")
    source = {
        "pilot_run_id": PILOT_RUN_ID,
        "pilot_commit": PILOT_COMMIT,
        "pilot_terminal": terminal,
        "pilot_prepared_json": str(run / "artifacts/prepared.json"),
        "pilot_prepared_npz": str(run / "artifacts/prepared.npz"),
        "labels": str(labels_path),
        "manifest": str(manifest_path),
        "image_directory": str(image_directory),
        "model": model,
    }
    return source, prepared, control_assignments, allocation


def prepare(data_root, out, source_commit):
    root, out = paths(data_root, out, source_commit)
    if any((out / name).exists() for name in ("cohort.json", "prepared.json", "prepared.npz")):
        raise ValueError("diagnostic preparation artifacts already exist")
    source, accepted_prepared, assignments, allocation = load_sources(root)
    out.mkdir(parents=True, exist_ok=True)
    indices = core.bootstrap_indices()
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
    arrays["bootstrap_indices"] = indices
    np.savez_compressed(out / "prepared.npz", **arrays)
    meta = {
        "source_commit": source_commit,
        "protocol": protocol(),
        "sources": source,
        "control_assignments": assignments,
        "cohort": allocation,
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
    ):
        raise ValueError("diagnostic preparation identity mismatch")
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
        raise ValueError("diagnostic prepared array schema mismatch")
    for key in expected_keys - {"bootstrap_indices"}:
        if not np.array_equal(prepared[key], accepted_prepared[key]):
            raise ValueError("frozen reader parameter mismatch: " + key)
    core.validate_bootstrap(prepared["bootstrap_indices"])
    return meta, allocation, prepared


def token_metadata(tokenizer):
    from src.intervene import NO_WORDS, YES_WORDS

    yes, no = singleton_ids(tokenizer, YES_WORDS, NO_WORDS)
    letter_a, letter_b = singleton_ids(tokenizer)
    result = {}
    for spec in core.CONDITIONS:
        encoding = spec["encoding"]
        first, second = (yes, no) if encoding == "Y" else (letter_a, letter_b)
        present, absent = (second, first) if encoding == "B" else (first, second)
        result[spec["name"]] = {
            "raw_first": first,
            "raw_second": second,
            "present": present,
            "absent": absent,
            "candidate_ids": sorted(set(first + second)),
        }
    return result


def validate_token_metadata(tokens):
    if set(tokens) != {spec["name"] for spec in core.CONDITIONS}:
        raise ValueError("token metadata must contain the six registered conditions")
    by_encoding = {}
    for spec in core.CONDITIONS:
        record = tokens[spec["name"]]
        if set(record) != {"raw_first", "raw_second", "present", "absent", "candidate_ids"}:
            raise ValueError("candidate token metadata schema mismatch")
        first, second = record["raw_first"], record["raw_second"]
        expected_first, expected_second = PINNED_TOKEN_GROUPS[spec["encoding"]]
        if (
            not first
            or not second
            or any(type(value) is not int or value < 0 for value in first + second)
            or set(first) & set(second)
            or record["candidate_ids"] != sorted(set(first + second))
            or first != expected_first
            or second != expected_second
        ):
            raise ValueError("candidate token groups differ from the pinned singleton IDs")
        expected_present, expected_absent = (
            (second, first) if spec["encoding"] == "B" else (first, second)
        )
        if record["present"] != expected_present or record["absent"] != expected_absent:
            raise ValueError("candidate token groups do not match semantic orientation")
        previous = by_encoding.setdefault(spec["encoding"], record)
        if previous != record:
            raise ValueError("wording conditions must reuse identical token groups")
    return True


def expected_mapping_cases():
    return [
        {
            "encoding": encoding,
            "case": case_index,
            "statement": statement,
            "prompt": f"{statement} Is the finding present? {core.SUFFIXES[encoding]}",
            "expected_present": expected,
        }
        for encoding in core.ENCODINGS
        for case_index, (statement, expected) in enumerate(STATEMENTS)
    ]


def load_scorer(source, gpu):
    from src.gpu_env import bind_gpu

    bind_gpu(gpu)
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from src.loci import as_hidden, loci_for
    from src.registry import REGISTRY

    arch = REGISTRY[accepted.ARCH]
    if (
        arch.hf_id != accepted.MODEL_ID
        or arch.family != "llava"
        or arch.vision_feature_layer != -2
        or torch.cuda.device_count() != 1
        or "A100" not in torch.cuda.get_device_name(0)
    ):
        raise ValueError("diagnostic requires the accepted LLaVA registry and one A100")
    model_path = source["model_source"]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True
    ).eval()
    accepted.validate_processor(processor, model.config)
    tokens = token_metadata(processor.tokenizer)
    locus = next(item for item in loci_for(arch) if item.name == "vis.last")
    modules = dict(model.named_modules())
    if locus.module != core.MODULE or locus.positions != "all" or arch.connector not in modules:
        raise ValueError("registered consumed visual locus mismatch")

    def score(condition, rows=(), capture=False, prompt_override=None):
        spec = core.CONDITIONS[condition]
        prompt_text = spec["prompt"] if prompt_override is None else prompt_override
        content = ([{"type": "image"}] if rows else []) + [{"type": "text", "text": prompt_text}]
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        images = []
        for row in rows:
            with Image.open(row["image_path"]) as image:
                images.append(image.convert("RGB"))
        inputs = processor(
            text=[text] * max(1, len(rows)),
            return_tensors="pt",
            padding=True,
            **({"images": images} if images else {}),
        ).to("cuda:0")
        captured, handles = {}, []
        if capture:

            def target_hook(_module, _inputs, output):
                hidden = as_hidden(output)
                captured["target"] = hidden.detach().float().cpu()

            def connector_hook(_module, values):
                captured["forwarded"] = values[0].detach().float().cpu()

            handles.append(modules[locus.module].register_forward_hook(target_hook))
            handles.append(modules[arch.connector].register_forward_pre_hook(connector_hook))
        try:
            with torch.inference_mode():
                logits = model(**inputs).logits[:, -1, :].float()
        finally:
            for handle in handles:
                handle.remove()
        metadata = tokens[spec["name"]]
        first = logits[:, metadata["raw_first"]].max(dim=-1).values
        second = logits[:, metadata["raw_second"]].max(dim=-1).values
        raw = first - second
        semantic = -raw if spec["encoding"] == "B" else raw
        candidate = logits[:, metadata["candidate_ids"]]
        log_partition = torch.logsumexp(logits, dim=-1)
        token_mass = torch.exp(torch.logsumexp(candidate, dim=-1) - log_partition)
        if spec["encoding"] == "Y":
            lse = torch.logsumexp(logits[:, metadata["raw_first"]], dim=-1) - torch.logsumexp(
                logits[:, metadata["raw_second"]], dim=-1
            )
        else:
            lse = None
        raw_margin = raw.cpu().numpy().astype(np.float64)
        semantic_margin = semantic.cpu().numpy().astype(np.float64)
        result = {
            "raw_margin": raw_margin,
            "semantic_margin": semantic_margin,
            "probability": core.sigmoid(semantic_margin),
            "answer_token_mass": token_mass.cpu().numpy().astype(np.float64),
            "candidate_logits": candidate.cpu().numpy().astype(np.float32),
            "log_partition": log_partition.cpu().numpy().astype(np.float32),
            "lse_margin": None if lse is None else lse.cpu().numpy().astype(np.float64),
        }
        if not all(np.isfinite(value).all() for value in result.values() if value is not None):
            raise ValueError("nonfinite native diagnostic score")
        if capture:
            target, forwarded = captured.get("target"), captured.get("forwarded")
            if target is None or forwarded is None or target.shape[1:] != (577, 1024):
                raise ValueError("consumed visual-block capture shape mismatch")
            result["activation"] = target.mean(dim=1).numpy().astype(np.float16)
            result["forwarded_no_cls_exact"] = bool(torch.equal(target[:, 1:, :], forwarded))
        return result

    runtime = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": 1,
        "chat_template": processor.chat_template,
        "module": locus.module,
        "positions": locus.positions,
        "model_source": model_path,
    }
    return score, tokens, runtime


def mapping_cases(score):
    cases = []
    for expected_case in expected_mapping_cases():
        encoding = expected_case["encoding"]
        condition = core.ENCODINGS.index(encoding)
        values = score(condition, prompt_override=expected_case["prompt"])
        record = {**expected_case, "semantic_margin": float(values["semantic_margin"][0])}
        if encoding == "Y":
            record["lse_margin"] = float(values["lse_margin"][0])
        cases.append(record)
    return cases


def budget(pilot_seconds, elapsed_seconds):
    if (
        not np.isfinite([pilot_seconds, elapsed_seconds]).all()
        or not 0 < pilot_seconds <= elapsed_seconds
    ):
        raise ValueError("invalid diagnostic timing measurements")
    projected = elapsed_seconds + 8406 * pilot_seconds / 512 + 600
    return {
        "pilot_equivalents": 512,
        "pilot_seconds": pilot_seconds,
        "elapsed_seconds": elapsed_seconds,
        "scientific_outcomes": 8406,
        "projected_seconds": projected,
        "limit_seconds": 3600,
        "passed": bool(projected <= 3600),
    }


def validate_preflight(flight, source_commit, model_source):
    if (
        flight.get("source_commit") != source_commit
        or flight.get("passed") is not True
        or flight.get("validation_row_ids") != accepted.VALIDATION_ROW_IDS
        or flight.get("module") != core.MODULE
        or flight.get("model_source") != model_source
        or flight.get("forwarded_no_cls_exact") is not True
        or flight.get("pooling") != "mean over all 577 block-output tokens"
    ):
        raise ValueError("diagnostic preflight identity or consumed-locus check failed")
    if not flight.get("clean_repeat_exact") or len(flight.get("mapping_cases", [])) != 12:
        raise ValueError("diagnostic clean repeat or mapping grid failed")
    validate_token_metadata(flight.get("token_ids", {}))
    for case, expected in zip(flight["mapping_cases"], expected_mapping_cases(), strict=True):
        expected_keys = (
            set(expected)
            | {"semantic_margin"}
            | ({"lse_margin"} if expected["encoding"] == "Y" else set())
        )
        if set(case) != expected_keys or any(
            case.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("registered semantic mapping identity mismatch")
        sign = 2 * case["expected_present"] - 1
        if not np.isfinite(case["semantic_margin"]) or case["semantic_margin"] * sign <= 0:
            raise ValueError("primary semantic orientation failed")
        if case["encoding"] == "Y" and (
            not np.isfinite(case["lse_margin"]) or case["lse_margin"] * sign <= 0
        ):
            raise ValueError("yes/no log-sum-exp semantic orientation failed")
    timing = flight["throughput"]
    if timing != budget(timing["pilot_seconds"], timing["elapsed_seconds"]) or not timing["passed"]:
        raise ValueError("registered 60-minute diagnostic projection failed")


def validate_text_only(records):
    expected_keys = {
        "condition",
        "wording",
        "encoding",
        "prompt",
        "raw_margin",
        "semantic_margin",
        "probability",
        "answer_token_mass",
        "lse_margin",
        "constant_auroc_against_index_labels",
        "subtracting_constant_preserves_image_auroc",
    }
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("six text-only outcomes are required")
    for record, spec in zip(records, core.CONDITIONS, strict=True):
        if (
            set(record) != expected_keys
            or record["condition"] != spec["name"]
            or record["wording"] != spec["wording"]
            or record["encoding"] != spec["encoding"]
            or record["prompt"] != spec["prompt"]
            or record["constant_auroc_against_index_labels"] != 0.5
            or record["subtracting_constant_preserves_image_auroc"] is not True
        ):
            raise ValueError("text-only condition identity or schema mismatch")
        values = [
            record[key]
            for key in ("raw_margin", "semantic_margin", "probability", "answer_token_mass")
        ]
        if not np.isfinite(values).all() or not 0 <= record["answer_token_mass"] <= 1:
            raise ValueError("text-only scores must be finite with valid token mass")
        expected_semantic = (
            -record["raw_margin"] if spec["encoding"] == "B" else record["raw_margin"]
        )
        if (
            record["semantic_margin"] != expected_semantic
            or not np.isclose(
                record["probability"], core.sigmoid(expected_semantic), atol=1e-12, rtol=1e-12
            )
            or (spec["encoding"] == "Y") != (record["lse_margin"] is not None)
            or (record["lse_margin"] is not None and not np.isfinite(record["lse_margin"]))
        ):
            raise ValueError("text-only semantic orientation or score mismatch")
    return True


def run(data_root, out, source_commit, gpu, preflight_only=False):
    root, out = paths(data_root, out, source_commit)
    if any((out / name).exists() for name in ("preflight.json", "per-image.csv", "scores.npz")):
        raise ValueError("diagnostic GPU artifacts already exist")
    meta, allocation, _prepared = load_prepared(root, out, source_commit)
    started = time.perf_counter()
    flight = {
        "passed": False,
        "source_commit": source_commit,
        "validation_row_ids": accepted.VALIDATION_ROW_IDS,
    }
    try:
        score, tokens, runtime = load_scorer(meta["sources"]["model"], gpu)
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
        for i in range(32):
            score(i % 6, validation)
        flight["throughput"] = budget(
            time.perf_counter() - pilot_start, time.perf_counter() - started
        )
        flight["passed"] = flight["throughput"]["passed"]
        validate_preflight(flight, source_commit, meta["sources"]["model"]["model_source"])
    except Exception as error:
        flight.update(passed=False, error=f"{type(error).__name__}: {error}")
        write_json(out / "preflight.json", flight)
        raise
    write_json(out / "preflight.json", flight)
    if preflight_only:
        return flight

    text_only = []
    for condition, spec in enumerate(core.CONDITIONS):
        values = score(condition)
        text_only.append(
            {
                "condition": spec["name"],
                "wording": spec["wording"],
                "encoding": spec["encoding"],
                "prompt": spec["prompt"],
                **{
                    key: float(values[key][0])
                    for key in ("raw_margin", "semantic_margin", "probability", "answer_token_mass")
                },
                "lse_margin": None
                if values["lse_margin"] is None
                else float(values["lse_margin"][0]),
                "constant_auroc_against_index_labels": 0.5,
                "subtracting_constant_preserves_image_auroc": True,
            }
        )
    write_json(out / "text-only.json", text_only)

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
                            raise ValueError(
                                "scientific capture is not the tensor consumed by the connector"
                            )
                        activation_batches.append(values["activation"])
                    for offset, row in enumerate(chunk):
                        i = start + offset
                        writer.writerow(
                            {
                                "source_commit": source_commit,
                                "condition": spec["name"],
                                "wording": spec["wording"],
                                "encoding": spec["encoding"],
                                "prompt": spec["prompt"],
                                "image_role": role,
                                "pair_index": i,
                                "patient_id": row["patient_id"],
                                "row_id": row["row_id"],
                                "image_path": row["image_path"],
                                "index_label": allocation["index"][i]["Effusion"],
                                "source_label": row["Effusion"],
                                "raw_margin": values["raw_margin"][offset],
                                "semantic_margin": values["semantic_margin"][offset],
                                "probability": values["probability"][offset],
                                "answer_token_mass": values["answer_token_mass"][offset],
                                "lse_margin": ""
                                if values["lse_margin"] is None
                                else values["lse_margin"][offset],
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
    return {"n_image_outcomes": 8400, "n_text_only_outcomes": 6}


def _interval(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return np.percentile(values, [2.5, 97.5], method="linear").tolist()


def summarize(data_root, out, source_commit):
    root, out = paths(data_root, out, source_commit)
    meta, allocation, prepared = load_prepared(root, out, source_commit)
    flight = accepted.read_json(out / "preflight.json")
    validate_preflight(flight, source_commit, meta["sources"]["model"]["model_source"])
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
        raise ValueError("candidate-logit archive schema mismatch")
    for condition, spec in enumerate(core.CONDITIONS):
        token = flight["token_ids"][spec["name"]]
        positions = {token_id: i for i, token_id in enumerate(token["candidate_ids"])}
        first = [positions[i] for i in token["raw_first"]]
        second = [positions[i] for i in token["raw_second"]]
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
            raw = logits[:, first].max(1) - logits[:, second].max(1)
            semantic = -raw if spec["encoding"] == "B" else raw
            if not np.allclose(scores["raw_margin"][condition, role_index], raw, atol=1e-6, rtol=0):
                raise ValueError("candidate logits do not reproduce raw margin")
            if not np.allclose(
                scores["semantic_margin"][condition, role_index], semantic, atol=1e-6, rtol=0
            ):
                raise ValueError("candidate logits do not reproduce semantic margin")
            log_mass = np.logaddexp.reduce(logits, axis=1) - partition
            if not np.allclose(
                scores["answer_token_mass"][condition, role_index],
                np.exp(log_mass),
                atol=1e-6,
                rtol=1e-6,
            ):
                raise ValueError("candidate logits do not reproduce answer-token mass")
            if spec["encoding"] == "Y":
                lse = np.logaddexp.reduce(logits[:, first], axis=1) - np.logaddexp.reduce(
                    logits[:, second], axis=1
                )
                if not np.allclose(
                    scores["lse_margin"][condition, role_index], lse, atol=1e-6, rtol=0
                ):
                    raise ValueError("candidate logits do not reproduce yes/no lse margin")
                scores["lse_margin"][condition, role_index] = lse

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
        gdraw = answer_arrays["bootstrap_image_advantage"][
            :, spec["wording"], core.ENCODINGS.index(spec["encoding"])
        ]
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
                "image_advantage": float(
                    answer_arrays["image_advantage"][
                        spec["wording"], core.ENCODINGS.index(spec["encoding"])
                    ]
                ),
                "image_advantage_ci95": _interval(gdraw),
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
    lse = []
    counts = core.multiplicities(indices)
    for condition in (0, 3):
        real, real_draw = core.auc_point_draws(scores["lse_margin"][condition, 0], labels, counts)
        donor, donor_draw = core.auc_point_draws(scores["lse_margin"][condition, 1], labels, counts)
        wording = core.CONDITIONS[condition]["wording"]
        max_g = answer_arrays["image_advantage"][wording, 0]
        max_draw = answer_arrays["bootstrap_image_advantage"][:, wording, 0]
        difference_draw = real_draw - donor_draw - max_draw
        lse.append(
            {
                "condition": core.CONDITIONS[condition]["name"],
                "real_auroc": real,
                "donor_auroc_against_index_label": donor,
                "image_advantage": real - donor,
                "image_advantage_ci95": _interval(real_draw - donor_draw),
                "image_advantage_difference_from_max_variant": real - donor - max_g,
                "image_advantage_difference_from_max_variant_ci95": _interval(difference_draw),
            }
        )
    result = {
        "source_commit": source_commit,
        "status": "OBSERVED",
        "gate": protocol()["gate"],
        "n_pairs": core.N_PAIRS,
        "n_image_outcomes": 8400,
        "n_text_only_outcomes": 6,
        "index_positive": int(labels.sum()),
        "donor_positive": int(donor_labels.sum()),
        "cells": cells,
        "primary_family": primary,
        "frozen_reader": reader,
        "same_logit_aggregation": lse,
        "text_only": text_only,
        "route": "mechanism_oriented_synthesis",
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
    np.savez_compressed(out / (SUMMARY + ".npz"), **arrays)
    write_json(out / (SUMMARY + ".json"), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "summarize"):
        sub = commands.add_parser(command)
        sub.add_argument("--data-root", type=Path, required=True)
        sub.add_argument("--out", type=Path, required=True)
        sub.add_argument("--source-commit", type=full_sha, required=True)
        if command == "run":
            sub.add_argument("--gpu", type=int, required=True)
            sub.add_argument("--preflight-only", action="store_true")
    args = vars(parser.parse_args(argv))
    return {"prepare": prepare, "run": run, "summarize": summarize}[args.pop("command")](**args)


if __name__ == "__main__":
    main()
