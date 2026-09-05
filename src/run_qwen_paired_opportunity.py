"""Run the registered clean-baseline paired opportunity gate.

Invoke from the source root with ``python -B -m src.run_qwen_paired_opportunity``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from scripts.audit_nih_paired_pool import RAW_FIELDS
from scripts.audit_paired_opportunity import COHORTS
from src import qwen_paired_opportunity as core


MASS_RUN_ID = "20260905T015725Z-1c9820d7b576-mass-prompts"
MASS_COMMIT = "1c9820d7b5765c2ec220d9f4afaafd4c743e43e4"
CONSOLIDATION_COHORT_RUN_ID = "20260905T052025Z-9bc918b414ff-paired-opportunity"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
BATCH_SIZE = 16
PILOT_EQUIVALENTS = 128
MAX_PREDICTED_SECONDS = 600
HARD_TIMEOUT_SECONDS = 900
OUTPUT_ALLOWANCE_SECONDS = 60
FIELDS = ("row_id", "patient_id", "pair_index", "prompt", "label",
          "raw_margin", "semantic_margin", "semantic_probability")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def terminal_route(mass_run, requested_concept=None):
    """Read the accepted routing summary only after terminal-success checks."""
    mass_run = Path(mass_run).resolve(strict=True)
    metadata = dict(line.split("=", 1) for line in
                    (mass_run / "metadata.env").read_text(encoding="utf-8").splitlines() if "=" in line)
    expected = {"RUN_ID": MASS_RUN_ID, "SOURCE_COMMIT": MASS_COMMIT, "COMMAND_STATUS": "0",
                "DISPATCHER_STATUS": "0", "CLEANUP_STATUS": "0", "ABORT_SIGNAL": "none", "GPU_COUNT": "1"}
    if mass_run.name != MASS_RUN_ID or any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Mass routing source must have accepted terminal metadata")
    for name in ("command_exit_status", "exit_status"):
        if (mass_run / name).read_text(encoding="utf-8").strip() != "0":
            raise ValueError("Mass routing source must be terminal-successful")
    protocol = {"source_commit": MASS_COMMIT, "n_eval": 200, "scientific_outcomes": 179200,
                "gate": "qwen7b-mass-prompt-encoding-specificity"}
    meta = read_json(mass_run / "artifacts/meta.json")
    summary = read_json(mass_run / "artifacts/mass-prompt-specificity-summary.json")
    if (any(obj.get(k) != v for obj in (meta, summary) for k, v in protocol.items())
            or summary.get("n_scientific_outcomes") != 179200
            or summary.get("n_outcomes") != 177800 or summary.get("n_calibration_outcomes") != 1400):
        raise ValueError("Mass terminal summary protocol or complete outcome counts mismatch")
    flags = {key: summary.get(key) for key in
             ("cross_wording_specificity", "discovery_wording_replication")}
    if any(type(value) is not bool for value in flags.values()) or (
            flags["cross_wording_specificity"] and not flags["discovery_wording_replication"]):
        raise ValueError("invalid Mass route flags")
    concept = "Mass" if any(flags.values()) else "Consolidation"
    if requested_concept is not None and requested_concept != concept:
        raise ValueError("requested concept disagrees with terminal Mass route")
    return {"concept": concept, "run_id": MASS_RUN_ID, "source_commit": MASS_COMMIT,
            "summary": str(mass_run / "artifacts/mass-prompt-specificity-summary.json"), **flags}


def protocol_metadata(concept):
    core._check_concept(concept)
    return {"gate": "qwen7b-paired-behavioral-opportunity", "concept": concept,
            "mapping_protocol": "generic_finding_explicit_instruction",
            "n_pairs": core.EVAL_PAIRS, "primary_prompts": list(core.PRIMARY_PROMPTS[concept]),
            "prompts": core.PROMPTS[concept], "scientific_outcomes": 200 * len(core.PRIMARY_PROMPTS[concept]),
            "selection_seed": core.SELECTION_SEED, "bootstrap_seed": core.BOOTSTRAP_SEED,
            "bootstrap_resamples": core.BOOTSTRAP_RESAMPLES, "batch_size": BATCH_SIZE,
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "model_local_only": True,
            "resolution": [336, 336], "hard_timeout_seconds": HARD_TIMEOUT_SECONDS}


def cohort(dataset_root, runs_root, out, source_commit, concept=None, evidence=None):
    """Bind source metadata and save the 200 selected image identities on CPU."""
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("a full source_commit is required")
    dataset_root, runs_root, out = map(Path, (dataset_root, runs_root, out))
    route = terminal_route(runs_root / MASS_RUN_ID, concept)
    concept = route["concept"]
    evidence = Path(evidence) if evidence else Path(__file__).resolve().parents[1] / "docs/evidence/paired-opportunity-metadata.json"
    audit = read_json(evidence)
    inputs = audit["original_pool"]["inputs"]
    image_directory = dataset_root / "png/images"
    if (audit.get("exit_codes") != [0, 0] or not audit.get("source_commit")
            or Path(inputs["dataset_root"]).resolve() != dataset_root.resolve()
            or Path(inputs["runs_root"]).resolve() != runs_root.resolve()
            or Path(inputs["image_directory"]).resolve() != image_directory.resolve()):
        raise ValueError("accepted metadata evidence source identities mismatch")
    raw_path, manifest_path = dataset_root / "Data_Entry_2017_v2020.csv", dataset_root / "manifest.csv"
    with raw_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(RAW_FIELDS) <= set(reader.fieldnames or []):
            raise ValueError("raw CSV is missing accepted source fields")
        raw = list(reader)
    observed = {label for row in raw for label in (row.get("Finding Labels") or "").split("|")}
    if observed != set(core.DISEASES) | {"No Finding"}:
        raise ValueError("raw CSV label schema differs from accepted NIH schema")
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    receipts = {name: read_json(runs_root / run / "artifacts" / filename)
                for name, run, filename, _ in COHORTS}
    have = {p.name for p in image_directory.iterdir() if p.suffix == ".png" and p.is_file()}
    _, receipt = core.select_pairs(raw, manifest, have, receipts, concept)
    receipt.update(source_commit=source_commit, route=route,
        sources={"raw_csv": str(raw_path.resolve()), "manifest": str(manifest_path.resolve()),
                 "runs_root": str(runs_root.resolve()), "image_directory": str(image_directory.resolve()),
                 "metadata_evidence": str(evidence.resolve()), "metadata_evidence_source_commit": audit["source_commit"]})
    validate_pairs(receipt)
    if concept == "Consolidation":
        reference_path = runs_root / CONSOLIDATION_COHORT_RUN_ID / "artifacts/registered-pairs.json"
        reference = read_json(reference_path)
        if any(receipt[key] != reference.get(key)
               for key in ("concept", "selection_seed", "patient_ids", "pairs")):
            raise ValueError("selection differs from the registered reference cohort")
        receipt["cohort_reference"] = str(reference_path.resolve())
    registered = out / "registered-pairs.json"
    if registered.exists() and read_json(registered) != receipt:
        raise ValueError("existing prospective cohort differs from selection")
    write_json(registered, receipt)
    return receipt


def validate_pairs(receipt):
    concept = receipt.get("concept")
    core._check_concept(concept)
    pairs = receipt.get("pairs", [])
    if (receipt.get("n_pairs") != core.EVAL_PAIRS or len(pairs) != core.EVAL_PAIRS
            or receipt.get("selection_seed") != core.SELECTION_SEED
            or len({p["patient_id"] for p in pairs}) != core.EVAL_PAIRS
            or receipt.get("patient_ids") != [p["patient_id"] for p in pairs]):
        raise ValueError("expected 100 distinct patients in registered order")
    images = set()
    for pair in pairs:
        sides = []
        for side in ("positive", "negative"):
            raw = pair[side + "_metadata"]
            labels, nuisance = core.validate_raw_row(raw)
            image = raw["Image Index"]
            if (image in images or image != pair[side + "_image_index"]
                    or image[:-4] != pair[side + "_row_id"]
                    or raw["Patient ID"] != pair["patient_id"]
                    or pair.get("split") != "test" or core.patient_split(pair["patient_id"]) != "test"
                    or sorted(labels) != pair[side + "_labels"]
                    or type(pair[side + "_no_finding"]) is not bool
                    or ("No Finding" in labels) != pair[side + "_no_finding"]
                    or (concept in labels) != (side == "positive")):
                raise ValueError("registered pair identity or raw labels mismatch")
            images.add(image)
            sides.append((nuisance, labels - {concept, "No Finding"}))
        if sides[0] != sides[1]:
            raise ValueError("registered pair nuisance or other-disease mismatch")
    return pairs


def encoding_for(cell):
    return "standard" if cell == "anchor" else cell.split("/")[1]


def validate_scores(scores, cell, count):
    scores = np.asarray(scores, dtype=float)
    sign = -1 if encoding_for(cell) == "B_present" else 1
    if (scores.shape != (count, 3) or not np.isfinite(scores).all()
            or not np.allclose(scores[:, 1], sign * scores[:, 0], rtol=0, atol=1e-7)
            or not np.allclose(scores[:, 2], core._sigmoid(scores[:, 1]), rtol=1e-6, atol=1e-7)
            or np.any((scores[:, 2] < 0) | (scores[:, 2] > 1))):
        raise ValueError("invalid finite scores or clinical orientation")
    return scores


def validate_grid(records, receipt):
    pairs = validate_pairs(receipt)
    cells = core.PRIMARY_PROMPTS[receipt["concept"]]
    records = list(records)
    if len(records) != len(cells) * 200:
        raise ValueError("incomplete opportunity grid")
    grid = np.empty((len(cells), core.EVAL_PAIRS, 2, 3))
    cursor = 0
    for c, cell in enumerate(cells):
        for i, pair in enumerate(pairs):
            for s, side in enumerate(("positive", "negative")):
                record = records[cursor]
                cursor += 1
                if (record["row_id"] != pair[side + "_row_id"] or record["patient_id"] != pair["patient_id"]
                        or str(record["pair_index"]) != str(i) or record["prompt"] != cell
                        or str(record["label"]) != str(1 - s)):
                    raise ValueError("duplicate, unordered or mismatched opportunity grid identity")
                grid[c, i, s] = validate_scores([[float(record[k]) for k in FIELDS[-3:]]], cell, 1)[0]
    return grid


def budget(pilot_seconds, elapsed_seconds, scientific_outcomes):
    if (not np.isfinite([pilot_seconds, elapsed_seconds]).all() or pilot_seconds <= 0
            or elapsed_seconds < pilot_seconds or scientific_outcomes not in (200, 400)):
        raise ValueError("invalid preflight timing or outcome count")
    predicted = elapsed_seconds + pilot_seconds * scientific_outcomes / PILOT_EQUIVALENTS + OUTPUT_ALLOWANCE_SECONDS
    return {"image_condition_equivalents": PILOT_EQUIVALENTS, "pilot_seconds": pilot_seconds,
            "elapsed_seconds": elapsed_seconds, "scientific_outcomes": scientific_outcomes,
            "output_allowance_seconds": OUTPUT_ALLOWANCE_SECONDS, "predicted_total_seconds": predicted,
            "maximum_seconds": MAX_PREDICTED_SECONDS, "passed": predicted <= MAX_PREDICTED_SECONDS}


def mapping_cases(concept):
    core._check_concept(concept)
    from src import qwen_mass_prompt_specificity as mass
    cases = []
    for cell in core.PRIMARY_PROMPTS[concept]:
        encoding = "yes_no" if cell == "anchor" else encoding_for(cell)
        cases.extend({"prompt_cell": cell, "case": i, "expected_present": label,
                      "prompt": f"{statement} Is the finding present? {mass.instruction(encoding)}"}
                     for i, (statement, label) in enumerate(mass.ae.STATEMENTS))
    return cases


def validate_preflight(preflight, concept):
    cells = core.PRIMARY_PROMPTS[concept]
    expected = mapping_cases(concept)
    cases = preflight.get("mapping_cases", [])
    repeated = preflight.get("repeated_scores", {})
    ids = preflight.get("token_ids", {})
    if len(cases) != len(expected) or set(repeated) != set(cells) or set(ids) != set(cells):
        raise ValueError("incomplete opportunity preflight")
    for cell, groups in ids.items():
        if (len(groups) != 2 or not all(groups) or set(groups[0]) & set(groups[1])
                or any(type(i) is not int or i < 0 for group in groups for i in group)):
            raise ValueError("invalid preflight singleton token identities")
    for actual, wanted in zip(cases, expected, strict=True):
        if any(actual.get(k) != v for k, v in wanted.items()):
            raise ValueError("preflight mapping identity mismatch")
        scores = validate_scores([actual["scores"]], wanted["prompt_cell"], 1)[0]
        if scores[1] * (2 * wanted["expected_present"] - 1) <= 0:
            raise ValueError("preflight mapping sign failed")
    for cell, scores in repeated.items():
        a, b = (validate_scores([value], cell, 1) for value in scores)
        if not np.array_equal(a, b):
            raise ValueError("clean validation scores must repeat exactly")
    timing = preflight["throughput"]
    expected_timing = budget(timing["pilot_seconds"], timing["elapsed_seconds"], 200 * len(cells))
    if timing != expected_timing or not timing["passed"] or preflight.get("passed") is not True:
        raise ValueError("opportunity preflight budget failed")
    return True


def summarize(per_image, out):
    per_image, out = Path(per_image), Path(out)
    receipt = read_json(per_image.parent / "registered-pairs.json")
    meta = read_json(per_image.parent / "meta.json")
    concept = receipt["concept"]
    source = meta.get("source", {})
    route = receipt["route"]
    flags = [route.get(key) for key in ("cross_wording_specificity", "discovery_wording_replication")]
    if (any(type(value) is not bool for value in flags) or flags[0] and not flags[1]
            or route.get("run_id") != MASS_RUN_ID or route.get("source_commit") != MASS_COMMIT
            or route.get("concept") != concept or concept != ("Mass" if any(flags) else "Consolidation")):
        raise ValueError("saved terminal route mismatch")
    if (any(meta.get(k) != v for k, v in protocol_metadata(concept).items())
            or meta.get("source_commit") != receipt["source_commit"] or meta.get("route") != receipt["route"]
            or meta.get("patient_ids") != receipt["patient_ids"]
            or source.get("model_id") != MODEL_ID or source.get("model_revision") != MODEL_REVISION
            or not source.get("model_source")):
        raise ValueError("opportunity summary metadata mismatch")
    preflight = read_json(per_image.parent / "preflight.json")
    validation_ids = meta.get("validation_row_ids", [])
    if (len(validation_ids) != BATCH_SIZE or len(set(validation_ids)) != BATCH_SIZE
            or preflight.get("validation_row_ids") != validation_ids):
        raise ValueError("validation cohort metadata mismatch")
    validate_preflight(preflight, concept)
    with per_image.open(newline="", encoding="utf-8") as stream:
        grid = validate_grid(csv.DictReader(stream), receipt)
    result, arrays = core.summarize_opportunity(grid[:, :, 0, 1], grid[:, :, 1, 1], concept,
        negative_no_finding=[p["negative_no_finding"] for p in receipt["pairs"]], return_arrays=True)
    arrays.update(raw_margin=grid[..., 0], semantic_probability=grid[..., 2],
                  patient_ids=np.asarray(receipt["patient_ids"]),
                  row_ids=np.asarray([[p[side + "_row_id"] for side in ("positive", "negative")]
                                      for p in receipt["pairs"]]))
    result.update(source_commit=meta["source_commit"], route=receipt["route"],
                  n_outcomes=int(grid[..., 0].size), patient_ids=receipt["patient_ids"],
                  cohort_receipt=receipt, source=meta["source"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    write_json(out, result)
    return result


def run(dataset_root, runs_root, ownership_run, out, source_commit, gpu, concept=None,
        evidence=None, preflight_only=False):
    started = time.perf_counter()
    out = Path(out)
    if (out / "per-image.csv").exists() or (out / "per-image.csv.tmp").exists():
        raise ValueError("scientific output already exists; use CPU summarize for a completed grid")
    receipt = cohort(dataset_root, runs_root, out, source_commit, concept, evidence)
    concept = receipt["concept"]
    from src.gpu_env import bind_gpu
    bind_gpu(gpu)
    from src import qwen_answer_encoding as ae
    source, _, _, validation, _ = ae.load_source(Path(ownership_run), Path(dataset_root) / "manifest.csv")
    if source.get("model_id") != MODEL_ID or source.get("model_revision") != MODEL_REVISION:
        raise ValueError("accepted model source identity mismatch")
    meta = {**protocol_metadata(concept), "source": source, "source_commit": source_commit,
            "route": receipt["route"], "patient_ids": receipt["patient_ids"],
            "validation_row_ids": [r["row_id"] for r in validation]}
    write_json(out / "meta.json", meta)
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from src.registry import REGISTRY
    arch = REGISTRY["qwen7b"]
    if (arch.hf_id != MODEL_ID or arch.processor_kwargs != {"min_pixels": 336 * 336, "max_pixels": 336 * 336}
            or torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0)):
        raise ValueError("opportunity requires the pinned Qwen registry and one A100")
    model_path = str(Path(source["model_source"]).resolve(strict=True))
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, **arch.processor_kwargs)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True).eval()
    cells = core.PRIMARY_PROMPTS[concept]
    ab_ids = ae.singleton_ids(tokenizer) if concept == "Mass" else None
    token_ids = {cell: ab_ids if concept == "Mass" else ae.singleton_ids(
        tokenizer, ("yes", "Yes", "YES"), ("no", "No", "NO")) for cell in cells}

    def score(prompt, cell, rows=()):
        if time.perf_counter() - started >= HARD_TIMEOUT_SECONDS:
            raise TimeoutError("paired opportunity reached its 900-second cap")
        content = ([{"type": "image"}] if rows else []) + [{"type": "text", "text": prompt}]
        text = processor.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)
        images = []
        for row in rows:
            with Image.open(row["image_path"]) as image:
                images.append(image.convert("RGB"))
        kwargs = {"images": images} if images else {}
        inputs = processor(text=[text] * max(1, len(rows)), return_tensors="pt", padding=True, **kwargs).to("cuda:0")
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1:, :].float()
            scores = ae.score_logits(logits, tokenizer, encoding_for(cell), ab_ids)
        if time.perf_counter() - started >= HARD_TIMEOUT_SECONDS:
            raise TimeoutError("paired opportunity reached its 900-second cap")
        return validate_scores(scores, cell, max(1, len(rows)))

    preflight = {"passed": False, "token_ids": token_ids, "mapping_cases": [], "repeated_scores": {},
                 "validation_row_ids": [r["row_id"] for r in validation]}
    try:
        for case in mapping_cases(concept):
            preflight["mapping_cases"].append({**case, "scores": score(case["prompt"], case["prompt_cell"])[0].tolist()})
        for cell in cells:
            preflight["repeated_scores"][cell] = [score(core.PROMPTS[concept][cell], cell, validation[:1])[0].tolist()
                                                    for _ in range(2)]
        pilot_start = time.perf_counter()
        for batch in range(PILOT_EQUIVALENTS // BATCH_SIZE):
            cell = cells[batch % len(cells)]
            score(core.PROMPTS[concept][cell], cell, validation)
        pilot_seconds = time.perf_counter() - pilot_start
        preflight["throughput"] = budget(pilot_seconds, time.perf_counter() - started, meta["scientific_outcomes"])
        preflight["passed"] = preflight["throughput"]["passed"]
        validate_preflight(preflight, concept)
    except Exception as error:
        preflight.update(passed=False, error=f"{type(error).__name__}: {error}")
        write_json(out / "preflight.json", preflight)
        raise
    write_json(out / "preflight.json", preflight)
    if preflight_only:
        return True
    image_rows = [{"row_id": pair[side + "_row_id"], "patient_id": pair["patient_id"],
                   "pair_index": i, "label": int(side == "positive"),
                   "image_path": str(Path(receipt["sources"]["image_directory"]) / pair[side + "_image_index"])}
                  for i, pair in enumerate(receipt["pairs"]) for side in ("positive", "negative")]
    temporary = out / "per-image.csv.tmp"
    with temporary.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for cell in cells:
            for start in range(0, len(image_rows), BATCH_SIZE):
                chunk = image_rows[start:start + BATCH_SIZE]
                scores = score(core.PROMPTS[concept][cell], cell, chunk)
                for row, values in zip(chunk, scores, strict=True):
                    writer.writerow({**{k: row[k] for k in FIELDS[:3]}, "prompt": cell, "label": row["label"],
                                     **dict(zip(FIELDS[-3:], values))})
            stream.flush()
            print(f"{cell}: {len(image_rows)} paired baseline outcomes complete", flush=True)
    with temporary.open(newline="", encoding="utf-8") as stream:
        validate_grid(csv.DictReader(stream), receipt)
    temporary.replace(out / "per-image.csv")
    summarize(out / "per-image.csv", out / "paired-opportunity-summary.json")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("cohort", "run"):
        p = sub.add_parser(command)
        p.add_argument("--dataset-root", type=Path, required=True)
        p.add_argument("--runs-root", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--source-commit", default=os.environ.get("SOURCE_COMMIT"))
        p.add_argument("--concept", choices=tuple(core.PRIMARY_PROMPTS))
        p.add_argument("--evidence", type=Path)
        if command == "run":
            p.add_argument("--ownership-run", type=Path, required=True)
            p.add_argument("--gpu", type=int, required=True)
            p.add_argument("--preflight-only", action="store_true")
    p = sub.add_parser("summarize")
    p.add_argument("--per-image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    {"cohort": cohort, "run": run, "summarize": summarize}[command](**args)


if __name__ == "__main__":
    main()
