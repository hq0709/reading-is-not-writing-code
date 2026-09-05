"""Registered independent-patient Mass prompt-conditioned specificity gate."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

try:
    from . import qwen_answer_encoding as ae
    from .qwen_causal_ownership_gate import _simultaneous_lower_bounds
    from .qwen_consolidation_input_closure import load_prior_row_ids, PREVIOUS_RUN_ID
except ImportError:
    import qwen_answer_encoding as ae
    from qwen_causal_ownership_gate import _simultaneous_lower_bounds
    from qwen_consolidation_input_closure import load_prior_row_ids, PREVIOUS_RUN_ID

EVAL_ROWS = 200
N_RANDOM = 119
SELECTION_SEED = 20260909
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_RESAMPLES = 5000
CALIBRATION_SEED = 20260908
SCIENTIFIC_OUTCOMES = 179200
PILOT_EQUIVALENTS = 1024
MAX_PREDICTED_SECONDS = 2.75 * 3600
CLOSURE_RUN_ID = "20260904T162317Z-8a55a4c2f6b6-consolidation-closure"
ENCODING_RUN_ID = "20260904T230152Z-21dad2f72726-answer-encoding"
ENCODINGS = ("yes_no", "A_present", "B_present")
QUESTIONS = {"show": "Does this chest radiograph show a lung mass?",
             "is": "Is there a lung mass in this chest radiograph?"}
CELLS = (*(f"{q}/{e}" for q in QUESTIONS for e in ENCODINGS), "anchor")
CODED_CELLS = tuple(c for c in CELLS if c.endswith(("A_present", "B_present")))
DIRECTIONS = (*ae.CONCEPTS, *(f"random{i}" for i in range(N_RANDOM)), "sham")
CONFIGURATIONS = (("baseline", 0.0), *((d, .25) for d in DIRECTIONS))
FIELDS = ("row_id", "patient_id", "cell", "direction", "alpha", "locus", "label",
          "raw_margin", "semantic_margin", "semantic_probability")


def instruction(encoding):
    if encoding == "yes_no":
        return ("Answer yes if the finding is present and no if it is absent. "
                "Reply with yes or no only.")
    if encoding not in ENCODINGS:
        raise ValueError("unknown response encoding")
    return ae.mapping_instruction(encoding)


def prompt_for(cell):
    if cell == "anchor":
        return ae.PROMPTS["Mass"]
    question, encoding = cell.split("/")
    return QUESTIONS[question] + " " + instruction(encoding)


def score_encoding(cell):
    if cell not in CELLS:
        raise ValueError("unknown prompt cell")
    return "standard" if cell == "anchor" or cell.endswith("yes_no") else cell.split("/")[1]


def orientation(cell):
    return -1 if score_encoding(cell) == "B_present" else 1


def mapping_cases():
    return [{"encoding": e, "case": i, "expected_present": label,
             "prompt": f"{statement} Is the finding present? {instruction(e)}"}
            for e in ENCODINGS for i, (statement, label) in enumerate(ae.STATEMENTS)]


def mapping_passed(cases):
    if len(cases) != 12:
        return False
    for case, expected in zip(cases, mapping_cases(), strict=True):
        margin = float(case.get("semantic_margin", "nan"))
        if (any(case.get(k) != v for k, v in expected.items()) or not np.isfinite(margin)
                or margin * (2 * expected["expected_present"] - 1) <= 0):
            return False
    return True


def common_directions(clinical):
    clinical = {q: np.asarray(clinical[q], dtype=np.float32) / np.linalg.norm(clinical[q])
                for q in ae.CONCEPTS}
    rng = np.random.default_rng(0)
    dimension = len(clinical["Mass"])
    directions = dict(clinical)
    for i in range(N_RANDOM):
        vector = rng.standard_normal(dimension).astype(np.float32)
        directions[f"random{i}"] = vector / np.linalg.norm(vector)
    sham = clinical["Mass"][rng.permutation(dimension)]
    directions["sham"] = sham / np.linalg.norm(sham)
    return directions


def select_rows(manifest_rows, excluded_row_ids):
    """Select patients and one row per patient using only identity and split metadata."""
    by_id = {r["row_id"]: r for r in manifest_rows}
    if len(by_id) != len(manifest_rows):
        raise ValueError("manifest row IDs must be unique")
    if not set(excluded_row_ids) <= by_id.keys():
        raise ValueError("exclusion row absent from manifest")
    excluded = {by_id[r]["patient_id"] for r in excluded_row_ids}
    by_patient = {}
    for row in manifest_rows:
        if row["split"] == "test" and row["patient_id"] not in excluded:
            by_patient.setdefault(row["patient_id"], []).append(row["row_id"])
    if len(by_patient) < EVAL_ROWS:
        raise ValueError("fewer than 200 independent eligible test patients")
    rng = np.random.default_rng(SELECTION_SEED)
    patients = rng.permutation(sorted(by_patient))[:EVAL_ROWS]
    selected = []
    for patient in patients:
        candidates = sorted(by_patient[patient])
        selected.append(by_id[candidates[rng.integers(len(candidates))]])
    receipt = {"selection_seed": SELECTION_SEED, "selection": "sorted patients, NumPy permutation, sorted rows, next integer draw",
               "eligible_patients": len(by_patient), "eligible_rows": sum(map(len, by_patient.values())),
               "excluded_patients": len(excluded), "n_eval": EVAL_ROWS,
               "row_ids": [r["row_id"] for r in selected],
               "patient_ids": [r["patient_id"] for r in selected]}
    return selected, receipt


def load_source(ownership_run, manifest):
    source, _, calibration, validation, _ = ae.load_source(ownership_run, manifest)
    runs = ownership_run.parent
    with manifest.open(newline="", encoding="utf-8") as stream:
        manifest_rows = list(csv.DictReader(stream))
    by_id = {r["row_id"]: r for r in manifest_rows}
    prior = load_prior_row_ids(runs)
    closure_path = runs / CLOSURE_RUN_ID / "artifacts/registered-pairs.json"
    pairs = ae.read_json(closure_path)["pairs"]
    if len(pairs) != 50 or len({p["patient_id"] for p in pairs}) != 50:
        raise ValueError("accepted closure must contain 50 distinct patients")
    for pair in pairs:
        for key in ("positive_row_id", "negative_row_id"):
            row = by_id[pair[key]]
            if row["patient_id"] != pair["patient_id"] or row["split"] != "test":
                raise ValueError("closure row/patient or split mismatch")
            prior.append(pair[key])
    encoding_path = runs / ENCODING_RUN_ID / "artifacts/meta.json"
    encoding = ae.read_json(encoding_path)
    for key in ("row_id", "patient_id"):
        if [r[key] for r in encoding["calibration_rows"]] != [r[key] for r in calibration]:
            raise ValueError("accepted encoding calibration differs from ownership last 200")
    rows, receipt = select_rows(manifest_rows, prior)
    for row in rows:
        if row["Mass"] not in ("0", "1"):
            raise ValueError("Mass labels must be binary")
    if {r["patient_id"] for r in rows} & {r["patient_id"] for r in calibration + validation}:
        raise ValueError("confirmation must be patient-disjoint from calibration and validation")
    receipt["exclusion_sources"] = [
        str(runs / ae.SOURCE_RUN_ID / "artifacts/intervention-summary.json"),
        str(runs / PREVIOUS_RUN_ID / "artifacts/registered-rows.json"),
        str(ownership_run / "artifacts/registered-rows.json"), str(closure_path)]
    receipt["calibration_source"] = str(encoding_path)
    receipt["calibration_row_ids"] = [r["row_id"] for r in calibration]
    receipt["calibration_patient_ids"] = [r["patient_id"] for r in calibration]
    return source, rows, calibration, validation, receipt


def validate_grid(records, rows, cells=CELLS, conditions=CONFIGURATIONS):
    n = len(rows)
    if n != EVAL_ROWS or len({r["row_id"] for r in rows}) != n or len({r["patient_id"] for r in rows}) != n:
        raise ValueError("expected 200 unique rows and patients")
    positions = {r["row_id"]: i for i, r in enumerate(rows)}
    configurations = {c: i for i, c in enumerate(conditions)}
    shape = (len(cells), len(conditions), n)
    values, seen = np.empty((*shape, 3)), np.zeros(shape, dtype=bool)
    for record in records:
        try:
            c = cells.index(record["cell"])
            d = configurations[(record["direction"], float(record["alpha"]))]
            i = positions[record["row_id"]]
        except (KeyError, ValueError) as error:
            raise ValueError("unexpected grid identity") from error
        if (seen[c, d, i] or record["patient_id"] != rows[i]["patient_id"]
                or record["locus"] != ae.LOCUS or float(record["label"]) != int(rows[i]["Mass"])):
            raise ValueError("duplicate grid entry or paired patient/label mismatch")
        score = np.array([float(record[k]) for k in FIELDS[-3:]])
        if (not np.isfinite(score).all() or not 0 <= score[2] <= 1
                or not np.isclose(score[1], orientation(cells[c]) * score[0], atol=1e-7, rtol=0)
                or not np.isclose(score[2], expit(score[1]), atol=1e-7, rtol=1e-6)):
            raise ValueError("invalid finite score or answer orientation")
        values[c, d, i], seen[c, d, i] = score, True
    if not seen.all():
        raise ValueError("incomplete Mass prompt grid")
    return values


def bootstrap_weights(n):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return np.array([np.bincount(rng.integers(n, size=n), minlength=n)
                     for _ in range(BOOTSTRAP_RESAMPLES)], dtype=np.float64)


def endpoint(estimate, draws):
    return {"estimate": float(estimate) if np.isfinite(estimate) else None, **ae.interval(draws)}


def calibration_metrics(labels, scores, weights):
    auc = ae.weighted_auroc(labels, scores[:, 1], np.vstack((np.ones(len(labels)), weights)))
    valid = auc[1:][np.isfinite(auc[1:])]
    lower = float(np.percentile(valid, 5)) if len(valid) else None
    positive, negative = int(labels.sum()), int(len(labels) - labels.sum())
    if min(positive, negative) < 10:
        status = "insufficient_labels"
    elif len(valid) < 1900:
        status = "insufficient_valid_bootstraps"
    elif lower is None or lower <= .5:
        status = "baseline_ineligible"
    else:
        status = "eligible"
    return {"eligible": status == "eligible", "status": status,
            "positive_patients": positive, "negative_patients": negative,
            "auroc": float(auc[0]) if np.isfinite(auc[0]) else None,
            "brier": float(np.mean((scores[:, 2] - labels) ** 2)),
            "mean_raw_margin": float(scores[:, 0].mean()), "auroc_bootstrap": ae.interval(auc[1:]),
            "auroc_one_sided_lower95": lower, "valid_bootstrap_count": len(valid),
            "bootstrap_seed": CALIBRATION_SEED}


def effect_summary(effects, weights, eligible):
    """Joint patient bootstrap; both decisions share all twenty clinical comparisons."""
    n = effects.shape[-1]
    means = effects.mean(axis=-1)
    boot = np.einsum("bi,cdi->bcd", weights / n, effects, optimize=True)
    mass = DIRECTIONS.index("Mass")
    competitors = [i for i, d in enumerate(ae.CONCEPTS) if d != "Mass"]
    pairs = [(CELLS.index(c), d) for c in CODED_CELLS for d in competitors]
    estimates = np.array([means[c, mass] - means[c, d] for c, d in pairs])
    draws = np.stack([boot[:, c, mass] - boot[:, c, d] for c, d in pairs], axis=1)
    lower, critical = _simultaneous_lower_bounds(estimates, draws)
    family = [{"cell": CELLS[c], "competitor": DIRECTIONS[d], "estimate": float(estimates[i]),
               "simultaneous_lower95": float(lower[i])} for i, (c, d) in enumerate(pairs)]
    result = {"clinical_family": family, "max_t_critical": float(critical), "cells": {},
              "bootstrap": {"resamples": len(weights), "seed": BOOTSTRAP_SEED,
                            "unit": "one image per patient", "method": "joint centered studentized max-T"}}
    margins = means[:, mass] - means[:, competitors].max(axis=1)
    margin_draws = boot[:, :, mass] - boot[:, :, competitors].max(axis=2)
    for c, cell in enumerate(CELLS):
        random_max = float(means[c, len(ae.CONCEPTS):-1].max())
        sham_abs = float(abs(means[c, -1]))
        bounds = [v["simultaneous_lower95"] for v in family if v["cell"] == cell]
        result["cells"][cell] = {
            "eligible": bool(eligible[cell]),
            "clinical_margin": endpoint(margins[c], margin_draws[:, c]),
            "direction_effects": {d: endpoint(means[c, j], boot[:, c, j]) for j, d in enumerate(DIRECTIONS)},
            "maximum_random_effect": random_max, "absolute_sham_effect": sham_abs,
            "random_reference_rank": float((1 + np.sum(means[c, 6:-1] >= means[c, mass])) / 120),
            "qualifies": bool(cell in CODED_CELLS and eligible[cell] and all(v > 0 for v in bounds)
                              and means[c, mass] > max(0, random_max, sham_abs))}
    result["cross_wording_specificity"] = all(result["cells"][c]["qualifies"] for c in CODED_CELLS)
    result["discovery_wording_replication"] = all(result["cells"][c]["qualifies"] for c in CODED_CELLS[:2])
    result["capability_status"] = "eligible" if all(eligible[c] for c in CODED_CELLS) else "inconclusive"
    comparisons = [(f"{q}/{e}", f"{q}/yes_no") for q in QUESTIONS for e in ENCODINGS[1:]]
    comparisons += [(f"is/{e}", f"show/{e}") for e in ENCODINGS]
    comparisons += [("is/yes_no", "anchor")]
    result["paired_margin_comparisons"] = {
        f"{a} - {b}": endpoint(margins[CELLS.index(a)] - margins[CELLS.index(b)],
                              margin_draws[:, CELLS.index(a)] - margin_draws[:, CELLS.index(b)])
        for a, b in comparisons}
    return result


def rank_correlations(scores):
    ranked = [rankdata(s) for s in scores]
    return {f"{CELLS[a]}:{CELLS[b]}": float(np.corrcoef(ranked[a], ranked[b])[0, 1])
            if np.ptp(ranked[a]) and np.ptp(ranked[b]) else None
            for a in range(len(CELLS)) for b in range(a + 1, len(CELLS))}


def protocol_metadata():
    return {"gate": "qwen7b-mass-prompt-encoding-specificity", "n_eval": EVAL_ROWS,
            "cell_order": list(CELLS), "prompts": {c: prompt_for(c) for c in CELLS},
            "directions": list(DIRECTIONS), "alpha": .25, "alpha_mode": "reltoken",
            "locus": ae.LOCUS, "module": ae.EXPECTED_LOCUS, "resolution": [336, 336],
            "scientific_outcomes": SCIENTIFIC_OUTCOMES, "selection_seed": SELECTION_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "calibration_seed": CALIBRATION_SEED, "model_id": ae.MODEL_ID,
            "model_revision": ae.MODEL_REVISION}


def summarize(per_image, out):
    meta = ae.read_json(per_image.parent / "meta.json")
    if any(meta.get(k) != v for k, v in protocol_metadata().items()):
        raise ValueError("summary metadata differs from registered protocol")
    rows, calibration_rows = meta["rows"], meta["calibration_rows"]
    receipt = ae.read_json(per_image.parent / "registered-rows.json")
    for key, field in (("row_id", "row_ids"), ("patient_id", "patient_ids")):
        if ({r[key] for r in rows} & {r[key] for r in calibration_rows}
                or [r[key] for r in rows] != receipt[field]
                or [r[key] for r in calibration_rows] != receipt["calibration_" + field]):
            raise ValueError("cohort receipt mismatch or cohorts not patient-disjoint")
    with per_image.open(newline="", encoding="utf-8") as stream:
        grid = validate_grid(csv.DictReader(stream), rows)
    with (per_image.parent / "calibration.csv").open(newline="", encoding="utf-8") as stream:
        calibration = validate_grid(csv.DictReader(stream), calibration_rows, conditions=CONFIGURATIONS[:1])
    preflight = ae.read_json(per_image.parent / "preflight.json")
    if not measurement_passed(preflight):
        raise ValueError("measurement preflight is not valid")
    weights = bootstrap_weights(EVAL_ROWS)
    cal_weights = ae.bootstrap_weights(EVAL_ROWS, CALIBRATION_SEED)
    cal_labels = np.array([int(r["Mass"]) for r in calibration_rows])
    eligibility = {c: calibration_metrics(cal_labels, calibration[i, 0], cal_weights)
                   for i, c in enumerate(CELLS)}
    effects = grid[:, 1:, :, 1] - grid[:, :1, :, 1]
    result = effect_summary(effects, weights, {c: v["eligible"] for c, v in eligibility.items()})
    result.update(protocol_metadata())
    result.update(n_outcomes=int(grid[..., 0].size), n_calibration_outcomes=int(calibration[..., 0].size),
                  n_scientific_outcomes=int(grid[..., 0].size + calibration[..., 0].size),
                  calibration=eligibility, source=meta["source"], source_commit=meta["source_commit"],
                  cohort_receipt=receipt, preflight=preflight,
                  calibration_spearman=rank_correlations(calibration[:, 0, :, 1]),
                  confirmation_baseline_spearman=rank_correlations(grid[:, 0, :, 1]))
    labels = np.array([int(r["Mass"]) for r in rows])
    all_weights = np.vstack((np.ones(EVAL_ROWS), weights))
    for c, cell in enumerate(CELLS):
        base = grid[c, 0]
        base_auc = ae.weighted_auroc(labels, base[:, 1], all_weights)
        base_brier = all_weights @ ((base[:, 2] - labels) ** 2) / EVAL_ROWS
        summary = result["cells"][cell]
        summary["baseline"] = {"auroc": endpoint(base_auc[0], base_auc[1:]),
                               "brier": endpoint(base_brier[0], base_brier[1:]),
                               "mean_raw_margin": float(base[:, 0].mean())}
        summary["metric_changes"] = {}
        for d, direction in enumerate(DIRECTIONS, 1):
            scores = grid[c, d]
            auc = ae.weighted_auroc(labels, scores[:, 1], all_weights)
            brier = all_weights @ ((scores[:, 2] - labels) ** 2) / EVAL_ROWS
            probability_delta = all_weights @ (scores[:, 2] - base[:, 2]) / EVAL_ROWS
            summary["metric_changes"][direction] = {
                "steered_auroc": endpoint(auc[0], auc[1:]), "steered_brier": endpoint(brier[0], brier[1:]),
                "auroc_change": endpoint(auc[0] - base_auc[0], auc[1:] - base_auc[1:]),
                "brier_change": endpoint(brier[0] - base_brier[0], brier[1:] - base_brier[1:]),
                "probability_effect": endpoint(probability_delta[0], probability_delta[1:])}
        print(f"summarized {cell}", flush=True)
    ae.write_json(out, result)
    return result


def measurement_passed(preflight):
    change = preflight.get("nonzero_max_logit_change")
    throughput = preflight.get("throughput", {})
    elapsed = throughput.get("elapsed_seconds", 0)
    predicted = elapsed * SCIENTIFIC_OUTCOMES / PILOT_EQUIVALENTS
    return bool(preflight.get("passed") is True and preflight.get("alpha_zero_exact") is True
                and preflight.get("module") == ae.EXPECTED_LOCUS
                and isinstance(change, (int, float)) and np.isfinite(change) and change > 0
                and mapping_passed(preflight.get("mapping_cases", []))
                and throughput.get("passed") is True and 0 < predicted <= MAX_PREDICTED_SECONDS
                and throughput.get("image_condition_equivalents") == PILOT_EQUIVALENTS
                and throughput.get("scientific_outcomes") == SCIENTIFIC_OUTCOMES
                and throughput.get("predicted_scientific_seconds") == predicted)


def run(ownership_run, manifest, gpu, out, preflight_only=False, cohort_only=False):
    source, rows, calibration, validation, receipt = load_source(ownership_run, manifest)
    out.mkdir(parents=True, exist_ok=True)
    meta = {**protocol_metadata(), "source": source,
            "source_commit": os.environ.get("SOURCE_COMMIT", "unknown"),
            "rows": [{k: r[k] for k in ("row_id", "patient_id", "Mass")} for r in rows],
            "calibration_rows": [{k: r[k] for k in ("row_id", "patient_id", "Mass")} for r in calibration],
            "validation_row_ids": [r["row_id"] for r in validation], "batch_size": 16}
    for name, payload in (("registered-rows.json", receipt), ("meta.json", meta)):
        if (out / name).exists() and ae.read_json(out / name) != payload:
            raise ValueError("checkpoint metadata differs from this registered run")
        ae.write_json(out / name, payload)
    if cohort_only:
        return True
    try:
        from .gpu_env import bind_gpu
    except ImportError:
        from gpu_env import bind_gpu
    bind_gpu(gpu)
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    try:
        from .intervene import Steerer, load_registered_directions
        from .loci import loci_for
        from .registry import REGISTRY
    except ImportError:
        from intervene import Steerer, load_registered_directions
        from loci import loci_for
        from registry import REGISTRY
    with np.load(source["directions"], allow_pickle=False) as bundle:
        dimension = int(bundle["raw_dim"])
    clinical, digest = load_registered_directions(source["directions"], dimension, "Mass")
    if digest != source["directions_sha256"]:
        raise ValueError("registered small direction bundle identity mismatch")
    directions = common_directions(clinical)
    arch = REGISTRY[ae.ARCH]
    locus = next(lo for lo in loci_for(arch) if lo.name == ae.LOCUS)
    if (locus.module != ae.EXPECTED_LOCUS or arch.hf_id != ae.MODEL_ID
            or arch.processor_kwargs != {"min_pixels": 336 * 336, "max_pixels": 336 * 336}):
        raise ValueError("pinned Qwen registry configuration changed")
    processor = AutoProcessor.from_pretrained(source["model_source"], local_files_only=True,
                                              **arch.processor_kwargs)
    tokenizer = processor.tokenizer
    preflight = {"validation_row_ids": [r["row_id"] for r in validation],
                 "mapping_cases": [], "module": locus.module, "passed": False}
    try:
        ab_ids = ae.singleton_ids(tokenizer)
        preflight["ab_token_ids"] = ab_ids
        preflight["yes_no_token_ids"] = ae.singleton_ids(tokenizer, ("yes", "Yes", "YES"), ("no", "No", "NO"))
    except ValueError as error:
        preflight["error"] = str(error)
        ae.write_json(out / "preflight.json", preflight)
        return False
    model = AutoModelForImageTextToText.from_pretrained(
        source["model_source"], dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True).eval()

    def inputs_for(prompt, image_rows=()):
        content = ([{"type": "image"}] if image_rows else []) + [{"type": "text", "text": prompt}]
        text = processor.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)
        images = []
        for row in image_rows:
            with Image.open(row["image_path"]) as image:
                images.append(image.convert("RGB"))
        kwargs = {"images": images} if images else {}
        return processor(text=[text] * max(1, len(images)), return_tensors="pt",
                         padding=True, **kwargs).to("cuda:0")

    def forward(inputs):
        with torch.inference_mode():
            return model(**inputs).logits[:, -1:, :].float().clone()

    steerer = Steerer(model, locus.module, positions=locus.positions, mode="reltoken")
    try:
        validation_inputs = inputs_for(prompt_for("anchor"), validation[:1])
        clean = forward(validation_inputs)
        with steerer:
            steerer.vec, steerer.alpha = directions["Mass"], 0.0
            zero = forward(validation_inputs)
            steerer.alpha = .25
            changed = forward(validation_inputs)
        preflight["alpha_zero_exact"] = bool(torch.equal(clean, zero))
        change = float((changed - clean).abs().max().item())
        preflight["nonzero_max_logit_change"] = change if np.isfinite(change) else None
        for case in mapping_cases():
            encoding = "standard" if case["encoding"] == "yes_no" else case["encoding"]
            score = ae.score_logits(forward(inputs_for(case["prompt"])), tokenizer, encoding, ab_ids)[0]
            preflight["mapping_cases"].append({**case, "raw_margin": float(score[0]),
                                               "semantic_margin": float(score[1])})
        if preflight["alpha_zero_exact"] and np.isfinite(change) and change > 0 and mapping_passed(preflight["mapping_cases"]):
            pilot_conditions = (CONFIGURATIONS[:7] + (("random0", .25), ("random118", .25), ("sham", .25)))
            started = time.perf_counter()
            with steerer:
                for batch in range(64):
                    cell = CELLS[batch % len(CELLS)]
                    name, alpha = pilot_conditions[batch % len(pilot_conditions)]
                    steerer.vec, steerer.alpha = (None if alpha == 0 else directions[name]), alpha
                    inputs = inputs_for(prompt_for(cell), validation)
                    ae.score_logits(forward(inputs), tokenizer, score_encoding(cell), ab_ids)
                    del inputs
                    if (batch + 1) % 16 == 0:
                        print(f"validation throughput: {(batch + 1) * 16}/{PILOT_EQUIVALENTS}", flush=True)
            elapsed = time.perf_counter() - started
            predicted = elapsed * SCIENTIFIC_OUTCOMES / PILOT_EQUIVALENTS
            preflight["throughput"] = {"image_condition_equivalents": PILOT_EQUIVALENTS,
                "elapsed_seconds": elapsed, "predicted_scientific_seconds": predicted,
                "scientific_outcomes": SCIENTIFIC_OUTCOMES, "maximum_seconds": MAX_PREDICTED_SECONDS,
                "passed": bool(0 < elapsed and predicted <= MAX_PREDICTED_SECONDS)}
            preflight["passed"] = preflight["throughput"]["passed"]
    except Exception as error:
        preflight["passed"] = False
        preflight["error"] = f"{type(error).__name__}: {error}"
        ae.write_json(out / "preflight.json", preflight)
        raise
    ae.write_json(out / "preflight.json", preflight)
    if preflight_only or not measurement_passed(preflight):
        return measurement_passed(preflight)
    del validation_inputs, clean, zero, changed

    def evaluate(cohort, conditions, prefix, combined_name):
        for cell in CELLS:
            checkpoint = out / f"{prefix}{cell.replace('/', '-')}.csv"
            if checkpoint.exists():
                with checkpoint.open(newline="", encoding="utf-8") as stream:
                    validate_grid(csv.DictReader(stream), cohort, (cell,), conditions)
                print(f"{prefix}{cell}: completed checkpoint reused", flush=True)
                continue
            temp = checkpoint.with_suffix(".csv.tmp")
            with temp.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                for name, alpha in conditions:
                    steerer.vec, steerer.alpha = (None if alpha == 0 else directions[name]), alpha
                    for start in range(0, EVAL_ROWS, 16):
                        chunk = cohort[start:start + 16]
                        inputs = inputs_for(prompt_for(cell), chunk)
                        scores = ae.score_logits(forward(inputs), tokenizer, score_encoding(cell), ab_ids)
                        for row, score in zip(chunk, scores, strict=True):
                            writer.writerow(dict(row_id=row["row_id"], patient_id=row["patient_id"],
                                cell=cell, direction=name, alpha=alpha, locus=ae.LOCUS, label=row["Mass"],
                                **dict(zip(FIELDS[-3:], score))))
                        del inputs
                    stream.flush()
                    print(f"{prefix}{cell}: {name} complete", flush=True)
            temp.replace(checkpoint)
        combined = out / combined_name
        temp = combined.with_suffix(".csv.tmp")
        with temp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for cell in CELLS:
                with (out / f"{prefix}{cell.replace('/', '-')}.csv").open(newline="", encoding="utf-8") as chunk:
                    writer.writerows(csv.DictReader(chunk))
        temp.replace(combined)

    with steerer:
        evaluate(calibration, CONFIGURATIONS[:1], "calibration-", "calibration.csv")
        evaluate(rows, CONFIGURATIONS, "", "per-image.csv")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "cohort"):
        child = sub.add_parser(command)
        for name in ("ownership-run", "manifest", "out"):
            child.add_argument(f"--{name}", type=Path, required=True)
        if command == "run":
            child.add_argument("--gpu", type=int, required=True)
            child.add_argument("--preflight-only", action="store_true")
    summary = sub.add_parser("summarize")
    for name in ("per-image", "out"):
        summary.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args.per_image, args.out)
    elif args.command == "cohort":
        run(args.ownership_run, args.manifest, None, args.out, cohort_only=True)
    elif not run(args.ownership_run, args.manifest, args.gpu, args.out, args.preflight_only):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
