"""Registered paired answer-encoding sensitivity on the accepted Qwen ownership cohort."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

try:
    from .qwen_causal_ownership_gate import CONCEPTS, PROMPTS, REGISTERED_ROWS_SHA256
    from .qwen_consolidation_input_closure import OWNERSHIP_COMMIT, OWNERSHIP_RUN_ID
    from .qwen_direction_specificity_gate import SOURCE_COMMIT, SOURCE_RUN_ID, _metadata
    from .qwen_effusion_gate import ARCH, EXPECTED_LOCUS, LOCUS, MODEL_ID, MODEL_REVISION
except ImportError:
    from qwen_causal_ownership_gate import CONCEPTS, PROMPTS, REGISTERED_ROWS_SHA256
    from qwen_consolidation_input_closure import OWNERSHIP_COMMIT, OWNERSHIP_RUN_ID
    from qwen_direction_specificity_gate import SOURCE_COMMIT, SOURCE_RUN_ID, _metadata
    from qwen_effusion_gate import ARCH, EXPECTED_LOCUS, LOCUS, MODEL_ID, MODEL_REVISION

EVAL_ROWS = 200
N_RANDOM = 20
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260907
CALIBRATION_SEED = 20260908
PILOT_EQUIVALENTS = 1024
SCIENTIFIC_OUTCOMES = 200400
MAX_PREDICTED_SECONDS = 3.75 * 3600
ENCODINGS = ("standard", "A_present", "B_present")
DOSES = (-0.25, 0.25)
DIRECTIONS = (*CONCEPTS, *(f"random{i}" for i in range(N_RANDOM)), "sham")
CONFIGURATIONS = (("baseline", 0.0), *((d, a) for d in DIRECTIONS for a in DOSES))
FINDINGS = dict(zip(CONCEPTS, (
    "a pleural effusion", "atelectasis", "a pneumothorax", "cardiomegaly",
    "a lung mass", "a lung nodule",
), strict=True))
STATEMENTS = (
    ("The finding is present.", 1), ("The finding is absent.", 0),
    ("The finding is explicitly reported as present.", 1),
    ("The finding is explicitly reported as absent.", 0),
)
FIELDS = ("row_id", "patient_id", "question", "encoding", "direction", "alpha", "locus",
          "label", "raw_margin", "semantic_margin", "semantic_probability")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mapping_instruction(encoding):
    present, absent = ("A", "B") if encoding == "A_present" else ("B", "A")
    return (f"Answer {present} if the finding is present and {absent} if it is absent. "
            "Reply with A or B only.")


def prompt_for(question, encoding):
    if encoding == "standard":
        return PROMPTS[question]
    if encoding not in ENCODINGS:
        raise ValueError("unknown answer encoding")
    return (f"Does this chest radiograph show {FINDINGS[question]}? "
            + mapping_instruction(encoding))


def singleton_ids(tokenizer, first=("A",), second=("B",)):
    groups = []
    for words in (first, second):
        ids = set()
        for word in words:
            for form in (word, " " + word):
                encoded = tokenizer.encode(form, add_special_tokens=False)
                if len(encoded) == 1:
                    ids.add(encoded[0])
        groups.append(sorted(ids))
    if not all(groups) or set(groups[0]) & set(groups[1]):
        raise ValueError("measurement requires nonempty disjoint singleton token IDs")
    return groups


def orient(raw_margin, encoding):
    if encoding not in ENCODINGS:
        raise ValueError("unknown answer encoding")
    return np.asarray(raw_margin, dtype=np.float64) * (-1 if encoding == "B_present" else 1)


def score_logits(logits, tokenizer, encoding, ab_ids):
    if encoding == "standard":
        try:
            from .intervene import yes_margin_batch, yes_prob_batch
        except ImportError:
            from intervene import yes_margin_batch, yes_prob_batch
        raw = np.asarray(yes_margin_batch(logits, tokenizer))
        probability = np.asarray(yes_prob_batch(logits, tokenizer))
    else:
        last = logits[:, -1, :].float()
        raw = (last[:, ab_ids[0]].max(dim=-1).values
               - last[:, ab_ids[1]].max(dim=-1).values).tolist()
        probability = expit(orient(raw, encoding))
    values = np.stack((raw, orient(raw, encoding), probability), axis=-1)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite answer scores")
    return values


def common_directions(clinical):
    """Replay seed zero: twenty isotropic normals, then one shared coordinate permutation."""
    clinical = {q: np.asarray(clinical[q], dtype=np.float32) / np.linalg.norm(clinical[q])
                for q in CONCEPTS}
    rng = np.random.default_rng(0)
    dimension = len(clinical[CONCEPTS[0]])
    random = {}
    for i in range(N_RANDOM):
        vector = rng.standard_normal(dimension).astype(np.float32)
        random[f"random{i}"] = vector / np.linalg.norm(vector)
    permutation = rng.permutation(dimension)
    return {
        q: {**clinical, **random,
            "sham": clinical[q][permutation] / np.linalg.norm(clinical[q][permutation])}
        for q in CONCEPTS
    }


def mapping_cases():
    return [{"encoding": e, "case": i, "expected_present": y,
                 "prompt": f"{statement} Is the finding present? {mapping_instruction(e)}"}
            for e in ENCODINGS[1:] for i, (statement, y) in enumerate(STATEMENTS)]


def mapping_passed(cases):
    expected = mapping_cases()
    if len(cases) != len(expected):
        return False
    for case, wanted in zip(cases, expected, strict=True):
        margin = float(case.get("semantic_margin", "nan"))
        if (any(case.get(k) != v for k, v in wanted.items()) or not np.isfinite(margin)
                or margin * (2 * wanted["expected_present"] - 1) <= 0):
            return False
    return True


def load_source(ownership_run, manifest):
    root = ownership_run.resolve(strict=True)
    metadata = _metadata(root / "metadata.env")
    expected = {"RUN_ID": OWNERSHIP_RUN_ID, "SOURCE_COMMIT": OWNERSHIP_COMMIT, "COMMAND_STATUS": "0",
                    "DISPATCHER_STATUS": "0", "CLEANUP_STATUS": "0", "ABORT_SIGNAL": "none"}
    if root.name != OWNERSHIP_RUN_ID or any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("accepted ownership run identity or terminal metadata mismatch")
    for name in ("command_exit_status", "exit_status"):
        if (root / name).read_text(encoding="utf-8").strip() != "0":
            raise ValueError("ownership run did not complete successfully")
    source = read_json(root / "artifacts/source-reference.json")
    for key, value in {"source_run_id": SOURCE_RUN_ID, "source_commit": SOURCE_COMMIT,
                           "model_id": MODEL_ID, "model_revision": MODEL_REVISION}.items():
        if source.get(key) != value:
            raise ValueError(f"accepted source reference mismatch: {key}")
    registered = read_json(root / "artifacts/registered-rows.json")
    if (registered.get("row_ids_sha256") != REGISTERED_ROWS_SHA256
            or len(registered["row_ids"]) != 400 or len(registered["patient_ids"]) != 400):
        raise ValueError("accepted ownership cohort receipt mismatch")
    with manifest.open(newline="", encoding="utf-8") as stream:
        manifest_rows = list(csv.DictReader(stream))
    by_id = {row["row_id"]: row for row in manifest_rows}
    ids, patients = registered["row_ids"], registered["patient_ids"]
    if (len(by_id) != len(manifest_rows)
            or len(set(ids)) != 400 or len(set(patients)) != 400):
        raise ValueError("registered paired cohort identity is incomplete")
    selected = []
    for row_id, patient in zip(ids, patients, strict=True):
        row = by_id[row_id]
        if row["patient_id"] != patient or row["split"] != "test":
            raise ValueError("registered row/patient or test split mismatch")
        if any(row[q] not in ("0", "1") for q in CONCEPTS):
            raise ValueError("registered labels must be binary")
        selected.append(row)
    validation = [row for row in manifest_rows if row["split"] in ("val", "validation")][:16]
    if len(validation) != 16:
        raise ValueError("throughput preflight requires 16 validation images")
    if any(row["patient_id"] in patients for row in validation):
        raise ValueError("validation patient overlaps the test cohort")
    ownership = read_json(root / "artifacts/causal-ownership-summary.json")
    if ownership.get("intervention_commit") != OWNERSHIP_COMMIT or ownership.get("n_eval") != 400:
        raise ValueError("accepted ownership summary identity mismatch")
    return source, selected[:EVAL_ROWS], selected[-EVAL_ROWS:], validation, ownership


def validate_grid(records, rows, encodings=ENCODINGS, questions=CONCEPTS, conditions=CONFIGURATIONS):
    """Return encoding x question x configuration x patient x score, in fixed order."""
    n = len(rows)
    if n != EVAL_ROWS or len({r["row_id"] for r in rows}) != n or len({r["patient_id"] for r in rows}) != n:
        raise ValueError("expected 200 unique paired patients")
    positions = {r["row_id"]: i for i, r in enumerate(rows)}
    configurations = {c: i for i, c in enumerate(conditions)}
    shape = (len(encodings), len(questions), len(conditions), n)
    values, seen = np.empty((*shape, 3)), np.zeros(shape, dtype=bool)
    for record in records:
        try:
            e, q = encodings.index(record["encoding"]), questions.index(record["question"])
            c = configurations[(record["direction"], float(record["alpha"]))]
            i = positions[record["row_id"]]
        except (KeyError, ValueError) as error:
            raise ValueError("unexpected grid identity") from error
        if (seen[e, q, c, i] or record["patient_id"] != rows[i]["patient_id"]
                or record["locus"] != LOCUS or float(record["label"]) != int(rows[i][questions[q]])):
            raise ValueError("duplicate grid entry or paired patient/label mismatch")
        score = np.array([float(record[k]) for k in FIELDS[-3:]])
        if (not np.isfinite(score).all() or not 0 <= score[2] <= 1
                or not np.isclose(score[1], orient(score[0], encodings[e]), atol=1e-7, rtol=0)
                or not np.isclose(score[2], expit(score[1]), atol=1e-7, rtol=0)):
            raise ValueError("invalid finite score or answer orientation")
        values[e, q, c, i], seen[e, q, c, i] = score, True
    if not seen.all():
        raise ValueError("incomplete answer-encoding grid")
    return values


def run(ownership_run, manifest, gpu, out, preflight_only=False):
    source, rows, calibration_rows, validation, ownership = load_source(ownership_run, manifest)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"analysis": "registered_paired_sensitivity", "n_eval": EVAL_ROWS, "batch_size": 16,
                "ownership_run_id": OWNERSHIP_RUN_ID, "ownership_commit": OWNERSHIP_COMMIT,
                "source": source, "original_ownership_decisions": ownership,
                "source_commit": os.environ.get("SOURCE_COMMIT", "unknown"),
                "locus": LOCUS, "module": EXPECTED_LOCUS, "alpha_mode": "reltoken", "seed": 0,
                "encodings": list(ENCODINGS), "directions": list(DIRECTIONS), "doses": list(DOSES),
                "model_local_only": True, "resolution": [336, 336],
                "prompts": {e: {q: prompt_for(q, e) for q in CONCEPTS} for e in ENCODINGS},
                "rows": [{k: r[k] for k in ("row_id", "patient_id", *CONCEPTS)} for r in rows],
                "calibration_rows": [{k: r[k] for k in ("row_id", "patient_id", *CONCEPTS)} for r in calibration_rows],
                "validation_row_ids": [r["row_id"] for r in validation],
                "calibration_seed": CALIBRATION_SEED, "scientific_outcomes": SCIENTIFIC_OUTCOMES}
    if (out / "meta.json").exists() and read_json(out / "meta.json") != meta:
        raise ValueError("checkpoint metadata differs from this registered run")
    write_json(out / "meta.json", meta)
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
    clinical, digest = load_registered_directions(source["directions"], dimension, CONCEPTS[0])
    if digest != source["directions_sha256"]:
        raise ValueError("registered small direction bundle identity mismatch")
    directions = common_directions(clinical)
    arch = REGISTRY[ARCH]
    locus = next(lo for lo in loci_for(arch) if lo.name == LOCUS)
    if (locus.module != EXPECTED_LOCUS or arch.hf_id != MODEL_ID
            or arch.processor_kwargs != {"min_pixels": 336 * 336, "max_pixels": 336 * 336}):
        raise ValueError("pinned Qwen registry configuration changed")
    processor = AutoProcessor.from_pretrained(source["model_source"], local_files_only=True,
                                              **arch.processor_kwargs)
    tokenizer = processor.tokenizer
    preflight = {"validation_row_ids": [r["row_id"] for r in validation], "mapping_cases": [], "passed": False}
    try:
        ab_ids = singleton_ids(tokenizer)
        preflight["ab_token_ids"] = ab_ids
        preflight["yes_no_token_ids"] = singleton_ids(tokenizer, ("yes", "Yes", "YES"), ("no", "No", "NO"))
    except ValueError as error:
        preflight["error"] = str(error)
        write_json(out / "preflight.json", preflight)
        return False
    model = AutoModelForImageTextToText.from_pretrained(
        source["model_source"], dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True,
    ).eval()

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
        validation_inputs = inputs_for(PROMPTS[CONCEPTS[0]], validation[:1])
        clean = forward(validation_inputs)
        with steerer:
            steerer.vec, steerer.alpha = clinical[CONCEPTS[0]], 0.0
            zero = forward(validation_inputs)
            steerer.alpha = 0.25
            changed = forward(validation_inputs)
        preflight["alpha_zero_exact"] = bool(torch.equal(clean, zero))
        preflight["nonzero_max_logit_change"] = float((changed - clean).abs().max().item())
        for case in mapping_cases():
            score = score_logits(forward(inputs_for(case["prompt"])), tokenizer, case["encoding"], ab_ids)[0]
            preflight["mapping_cases"].append({**case, "raw_margin": float(score[0]),
                                                "semantic_margin": float(score[1])})
        change = preflight["nonzero_max_logit_change"]
        preflight["passed"] = bool(preflight["alpha_zero_exact"] and np.isfinite(change)
                                   and change > 0 and mapping_passed(preflight["mapping_cases"]))
        if not np.isfinite(change):
            preflight["nonzero_max_logit_change"] = None
        if preflight["passed"]:
            started = time.perf_counter()
            with steerer:
                for batch in range(PILOT_EQUIVALENTS // 16):
                    encoding = ENCODINGS[1 + batch % 2]
                    steerer.vec, steerer.alpha = clinical[CONCEPTS[0]], DOSES[batch % 2]
                    inputs = inputs_for(prompt_for(CONCEPTS[0], encoding), validation)
                    score_logits(forward(inputs), tokenizer, encoding, ab_ids)
                    del inputs
                    if (batch + 1) % 16 == 0:
                        print(f"validation throughput: {(batch + 1) * 16}/{PILOT_EQUIVALENTS} equivalents", flush=True)
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
        write_json(out / "preflight.json", preflight)
        raise
    write_json(out / "preflight.json", preflight)
    print(f"preflight passed={preflight['passed']}", flush=True)
    if preflight_only or not preflight["passed"]:
        return preflight["passed"]
    del validation_inputs, clean, zero, changed
    def evaluate_cohort(cohort, encodings, conditions, prefix, combined_name):
        for encoding in encodings:
            for question in CONCEPTS:
                checkpoint = out / f"{prefix}{question.lower()}-{encoding}.csv"
                if checkpoint.exists():
                    with checkpoint.open(newline="", encoding="utf-8") as stream:
                        validate_grid(csv.DictReader(stream), cohort, (encoding,), (question,), conditions)
                    print(f"{prefix}{encoding}/{question}: completed checkpoint reused", flush=True)
                    continue
                temp = checkpoint.with_suffix(".csv.tmp")
                with temp.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    writer.writeheader()
                    for start in range(0, EVAL_ROWS, 16):
                        chunk = cohort[start:start + 16]
                        inputs = inputs_for(prompt_for(question, encoding), chunk)
                        for name, alpha in conditions:
                            steerer.vec = None if alpha == 0 else directions[question][name]
                            steerer.alpha = alpha
                            scores = score_logits(forward(inputs), tokenizer, encoding, ab_ids)
                            for row, score in zip(chunk, scores, strict=True):
                                writer.writerow(dict(row_id=row["row_id"], patient_id=row["patient_id"],
                                    question=question, encoding=encoding, direction=name, alpha=alpha,
                                    locus=LOCUS, label=row[question], **dict(zip(FIELDS[-3:], score))))
                        stream.flush()
                        print(f"{prefix}{encoding}/{question}: {start + len(chunk)}/{EVAL_ROWS} images", flush=True)
                        del inputs
                temp.replace(checkpoint)
                print(f"{prefix}{encoding}/{question}: checkpoint complete", flush=True)
        combined = out / combined_name
        temp = combined.with_suffix(".csv.tmp")
        with temp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for encoding in encodings:
                for question in CONCEPTS:
                    with (out / f"{prefix}{question.lower()}-{encoding}.csv").open(newline="", encoding="utf-8") as chunk:
                        writer.writerows(csv.DictReader(chunk))
        temp.replace(combined)

    with steerer:
        evaluate_cohort(calibration_rows, ENCODINGS[1:], CONFIGURATIONS[:1], "calibration-", "calibration.csv")
        evaluate_cohort(rows, ENCODINGS, CONFIGURATIONS, "", "per-image.csv")
    return True


def bootstrap_weights(n, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    return np.array([np.bincount(rng.integers(n, size=n), minlength=n)
                     for _ in range(BOOTSTRAP_RESAMPLES)], dtype=np.float64)


def weighted_auroc(labels, scores, weights):
    """Tie-aware AUROC for joint patient count weights; constant-label draws are NaN."""
    order = np.argsort(scores, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
    positive = np.add.reduceat(weights[:, order] * labels[order], starts, axis=1)
    negative = np.add.reduceat(weights[:, order] * (1 - labels[order]), starts, axis=1)
    denominator = positive.sum(axis=1) * negative.sum(axis=1)
    numerator = (positive * (np.cumsum(negative, axis=1) - negative / 2)).sum(axis=1)
    return np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)


def interval(draws):
    valid = draws[np.isfinite(draws)]
    return {"descriptive_ci95": np.percentile(valid, [2.5, 97.5]).tolist() if len(valid) else None,
                "valid_bootstrap_count": len(valid)}


def crossover(raw_a_effect, raw_b_effect):
    return (raw_a_effect - raw_b_effect) / 2, (raw_a_effect + raw_b_effect) / 2


def unbiased_squared_mean(values, weights):
    """U-statistic of the squared mean, treating repeated bootstrap patients as a multiset."""
    n = weights.sum(axis=1)[:, None, None]
    mean = np.einsum("bi,dqi->bdq", weights, values) / n
    second_sum = np.einsum("bi,dqi->bdq", weights, values ** 2)
    return n / (n - 1) * (mean ** 2 - second_sum / n ** 2)


def energy_summary(semantic, token, weights, eligible):
    all_weights = np.vstack((np.ones(semantic.shape[-1]), weights))
    corrected = unbiased_squared_mean(token, all_weights) - unbiased_squared_mean(semantic, all_weights)
    mean_s = np.einsum("bi,dqi->bdq", all_weights / semantic.shape[-1], semantic)
    mean_t = np.einsum("bi,dqi->bdq", all_weights / semantic.shape[-1], token)
    naive = mean_t ** 2 - mean_s ** 2

    def endpoint(samples):
        return dict(estimate=float(samples[0]), **interval(samples[1:]))

    primary = {"eligible_question_count": len(eligible), "eligible_questions": [CONCEPTS[q] for q in eligible],
                   "status": "estimable" if eligible else "inconclusive", "estimate": None, "descriptive_ci95": None}
    controls = {}
    if eligible:
        per_direction = corrected[:, :, eligible].sum(axis=-1)
        primary.update(endpoint(per_direction[:, :6].mean(axis=1)))
        controls = {d: endpoint(per_direction[:, i]) for i, d in enumerate(DIRECTIONS[6:], 6)}
    maximum = max((v["estimate"] for d, v in controls.items() if d != "sham"), default=None)
    sham = controls.get("sham", {}).get("estimate")
    primary["encoding_even_beyond_controls"] = bool(eligible and primary["descriptive_ci95"][0] > 0
        and primary["estimate"] > maximum and primary["estimate"] > abs(sham))
    primary["ci95"] = primary["descriptive_ci95"]
    return {"semantic_matrix": mean_s[0, :6].tolist(), "token_matrix": mean_t[0, :6].tolist(),
            "component_coordinates": {"semantic": "mapping-odd: (raw_A_present - raw_B_present)/2",
                                      "token": "mapping-even: (raw_A_present + raw_B_present)/2"},
            "estimator": "unbiased squared mean; sum questions, mean clinical directions",
            "primary": primary, "eligible_controls": controls, "maximum_random_estimate": maximum,
            "absolute_sham_estimate": abs(sham) if sham is not None else None,
            "all_questions_descriptive": endpoint(corrected[:, :6].sum(axis=-1).mean(axis=-1)),
            "naive_all_questions_descriptive": endpoint(naive[:, :6].sum(axis=-1).mean(axis=-1)),
            "all_question_controls_descriptive": {d: endpoint(corrected[:, i].sum(axis=-1))
                                                    for i, d in enumerate(DIRECTIONS[6:], 6)}}


def summarize(per_image, out):
    meta, preflight = read_json(per_image.parent / "meta.json"), read_json(per_image.parent / "preflight.json")
    expected = {"analysis": "registered_paired_sensitivity", "n_eval": EVAL_ROWS,
                    "ownership_run_id": OWNERSHIP_RUN_ID, "ownership_commit": OWNERSHIP_COMMIT,
                    "locus": LOCUS, "module": EXPECTED_LOCUS, "alpha_mode": "reltoken",
                    "encodings": list(ENCODINGS), "directions": list(DIRECTIONS), "doses": list(DOSES),
                    "prompts": {e: {q: prompt_for(q, e) for q in CONCEPTS} for e in ENCODINGS}}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("summary metadata differs from registered protocol")
    for key in ("row_id", "patient_id"):
        if {r[key] for r in meta["rows"]} & {r[key] for r in meta["calibration_rows"]}:
            raise ValueError("capability calibration must be patient-disjoint from effect cohort")
    with per_image.open(newline="", encoding="utf-8") as stream:
        grid = validate_grid(csv.DictReader(stream), meta["rows"])
    with (per_image.parent / "calibration.csv").open(newline="", encoding="utf-8") as stream:
        calibration = validate_grid(csv.DictReader(stream), meta["calibration_rows"],
                                    ENCODINGS[1:], conditions=CONFIGURATIONS[:1])
    weights = bootstrap_weights(EVAL_ROWS)
    all_weights = np.vstack((np.ones(EVAL_ROWS), weights))
    labels = np.array([[int(r[q]) for r in meta["rows"]] for q in CONCEPTS])
    mapping_ok = mapping_passed(preflight.get("mapping_cases", []))
    measurement_ok = (preflight.get("passed") is True and preflight.get("alpha_zero_exact") is True
                      and isinstance(preflight.get("nonzero_max_logit_change"), (int, float))
                      and np.isfinite(preflight["nonzero_max_logit_change"])
                      and preflight["nonzero_max_logit_change"] > 0
                      and preflight.get("throughput", {}).get("passed") is True)
    result = {"analysis": "registered_paired_sensitivity", "n_eval": EVAL_ROWS, "n_outcomes": int(grid[..., 0].size),
                  "concepts": list(CONCEPTS), "matrix_orientation": "rows directions; columns questions",
                  "ownership_run_id": OWNERSHIP_RUN_ID, "ownership_commit": OWNERSHIP_COMMIT,
                  "original_ownership_decisions": meta["original_ownership_decisions"],
                  "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                                 "unit": "one image per patient", "method": "joint count weights"},
                  "preflight": preflight, "encodings": {}, "eligibility": {}, "baseline_spearman": {},
                  "n_calibration_outcomes": int(calibration[..., 0].size),
                  "n_scientific_outcomes": int(grid[..., 0].size + calibration[..., 0].size),
                  "auroc_score": "semantic_margin", "calibration_bootstrap_seed": CALIBRATION_SEED,
                  "calibration_baseline": {}}
    for e, encoding in enumerate(ENCODINGS):
        summary = {"baseline": {}, "doses": {}, "metric_changes": {}}
        for q, question in enumerate(CONCEPTS):
            y = labels[q]
            base_p = grid[e, q, 0, :, 2]
            base_auc = weighted_auroc(y, grid[e, q, 0, :, 1], all_weights)
            base_brier = all_weights @ ((base_p - y) ** 2) / EVAL_ROWS
            summary["baseline"][question] = {"auroc": float(base_auc[0]) if np.isfinite(base_auc[0]) else None,
                "brier": float(base_brier[0]), "auroc_bootstrap": interval(base_auc[1:]), "brier_bootstrap": interval(base_brier[1:]),
                "mean_raw_margin": float(grid[e, q, 0, :, 0].mean())}
            summary["metric_changes"][question] = {}
            for c, (direction, alpha) in enumerate(CONFIGURATIONS[1:], 1):
                probability = grid[e, q, c, :, 2]
                auc = weighted_auroc(y, grid[e, q, c, :, 1], all_weights) - base_auc
                brier = all_weights @ ((probability - y) ** 2) / EVAL_ROWS - base_brier
                summary["metric_changes"][question][f"{direction}:{alpha}"] = {
                    "auroc_change": float(auc[0]) if np.isfinite(auc[0]) else None,
                    "auroc_bootstrap": interval(auc[1:]), "brier_change": float(brier[0]), "brier_bootstrap": interval(brier[1:])}
        for alpha in DOSES:
            indexes = [CONFIGURATIONS.index((d, alpha)) for d in DIRECTIONS]
            effects = grid[e, :, indexes] - grid[e, :, 0][None, ...]
            mean = effects.mean(axis=2)
            clinical = mean[:len(CONCEPTS)]
            ownership = {}
            for coordinate, name in ((1, "semantic_margin"), (2, "semantic_probability")):
                boot = np.einsum("bi,dqi->bdq", weights / EVAL_ROWS, effects[:6, :, :, coordinate])
                ownership[name] = {
                    q: dict(estimate=float(clinical[j, j, coordinate] - np.delete(clinical[:, j, coordinate], j).max()),
                            **interval(boot[:, j, j] - np.delete(boot[:, :, j], j, axis=1).max(axis=1)))
                    for j, q in enumerate(CONCEPTS)
                }
            summary["doses"][str(alpha)] = {
                "raw_margin_matrix": clinical[:, :, 0].tolist(), "semantic_margin_matrix": clinical[:, :, 1].tolist(),
                "semantic_probability_matrix": clinical[:, :, 2].tolist(), "ownership_margin": ownership,
                "controls": {d: {q: dict(zip(FIELDS[-3:], mean[i, j].tolist())) for j, q in enumerate(CONCEPTS)}
                          for i, d in enumerate(DIRECTIONS) if d not in CONCEPTS}}
        result["encodings"][encoding] = summary
        print(f"summarized {encoding}", flush=True)
    eligible = []
    calibration_labels = np.array([[int(r[q]) for r in meta["calibration_rows"]] for q in CONCEPTS])
    calibration_weights = np.vstack((np.ones(EVAL_ROWS), bootstrap_weights(EVAL_ROWS, CALIBRATION_SEED)))
    for q, question in enumerate(CONCEPTS):
        positive = int(calibration_labels[q].sum())
        lower, valid = {}, {}
        result["calibration_baseline"][question] = {}
        for e, encoding in enumerate(ENCODINGS[1:]):
            scores = calibration[e, q, 0]
            auc = weighted_auroc(calibration_labels[q], scores[:, 1], calibration_weights)
            draws = auc[1:][np.isfinite(auc[1:])]
            valid[encoding] = len(draws)
            lower[encoding] = float(np.percentile(draws, 5)) if len(draws) else None
            result["calibration_baseline"][question][encoding] = {
                "auroc": float(auc[0]) if np.isfinite(auc[0]) else None,
                "brier": float(np.mean((scores[:, 2] - calibration_labels[q]) ** 2)),
                "mean_raw_margin": float(scores[:, 0].mean()), "auroc_bootstrap": interval(auc[1:])}
        if not mapping_ok:
            status = "mapping_failed"
        elif not measurement_ok:
            status = "measurement_failed"
        elif min(positive, EVAL_ROWS - positive) < 10:
            status = "insufficient_labels"
        elif min(valid.values()) < 1900:
            status = "insufficient_valid_bootstraps"
        elif not all(v is not None and v > 0.5 for v in lower.values()):
            status = "baseline_ineligible"
        else:
            status = "eligible"
            eligible.append(q)
        result["eligibility"][question] = {"status": status, "positive_patients": positive,
            "negative_patients": EVAL_ROWS - positive, "mapping_passed": mapping_ok,
            "cohort": "independent_capability_calibration", "bootstrap_seed": CALIBRATION_SEED,
            "baseline_auroc_one_sided_lower95": lower, "valid_bootstrap_count": valid}
        correlations = {}
        for a, b in ((0, 1), (0, 2), (1, 2)):
            x, y = (rankdata(grid[e, q, 0, :, 1]) for e in (a, b))
            rho = float(np.corrcoef(x, y)[0, 1]) if np.ptp(x) and np.ptp(y) else None
            correlations[f"{ENCODINGS[a]}:{ENCODINGS[b]}"] = rho
        result["baseline_spearman"][question] = {"correlations": correlations, "interpretation": "diagnostic"}
    indexes = [CONFIGURATIONS.index((d, 0.25)) for d in DIRECTIONS]
    raw_a = grid[1, :, indexes, :, 0] - grid[1, :, 0, :, 0][None, ...]
    raw_b = grid[2, :, indexes, :, 0] - grid[2, :, 0, :, 0][None, ...]
    semantic, token = crossover(raw_a, raw_b)
    result["crossover_plus_025"] = energy_summary(semantic, token, weights, eligible)
    result["interpretation"] = "paired sensitivity; scientific interpretation requires eligibility and compatible baseline rankings"
    write_json(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    for name in ("ownership-run", "manifest", "out"):
        run_parser.add_argument(f"--{name}", type=Path, required=True)
    run_parser.add_argument("--gpu", type=int, required=True)
    run_parser.add_argument("--preflight-only", action="store_true")
    summary_parser = sub.add_parser("summarize")
    for name in ("per-image", "out"):
        summary_parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if not run(args.ownership_run, args.manifest, args.gpu, args.out, args.preflight_only):
            raise SystemExit(1)
    else:
        summarize(args.per_image, args.out)


if __name__ == "__main__":
    main()
