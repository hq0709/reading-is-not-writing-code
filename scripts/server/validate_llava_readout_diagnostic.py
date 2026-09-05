"""Independent terminal replay for the LLaVA Effusion readout diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

N = 700
B = 10_000
SUMMARY = "llava-readout-diagnostic-summary"
PILOT_RUN_ID = "20260905T093108Z-42a43207c848-llava-validation"
PILOT_COMMIT = "42a43207c848acfcadecf2f3e0bf866ea71d9dc7"
PILOT_PREPARED_DIGESTS = {
    "artifacts/prepared.json": "ff72f3abbc6a83880e129828fffd5684a21f205b83f8eb2f1932a37dfeac8164",
    "artifacts/prepared.npz": "2bd04f9d145bffe7aeccd08a8fab5cefd76d0e8f76081eb4d92f5d47cc80fb7d",
}
CONDITIONS = ("0Y", "0A", "0B", "1Y", "1A", "1B")
ENCODINGS = ("Y", "A", "B", "Y", "A", "B")
WORDINGS = (
    "Is there a pleural effusion in this chest radiograph?",
    "Does this chest radiograph show a pleural effusion?",
)
SUFFIXES = {
    "Y": "Answer yes or no.",
    "A": "Answer A if the finding is present and B if it is absent. Reply with A or B only.",
    "B": "Answer B if the finding is present and A if it is absent. Reply with A or B only.",
}
STATEMENTS = (
    ("The finding is present.", 1),
    ("The finding is absent.", 0),
    ("The finding is explicitly reported as present.", 1),
    ("The finding is explicitly reported as absent.", 0),
)
TOKEN_GROUPS = {
    "Y": ([3869, 4874, 22483], [694, 1939, 11698]),
    "A": ([319], [350]),
    "B": ([319], [350]),
}
VALIDATION_ROW_IDS = (
    "00010613_005",
    "00007833_003",
    "00003974_003",
    "00009259_000",
    "00005977_010",
    "00009349_012",
    "00003459_018",
    "00003933_000",
    "00008451_011",
    "00010722_006",
    "00005496_000",
    "00005759_028",
    "00010294_050",
    "00000557_000",
    "00009107_007",
    "00003393_010",
)
PRIMARY = (
    "accepted_condition_image_advantage",
    "wording_effect",
    "symbol_vs_yes_no_effect",
    "mapping_asymmetry",
    "wording_by_encoding",
    "wording_by_mapping",
)
MAX_ERROR = 0.0


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def same(actual, expected, name, tolerance=2e-10):
    global MAX_ERROR
    actual, expected = np.asarray(actual), np.asarray(expected)
    check(actual.shape == expected.shape, name + " shape")
    check(np.array_equal(np.isnan(actual), np.isnan(expected)), name + " NaN identity")
    mask = np.isfinite(expected)
    check(np.isfinite(actual[mask]).all(), name + " finite")
    error = float(np.max(np.abs(actual[mask] - expected[mask]))) if mask.any() else 0.0
    MAX_ERROR = max(MAX_ERROR, error)
    check(error <= tolerance, name + " discrepancy " + str(error))


def inventory(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        check(len(fields) == 2, "malformed dispatcher manifest")
        digest, name = fields
        name = name.strip().lstrip("*").removeprefix("./")
        check(
            len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
            "malformed dispatcher digest",
        )
        check(name and name not in result, "duplicate dispatcher manifest member")
        result[name] = digest
    return result


def expected_conditions():
    return [
        {
            "name": CONDITIONS[i],
            "wording": i // 3,
            "encoding": ENCODINGS[i],
            "prompt": f"{WORDINGS[i // 3]} {SUFFIXES[ENCODINGS[i]]}",
        }
        for i in range(6)
    ]


def expected_mapping_cases():
    return [
        {
            "encoding": encoding,
            "case": case_index,
            "statement": statement,
            "prompt": f"{statement} Is the finding present? {SUFFIXES[encoding]}",
            "expected_present": expected,
        }
        for encoding in ("Y", "A", "B")
        for case_index, (statement, expected) in enumerate(STATEMENTS)
    ]


def validate_token_metadata(tokens):
    check(set(tokens) == set(CONDITIONS), "six-condition token metadata identity")
    for condition, encoding in zip(CONDITIONS, ENCODINGS, strict=True):
        record = tokens[condition]
        check(
            set(record) == {"raw_first", "raw_second", "present", "absent", "candidate_ids"},
            "candidate token metadata schema",
        )
        first, second = TOKEN_GROUPS[encoding]
        present, absent = (second, first) if encoding == "B" else (first, second)
        check(
            record
            == {
                "raw_first": first,
                "raw_second": second,
                "present": present,
                "absent": absent,
                "candidate_ids": sorted(set(first + second)),
            },
            "pinned candidate token identity and semantic orientation",
        )


def validate_semantic_preflight(preflight):
    check(
        preflight.get("validation_row_ids") == list(VALIDATION_ROW_IDS)
        and preflight.get("module") == "model.vision_tower.encoder.layers.22"
        and preflight.get("pooling") == "mean over all 577 block-output tokens"
        and preflight.get("forwarded_no_cls_exact") is True
        and preflight.get("clean_repeat_exact") is True,
        "preflight source and consumed-locus identity",
    )
    validate_token_metadata(preflight["token_ids"])
    expected = expected_mapping_cases()
    check(len(preflight["mapping_cases"]) == len(expected), "twelve semantic mapping cases")
    for case, identity in zip(preflight["mapping_cases"], expected, strict=True):
        keys = set(identity) | {"semantic_margin"}
        if identity["encoding"] == "Y":
            keys.add("lse_margin")
        check(set(case) == keys, "semantic mapping case schema")
        check(
            all(case.get(key) == value for key, value in identity.items()),
            "semantic mapping identity",
        )
        sign = 2 * identity["expected_present"] - 1
        check(
            np.isfinite(case["semantic_margin"]) and case["semantic_margin"] * sign > 0,
            "primary semantic orientation",
        )
        if identity["encoding"] == "Y":
            check(
                np.isfinite(case["lse_margin"]) and case["lse_margin"] * sign > 0,
                "lse semantic orientation",
            )
    timing = preflight["throughput"]
    expected_timing = {
        "pilot_equivalents": 512,
        "pilot_seconds": timing["pilot_seconds"],
        "elapsed_seconds": timing["elapsed_seconds"],
        "scientific_outcomes": 8406,
        "projected_seconds": timing["elapsed_seconds"] + 8406 * timing["pilot_seconds"] / 512 + 600,
        "limit_seconds": 3600,
        "passed": timing["elapsed_seconds"] + 8406 * timing["pilot_seconds"] / 512 + 600 <= 3600,
    }
    check(timing == expected_timing and timing["passed"] is True, "registered timing projection")


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
    check(isinstance(records, list) and len(records) == 6, "six text-only outputs")
    for record, spec in zip(records, expected_conditions(), strict=True):
        check(set(record) == expected_keys, "text-only schema")
        check(
            record["condition"] == spec["name"]
            and record["wording"] == spec["wording"]
            and record["encoding"] == spec["encoding"]
            and record["prompt"] == spec["prompt"]
            and record["constant_auroc_against_index_labels"] == 0.5
            and record["subtracting_constant_preserves_image_auroc"] is True,
            "text-only condition identity",
        )
        values = np.asarray(
            [
                record[name]
                for name in ("raw_margin", "semantic_margin", "probability", "answer_token_mass")
            ]
        )
        check(
            np.isfinite(values).all() and 0 <= record["answer_token_mass"] <= 1,
            "finite text-only score",
        )
        semantic = -record["raw_margin"] if spec["encoding"] == "B" else record["raw_margin"]
        check(record["semantic_margin"] == semantic, "text-only semantic orientation")
        same(record["probability"], 1 / (1 + np.exp(-semantic)), "text-only sigmoid", 1e-12)
        check(
            (spec["encoding"] == "Y") == (record["lse_margin"] is not None)
            and (record["lse_margin"] is None or np.isfinite(record["lse_margin"])),
            "text-only lse schema",
        )


def verify_frozen_reader_source(root, prepared, prep):
    pilot = root / "runs" / PILOT_RUN_ID
    receipt = read(
        root / "state/llava-validation-opportunity-internal-20260905T093743Z/receipt.json"
    )
    manifest = inventory(pilot / "SHA256SUMS")
    metadata = dict(
        line.split("=", 1)
        for line in (pilot / "metadata.env").read_text().splitlines()
        if "=" in line
    )
    check(
        receipt.get("status") == "PASS"
        and receipt.get("run_id") == PILOT_RUN_ID
        and receipt.get("source_commit") == PILOT_COMMIT,
        "accepted pilot internal receipt identity",
    )
    check(
        metadata.get("RUN_ID") == PILOT_RUN_ID
        and metadata.get("SOURCE_COMMIT") == PILOT_COMMIT
        and metadata.get("COMMAND_STATUS") == metadata.get("DISPATCHER_STATUS") == "0"
        and metadata.get("CLEANUP_STATUS") == "0"
        and metadata.get("ABORT_SIGNAL") == "none"
        and (pilot / "command_exit_status").read_text().strip() == "0"
        and (pilot / "exit_status").read_text().strip() == "0",
        "accepted pilot terminal identity",
    )
    for name, digest in PILOT_PREPARED_DIGESTS.items():
        check(manifest.get(name) == digest, "accepted pilot dispatcher checksum")
        check(receipt["verified_new_files"].get(name) == digest, "accepted pilot receipt checksum")
    pilot_json = read(pilot / "artifacts/prepared.json")
    check(
        prepared["sources"]["pilot_run_id"] == PILOT_RUN_ID
        and prepared["sources"]["pilot_commit"] == PILOT_COMMIT
        and prepared["sources"]["pilot_prepared_json"] == str(pilot / "artifacts/prepared.json")
        and prepared["sources"]["pilot_prepared_npz"] == str(pilot / "artifacts/prepared.npz")
        and prepared["control_assignments"] == pilot_json["control_assignments"],
        "new frozen reader JSON provenance",
    )
    with np.load(pilot / "artifacts/prepared.npz", allow_pickle=False) as archive:
        for key in (
            "projection",
            "scale",
            "train_mean",
            "coefficients",
            "control_coefficients",
            "control_intercepts",
        ):
            check(
                np.array_equal(prep[key], archive[key]), "new frozen reader array provenance " + key
            )


def weights(indices):
    return np.stack([np.bincount(row, minlength=N) for row in indices])


def validate_bootstrap(indices):
    expected = np.random.default_rng(20260917).integers(0, N, size=(B, N), dtype=np.int64)
    check(
        np.asarray(indices).dtype == np.int64 and np.array_equal(indices, expected),
        "bootstrap index identity",
    )
    return expected


def auc(scores, labels, counts):
    scores, labels = np.asarray(scores), np.asarray(labels)
    order = np.argsort(scores, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
    positive = np.add.reduceat(counts[:, order] * labels[order], boundaries, axis=1)
    negative = np.add.reduceat(counts[:, order] * (1 - labels[order]), boundaries, axis=1)
    product = positive.sum(1) * negative.sum(1)
    numerator = np.sum(positive * (np.cumsum(negative, axis=1) - 0.5 * negative), axis=1)
    return np.divide(numerator, product, out=np.full(len(counts), np.nan), where=product != 0)


def contrasts(values):
    values = np.asarray(values)
    s = (values[..., 1] + values[..., 2]) / 2
    return np.stack(
        (
            values[..., 0, 0],
            (values[..., 1, :] - values[..., 0, :]).mean(-1),
            (s - values[..., :, 0]).mean(-1),
            (values[..., :, 1] - values[..., :, 2]).mean(-1),
            (s[..., 1] - values[..., 1, 0]) - (s[..., 0] - values[..., 0, 0]),
            (values[..., 1, 1] - values[..., 1, 2]) - (values[..., 0, 1] - values[..., 0, 2]),
        ),
        axis=-1,
    )


def source_record(raw, image_directory):
    from src.build_manifest import CONCEPTS, patient_split

    found = set(raw["Finding Labels"].split("|"))
    filename, patient = raw["Image Index"], str(raw["Patient ID"])
    return {
        "row_id": filename.removesuffix(".png"),
        "image_index": filename,
        "image_path": str(image_directory / filename),
        "patient_id": patient,
        "split": patient_split(patient),
        "age": int(raw["Patient Age"]),
        "view_AP": int(raw["View Position"] == "AP"),
        "sex_M": int(raw["Patient Sex"] == "M"),
        "no_finding": int("No Finding" in found),
        "finding_labels": raw["Finding Labels"],
        **{concept: int(concept in found) for concept in CONCEPTS},
        "source_fields": dict(raw),
    }


def replay_allocation(root):
    from src.build_manifest import patient_split

    dataset = root / "datasets/nih-chestxray14"
    with (dataset / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    with (dataset / "Data_Entry_2017_v2020.csv").open(newline="", encoding="utf-8-sig") as stream:
        raw = list(csv.DictReader(stream))
    check(len(manifest) == 26_229 and len(raw) == 112_120, "NIH source row counts")
    excluded = {str(row["patient_id"]) for row in manifest}
    representatives = {}
    for row in raw:
        patient = str(row["Patient ID"])
        if patient in excluded or patient_split(patient) != "val":
            continue
        if (
            patient not in representatives
            or row["Image Index"] < representatives[patient]["Image Index"]
        ):
            representatives[patient] = row
    order = np.random.default_rng(20260916).permutation(sorted(representatives)).tolist()
    selected = [source_record(representatives[p], dataset / "png/images") for p in order[:1400]]
    return {
        "index": selected[:700],
        "donor": selected[700:],
        "unused_patient_ids": order[1400:],
        "eligible_patient_ids": order,
        "selection_seed": 20260916,
        "pairing": "allocation position",
    }


def csv_scores(path, cohort, commit, prompts):
    result = {
        key: np.empty((6, 2, N))
        for key in (
            "raw_margin",
            "semantic_margin",
            "probability",
            "answer_token_mass",
            "lse_margin",
        )
    }
    with path.open(newline="", encoding="utf-8") as stream:
        records = iter(csv.DictReader(stream))
        for c, condition in enumerate(CONDITIONS):
            for role_index, role in enumerate(("index", "donor")):
                for i, row in enumerate(cohort[role]):
                    record = next(records, None)
                    check(record is not None, "complete 8,400-row score grid")
                    index = cohort["index"][i]
                    expected = {
                        "source_commit": commit,
                        "condition": condition,
                        "wording": str(c // 3),
                        "encoding": ENCODINGS[c],
                        "prompt": prompts[c]["prompt"],
                        "image_role": role,
                        "pair_index": str(i),
                        "patient_id": str(row["patient_id"]),
                        "row_id": row["row_id"],
                        "image_path": row["image_path"],
                        "index_label": str(index["Effusion"]),
                        "source_label": str(row["Effusion"]),
                    }
                    check(
                        all(record.get(key) == value for key, value in expected.items()),
                        "ordered score identity",
                    )
                    for key in (
                        "raw_margin",
                        "semantic_margin",
                        "probability",
                        "answer_token_mass",
                    ):
                        result[key][c, role_index, i] = float(record[key])
                    result["lse_margin"][c, role_index, i] = (
                        float(record["lse_margin"]) if ENCODINGS[c] == "Y" else np.nan
                    )
                    check(
                        0 <= result["answer_token_mass"][c, role_index, i] <= 1,
                        "answer-token mass range",
                    )
        check(next(records, None) is None, "no duplicate score rows")
    same(result["probability"], 1 / (1 + np.exp(-result["semantic_margin"])), "semantic sigmoid")
    return result


def verify(run_id, commit):
    root = Path("/home/qingchan/data/concept-flow")
    run, out = root / "runs" / run_id, root / "runs" / run_id / "artifacts"
    check(run.parent == root / "runs" and len(commit) == 40, "run identity")
    metadata = dict(
        line.split("=", 1)
        for line in (run / "metadata.env").read_text().splitlines()
        if "=" in line
    )
    expected = {
        "RUN_ID": run_id,
        "SOURCE_COMMIT": commit,
        "COMMAND_STATUS": "0",
        "DISPATCHER_STATUS": "0",
        "CLEANUP_STATUS": "0",
        "ABORT_SIGNAL": "none",
        "GPU_COUNT": "1",
    }
    check(all(metadata.get(key) == value for key, value in expected.items()), "terminal metadata")
    check(
        all(
            (run / name).read_text().strip() == "0"
            for name in ("command_exit_status", "exit_status")
        ),
        "terminal statuses",
    )
    manifest = inventory(run / "SHA256SUMS")
    names = ["metadata.env", "command_exit_status", "exit_status"] + [
        "artifacts/" + name
        for name in (
            "cohort.json",
            "prepared.json",
            "prepared.npz",
            "preflight.json",
            "text-only.json",
            "per-image.csv",
            "scores.npz",
            SUMMARY + ".json",
            SUMMARY + ".npz",
        )
    ]
    digests = {}
    for name in names:
        with (run / name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        check(manifest.get(name) == digest, "decision-bearing terminal manifest " + name)
        digests[name] = digest

    cohort, prepared, preflight, text, summary = [
        read(out / name)
        for name in (
            "cohort.json",
            "prepared.json",
            "preflight.json",
            "text-only.json",
            SUMMARY + ".json",
        )
    ]
    check(cohort == replay_allocation(root), "identifier-first allocation replay")
    check(prepared["cohort"] == cohort == summary["cohort"], "cohort copies")
    check(prepared["source_commit"] == summary["source_commit"] == commit, "source commit copies")
    check(
        prepared["protocol"] == summary["protocol"] and prepared["sources"] == summary["sources"],
        "protocol/source copies",
    )
    check(
        prepared["protocol"]["conditions"] == expected_conditions(), "registered condition protocol"
    )
    validate_text_only(text)
    check(text == summary["text_only"], "text-only summary copy")
    check(
        preflight == summary["preflight"]
        and preflight["passed"] is True
        and preflight["clean_repeat_exact"] is True
        and preflight["forwarded_no_cls_exact"] is True,
        "preflight acceptance",
    )
    validate_semantic_preflight(preflight)

    with np.load(out / "prepared.npz", allow_pickle=False) as archive:
        prep = {key: archive[key] for key in archive.files}
    verify_frozen_reader_source(root, prepared, prep)
    indices = validate_bootstrap(prep["bootstrap_indices"])
    prompts = summary["protocol"]["conditions"]
    scores = csv_scores(out / "per-image.csv", cohort, commit, prompts)
    with np.load(out / "scores.npz", allow_pickle=False) as archive:
        stored = {key: archive[key] for key in archive.files}
    with np.load(out / (SUMMARY + ".npz"), allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    check(
        saved["source_commit"].item() == commit and saved["route"].item() == summary["route"],
        "summary NPZ identity",
    )
    for key, value in prep.items():
        same(saved[key], value, "prepared array copy " + key)
    for key, value in stored.items():
        same(saved[key], value, "native score array copy " + key)
    token_meta = preflight["token_ids"]
    for c, condition in enumerate(CONDITIONS):
        token = token_meta[condition]
        lookup = {token_id: i for i, token_id in enumerate(token["candidate_ids"])}
        first, second = (
            [lookup[i] for i in token["raw_first"]],
            [lookup[i] for i in token["raw_second"]],
        )
        for role_index, role in enumerate(("index", "donor")):
            key = condition + "__" + role
            logits, partition = stored["candidate_logits__" + key], stored["log_partition__" + key]
            check(
                logits.shape == (N, len(lookup)) and logits.dtype == np.float32,
                "candidate-logit archive",
            )
            raw = logits[:, first].max(1) - logits[:, second].max(1)
            semantic = -raw if ENCODINGS[c] == "B" else raw
            same(scores["raw_margin"][c, role_index], raw, "candidate raw margin", 1e-6)
            same(
                scores["semantic_margin"][c, role_index],
                semantic,
                "candidate semantic margin",
                1e-6,
            )
            same(
                scores["answer_token_mass"][c, role_index],
                np.exp(np.logaddexp.reduce(logits, axis=1) - partition),
                "candidate token mass",
                1e-6,
            )
            if ENCODINGS[c] == "Y":
                lse = np.logaddexp.reduce(logits[:, first], axis=1) - np.logaddexp.reduce(
                    logits[:, second], axis=1
                )
                same(scores["lse_margin"][c, role_index], lse, "candidate lse margin", 1e-6)
                scores["lse_margin"][c, role_index] = lse
    for key, value in scores.items():
        same(saved[key], value, "score array copy " + key)

    labels = np.asarray([row["Effusion"] for row in cohort["index"]])
    donor_labels = np.asarray([row["Effusion"] for row in cohort["donor"]])
    counts = np.vstack((np.ones(N, dtype=np.int64), weights(indices)))
    aurocs = np.empty((6, 2, B + 1))
    donor_own = np.empty(6)
    for c in range(6):
        for role in range(2):
            aurocs[c, role] = auc(scores["semantic_margin"][c, role], labels, counts)
        donor_own[c] = auc(scores["semantic_margin"][c, 1], donor_labels, counts[:1])[0]
    advantages = (aurocs[:, 0] - aurocs[:, 1]).T.reshape(B + 1, 2, 3)
    theta = contrasts(advantages[0])
    theta_draws = contrasts(advantages[1:])
    valid = np.isfinite(theta_draws).all(1)
    check(valid.sum() >= 9500, "common valid primary draws")
    radius = np.percentile(np.max(np.abs(theta_draws[valid] - theta), axis=1), 95, method="linear")
    same(saved["auroc"], aurocs[:, :, 0], "saved answer AUROC")
    same(saved["bootstrap_auroc"], aurocs[:, :, 1:], "saved answer AUROC draws")
    same(saved["donor_own_label_auroc"], donor_own, "saved donor own-label AUROC")
    same(saved["image_advantage"], advantages[0], "saved image advantage")
    same(saved["bootstrap_image_advantage"], advantages[1:], "saved image-advantage draws")
    same(saved["primary"], theta, "saved primary estimates")
    same(saved["bootstrap_primary"], theta_draws, "saved primary draws")
    check(np.array_equal(saved["common_valid"], valid), "saved primary valid-draw mask")
    same(saved["simultaneous_radius"], radius, "saved simultaneous radius")
    family = summary["primary_family"]
    same(family["simultaneous_radius"], radius, "simultaneous radius")
    check(
        family["common_valid_draws"] == int(valid.sum())
        and family["bootstrap_degenerate"] is bool(radius == 0),
        "primary family decision",
    )
    check([row["name"] for row in family["contrasts"]] == list(PRIMARY), "primary contrast order")
    for i, row in enumerate(family["contrasts"]):
        same(row["estimate"], theta[i], "primary estimate")
        same(
            row["simultaneous_ci95"],
            [theta[i] - radius, theta[i] + radius],
            "simultaneous interval",
        )
    check(len(summary["cells"]) == 6, "six reported answer cells")
    for c, row in enumerate(summary["cells"]):
        wording, encoding_index = c // 3, c % 3
        expected_scalars = {
            "real_auroc": aurocs[c, 0, 0],
            "donor_auroc_against_index_label": aurocs[c, 1, 0],
            "donor_auroc_against_donor_label": donor_own[c],
            "image_advantage": advantages[0, wording, encoding_index],
            "real_mean_margin": scores["semantic_margin"][c, 0].mean(),
            "donor_mean_margin": scores["semantic_margin"][c, 1].mean(),
            "real_brier": np.mean((scores["probability"][c, 0] - labels) ** 2),
            "donor_brier_against_index_label": np.mean((scores["probability"][c, 1] - labels) ** 2),
            "real_mean_answer_token_mass": scores["answer_token_mass"][c, 0].mean(),
            "donor_mean_answer_token_mass": scores["answer_token_mass"][c, 1].mean(),
        }
        check(
            row["condition"] == CONDITIONS[c] and row["prompt"] == prompts[c]["prompt"],
            "reported cell identity",
        )
        for key, value in expected_scalars.items():
            same(row[key], value, "reported cell " + key)
        same(
            row["real_auroc_ci95"],
            np.percentile(
                aurocs[c, 0, 1:][np.isfinite(aurocs[c, 0, 1:])], [2.5, 97.5], method="linear"
            ),
            "reported real interval",
        )
        same(
            row["donor_auroc_ci95"],
            np.percentile(
                aurocs[c, 1, 1:][np.isfinite(aurocs[c, 1, 1:])], [2.5, 97.5], method="linear"
            ),
            "reported donor interval",
        )
        same(
            row["image_advantage_ci95"],
            np.percentile(advantages[1:, wording, encoding_index], [2.5, 97.5], method="linear"),
            "reported advantage interval",
        )
    check(len(summary["same_logit_aggregation"]) == 2, "two same-logit aggregation contrasts")
    for report_index, c in enumerate((0, 3)):
        real = auc(scores["lse_margin"][c, 0], labels, counts)
        donor = auc(scores["lse_margin"][c, 1], labels, counts)
        lse_advantage = real - donor
        max_advantage = advantages[:, c // 3, 0]
        row = summary["same_logit_aggregation"][report_index]
        check(row["condition"] == CONDITIONS[c], "same-logit condition identity")
        same(row["real_auroc"], real[0], "same-logit real AUROC")
        same(row["donor_auroc_against_index_label"], donor[0], "same-logit donor AUROC")
        same(row["image_advantage"], lse_advantage[0], "same-logit advantage")
        same(
            row["image_advantage_ci95"],
            np.percentile(lse_advantage[1:], [2.5, 97.5], method="linear"),
            "same-logit advantage interval",
        )
        difference = lse_advantage - max_advantage
        same(
            row["image_advantage_difference_from_max_variant"],
            difference[0],
            "same-logit scoring difference",
        )
        same(
            row["image_advantage_difference_from_max_variant_ci95"],
            np.percentile(difference[1:], [2.5, 97.5], method="linear"),
            "same-logit difference interval",
        )

    raw = stored["reader_raw_activation"]
    check(
        raw.shape == (N, 1024) and raw.dtype == np.float16 and np.isfinite(raw).all(),
        "float16 reader capture",
    )
    projected = raw.astype(np.float32) @ prep["projection"].astype(np.float32)
    features = projected.copy()
    features -= prep["train_mean"]
    features /= prep["scale"]
    clinical = features @ prep["coefficients"][0]
    controls = np.stack(
        [
            features @ coefficient + intercept
            for coefficient, intercept in zip(
                prep["control_coefficients"], prep["control_intercepts"], strict=True
            )
        ]
    )
    types = [
        f"{row['view_AP']}_{row['sex_M']}_{min(max(int(row['age']) // 10, 0), 9)}"
        for row in cohort["index"]
    ]
    check(len(prepared["control_assignments"]) == 20, "twenty frozen control assignments")
    check(
        all(
            all(kind in record.get("assignment", {}) for kind in types)
            for record in prepared["control_assignments"]
        ),
        "frozen control assignment cohort coverage",
    )
    control_labels = np.asarray(
        [
            [record["assignment"][kind] for kind in types]
            for record in prepared["control_assignments"]
        ]
    )
    clinical_auc = auc(clinical, labels, counts)
    control_auc = np.stack(
        [auc(score, target, counts) for score, target in zip(controls, control_labels, strict=True)]
    )
    selectivity = clinical_auc - control_auc.mean(0)
    selectivity[~np.isfinite(control_auc).all(0)] = np.nan
    valid_reader = np.isfinite(selectivity[1:])
    interval = np.percentile(selectivity[1:][valid_reader], [2.5, 97.5], method="linear")
    report = summary["frozen_reader"]
    same(report["clinical_auroc"], clinical_auc[0], "reader AUROC")
    same(report["control_auroc"], control_auc[:, 0], "control AUROCs")
    same(report["selectivity"], selectivity[0], "reader selectivity")
    same(report["selectivity_ci95"], interval, "reader interval")
    check(
        report["valid_draws"] == int(valid_reader.sum())
        and report["replicated"] is bool(valid_reader.sum() >= 9500 and interval[0] > 0),
        "reader replication decision",
    )
    same(saved["reader_raw_activation"], raw, "saved reader activations")
    same(saved["reader_projected"], projected, "saved reader projection")
    same(saved["reader_features"], features, "saved reader features")
    same(saved["reader_clinical_score"], clinical, "saved reader clinical scores")
    same(saved["reader_control_score"], controls, "saved reader control scores")
    check(np.array_equal(saved["reader_control_labels"], control_labels), "saved control labels")
    same(saved["reader_clinical_auroc"], clinical_auc[0], "saved reader point AUROC")
    same(saved["reader_control_auroc"], control_auc[:, 0], "saved control point AUROCs")
    same(saved["bootstrap_reader_clinical_auroc"], clinical_auc[1:], "saved reader AUROC draws")
    same(saved["bootstrap_reader_control_auroc"], control_auc[:, 1:], "saved control AUROC draws")
    same(saved["bootstrap_reader_selectivity"], selectivity[1:], "saved selectivity draws")
    check(
        summary["n_image_outcomes"] == 8400
        and summary["n_text_only_outcomes"] == 6
        and summary["route"] == "mechanism_oriented_synthesis",
        "summary counts and route",
    )
    return {
        "status": "PASS",
        "run_id": run_id,
        "source_commit": commit,
        "verified_new_files": digests,
        "n_image_outcomes": 8400,
        "reader_replicated": report["replicated"],
        "simultaneous_radius": float(radius),
        "max_numerical_discrepancy": MAX_ERROR,
    }


if __name__ == "__main__":
    print(json.dumps(verify(sys.argv[1], sys.argv[2]), allow_nan=False))
