#!/usr/bin/env python3
"""Extract the evidence-locked paper data from the accepted run receipts."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("CONCEPT_FLOW_DATA_ROOT", "/home/qingchan/data/concept-flow"))
RUNS = {
    "llava_effusion": "20260902T191411Z-ffd523c464c8-f99e2f39",
    "llava_edema": "20260902T221534Z-110b84618d1b-edema",
    "qwen_effusion": "20260903T000321Z-caaae3ef346d-qwen-full",
    "qwen_specificity": "20260903T103508Z-456c81bad460-7a3edc4b",
    "qwen_ownership": "20260904T125710Z-a3bd883540eb-causal-ownership",
    "qwen_closure": "20260904T162317Z-8a55a4c2f6b6-consolidation-closure",
}
EXPECTED = {
    "llava_effusion": {
        "auroc": 0.7788163441124888,
        "auroc_ci95": [0.7537593969935978, 0.803081179537482],
        "control_mean": 0.6648838502302586,
        "selectivity": 0.11393249388223015,
        "selectivity_ci95": [0.08754204794557874, 0.14039523338826207],
        "primary_effect": 0.1469851206243038,
        "random_p95": 0.12022438084706669,
        "maximum_sham": 0.15463428646326066,
        "spearman": 1.0,
    },
    "llava_edema": {
        "auroc": 0.8009024661494056,
        "auroc_ci95": [0.7648211929185149, 0.8329276519760651],
        "control_mean": 0.6648838502302586,
        "selectivity": 0.136018615919147,
        "selectivity_ci95": [0.09935901397117824, 0.1693054728441684],
        "primary_effect": 0.06593271091580388,
        "random_p95": 0.14702408102527262,
        "maximum_sham": 0.09133242860436441,
        "spearman": -0.42857142857142866,
        "paired": 0.022086122036916844,
        "paired_ci95": [-0.017479364808838334, 0.05901509222566426],
    },
    "qwen_effusion": {
        "auroc": 0.7742131011353497,
        "auroc_ci95": [0.7488153720964997, 0.79929749997042],
        "control_mean": 0.6576339607365205,
        "selectivity": 0.11657914039882922,
        "selectivity_ci95": [0.08974446529282622, 0.14283241566000443],
        "primary_effect": 0.23430159127572553,
        "random_p95": 0.09631123471958565,
        "maximum_sham": 0.08516594238812104,
        "spearman": 0.9642857142857145,
        "paired": 0.0026466465165990716,
        "paired_ci95": [-0.015509162117536269, 0.021500020623000344],
        "nodule": 0.286263411717955,
    },
    "specificity": {
        "effect": 0.1945347837949521,
        "random_p95": 0.08776638119568816,
        "sham": 0.0156513355919742,
        "nodule": 0.2518629158087424,
        "margin": -0.057328132013790334,
        "ci95": [-0.0693829501046566, -0.04500512929691468],
        "one_sided": -0.06779789115232414,
    },
    "ownership": {
        "n_owned": 0,
        "n_off_diagonal_winners": 6,
        "candidate_effect": 0.2909364313397673,
        "candidate_effect_lower": 0.28083605156555386,
        "candidate_dominance_lower": 0.13321401700916094,
        "global_random_max": 0.34022399436600975,
    },
    "closure": {
        "probe_auroc": 0.7218194409485241,
        "probe_selectivity": 0.06418548021200365,
        "probe_selectivity_ci95": [0.030158249563824062, 0.09758492650943741],
        "displacement": 0.10906783267855644,
        "displacement_lower": -0.036407997652888296,
        "concept_gain": 0.0022028424963355065,
        "random_max": 0.0024608347192406656,
        "clinical_margin": 0.0019651266001164914,
        "clinical_lower": -0.0015283454516902566,
    },
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def exact(name: str, observed: float, expected: float) -> None:
    if observed != expected:
        raise RuntimeError(f"{name} drifted: {observed!r} != {expected!r}")


def cell(key: str, model: str, concept: str) -> dict:
    run_id = RUNS[key]
    artifact_root = DATA_ROOT / "runs" / run_id / "artifacts"
    probe_path = artifact_root / "probe" / "probe.json"
    summary_path = artifact_root / "intervention-summary.json"
    csv_path = artifact_root / "intervention" / "intervention.csv"
    probe = load_json(probe_path)
    summary = load_json(summary_path)
    exact(f"{key}.auroc", probe["real_auroc"], EXPECTED[key]["auroc"])
    exact(
        f"{key}.selectivity",
        probe["real_minus_mean_control"],
        EXPECTED[key]["selectivity"],
    )
    for index, value in enumerate(probe["real_auroc_ci95"]):
        exact(f"{key}.auroc_ci95[{index}]", value, EXPECTED[key]["auroc_ci95"][index])
    exact(f"{key}.control_mean", probe["control_auroc_mean"], EXPECTED[key]["control_mean"])
    for index, value in enumerate(probe["selectivity_ci95"]):
        exact(
            f"{key}.selectivity_ci95[{index}]",
            value,
            EXPECTED[key]["selectivity_ci95"][index],
        )
    exact(
        f"{key}.primary_effect",
        summary["primary"]["concept_consistent_change"],
        EXPECTED[key]["primary_effect"],
    )
    exact(f"{key}.random_p95", summary["primary"]["random_p95"], EXPECTED[key]["random_p95"])
    exact(
        f"{key}.maximum_sham",
        summary["maximum_absolute_sham_effect"],
        EXPECTED[key]["maximum_sham"],
    )
    exact(f"{key}.spearman", summary["monotonicity"]["spearman_rho"], EXPECTED[key]["spearman"])
    if "paired" in EXPECTED[key]:
        paired_key = (
            "edema_minus_effusion_selectivity" if key == "llava_edema" else "qwen_minus_llava_selectivity"
        )
        exact(f"{key}.paired", probe[paired_key], EXPECTED[key]["paired"])
        for index, value in enumerate(probe[f"{paired_key}_ci95"]):
            exact(f"{key}.paired_ci95[{index}]", value, EXPECTED[key]["paired_ci95"][index])
    if key == "qwen_effusion":
        exact(
            "qwen_effusion.nodule",
            summary["primary"]["unrelated_effects"]["unrelated_Nodule"],
            EXPECTED[key]["nodule"],
        )

    dose = []
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["direction"] == "concept":
                mean = float(row["mean_p_yes"])
                dose.append(
                    {
                        "alpha": float(row["alpha"]),
                        "mean_p_yes": mean,
                        "raw_change": mean - summary["baseline_mean_p_yes"],
                    }
                )
    controls = {
        str(item["alpha"]): {
            "random_p05": item["random_p05"],
            "random_p95": item["random_p95"],
            "sham_effect": item["sham_effect"],
        }
        for item in summary["candidates"]
    }
    return {
        "key": key,
        "run_id": run_id,
        "model": model,
        "concept": concept,
        "locus": probe["locus"],
        "module": probe.get("module", "model.vision_tower.encoder.layers.22"),
        "n_train": probe["n_train"],
        "n_test": probe["n_test"],
        "n_test_patients": probe["n_test_patients"],
        "auroc": probe["real_auroc"],
        "auroc_ci95": probe["real_auroc_ci95"],
        "control_mean": probe["control_auroc_mean"],
        "control_spread": probe["control_auroc_spread"],
        "selectivity": probe["real_minus_mean_control"],
        "selectivity_ci95": probe["selectivity_ci95"],
        "paired_difference": (
            probe.get("edema_minus_effusion_selectivity")
            if key == "llava_edema"
            else probe.get("qwen_minus_llava_selectivity")
        ),
        "paired_difference_ci95": (
            probe.get("edema_minus_effusion_selectivity_ci95")
            if key == "llava_edema"
            else probe.get("qwen_minus_llava_selectivity_ci95")
        ),
        "baseline_mean_p_yes": summary["baseline_mean_p_yes"],
        "dose": sorted(dose, key=lambda item: item["alpha"]),
        "controls": controls,
        "primary": summary["primary"],
        "maximum_absolute_sham_effect": summary["maximum_absolute_sham_effect"],
        "selective_cell": summary["selective_cell"],
        "spearman_rho": summary["monotonicity"]["spearman_rho"],
        "sources": {
            "probe": str(probe_path),
            "intervention_summary": str(summary_path),
            "intervention_csv": str(csv_path),
        },
    }


def main() -> None:
    cells = [
        cell("llava_effusion", "LLaVA-1.5-7B", "Effusion"),
        cell("llava_edema", "LLaVA-1.5-7B", "Edema"),
        cell("qwen_effusion", "Qwen2.5-VL-7B", "Effusion"),
    ]
    supplement_run = RUNS["qwen_specificity"]
    supplement_path = (
        DATA_ROOT
        / "runs"
        / supplement_run
        / "artifacts"
        / "direction-specificity-summary.json"
    )
    supplement = load_json(supplement_path)
    specificity_expected = EXPECTED["specificity"]
    exact("specificity.effect", supplement["direction_effects"]["concept"], specificity_expected["effect"])
    exact("specificity.random_p95", supplement["random_effect_p95"], specificity_expected["random_p95"])
    exact("specificity.sham", supplement["absolute_sham_effect"], specificity_expected["sham"])
    exact(
        "specificity.nodule",
        supplement["direction_effects"]["unrelated_Nodule"],
        specificity_expected["nodule"],
    )
    exact("specificity.margin", supplement["primary"]["margin"], specificity_expected["margin"])
    for index, value in enumerate(supplement["primary"]["ci95"]):
        exact(f"specificity.ci95[{index}]", value, specificity_expected["ci95"][index])
    exact(
        "specificity.one_sided",
        supplement["primary"]["one_sided_lower_95"],
        specificity_expected["one_sided"],
    )
    ownership_run = RUNS["qwen_ownership"]
    ownership_path = (
        DATA_ROOT / "runs" / ownership_run / "artifacts" / "causal-ownership-summary.json"
    )
    ownership = load_json(ownership_path)
    ownership_expected = EXPECTED["ownership"]
    exact("ownership.n_owned", ownership["n_owned_concepts"], ownership_expected["n_owned"])
    exact(
        "ownership.n_off_diagonal_winners",
        ownership["n_off_diagonal_winners"],
        ownership_expected["n_off_diagonal_winners"],
    )
    shared_alias = ownership["shared_alias"]
    exact("ownership.candidate_effect", shared_alias["off_diagonal_effect"], ownership_expected["candidate_effect"])
    exact(
        "ownership.candidate_effect_lower",
        shared_alias["off_diagonal_effect_simultaneous_lower_95"],
        ownership_expected["candidate_effect_lower"],
    )
    exact(
        "ownership.candidate_dominance_lower",
        shared_alias["relative_dominance_simultaneous_lower_95"],
        ownership_expected["candidate_dominance_lower"],
    )
    exact(
        "ownership.global_random_max",
        shared_alias["global_random_effect_max"],
        ownership_expected["global_random_max"],
    )

    closure_run = RUNS["qwen_closure"]
    closure_path = DATA_ROOT / "runs" / closure_run / "artifacts" / "input-closure-summary.json"
    closure = load_json(closure_path)
    closure_expected = EXPECTED["closure"]
    exact("closure.probe_auroc", closure["probe"]["real_auroc"], closure_expected["probe_auroc"])
    exact(
        "closure.probe_selectivity",
        closure["probe"]["selectivity"],
        closure_expected["probe_selectivity"],
    )
    for index, value in enumerate(closure["probe"]["selectivity_ci95"]):
        exact(
            f"closure.probe_selectivity_ci95[{index}]",
            value,
            closure_expected["probe_selectivity_ci95"][index],
        )
    exact("closure.displacement", closure["input_displacement"]["estimate"], closure_expected["displacement"])
    exact(
        "closure.displacement_lower",
        closure["input_displacement"]["one_sided_lower_95"],
        closure_expected["displacement_lower"],
    )
    exact("closure.concept_gain", closure["concept_closure_gain"], closure_expected["concept_gain"])
    exact("closure.random_max", closure["random_closure_gain_max"], closure_expected["random_max"])
    exact(
        "closure.clinical_margin",
        closure["clinical_familywise_margin"]["estimate"],
        closure_expected["clinical_margin"],
    )
    exact(
        "closure.clinical_lower",
        closure["clinical_familywise_margin"]["one_sided_lower_95"],
        closure_expected["clinical_lower"],
    )
    output = {
        "schema": "concept-flow-paper-evidence-v2",
        "cells": cells,
        "specificity": {
            "run_id": supplement_run,
            "alpha": supplement["alpha"],
            "n_eval": supplement["n_eval"],
            "n_eval_patients": supplement["n_eval_patients"],
            "bootstrap_resamples": supplement["bootstrap_resamples"],
            "direction_effects": supplement["direction_effects"],
            "random_effect_p95": supplement["random_effect_p95"],
            "absolute_sham_effect": supplement["absolute_sham_effect"],
            "primary": supplement["primary"],
            "direction_specific": supplement["direction_specific"],
            "source": str(supplement_path),
        },
        "causal_ownership": {
            "run_id": ownership_run,
            "n_eval_patients": ownership["n_eval_patients"],
            "n_random": ownership["n_random"],
            "bootstrap_resamples": ownership["bootstrap_resamples"],
            "n_owned_concepts": ownership["n_owned_concepts"],
            "n_off_diagonal_winners": ownership["n_off_diagonal_winners"],
            "columns": ownership["columns"],
            "shared_alias": shared_alias,
            "source": str(ownership_path),
        },
        "input_closure": {
            "run_id": closure_run,
            "n_pairs": closure["n_pairs"],
            "n_random": closure["n_random"],
            "bootstrap_resamples": closure["bootstrap_resamples"],
            "probe": closure["probe"],
            "input_displacement": closure["input_displacement"],
            "concept_closure_gain": closure["concept_closure_gain"],
            "random_closure_gain_max": closure["random_closure_gain_max"],
            "exact_random_rank_p": closure["exact_random_rank_p"],
            "sham_closure_gain": closure["sham_closure_gain"],
            "maximum_unrelated": closure["maximum_unrelated"],
            "clinical_familywise_margin": closure["clinical_familywise_margin"],
            "alpha": closure["alpha"],
            "input_closure": closure["input_closure"],
            "source": str(closure_path),
        },
    }
    destination = PAPER / "data" / "accepted_results.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
