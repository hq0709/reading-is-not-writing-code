#!/usr/bin/env python3
"""Extract the evidence-locked paper data from the four accepted run receipts."""

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
}
EXPECTED = {
    "llava_effusion": {"auroc": 0.7788163441124888, "selectivity": 0.11393249388223015},
    "llava_edema": {"auroc": 0.8009024661494056, "selectivity": 0.136018615919147},
    "qwen_effusion": {"auroc": 0.7742131011353497, "selectivity": 0.11657914039882922},
    "specificity_margin": -0.057328132013790334,
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
    exact("specificity.margin", supplement["primary"]["margin"], EXPECTED["specificity_margin"])
    output = {
        "schema": "concept-flow-paper-evidence-v1",
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
    }
    destination = PAPER / "data" / "accepted_results.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
