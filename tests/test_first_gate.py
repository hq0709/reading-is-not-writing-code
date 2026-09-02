from __future__ import annotations

import unittest
from pathlib import Path

from src.extract import select_loci
from src.first_gate import summarize_intervention_rows
from src.loci import Locus


class FirstGateContractTests(unittest.TestCase):
    def test_immutable_runner_pins_the_registered_protocol(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner = (root / "scripts/server/run_first_gate.sh").read_text(encoding="utf-8")
        for token in (
            "--arch llava15_7b",
            "--concept Effusion",
            "--loci vis.last",
            "--n-boot 2000",
            "--alphas=-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1",
            "--alpha-mode reltoken",
            "--control-alphas=-1,-0.5,-0.25,0.25,0.5,1",
            "--n-random 20",
            "--n-eval 200",
            "--eval-split test",
        ):
            self.assertIn(token, runner)

    def test_extract_selects_only_the_registered_locus(self) -> None:
        loci = [
            Locus("vis.block0", "vision.0", "vision", 0, "all"),
            Locus("vis.last", "vision.22", "vision", 1, "all"),
            Locus("connector", "projector", "connector", 2, "all"),
        ]
        selected = select_loci(loci, "vis.last")
        self.assertEqual(["vis.last"], [locus.name for locus in selected])
        with self.assertRaisesRegex(ValueError, "unknown loci"):
            select_loci(loci, "vis.missing")

    def test_intervention_summary_uses_only_controlled_alphas(self) -> None:
        alphas = [-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0]
        control_alphas = [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]
        rows: list[dict[str, str]] = []

        def add(direction: str, alpha: float, probability: float) -> None:
            rows.append(
                {
                    "locus": "vis.last",
                    "direction": direction,
                    "alpha": str(alpha),
                    "mean_p_yes": str(probability),
                    "n": "200",
                }
            )

        for alpha in alphas:
            # The uncontrolled +0.1 dose is the largest change, so it must not become primary.
            p = 0.9 if alpha == 0.1 else 0.5 + 0.2 * alpha
            add("concept", alpha, p)
        for index in range(20):
            for alpha in control_alphas:
                add(f"random{index}", alpha, 0.5 + 0.01 * alpha)
        for alpha in control_alphas:
            add("sham", alpha, 0.5 + 0.02 * alpha)
            add("unrelated_Atelectasis", alpha, 0.5 + 0.03 * alpha)

        summary = summarize_intervention_rows(
            rows,
            alphas=alphas,
            control_alphas=control_alphas,
            n_random=20,
            n_eval=200,
        )
        self.assertIn(summary["primary"]["alpha"], control_alphas)
        self.assertNotEqual(0.1, summary["primary"]["alpha"])
        self.assertTrue(summary["primary"]["selective"])
        self.assertEqual([-0.5, 0.5], summary["monotonicity"]["range"])

    def test_intervention_summary_rejects_missing_random_direction(self) -> None:
        rows = [
            {"locus": "vis.last", "direction": "concept", "alpha": "0", "mean_p_yes": "0.5", "n": "200"},
            {"locus": "vis.last", "direction": "concept", "alpha": "1", "mean_p_yes": "0.6", "n": "200"},
            {"locus": "vis.last", "direction": "sham", "alpha": "1", "mean_p_yes": "0.5", "n": "200"},
        ]
        with self.assertRaisesRegex(ValueError, "random directions"):
            summarize_intervention_rows(
                rows, alphas=[0.0, 1.0], control_alphas=[1.0], n_random=20, n_eval=200
            )


if __name__ == "__main__":
    unittest.main()
