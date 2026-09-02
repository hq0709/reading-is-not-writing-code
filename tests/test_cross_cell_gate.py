from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src import cross_cell_gate as gate
from src.cross_cell_gate import (
    ALPHAS,
    CONTROL_ALPHAS,
    SOURCE_COMMIT,
    SOURCE_RUN_ID,
    SOURCE_SHA256,
    UNRELATED_CONCEPTS,
    load_source_probe_baseline,
    paired_selectivity_difference,
    summarize_intervention_rows,
    validate_activation_source,
    validate_manifest_identity,
)
from src.first_gate import capacity_direction
from src.intervene import load_registered_directions, registered_unrelated


class CrossCellGateTests(unittest.TestCase):
    def test_paired_selectivity_difference_subtracts_matched_draws(self) -> None:
        point, interval, draws = paired_selectivity_difference(
            0.3,
            np.asarray([0.2, 0.3, 0.4, 0.5]),
            0.1,
            np.asarray([0.05, 0.1, 0.2, 0.25]),
        )
        np.testing.assert_allclose([0.15, 0.2, 0.2, 0.25], draws)
        self.assertAlmostEqual(0.2, point)
        np.testing.assert_allclose(np.percentile(draws, [2.5, 97.5]), interval)

    def test_source_probe_baseline_requires_identical_test_rows(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source_probe = root / "probe.json"
            source_bootstrap = root / "probe_scores_and_bootstrap.npz"
            test_ids = np.asarray([f"r{i}" for i in range(5343)], dtype=object)
            source_probe.write_text(
                json.dumps(
                    {
                        "arch": "llava15_7b",
                        "concept": "Effusion",
                        "locus": "vis.last",
                        "projection_dim": 512,
                        "projection_seed": 0,
                        "C": 1.0,
                        "probe_seed": 0,
                        "control_seeds": list(range(20)),
                        "bootstrap_resamples": 2000,
                        "bootstrap_seed": 20260827,
                        "bootstrap_unit": "patient",
                        "n_test": 5343,
                        "n_test_patients": 1659,
                        "source_commit": SOURCE_COMMIT,
                        "bootstrap_sha256": SOURCE_SHA256[
                            "./artifacts/probe/probe_scores_and_bootstrap.npz"
                        ],
                        "real_minus_mean_control": 0.1,
                    }
                ),
                encoding="utf-8",
            )
            np.savez_compressed(
                source_bootstrap,
                test_row_id=test_ids,
                effusion_label=np.zeros(5343),
                real_scores=np.zeros(5343),
                control_scores=np.zeros((20, 5343)),
                control_labels=np.zeros((20, 5343)),
                real_bootstrap=np.zeros(2000),
                mean_control_bootstrap=np.zeros(2000),
                selectivity_bootstrap=np.linspace(0.0, 0.2, 2000),
            )
            reuse = {
                "source_probe": str(source_probe),
                "source_probe_bootstrap": str(source_bootstrap),
            }

            point, draws = load_source_probe_baseline(reuse, test_ids, 2000)
            self.assertEqual(0.1, point)
            self.assertEqual((2000,), draws.shape)

            test_ids[0] = "different"
            with self.assertRaisesRegex(ValueError, "identical test row IDs"):
                load_source_probe_baseline(reuse, test_ids, 2000)

    def test_probe_rejects_manifest_content_drift(self) -> None:
        with TemporaryDirectory() as temp:
            manifest = Path(temp) / "manifest.csv"
            manifest.write_text("row_id\nr1\n", encoding="utf-8")
            with patch.object(gate, "MANIFEST_SHA256", gate.sha256(manifest)):
                validate_manifest_identity(manifest)
                manifest.write_text("row_id\nr2\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "manifest hash"):
                    validate_manifest_identity(manifest)

    def test_runner_reuses_accepted_activations_without_extraction(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner = (root / "scripts/server/run_cross_cell_gate.sh").read_text(encoding="utf-8")
        for token in (
            SOURCE_RUN_ID,
            "--concept Edema",
            "--loci vis.last",
            "--n-boot 2000",
            "--alphas=-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1",
            "--control-alphas=-1,-0.5,-0.25,0.25,0.5,1",
            "--n-random 20",
            "--n-eval 200",
            "--eval-split test",
            "Is there pulmonary edema in this chest radiograph? Answer yes or no.",
        ):
            self.assertIn(token, runner)
        self.assertNotIn("src/extract.py", runner)

    def test_registered_edema_bundle_has_six_unique_unrelated_directions(self) -> None:
        self.assertEqual(
            ("Effusion", "Atelectasis", "Pneumothorax", "Cardiomegaly", "Mass", "Nodule"),
            registered_unrelated("Edema"),
        )
        rng = np.random.default_rng(4)
        projection = rng.standard_normal((3, 512)).astype(np.float32)
        scale = np.linspace(0.5, 2.0, 512)
        coefficients = rng.standard_normal((7, 512))
        vectors = np.stack(
            [capacity_direction(projection, scale, coefficient) for coefficient in coefficients]
        )
        names = np.asarray(["Edema", *UNRELATED_CONCEPTS])
        with TemporaryDirectory() as temp:
            path = Path(temp) / "directions.npz"
            np.savez_compressed(
                path,
                names=names,
                vectors=vectors,
                projection=projection,
                scale=scale,
                coefficients=coefficients,
                locus=np.asarray("vis.last"),
                raw_dim=np.asarray(3),
                projection_dim=np.asarray(512),
                projection_seed=np.asarray(0),
                C=np.asarray(1.0),
                probe_seed=np.asarray(0),
            )
            loaded, _ = load_registered_directions(path, 3, "Edema")
        self.assertEqual(set(names), set(loaded))

    def test_source_receipt_allows_pre_fusion_prompt_reuse(self) -> None:
        with TemporaryDirectory() as temp:
            run = Path(temp) / SOURCE_RUN_ID
            acts = run / "artifacts/activations"
            acts.mkdir(parents=True)
            source_probe = run / "artifacts/probe"
            source_probe.mkdir()
            (run / "metadata.env").write_text(
                f"SOURCE_COMMIT={SOURCE_COMMIT}\nCOMMAND_STATUS=0\nDISPATCHER_STATUS=0\n",
                encoding="utf-8",
            )
            (run / "command_exit_status").write_text("0\n", encoding="utf-8")
            (run / "exit_status").write_text("0\n", encoding="utf-8")
            (run / "SHA256SUMS").write_text(
                "\n".join(f"{digest}  {relative}" for relative, digest in SOURCE_SHA256.items())
                + "\n",
                encoding="utf-8",
            )
            (acts / "meta.json").write_text(
                json.dumps(
                    {
                        "arch": "llava15_7b",
                        "model": "llava-hf/llava-1.5-7b-hf",
                        "model_source": str(Path(temp) / "model-snapshot"),
                        "prompt": "Is there a pleural effusion in this chest radiograph? Answer yes or no.",
                        "n_rows": 26_229,
                        "git_sha": SOURCE_COMMIT,
                        "loci": ["vis.last"],
                    }
                ),
                encoding="utf-8",
            )
            (acts / "shard0.json").write_text(
                json.dumps(
                    {
                        "n_in": 26_229,
                        "n_out": 26_229,
                        "failed": [],
                    }
                ),
                encoding="utf-8",
            )
            (acts / "shard0.npz").write_bytes(b"accepted immutable shard")
            (run / "artifacts/input-verification.json").write_text("{}", encoding="utf-8")
            (run / "artifacts/hook-verification.json").write_text("{}", encoding="utf-8")
            summary = run / "artifacts/intervention-summary.json"
            summary.write_text(
                json.dumps({"eval_row_ids": [f"r{i}" for i in range(200)]}), encoding="utf-8"
            )
            (source_probe / "probe.json").write_text("{}", encoding="utf-8")
            (source_probe / "probe_scores_and_bootstrap.npz").write_bytes(b"accepted probe")
            out = Path(temp) / "reuse.json"

            validate_activation_source(run, out)
            receipt = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(200, len(receipt["eval_row_ids"]))
            self.assertEqual("vis.last", receipt["locus"])

            meta = json.loads((acts / "meta.json").read_text(encoding="utf-8"))
            meta["loci"] = ["connector"]
            (acts / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "activation identity"):
                validate_activation_source(run, out)

    def test_edema_summary_requires_complete_six_control_grid(self) -> None:
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

        for alpha in ALPHAS:
            add("concept", alpha, 0.5 + 0.1 * alpha)
        for direction in [
            *(f"random{i}" for i in range(20)),
            "sham",
            *(f"unrelated_{name}" for name in UNRELATED_CONCEPTS),
        ]:
            for alpha in CONTROL_ALPHAS:
                add(direction, alpha, 0.5 + 0.01 * alpha)

        summary = summarize_intervention_rows(rows)
        self.assertEqual("Edema", summary["concept"])
        self.assertEqual(171, len(rows))
        self.assertTrue(summary["primary"]["selective"])

        rows = [row for row in rows if row["direction"] != "unrelated_Effusion"]
        with self.assertRaisesRegex(ValueError, "unrelated directions"):
            summarize_intervention_rows(rows)


if __name__ == "__main__":
    unittest.main()
