from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.extract import select_loci
from src.first_gate import (
    EXPECTED_LOCUS,
    MODEL_ID,
    MODEL_REVISION,
    REGISTERED_PROMPT,
    capacity_direction,
    hook_implementation_hashes,
    summarize_intervention_rows,
    validate_hook_receipt,
)
from src.intervene import Steerer, load_registered_directions
from src.loci import Locus, loci_for
from src.registry import REGISTRY


class FirstGateContractTests(unittest.TestCase):
    def test_steerer_precedes_an_existing_hidden_state_recorder(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = torch.nn.Identity()

        model = TinyModel()
        recorded = []
        handle = model.layer.register_forward_hook(
            lambda _module, _inputs, output: recorded.append(output.detach().clone())
        )
        steer = Steerer(model, "layer", positions="all", mode="relnorm", global_scale=1.0)
        steer.vec = np.array([1.0, 0.0], dtype=np.float32)
        steer.alpha = 1.0
        with steer:
            output = model.layer(torch.ones(1, 1, 2))
        handle.remove()
        self.assertTrue(torch.equal(torch.tensor([[[2.0, 1.0]]]), output))
        self.assertTrue(torch.equal(output, recorded[0]))

    def test_llava_vislast_resolves_to_loaded_transformers_namespace(self) -> None:
        locus = next(locus for locus in loci_for(REGISTRY["llava15_7b"]) if locus.name == "vis.last")
        self.assertEqual("model.vision_tower.encoder.layers.22", locus.module)

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
            '--model-path "$snapshot"',
            '--directions "$probe/directions.npz"',
            "--input-verification",
            "--extract-meta",
            "--prompt 'Is there a pleural effusion in this chest radiograph? Answer yes or no.'",
        ):
            self.assertIn(token, runner)
        self.assertIn('run_dir=${RUN_DIR:?', runner)
        self.assertIn("grep -Fx 'GPU_COUNT=1'", runner)
        self.assertNotIn("hook|full RUN_DIR GPU", runner)

    def test_capacity_direction_maps_the_registered_probe_back_to_raw_space(self) -> None:
        projection = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scale = np.array([2.0, 4.0])
        coefficient = np.array([4.0, -2.0])
        expected = projection @ (coefficient / scale)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(expected, capacity_direction(projection, scale, coefficient))

    def test_direction_bundle_reconstructs_registered_vectors(self) -> None:
        rng = np.random.default_rng(0)
        projection = rng.standard_normal((3, 512)).astype(np.float32)
        scale = np.linspace(0.5, 2.0, 512)
        coefficients = rng.standard_normal((6, 512))
        vectors = np.stack([
            capacity_direction(projection, scale, coefficient) for coefficient in coefficients
        ])
        names = np.asarray(["Effusion", "Atelectasis", "Pneumothorax", "Cardiomegaly", "Mass", "Nodule"])
        with TemporaryDirectory() as temp:
            path = Path(temp) / "directions.npz"
            np.savez_compressed(
                path, names=names, vectors=vectors, projection=projection, scale=scale,
                coefficients=coefficients, locus=np.asarray("vis.last"), raw_dim=np.asarray(3),
                projection_dim=np.asarray(512), projection_seed=np.asarray(0), C=np.asarray(1.0),
                probe_seed=np.asarray(0),
            )
            loaded, digest = load_registered_directions(path, 3, "Effusion")
        self.assertEqual(set(names), set(loaded))
        self.assertEqual(64, len(digest))

    def test_direction_bundle_rejects_vectors_unrelated_to_readout(self) -> None:
        rng = np.random.default_rng(1)
        projection = rng.standard_normal((3, 512)).astype(np.float32)
        scale = np.ones(512)
        coefficients = rng.standard_normal((6, 512))
        vectors = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (6, 1))
        names = np.asarray(["Effusion", "Atelectasis", "Pneumothorax", "Cardiomegaly", "Mass", "Nodule"])
        with TemporaryDirectory() as temp:
            path = Path(temp) / "directions.npz"
            np.savez_compressed(
                path, names=names, vectors=vectors, projection=projection, scale=scale,
                coefficients=coefficients, locus=np.asarray("vis.last"), raw_dim=np.asarray(3),
                projection_dim=np.asarray(512), projection_seed=np.asarray(0), C=np.asarray(1.0),
                probe_seed=np.asarray(0),
            )
            with self.assertRaisesRegex(ValueError, "do not match"):
                load_registered_directions(path, 3, "Effusion")

    def test_gate_reuses_asset_receipts_instead_of_rehashing_model_shards(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "src/first_gate.py").read_text(encoding="utf-8")
        self.assertNotIn("actual = sha256(path)", source)
        self.assertIn('"verification_mode": "accepted-asset-receipt"', source)
        self.assertIn('"alpha_zero_bitwise_noop": True', source)

    def test_hook_identity_tracks_behavior_not_whole_validator_files(self) -> None:
        identities = hook_implementation_hashes()
        self.assertIn("verify_hook", identities)
        self.assertIn("Steerer", identities)
        self.assertIn("synthetic_cxr", identities)
        self.assertIn("llava15_7b_registry", identities)
        self.assertNotIn("first_gate.py", identities)
        self.assertTrue(all(len(digest) == 64 for digest in identities.values()))

    def test_hook_receipt_rejects_asset_drift(self) -> None:
        inputs = {
            "manifest_sha256": "manifest",
            "dataset_receipt_sha256": "dataset",
            "model_receipt_sha256": "model",
        }
        hook = {
            "arch": "llava15_7b",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "locus": "vis.last",
            "module": EXPECTED_LOCUS,
            "alpha_zero_bitwise_noop": True,
            "forward_path_proved": True,
            "prompt": REGISTERED_PROMPT,
            "hook_implementation_hashes": hook_implementation_hashes(),
            **inputs,
        }
        with TemporaryDirectory() as temp:
            input_path = Path(temp) / "input.json"
            hook_path = Path(temp) / "hook.json"
            input_path.write_text(json.dumps(inputs), encoding="utf-8")
            hook_path.write_text(json.dumps(hook), encoding="utf-8")
            validate_hook_receipt(hook_path, input_path)
            hook["model_receipt_sha256"] = "different"
            hook_path.write_text(json.dumps(hook), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "asset mismatch"):
                validate_hook_receipt(hook_path, input_path)

    def test_dispatch_receipt_uses_the_declared_gpu_allocation(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dispatcher = (root / "scripts/server/dispatch_run.sh").read_text(encoding="utf-8")
        server = (root / "scripts/server_run.sh").read_text(encoding="utf-8")
        self.assertIn("declared_gpu_count=${4:?", dispatcher)
        self.assertIn("GPU_INVENTORY_COUNT", dispatcher)
        self.assertIn('"$declared_gpu_count"', server)

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
            for concept in ("Atelectasis", "Pneumothorax", "Cardiomegaly", "Mass", "Nodule"):
                add(f"unrelated_{concept}", alpha, 0.5 + 0.03 * alpha)

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

    def test_intervention_summary_rejects_an_incomplete_unrelated_control_set(self) -> None:
        rows: list[dict[str, str]] = []
        for alpha in (0.0, 1.0):
            rows.append({"locus": "vis.last", "direction": "concept", "alpha": str(alpha),
                         "mean_p_yes": "0.5", "n": "200"})
        for index in range(20):
            rows.append({"locus": "vis.last", "direction": f"random{index}", "alpha": "1.0",
                         "mean_p_yes": "0.5", "n": "200"})
        rows.extend([
            {"locus": "vis.last", "direction": "sham", "alpha": "1.0",
             "mean_p_yes": "0.5", "n": "200"},
            {"locus": "vis.last", "direction": "unrelated_Atelectasis", "alpha": "1.0",
             "mean_p_yes": "0.5", "n": "200"},
        ])
        with self.assertRaisesRegex(ValueError, "unrelated directions"):
            summarize_intervention_rows(
                rows, alphas=[0.0, 1.0], control_alphas=[1.0], n_random=20, n_eval=200
            )


if __name__ == "__main__":
    unittest.main()
