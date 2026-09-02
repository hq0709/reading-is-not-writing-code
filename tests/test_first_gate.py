from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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
    summarize_intervention,
    summarize_intervention_rows,
    validate_hook_receipt,
    verify_inputs,
)
from src.gpu_env import bind_gpu
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
        self.assertIn('${RUN_DIR:?first gate refused: trusted RUN_DIR is missing}', runner)
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
        self.assertIn("bind_gpu", identities)
        self.assertIn("Steerer", identities)
        self.assertIn("Locus", identities)
        self.assertIn("synthetic_cxr", identities)
        self.assertIn("llava15_7b_registry", identities)
        self.assertNotIn("first_gate.py", identities)
        self.assertTrue(all(len(digest) == 64 for digest in identities.values()))

    def test_verify_inputs_accepts_receipts_and_rejects_size_drift(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "image.png"
            image.write_bytes(b"image")
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                stream.write("row_id,patient_id,split,image_path\n")
                for index in range(26_229):
                    stream.write(f"r{index},p{index},{('train', 'val', 'test')[index % 3]},{image}\n")
            import src.first_gate as gate
            manifest_hash = gate.sha256(manifest)
            dataset_receipt = root / "dataset.json"
            dataset_receipt.write_text(json.dumps({
                "registered_manifest_sha256": manifest_hash,
                "registered_manifest_rows": 26_229,
            }), encoding="utf-8")
            model_root = root / "model"
            snapshot = model_root / "snapshot"
            snapshot.mkdir(parents=True)
            verified_files = {}
            for name, digest in gate.MODEL_HASHES.items():
                (snapshot / name).write_bytes(b"x")
                verified_files[name] = {"bytes": 1, "sha256": digest}
            (snapshot / "config.json").write_text(
                json.dumps({"vision_feature_layer": -2}), encoding="utf-8"
            )
            receipt = model_root / "asset-receipt.json"
            receipt.write_text(json.dumps({
                "repo_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "snapshot_relative_path": "snapshot",
                "verified_files": verified_files,
            }), encoding="utf-8")
            out = root / "verified.json"
            with patch.object(gate, "MANIFEST_SHA256", manifest_hash):
                verify_inputs(manifest, dataset_receipt, model_root, out)
                self.assertEqual(26_229, json.loads(out.read_text(encoding="utf-8"))["manifest_rows"])
                verified_files[next(iter(verified_files))]["bytes"] = 2
                receipt.write_text(json.dumps({
                    "repo_id": MODEL_ID,
                    "revision": MODEL_REVISION,
                    "snapshot_relative_path": "snapshot",
                    "verified_files": verified_files,
                }), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "file size drift"):
                    verify_inputs(manifest, dataset_receipt, model_root, out)

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
        self.assertIn("gpu_ids=${4:?", dispatcher)
        self.assertIn('export CUDA_VISIBLE_DEVICES="$cuda_visible_devices"', dispatcher)
        self.assertIn("GPU_IDS", dispatcher)
        self.assertIn("GPU_INVENTORY_COUNT", dispatcher)
        self.assertIn('"$gpu_ids"', server)

    def test_python_entrypoints_use_the_allocation_guard(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for path in (root / "src").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            if "CUDA_VISIBLE_DEVICES" in source and path.name != "gpu_env.py":
                self.assertIn("bind_gpu", source, path)

    def test_gpu_guard_rejects_ids_outside_dispatcher_allocation(self) -> None:
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3"}, clear=False):
            bind_gpu(3)
            self.assertEqual("3", os.environ["CUDA_VISIBLE_DEVICES"])
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "outside dispatcher allocation"):
                bind_gpu(0)

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

    def test_complete_evidence_chain_is_bound_and_rejects_prompt_drift(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source_commit = "a" * 40
            inputs = {
                "manifest_sha256": "manifest",
                "dataset_receipt_sha256": "dataset",
                "model_receipt_sha256": "model",
                "snapshot": os.fspath(root / "snapshot"),
                "source_commit": source_commit,
            }
            hook = {
                "arch": "llava15_7b", "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "locus": "vis.last", "module": EXPECTED_LOCUS,
                "alpha_zero_bitwise_noop": True, "forward_path_proved": True,
                "prompt": REGISTERED_PROMPT,
                "hook_implementation_hashes": hook_implementation_hashes(),
                **{key: inputs[key] for key in (
                    "manifest_sha256", "dataset_receipt_sha256", "model_receipt_sha256"
                )},
            }
            directions = root / "directions.npz"
            directions.write_bytes(b"registered directions")
            import src.first_gate as gate
            directions_hash = gate.sha256(directions)
            extract = {
                "arch": "llava15_7b", "model_source": inputs["snapshot"],
                "model_local_only": True, "prompt": REGISTERED_PROMPT,
                "n_rows": 26_229, "loci": ["vis.last"], "gpus": [0],
                "git_sha": source_commit,
            }
            probe = {
                "arch": "llava15_7b", "concept": "Effusion", "locus": "vis.last",
                "projection_dim": 512, "projection_seed": 0, "C": 1.0, "probe_seed": 0,
                "control_seeds": list(range(20)), "bootstrap_resamples": 2000,
                "bootstrap_seed": 20260827, "bootstrap_unit": "patient",
                "direction_names": ["Effusion", "Atelectasis", "Pneumothorax", "Cardiomegaly", "Mass", "Nodule"],
                "n_test": 200, "n_test_patients": 200, "test_positive": 100,
                "directions_sha256": directions_hash, "source_commit": source_commit,
            }
            bootstrap = root / "probe_scores_and_bootstrap.npz"
            np.savez_compressed(
                bootstrap,
                test_row_id=np.asarray([f"r{i}" for i in range(200)], dtype=object),
                effusion_label=np.asarray([i % 2 for i in range(200)], dtype=np.int8),
                real_scores=np.linspace(0, 1, 200),
                control_scores=np.zeros((20, 200)),
                control_labels=np.zeros((20, 200), dtype=np.int8),
                real_bootstrap=np.zeros(2000),
                mean_control_bootstrap=np.zeros(2000),
                selectivity_bootstrap=np.zeros(2000),
            )
            probe["bootstrap_sha256"] = gate.sha256(bootstrap)
            meta = {
                "arch": "llava15_7b", "concept": "Effusion", "alpha_mode": "reltoken",
                "seed": 0, "eval_split": "test", "eval_row_ids": [f"r{i}" for i in range(200)],
                "alphas": gate.ALPHAS, "control_alphas": gate.CONTROL_ALPHAS,
                "n_random": 20, "n_eval": 200, "directions_sha256": directions_hash,
                "model_source": inputs["snapshot"], "model_local_only": True,
                "prompt": REGISTERED_PROMPT, "source_commit": source_commit,
            }
            rows = []
            for alpha in gate.ALPHAS:
                rows.append({"locus": "vis.last", "direction": "concept", "alpha": alpha,
                             "mean_p_yes": 0.5 + 0.1 * alpha, "n": 200})
            for direction in [*(f"random{i}" for i in range(20)), "sham",
                              *(f"unrelated_{name}" for name in gate.UNRELATED_CONCEPTS)]:
                for alpha in gate.CONTROL_ALPHAS:
                    rows.append({"locus": "vis.last", "direction": direction, "alpha": alpha,
                                 "mean_p_yes": 0.5 + 0.01 * alpha, "n": 200})
            paths = {name: root / name for name in (
                "input.json", "hook.json", "extract.json", "probe.json", "meta.json", "done.json"
            )}
            for path, payload in (
                (paths["input.json"], inputs), (paths["hook.json"], hook),
                (paths["extract.json"], extract), (paths["probe.json"], probe),
                (paths["meta.json"], meta),
                (paths["done.json"], {"loci": ["vis.last"], "rows": 165}),
            ):
                path.write_text(json.dumps(payload), encoding="utf-8")
            csv_path = root / "intervention.csv"
            import csv
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            out = root / "summary.json"
            summarize_intervention(
                csv_path, paths["meta.json"], paths["input.json"], paths["hook.json"],
                paths["extract.json"], paths["probe.json"], bootstrap, directions,
                paths["done.json"], out,
            )
            self.assertIn("evidence", json.loads(out.read_text(encoding="utf-8")))
            extract["prompt"] = "drifted"
            paths["extract.json"].write_text(json.dumps(extract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "extraction contract mismatch for prompt"):
                summarize_intervention(
                    csv_path, paths["meta.json"], paths["input.json"], paths["hook.json"],
                    paths["extract.json"], paths["probe.json"], bootstrap, directions,
                    paths["done.json"], out,
                )
            extract["prompt"] = REGISTERED_PROMPT
            paths["extract.json"].write_text(json.dumps(extract), encoding="utf-8")
            bootstrap.unlink()
            with self.assertRaises(FileNotFoundError):
                summarize_intervention(
                    csv_path, paths["meta.json"], paths["input.json"], paths["hook.json"],
                    paths["extract.json"], paths["probe.json"], bootstrap, directions,
                    paths["done.json"], out,
                )


if __name__ == "__main__":
    unittest.main()
