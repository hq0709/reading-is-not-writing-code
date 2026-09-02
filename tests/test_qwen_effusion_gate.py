from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.first_gate import ALPHAS, CONTROL_ALPHAS, UNRELATED_CONCEPTS
from src.registry import REGISTRY

ROOT = Path(__file__).resolve().parents[1]


class QwenEffusionGateContractTests(unittest.TestCase):
    def test_fixed_model_and_locus_identity(self) -> None:
        from src.qwen_effusion_gate import (
            ARCH,
            EXPECTED_LOCUS,
            MODEL_ID,
            MODEL_REVISION,
        )

        self.assertEqual("qwen7b", ARCH)
        self.assertEqual("Qwen/Qwen2.5-VL-7B-Instruct", MODEL_ID)
        self.assertEqual("cc594898137f460bfe9f0759e9844b3ce807cfb5", MODEL_REVISION)
        locus = REGISTRY[ARCH].vision_block_fmt.format(i=31)
        self.assertEqual("model.visual.blocks.31", locus)
        self.assertEqual(locus, EXPECTED_LOCUS)

    def test_model_stage_registers_all_five_official_weight_shards(self) -> None:
        from scripts.server.stage_qwen7b_asset import MODEL_HASHES, MODEL_REVISION, MODEL_ROOT

        self.assertEqual("cc594898137f460bfe9f0759e9844b3ce807cfb5", MODEL_REVISION)
        self.assertEqual(
            {
                "model-00001-of-00005.safetensors",
                "model-00002-of-00005.safetensors",
                "model-00003-of-00005.safetensors",
                "model-00004-of-00005.safetensors",
                "model-00005-of-00005.safetensors",
            },
            set(MODEL_HASHES),
        )
        self.assertEqual(
            Path("/home/qingchan/data/concept-flow/models/qwen7b-huggingface"), MODEL_ROOT
        )
        self.assertTrue(all(len(digest) == 64 for digest in MODEL_HASHES.values()))

    def test_runner_fixes_the_registered_protocol_and_extracts_fresh_activations(self) -> None:
        runner = (ROOT / "scripts/server/run_qwen_effusion_vislast_gate.sh").read_text(
            encoding="utf-8"
        )
        for token in (
            "hook|full",
            "--arch qwen7b",
            "--concept Effusion",
            "--loci vis.last",
            "--n-boot 2000",
            "--alphas=-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1",
            "--control-alphas=-1,-0.5,-0.25,0.25,0.5,1",
            "--n-random 20",
            "--n-eval 200",
            "--eval-split test",
            "Is there a pleural effusion in this chest radiograph? Answer yes or no.",
            "src/extract.py",
            "src/cross_cell_gate.py",
            "validate-hook-receipt",
        ):
            self.assertIn(token, runner)

    def test_qwen_summary_uses_the_unchanged_165_row_control_grid(self) -> None:
        from src.qwen_effusion_gate import summarize_intervention_rows

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

        self.assertEqual(165, len(rows))
        self.assertEqual("qwen7b", summary["arch"])
        self.assertEqual("Effusion", summary["concept"])
        self.assertTrue(summary["selective_cell"])

    def test_probe_augmentation_records_paired_llava_difference(self) -> None:
        from src.qwen_effusion_gate import augment_probe_with_llava_reference

        with TemporaryDirectory() as temp:
            root = Path(temp)
            probe_path = root / "probe.json"
            bootstrap_path = root / "probe_scores_and_bootstrap.npz"
            reference_path = root / "reference.json"
            source_probe = root / "source-probe.json"
            source_bootstrap = root / "source-bootstrap.npz"
            test_ids = np.asarray([f"r{i}" for i in range(5343)], dtype=object)
            probe_path.write_text(
                json.dumps(
                    {
                        "arch": "llava15_7b",
                        "concept": "Effusion",
                        "module": "model.vision_tower.encoder.layers.22",
                        "real_minus_mean_control": 0.2,
                    }
                ),
                encoding="utf-8",
            )
            np.savez_compressed(
                bootstrap_path,
                test_row_id=test_ids,
                effusion_label=np.zeros(5343),
                real_scores=np.zeros(5343),
                control_scores=np.zeros((20, 5343)),
                control_labels=np.zeros((20, 5343)),
                real_bootstrap=np.zeros(2000),
                mean_control_bootstrap=np.zeros(2000),
                selectivity_bootstrap=np.full(2000, 0.2),
            )
            source_probe.write_text("{}", encoding="utf-8")
            np.savez_compressed(source_bootstrap, placeholder=np.zeros(1))
            reference_path.write_text(
                json.dumps(
                    {
                        "source_run_id": "source",
                        "source_probe": str(source_probe),
                        "source_probe_bootstrap": str(source_bootstrap),
                    }
                ),
                encoding="utf-8",
            )

            def load_reference(_reference: dict, ids: np.ndarray, n_boot: int):
                np.testing.assert_array_equal(test_ids, ids)
                self.assertEqual(2000, n_boot)
                return 0.1, np.full(2000, 0.1)

            augment_probe_with_llava_reference(
                probe_path, bootstrap_path, reference_path, load_reference=load_reference
            )

            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            self.assertEqual("qwen7b", probe["arch"])
            self.assertEqual("model.visual.blocks.31", probe["module"])
            self.assertAlmostEqual(0.1, probe["qwen_minus_llava_selectivity"])
            with np.load(bootstrap_path, allow_pickle=True) as bootstrap:
                np.testing.assert_allclose(
                    0.1, bootstrap["qwen_minus_llava_selectivity_bootstrap"]
                )

    def test_hook_receipt_cannot_cross_source_commits(self) -> None:
        from src.qwen_effusion_gate import (
            ARCH,
            EXPECTED_LOCUS,
            LOCUS,
            MODEL_ID,
            MODEL_REVISION,
            hook_implementation_hashes,
            validate_hook_receipt,
        )

        with TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = {
                "source_commit": "full-commit",
                "manifest_sha256": "manifest",
                "dataset_receipt_sha256": "dataset",
                "model_receipt_sha256": "model",
            }
            hook = {
                "arch": ARCH,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "locus": LOCUS,
                "module": EXPECTED_LOCUS,
                "alpha_zero_bitwise_noop": True,
                "forward_path_proved": True,
                "prompt": "Is there a pleural effusion in this chest radiograph? Answer yes or no.",
                "source_commit": "hook-commit",
                "manifest_sha256": "manifest",
                "dataset_receipt_sha256": "dataset",
                "model_receipt_sha256": "model",
                "hook_implementation_hashes": hook_implementation_hashes(),
            }
            input_path = root / "inputs.json"
            hook_path = root / "hook.json"
            input_path.write_text(json.dumps(inputs), encoding="utf-8")
            hook_path.write_text(json.dumps(hook), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "one immutable source commit"):
                validate_hook_receipt(hook_path, input_path)

    def test_model_receipt_rejects_wrong_official_revision(self) -> None:
        from scripts.server.stage_qwen7b_asset import MODEL_HASHES, official_shard_hashes

        class Lfs:
            def __init__(self, digest: str):
                self.sha256 = digest

        class Sibling:
            def __init__(self, name: str, digest: str):
                self.rfilename = name
                self.lfs = Lfs(digest)

        siblings = [Sibling(name, digest) for name, digest in MODEL_HASHES.items()]
        self.assertEqual(MODEL_HASHES, official_shard_hashes(siblings))
        siblings[0].lfs.sha256 = "0" * 64
        with self.assertRaisesRegex(SystemExit, "official weight shard identity"):
            official_shard_hashes(siblings)


if __name__ == "__main__":
    unittest.main()
