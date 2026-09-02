"""Registered Qwen2.5-VL Effusion architecture gate at the consumed visual locus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

try:
    from .bootstrap_probe import pct_ci
    from .cross_cell_gate import (
        SOURCE_COMMIT as LLAVA_SOURCE_COMMIT,
    )
    from .cross_cell_gate import (
        SOURCE_RUN_ID as LLAVA_SOURCE_RUN_ID,
    )
    from .cross_cell_gate import (
        SOURCE_SHA256 as LLAVA_SOURCE_SHA256,
    )
    from .cross_cell_gate import (
        load_source_probe_baseline,
    )
    from .first_gate import (
        ALPHAS,
        CONTROL_ALPHAS,
        EXPECTED_ROWS,
        MANIFEST_SHA256,
        REGISTERED_PROMPT,
        UNRELATED_CONCEPTS,
        sha256,
        write_json,
    )
    from .first_gate import (
        run_probe as run_fixed_effusion_probe,
    )
    from .first_gate import (
        summarize_intervention_rows as summarize_fixed_rows,
    )
except ImportError:
    from bootstrap_probe import pct_ci
    from cross_cell_gate import (
        SOURCE_COMMIT as LLAVA_SOURCE_COMMIT,
    )
    from cross_cell_gate import (
        SOURCE_RUN_ID as LLAVA_SOURCE_RUN_ID,
    )
    from cross_cell_gate import (
        SOURCE_SHA256 as LLAVA_SOURCE_SHA256,
    )
    from cross_cell_gate import (
        load_source_probe_baseline,
    )
    from first_gate import (
        ALPHAS,
        CONTROL_ALPHAS,
        EXPECTED_ROWS,
        MANIFEST_SHA256,
        REGISTERED_PROMPT,
        UNRELATED_CONCEPTS,
        sha256,
        write_json,
    )
    from first_gate import (
        run_probe as run_fixed_effusion_probe,
    )
    from first_gate import (
        summarize_intervention_rows as summarize_fixed_rows,
    )


ARCH = "qwen7b"
CONCEPT = "Effusion"
LOCUS = "vis.last"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
EXPECTED_LOCUS = "model.visual.blocks.31"
MODEL_HASHES = {
    "model-00001-of-00005.safetensors": (
        "e97b877e47fde53a6c6e77aafb36e58e91ee9d95c4a3eeac6f1b5c0e6a1c986e"
    ),
    "model-00002-of-00005.safetensors": (
        "a9a300a43b4724eee2abe7c18ceb26768d0ab011eb0cad19d9bfd2476a24d024"
    ),
    "model-00003-of-00005.safetensors": (
        "111223d173e00bbee81cba1216fad28668df3476706b7fd26f4d5b50f8b3a507"
    ),
    "model-00004-of-00005.safetensors": (
        "ef47f634fa57d46ee134edcc09f34085a47da1e16c12a2abe0d67118be6d72ed"
    ),
    "model-00005-of-00005.safetensors": (
        "0c859795ad3a627a9b95bcb762e059d5b768a4a36fdd4affeff269d93fdecc67"
    ),
}


def verify_inputs(manifest: Path, dataset_receipt: Path, model_root: Path, out: Path) -> None:
    dataset = json.loads(dataset_receipt.read_text(encoding="utf-8"))
    model_receipt_path = model_root / "asset-receipt.json"
    model = json.loads(model_receipt_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest)
    if manifest_hash != MANIFEST_SHA256 or dataset.get("registered_manifest_sha256") != manifest_hash:
        raise ValueError("registered NIH manifest hash mismatch")
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    if len(rows) != EXPECTED_ROWS or dataset.get("registered_manifest_rows") != EXPECTED_ROWS:
        raise ValueError("registered NIH manifest row count mismatch")
    row_ids = [row["row_id"] for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("duplicate NIH row_id")
    patient_splits: dict[str, set[str]] = {}
    missing_images = []
    for row in rows:
        patient_splits.setdefault(row["patient_id"], set()).add(row["split"])
        image = Path(row["image_path"])
        if not image.is_file():
            missing_images.append(os.fspath(image))
    if missing_images:
        raise ValueError(f"missing registered images: {missing_images[:3]}")
    if any(len(splits) != 1 for splits in patient_splits.values()):
        raise ValueError("patient appears in multiple splits")

    if (
        model.get("asset") != ARCH
        or model.get("repo_id") != MODEL_ID
        or model.get("revision") != MODEL_REVISION
    ):
        raise ValueError("staged Qwen model identity or revision mismatch")
    snapshot = (model_root / model["snapshot_relative_path"]).resolve(strict=True)
    if not snapshot.is_relative_to(model_root.resolve(strict=True)):
        raise ValueError("Qwen snapshot escapes staged model root")
    receipt_files = model.get("verified_files")
    if not isinstance(receipt_files, dict):
        raise TypeError("Qwen model receipt has no verified file inventory")
    verified_files = {}
    for name, expected in MODEL_HASHES.items():
        path = snapshot / name
        record = receipt_files.get(name)
        if not isinstance(record, dict) or record.get("sha256") != expected:
            raise ValueError(f"accepted Qwen model receipt mismatch for {name}")
        expected_bytes = record.get("bytes")
        if not isinstance(expected_bytes, int) or path.stat().st_size != expected_bytes:
            raise ValueError(f"Qwen model file size drift for {name}")
        verified_files[name] = {"bytes": expected_bytes, "sha256": expected}
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    vision = config.get("vision_config", {})
    if (
        config.get("model_type") != "qwen2_5_vl"
        or vision.get("depth") != 32
        or vision.get("hidden_size") != 1280
        or vision.get("spatial_merge_size") != 2
    ):
        raise ValueError("Qwen architecture identity mismatch")
    write_json(
        out,
        {
            "dataset_receipt": os.fspath(dataset_receipt),
            "dataset_receipt_sha256": sha256(dataset_receipt),
            "manifest": os.fspath(manifest),
            "manifest_sha256": manifest_hash,
            "manifest_rows": len(rows),
            "patients": len(patient_splits),
            "splits": {
                name: sum(row["split"] == name for row in rows)
                for name in ("train", "val", "test")
            },
            "model_receipt": os.fspath(model_receipt_path),
            "model_receipt_sha256": sha256(model_receipt_path),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "snapshot": os.fspath(snapshot),
            "verified_model_files": verified_files,
            "verification_mode": "accepted-asset-receipt",
            "architecture": {
                "model_type": "qwen2_5_vl",
                "vision_depth": 32,
                "vision_hidden_size": 1280,
                "spatial_merge_size": 2,
            },
            "source_commit": os.environ.get("SOURCE_COMMIT", "unknown"),
        },
    )


def hook_implementation_hashes() -> dict[str, str]:
    try:
        from .gpu_env import bind_gpu
        from .intervene import Steerer
        from .loci import Locus, as_hidden, loci_for
        from .registry import REGISTRY
        from .smoke_extract import synthetic_cxr
    except ImportError:
        from gpu_env import bind_gpu
        from intervene import Steerer
        from loci import Locus, as_hidden, loci_for
        from registry import REGISTRY
        from smoke_extract import synthetic_cxr

    sources = {
        "verify_hook": inspect.getsource(verify_hook),
        "bind_gpu": inspect.getsource(bind_gpu),
        "Steerer": inspect.getsource(Steerer),
        "Locus": inspect.getsource(Locus),
        "as_hidden": inspect.getsource(as_hidden),
        "loci_for": inspect.getsource(loci_for),
        "synthetic_cxr": inspect.getsource(synthetic_cxr),
        "qwen7b_registry": json.dumps(vars(REGISTRY[ARCH]), sort_keys=True),
        "registered_hook_protocol": json.dumps(
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "expected_locus": EXPECTED_LOCUS,
                "prompt": REGISTERED_PROMPT,
            },
            sort_keys=True,
        ),
    }
    return {
        name: hashlib.sha256(source.encode("utf-8")).hexdigest()
        for name, source in sources.items()
    }


def validate_hook_receipt(hook_path: Path, input_path: Path) -> None:
    hook = json.loads(hook_path.read_text(encoding="utf-8"))
    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    expected = {
        "arch": ARCH,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "locus": LOCUS,
        "module": EXPECTED_LOCUS,
        "alpha_zero_bitwise_noop": True,
        "forward_path_proved": True,
        "prompt": REGISTERED_PROMPT,
    }
    for key, value in expected.items():
        if hook.get(key) != value:
            raise ValueError(f"Qwen hook receipt mismatch for {key}")
    for key in ("manifest_sha256", "dataset_receipt_sha256", "model_receipt_sha256"):
        if hook.get(key) != inputs.get(key):
            raise ValueError(f"Qwen hook receipt asset mismatch for {key}")
    if hook.get("source_commit") != inputs.get("source_commit"):
        raise ValueError("Qwen hook and full run must use one immutable source commit")
    if hook.get("hook_implementation_hashes") != hook_implementation_hashes():
        raise ValueError("Qwen hook implementation changed after the accepted preflight")


def verify_hook(model_root: Path, input_path: Path, gpu: int, out: Path) -> None:
    try:
        from .gpu_env import bind_gpu
    except ImportError:
        from gpu_env import bind_gpu

    bind_gpu(gpu)
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    try:
        from .intervene import Steerer
        from .loci import as_hidden, loci_for
        from .registry import REGISTRY
        from .smoke_extract import synthetic_cxr
    except ImportError:
        from intervene import Steerer
        from loci import as_hidden, loci_for
        from registry import REGISTRY
        from smoke_extract import synthetic_cxr

    receipt = json.loads((model_root / "asset-receipt.json").read_text(encoding="utf-8"))
    snapshot = model_root / receipt["snapshot_relative_path"]
    arch = REGISTRY[ARCH]
    locus = next(item for item in loci_for(arch) if item.name == LOCUS)
    if locus.module != EXPECTED_LOCUS:
        raise ValueError(f"vis.last resolved to {locus.module}, expected {EXPECTED_LOCUS}")
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    processor = AutoProcessor.from_pretrained(
        snapshot, local_files_only=True, **arch.processor_kwargs
    )
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True
    ).eval()
    modules = dict(model.named_modules())
    if locus.module not in modules or arch.connector not in modules:
        raise ValueError("registered Qwen target or connector module is absent")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": REGISTERED_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[synthetic_cxr()], return_tensors="pt").to("cuda:0")
    target: dict[str, object] = {}
    connector: dict[str, object] = {}

    def target_hook(_module, _inputs, output):
        hidden = as_hidden(output)
        if hidden is not None:
            target["count"] = int(target.get("count", 0)) + 1
            target["shape"] = list(hidden.shape)

    def connector_hook(_module, _inputs, output):
        hidden = as_hidden(output)
        if hidden is not None:
            connector["value"] = hidden.detach().float().cpu()

    target_handle = modules[locus.module].register_forward_hook(target_hook)
    connector_handle = modules[arch.connector].register_forward_hook(connector_hook)
    with torch.inference_mode():
        clean = model(**inputs).logits[:, -1, :].detach().float().cpu()
    clean_connector = connector["value"]
    target_handle.remove()
    connector_handle.remove()
    if target.get("count") != 1:
        raise ValueError(f"registered Qwen target hook fired {target.get('count', 0)} times")

    generator = np.random.default_rng(0)
    vector = generator.standard_normal(target["shape"][-1]).astype(np.float32)
    vector /= np.linalg.norm(vector)
    connector = {}
    connector_handle = modules[arch.connector].register_forward_hook(connector_hook)
    steerer = Steerer(model, locus.module, positions="all", mode="reltoken")
    steerer.vec = vector
    steerer.alpha = 0.0
    with steerer, torch.inference_mode():
        zero = model(**inputs).logits[:, -1, :].detach().float().cpu()
    zero_connector = connector["value"]
    connector_handle.remove()
    if not torch.equal(clean, zero) or not torch.equal(clean_connector, zero_connector):
        raise ValueError("Qwen alpha=0 steering is not a bitwise no-op")

    connector = {}
    connector_handle = modules[arch.connector].register_forward_hook(connector_hook)
    steerer.alpha = 0.1
    with steerer, torch.inference_mode():
        changed = model(**inputs).logits[:, -1, :].detach().float().cpu()
    changed_connector = connector["value"]
    connector_handle.remove()
    connector_delta = float((changed_connector - clean_connector).abs().max())
    logit_delta = float((changed - clean).abs().max())
    if connector_delta <= 0 or logit_delta <= 0:
        raise ValueError("Qwen vis.last perturbation did not change connector and logits")

    verified_inputs = json.loads(input_path.read_text(encoding="utf-8"))
    write_json(
        out,
        {
            "arch": ARCH,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "source_commit": os.environ.get("SOURCE_COMMIT", "unknown"),
            "prompt": REGISTERED_PROMPT,
            "config_model_type": getattr(config, "model_type", None),
            "config_vision_depth": getattr(config.vision_config, "depth", None),
            "locus": LOCUS,
            "module": locus.module,
            "expected_module": EXPECTED_LOCUS,
            "connector": arch.connector,
            "hook_fires": target["count"],
            "hook_shape": target["shape"],
            "synthetic_only": True,
            "steering_mode": "reltoken",
            "steering_alpha": 0.1,
            "alpha_zero_bitwise_noop": True,
            "connector_max_abs_delta": connector_delta,
            "last_logit_max_abs_delta": logit_delta,
            "forward_path_proved": True,
            "hook_implementation_hashes": hook_implementation_hashes(),
            **{
                key: verified_inputs[key]
                for key in (
                    "manifest_sha256",
                    "dataset_receipt_sha256",
                    "model_receipt_sha256",
                )
            },
        },
    )


def augment_probe_with_llava_reference(
    probe_path: Path,
    bootstrap_path: Path,
    reference_path: Path,
    *,
    load_reference: Callable[[dict, np.ndarray, int], tuple[float, np.ndarray]] = (
        load_source_probe_baseline
    ),
) -> None:
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    with np.load(bootstrap_path, allow_pickle=True) as source:
        arrays = {name: source[name].copy() for name in source.files}
    qwen_draws = np.asarray(arrays["selectivity_bootstrap"], dtype=np.float64)
    test_ids = np.asarray(arrays["test_row_id"], dtype=object)
    llava_point, llava_draws = load_reference(reference, test_ids, len(qwen_draws))
    paired_draws = qwen_draws - llava_draws
    qwen_point = float(probe["real_minus_mean_control"])
    paired_interval = pct_ci(paired_draws)
    arrays["qwen_minus_llava_selectivity_bootstrap"] = paired_draws
    temp = bootstrap_path.with_name(f".{bootstrap_path.name}.tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temp, bootstrap_path)

    probe.update(
        {
            "arch": ARCH,
            "module": EXPECTED_LOCUS,
            "bootstrap_sha256": sha256(bootstrap_path),
            "llava_effusion_real_minus_mean_control": llava_point,
            "qwen_minus_llava_selectivity": qwen_point - llava_point,
            "qwen_minus_llava_selectivity_ci95": list(paired_interval),
            "paired_selectivity_reference": (
                "accepted LLaVA Effusion probe with identical test rows and patient bootstrap draws"
            ),
            "llava_reference_run": reference.get("source_run_id"),
            "llava_probe_sha256": LLAVA_SOURCE_SHA256["./artifacts/probe/probe.json"],
            "llava_probe_bootstrap_sha256": LLAVA_SOURCE_SHA256[
                "./artifacts/probe/probe_scores_and_bootstrap.npz"
            ],
        }
    )
    write_json(probe_path, probe)


def run_probe(
    acts_dir: Path,
    manifest: Path,
    llava_reference: Path,
    out_dir: Path,
    n_boot: int = 2000,
) -> None:
    reference = json.loads(llava_reference.read_text(encoding="utf-8"))
    if (
        reference.get("source_run_id") != LLAVA_SOURCE_RUN_ID
        or reference.get("source_commit") != LLAVA_SOURCE_COMMIT
    ):
        raise ValueError("accepted LLaVA comparison receipt identity mismatch")
    run_fixed_effusion_probe(acts_dir, manifest, out_dir, n_boot)
    augment_probe_with_llava_reference(
        out_dir / "probe.json",
        out_dir / "probe_scores_and_bootstrap.npz",
        llava_reference,
    )


def summarize_intervention_rows(rows: list[dict[str, str]]) -> dict:
    summary = summarize_fixed_rows(
        rows,
        alphas=ALPHAS,
        control_alphas=CONTROL_ALPHAS,
        n_random=20,
        n_eval=200,
    )
    summary["arch"] = ARCH
    summary["concept"] = CONCEPT
    return summary


def summarize_intervention(
    csv_path: Path,
    meta_path: Path,
    input_path: Path,
    hook_path: Path,
    extract_meta_path: Path,
    probe_path: Path,
    bootstrap_path: Path,
    directions_path: Path,
    done_path: Path,
    llava_reference_path: Path,
    out: Path,
) -> None:
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    reference = json.loads(llava_reference_path.read_text(encoding="utf-8"))
    expected_meta = {
        "arch": ARCH,
        "concept": CONCEPT,
        "alpha_mode": "reltoken",
        "seed": 0,
        "eval_split": "test",
        "alphas": ALPHAS,
        "control_alphas": CONTROL_ALPHAS,
        "n_random": 20,
        "n_eval": 200,
        "model_source": inputs.get("snapshot"),
        "model_local_only": True,
        "prompt": REGISTERED_PROMPT,
    }
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            raise ValueError(f"Qwen intervention contract mismatch for {key}")
    if meta.get("eval_row_ids") != reference.get("eval_row_ids"):
        raise ValueError("Qwen intervention did not reuse the exact LLaVA evaluation rows")
    validate_hook_receipt(hook_path, input_path)

    extract_meta = json.loads(extract_meta_path.read_text(encoding="utf-8"))
    expected_extract = {
        "arch": ARCH,
        "model": MODEL_ID,
        "model_source": inputs.get("snapshot"),
        "model_local_only": True,
        "prompt": REGISTERED_PROMPT,
        "n_rows": EXPECTED_ROWS,
        "loci": [LOCUS],
        "batch_size": 8,
    }
    for key, value in expected_extract.items():
        if extract_meta.get(key) != value:
            raise ValueError(f"Qwen extraction contract mismatch for {key}")
    if len(extract_meta.get("gpus", [])) != 1:
        raise ValueError("Qwen extraction must use exactly one allocated GPU")

    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    expected_probe = {
        "arch": ARCH,
        "concept": CONCEPT,
        "locus": LOCUS,
        "module": EXPECTED_LOCUS,
        "projection_dim": 512,
        "projection_seed": 0,
        "C": 1.0,
        "probe_seed": 0,
        "control_seeds": list(range(20)),
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20260827,
        "bootstrap_unit": "patient",
        "direction_names": [CONCEPT, *UNRELATED_CONCEPTS],
        "llava_reference_run": LLAVA_SOURCE_RUN_ID,
    }
    for key, value in expected_probe.items():
        if probe.get(key) != value:
            raise ValueError(f"Qwen probe contract mismatch for {key}")
    bootstrap_hash = sha256(bootstrap_path)
    directions_hash = sha256(directions_path)
    if probe.get("bootstrap_sha256") != bootstrap_hash:
        raise ValueError("Qwen probe bootstrap evidence hash mismatch")
    if probe.get("directions_sha256") != directions_hash or meta.get("directions_sha256") != directions_hash:
        raise ValueError("Qwen intervention did not use the registered direction bundle")
    with np.load(bootstrap_path, allow_pickle=True) as bootstrap:
        expected_arrays = {
            "test_row_id",
            "effusion_label",
            "real_scores",
            "control_scores",
            "control_labels",
            "real_bootstrap",
            "mean_control_bootstrap",
            "selectivity_bootstrap",
            "qwen_minus_llava_selectivity_bootstrap",
        }
        if set(bootstrap.files) != expected_arrays:
            raise ValueError("Qwen bootstrap evidence inventory mismatch")
        n_test = probe.get("n_test")
        if (
            not isinstance(n_test, int)
            or n_test <= 0
            or bootstrap["test_row_id"].shape != (n_test,)
            or bootstrap["effusion_label"].shape != (n_test,)
            or bootstrap["real_scores"].shape != (n_test,)
            or bootstrap["control_scores"].shape != (20, n_test)
            or bootstrap["control_labels"].shape != (20, n_test)
            or bootstrap["real_bootstrap"].shape != (2000,)
            or bootstrap["mean_control_bootstrap"].shape != (2000,)
            or bootstrap["selectivity_bootstrap"].shape != (2000,)
            or bootstrap["qwen_minus_llava_selectivity_bootstrap"].shape != (2000,)
        ):
            raise ValueError("Qwen bootstrap evidence shape mismatch")
    selectivity_ci = probe.get("selectivity_ci95")
    paired_ci = probe.get("qwen_minus_llava_selectivity_ci95")
    if (
        not isinstance(selectivity_ci, list)
        or len(selectivity_ci) != 2
        or not all(
            isinstance(value, (int, float)) and np.isfinite(value)
            for value in selectivity_ci
        )
        or not isinstance(probe.get("qwen_minus_llava_selectivity"), (int, float))
        or not isinstance(paired_ci, list)
        or len(paired_ci) != 2
        or not all(isinstance(value, (int, float)) and np.isfinite(value) for value in paired_ci)
    ):
        raise ValueError("Qwen paired LLaVA comparison is invalid")
    source_commit = inputs.get("source_commit")
    if (
        extract_meta.get("git_sha") != source_commit
        or probe.get("source_commit") != source_commit
        or meta.get("source_commit") != source_commit
    ):
        raise ValueError("Qwen gate artifacts do not share one immutable source commit")
    if meta.get("model_source") != inputs.get("snapshot"):
        raise ValueError("Qwen intervention model snapshot mismatch")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("loci") != [LOCUS] or done.get("rows") != 165:
        raise ValueError("Qwen intervention completion marker is incomplete")

    summary = summarize_intervention_rows(rows)
    summary["eval_row_ids"] = meta["eval_row_ids"]
    summary["availability_eligible"] = bool(selectivity_ci[0] > 0)
    summary["probe"] = {
        "real_auroc": probe["real_auroc"],
        "control_auroc_mean": probe["control_auroc_mean"],
        "real_minus_mean_control": probe["real_minus_mean_control"],
        "selectivity_ci95": probe["selectivity_ci95"],
        "llava_effusion_real_minus_mean_control": probe[
            "llava_effusion_real_minus_mean_control"
        ],
        "qwen_minus_llava_selectivity": probe["qwen_minus_llava_selectivity"],
        "qwen_minus_llava_selectivity_ci95": paired_ci,
    }
    summary["llava_reference_run"] = LLAVA_SOURCE_RUN_ID
    summary["evidence"] = {
        "input_verification_sha256": sha256(input_path),
        "hook_verification_sha256": sha256(hook_path),
        "extraction_meta_sha256": sha256(extract_meta_path),
        "llava_reference_sha256": sha256(llava_reference_path),
        "probe_sha256": sha256(probe_path),
        "probe_bootstrap_sha256": bootstrap_hash,
        "directions_sha256": directions_hash,
        "intervention_csv_sha256": sha256(csv_path),
        "intervention_meta_sha256": sha256(meta_path),
        "completion_marker_sha256": sha256(done_path),
    }
    write_json(out, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-inputs")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--dataset-receipt", type=Path, required=True)
    verify.add_argument("--model-root", type=Path, required=True)
    verify.add_argument("--out", type=Path, required=True)
    hook = sub.add_parser("verify-hook")
    hook.add_argument("--model-root", type=Path, required=True)
    hook.add_argument("--input-verification", type=Path, required=True)
    hook.add_argument("--gpu", type=int, default=0)
    hook.add_argument("--out", type=Path, required=True)
    reuse = sub.add_parser("validate-hook-receipt")
    reuse.add_argument("--hook", type=Path, required=True)
    reuse.add_argument("--input-verification", type=Path, required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--acts", type=Path, required=True)
    probe.add_argument("--manifest", type=Path, required=True)
    probe.add_argument("--llava-reference", type=Path, required=True)
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--n-boot", type=int, default=2000)
    summary = sub.add_parser("summarize-intervention")
    summary.add_argument("--csv", type=Path, required=True)
    summary.add_argument("--meta", type=Path, required=True)
    summary.add_argument("--input-verification", type=Path, required=True)
    summary.add_argument("--hook-verification", type=Path, required=True)
    summary.add_argument("--extract-meta", type=Path, required=True)
    summary.add_argument("--probe", type=Path, required=True)
    summary.add_argument("--bootstrap", type=Path, required=True)
    summary.add_argument("--directions", type=Path, required=True)
    summary.add_argument("--done", type=Path, required=True)
    summary.add_argument("--llava-reference", type=Path, required=True)
    summary.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify-inputs":
        verify_inputs(args.manifest, args.dataset_receipt, args.model_root, args.out)
    elif args.command == "verify-hook":
        verify_hook(args.model_root, args.input_verification, args.gpu, args.out)
    elif args.command == "validate-hook-receipt":
        validate_hook_receipt(args.hook, args.input_verification)
    elif args.command == "probe":
        run_probe(args.acts, args.manifest, args.llava_reference, args.out, args.n_boot)
    else:
        summarize_intervention(
            args.csv,
            args.meta,
            args.input_verification,
            args.hook_verification,
            args.extract_meta,
            args.probe,
            args.bootstrap,
            args.directions,
            args.done,
            args.llava_reference,
            args.out,
        )


if __name__ == "__main__":
    main()
