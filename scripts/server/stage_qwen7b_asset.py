#!/usr/bin/env python3
"""Atomically stage the pinned Qwen2.5-VL asset for the architecture gate."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

try:
    from .stage_first_gate_assets import (
        DATA_ROOT,
        assert_managed_path,
        assert_model_tree_symlinks_stay_within,
        assert_runtime_safe,
        assert_safe,
        atomic_publish,
        fail,
        sha256,
        staging_lock,
        write_json,
    )
except ImportError:
    from stage_first_gate_assets import (
        DATA_ROOT,
        assert_managed_path,
        assert_model_tree_symlinks_stay_within,
        assert_runtime_safe,
        assert_safe,
        atomic_publish,
        fail,
        sha256,
        staging_lock,
        write_json,
    )


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MODEL_ROOT = DATA_ROOT / "models/qwen7b-huggingface"
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


def _lfs_sha256(sibling: object) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, dict):
        value = lfs.get("sha256")
    else:
        value = getattr(lfs, "sha256", None)
    return value if isinstance(value, str) else None


def official_shard_hashes(siblings: list[object]) -> dict[str, str]:
    observed = {
        str(getattr(sibling, "rfilename", "")): _lfs_sha256(sibling)
        for sibling in siblings
        if getattr(sibling, "rfilename", "") in MODEL_HASHES
    }
    if observed != MODEL_HASHES:
        fail("official weight shard identity differs from the registered Qwen revision")
    return {name: digest for name, digest in observed.items() if digest is not None}


def stage_model(head: str) -> None:
    partial = DATA_ROOT / f"models/.partial-qwen7b-{MODEL_REVISION}"
    hub = partial / "hub"
    for path in (MODEL_ROOT.parent, MODEL_ROOT, partial, hub):
        assert_managed_path(path)
    if MODEL_ROOT.exists():
        fail(f"model target already exists: {MODEL_ROOT}")
    partial.mkdir(parents=True, exist_ok=True)
    assert_model_tree_symlinks_stay_within(partial)
    hub.mkdir(exist_ok=True)
    os.environ["HF_HOME"] = os.fspath(partial)

    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig, AutoProcessor

    assert_runtime_safe(head)
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION or info.private or info.gated:
        fail("Qwen model identity, access, or revision mismatch")
    official_shard_hashes(info.siblings)
    assert_runtime_safe(head)
    print(f"staging {MODEL_ID}@{MODEL_REVISION}", flush=True)
    snapshot = Path(
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=hub)
    ).resolve()
    if snapshot.name != MODEL_REVISION or not snapshot.is_relative_to(partial.resolve()):
        fail(f"unexpected Qwen snapshot path: {snapshot}")
    assert_model_tree_symlinks_stay_within(partial)

    verified_files: dict[str, dict[str, int | str]] = {}
    for name, expected in MODEL_HASHES.items():
        path = snapshot / name
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(partial.resolve(strict=True)):
            fail(f"Qwen model blob escapes staging partial: {path} -> {resolved}")
        assert_runtime_safe(head)
        actual = sha256(path)
        if actual != expected:
            fail(f"Qwen model hash mismatch for {name}: {actual}")
        verified_files[name] = {"bytes": path.stat().st_size, "sha256": actual}

    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    AutoProcessor.from_pretrained(
        snapshot,
        local_files_only=True,
        min_pixels=336 * 336,
        max_pixels=336 * 336,
    )
    vision = getattr(config, "vision_config", None)
    if (
        getattr(config, "model_type", None) != "qwen2_5_vl"
        or getattr(vision, "depth", None) != 32
        or getattr(vision, "hidden_size", None) != 1280
        or getattr(vision, "spatial_merge_size", None) != 2
    ):
        fail("Qwen architecture identity differs from the registered model")

    assert_runtime_safe(head)
    write_json(
        partial / "asset-receipt.json",
        {
            "asset": "qwen7b",
            "completed_utc": datetime.now(UTC).isoformat(),
            "source": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "source_commit": head,
            "snapshot_relative_path": os.fspath(snapshot.relative_to(partial.resolve())),
            "verified_files": verified_files,
            "architecture": {
                "model_type": "qwen2_5_vl",
                "vision_depth": 32,
                "vision_hidden_size": 1280,
                "spatial_merge_size": 2,
            },
        },
    )
    atomic_publish(partial, MODEL_ROOT, head, allow_internal_symlinks_within_partial=True)
    print(f"Qwen model staged at {MODEL_ROOT}", flush=True)


def main() -> None:
    with staging_lock():
        stage_model(assert_safe())


if __name__ == "__main__":
    main()
