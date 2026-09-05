"""Execute and replay the registered LLaVA text semantic calibration."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from src import llava_text_semantic_calibration as core
from src import run_llava_paired_opportunity as accepted

SUMMARY = "llava-text-semantic-calibration-summary.json"
OUTCOMES = "llava-text-semantic-calibration-outcomes.json"
META = "llava-text-semantic-calibration-meta.json"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def full_sha(value):
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise argparse.ArgumentTypeError("source-commit must be a full lowercase SHA")
    return value


def resolved_paths(data_root, out, source_commit):
    full_sha(source_commit)
    if os.environ.get("SOURCE_COMMIT", source_commit) != source_commit:
        raise ValueError("source-commit differs from immutable dispatcher metadata")
    root = Path(data_root).resolve(strict=True)
    out = Path(out).resolve()
    if out == root or not out.is_relative_to(root):
        raise ValueError("output must remain below the accepted data root")
    return root, out


def protocol():
    return {
        "gate": "llava-text-semantic-calibration",
        "model_id": accepted.MODEL_ID,
        "model_revision": accepted.MODEL_REVISION,
        "dtype": "bfloat16",
        "interface": "image-free processor conversation, one independent forward pass per case",
        "objects": list(core.OBJECTS),
        "conditions": [condition["name"] for condition in core.CONDITIONS],
        "token_groups": {
            key: [list(first), list(second)]
            for key, (first, second) in core.PINNED_TOKEN_GROUPS.items()
        },
        "n_descriptions": 16,
        "n_outcomes": 80,
        "resource_envelope_seconds": 900,
    }


def load_source(root):
    model_root = (root / "models/huggingface").resolve(strict=True)
    receipt_path = (model_root / "asset-receipt.json").resolve(strict=True)
    snapshot = (model_root / accepted.SNAPSHOT_RELATIVE_PATH).resolve(strict=True)
    receipt = read_json(receipt_path)
    expected_receipt = {
        "asset": accepted.ARCH,
        "repo_id": accepted.MODEL_ID,
        "revision": accepted.MODEL_REVISION,
        "snapshot_relative_path": accepted.SNAPSHOT_RELATIVE_PATH,
    }
    if any(receipt.get(key) != value for key, value in expected_receipt.items()):
        raise ValueError("accepted LLaVA asset receipt identity mismatch")
    source = {
        "model_id": accepted.MODEL_ID,
        "model_revision": accepted.MODEL_REVISION,
        "model_source": str(snapshot),
        "model_receipt": str(receipt_path),
        "asset_receipt": receipt,
    }
    for key in ("model_source", "model_receipt"):
        if not Path(source[key]).resolve(strict=True).is_relative_to(root):
            raise ValueError("accepted model provenance escapes the data root")
    return source


def load_scorer(source, gpu):
    from src.gpu_env import bind_gpu

    bind_gpu(gpu)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from src.intervene import NO_WORDS, YES_WORDS
    from src.qwen_answer_encoding import singleton_ids
    from src.registry import REGISTRY

    arch = REGISTRY[accepted.ARCH]
    if (
        arch.hf_id != accepted.MODEL_ID
        or arch.family != "llava"
        or arch.processor_kwargs != {}
        or torch.cuda.device_count() != 1
        or "A100" not in torch.cuda.get_device_name(0)
    ):
        raise ValueError("calibration requires the accepted LLaVA registry and one A100")
    model_path = source["model_source"]
    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=True, **arch.processor_kwargs
    )
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True
    ).eval()
    accepted.validate_processor(processor, model.config)
    yes, no = singleton_ids(processor.tokenizer, YES_WORDS, NO_WORDS)
    letter_a, letter_b = singleton_ids(processor.tokenizer)
    observed = {
        "Y": (yes, no),
        "A": (letter_a, letter_b),
        "B": (letter_a, letter_b),
    }
    tokens = {}
    for condition in core.CONDITIONS:
        encoding = condition["encoding"]
        if tuple(map(list, observed[encoding])) != tuple(
            map(list, core.PINNED_TOKEN_GROUPS[encoding])
        ):
            raise ValueError("tokenizer differs from the pinned singleton identities")
        tokens[condition["name"]] = core.token_metadata(encoding)
    core.validate_token_metadata(tokens)

    def score(identity):
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": identity["user_prompt"]}]}
        ]
        rendered = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[rendered], return_tensors="pt", padding=True).to("cuda:0")
        if "pixel_values" in inputs or "image_sizes" in inputs:
            raise ValueError("image-free calibration unexpectedly produced image inputs")
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1, :].float()
        metadata = core.token_metadata(identity["encoding"])
        token_ids = metadata["candidate_ids"]
        candidate = logits[token_ids].cpu().numpy().astype(np.float32)
        partition = float(torch.logsumexp(logits, dim=-1).cpu())
        return core.derive_record(identity, rendered, candidate, partition, token_ids)

    runtime = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": 1,
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(processor.tokenizer).__name__,
        "chat_template": processor.chat_template,
    }
    return score, tokens, runtime


def collect(score):
    outcomes = [score(identity) for identity in core.cases()]
    core.validate_outcomes(outcomes)
    return outcomes


def run(data_root, out, source_commit, gpu):
    root, out = resolved_paths(data_root, out, source_commit)
    if any((out / name).exists() for name in (META, OUTCOMES, SUMMARY)):
        raise ValueError("calibration artifacts already exist")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source = load_source(root)
    score, tokens, runtime = load_scorer(source, gpu)
    meta = {
        "source_commit": source_commit,
        "protocol": protocol(),
        "source": source,
        "token_ids": tokens,
        "runtime": runtime,
    }
    write_json(out / META, meta)
    outcomes = collect(score)
    write_json(out / OUTCOMES, outcomes)
    summary = {
        **core.summarize(outcomes),
        "source_commit": source_commit,
        "elapsed_seconds": time.perf_counter() - started,
        "protocol": protocol(),
        "source": source,
        "token_ids": tokens,
    }
    if summary["elapsed_seconds"] > 900:
        raise TimeoutError("calibration exceeded its fifteen-minute resource envelope")
    write_json(out / SUMMARY, summary)
    return summary


def replay(data_root, out, source_commit):
    root, out = resolved_paths(data_root, out, source_commit)
    meta = read_json(out / META)
    source = load_source(root)
    if (
        meta.get("source_commit") != source_commit
        or meta.get("protocol") != protocol()
        or meta.get("source") != source
    ):
        raise ValueError("calibration metadata or source identity mismatch")
    core.validate_token_metadata(meta.get("token_ids"))
    outcomes = read_json(out / OUTCOMES)
    expected = {
        **core.summarize(outcomes),
        "source_commit": source_commit,
        "protocol": protocol(),
        "source": source,
        "token_ids": meta["token_ids"],
    }
    summary = read_json(out / SUMMARY)
    elapsed = summary.pop("elapsed_seconds", None)
    if not isinstance(elapsed, (int, float)) or not 0 < elapsed <= 900 or summary != expected:
        raise ValueError("calibration summary does not replay from retained logits")
    summary["elapsed_seconds"] = elapsed
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "replay"):
        sub = commands.add_parser(command)
        sub.add_argument("--data-root", type=Path, required=True)
        sub.add_argument("--out", type=Path, required=True)
        sub.add_argument("--source-commit", type=full_sha, required=True)
        if command == "run":
            sub.add_argument("--gpu", type=int, required=True)
    args = vars(parser.parse_args(argv))
    return {"run": run, "replay": replay}[args.pop("command")](**args)


if __name__ == "__main__":
    main()
