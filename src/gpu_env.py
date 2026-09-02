"""Bind a process to one GPU without escaping a dispatcher allocation."""
from __future__ import annotations

import os


def bind_gpu(gpu: int) -> None:
    """Select a physical GPU, rejecting IDs outside an inherited allocation."""
    requested = str(gpu)
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is not None:
        allocated = {item.strip() for item in inherited.split(",") if item.strip()}
        if requested not in allocated:
            raise RuntimeError(
                f"GPU {requested} is outside dispatcher allocation {inherited!r}"
            )
    os.environ["CUDA_VISIBLE_DEVICES"] = requested
