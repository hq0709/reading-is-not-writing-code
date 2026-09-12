"""Where run artefacts live on this machine: <RUN_ROOT>/<model_key>/<dataset_id>/... (return-format layout)."""
from pathlib import Path

RUN_ROOT = Path("/rodata/azradonc_dev/m253405/cf-transfer/runs")


def run_dir(model_key: str, dataset_id: str) -> Path:
    return RUN_ROOT / model_key / dataset_id


def features_dir(model_key: str, dataset_id: str) -> Path:
    return run_dir(model_key, dataset_id) / "features"


def fits_dir(model_key: str, dataset_id: str, locus_id: str) -> Path:
    return run_dir(model_key, dataset_id) / "fits" / locus_id


def outcomes_dir(model_key: str, dataset_id: str) -> Path:
    return run_dir(model_key, dataset_id) / "outcomes"
