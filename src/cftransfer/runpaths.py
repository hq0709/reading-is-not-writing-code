"""Where run artefacts live on this machine: <RUN_ROOT>/<model_key>/<dataset_id>/... (return-format layout)."""
from pathlib import Path

RUN_ROOT = Path("/rodata/azradonc_dev/m253405/cf-transfer/runs")


def run_dir(model_key: str, dataset_id: str) -> Path:
    return RUN_ROOT / model_key / dataset_id


def features_dir(model_key: str, dataset_id: str) -> Path:
    return run_dir(model_key, dataset_id) / "features"


def valid_features_dir(model_key: str, dataset_id: str) -> Path:
    """Features of the `valid` role (VALID module), extracted separately so features/<locus>.npz stays the campaign file."""
    return features_dir(model_key, dataset_id) / "valid"


def fits_dir(model_key: str, dataset_id: str, locus_id: str) -> Path:
    return run_dir(model_key, dataset_id) / "fits" / locus_id


def outcomes_dir(model_key: str, dataset_id: str) -> Path:
    return run_dir(model_key, dataset_id) / "outcomes"
