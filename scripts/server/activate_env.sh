#!/usr/bin/env bash
# Source this file from the repository root before running project commands.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "activate_env.sh must be sourced, not executed" >&2
  exit 2
fi

EXPECTED_HOME="/home/qingchan"
ENV_PREFIX="$EXPECTED_HOME/miniforge3/envs/conceptflow"
CONDA_SH="$EXPECTED_HOME/miniforge3/etc/profile.d/conda.sh"

if [[ "${HOME:-}" != "$EXPECTED_HOME" ]]; then
  echo "refusing activation outside $EXPECTED_HOME (HOME=${HOME:-unset})" >&2
  return 1
fi

if [[ ! -r "$CONDA_SH" ]]; then
  echo "missing Miniforge activation hook: $CONDA_SH" >&2
  return 1
fi

if [[ ! -d "$ENV_PREFIX" ]]; then
  echo "missing Concept Flow environment: $ENV_PREFIX" >&2
  return 1
fi

# uv installs into the same dedicated Conda prefix; it must never create a
# repository-local environment or mutate Conda's base environment.
export UV_PROJECT_ENVIRONMENT="$ENV_PREFIX"
source "$CONDA_SH" || return 1
conda activate "$ENV_PREFIX" || return 1

python - <<'PY' || return 1
import sys

if not (3, 12) <= sys.version_info[:2] < (3, 14):
    raise SystemExit(
        f"Concept Flow requires Python 3.12 or 3.13; found {sys.version.split()[0]}"
    )
PY
