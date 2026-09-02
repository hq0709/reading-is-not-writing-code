#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZED_HOME=/home/qingchan
run_dir=${1:?usage: run_command.sh RUN_DIR}
case "$(realpath -m -- "$run_dir")/" in
  "$AUTHORIZED_HOME"/*) ;;
  *) echo "run directory escapes authorized home" >&2; exit 64 ;;
esac

source_dir="$run_dir/source"
test -f "$source_dir/pyproject.toml"
export RUN_DIR="$run_dir"
cd "$source_dir"
source scripts/server/activate_env.sh
exec bash "$run_dir/command.sh"
