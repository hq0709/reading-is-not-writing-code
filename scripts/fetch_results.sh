#!/usr/bin/env bash
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
set -a; source config/autoresearch.env; set +a
run_id=${1:?usage: fetch_results.sh RUN_ID}
[[ "$run_id" =~ ^[A-Za-z0-9._-]{1,128}$ ]] && [[ "$run_id" != .* ]] || { echo 'unsafe run id' >&2; exit 64; }
dest="$repo_root/results/remote/$run_id"
partial="$repo_root/results/remote/.partial-$run_id-$$"
test ! -e "$dest" && test ! -e "$partial"
mkdir -p "$partial"
trap 'rm -rf -- "$partial"' EXIT
scp -P "$SERVER_PORT" -r "$SERVER_USER@$SERVER_HOST:$DATA_ROOT/runs/$run_id/." "$partial/"
(cd "$partial" && sha256sum -c SHA256SUMS)
test -f "$partial/metadata.env" && test -f "$partial/exit_status"
mv "$partial" "$dest"
trap - EXIT
printf '%s\n' "$dest"
