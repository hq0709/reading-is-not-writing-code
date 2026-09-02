#!/usr/bin/env bash
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
set -a; source config/autoresearch.env; set +a
test "$#" -gt 0 || { echo 'usage: remote_run.sh COMMAND [ARG...]' >&2; exit 64; }
test -z "$(git status --porcelain --untracked-files=all)" || { echo 'local tree is dirty' >&2; exit 64; }
git fetch origin
sha=$(git rev-parse HEAD)
test "$(git rev-parse '@{upstream}')" = "$sha" || { echo 'HEAD is not pushed' >&2; exit 64; }
run_id="$(date -u +%Y%m%dT%H%M%SZ)-${sha:0:12}-$(printf '%s:%s' "$$" "$RANDOM" | sha256sum | head -c 8)"
payload=$(python3 -c 'import base64,json,sys; print(base64.b64encode(json.dumps(sys.argv[1:]).encode()).decode())' "$@")
ssh -p "$SERVER_PORT" "$SERVER_USER@$SERVER_HOST" \
  "$SERVER_REPO/scripts/server_run.sh" "$run_id" "$sha" "$payload"
