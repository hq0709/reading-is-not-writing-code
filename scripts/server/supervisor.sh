#!/usr/bin/env bash
set -euo pipefail
readonly AUTHORIZED_HOME=/home/qingchan
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
set -a; source config/autoresearch.env; set +a
test "$(id -u)" -ne 0 && test "$HOME" = "$AUTHORIZED_HOME"
state_dir="$DATA_ROOT/state"
case "$(realpath -m -- "$state_dir")/" in "$AUTHORIZED_HOME"/*) ;; *) exit 64;; esac
mkdir -p "$state_dir"
exec 9>"$state_dir/supervisor.lock"
flock -n 9 || { echo 'supervisor already running' >&2; exit 64; }
once=${1:-}
while :; do
  tmp=$(mktemp "$state_dir/.health.XXXXXX")
  {
    printf 'CHECKED_UTC=%q\n' "$(date -u +%FT%TZ)"
    printf 'FREE_KB=%q\n' "$(df -Pk "$DATA_ROOT" | awk 'NR==2 {print $4}')"
    printf 'STOP_PRESENT=%q\n' "$(test -e "$STOP_SENTINEL" && echo 1 || echo 0)"
    printf 'GIT_CLEAN=%q\n' "$(test -z "$(git status --porcelain --untracked-files=all)" && echo 1 || echo 0)"
    printf 'CODEX_AUTH=%q\n' "$("$HOME/.local/bin/codex" login status >/dev/null 2>&1 && echo 1 || echo 0)"
  } >"$tmp"
  mv -f "$tmp" "$state_dir/health.env"
  test "$once" = --once && exit 0
  sleep "$SUPERVISOR_INTERVAL_SECONDS"
done
