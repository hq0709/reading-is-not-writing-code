#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZED_HOME=/home/qingchan
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
set -a; source config/autoresearch.env; set +a
fail() { echo "server run refused: $*" >&2; exit 64; }
test "$(id -u)" -ne 0 || fail 'root is forbidden'
test "$HOME" = "$AUTHORIZED_HOME" || fail 'unexpected HOME'
for path in "$repo_root" "$DATA_ROOT" "$TMUX_SOCKET"; do
  case "$(realpath -m -- "$path")/" in "$AUTHORIZED_HOME"/*) ;; *) fail "$path escapes authorized home";; esac
done
run_id=${1:?usage: server_run.sh RUN_ID FULL_SHA PAYLOAD}
sha=${2:?usage: server_run.sh RUN_ID FULL_SHA PAYLOAD}
payload=${3:?usage: server_run.sh RUN_ID FULL_SHA PAYLOAD}
[[ "$run_id" =~ ^[A-Za-z0-9._-]{1,128}$ ]] && [[ "$run_id" != .* ]] || fail 'unsafe run id'
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] || fail 'unsafe SHA'
[[ "$payload" =~ ^[A-Za-z0-9+/=]+$ ]] || fail 'unsafe payload'
test -z "$(git status --porcelain --untracked-files=all)" || fail 'server tree is dirty'
git fetch origin
git pull --ff-only
test "$(git rev-parse HEAD)" = "$sha" || fail 'server SHA differs from requested SHA'
test "$(git rev-parse '@{upstream}')" = "$sha" || fail 'server SHA is not pushed upstream'
mkdir -p "$(dirname "$TMUX_SOCKET")"
if ! tmux -S "$TMUX_SOCKET" has-session -t "$EXPERIMENT_TMUX" 2>/dev/null; then
  tmux -S "$TMUX_SOCKET" new-session -d -s "$EXPERIMENT_TMUX" -n control -c "$SERVER_HOME" 'exec tail -f /dev/null'
fi
window="run-${run_id:0:24}"
printf -v launch 'exec %q %q %q %q' "$repo_root/scripts/server/dispatch_run.sh" "$run_id" "$sha" "$payload"
tmux -S "$TMUX_SOCKET" new-window -d -t "$EXPERIMENT_TMUX" -n "$window" -c "$repo_root" "$launch"
printf '%s\n' "$run_id"
