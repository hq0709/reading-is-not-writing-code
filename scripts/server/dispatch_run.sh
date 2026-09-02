#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZED_HOME=/home/qingchan
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
set -a; source config/autoresearch.env; set +a
fail() { printf 'dispatch refused: %s\n' "$*" >&2; exit 64; }
inside_home() { case "$(realpath -m -- "$1")/" in "$AUTHORIZED_HOME"/*) return 0;; *) return 1;; esac; }

test "$(id -u)" -ne 0 || fail 'root is forbidden'
test "$HOME" = "$AUTHORIZED_HOME" || fail "HOME is not $AUTHORIZED_HOME"
for path in "$repo_root" "$DATA_ROOT" "$STOP_SENTINEL"; do inside_home "$path" || fail "$path escapes authorized home"; done
test ! -e "$STOP_SENTINEL" || fail 'stop sentinel exists'
test -z "$(git status --porcelain --untracked-files=all)" || fail 'source tree is dirty'

run_id=${1:?usage: dispatch_run.sh RUN_ID FULL_SHA PAYLOAD}
requested_sha=${2:?usage: dispatch_run.sh RUN_ID FULL_SHA PAYLOAD}
payload=${3:?usage: dispatch_run.sh RUN_ID FULL_SHA PAYLOAD}
[[ "$run_id" =~ ^[A-Za-z0-9._-]{1,128}$ ]] && [[ "$run_id" != .* ]] || fail 'unsafe run id'
[[ "$requested_sha" =~ ^[0-9a-f]{40}$ ]] || fail 'commit must be a full SHA'
[[ "$payload" =~ ^[A-Za-z0-9+/=]+$ ]] || fail 'unsafe command payload'
test "$(git rev-parse HEAD)" = "$requested_sha" || fail 'checkout SHA differs'
test "$(git rev-parse '@{upstream}')" = "$requested_sha" || fail 'commit is not pushed upstream'
mapfile -d '' -t command < <(python3 -c 'import base64,json,sys; [sys.stdout.buffer.write(x.encode()+b"\0") for x in json.loads(base64.b64decode(sys.argv[1]))]' "$payload")
test "${#command[@]}" -gt 0 || fail 'empty decoded command'

state_dir="$DATA_ROOT/state"
runs_dir="$DATA_ROOT/runs"
test -f "$state_dir/bootstrap_started_epoch" || fail 'budget epoch is missing'
now_epoch=$(date +%s)
start_budget_epoch=$(cat "$state_dir/bootstrap_started_epoch")
[[ "$start_budget_epoch" =~ ^[0-9]+$ ]] || fail 'invalid budget epoch'
test $((now_epoch - start_budget_epoch)) -lt $((MAX_WALL_CLOCK_HOURS * 3600)) || fail 'wall-clock budget exhausted'
free_kb=$(df -Pk "$DATA_ROOT" | awk 'NR==2 {print $4}')
test "$free_kb" -ge $((MIN_FREE_DISK_GB * 1024 * 1024)) || fail 'free disk below threshold'
gpu_seconds=$(awk -F= '/^ELAPSED_SECONDS=/{e=$2}/^GPU_COUNT=/{g=$2; s+=e*g} END{print s+0}' "$runs_dir"/*/metadata.env 2>/dev/null || true)
test "${gpu_seconds:-0}" -lt $((MAX_GPU_HOURS * 3600)) || fail 'GPU budget exhausted'
failures=$(find "$runs_dir" -mindepth 2 -maxdepth 2 -name command_exit_status -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n "$MAX_CONSECUTIVE_FAILURES" | cut -d' ' -f2-)
failure_count=0
while IFS= read -r file; do test -n "$file" || continue; test "$(cat "$file")" -eq 0 && break; failure_count=$((failure_count + 1)); done <<<"$failures"
test "$failure_count" -lt "$MAX_CONSECUTIVE_FAILURES" || fail 'consecutive failure threshold reached'

run_dir="$runs_dir/$run_id"
inside_home "$run_dir" || fail 'run path escapes authorized home'
test ! -e "$run_dir" || fail 'run already exists'
mkdir -p "$run_dir"
start_epoch=$(date +%s)
gpu_count=$(nvidia-smi -L | wc -l)
abort_signal=none
command_status=125
child_pid=

terminate_group() {
  test -n "${child_pid:-}" || return 0
  kill -TERM -- "-$child_pid" 2>/dev/null || true
  for _ in {1..20}; do kill -0 "$child_pid" 2>/dev/null || return 0; sleep 1; done
  kill -KILL -- "-$child_pid" 2>/dev/null || true
  while kill -0 "$child_pid" 2>/dev/null; do sleep 1; done
}
on_signal() { abort_signal=$1; command_status=130; terminate_group; exit 130; }
finalize() {
  dispatcher_status=$?
  trap - EXIT INT TERM HUP
  set +e
  terminate_group
  finish_epoch=$(date +%s)
  {
    printf 'FINISHED_UTC=%q\n' "$(date -u +%FT%TZ)"
    printf 'FINISH_EPOCH=%q\n' "$finish_epoch"
    printf 'ELAPSED_SECONDS=%q\n' "$((finish_epoch - start_epoch))"
    printf 'ABORT_SIGNAL=%q\n' "$abort_signal"
    printf 'COMMAND_STATUS=%q\n' "$command_status"
    printf 'DISPATCHER_STATUS=%q\n' "$dispatcher_status"
    printf 'CLEANUP_STATUS=0\n'
  } >>"$run_dir/metadata.env"
  printf '%s\n' "$command_status" >"$run_dir/command_exit_status"
  printf '%s\n' "$dispatcher_status" >"$run_dir/exit_status"
  (cd "$run_dir" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
}
trap finalize EXIT
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap 'on_signal HUP' HUP

{
  printf 'RUN_ID=%q\n' "$run_id"
  printf 'SOURCE_COMMIT=%q\n' "$requested_sha"
  printf 'HOST=%q\n' "$(hostname)"
  printf 'STARTED_UTC=%q\n' "$(date -u +%FT%TZ)"
  printf 'START_EPOCH=%q\n' "$start_epoch"
  printf 'SOURCE_PATH=%q\n' "$run_dir/source"
  printf 'PYTHON_VERSION=%q\n' "$PYTHON_VERSION"
  printf 'GPU_COUNT=%q\n' "$gpu_count"
  printf 'GPU_NAMES=%q\n' "$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';' -)"
} >"$run_dir/metadata.env"
mkdir -p "$run_dir/source"
git archive "$requested_sha" | tar -x -C "$run_dir/source"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\nexec '
  printf '%q ' "${command[@]}"
  printf '\n'
} >"$run_dir/command.sh"
chmod 700 "$run_dir/command.sh"
set +e
setsid timeout --signal=TERM --kill-after=60 "${MAX_SINGLE_RUN_HOURS}h" \
  "$run_dir/source/scripts/server/run_command.sh" "$run_dir" >"$run_dir/stdout.log" 2>"$run_dir/stderr.log" &
child_pid=$!
wait "$child_pid"
command_status=$?
child_pid=
set -e
exit "$command_status"
