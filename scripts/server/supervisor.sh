#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZED_HOME=/home/qingchan
readonly MANAGED_STOP_CONTENT='AUTORESEARCH_SUPERVISOR_BLOCK_V1'
readonly REVIEW_RECEIPT_MAX_AGE_SECONDS=86400

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
set -a; source config/autoresearch.env; set +a

fail() { printf 'supervisor refused: %s\n' "$*" >&2; exit 64; }
inside_home() { case "$(realpath -m -- "$1")/" in "$AUTHORIZED_HOME"/*) return 0;; *) return 1;; esac; }
is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }

test "$(id -u)" -ne 0 || fail 'root is forbidden'
test "$HOME" = "$AUTHORIZED_HOME" || fail 'unexpected HOME'
for path in "$repo_root" "$DATA_ROOT" "$ARIS_REPO" "$TMUX_SOCKET" "$STOP_SENTINEL"; do
  inside_home "$path" || fail "$path escapes authorized home"
done

state_dir="$DATA_ROOT/state"
runs_dir="$DATA_ROOT/runs"
health_file="$state_dir/health.env"
mkdir -p "$state_dir" "$runs_dir"

mode=${1:-}
case "$mode" in ''|--once|--gate-only) ;; *) fail 'usage: supervisor.sh [--once|--gate-only]';; esac
if test "$mode" != --gate-only; then
  exec 9>"$state_dir/supervisor.lock"
  flock -n 9 || fail 'supervisor already running'
fi

check_health() {
  local now_epoch start_epoch free_kb gpu_seconds failure_count api_spend
  local git_head git_upstream latest_receipt receipt_epoch latest_progress active_windows audit
  local reasons=()
  now_epoch=$(date +%s)

  STOP_OK=1
  if test -e "$STOP_SENTINEL" && ! grep -Fx "$MANAGED_STOP_CONTENT" "$STOP_SENTINEL" >/dev/null 2>&1; then
    STOP_OK=0; reasons+=(stop_sentinel)
  fi

  DISK_OK=0
  free_kb=$(df -Pk "$DATA_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')
  if is_uint "${free_kb:-}" && is_uint "$MIN_FREE_DISK_GB" &&
     test "$free_kb" -ge $((MIN_FREE_DISK_GB * 1024 * 1024)); then DISK_OK=1; else reasons+=(disk); fi

  WALL_CLOCK_OK=0
  if test -f "$state_dir/bootstrap_started_epoch"; then
    start_epoch=$(<"$state_dir/bootstrap_started_epoch")
    if is_uint "$start_epoch" && is_uint "$MAX_WALL_CLOCK_HOURS" && test "$now_epoch" -ge "$start_epoch" &&
       test $((now_epoch - start_epoch)) -lt $((MAX_WALL_CLOCK_HOURS * 3600)); then WALL_CLOCK_OK=1; fi
  fi
  test "$WALL_CLOCK_OK" -eq 1 || reasons+=(wall_clock)

  GPU_BUDGET_OK=0
  gpu_seconds=$(find "$runs_dir" -mindepth 2 -maxdepth 2 -name metadata.env -print0 2>/dev/null |
    xargs -0 -r awk -F= '/^ELAPSED_SECONDS=/{e=$2}/^GPU_COUNT=/{g=$2; s+=e*g} END{print s+0}')
  gpu_seconds=${gpu_seconds:-0}
  if is_uint "$MAX_GPU_HOURS" && awk -v used="$gpu_seconds" -v limit="$((MAX_GPU_HOURS * 3600))" 'BEGIN {exit !(used < limit)}'; then GPU_BUDGET_OK=1; else reasons+=(gpu_budget); fi

  FAILURE_BUDGET_OK=0
  failure_count=0
  while IFS= read -r status_file; do
    test -n "$status_file" || continue
    if test "$(<"$status_file")" -eq 0 2>/dev/null; then break; fi
    failure_count=$((failure_count + 1))
  done < <(find "$runs_dir" -mindepth 2 -maxdepth 2 -name command_exit_status -printf '%T@ %p\n' 2>/dev/null |
    sort -nr | head -n "$MAX_CONSECUTIVE_FAILURES" | cut -d' ' -f2-)
  if is_uint "$MAX_CONSECUTIVE_FAILURES" && test "$failure_count" -lt "$MAX_CONSECUTIVE_FAILURES"; then FAILURE_BUDGET_OK=1; else reasons+=(consecutive_failures); fi

  API_BUDGET_OK=0
  api_spend=$(find "$runs_dir" -mindepth 2 -maxdepth 2 -name metadata.env -print0 2>/dev/null |
    xargs -0 -r awk -F= '/^API_SPEND_USD=/{s+=$2} END{printf "%.6f", s+0}')
  api_spend=${api_spend:-0.000000}
  if awk -v used="$api_spend" -v limit="$MAX_API_SPEND_USD" 'BEGIN {exit !(used <= limit)}'; then API_BUDGET_OK=1; else reasons+=(api_budget); fi

  GIT_OK=0
  if test -z "$(git status --porcelain --untracked-files=all)" && git fetch origin --quiet 2>/dev/null; then
    git_head=$(git rev-parse HEAD 2>/dev/null || true)
    git_upstream=$(git rev-parse '@{upstream}' 2>/dev/null || true)
    test -n "$git_head" && test "$git_head" = "$git_upstream" && GIT_OK=1
  fi
  test "$GIT_OK" -eq 1 || reasons+=(git)

  CODEX_AUTH_OK=0
  "$HOME/.local/bin/codex" login status >/dev/null 2>&1 && CODEX_AUTH_OK=1
  test "$CODEX_AUTH_OK" -eq 1 || reasons+=(codex_auth)

  ARIS_AUDIT_OK=0
  audit="$state_dir/aris-audit.env"
  if test "$(git -C "$ARIS_REPO" rev-parse HEAD 2>/dev/null || true)" = "$ARIS_FULL_SHA" && test -f "$audit" &&
     grep -Fx "ARIS_FULL_SHA=$ARIS_FULL_SHA" "$audit" >/dev/null 2>&1 &&
     grep -Fx "SKILL_LIST_SHA256=$(sha256sum config/aris-skills.txt | cut -d ' ' -f1)" "$audit" >/dev/null 2>&1 &&
     grep -Fx "FORBIDDEN_LIST_SHA256=$(sha256sum config/aris-forbidden-skills.txt | cut -d ' ' -f1)" "$audit" >/dev/null 2>&1 &&
     grep -Fx "REVIEWER_ROUTING_SHA256=$(sha256sum config/reviewer-routing.tsv | cut -d ' ' -f1)" "$audit" >/dev/null 2>&1 &&
     grep -Fx "REVIEWER_WRAPPER_SHA256=$(sha256sum scripts/server/claude_review_adapter.py | cut -d ' ' -f1)" "$audit" >/dev/null 2>&1; then ARIS_AUDIT_OK=1; fi
  test "$ARIS_AUDIT_OK" -eq 1 || reasons+=(aris_audit)

  REVIEWER_RECEIPT_OK=0
  latest_receipt=$(find "$HOME/.codex/state/$REVIEW_TRANSPORT" -maxdepth 1 -type f -name 'review-*.json' -printf '%T@ %p\n' 2>/dev/null |
    sort -nr | head -n 1 | cut -d' ' -f2-)
  receipt_epoch=0
  test -z "$latest_receipt" || receipt_epoch=$(stat -c %Y "$latest_receipt")
  if test -n "$latest_receipt" && test "$now_epoch" -ge "$receipt_epoch" &&
     test $((now_epoch - receipt_epoch)) -le "$REVIEW_RECEIPT_MAX_AGE_SECONDS"; then
    "$CONDA_ROOT/envs/$CONDA_ENV/bin/python" - "$latest_receipt" "$CLAUDE_REVIEW_MODEL" "$ARIS_FULL_SHA" "$(sha256sum scripts/server/claude_review_adapter.py | cut -d ' ' -f1)" <<'PY' >/dev/null 2>&1 && REVIEWER_RECEIPT_OK=1
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    receipt = json.load(stream)
assert receipt.get("valid") is True
assert receipt.get("readOnly") is True
assert receipt.get("modelsObserved") == [sys.argv[2]]
assert receipt.get("arisFullSha") == sys.argv[3]
assert receipt.get("wrapperSha256") == sys.argv[4]
PY
  fi
  test "$REVIEWER_RECEIPT_OK" -eq 1 || reasons+=(reviewer_receipt)

  AGENT_LIVE=1; EXPERIMENT_TMUX_LIVE=1; EXPERIMENT_PROGRESS_OK=1
  active_windows=0; latest_progress=0
  if test "$mode" != --gate-only; then
    tmux -S "$TMUX_SOCKET" has-session -t "$AGENT_TMUX" 2>/dev/null || AGENT_LIVE=0
    tmux -S "$TMUX_SOCKET" has-session -t "$EXPERIMENT_TMUX" 2>/dev/null || EXPERIMENT_TMUX_LIVE=0
    test "$AGENT_LIVE" -eq 1 || reasons+=(agent_tmux)
    test "$EXPERIMENT_TMUX_LIVE" -eq 1 || reasons+=(experiment_tmux)
    if test "$EXPERIMENT_TMUX_LIVE" -eq 1; then
      active_windows=$(tmux -S "$TMUX_SOCKET" list-windows -t "$EXPERIMENT_TMUX" -F '#{window_name} #{window_active}' 2>/dev/null |
        awk '$1 != "control" && $1 != "supervisor" {n++} END{print n+0}')
      if test "$active_windows" -gt 0; then
        latest_progress=$(find "$runs_dir" -mindepth 2 -maxdepth 2 -type f ! -name SHA256SUMS -printf '%T@\n' 2>/dev/null |
          sort -nr | head -n 1 | cut -d. -f1)
        latest_progress=${latest_progress:-0}
        if test $((now_epoch - latest_progress)) -gt $((MAX_SINGLE_RUN_HOURS * 3600)); then EXPERIMENT_PROGRESS_OK=0; reasons+=(experiment_progress); fi
      fi
    fi
  fi

  HEALTHY=1
  for value in "$STOP_OK" "$DISK_OK" "$WALL_CLOCK_OK" "$GPU_BUDGET_OK" "$FAILURE_BUDGET_OK" "$API_BUDGET_OK" "$GIT_OK" "$CODEX_AUTH_OK" "$ARIS_AUDIT_OK" "$REVIEWER_RECEIPT_OK" "$AGENT_LIVE" "$EXPERIMENT_TMUX_LIVE" "$EXPERIMENT_PROGRESS_OK"; do
    test "$value" -eq 1 || HEALTHY=0
  done

  tmp=$(mktemp "$state_dir/.health.XXXXXX")
  {
    printf 'CHECKED_UTC=%q\n' "$(date -u +%FT%TZ)"
    printf 'HEALTHY=%s\nSTOP_OK=%s\nDISK_OK=%s\nWALL_CLOCK_OK=%s\n' "$HEALTHY" "$STOP_OK" "$DISK_OK" "$WALL_CLOCK_OK"
    printf 'GPU_BUDGET_OK=%s\nGPU_SECONDS=%q\nFAILURE_BUDGET_OK=%s\nCONSECUTIVE_FAILURES=%q\n' "$GPU_BUDGET_OK" "$gpu_seconds" "$FAILURE_BUDGET_OK" "$failure_count"
    printf 'API_BUDGET_OK=%s\nAPI_SPEND_USD=%q\nGIT_OK=%s\nCODEX_AUTH_OK=%s\n' "$API_BUDGET_OK" "$api_spend" "$GIT_OK" "$CODEX_AUTH_OK"
    printf 'ARIS_AUDIT_OK=%s\nREVIEWER_RECEIPT_OK=%s\nAGENT_LIVE=%s\nEXPERIMENT_TMUX_LIVE=%s\n' "$ARIS_AUDIT_OK" "$REVIEWER_RECEIPT_OK" "$AGENT_LIVE" "$EXPERIMENT_TMUX_LIVE"
    printf 'EXPERIMENT_PROGRESS_OK=%s\nACTIVE_EXPERIMENT_WINDOWS=%q\nLATEST_PROGRESS_EPOCH=%q\n' "$EXPERIMENT_PROGRESS_OK" "$active_windows" "$latest_progress"
    printf 'FREE_KB=%q\nREASONS=%q\n' "$free_kb" "${reasons[*]:-none}"
  } >"$tmp"
  chmod 600 "$tmp"
  mv -f "$tmp" "$health_file"

  if test "$HEALTHY" -eq 1; then
    if test -f "$STOP_SENTINEL" && grep -Fx "$MANAGED_STOP_CONTENT" "$STOP_SENTINEL" >/dev/null 2>&1; then rm -f -- "$STOP_SENTINEL"; fi
    return 0
  fi
  if test ! -e "$STOP_SENTINEL"; then printf '%s\n' "$MANAGED_STOP_CONTENT" >"$STOP_SENTINEL"; chmod 600 "$STOP_SENTINEL"; fi
  return 1
}

while :; do
  if check_health; then status=0; else status=64; fi
  if test "$mode" = --once || test "$mode" = --gate-only; then exit "$status"; fi
  sleep "$SUPERVISOR_INTERVAL_SECONDS"
done
