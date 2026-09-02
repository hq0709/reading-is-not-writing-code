#!/usr/bin/env bash
set -euo pipefail
readonly AUTHORIZED_HOME=/home/qingchan
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
set -a; source config/autoresearch.env; set +a
fail() { echo "ARIS launch refused: $*" >&2; exit 64; }
test "$(id -u)" -ne 0 || fail 'root is forbidden'
test "$HOME" = "$AUTHORIZED_HOME" || fail 'unexpected HOME'
for path in "$repo_root" "$DATA_ROOT" "$ARIS_REPO" "$TMUX_SOCKET"; do case "$(realpath -m -- "$path")/" in "$AUTHORIZED_HOME"/*) ;; *) fail "$path escapes authorized home";; esac; done
if test -e "$STOP_SENTINEL" && ! grep -Fx 'AUTORESEARCH_SUPERVISOR_BLOCK_V1' "$STOP_SENTINEL" >/dev/null 2>&1; then
  fail 'stop sentinel exists'
fi
test -z "$(git status --porcelain --untracked-files=all)" || fail 'dirty source tree'
git fetch origin
head=$(git rev-parse HEAD)
test "$(git rev-parse '@{upstream}')" = "$head" || fail 'HEAD is not pushed'
test "$(git -C "$ARIS_REPO" rev-parse HEAD)" = "$ARIS_FULL_SHA" || fail 'ARIS SHA mismatch'
profile="$HOME/.codex/$CODEX_PROFILE.config.toml"
test "$(stat -c %a "$profile")" = 600 || fail 'profile missing or wrong mode'
grep -F "model = \"$CODEX_MODEL\"" "$profile" >/dev/null || fail 'profile model mismatch'
grep -F 'model_reasoning_effort = "high"' "$profile" >/dev/null || fail 'profile effort mismatch'
! grep -Eq 'service_tier|priority|fast' "$profile" || fail 'Fast/service tier override present'
"$HOME/.local/bin/codex" --strict-config --profile "$CODEX_PROFILE" --version >/dev/null
"$HOME/.local/bin/codex" login status >/dev/null || fail 'codex login status failed'
source scripts/server/activate_env.sh
python scripts/server/claude_review_adapter.py --probe >/dev/null || fail 'Claude reviewer probe failed'
audit="$DATA_ROOT/state/aris-audit.env"
test -f "$audit" && grep -Fx "ARIS_FULL_SHA=$ARIS_FULL_SHA" "$audit" >/dev/null || fail 'ARIS audit receipt missing'
audit_expect() { grep -Fx "$1=$2" "$audit" >/dev/null || fail "ARIS audit drift: $1"; }
audit_expect SKILL_LIST_SHA256 "$(sha256sum config/aris-skills.txt | cut -d ' ' -f1)"
audit_expect FORBIDDEN_LIST_SHA256 "$(sha256sum config/aris-forbidden-skills.txt | cut -d ' ' -f1)"
audit_expect REVIEWER_ROUTING_SHA256 "$(sha256sum config/reviewer-routing.tsv | cut -d ' ' -f1)"
audit_expect REVIEWER_WRAPPER_SHA256 "$(sha256sum scripts/server/claude_review_adapter.py | cut -d ' ' -f1)"
plugin_listing=$("$HOME/.local/bin/codex" plugin list)
test "$plugin_listing" = 'No marketplace plugins found.' || fail 'project launch refuses discovered plugins'
"$CONDA_ROOT/envs/$CONDA_ENV/bin/python" - "$CODEX_PROFILE" <<'PY'
import json
import subprocess
import sys

profile = sys.argv[1]
servers = json.loads(subprocess.check_output(
    ["/home/qingchan/.local/bin/codex", "--profile", profile, "mcp", "list", "--json"],
    text=True,
))
assert len(servers) == 1, servers
assert servers[0]["name"] == "claude-review-concept-flow", servers
assert servers[0]["enabled"] is True, servers
PY
scripts/server/supervisor.sh --gate-only || fail 'runtime health gate failed'
mkdir -p "$(dirname "$TMUX_SOCKET")" "$DATA_ROOT/logs"
if ! tmux -S "$TMUX_SOCKET" has-session -t "$EXPERIMENT_TMUX" 2>/dev/null; then
  tmux -S "$TMUX_SOCKET" new-session -d -s "$EXPERIMENT_TMUX" -n control -c "$SERVER_HOME" 'exec tail -f /dev/null'
fi
tmux -S "$TMUX_SOCKET" has-session -t "$AGENT_TMUX" 2>/dev/null && { echo "agent tmux already running: $AGENT_TMUX"; exit 0; }
log="$DATA_ROOT/logs/aris-$(date -u +%Y%m%dT%H%M%SZ)-${head:0:12}.log"
printf -v launch 'exec %q exec --profile %q --dangerously-bypass-approvals-and-sandbox -C %q - < %q' \
  "$HOME/.local/bin/codex" "$CODEX_PROFILE" "$repo_root" "$repo_root/config/aris-run-prompt.md"
tmux -S "$TMUX_SOCKET" new-session -d -s "$AGENT_TMUX" -c "$repo_root"
printf -v log_pipe 'cat >> %q' "$log"
tmux -S "$TMUX_SOCKET" pipe-pane -o -t "$AGENT_TMUX" "$log_pipe"
tmux -S "$TMUX_SOCKET" send-keys -t "$AGENT_TMUX" -l "$launch"
tmux -S "$TMUX_SOCKET" send-keys -t "$AGENT_TMUX" Enter
if ! tmux -S "$TMUX_SOCKET" list-windows -t "$EXPERIMENT_TMUX" -F '#{window_name}' | grep -Fx supervisor >/dev/null; then
  printf -v supervise 'exec %q' "$repo_root/scripts/server/supervisor.sh"
  tmux -S "$TMUX_SOCKET" new-window -d -t "$EXPERIMENT_TMUX" -n supervisor -c "$repo_root" "$supervise"
fi
printf 'started tmux %s; log %s\n' "$AGENT_TMUX" "$log"
