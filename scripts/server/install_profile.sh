#!/usr/bin/env bash
set -euo pipefail
readonly AUTHORIZED_HOME=/home/qingchan
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
set -a; source config/autoresearch.env; set +a
test "$(id -u)" -ne 0 && test "$HOME" = "$AUTHORIZED_HOME"
chmod 700 "$repo_root/scripts/server/claude_review_adapter.py"
profile="$HOME/.codex/$CODEX_PROFILE.config.toml"
mkdir -p "$HOME/.codex"
umask 077
tmp=$(mktemp "$HOME/.codex/.${CODEX_PROFILE}.config.XXXXXX")
trap 'rm -f -- "$tmp"' EXIT
{
  printf 'model = "%s"\n' "$CODEX_MODEL"
  printf 'model_reasoning_effort = "%s"\n' "$CODEX_REASONING_EFFORT"
  printf 'approval_policy = "never"\n'
  printf 'sandbox_mode = "danger-full-access"\n\n'
  printf '[apps._default]\nenabled = false\n\n'
  printf '[mcp_servers."%s"]\n' "$REVIEW_TRANSPORT"
  printf 'command = "%s"\n' "$CONDA_ROOT/envs/$CONDA_ENV/bin/python"
  printf 'args = ["%s/scripts/server/claude_review_adapter.py"]\n' "$SERVER_REPO"
  printf 'enabled = true\nrequired = true\nenabled_tools = ["review"]\n'
  printf 'startup_timeout_sec = 10\ntool_timeout_sec = 1800\n'
  printf 'env = { CLAUDE_REVIEW_MODEL = "%s", CLAUDE_CODE_EFFORT_LEVEL = "medium" }\n' "$CLAUDE_REVIEW_MODEL"
} >"$tmp"
chmod 600 "$tmp"
mv -f "$tmp" "$profile"
trap - EXIT
test "$(stat -c %a "$profile")" = 600
"$HOME/.local/bin/codex" --strict-config --profile "$CODEX_PROFILE" --version
