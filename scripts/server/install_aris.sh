#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZED_HOME=/home/qingchan
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
set -a; source config/autoresearch.env; set +a
fail() { echo "ARIS install refused: $*" >&2; exit 64; }
test "$(id -u)" -ne 0 || fail 'root is forbidden'
test "$HOME" = "$AUTHORIZED_HOME" || fail 'unexpected HOME'
for path in "$repo_root" "$ARIS_REPO" "$DATA_ROOT"; do
  case "$(realpath -m -- "$path")/" in "$AUTHORIZED_HOME"/*) ;; *) fail "$path escapes authorized home";; esac
done
test "$(git -C "$ARIS_REPO" rev-parse HEAD)" = "$ARIS_FULL_SHA" || fail 'ARIS SHA mismatch'
test -z "$(git status --porcelain --untracked-files=all)" || fail 'dirty source tree'

list_csv() {
  awk '
    { sub(/[[:space:]]*#.*/, ""); gsub(/^[[:space:]]+|[[:space:]]+$/, "") }
    $0 != "" && !seen[$0]++ { out = out (out == "" ? "" : ",") $0 }
    END { print out }
  ' "$1"
}

skills_csv=$(list_csv config/aris-skills.txt)
forbidden_csv=$(list_csv config/aris-forbidden-skills.txt)
test -z "$ARIS_APPROVED_GROUPS" || fail 'whole ARIS groups are forbidden for this project'
test -n "$skills_csv" || fail 'ARIS allowlist is empty'
test "$skills_csv" = "$ARIS_APPROVED_EXTRA_SKILLS" || fail 'approved skill list/config drift'
[[ "$skills_csv" =~ ^[A-Za-z0-9._-]+(,[A-Za-z0-9._-]+)*$ ]] || fail 'invalid skill name'
[[ "$forbidden_csv" =~ ^[A-Za-z0-9._-]+(,[A-Za-z0-9._-]+)*$ ]] || fail 'invalid forbidden name'

aris_args=(
  "$repo_root"
  --aris-repo "$ARIS_REPO"
  --with-claude-review-overlay
  --skills "$skills_csv"
  --exclude "$forbidden_csv"
  --quiet
  --no-doc
)
ARIS_NO_PICKER=1 bash "$ARIS_REPO/tools/install_aris_codex.sh" "${aris_args[@]}" --dry-run
test "${1:-}" = --apply || { echo 'dry-run complete; pass --apply to reconcile and audit'; exit 0; }
if test -s "$repo_root/.aris/installed-skills-codex.txt"; then
  ARIS_NO_PICKER=1 bash "$ARIS_REPO/tools/install_aris_codex.sh" "$repo_root" \
    --aris-repo "$ARIS_REPO" --uninstall --quiet --no-doc
fi
ARIS_NO_PICKER=1 bash "$ARIS_REPO/tools/install_aris_codex.sh" "${aris_args[@]}"

manifest="$repo_root/.aris/installed-skills-codex.txt"
test -s "$manifest" || fail 'installer manifest missing'
aris_real=$(realpath -e -- "$ARIS_REPO")
installed=0
while IFS=$'\t' read -r kind name source_rel target_rel mode; do
  case "$kind" in skill|support) ;; *) continue;; esac
  target="$repo_root/$target_rel"
  test "$mode" = symlink && test -L "$target" || fail "invalid managed entry: $name"
  resolved=$(realpath -e -- "$target")
  case "$resolved/" in "$aris_real"/*) ;; *) fail "skill escapes pinned ARIS repo: $name";; esac
  installed=$((installed + 1))
done < "$manifest"
test "$installed" -gt 0 || fail 'empty installed dependency closure'
grep -F $'skills-codex-claude-review/' "$manifest" >/dev/null || fail 'Claude reviewer overlay missing'
installed_csv=$(awk -F '\t' '$1 == "skill" || $1 == "support" { print $2 }' "$manifest" | sort -u | paste -sd, -)
approved_csv=$(printf '%s\n' "$skills_csv,shared-references" | tr ',' '\n' | sort -u | paste -sd, -)
test "$installed_csv" = "$approved_csv" || fail 'installed dependency closure exceeds exact allowlist'
while IFS= read -r name; do
  test ! -e "$repo_root/.agents/skills/$name" || fail "forbidden skill installed: $name"
done < <(printf '%s\n' "$forbidden_csv" | tr ',' '\n')

state_dir="$DATA_ROOT/state"
mkdir -p "$state_dir"
umask 077
receipt=$(mktemp "$state_dir/.aris-audit.XXXXXX")
trap 'rm -f -- "$receipt"' EXIT
{
  printf 'AUDITED_UTC=%q\n' "$(date -u +%FT%TZ)"
  printf 'ARIS_FULL_SHA=%s\n' "$ARIS_FULL_SHA"
  printf 'ARIS_APPROVED_SKILLS=%s\n' "$skills_csv"
  printf 'ARIS_FORBIDDEN_SKILLS=%s\n' "$forbidden_csv"
  printf 'INSTALLED_ENTRY_COUNT=%s\n' "$installed"
  printf 'MANIFEST_SHA256=%s\n' "$(sha256sum "$manifest" | cut -d ' ' -f1)"
  printf 'SKILL_LIST_SHA256=%s\n' "$(sha256sum config/aris-skills.txt | cut -d ' ' -f1)"
  printf 'FORBIDDEN_LIST_SHA256=%s\n' "$(sha256sum config/aris-forbidden-skills.txt | cut -d ' ' -f1)"
  printf 'REVIEWER_ROUTING_SHA256=%s\n' "$(sha256sum config/reviewer-routing.tsv | cut -d ' ' -f1)"
  printf 'REVIEWER_WRAPPER_SHA256=%s\n' "$(sha256sum scripts/server/claude_review_adapter.py | cut -d ' ' -f1)"
} > "$receipt"
chmod 600 "$receipt"
mv -f "$receipt" "$state_dir/aris-audit.env"
trap - EXIT
printf 'audited %s ARIS entries at %s\n' "$installed" "$state_dir/aris-audit.env"
