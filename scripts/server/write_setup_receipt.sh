#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZED_HOME=/home/qingchan
readonly PROVENANCE_REL=config/toolchain-provenance.tsv

fail() {
  echo "setup receipt refused: $*" >&2
  exit 64
}

test "$(id -u)" -ne 0 || fail 'root is forbidden'
test "$HOME" = "$AUTHORIZED_HOME" || fail 'unexpected HOME'

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
set -a
source config/autoresearch.env
set +a

for path in "$repo_root" "$DATA_ROOT" "$ARIS_REPO"; do
  case "$(realpath -m -- "$path")/" in
    "$AUTHORIZED_HOME"/*) ;;
    *) fail "$path escapes authorized home" ;;
  esac
done
test -z "$(git status --porcelain --untracked-files=all)" || fail 'dirty source tree'
test "$(git -C "$ARIS_REPO" rev-parse HEAD)" = "$ARIS_FULL_SHA" || fail 'ARIS SHA mismatch'

provenance="$repo_root/$PROVENANCE_REL"
test -r "$provenance" || fail 'toolchain provenance manifest is missing'
expected_header=$'component\tofficial_source_url\tinstaller_sha256\tbinary_home_relative_path\texpected_version\texpected_binary_sha256\tsanitized_install_command'
IFS= read -r actual_header < "$provenance"
test "$actual_header" = "$expected_header" || fail 'unexpected provenance manifest schema'

state_dir="$DATA_ROOT/state"
case "$(realpath -m -- "$state_dir")/" in
  "$AUTHORIZED_HOME"/*) ;;
  *) fail 'state directory escapes authorized home' ;;
esac
mkdir -p "$state_dir"
umask 077
receipt=$(mktemp "$state_dir/.setup-receipt.XXXXXX")
trap 'rm -f -- "$receipt"' EXIT

generated_utc=$(date -u +%FT%TZ)
host=$(hostname -f 2>/dev/null || hostname)
source_commit=$(git rev-parse HEAD)
manifest_sha256=$(sha256sum "$provenance" | awk '{print $1}')

{
  printf 'record_type\tcomponent_or_key\tofficial_source_or_value\tinstaller_sha256\tinstalled_path\tinstalled_version\tinstalled_sha256\tsanitized_install_command\n'
  printf 'META\tgenerated_utc\t%s\t-\t-\t-\t-\t-\n' "$generated_utc"
  printf 'META\thost\t%s\t-\t-\t-\t-\t-\n' "$host"
  printf 'META\tsource_commit\t%s\t-\t-\t-\t-\t-\n' "$source_commit"
  printf 'META\tmanifest_sha256\t%s\t-\t-\t-\t-\t-\n' "$manifest_sha256"
  printf 'SOURCE\taris\thttps://github.com/wanshuiyin/Auto-claude-code-research-in-sleep\t-\t%s\t%s\t-\tgit -C "$HOME/aris_repo" checkout --detach %s\n' \
    "$ARIS_REPO" "$ARIS_FULL_SHA" "$ARIS_FULL_SHA"

  while IFS=$'\t' read -r component source_url installer_sha binary_rel expected_version expected_binary_sha install_command; do
    test -n "$component" || fail 'empty component in provenance manifest'
    [[ "$source_url" == https://* ]] || fail "non-HTTPS source for $component"
    [[ "$installer_sha" =~ ^[0-9a-f]{64}$ ]] || fail "invalid installer hash for $component"
    [[ "$expected_binary_sha" =~ ^[0-9a-f]{64}$ ]] || fail "invalid binary hash for $component"
    [[ "$binary_rel" != /* ]] || fail "absolute binary path for $component"
    binary="$AUTHORIZED_HOME/$binary_rel"
    resolved=$(realpath -e -- "$binary") || fail "missing binary for $component"
    case "$resolved/" in
      "$AUTHORIZED_HOME"/*) ;;
      *) fail "binary escapes authorized home: $component" ;;
    esac

    case "$component" in
      miniforge-conda) installed_version=$("$binary" --version) ;;
      uv) installed_version=$("$binary" --version) ;;
      uvx) installed_version=$("$binary" --version) ;;
      codex) installed_version=$("$binary" --version) ;;
      claude) installed_version=$("$binary" --version) ;;
      *) fail "unknown component: $component" ;;
    esac
    installed_version=$(printf '%s' "$installed_version" | tr '\t\r\n' '   ')
    [[ "$installed_version" == *"$expected_version"* ]] || fail "version mismatch for $component"
    installed_sha=$(sha256sum "$binary" | awk '{print $1}')
    test "$installed_sha" = "$expected_binary_sha" || fail "binary hash mismatch for $component"
    [[ "$install_command" != *$'\n'* && "$install_command" != *$'\r'* ]] || fail "invalid command for $component"
    printf 'TOOL\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$component" "$source_url" "$installer_sha" "$binary" "$installed_version" "$installed_sha" "$install_command"
  done < <(tail -n +2 "$provenance")
} > "$receipt"

chmod 600 "$receipt"
mv -f "$receipt" "$state_dir/setup-receipt.tsv"
trap - EXIT
test "$(stat -c '%a' "$state_dir/setup-receipt.tsv")" = 600 || fail 'receipt mode is not 600'
printf '%s\n' "$state_dir/setup-receipt.tsv"
