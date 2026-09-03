#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZED_HOME=/home/qingchan
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

fail() { printf 'TeX toolchain install refused: %s\n' "$*" >&2; exit 64; }
inside_home() { case "$(realpath -m -- "$1")/" in "$AUTHORIZED_HOME"/*) return 0;; *) return 1;; esac; }

test "$(id -u)" -ne 0 || fail 'root is forbidden'
test "$HOME" = "$AUTHORIZED_HOME" || fail 'unexpected HOME'
inside_home "$repo_root" || fail 'repository escapes authorized home'

spec="$repo_root/config/tex-toolchain.json"
test -r "$spec" || fail 'toolchain spec is missing'
mapfile -d '' -t values < <("$AUTHORIZED_HOME/miniforge3/envs/conceptflow/bin/python" - "$spec" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    spec = json.load(stream)
assert spec["component"] == "tectonic"
assert spec["version"] == "0.17.0"
assert spec["release_tag"] == "tectonic@0.17.0"
assert spec["build_command"] == ["tectonic", "-X", "compile", "main.tex", "--keep-logs"]
for value in (
    spec["archive"]["url"],
    spec["archive"]["sha256"],
    spec["install_path"],
    spec["cache_root"],
):
    sys.stdout.buffer.write(value.encode() + b"\0")
PY
)
test "${#values[@]}" -eq 4 || fail 'unexpected toolchain spec shape'
archive_url=${values[0]}
archive_sha256=${values[1]}
install_path=${values[2]}
cache_root=${values[3]}

[[ "$archive_url" == https://github.com/tectonic-typesetting/tectonic/releases/download/* ]] || fail 'unexpected release URL'
[[ "$archive_sha256" =~ ^[0-9a-f]{64}$ ]] || fail 'invalid archive SHA-256'
inside_home "$install_path" || fail 'install path escapes authorized home'
inside_home "$cache_root" || fail 'cache path escapes authorized home'
test "$(dirname "$install_path")" = "$AUTHORIZED_HOME/.local/bin" || fail 'unexpected install directory'

mkdir -p "$AUTHORIZED_HOME/.cache/tmp" "$(dirname "$install_path")" "$cache_root"
tmp_dir=$(mktemp -d "$AUTHORIZED_HOME/.cache/tmp/tectonic-install.XXXXXX")
trap 'rm -rf -- "$tmp_dir"' EXIT
archive="$tmp_dir/tectonic.tar.gz"
curl --proto '=https' --tlsv1.2 -fL "$archive_url" -o "$archive"
printf '%s  %s\n' "$archive_sha256" "$archive" | sha256sum -c -
test "$(tar -tzf "$archive" | sed '/\/$/d' | wc -l)" -eq 1 || fail 'release archive has unexpected contents'
test "$(tar -tzf "$archive" | sed '/\/$/d')" = tectonic || fail 'release archive does not contain tectonic'
tar -xzf "$archive" -C "$tmp_dir" tectonic
test -f "$tmp_dir/tectonic" || fail 'tectonic executable was not extracted'

install_tmp=$(mktemp "$AUTHORIZED_HOME/.local/bin/.tectonic.XXXXXX")
trap 'rm -rf -- "$tmp_dir"; rm -f -- "$install_tmp"' EXIT
cp "$tmp_dir/tectonic" "$install_tmp"
chmod 755 "$install_tmp"
"$install_tmp" --help >/dev/null
test "$("$install_tmp" -V)" = 'Tectonic 0.17.0' || fail 'unexpected tectonic version'
"$install_tmp" -X compile --help >/dev/null
mv -f "$install_tmp" "$install_path"
trap 'rm -rf -- "$tmp_dir"' EXIT

printf 'installed=%s\n' "$install_path"
printf 'archive_sha256=%s\n' "$archive_sha256"
printf 'binary_sha256=%s\n' "$(sha256sum "$install_path" | awk '{print $1}')"
printf 'version=%s\n' "$("$install_path" -V)"
