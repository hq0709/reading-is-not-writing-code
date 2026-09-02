#!/usr/bin/env bash
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
mirror="$repo_root/paper-overleaf"
test -d "$mirror/.git" || { echo 'Overleaf clone is not configured (CONDITIONAL N/A)' >&2; exit 2; }
test ! -L "$mirror" || { echo 'Overleaf clone must not be a symlink' >&2; exit 64; }
case "$(realpath -m -- "$mirror")/" in "$repo_root"/*) ;; *) echo 'Overleaf clone escapes repository root' >&2; exit 64;; esac
mode=${1:---dry-run}
test "$mode" = --dry-run || test "$mode" = --apply
files=(main.tex refs.bib figures tables)
for item in "${files[@]}"; do
  test -e "$repo_root/paper/$item" || continue
  if test "$mode" = --dry-run; then
    diff -qr "$repo_root/paper/$item" "$mirror/$item" || true
  else
    rm -rf -- "$mirror/$item"
    cp -a -- "$repo_root/paper/$item" "$mirror/$item"
  fi
done
test "$mode" = --dry-run || git -C "$mirror" diff --stat
