#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZED_HOME=/home/qingchan
readonly EXPECTED_TECTONIC_VERSION='Tectonic 0.17.0'
readonly EXPECTED_TECTONIC_SHA256='a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da'
readonly MAIN_BODY_LIMIT=9

fail() { printf 'paper compilation gate refused: %s\n' "$*" >&2; exit 64; }
inside_home() { case "$(realpath -m -- "$1")/" in "$AUTHORIZED_HOME"/*) return 0;; *) return 1;; esac; }

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
set -a; source config/autoresearch.env; set +a

test "$(id -u)" -ne 0 || fail 'root is forbidden'
test "$HOME" = "$AUTHORIZED_HOME" || fail 'unexpected HOME'
test -n "${RUN_DIR:-}" || fail 'RUN_DIR is required from the immutable dispatcher'
[[ "${SOURCE_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || fail 'SOURCE_COMMIT is not a full SHA'
for path in "$repo_root" "$RUN_DIR" "$DATA_ROOT" "$STOP_SENTINEL"; do inside_home "$path" || fail "$path escapes authorized home"; done
test ! -e "$STOP_SENTINEL" || fail 'stop sentinel exists'
grep -Fx "SOURCE_COMMIT=$SOURCE_COMMIT" "$RUN_DIR/metadata.env" >/dev/null || fail 'snapshot source identity mismatch'

tectonic="$AUTHORIZED_HOME/.local/bin/tectonic"
setup_receipt="$DATA_ROOT/state/setup-receipt.tsv"
test -x "$tectonic" || fail 'tectonic is missing'
test "$("$tectonic" -V)" = "$EXPECTED_TECTONIC_VERSION" || fail 'tectonic version mismatch'
test "$(sha256sum "$tectonic" | awk '{print $1}')" = "$EXPECTED_TECTONIC_SHA256" || fail 'tectonic binary hash mismatch'
test -r "$setup_receipt" || fail 'setup receipt is missing'
grep -F $'TOOL\ttectonic\t' "$setup_receipt" >/dev/null || fail 'setup receipt does not cover tectonic'
grep -F "$EXPECTED_TECTONIC_SHA256" "$setup_receipt" >/dev/null || fail 'setup receipt has the wrong tectonic identity'

artifacts="$RUN_DIR/artifacts"
mkdir -p "$artifacts"
paper_dir="$repo_root/paper"
test -f "$paper_dir/main.tex" || fail 'paper/main.tex is missing'
test -f "$paper_dir/refs.bib" || fail 'paper/refs.bib is missing'

"$tectonic" -X show user-cache-dir >"$artifacts/tectonic-cache-dir.txt"
inside_home "$(<"$artifacts/tectonic-cache-dir.txt")" || fail 'tectonic cache escapes authorized home'

make -C "$paper_dir" ENGINE="$tectonic" main.pdf
test -s "$paper_dir/main.pdf" || fail 'compiler did not produce main.pdf'
test "$(stat -c %s "$paper_dir/main.pdf")" -gt 102400 || fail 'PDF is unexpectedly small'

cp "$paper_dir/main.pdf" "$artifacts/main.pdf"
cp "$paper_dir/main.log" "$artifacts/main.log"
for name in main.aux main.bbl main.blg main.out; do
  test ! -f "$paper_dir/$name" || cp "$paper_dir/$name" "$artifacts/$name"
done
pdfinfo "$artifacts/main.pdf" >"$artifacts/pdfinfo.txt"
pdffonts "$artifacts/main.pdf" >"$artifacts/pdffonts.txt"
pdftotext -layout "$artifacts/main.pdf" "$artifacts/main.txt"

python - "$artifacts" "$SOURCE_COMMIT" "$MAIN_BODY_LIMIT" "$EXPECTED_TECTONIC_VERSION" "$EXPECTED_TECTONIC_SHA256" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

artifacts = pathlib.Path(sys.argv[1])
source_commit = sys.argv[2]
main_body_limit = int(sys.argv[3])
tectonic_version = sys.argv[4]
tectonic_sha256 = sys.argv[5]

def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

pdfinfo = (artifacts / "pdfinfo.txt").read_text(encoding="utf-8")
pages_match = re.search(r"^Pages:\s+(\d+)$", pdfinfo, re.MULTILINE)
if pages_match is None:
    raise SystemExit("unable to determine PDF page count")
total_pages = int(pages_match.group(1))

text_pages = (artifacts / "main.txt").read_text(encoding="utf-8").split("\f")
if text_pages and not text_pages[-1].strip():
    text_pages.pop()

def heading_page(pattern: str) -> int:
    regex = re.compile(pattern, re.MULTILINE)
    matches = [index + 1 for index, page in enumerate(text_pages) if regex.search(page)]
    if len(matches) != 1:
        raise SystemExit(f"expected one heading matching {pattern!r}; found pages {matches}")
    return matches[0]

conclusion_page = heading_page(r"^\s*6\s+Conclusion\s*$")
references_page = heading_page(r"^\s*References\s*$")
appendix_page = heading_page(r"^\s*A\s+Registered protocol and provenance\s*$")
if not conclusion_page <= references_page <= appendix_page:
    raise SystemExit("section ordering is invalid")

# References beginning on page 10 means the main body occupies at most pages 1--9.
# If references share a page with the conclusion, that shared page counts toward the limit.
main_body_last_page = references_page if references_page <= main_body_limit else references_page - 1
within_page_limit = references_page <= main_body_limit + 1

font_lines = (artifacts / "pdffonts.txt").read_text(encoding="utf-8").splitlines()[2:]
font_rows = [line.split() for line in font_lines if line.strip()]
if not font_rows:
    raise SystemExit("PDF font inventory is empty")
unembedded = [row[0] for row in font_rows if len(row) < 4 or row[3].lower() != "yes"]

log = (artifacts / "main.log").read_text(encoding="utf-8", errors="replace")
undefined_references = len(re.findall(r"Reference .* undefined", log, re.IGNORECASE))
undefined_citations = len(re.findall(r"Citation .* undefined", log, re.IGNORECASE))
undefined_summary = len(re.findall(r"undefined references", log, re.IGNORECASE))
overfull = [float(value) for value in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log)]
max_overfull_pt = max(overfull, default=0.0)

pdf_path = artifacts / "main.pdf"
receipt = {
    "run": "evidence-locked-paper-compilation",
    "sourceCommit": source_commit,
    "compiler": {"version": tectonic_version, "sha256": tectonic_sha256},
    "pdf": {
        "path": str(pdf_path),
        "sha256": sha256(pdf_path),
        "sizeBytes": pdf_path.stat().st_size,
        "totalPages": total_pages,
        "conclusionStartsPage": conclusion_page,
        "referencesStartPage": references_page,
        "appendixStartsPage": appendix_page,
        "mainBodyLastPage": main_body_last_page,
        "mainBodyLimit": main_body_limit,
        "withinPageLimit": within_page_limit,
    },
    "validation": {
        "fontCount": len(font_rows),
        "allFontsEmbedded": not unembedded,
        "unembeddedFonts": unembedded,
        "undefinedReferences": undefined_references,
        "undefinedCitations": undefined_citations,
        "undefinedSummaryWarnings": undefined_summary,
        "overfullBoxCount": len(overfull),
        "maxOverfullPt": max_overfull_pt,
        "noSevereOverfullBoxes": max_overfull_pt <= 20.0,
    },
}
receipt["gatePass"] = all(
    [
        within_page_limit,
        not unembedded,
        undefined_references == 0,
        undefined_citations == 0,
        undefined_summary == 0,
        max_overfull_pt <= 20.0,
    ]
)
(artifacts / "compile-receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
if not receipt["gatePass"]:
    raise SystemExit("paper compilation validation failed")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
