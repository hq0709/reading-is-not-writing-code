#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZED_HOME=/home/qingchan
readonly SOURCE_RUN_ID=20260903T123856Z-00f0adc-paper-compile
readonly SOURCE_PAPER_COMMIT=00f0adc32a4ac36dba3d0f21feca485f1c8ef972
readonly SOURCE_DATE_EPOCH=1788439147
readonly EXPECTED_PDF_SHA256=e88c7f9a42baa4f04c7ccdbfdf9653501e4f96a81bd1c1ee238163671e5106e2
readonly EXPECTED_COMPILE_RECEIPT_SHA256=f97889371ffe04aa710f894a3a4d6902b221dc351e2659837d01d230a70014e8
readonly EXPECTED_TECTONIC_VERSION='Tectonic 0.17.0'
readonly EXPECTED_TECTONIC_SHA256=a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da

fail() { printf 'submission bundle gate refused: %s\n' "$*" >&2; exit 64; }
inside_home() { case "$(realpath -m -- "$1")/" in "$AUTHORIZED_HOME"/*) return 0;; *) return 1;; esac; }

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
set -a; source config/autoresearch.env; set +a

test "$(id -u)" -ne 0 || fail 'root is forbidden'
test "$HOME" = "$AUTHORIZED_HOME" || fail 'unexpected HOME'
test -n "${RUN_DIR:-}" || fail 'RUN_DIR is required from the immutable dispatcher'
[[ "${SOURCE_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || fail 'SOURCE_COMMIT is not a full SHA'
for path in "$repo_root" "$RUN_DIR" "$DATA_ROOT" "$STOP_SENTINEL"; do inside_home "$path" || fail "$path escapes authorized home"; done
test "$repo_root" = "$RUN_DIR/source" || fail 'RUN_DIR does not match archived source'
test ! -e "$STOP_SENTINEL" || fail 'stop sentinel exists'
grep -Fx "SOURCE_COMMIT=$SOURCE_COMMIT" "$RUN_DIR/metadata.env" >/dev/null || fail 'snapshot source identity mismatch'

source_run="$DATA_ROOT/runs/$SOURCE_RUN_ID"
source_paper="$source_run/source/paper"
source_pdf="$source_run/artifacts/main.pdf"
compile_receipt="$source_run/artifacts/compile-receipt.json"
for path in "$source_run" "$source_paper" "$source_pdf" "$compile_receipt"; do inside_home "$path" || fail "$path escapes authorized home"; done
test -d "$source_paper" || fail 'accepted paper source snapshot is missing'
test -f "$source_pdf" || fail 'accepted PDF is missing'
test -f "$compile_receipt" || fail 'accepted compile receipt is missing'
test "$(<"$source_run/command_exit_status")" = 0 || fail 'accepted compile command did not pass'
test "$(<"$source_run/exit_status")" = 0 || fail 'accepted compile dispatcher did not pass'
grep -Fx "SOURCE_COMMIT=$SOURCE_PAPER_COMMIT" "$source_run/metadata.env" >/dev/null || fail 'accepted paper source commit mismatch'
grep -Fx "$EXPECTED_COMPILE_RECEIPT_SHA256  ./artifacts/compile-receipt.json" "$source_run/SHA256SUMS" >/dev/null || fail 'compile receipt terminal checksum mismatch'
grep -Fx "$EXPECTED_PDF_SHA256  ./artifacts/main.pdf" "$source_run/SHA256SUMS" >/dev/null || fail 'accepted PDF terminal checksum mismatch'
test "$(sha256sum "$compile_receipt" | awk '{print $1}')" = "$EXPECTED_COMPILE_RECEIPT_SHA256" || fail 'compile receipt content mismatch'
test "$(sha256sum "$source_pdf" | awk '{print $1}')" = "$EXPECTED_PDF_SHA256" || fail 'accepted PDF content mismatch'

tectonic="$AUTHORIZED_HOME/.local/bin/tectonic"
test -x "$tectonic" || fail 'tectonic is missing'
test "$("$tectonic" -V)" = "$EXPECTED_TECTONIC_VERSION" || fail 'tectonic version mismatch'
test "$(sha256sum "$tectonic" | awk '{print $1}')" = "$EXPECTED_TECTONIC_SHA256" || fail 'tectonic binary hash mismatch'

python - "$compile_receipt" "$SOURCE_PAPER_COMMIT" "$EXPECTED_PDF_SHA256" "$EXPECTED_TECTONIC_VERSION" "$EXPECTED_TECTONIC_SHA256" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
assert receipt["gatePass"] is True
assert receipt["sourceCommit"] == sys.argv[2]
assert receipt["pdf"]["sha256"] == sys.argv[3]
assert receipt["compiler"]["version"] == sys.argv[4]
assert receipt["compiler"]["sha256"] == sys.argv[5]
PY

artifacts="$RUN_DIR/artifacts"
mkdir -p "$artifacts"
python scripts/server/build_submission_bundle.py \
  --source "$source_paper" \
  --output "$artifacts/bundle" \
  --source-date-epoch "$SOURCE_DATE_EPOCH" \
  --tectonic-version "$EXPECTED_TECTONIC_VERSION" \
  --tectonic-sha256 "$EXPECTED_TECTONIC_SHA256" \
  --pdf-sha256 "$EXPECTED_PDF_SHA256" \
  >"$artifacts/bundle-build.txt"

cp "$source_pdf" "$artifacts/accepted-main.pdf"
mkdir -p "$artifacts/reproduction"
tar -xzf "$artifacts/bundle/submission-source.tar.gz" -C "$artifacts/reproduction"
reproduction="$artifacts/reproduction/submission-source"
(cd "$reproduction" && \
  SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" FORCE_SOURCE_DATE=1 \
  "$tectonic" -X compile main.tex --keep-logs \
    >"$artifacts/reproduction-stdout.log" 2>"$artifacts/reproduction-stderr.log")
test -s "$reproduction/main.pdf" || fail 'source archive did not produce a PDF'
reproduced_sha=$(sha256sum "$reproduction/main.pdf" | awk '{print $1}')
test "$reproduced_sha" = "$EXPECTED_PDF_SHA256" || fail 'source archive PDF is not byte-identical to the accepted PDF'
(cd "$reproduction" && sha256sum -c MANIFEST.sha256) >"$artifacts/manifest-validation.txt"

pdfinfo "$reproduction/main.pdf" >"$artifacts/reproduced-pdfinfo.txt"
pdffonts "$reproduction/main.pdf" >"$artifacts/reproduced-pdffonts.txt"
pdftotext -layout "$reproduction/main.pdf" "$artifacts/reproduced-main.txt"

python - "$artifacts" "$SOURCE_COMMIT" "$SOURCE_RUN_ID" "$SOURCE_PAPER_COMMIT" "$SOURCE_DATE_EPOCH" "$EXPECTED_PDF_SHA256" <<'PY'
import hashlib
import json
import pathlib
import re
import sys
import tarfile

artifacts = pathlib.Path(sys.argv[1])
source_commit, source_run_id, paper_commit = sys.argv[2:5]
source_date_epoch = int(sys.argv[5])
expected_pdf_sha = sys.argv[6]
archive = artifacts / "bundle/submission-source.tar.gz"
accepted_pdf = artifacts / "accepted-main.pdf"
reproduced_pdf = artifacts / "reproduction/submission-source/main.pdf"

def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

with tarfile.open(archive, "r:gz") as bundle:
    members = bundle.getmembers()
    names = [member.name for member in members]
    regular_only = all(member.isfile() for member in members)
    safe_paths = all(
        name.startswith("submission-source/") and ".." not in pathlib.PurePosixPath(name).parts
        for name in names
    )

main_tex = (artifacts / "reproduction/submission-source/main.tex").read_text(encoding="utf-8")
pdf_text = (artifacts / "reproduced-main.txt").read_text(encoding="utf-8", errors="replace")
anonymous_source = r"\author{Anonymous Authors}" in main_tex
identity_patterns = (r"qingchan", r"wy-coliney", r"asimov", r"uga\.edu", r"/home/")
identity_hits = [pattern for pattern in identity_patterns if re.search(pattern, main_tex + pdf_text, re.I)]
accepted_sha = sha256(accepted_pdf)
reproduced_sha = sha256(reproduced_pdf)
receipt = {
    "run": "anonymous-submission-bundle",
    "sourceCommit": source_commit,
    "acceptedCompilation": {
        "runId": source_run_id,
        "sourceCommit": paper_commit,
        "pdfSha256": accepted_sha,
    },
    "archive": {
        "path": str(archive),
        "sha256": sha256(archive),
        "sizeBytes": archive.stat().st_size,
        "memberCount": len(members),
        "regularFilesOnly": regular_only,
        "safePaths": safe_paths,
    },
    "reproduction": {
        "sourceDateEpoch": source_date_epoch,
        "pdfSha256": reproduced_sha,
        "byteIdenticalToAcceptedPdf": reproduced_sha == accepted_sha == expected_pdf_sha,
    },
    "anonymity": {
        "anonymousAuthorDeclaration": anonymous_source,
        "identityMarkerHits": identity_hits,
    },
}
receipt["gatePass"] = all(
    [
        regular_only,
        safe_paths,
        reproduced_sha == accepted_sha == expected_pdf_sha,
        anonymous_source,
        not identity_hits,
    ]
)
(artifacts / "submission-bundle-receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
if not receipt["gatePass"]:
    raise SystemExit("submission bundle validation failed")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
