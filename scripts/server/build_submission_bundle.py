#!/usr/bin/env python3
"""Build the minimal anonymous paper source archive for the submission gate."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import pathlib
import shutil
import tarfile

BUNDLE_ROOT = "submission-source"
SOURCE_FILES = (
    "LICENSE",
    "main.tex",
    "math_commands.tex",
    "refs.bib",
    "fancyhdr.sty",
    "natbib.sty",
    "iclr2025_conference.bst",
    "iclr2025_conference.sty",
    "sections/0_abstract.tex",
    "sections/1_introduction.tex",
    "sections/2_related_work.tex",
    "sections/3_protocol.tex",
    "sections/4_results.tex",
    "sections/5_discussion.tex",
    "sections/6_conclusion.tex",
    "sections/A_appendix.tex",
    "tables/table_decoding.tex",
    "tables/table_gates.tex",
    "tables/table_mechanisms.tex",
    "figures/fig1_evidence_ladder.pdf",
    "figures/fig2_dose_responses.pdf",
    "figures/fig3_direction_specificity.pdf",
)
GENERATED_FILES = ("MANIFEST.sha256", "REPRODUCE.md")
FORBIDDEN_IDENTITY_MARKERS = (
    b"qingchan",
    b"wy-coliney",
    b"asimov",
    b"uga.edu",
    b"/home/",
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source(source: pathlib.Path) -> None:
    for relative in SOURCE_FILES:
        path = source / relative
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"required regular source file is missing: {relative}")
        lowered = path.read_bytes().lower()
        marker = next((item for item in FORBIDDEN_IDENTITY_MARKERS if item in lowered), None)
        if marker is not None:
            raise SystemExit(
                f"anonymous source contains forbidden identity marker {marker.decode()!r}: {relative}"
            )
    main = (source / "main.tex").read_text(encoding="utf-8")
    if r"\author{Anonymous Authors}" not in main:
        raise SystemExit("main.tex does not declare Anonymous Authors")


def _reproduce_text(
    *, source_date_epoch: int, tectonic_version: str, tectonic_sha256: str, pdf_sha256: str
) -> str:
    return f"""# Reproduce the accepted paper PDF

Use Tectonic `{tectonic_version}` with binary SHA-256
`{tectonic_sha256}`. From this directory, run:

```bash
SOURCE_DATE_EPOCH={source_date_epoch} FORCE_SOURCE_DATE=1 tectonic -X compile main.tex --keep-logs
```

The resulting `main.pdf` must have SHA-256 `{pdf_sha256}`.
`MANIFEST.sha256` records every submitted source asset.
"""


def _write_archive(bundle: pathlib.Path, archive: pathlib.Path, epoch: int) -> None:
    members = sorted((*SOURCE_FILES, *GENERATED_FILES))
    with (
        archive.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT) as output,
    ):
        for relative in members:
            path = bundle / relative
            info = output.gettarinfo(os.fspath(path), arcname=f"{BUNDLE_ROOT}/{relative}")
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            info.mtime = epoch
            info.mode = 0o644
            with path.open("rb") as stream:
                output.addfile(info, stream)


def build(
    *,
    source: pathlib.Path,
    output: pathlib.Path,
    source_date_epoch: int,
    tectonic_version: str,
    tectonic_sha256: str,
    pdf_sha256: str,
) -> tuple[pathlib.Path, pathlib.Path]:
    _validate_source(source)
    if output.exists():
        raise SystemExit(f"bundle output already exists: {output}")
    output.mkdir(parents=True)
    bundle = output / BUNDLE_ROOT
    bundle.mkdir()
    for relative in SOURCE_FILES:
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)

    manifest = "".join(f"{sha256(bundle / item)}  {item}\n" for item in sorted(SOURCE_FILES))
    (bundle / "MANIFEST.sha256").write_text(manifest, encoding="utf-8", newline="\n")
    (bundle / "REPRODUCE.md").write_text(
        _reproduce_text(
            source_date_epoch=source_date_epoch,
            tectonic_version=tectonic_version,
            tectonic_sha256=tectonic_sha256,
            pdf_sha256=pdf_sha256,
        ),
        encoding="utf-8",
        newline="\n",
    )
    archive = output / "submission-source.tar.gz"
    _write_archive(bundle, archive, source_date_epoch)
    return bundle, archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    parser.add_argument("--tectonic-version", required=True)
    parser.add_argument("--tectonic-sha256", required=True)
    parser.add_argument("--pdf-sha256", required=True)
    args = parser.parse_args()
    bundle, archive = build(
        source=args.source,
        output=args.output,
        source_date_epoch=args.source_date_epoch,
        tectonic_version=args.tectonic_version,
        tectonic_sha256=args.tectonic_sha256,
        pdf_sha256=args.pdf_sha256,
    )
    print(f"bundle={bundle}")
    print(f"archive={archive}")
    print(f"archive_sha256={sha256(archive)}")


if __name__ == "__main__":
    main()
