"""Fetch the dermoscopy images named by the manifest, in parallel, resumably.

Files come from the archive's public S3 bucket, so the fetch is independent of the API and can run wide.
Skips anything already on disk, verifies each file opens as an image before counting it, and records
failures rather than leaving a truncated JPEG where a picture should be.

    python src/download_isic.py --manifest data/isic_manifest.csv --workers 16
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# The ISIC API asks for a contact address in the User-Agent. Set CONTACT_EMAIL to yours before
# crawling; the archive is entitled to know who is hitting it.
UA = ("concept-flow-research/1.0 (academic interpretability study; contact "
      + os.environ.get("CONTACT_EMAIL", "set-CONTACT_EMAIL") + ")")


def fetch(row, out_dir, tries=4):
    path = row["image_path"]
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return ("skip", row["row_id"], None)
    tmp = path + ".part"
    for k in range(tries):
        try:
            req = urllib.request.Request(row["url"], headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as fh:
                fh.write(r.read())
            from PIL import Image
            with Image.open(tmp) as im:
                im.verify()                       # a truncated jpeg is worse than a missing one
            os.replace(tmp, path)
            return ("ok", row["row_id"], None)
        except Exception as e:
            if k == tries - 1:
                if os.path.exists(tmp):
                    os.remove(tmp)
                return ("fail", row["row_id"], f"{type(e).__name__}: {str(e)[:120]}")
            time.sleep(min(2 ** k, 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/isic_manifest.csv")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest)))
    if args.limit:
        rows = rows[:args.limit]
    out_dir = os.path.dirname(rows[0]["image_path"])
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(rows)} images -> {out_dir}  ({args.workers} workers)")

    t0, done, fails = time.time(), {"ok": 0, "skip": 0, "fail": 0}, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, r, out_dir): r for r in rows}
        for i, f in enumerate(as_completed(futs), 1):
            status, rid, err = f.result()
            done[status] += 1
            if status == "fail":
                fails.append({"row_id": rid, "error": err})
            if i % 500 == 0:
                rate = i / max(time.time() - t0, 1e-9)
                print(f"  {i}/{len(rows)}  {rate:.0f} img/s  ok={done['ok']} skip={done['skip']} "
                      f"fail={done['fail']}  eta {(len(rows) - i) / max(rate, 1e-9) / 60:.0f} min", flush=True)

    json.dump({"n": len(rows), **done, "failures": fails[:200]},
              open(os.path.join(os.path.dirname(args.manifest), "isic_download_report.json"), "w"), indent=1)
    print(f"\ndone in {(time.time() - t0) / 60:.0f} min: ok={done['ok']} skip={done['skip']} fail={done['fail']}")
    if fails:
        print(f"first failures: {fails[:3]}")


if __name__ == "__main__":
    main()
