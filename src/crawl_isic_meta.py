"""Page the ISIC archive API and store every image's metadata as JSONL.

Only metadata, no pixels. The archive holds 553k images but the fields the validity battery needs are
sparse: about 57% carry an age, 60% a sex, 67% an anatomic site. Selecting a usable subset therefore has to
happen after seeing the whole index rather than by taking the first N, and the server side `query` parameter
is ignored by this endpoint, so filtering has to be done here.

Resumable: the next cursor is written beside the output after every page, so an interruption costs one page.
"""
from __future__ import os
import annotations
import json, os, sys, time, urllib.request, urllib.error

OUT = "data/isic/images.jsonl"
CUR = "data/isic/cursor.txt"
# The ISIC API asks for a contact address in the User-Agent. Set CONTACT_EMAIL to yours before
# crawling; the archive is entitled to know who is hitting it.
UA = ("concept-flow-research/1.0 (academic interpretability study; contact "
      + os.environ.get("CONTACT_EMAIL", "set-CONTACT_EMAIL") + ")")
BASE = "https://api.isic-archive.com/api/v2/images/?limit=100"


def get(url, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            wait = min(2 ** k, 60)
            print(f"  retry {k+1}/{tries} in {wait}s: {type(e).__name__} {str(e)[:80]}", flush=True)
            time.sleep(wait)
    raise SystemExit(f"gave up on {url}")


def main():
    nxt = open(CUR).read().strip() if os.path.exists(CUR) else BASE
    if nxt == "DONE":
        print("already complete"); return
    mode = "a" if os.path.exists(OUT) else "w"
    n = sum(1 for _ in open(OUT)) if mode == "a" else 0
    keys, t0, total = {}, time.time(), 0
    with open(OUT, mode) as fh:
        while nxt:
            j = get(nxt)
            total = j.get("count") or total   # only the first page carries a count
            for r in j["results"]:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
                for grp, d in r.get("metadata", {}).items():
                    for k in d:
                        keys[f"{grp}.{k}"] = keys.get(f"{grp}.{k}", 0) + 1
                for k in r:
                    if k not in ("metadata", "files"):
                        keys[f"top.{k}"] = keys.get(f"top.{k}", 0) + 1
                n += 1
            fh.flush()
            nxt = j.get("next")
            open(CUR, "w").write(nxt or "DONE")
            if (n // 100) % 20 == 0:
                rate = n / max(time.time() - t0, 1e-9)
                eta = (total - n) / max(rate, 1e-9) / 60 if total else float("nan")
                print(f"{n}/{total or '?'}  {rate:.0f} rec/s  eta {eta:.0f} min", flush=True)
            time.sleep(0.35)                      # be a polite client of a public archive
    print(f"\ndone: {n} records")
    print("fields ever seen:")
    for k, v in sorted(keys.items(), key=lambda x: -x[1]):
        print(f"  {v:8d}  {k}")


if __name__ == "__main__":
    main()
