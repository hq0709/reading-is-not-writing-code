"""Downscale the dermoscopy images once, so five extraction passes do not each decode a 29-megapixel JPEG.

The archive stores originals at up to 6642x4421. Every model resizes to its own fixed budget anyway (144
visual tokens for the Qwen family, 576 for LLaVA), so decoding at full resolution is work thrown away, and
it dominated the extraction: 2.7 images per second against 11 for the chest radiographs, which are already
small. Capping the long side at 1024 leaves far more detail than any of these encoders consumes.

Every model reads the identical resized file, so no comparison in the paper is affected by the choice.
"""
from __future__ import annotations
import csv, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

MAX_SIDE = 1024


def one(args):
    src, dst = args
    if os.path.exists(dst) and os.path.getsize(dst) > 512:
        return "skip"
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            if max(im.size) > MAX_SIDE:
                s = MAX_SIDE / max(im.size)
                im = im.resize((max(int(im.width * s), 1), max(int(im.height * s), 1)), Image.LANCZOS)
            im.save(dst, "JPEG", quality=95)
        return "ok"
    except Exception as e:
        return f"fail {src}: {type(e).__name__}"


def main():
    man = sys.argv[1] if len(sys.argv) > 1 else "data/isic_manifest.csv"
    rows = list(csv.DictReader(open(man)))
    out_dir = "data/isic/images_1024"
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(r["image_path"], os.path.join(out_dir, r["row_id"] + ".jpg")) for r in rows]
    counts = {}
    with ProcessPoolExecutor(max_workers=32) as ex:
        for i, f in enumerate(as_completed([ex.submit(one, j) for j in jobs]), 1):
            r = f.result(); counts[r.split()[0]] = counts.get(r.split()[0], 0) + 1
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)} {counts}", flush=True)
    print(f"done: {counts}")

    for r in rows:
        r["image_path"] = os.path.join(out_dir, r["row_id"] + ".jpg")
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"manifest {man} now points at {out_dir}")


if __name__ == "__main__":
    main()
