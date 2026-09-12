"""Extract only the COCO images the frozen manifest uses (20,416 train2017 + 600 val2017) from the official zips."""
import csv, os, sys, time, zipfile
ROOT = "/rodata/azradonc_dev/m253405/cf-transfer/data/coco"
while not os.path.exists(f"{ROOT}/zips/train2017.zip.done") or not os.path.exists(f"{ROOT}/zips/val2017.zip.done"):
    time.sleep(30)
rows = list(csv.DictReader(open(f"{ROOT}/manifests/cohort.csv")))
need = {r["relative_image_path"] for r in rows}
have = {p for p in need if os.path.exists(f"{ROOT}/{p}")}
todo = need - have
print(f"need {len(need)} have {len(have)} todo {len(todo)}", flush=True)
t0 = time.time(); n = 0
for z in ("val2017.zip", "train2017.zip"):
    with zipfile.ZipFile(f"{ROOT}/zips/{z}") as zf:
        members = [m for m in zf.namelist() if m in todo]
        for m in members:
            os.makedirs(os.path.dirname(f"{ROOT}/{m}"), exist_ok=True)
            with open(f"{ROOT}/{m}.part", "wb") as f:
                f.write(zf.read(m))
            os.replace(f"{ROOT}/{m}.part", f"{ROOT}/{m}")
            n += 1
        print(f"{z}: {len(members)} extracted ({time.time()-t0:.0f}s)", flush=True)
missing = sorted(p for p in need if not os.path.exists(f"{ROOT}/{p}"))
print(f"final missing {len(missing)}", missing[:5], flush=True)
sys.exit(1 if missing else 0)
