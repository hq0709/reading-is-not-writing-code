"""Extract exactly the NIH ChestX-ray14 PNGs that cf-transfer-v1 needs from the official zips.

Needed rows = every `train` row of data/manifest.csv (probe fitting) + the 1,016 rows of
docs/external-replication/cohorts.csv (preflight/calibration/test). Nothing else is unpacked.
"""
import csv, os, sys, zipfile, time
REPO = "/rodata/azradonc_dev/m253405/concept-flow"
ZIPS = "/rodata/azradonc_dev/m253405/cache/nih/zips"
OUT = "/rodata/azradonc_dev/m253405/cache/nih/images"
rows = list(csv.DictReader(open(f"{REPO}/data/manifest.csv")))
need = {r["row_id"] + ".png" for r in rows if r["split"] == "train"}
need |= {r["row_id"] + ".png" for r in csv.DictReader(open(f"{REPO}/docs/external-replication/cohorts.csv"))}
have = set(os.listdir(OUT))
todo = need - have
print(f"need {len(need)} have {len(have & need)} todo {len(todo)}", flush=True)
t0 = time.time(); n = 0
for z in sorted(os.listdir(ZIPS)):
    if not z.endswith(".zip"):
        continue
    with zipfile.ZipFile(os.path.join(ZIPS, z)) as zf:
        members = [m for m in zf.namelist() if os.path.basename(m) in todo]
        for m in members:
            data = zf.read(m)
            tmp = os.path.join(OUT, os.path.basename(m) + ".part")
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, os.path.join(OUT, os.path.basename(m)))
            n += 1
        print(f"{z}: {len(members)} extracted (total {n}, {time.time()-t0:.0f}s)", flush=True)
have = set(os.listdir(OUT))
missing = sorted(need - have)
print(f"final: have {len(need & have)}/{len(need)}; missing {len(missing)}", flush=True)
if missing:
    print("MISSING:", missing[:20])
    sys.exit(1)
print("NIH_EXTRACT_DONE")
