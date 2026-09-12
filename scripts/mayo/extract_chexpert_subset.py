"""Extract only the frozen-manifest CheXpert Plus PNGs from the downloaded zip chunks; write images/.complete when done."""
import csv, os, sys, time, zipfile
ROOT = "/rodata/azradonc_dev/m253405/cf-transfer/data/chexpert"
ZIPS = f"{ROOT}/zips"
chunks = [f"png_chexpert_plus_chunk_{i}.zip" for i in range(5)]
while not all(os.path.exists(f"{ZIPS}/{z}.done") for z in chunks):
    time.sleep(120)
rows = list(csv.DictReader(open(f"{ROOT}/manifests/cohort.csv")))
need = {r["relative_image_path"] for r in rows}          # train/patientXXXXX/studyY/view1_frontal.png
have = {p for p in need if os.path.exists(f"{ROOT}/images/{p}")}
todo = need - have
print(f"need {len(need)} have {len(have)} todo {len(todo)}", flush=True)
t0 = time.time(); n = 0
for z in chunks:
    with zipfile.ZipFile(f"{ZIPS}/{z}") as zf:
        names = zf.namelist()
        if n == 0:
            print(z, "example members:", names[:3], flush=True)
        # members may carry a prefix (e.g. 'PNG/train/...' or 'train/...'); match on the trailing path
        by_tail = {}
        for m in names:
            for p in ("train/", ):
                k = m.find(p)
                if k >= 0:
                    by_tail[m[k:]] = m
                    break
        members = [(by_tail[p], p) for p in todo if p in by_tail]
        for m, p in members:
            dest = f"{ROOT}/images/{p}"
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(m) as src, open(dest + ".part", "wb") as f:
                f.write(src.read())
            os.replace(dest + ".part", dest); n += 1
        print(f"{z}: {len(members)} extracted (total {n}, {time.time()-t0:.0f}s)", flush=True)
missing = sorted(p for p in need if not os.path.exists(f"{ROOT}/images/{p}"))
print(f"final missing {len(missing)}", missing[:5], flush=True)
if not missing:
    open(f"{ROOT}/images/.complete", "w").write(time.strftime("%FT%TZ", time.gmtime()))
    print("CHEXPERT_EXTRACT_DONE", flush=True)
sys.exit(1 if missing else 0)
