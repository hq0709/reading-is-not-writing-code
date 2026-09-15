"""Extract the `valid`-role PNGs of the frozen CheXpert manifest (members PNG/valid/... of the CheXpert Plus zip chunks) into
images/valid/...; size checked against the central directory and CRC checked by zipfile on read; idempotent (present
files are skipped); writes images/valid/.complete, the marker the VALID queue tasks require."""
import csv, os, sys, time, zipfile
ROOT = "/rodata/azradonc_dev/m253405/cf-transfer/data/chexpert"
ZIPS = f"{ROOT}/zips"
chunks = [f"png_chexpert_plus_chunk_{i}.zip" for i in range(5)]
rows = [r for r in csv.DictReader(open(f"{ROOT}/manifests/cohort.csv")) if r["role"] == "valid"]
need = {r["relative_image_path"] for r in rows}          # valid/patientNNNNN/studyN/viewN_frontal.png
if not need:
    sys.exit("cohort.csv has no valid rows: run  python -m cftransfer.manifests chexpert_valid  first")
todo = {p for p in need if not os.path.exists(f"{ROOT}/images/{p}")}
print(f"need {len(need)} have {len(need) - len(todo)} todo {len(todo)}", flush=True)
t0 = time.time(); n = 0; nbytes = 0
for z in chunks:
    with zipfile.ZipFile(f"{ZIPS}/{z}") as zf:
        members = [(m, m[m.find("valid/"):]) for m in zf.namelist() if m.find("valid/") >= 0 and m[m.find("valid/"):] in todo]
        for m, p in members:
            info = zf.getinfo(m)
            data = zf.read(m)                                # zipfile raises BadZipFile on a CRC-32 mismatch
            if len(data) != info.file_size:
                sys.exit(f"{m}: read {len(data)} bytes, central directory says {info.file_size}")
            dest = f"{ROOT}/images/{p}"
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest + ".part", "wb") as f:
                f.write(data)
            os.replace(dest + ".part", dest); n += 1; nbytes += len(data)
        print(f"{z}: {len(members)} extracted (total {n}, {nbytes / 2**20:.0f} MiB, {time.time() - t0:.0f}s)", flush=True)
missing = sorted(p for p in need if not os.path.exists(f"{ROOT}/images/{p}"))
print(f"final missing {len(missing)}", missing[:5], flush=True)
if not missing:
    open(f"{ROOT}/images/valid/.complete", "w").write(time.strftime("%FT%TZ", time.gmtime()))
    print("CHEXPERT_VALID_EXTRACT_DONE", flush=True)
sys.exit(1 if missing else 0)
