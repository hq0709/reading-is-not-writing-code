"""Watch failed CheXpert preparation tasks whose features were written with the truncated (U32) row_id dtype:
delete those feature files and move the task back to pending so a worker redoes them with the fixed writer."""
import glob, json, os, sys, time
import numpy as np
Q = "/rodata/azradonc_dev/m253405/cf-transfer/queue"; R = "/rodata/azradonc_dev/m253405/cf-transfer/runs"
def truncated(path):
    try:
        f = np.load(path); return f["row_id"].dtype.itemsize // 4 < 36
    except Exception:
        return False
deadline = time.time() + float(sys.argv[1]) * 3600 if len(sys.argv) > 1 else float("inf")
done_once = False
while time.time() < deadline:
    for p in sorted(glob.glob(f"{Q}/failed/*prep-*-chexpert.json")):
        t = json.load(open(p)); m = t["name"].split("prep-")[1].rsplit("-chexpert", 1)[0]
        feats = glob.glob(f"{R}/{m}/chexpert/features/*.npz") + glob.glob(f"{R}/{m}/chexpert/features_train/*.npz")
        bad = [f for f in feats if truncated(f)]
        for f in bad:
            os.remove(f)
        for k in ("worker", "started_utc", "ended_utc", "returncode", "log"):
            t.pop(k, None)
        t["requeued_reason"] = f"row_id truncation fix; removed {len(bad)} truncated feature files"
        dest = f"{Q}/pending/{os.path.basename(p)}"
        with open(dest, "w") as fh: json.dump(t, fh, indent=1)
        os.remove(p)
        print(f"[{time.strftime('%FT%TZ', time.gmtime())}] requeued {t['name']} (removed {len(bad)} files)", flush=True)
    time.sleep(120)
