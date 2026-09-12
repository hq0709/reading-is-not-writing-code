"""Download the CheXpert Plus PNG zip chunks from Redivis (large files stream fast; per-file PNG downloads are throttled)."""
import os, sys, time, json
os.environ["REDIVIS_API_ENDPOINT"] = "https://stanford.redivis.com/api/v1"
os.environ["REDIVIS_API_TOKEN"] = open("/rodata/azradonc_dev/m253405/cf-transfer/data/chexpert/redivis_token").read().strip()
import redivis
ROOT = "/rodata/azradonc_dev/m253405/cf-transfer/data/chexpert/zips"; os.makedirs(ROOT, exist_ok=True)
ds = redivis.organization("aimi").dataset("chexpert_plus:5yyj")
t = ds.table("png_compressed:wsd7")
idx = t.to_pandas_dataframe()
order = sys.argv[1].split(",") if len(sys.argv) > 1 else sorted(idx.file_name)
for name in order:
    row = idx[idx.file_name == name].iloc[0]
    dest = f"{ROOT}/{name}"
    if os.path.exists(dest + ".done"):
        print("skip", name, flush=True); continue
    t0 = time.time(); print(f"[{time.strftime('%FT%TZ', time.gmtime())}] downloading {name} ({row['size']/1e9:.1f} GB)", flush=True)
    t.file(row.file_id).download(dest, overwrite=True, progress=False)
    sz = os.path.getsize(dest); ok = sz == int(row["size"])
    print(f"[{time.strftime('%FT%TZ', time.gmtime())}] done {name} {sz} bytes ok={ok} {sz/1e6/(time.time()-t0):.1f} MB/s", flush=True)
    if ok:
        open(dest + ".done", "w").write(json.dumps({"size": sz, "md5_hash": row["md5_hash"]}))
print("CHEXPERT_ZIPS_DONE")
