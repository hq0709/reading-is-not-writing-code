"""Shared task queue + worker so any GPU (login node or Slurm) drains the campaign without duplication.

Queue layout: <QUEUE>/pending/<prio>-<name>.json -> running/ -> done/ | failed/. A task is claimed by an atomic
rename, so concurrent workers never run the same task. Task JSON: {"name", "lane": "gpu1|gpu2|gpu4", "cmd": [..],
"requires": [paths that must exist], "produces": [paths or globs; if all exist the task is marked done unrun],
"env": {...}}. Workers exit when nothing in their lane is claimable and `--exit-when-empty`, or when the time
budget is exhausted (they finish the current task first).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

QUEUE = Path("/rodata/azradonc_dev/m253405/cf-transfer/queue")
LOGS = Path("/rodata/azradonc_dev/m253405/cf-transfer/logs/worker")


def satisfied(paths: list[str]) -> bool:
    return all(glob.glob(p) if any(ch in p for ch in "*?[") else os.path.exists(p) for p in paths)


def claim(lane: str) -> tuple[Path, dict] | None:
    for p in sorted((QUEUE / "pending").glob("*.json")):
        try:
            t = json.loads(p.read_text())
        except Exception:
            continue
        if t.get("lane", "gpu1") != lane:
            continue
        if t.get("produces") and satisfied(t["produces"]):
            try:
                p.rename(QUEUE / "done" / p.name)
            except OSError:
                pass
            continue
        if not satisfied(t.get("requires", [])):
            continue
        target = QUEUE / "running" / p.name
        try:
            p.rename(target)                     # atomic on the same filesystem; fails if someone else took it
        except OSError:
            continue
        return target, t
    return None


def gpu_free(min_free_mib: int = 70000) -> bool:
    """Only the GPUs visible to this worker (nvidia-smi ignores CUDA_VISIBLE_DEVICES, so pass -i explicitly)."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    cmd = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"] + (["-i", vis] if vis else [])
    try:
        out = subprocess.check_output(cmd, text=True)
        return all(int(x) >= min_free_mib for x in out.split())
    except Exception:
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", default="gpu1")
    ap.add_argument("--max-hours", type=float, default=1e9, help="stop claiming new tasks after this many hours")
    ap.add_argument("--exit-when-empty", action="store_true")
    ap.add_argument("--wait-for-free-gpu", action="store_true", help="login node: wait until the visible GPU(s) are idle")
    ap.add_argument("--worker-id", default=f"{os.uname().nodename}-{os.environ.get('SLURM_JOB_ID', os.getpid())}")
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.wait_for_free_gpu:
        while not gpu_free():
            time.sleep(60)
    print(f"[worker {a.worker_id}] lane={a.lane} start", flush=True)
    while time.time() - t0 < a.max_hours * 3600:
        got = claim(a.lane)
        if got is None:
            if a.exit_when_empty and not any(json.loads(p.read_text()).get("lane", "gpu1") == a.lane
                                              for p in (QUEUE / "pending").glob("*.json")):
                break
            time.sleep(60)
            continue
        path, t = got
        log = LOGS / f"{path.stem}.{a.worker_id}.log"
        env = {**os.environ, **t.get("env", {}), "PYTHONUNBUFFERED": "1"}
        t["worker"] = a.worker_id; t["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); path.write_text(json.dumps(t, indent=1))
        print(f"[worker {a.worker_id}] running {path.stem} -> {log}", flush=True)
        with log.open("a") as f:
            rc = subprocess.call(t["cmd"], cwd=t.get("cwd", "/rodata/azradonc_dev/m253405/concept-flow/src"), env=env, stdout=f, stderr=subprocess.STDOUT)
        t["returncode"] = rc; t["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); t["log"] = str(log)
        dest = QUEUE / ("done" if rc == 0 and (not t.get("produces") or satisfied(t["produces"])) else "failed") / path.name
        path.write_text(json.dumps(t, indent=1)); path.rename(dest)
        print(f"[worker {a.worker_id}] {'done' if dest.parent.name == 'done' else 'FAILED'} {path.stem} rc={rc}", flush=True)
    print(f"[worker {a.worker_id}] exit after {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
