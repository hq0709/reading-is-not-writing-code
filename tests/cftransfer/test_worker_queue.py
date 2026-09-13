"""Queue worker: stale running tasks (Slurm job gone) are requeued; live and non-Slurm tasks are left alone."""
import json
import time

from cftransfer import worker


def _task(d, name, wid, age_s):
    p = d / "running" / f"{name}.json"
    p.write_text(json.dumps({"name": name, "lane": "gpu1", "cmd": ["true"], "worker": wid, "started_utc": "x"}))
    import os
    os.utime(p, (time.time() - age_s, time.time() - age_s))
    return p


def test_reclaim_stale(tmp_path, monkeypatch):
    for sub in ("pending", "running", "done", "failed"):
        (tmp_path / sub).mkdir()
    monkeypatch.setattr(worker, "QUEUE", tmp_path)
    _task(tmp_path, "010-CORE-a", "rohpcgpu37.mayo.edu-111", 3600)      # job gone -> requeue
    _task(tmp_path, "011-CORE-b", "rohpcgpu38.mayo.edu-222", 3600)      # job alive -> keep
    _task(tmp_path, "012-CORE-c", "login-gpu2", 3600)                    # login worker -> keep
    _task(tmp_path, "013-CORE-d", "rohpcgpu39.mayo.edu-333", 30)        # gone but inside grace -> keep
    moved = worker.reclaim_stale(live={"222"})
    assert moved == ["010-CORE-a"]
    assert sorted(p.name for p in (tmp_path / "pending").glob("*.json")) == ["010-CORE-a.json"]
    assert sorted(p.stem for p in (tmp_path / "running").glob("*.json")) == ["011-CORE-b", "012-CORE-c", "013-CORE-d"]
    t = json.loads((tmp_path / "pending" / "010-CORE-a.json").read_text())
    assert "worker" not in t and t["requeued"][0]["from"] == "rohpcgpu37.mayo.edu-111"
    monkeypatch.setattr(worker, "live_slurm_jobs", lambda: None)
    assert worker.reclaim_stale() == []                   # squeue unavailable -> no action
