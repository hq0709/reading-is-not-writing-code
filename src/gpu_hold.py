"""Reserve GPUs on a shared box by occupying them, and release them when a real job needs one.

The box has no scheduler, so a free GPU is free until someone else takes it. This holds a card by
allocating most of its memory and keeping a small matmul running, which is what makes it visible to other
users in nvidia-smi as genuinely busy rather than idle-but-allocated.

    python src/gpu_hold.py hold 4 5 6 7      # reserve those four
    python src/gpu_hold.py status            # what is held, by whom
    python src/gpu_hold.py release 5         # free one before a real run uses it
    python src/gpu_hold.py release all

Release before launching work on a card. A held card cannot be used by our own jobs either.
"""
import os
import signal
import subprocess
import sys
import time

PIDDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", ".gpu_hold")


def _pidfile(idx):
    return os.path.join(PIDDIR, f"gpu{idx}.pid")


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _held():
    out = {}
    if not os.path.isdir(PIDDIR):
        return out
    for f in sorted(os.listdir(PIDDIR)):
        if not f.endswith(".pid"):
            continue
        idx = int(f[3:-4])
        try:
            pid = int(open(os.path.join(PIDDIR, f)).read().strip())
        except (ValueError, OSError):
            continue
        if _alive(pid):
            out[idx] = pid
        else:
            os.unlink(os.path.join(PIDDIR, f))
    return out


def _worker(idx, frac):
    """Runs in the child. Fills the card and keeps a little compute going."""
    import torch

    torch.cuda.set_device(idx)
    free, total = torch.cuda.mem_get_info(idx)
    want, block = int(free * frac), None
    while want > (1 << 28):
        try:
            block = torch.empty(want, dtype=torch.uint8, device=f"cuda:{idx}")
            break
        except torch.cuda.OutOfMemoryError:
            want = int(want * 0.9)
    if block is None:
        print(f"gpu{idx}: could not reserve anything", flush=True)
        return
    print(f"gpu{idx}: holding {block.numel() / 2**30:.1f} GiB of {total / 2**30:.1f} GiB", flush=True)
    x = torch.randn(2048, 2048, device=f"cuda:{idx}", dtype=torch.bfloat16)
    while True:
        x = (x @ x).div_(64)
        torch.cuda.synchronize(idx)
        time.sleep(0.2)


def hold(indices, frac=0.92):
    os.makedirs(PIDDIR, exist_ok=True)
    already = _held()
    for idx in indices:
        if idx in already:
            print(f"gpu{idx}: already held by pid {already[idx]}")
            continue
        log = os.path.join(PIDDIR, f"gpu{idx}.log")
        with open(log, "w") as fh:
            p = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "_worker", str(idx), str(frac)],
                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
            )
        open(_pidfile(idx), "w").write(str(p.pid))
        print(f"gpu{idx}: holding, pid {p.pid}, log {log}")
    time.sleep(8)
    status()


def release(indices):
    held = _held()
    targets = sorted(held) if indices == ["all"] else [int(i) for i in indices]
    for idx in targets:
        pid = held.get(idx)
        if pid is None:
            print(f"gpu{idx}: not held")
            continue
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        if os.path.exists(_pidfile(idx)):
            os.unlink(_pidfile(idx))
        print(f"gpu{idx}: released (pid {pid})")


def status():
    held = _held()
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    print(f"\n{'gpu':>4} {'used':>7} {'total':>7} {'util':>5}  holder")
    for line in q.splitlines():
        idx, used, total, util = [s.strip() for s in line.split(",")]
        who = f"ours, pid {held[int(idx)]}" if int(idx) in held else ""
        print(f"{idx:>4} {used:>7} {total:>7} {util:>4}%  {who}")
    print(f"\nheld by us: {sorted(held) if held else 'none'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "_worker":
        _worker(int(sys.argv[2]), float(sys.argv[3]))
    elif cmd == "hold":
        hold([int(a) for a in sys.argv[2:]] or [0])
    elif cmd == "release":
        release(sys.argv[2:] or ["all"])
    else:
        status()
