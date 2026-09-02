"""Fill every free GPU with an intervention job and keep them fed until the queue is empty.

The box is shared and has no scheduler, so a card left idle for a few minutes is taken by someone else.
This builds the full (model, concept) queue, starts one job per free card, and starts the next queued job
the moment a card frees, until everything is done.

Loci are chosen per model at matched relative depths, not at fixed layer indices, because the models have
28 or 32 LLM layers. That keeps the curves comparable across architectures and still covers each model's
own decodability peak.

    python src/launch_interventions.py                    # everything, auto GPUs
    python src/launch_interventions.py --dry-run
    python src/launch_interventions.py --gpus 0,1,2,3,6 --concepts Effusion,Cardiomegaly
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from registry import REGISTRY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def loci_spec(arch_key: str) -> str:
    """Matched relative depths: encoder output, connector, then quarters through the LLM."""
    n = REGISTRY[arch_key].n_llm_layers
    layers = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    return ",".join(["vis.last", "connector"] + [f"llm.L{i}.vis" for i in layers])


def free_gpus(min_free_gb=30):
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    out = []
    for line in q.strip().splitlines():
        i, used, total = [int(x) for x in line.split(",")]
        if (total - used) / 1024 >= min_free_gb:
            out.append(i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="lingshu7b,qwen7b,llavamed7b,llava15_7b,internvl3_8b")
    ap.add_argument("--concepts", default="Effusion,Cardiomegaly,Pneumothorax")
    ap.add_argument("--gpus", default="", help="comma separated; default is every card with room")
    ap.add_argument("--n-eval", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll", type=int, default=45)
    args = ap.parse_args()

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]

    queue = []
    for concept in concepts:                       # concept-major, so every model gets concept 1 first
        for arch in archs:
            name = f"int_{arch}_{concept.lower()}"
            if os.path.exists(os.path.join(ROOT, "runs", name, "intervention.csv")):
                print(f"skip {name}, already done")
                continue
            queue.append((arch, concept, name))

    gpus = ([int(g) for g in args.gpus.split(",") if g.strip()] if args.gpus else free_gpus())
    print(f"{len(queue)} jobs over {len(gpus)} gpus: {gpus}")
    for arch, concept, name in queue:
        print(f"  {name:34s} loci {loci_spec(arch)}")
    if args.dry_run or not queue:
        return

    env = dict(os.environ, HF_HOME=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false")
    running: dict[int, tuple] = {}
    pending = list(queue)
    t0 = time.time()

    def start(gpu):
        arch, concept, name = pending.pop(0)
        log = os.path.join(ROOT, "runs", f"{name}.log")
        cmd = [sys.executable, "src/intervene.py",
               "--acts", f"runs/act_{arch}", "--arch", arch, "--concept", concept,
               "--loci", loci_spec(arch), "--n-eval", str(args.n_eval),
               "--batch-size", str(args.batch_size), "--gpu", str(gpu),
               "--out", f"runs/{name}"]
        with open(log, "w") as fh:
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        running[gpu] = (p, name, time.time())
        print(f"[{time.strftime('%H:%M:%S')}] gpu{gpu} <- {name} (pid {p.pid})", flush=True)

    for g in gpus:
        if pending:
            start(g)

    while running:
        time.sleep(args.poll)
        for gpu in list(running):
            p, name, started = running[gpu]
            if p.poll() is None:
                continue
            mins = (time.time() - started) / 60
            ok = os.path.exists(os.path.join(ROOT, "runs", name, "intervention.csv"))
            print(f"[{time.strftime('%H:%M:%S')}] gpu{gpu} done {name} in {mins:.0f}m "
                  f"exit {p.returncode} {'OK' if ok else 'NO OUTPUT'}", flush=True)
            del running[gpu]
            if pending:
                start(gpu)
        if not pending and not running:
            break

    print(f"all interventions finished in {(time.time() - t0) / 60:.0f} minutes")


if __name__ == "__main__":
    main()
