"""Run Experiment 1.

    python -m lawlike.run --preset smoke          # ~1-2 min sanity run
    python -m lawlike.run --preset quick          # first real look
    python -m lawlike.run --preset full           # press go, walk away
    python -m lawlike.run --preset quick --analyze-only
    python -m lawlike.run --preset full2a         # Phase 2a confirmation run (README 3b.6)

Results stream to <out>/results.jsonl one world at a time, so you can stop (Ctrl-C) and
re-run the same command to resume, and you can run the analysis on partial results at any
time. Worlds are processed highest-prior-weight first, interleaving the three arms, so
partial results are already a fair (if noisier) estimate.
"""
from __future__ import annotations

import os

# one numpy thread per worker process; parallelism comes from processes
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback

from .config import get_preset
from .worlds import make_world_list

_CFG = None
_OUT = None


def _init(cfg, out):
    global _CFG, _OUT
    _CFG, _OUT = cfg, out


def _work(world):
    from .experiment1 import process_world
    t = time.time()
    try:
        s = process_world(world, _CFG, _OUT)
        s["seconds"] = time.time() - t
        return ("ok", world.world_id, s)
    except Exception:
        return ("err", world.world_id, traceback.format_exc())


def ordered(worlds):
    """Highest weight first within each arm, arms interleaved round-robin."""
    by_arm: dict[str, list] = {}
    for w in worlds:
        by_arm.setdefault(w.arm, []).append(w)
    for v in by_arm.values():
        v.sort(key=lambda w: -w.weight)
    out = []
    iters = {a: iter(v) for a, v in by_arm.items()}
    # interleave in proportion to arm size so every arm finishes around the same time
    sizes = {a: len(v) for a, v in by_arm.items()}
    taken = {a: 0 for a in by_arm}
    total = sum(sizes.values())
    for _ in range(total):
        a = min((a for a in by_arm if taken[a] < sizes[a]), key=lambda a: taken[a] / sizes[a])
        out.append(next(iters[a]))
        taken[a] += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="smoke")
    ap.add_argument("--out", default=None, help="output directory (default results/<preset>)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=None, help="process at most this many new worlds")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--no-analyze", action="store_true")
    args = ap.parse_args(argv)

    cfg = get_preset(args.preset)
    out = args.out or os.path.join("results", cfg.name)
    os.makedirs(out, exist_ok=True)
    cfg_path = os.path.join(out, "config.json")
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        if old.get("hash") != cfg.hash():
            sys.exit(f"{out} holds results from a different config (hash {old.get('hash')} != "
                     f"{cfg.hash()}). Use a different --out, or delete that directory.")
    else:
        json.dump({"hash": cfg.hash(), "config": cfg.to_dict()}, open(cfg_path, "w"), indent=2, default=str)

    res_path = os.path.join(out, "results.jsonl")
    if not args.analyze_only:
        done = set()
        if os.path.exists(res_path):
            with open(res_path) as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["world"]["world_id"])
                    except Exception:
                        pass
        todo = [w for w in ordered(make_world_list(cfg)) if w.world_id not in done]
        if args.limit:
            todo = todo[: args.limit]
        n_total = len(todo) + len(done)
        print(f"[lawlike] preset={cfg.name} config={cfg.hash()} out={out}")
        print(f"[lawlike] {len(done)} worlds already done, {len(todo)} to go, {args.workers} workers")
        if todo:
            t0 = time.time()
            last = 0.0
            n = 0
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.workers, initializer=_init, initargs=(cfg, out)) as pool, \
                    open(res_path, "a") as fres, open(os.path.join(out, "errors.log"), "a") as ferr:
                try:
                    for status, wid, payload in pool.imap_unordered(_work, todo, chunksize=1):
                        n += 1
                        if status == "ok":
                            fres.write(json.dumps(payload) + "\n")
                            fres.flush()
                        else:
                            ferr.write(f"=== {wid}\n{payload}\n")
                            ferr.flush()
                            print(f"[lawlike] ERROR in {wid} (see errors.log)")
                        now = time.time()
                        if now - last > 10 or n == len(todo):
                            last = now
                            rate = n / (now - t0)
                            eta = (len(todo) - n) / rate if rate > 0 else float("inf")
                            print(f"[lawlike] {n + len(done)}/{n_total} worlds | "
                                  f"{(now - t0) / 60:.1f} min elapsed | ETA {eta / 60:.1f} min", flush=True)
                except KeyboardInterrupt:
                    print("\n[lawlike] interrupted; progress is saved. Re-run the same command to resume.")
                    pool.terminate()
                    return
    if not args.no_analyze:
        from .analyze import analyze
        analyze(out)
        if getattr(cfg, "seed_mode", "crossed") == "per_rule":   # Phase 2a design
            from .analyze2a import analyze2a
            p1 = "results/full" if os.path.exists("results/full/results.jsonl") else None
            analyze2a(out, phase1_dir=p1)


if __name__ == "__main__":
    main()
