"""Finite-size check (README 3b.7 A7).

    python -m lawlike.size_check results/full2a

Takes the highest-weight dynamically rich worlds of the M_A simplicity arm, until they
cover 50% of that arm's dynamically rich weight (at most 40 worlds), re-runs each at 2N and
2T (transient and observer start grid scaled to match; config.scaled_for_size_check), and
compares lawful_nt per counted filter. Resumable: results go to <dir>/size_check/.

A world "moves" if lawful_nt changes by more than 0.1 under ANY counted filter (the
conservative reading of A7). If more than a third of the checked worlds move, every rich-world
verdict carries a finite-size caveat.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

import numpy as np

from .analyze import load
from .analyze2a import COUNTED, Table, fmt
from .config import Exp1Config, scaled_for_size_check
from .worlds import World

COVER, CAP, MOVE, CAVEAT = 0.5, 40, 0.1, 1 / 3


def select(rows):
    tab = Table(rows)
    m = (tab.arm == "simplicity") & tab.dyn
    idx = np.where(m)[0]
    w = tab.wp["M_A"][idx]
    order = idx[np.argsort(-w)]
    tot, acc, pick = w.sum(), 0.0, []
    for i in order:
        if acc >= COVER * tot or len(pick) >= CAP:
            break
        pick.append(int(i))
        acc += tab.wp["M_A"][i]
    return pick, acc / tot if tot > 0 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Finite-size check (README 3b.7 A7).")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    rows = load(a.out_dir)
    cfg = Exp1Config(**{k: (tuple(map(tuple, v)) if k == "velocities" else tuple(v) if isinstance(v, list) else v)
                        for k, v in json.load(open(os.path.join(a.out_dir, "config.json")))["config"].items()})
    big = scaled_for_size_check(cfg)
    big.save_observers = False
    sdir = os.path.join(a.out_dir, "size_check")
    os.makedirs(sdir, exist_ok=True)
    pick, cover = select(rows)
    res_path = os.path.join(sdir, "results.jsonl")
    done = {json.loads(l)["world"]["world_id"] for l in open(res_path)} if os.path.exists(res_path) else set()
    worlds = [World(**{k: v for k, v in rows[i]["world"].items() if k in World.__dataclass_fields__}) for i in pick]
    todo = [w for w in worlds if w.world_id not in done]
    print(f"[size_check] {len(pick)} worlds covering {cover:.0%} of dynamically rich M_A weight; "
          f"{len(todo)} to run at N={big.N}, T={big.T}")
    if todo:
        from .run import _init, _work
        with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(big, sdir)) as pool, \
                open(res_path, "a") as f:
            for n, (status, wid, payload) in enumerate(pool.imap_unordered(_work, todo), 1):
                if status == "ok":
                    f.write(json.dumps(payload) + "\n")
                    f.flush()
                else:
                    print(f"[size_check] ERROR {wid}\n{payload}")
                print(f"[size_check] {n}/{len(todo)}", flush=True)
    bigrows = {r["world"]["world_id"]: r for r in load(sdir)}
    L = ["# Finite-size check (README §3b.7 A7)\n",
         f"{len(pick)} highest-weight dynamically rich worlds (M_A simplicity arm), covering {cover:.0%} of its "
         f"dynamically rich weight. Each re-run at N={big.N}, T={big.T}, T0={big.T0}, pos_stride={big.pos_stride} "
         f"(base N={cfg.N}, T={cfg.T}). A world moves if lawful_nt changes by > {MOVE} under any counted filter.\n",
         "| world | " + " | ".join(f"{f} base → 2×" for f in COUNTED) + " | moved |",
         "|---|" + "---|" * (len(COUNTED) + 1)]
    moved = 0
    n_cmp = 0
    for i in pick:
        r = rows[i]
        b = bigrows.get(r["world"]["world_id"])
        if b is None:
            continue
        n_cmp += 1
        cells, mv = [], False
        for f in COUNTED:
            x, y = r["filters"][f].get("lawful_nt"), b["filters"][f].get("lawful_nt")
            cells.append(f"{fmt(x)} → {fmt(y)}")
            if (x is None) != (y is None) or (x is not None and abs(x - y) > MOVE):
                mv = True
        moved += mv
        L.append(f"| {r['world']['world_id']} | " + " | ".join(cells) + f" | {'yes' if mv else ''} |")
    share = moved / n_cmp if n_cmp else float("nan")
    L.append(f"\n**{moved}/{n_cmp} worlds moved ({share:.0%}).** " +
             ("More than a third: every rich-world verdict carries a finite-size caveat." if share > CAVEAT
              else "At most a third: no finite-size caveat."))
    with open(os.path.join(a.out_dir, "report_size_check.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[size_check] wrote {os.path.join(a.out_dir, 'report_size_check.md')}")


if __name__ == "__main__":
    main()
