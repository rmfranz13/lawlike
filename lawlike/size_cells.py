"""Separated size checks (README 3b.8 A10).

    python -m lawlike.size_cells results/full2a_sf

A7 doubled N and T together. Here the highest-weight dynamically rich worlds of the M_A
simplicity arm (A7's selection rule, on this run) are re-run in four cells:

    N2    2N, same T0 and T     ring size, with the same amount of data
    late  T0 + T, same N and T  the same amount of data, taken later
    T2    2T, same N and T0     twice the data (it spans both epochs)
    NT2   2N, 2T0, 2T           A7 as it was run

A world "moves" in a cell if lawful_nt changes by more than 0.1 under any counted filter
(defined <-> undefined counts). Caveats trigger above one third of the checked worlds:
ring-size (N2), epoch (late), data-length (T2 differs from BOTH base and late).
Resumable: results go to <dir>/size_cells/<cell>/.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

import numpy as np

from .analyze import load
from .analyze2a import COUNTED, fmt
from .config import Exp1Config, size_cells
from .size_check import CAVEAT, MOVE, select
from .worlds import World

CELLS = ("N2", "late", "T2", "NT2")
EXPLORE = ("lawful_nt_m1", "mean_gap_nt", "mean_depth_nt", "mean_antic", "antic_gap")   # README 3c, exploratory


def load_cfg(out_dir) -> Exp1Config:
    raw = json.load(open(os.path.join(out_dir, "config.json")))["config"]
    return Exp1Config(**{k: (tuple(map(tuple, v)) if k == "velocities" else tuple(v) if isinstance(v, list) else v)
                         for k, v in raw.items()})


def differs(x, y) -> bool:
    if (x is None) != (y is None):
        return True
    return x is not None and abs(x - y) > MOVE


def main(argv=None):
    ap = argparse.ArgumentParser(description="Separated size checks (README 3b.8 A10).")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    rows = load(a.out_dir)
    cfg = load_cfg(a.out_dir)
    pick, cover = select(rows)
    worlds = [World(**{k: v for k, v in rows[i]["world"].items() if k in World.__dataclass_fields__}) for i in pick]
    cells = size_cells(cfg)
    res = {}
    for name in CELLS:
        c = cells[name]
        c.save_observers = False
        cdir = os.path.join(a.out_dir, "size_cells", name)
        os.makedirs(cdir, exist_ok=True)
        path = os.path.join(cdir, "results.jsonl")
        done = {json.loads(l)["world"]["world_id"] for l in open(path)} if os.path.exists(path) else set()
        todo = [w for w in worlds if w.world_id not in done]
        print(f"[size_cells] {name}: N={c.N} T0={c.T0} T={c.T}; {len(todo)} of {len(worlds)} worlds to run", flush=True)
        if todo:
            from .run import _init, _work
            with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(c, cdir)) as pool, \
                    open(path, "a") as f:
                for status, wid, payload in pool.imap_unordered(_work, todo):
                    if status == "ok":
                        f.write(json.dumps(payload) + "\n")
                        f.flush()
                    else:
                        print(f"[size_cells] ERROR {name} {wid}\n{payload}")
        res[name] = {r["world"]["world_id"]: r for r in load(cdir)}

    def lnt(r, f):
        return None if r is None else r["filters"][f].get("lawful_nt")

    L = ["# Separated size checks (README §3b.8 A10)\n",
         f"{len(pick)} highest-weight dynamically rich worlds (M_A simplicity arm), covering {cover:.0%} of its "
         f"dynamically rich weight. Base: N={cfg.N}, T0={cfg.T0}, T={cfg.T}. Cells: "
         + "; ".join(f"`{n}` N={cells[n].N}, T0={cells[n].T0}, T={cells[n].T}" for n in CELLS)
         + f". A world moves if lawful_nt changes by > {MOVE} under any counted filter.\n"]
    moved = {n: 0 for n in CELLS}
    up = {n: 0 for n in CELLS}
    down = {n: 0 for n in CELLS}
    data_len = 0
    n_cmp = 0
    T = ["| world | ring | " + " | ".join(f"{f}: base / N2 / late / T2 / NT2" for f in COUNTED) + " | moved in | data-length |",
         "|---|---|" + "---|" * (len(COUNTED) + 2)]
    for i in pick:
        r = rows[i]
        wid = r["world"]["world_id"]
        cr = {n: res[n].get(wid) for n in CELLS}
        if any(v is None for v in cr.values()):
            continue
        n_cmp += 1
        cells_txt, mv = [], []
        for f in COUNTED:
            cells_txt.append(" / ".join(fmt(lnt(x, f)) for x in (r, *cr.values())))
        for n in CELLS:
            ds = [(lnt(r, f), lnt(cr[n], f)) for f in COUNTED]
            if any(differs(x, y) for x, y in ds):
                moved[n] += 1
                mv.append(n)
                sg = [np.sign(y - x) for x, y in ds if x is not None and y is not None and abs(y - x) > MOVE]
                up[n] += bool(sg) and all(s > 0 for s in sg)
                down[n] += bool(sg) and all(s < 0 for s in sg)
        dl = any(differs(lnt(r, f), lnt(cr["T2"], f)) and differs(lnt(cr["late"], f), lnt(cr["T2"], f)) for f in COUNTED)
        data_len += dl
        T.append(f"| {wid} | {r['world_stats'].get('ring', cfg.N)} | " + " | ".join(cells_txt)
                 + f" | {', '.join(mv)} | {'yes' if dl else ''} |")

    def share(k):
        return k / n_cmp if n_cmp else float("nan")

    L.append("| cell | isolates | worlds moved | all moves up | all moves down | caveat |")
    L.append("|---|---|---|---|---|---|")
    what = dict(N2="ring size", late="epoch", T2="twice the data", NT2="A7 as run")
    cav = dict(N2="ring-size", late="epoch", NT2="A7 finite-size")
    for n in CELLS:
        c = f"**{cav[n]} caveat triggered**" if n in cav and share(moved[n]) > CAVEAT else ("" if n in cav else "see data-length")
        L.append(f"| `{n}` | {what[n]} | {moved[n]}/{n_cmp} ({share(moved[n]):.0%}) | {up[n]} | {down[n]} | {c} |")
    L.append(f"\n**Data-length** (T2 differs by > {MOVE} from both base and late, any counted filter): "
             f"{data_len}/{n_cmp} ({share(data_len):.0%}). "
             + ("**Data-length caveat triggered:** the lawful/noisy classification depends on stream length; a "
                "length-robust lawfulness measure must be pre-registered before 2b." if share(data_len) > CAVEAT
                else "At most a third: no data-length caveat."))
    L.append("\n## Per world\n")
    L += T
    L.append("\n## README §3c metrics by cell (EXPLORATORY; unweighted mean over the checked worlds)\n")
    L.append("| metric | filter | base | " + " | ".join(CELLS) + " |")
    L.append("|---|---|---|" + "---|" * len(CELLS))
    ids = [rows[i]["world"]["world_id"] for i in pick if all(rows[i]["world"]["world_id"] in res[n] for n in CELLS)]
    base = {rows[i]["world"]["world_id"]: rows[i] for i in pick}
    for mname in EXPLORE:
        for f in COUNTED:
            vals = []
            for src in (base, *(res[n] for n in CELLS)):
                v = [src[w]["filters"][f].get(mname) for w in ids]
                v = [x for x in v if x is not None and np.isfinite(x)]
                vals.append(fmt(float(np.mean(v))) if v else "—")
            L.append(f"| {mname} | {f} | " + " | ".join(vals) + " |")
    out = os.path.join(a.out_dir, "report_size_cells.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[size_cells] wrote {out}")


if __name__ == "__main__":
    main()
