"""Seam-free close-out of 2a (README 3b.8 A9): what changed relative to the seamed run.

    python -m lawlike.seam_check results/full2a_sf results/full2a

  * regression: every world whose ring is unchanged must reproduce every pre-existing field
    of the seamed run exactly;
  * where the dynamically rich weight went, and the translation periods of tile worlds;
  * the verdict tables side by side; a headline verdict that differs is "seam-sensitive".
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from .analyze import load
from .analyze2a import Table, fmt


def _same(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and all(k in b and _same(v, b[k]) for k, v in a.items())
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def main(argv=None):
    ap = argparse.ArgumentParser(description="Seam-free vs seamed 2a (README 3b.8 A9).")
    ap.add_argument("sf_dir")
    ap.add_argument("seamed_dir")
    a = ap.parse_args(argv)
    new, old = load(a.sf_dir), load(a.seamed_dir)
    N = json.load(open(os.path.join(a.sf_dir, "config.json")))["config"]["N"]
    oldby = {r["world"]["world_id"]: r for r in old}
    L = [f"# Seam-free close-out of 2a (README §3b.8 A9): `{a.sf_dir}` vs `{a.seamed_dir}`\n"]

    # ---- regression
    same_list = {r["world"]["world_id"] for r in new} == set(oldby)
    changed = [r for r in new if r["world_stats"]["ring"] != N]
    kept = [r for r in new if r["world_stats"]["ring"] == N]
    bad = []
    for r in kept:
        o = oldby.get(r["world"]["world_id"])
        if o is None or not all(_same(o[k], r[k]) for k in ("world", "world_stats", "filters", "jaccard")):
            bad.append(r["world"]["world_id"])
    wsame = all(_same(oldby[r["world"]["world_id"]]["world"], r["world"]) for r in new if r["world"]["world_id"] in oldby)
    L.append("## Regression check\n")
    L.append(f"- Same world list: **{'yes' if same_list else 'NO'}** ({len(new)} worlds). "
             f"World records (weights, RNG seeds) identical: **{'yes' if wsame else 'NO'}**.")
    L.append(f"- {len(changed)} tile worlds run on a different ring (seed length does not divide {N}); "
             f"{len(kept)} worlds are unchanged.")
    L.append(f"- Unchanged worlds reproducing every pre-existing field exactly: **{len(kept) - len(bad)}/{len(kept)}**"
             + (f". MISMATCHES: {bad[:10]}{' …' if len(bad) > 10 else ''}" if bad else "."))

    # ---- dynamical richness
    tn, to = Table(new), Table([oldby[r["world"]["world_id"]] for r in new])
    L.append("\n## Where the dynamically rich weight went (simplicity arm)\n")
    L.append("| prior | seamed | seam-free |")
    L.append("|---|---|---|")
    simp = tn.arm == "simplicity"
    for P in ("phase1", "M_A", "M_B"):
        w = tn.wp[P]
        L.append(f"| {P} | {w[simp & to.dyn].sum() / w[simp].sum():.4f} | {w[simp & tn.dyn].sum() / w[simp].sum():.4f} |")
    tile = np.array([r["world"]["ic_kind"] == "tile" for r in new])
    ch = np.array([r["world_stats"]["ring"] != N for r in new])
    w = tn.wp["M_A"]
    L.append(f"\nOf the {int(ch.sum())} re-ringed tile worlds, {int((ch & to.dyn).sum())} were dynamically rich when seamed "
             f"and {int((ch & tn.dyn).sum())} are when seam-free. Their share of the M_A simplicity arm's dynamically rich "
             f"weight: {w[ch & to.dyn].sum() / w[simp & to.dyn].sum():.3f} seamed, "
             f"{w[ch & tn.dyn].sum() / max(w[simp & tn.dyn].sum(), 1e-300):.3f} seam-free.")

    def bucket(p):
        return "none ≤ 512" if p is None else "≤ 16" if p <= 16 else "17–64" if p <= 64 else "65–512"
    names = ("≤ 16", "17–64", "65–512", "none ≤ 512")
    L.append("\nTranslation periods (analysed window), number of worlds:\n")
    L.append("| worlds | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    for lab, m in (("re-ringed tile worlds, seamed", ch), ("re-ringed tile worlds, seam-free", ch),
                   ("all tile worlds, seam-free", tile), ("all simplicity worlds, seam-free", simp)):
        src = old if lab.endswith(", seamed") else new
        rr = [oldby[new[i]["world"]["world_id"]] if src is old else new[i] for i in np.where(m)[0]]
        c = {k: 0 for k in names}
        for r in rr:
            c[bucket(r["world_stats"]["tperiod"])] += 1
        L.append(f"| {lab} | " + " | ".join(str(c[k]) for k in names) + " |")
    gap = [r["world"]["world_id"] for r in new if r["world"]["arm"] != "noise"
           and r["world_stats"]["tperiod"] is not None and 17 <= r["world_stats"]["tperiod"] <= 64]
    L.append(f"\nThe 17–64 gap that motivated A1's cutoff: **{'still empty' if not gap else f'{len(gap)} worlds now fall in it'}** "
             f"(all non-noise worlds){'' if not gap else ': ' + ', '.join(gap[:8]) + (' …' if len(gap) > 8 else '')}. "
             "The cutoff stays at 16 either way (A9).")

    # ---- verdicts
    ps, po = (os.path.join(d, "summary_2a.json") for d in (a.sf_dir, a.seamed_dir))
    if os.path.exists(ps) and os.path.exists(po):
        s, o = json.load(open(ps)), json.load(open(po))
        L.append("\n## Verdicts, seamed vs seam-free (the seam-free column is the 2a verdict of record)\n")
        L.append("| hypothesis | prior / column | analysis | seamed | seam-free | |")
        L.append("|---|---|---|---|---|---|")
        for k in sorted(set(s) | set(o)):
            h, P, lab = k.split("|")
            head = h in ("H1", "H7a", "H7b") and P.startswith("M_A") and P != "M_A*" and "secondary" not in lab
            diff = s.get(k) != o.get(k)
            flag = ("**seam-sensitive**" if head else "changed") if diff else ""
            L.append(f"| {h} | {P} | {lab} | {o.get(k, '—')} | {s.get(k, '—')} | {flag} |")
    out = os.path.join(a.sf_dir, "report_seam.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[seam_check] wrote {out}")


if __name__ == "__main__":
    main()
