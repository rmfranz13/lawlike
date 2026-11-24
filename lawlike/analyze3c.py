"""README 3c: memory horizons, anticipation (the self-modelling check), open-endedness.

    python -m lawlike.analyze3c results/full2a_sf

Predictions M0, M1, S1, O1, O2 (README 3c.5): per counted filter, "most" = at least 3 of 4,
SSA, M_A, all worlds and dynamically rich worlds, cluster-bootstrap intervals as in 3b.4.
On 2a worlds this is the EXPLORATORY first pass (3c.6): it cannot confirm anything.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .analyze import load, program_key, verdict
from .analyze2a import COUNTED, Table, ci, contrast, ensure_dyn, est, fmt, term, _mark
from .bootstrap import ClusterBootstrap
from .experiment1 import ANTIC_LAGS, FILTERS, HORIZONS, RETAIN_LAGS
from .worlds import elias_gamma_len

SIMP, UNIF = "simp_M_A", "uniform"


def tally(vs, what="SUPPORTED"):
    return sum(vs[f] == what for f in COUNTED)


def analyze3c(out_dir):
    rows = load(out_dir)
    cfg = json.load(open(os.path.join(out_dir, "config.json")))["config"]
    rows = ensure_dyn(rows, cfg, out_dir)
    nested = cfg.get("seed_mode") == "per_rule"
    boot = ClusterBootstrap([r["world"] for r in rows], int(cfg.get("ic_exact_len", 0)), np.random.default_rng(0), nested=nested)
    tab = Table(rows)
    ALL = np.ones(tab.n, bool)
    noise = tab.arm == "noise"
    analyses = (("all worlds", ALL), ("dynamically rich", tab.dyn | noise))
    excl = tuple(cfg.get("exclude_programs_from") or ())
    fresh = None
    if excl and all(os.path.exists(os.path.join(d, "results.jsonl")) for d in excl):
        old = {program_key(r["world"]) for d in excl for r in load(d) if r["world"]["arm"] == "simplicity"}
        fresh = np.array([not (r["world"]["arm"] == "simplicity" and program_key(r["world"]) in old) for r in rows])
    mlo, mhi = HORIZONS[0], HORIZONS[-1]
    summary = {}
    L = [f"# README §3c report — `{os.path.basename(os.path.abspath(out_dir))}`\n",
         "**EXPLORATORY first pass (README §3c.6).** These worlds were examined before the metrics were defined, "
         "so nothing here is a confirmation; M1, S1 and O1 are confirmed or not on fresh Phase 2b worlds. "
         "SSA, M_A, counted filters " + ", ".join(COUNTED) + "; \"most\" = at least 3 of 4. † = deprecated.\n"]
    head_at = len(L)

    # ---- lawfulness by horizon
    for lab, keep in analyses:
        L.append(f"\n## lawful_nt by memory horizon ({lab})\n")
        L.append("| filter | " + " | ".join(f"{a} m={m}" for a in (SIMP, UNIF, "noise") for m in HORIZONS) + " |")
        L.append("|---|" + "---|" * (3 * len(HORIZONS)))
        for f in FILTERS:
            L.append(f"| {_mark(f)} | " + " | ".join(
                fmt(est(term(tab, a, tab.col(f, f"lawful_nt_m{m}"), keep))) for a in (SIMP, UNIF, "noise") for m in HORIZONS) + " |")

    # ---- M0, M1
    L.append(f"\n## M0 (instrument check) and M1 — law needs memory\n")
    L.append(f"M0: lawful_nt(m={mhi}) − lawful_nt(m={mlo}) within each simplicity world, should not be negative. "
             f"M1: that difference, simplicity − uniform-program; falsified if the interval lies below 0 for most counted filters.\n")
    for lab, keep in analyses:
        L.append(f"\n**{lab}**\n")
        L.append("| filter | M0: simplicity | M1: simplicity − uniform | verdict | M1, fresh programs only | verdict |")
        L.append("|---|---|---|---|---|---|")
        vs, vf, m0ok = {}, {}, True
        for f in FILTERS:
            d = tab.col(f, f"lawful_nt_m{mhi}") - tab.col(f, f"lawful_nt_m{mlo}")
            m0 = contrast([(1, term(tab, SIMP, d, keep))], boot)
            m1 = contrast([(1, term(tab, SIMP, d, keep)), (-1, term(tab, UNIF, d, keep))], boot)
            vs[f] = verdict(m1[1], m1[2])
            cell = "—", "—"
            if fresh is not None:
                mf = contrast([(1, term(tab, SIMP, d, keep & fresh)), (-1, term(tab, UNIF, d, keep & fresh))], boot)
                vf[f] = verdict(mf[1], mf[2])
                cell = ci(mf), vf[f]
            if f in COUNTED and np.isfinite(m0[0]) and m0[0] < -0.005:
                m0ok = False
            L.append(f"| {_mark(f)} | {ci(m0)} | {ci(m1)} | {vs[f]} | {cell[0]} | {cell[1]} |")
        ns, nc = tally(vs), tally(vs, "CONTRADICTED")
        summary[("M0", lab)] = "passes" if m0ok else "FAILS (check for a bug)"
        summary[("M1", lab)] = f"{ns}/4 supported, {nc}/4 contradicted: " + (
            "SUPPORTED" if ns >= 3 else "FALSIFIED" if nc >= 3 else "not supported")
        if vf:
            summary[("M1 fresh", lab)] = f"{tally(vf)}/4 supported, {tally(vf, 'CONTRADICTED')}/4 contradicted"

    # horizon-marginal lawfulness (secondary; description length 1 + EliasGamma(m))
    pm = np.array([2.0 ** -(1 + elias_gamma_len(m)) for m in HORIZONS])
    pm /= pm.sum()
    L.append("\n*Secondary — horizon-marginal lawful_nt (horizon prior " + ", ".join(
        f"m={m}: {p:.3f}" for m, p in zip(HORIZONS, pm)) + "), all worlds:* " + "; ".join(
        f"{f} simp {fmt(est(term(tab, SIMP, sum(p * tab.col(f, f'lawful_nt_m{m}') for m, p in zip(HORIZONS, pm)), ALL)))}"
        f" / unif {fmt(est(term(tab, UNIF, sum(p * tab.col(f, f'lawful_nt_m{m}') for m, p in zip(HORIZONS, pm)), ALL)))}"
        for f in COUNTED))

    # ---- S1
    L.append("\n## S1 — self-modelling emerges (anticipation gap)\n")
    L.append(f"Anticipation A = mean over k ∈ {set(ANTIC_LAGS)} of I(S_t; B_t+k | B_t) / H(B_t+k | B_t). Gap = an admitted "
             "observer's A minus the mean A of the same program at the other start positions in its world. "
             "Supported for a filter if the simplicity gap and (simplicity gap − noise gap) both have intervals above 0.\n")
    for lab, keep in analyses:
        L.append(f"\n**{lab}**\n")
        L.append("| filter | simplicity gap | noise gap | simplicity − noise | verdict | mean A admitted (simp) | diary (simp) | a_1 (simp) |")
        L.append("|---|---|---|---|---|---|---|---|")
        vs = {}
        for f in FILTERS:
            g = tab.col(f, "antic_gap")
            a, c = term(tab, SIMP, g, keep), term(tab, "noise", g, keep)
            ga = contrast([(1, a)], boot)
            if c[1].size:
                gd, gn = contrast([(1, a), (-1, c)], boot), fmt(est(c))
            else:                                      # as for H5: none admitted in noise -> noise gap 0
                gd, gn = ga, "0 (none admitted)"
            va, vd = verdict(ga[1], ga[2]), verdict(gd[1], gd[2])
            vs[f] = ("SUPPORTED" if va == vd == "SUPPORTED" else
                     "CONTRADICTED" if "CONTRADICTED" in (va, vd) else
                     "INSUFFICIENT DATA" if "INSUFFICIENT DATA" in (va, vd) else "INCONCLUSIVE")
            L.append(f"| {_mark(f)} | {ci(ga)} | {gn} | {ci(gd)} | {vs[f]} | "
                     + " | ".join(fmt(est(term(tab, SIMP, tab.col(f, m), keep))) for m in ("mean_antic", "mean_diary", "mean_a1")) + " |")
        summary[("S1", lab)] = f"{tally(vs)}/4: " + ("SUPPORTED" if tally(vs) >= 3 else "not supported")
    L.append("\n*Secondary — retention r_k = I(S_t; B_t−k | B_t) / H(B_t−k | B_t), simplicity arm, dynamically rich worlds:*\n")
    L.append("| filter | " + " | ".join(f"k={k}" for k in RETAIN_LAGS) + " |")
    L.append("|---|" + "---|" * len(RETAIN_LAGS))
    for f in FILTERS:
        L.append(f"| {_mark(f)} | " + " | ".join(fmt(est(term(tab, SIMP, tab.col(f, f"mean_ret{k}"), tab.dyn))) for k in RETAIN_LAGS) + " |")

    # ---- O1, O2
    L.append("\n## O1 and O2 — open-endedness (admitted non-trivial observers)\n")
    L.append("Depth = h^(2) − h^(8) (bits/symbol that memory beyond two steps buys). Learning progress = bits/symbol saved in "
             "the second half by continuing to learn. O1: each is higher in the simplicity arm than in noise, for most "
             "counted filters. O2: simplicity − uniform-program, reported without a prediction.\n")
    for lab, keep in analyses:
        L.append(f"\n**{lab}**\n")
        L.append("| filter | depth: simp | simp − noise | verdict | simp − uniform (O2) | progress: simp | simp − noise | verdict | simp − uniform (O2) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        vd, vp = {}, {}
        for f in FILTERS:
            cells = []
            for metric, store in (("mean_depth_nt", vd), ("mean_gap_nt", vp)):
                v = tab.col(f, metric)
                s = term(tab, SIMP, v, keep)
                dn = contrast([(1, s), (-1, term(tab, "noise", v, keep))], boot)
                du = contrast([(1, s), (-1, term(tab, UNIF, v, keep))], boot)
                store[f] = verdict(dn[1], dn[2])
                cells += [fmt(est(s)), ci(dn), store[f], f"{ci(du)} {verdict(du[1], du[2])}"]
            L.append(f"| {_mark(f)} | " + " | ".join(cells) + " |")
        nd, npg = tally(vd), tally(vp)
        summary[("O1", lab)] = f"depth {nd}/4, progress {npg}/4: " + ("SUPPORTED" if nd >= 3 and npg >= 3 else "not supported")

    S = ["\n## Summary (exploratory)\n", "| prediction | " + " | ".join(lab for lab, _ in analyses) + " |", "|---|---|---|"]
    for h in ("M0", "M1", "M1 fresh", "S1", "O1"):
        if any((h, lab) in summary for lab, _ in analyses):
            S.append(f"| {h} | " + " | ".join(summary.get((h, lab), "—") for lab, _ in analyses) + " |")
    L[head_at:head_at] = S
    path = os.path.join(out_dir, "report_3c.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(out_dir, "summary_3c.json"), "w") as f:
        json.dump({"|".join(k): v for k, v in summary.items()}, f, indent=1)
    print(f"[analyze3c] wrote {path}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="README 3c analysis (exploratory on 2a worlds).")
    ap.add_argument("out_dir")
    analyze3c(ap.parse_args(argv).out_dir)


if __name__ == "__main__":
    main()
