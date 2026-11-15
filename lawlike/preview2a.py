"""Phase 2a preview: reweight Phase 1 results under the Phase 2 rule priors (README 3b.6).

    python -m lawlike.preview2a results/full

At r = 1 both machines give every member of an ECA symmetry class the same weight
(rule_priors.check_class_invariance), so 2a's worlds are exactly Phase 1's, with different
rule weights. No new simulation. Labelled a preview because it inherits Phase 1's crossed
seeds and the complement approximation for embed seeds (README 6).

Arms built from the Phase 1 rows (README 3b.3):
  simp_<M>        simplicity arm, rule weight from machine M (seed prior unchanged)
  uic_<M>         Phase 1 uniform-arm worlds, rule weight from machine M  (uniform-IC)
  simplicity      Phase 1 simplicity arm (rule weights uniform)           (reference)
  uniform         Phase 1 uniform arm = uniform-program                   (Boltzmann)
  noise           unchanged

Each virtual row keeps the bootstrap index of the Phase 1 row it came from, so every
contrast (including simp_M_A − simplicity, the same worlds under two priors) is paired on
the same rules, seeds and replicates.

H5: Phase 1 rows predate the aligned-MI statistic, so the pre-registered H5 cannot be
computed in the preview. The NCD version (secondary) is reported.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .analyze import (CHANCE_JACCARD, RICH_BLOCK_ENTROPY, _arrays, _est, _mark, _world_arrays,
                      contrast_ci, diff_ci, fmt, load, verdict, wmean_ci)
from .bootstrap import ClusterBootstrap
from .experiment1 import DEPRECATED_FILTERS, FILTERS
from .rule_priors import class_weights, entropy_bits, PRIORS

MACHINES = ("M_A", "M_B")
COUNTED = [f for f in FILTERS if f != "none" and f not in DEPRECATED_FILTERS]
SCORED = [f for f in FILTERS if f != "none"]
PHASE1_NONRICH_SHARE = 0.88   # H6 threshold, fixed in README 3b.5


def build_rows(rows):
    """Phase 1 rows plus reweighted virtual arms (sharing bootstrap indices)."""
    out = list(rows)
    p1 = class_weights("phase1")
    for m in MACHINES:
        cw = class_weights(m)
        for src, dst in (("simplicity", f"simp_{m}"), ("uniform", f"uic_{m}")):
            sub = [r for r in rows if r["world"]["arm"] == src]
            neww = [r["world"]["weight"] * cw[r["world"]["rule"]] / p1[r["world"]["rule"]] for r in sub]
            Z = sum(neww)
            for r, w in zip(sub, neww):
                out.append({**r, "world": {**r["world"], "arm": dst, "weight": w / Z}})
    return out


def is_rich(r):
    return r["world_stats"]["block_entropy"] >= RICH_BLOCK_ENTROPY


def preview(out_dir: str, resample_rules: bool = True) -> str:
    base = load(out_dir)
    cfg = json.load(open(os.path.join(out_dir, "config.json"))).get("config", {})
    for i, r in enumerate(base):
        r["_i"] = i
    boot = ClusterBootstrap([r["world"] for r in base], int(cfg.get("ic_exact_len", 0)),
                            np.random.default_rng(0), resample_rules=resample_rules)
    rows = build_rows(base)
    rich = [r for r in rows if is_rich(r) or r["world"]["arm"] == "noise"]
    analyses = (("all worlds", rows), ("rich worlds", rich))

    def lawful(sub, arm, filt, mode="SSA"):
        return _arrays(sub, arm, filt, "lawful_nt", mode)

    def ci(d):
        return f"{fmt(d[0])} [{fmt(d[1], 2)}, {fmt(d[2], 2)}]"

    def tally(vs):
        return sum(v == "SUPPORTED" for f, v in vs.items() if f in COUNTED)

    L = [f"# Phase 2a preview — reweighted `{os.path.basename(os.path.abspath(out_dir))}`\n"]
    L.append("**Preview, not confirmation.** Phase 1 worlds reweighted under the Phase 2 rule priors "
             "(README §3b.6). Inherits Phase 1's crossed seeds and the complement approximation for "
             "embed seeds. Verdicts here are indicative; the 2a confirmation run is what counts.\n")
    L.append(f"Intervals: 95% two-way cluster bootstrap "
             f"({'rules × sampled seeds' if resample_rules else 'rules held fixed'}), all arms on the same "
             "replicates, so every contrast is paired. Counted filters: " + ", ".join(COUNTED) +
             "; \"most\" = at least 3 of 4. † = deprecated, reported only. Verdicts use SSA.\n")

    # ---- the priors themselves
    L.append("## The rule priors\n")
    L.append("| prior | rule-weight entropy (bits) | top classes (rep: class weight) |")
    L.append("|---|---|---|")
    for name in ("phase1",) + MACHINES:
        cw = class_weights(name)
        top = sorted(cw.items(), key=lambda kv: -kv[1])[:6]
        L.append(f"| {name} | {entropy_bits(PRIORS[name]()):.2f} | " +
                 ", ".join(f"{k}: {v:.3f}" for k, v in top) + " |")

    # ---- where the weight sits
    L.append("\n## Weight in rich worlds (spatial 3-block entropy ≥ 0.5)\n")
    L.append("| arm | share of weight in rich worlds | non-rich share |")
    L.append("|---|---|---|")
    arm_list = ["simplicity", "simp_M_A", "simp_M_B", "uniform", "uic_M_A", "uic_M_B"]
    nonrich = {}
    for arm in arm_list:
        a = _world_arrays(rows, arm, lambda r: 0.0 if is_rich(r) else 1.0)
        e = wmean_ci(a, boot)
        nonrich[arm] = e
        L.append(f"| {arm} | {fmt(1 - e[0])} | {ci(e)} |")

    # ---- lawfulness table
    for mode in ("SSA", "SIA"):
        L.append(f"\n## lawful_nt by arm ({mode}, all worlds)\n")
        L.append("| filter | " + " | ".join(arm_list + ["noise"]) + " |")
        L.append("|---|" + "---|" * (len(arm_list) + 1))
        for filt in FILTERS:
            L.append(f"| {_mark(filt)} | " +
                     " | ".join(fmt(_est(lawful(rows, a, filt, mode))) for a in arm_list + ["noise"]) + " |")

    # ---- hypotheses
    summary = {}   # (hyp, machine, analysis) -> verdict string
    L.append("\n## H1 — simplicity > uniform-program (Boltzmann) > noise\n")
    for m in MACHINES:
        for label, sub in analyses:
            L.append(f"\n**{m}, {label}**\n")
            L.append("| filter | simp | unif-program | simp − unif | verdict | simp − noise | verdict |")
            L.append("|---|---|---|---|---|---|---|")
            v1s = {}
            for filt in FILTERS:
                a, b, c = lawful(sub, f"simp_{m}", filt), lawful(sub, "uniform", filt), lawful(sub, "noise", filt)
                d1, d2 = diff_ci(a, b, boot), diff_ci(a, c, boot)
                v1s[filt] = verdict(d1[1], d1[2])
                L.append(f"| {_mark(filt)} | {fmt(_est(a))} | {fmt(_est(b))} | {ci(d1)} | {v1s[filt]} | "
                         f"{ci(d2)} | {verdict(d2[1], d2[2])} |")
            n = tally(v1s)
            summary[("H1", m, label)] = f"{n}/4 → {'SUPPORTED' if n >= 3 else 'not supported'}"
            L.append(f"\nsimp > unif SUPPORTED for {n} of 4 counted filters.")
        # SIA headline, reported not voted
        sia = {f: verdict(*diff_ci(lawful(rows, f"simp_{m}", f, "SIA"), lawful(rows, "uniform", f, "SIA"), boot)[1:])
               for f in FILTERS}
        L.append(f"\n*SIA, all worlds ({m}):* simp > unif SUPPORTED for {tally(sia)} of 4 counted filters.\n")

    L.append("\n## H2 — trivial-observer catch (simplicity arm under each machine)\n")
    L.append("| filter | null share M_A | trivial M_A | null share M_B | trivial M_B |")
    L.append("|---|---|---|---|---|")
    for filt in FILTERS:
        cells = []
        for m in MACHINES:
            for met in ("famshare_null", "frac_trivial"):
                cells.append(fmt(_est(_arrays(rows, f"simp_{m}", filt, met, "SSA"))))
        L.append(f"| {_mark(filt)} | " + " | ".join(cells) + " |")

    L.append(f"\n## H3 — convergence of criteria (mean pairwise Jaccard vs chance {CHANCE_JACCARD:.3f})\n")
    L.append("| machine | analysis | counted filters | verdict | all filters (as in §3.6) | verdict |")
    L.append("|---|---|---|---|---|---|")
    for m in MACHINES:
        for label, sub in analyses:
            cells = []
            for keep in (COUNTED, SCORED):
                pairs = sorted({k for r in sub if r["world"]["arm"] == f"simp_{m}" for k in r["jaccard"]
                                if all(x in keep for x in k.split("|"))})
                arrs = [_world_arrays(sub, f"simp_{m}", lambda r, k=k: r["jaccard"].get(k)) for k in pairs]
                arrs = [a for a in arrs if a[1].size >= 3]
                if not arrs:
                    cells += ["—", "INSUFFICIENT DATA"]
                    continue
                est = float(np.mean([_est(a) for a in arrs]))
                d = np.mean([boot.means(*a) for a in arrs], axis=0)
                d = d[np.isfinite(d)]
                lo, hi = np.quantile(d, 0.025), np.quantile(d, 0.975)
                v = verdict(lo - CHANCE_JACCARD, hi - CHANCE_JACCARD)
                cells += [f"{fmt(est)} [{fmt(lo, 2)}, {fmt(hi, 2)}]", v]
            summary[("H3", m, label)] = cells[1]
            L.append(f"| {m} | {label} | " + " | ".join(cells) + " |")

    L.append("\n## H4 — facts change, laws hold (simplicity arm vs noise, all worlds)\n")
    L.append("| filter | facts changed M_A | …laws held M_A | facts changed M_B | …laws held M_B | facts changed noise | …laws held noise |")
    L.append("|---|---|---|---|---|---|---|")
    for filt in FILTERS:
        cells = []
        for arm in ("simp_M_A", "simp_M_B", "noise"):
            for met in ("facts_change_nt", "facts_change_laws_hold_nt"):
                cells.append(fmt(_est(_arrays(rows, arm, filt, met, "SSA"))))
        L.append(f"| {_mark(filt)} | " + " | ".join(cells) + " |")
    L.append("\nH4 has no interval-based verdict in the pre-registration (§3.6 states it descriptively); "
             "reported as in Phase 1.")

    L.append("\n## H5 — intersubjectivity\n")
    L.append("**The pre-registered statistic (aligned MI) is not in the Phase 1 rows and cannot be "
             "computed in the preview.** It first appears in the confirmation run. Shown: the NCD "
             "version (secondary), gap paired per world, verdict on simplicity gap − noise gap.\n")
    L.append("| filter | " + " | ".join(f"{m} {lab}" for m in MACHINES for lab, _ in analyses) + " |")
    L.append("|---|" + "---|" * 4)

    def ncd_gap(filt):
        def fn(r):
            x, y = r["filters"][filt].get("ncd_base"), r["filters"][filt].get("ncd_adm")
            return None if x is None or y is None else x - y
        return fn
    for filt in FILTERS:
        cells = []
        for m in MACHINES:
            for _, sub in analyses:
                a = _world_arrays(sub, f"simp_{m}", ncd_gap(filt))
                c = _world_arrays(sub, "noise", ncd_gap(filt))
                d = wmean_ci(a, boot) if (c[1].size == 0 and a[1].size) else diff_ci(a, c, boot)
                cells.append(f"{ci(d)} {verdict(d[1], d[2])}")
        L.append(f"| {_mark(filt)} | " + " | ".join(cells) + " |")

    L.append(f"\n## H6 — dead worlds grow (non-rich share of simplicity weight under M_A > {PHASE1_NONRICH_SHARE})\n")
    e = nonrich["simp_M_A"]
    # the pre-registered threshold is a fixed number; also show the paired change vs the Phase 1 weighting
    d = contrast_ci([(1, _world_arrays(rows, "simp_M_A", lambda r: 0.0 if is_rich(r) else 1.0)),
                     (-1, _world_arrays(rows, "simplicity", lambda r: 0.0 if is_rich(r) else 1.0))], boot)
    v6 = verdict(e[1] - PHASE1_NONRICH_SHARE, e[2] - PHASE1_NONRICH_SHARE)
    summary[("H6", "M_A", "—")] = f"{fmt(e[0])} → {'falsified' if e[0] <= PHASE1_NONRICH_SHARE else 'holds'} (interval vs 0.88: {v6})"
    L.append(f"- M_A non-rich share: {ci(e)}. Pre-registered falsifier: ≤ {PHASE1_NONRICH_SHARE}. "
             f"Point estimate {'≤' if e[0] <= PHASE1_NONRICH_SHARE else '>'} threshold; interval vs threshold: **{v6}**.")
    L.append(f"- Paired change vs the Phase 1 weighting (same worlds): {ci(d)}.")
    eb = nonrich["simp_M_B"]
    L.append(f"- M_B (sensitivity, not pre-registered as a test): non-rich share {ci(eb)}.")

    L.append("\n## H7 — a simplicity prior over dynamics makes lawfulness more typical\n")
    for m in MACHINES:
        for label, sub in analyses:
            L.append(f"\n**{m}, {label}**\n")
            L.append("| filter | (a) simp_M − simp_phase1 | verdict | (b) uniform-IC_M − uniform-program | verdict |")
            L.append("|---|---|---|---|---|")
            va, vb = {}, {}
            for filt in FILTERS:
                da = diff_ci(lawful(sub, f"simp_{m}", filt), lawful(sub, "simplicity", filt), boot)
                db = diff_ci(lawful(sub, f"uic_{m}", filt), lawful(sub, "uniform", filt), boot)
                va[filt], vb[filt] = verdict(da[1], da[2]), verdict(db[1], db[2])
                L.append(f"| {_mark(filt)} | {ci(da)} | {va[filt]} | {ci(db)} | {vb[filt]} |")
            na, nb = tally(va), tally(vb)
            summary[("H7a", m, label)] = f"{na}/4 → {'SUPPORTED' if na >= 3 else 'not supported'}"
            summary[("H7b", m, label)] = f"{nb}/4 → {'SUPPORTED' if nb >= 3 else 'not supported'}"
            L.append(f"\n(a) SUPPORTED for {na}/4, (b) SUPPORTED for {nb}/4 counted filters.")

    # ---- summary up top
    S = ["\n## Summary of verdicts (SSA)\n", "| hypothesis | M_A all | M_A rich | M_B all | M_B rich |",
         "|---|---|---|---|---|"]
    for h in ("H1", "H3", "H7a", "H7b"):
        S.append(f"| {h} | " + " | ".join(summary.get((h, m, lab), "—") for m in MACHINES for lab, _ in analyses) + " |")
    S.append(f"| H6 | {summary[('H6', 'M_A', '—')]} | | | |")
    S.append("| H2, H4 | descriptive (tables below) | | | |")
    S.append("| H5 | pre-registered statistic unavailable in preview; NCD version below | | | |")
    L[3:3] = S

    report = "\n".join(L) + "\n"
    path = os.path.join(out_dir, "report_preview2a.md")
    with open(path, "w") as f:
        f.write(report)
    print(f"[preview2a] wrote {path}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", nargs="?", default="results/full")
    ap.add_argument("--fixed-rules", action="store_true")
    a = ap.parse_args(argv)
    preview(a.out_dir, resample_rules=not a.fixed_rules)


if __name__ == "__main__":
    main()
