"""Phase 2a analysis: hypotheses H1-H7 under the Phase 2 rule priors (README 3b, amendments 3b.7).

    python -m lawlike.analyze2a results/full2a
    python -m lawlike.analyze2a results/full2a --phase1 results/full      # adds the "vs Phase 1" table

Arms (README 3b.3), built from the simulated worlds by reweighting rules; every prior used
here is class-invariant (rule_priors.check_class_invariance), so this is exact:
  simp_<P>   simplicity arm, rule prior P in {phase1, M_A, M_B, M_A*}
  uic_<P>    uniform-IC: uniform-arm worlds, rule prior P
  uniform    uniform-program: the uniform arm under the phase1 (uniform) rule weights
  noise      no-law baseline

Analyses: all worlds and dynamically rich worlds are co-primary (A1); spatially rich worlds
are secondary. Headline (A6): H1, H7a, H7b under M_A in the two co-primary analyses, SSA,
each with a leave-one-class-out fragility label (A4) and a fixed-rule interval (A5).
Everything else is secondary and is reported in full.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .analyze import CHANCE_JACCARD, RICH_BLOCK_ENTROPY, fmt, load, program_key, verdict
from .bootstrap import ClusterBootstrap
from .experiment1 import DEPRECATED_FILTERS, FILTERS
from .rule_priors import class_weights

PRIORS = ("phase1", "M_A", "M_B", "M_A*")
COUNTED = tuple(f for f in FILTERS if f != "none" and f not in DEPRECATED_FILTERS)
SCORED = tuple(f for f in FILTERS if f != "none")
PHASE1_NONRICH_SHARE = 0.88    # H6 threshold (README 3b.5)
THIN_SHARE = 0.02              # A1: flag an analysis resting on < 2% of an arm's weight
HEADLINE = (("H1", "simp_M_A", "uniform"), ("H7a", "simp_M_A", "simp_phase1"), ("H7b", "uic_M_A", "uniform"))


# ----------------------------------------------------------------------------- data

def ensure_dyn(rows, cfg, out_dir):
    """Phase 1 rows predate the translation-period statistic: recompute it by re-simulating
    (world properties only), cached in <out_dir>/world_tperiod.json."""
    if all("dyn_rich" in r["world_stats"] for r in rows):
        return rows
    cache = os.path.join(out_dir, "world_tperiod.json")
    if os.path.exists(cache):
        tp = json.load(open(cache))
    else:
        from multiprocessing import get_context
        todo = [r["world"] for r in rows]
        with get_context("spawn").Pool(initializer=_tp_init, initargs=(cfg,)) as pool:
            res = pool.map(_tp_one, todo, chunksize=16)
        tp = {w["world_id"]: v for w, v in zip(todo, res)}
        json.dump(tp, open(cache, "w"))
    pmax = int(cfg.get("dyn_pmax", 16))
    for r in rows:
        p = tp[r["world"]["world_id"]]
        r["world_stats"]["tperiod"] = p
        r["world_stats"]["dyn_rich"] = bool(p is None or p > pmax)
    return rows


_TPCFG = None


def _tp_init(cfg):
    global _TPCFG
    from types import SimpleNamespace
    _TPCFG = SimpleNamespace(**cfg)


def _tp_one(w):
    from .experiment1 import TPERIOD_SCAN
    from .worlds import World, realise, translation_period
    world = World(**{k: v for k, v in w.items() if k in World.__dataclass_fields__})
    st, _, _ = realise(world, _TPCFG)
    tp = translation_period(st[_TPCFG.T0:_TPCFG.T0 + _TPCFG.T], TPERIOD_SCAN)
    return None if tp is None else tp[0]


class Table:
    """Column view of the results: one entry per simulated world."""

    def __init__(self, rows):
        self.n = len(rows)
        self.arm = np.array([r["world"]["arm"] for r in rows])
        self.rule = np.array([r["world"]["rule"] for r in rows])
        self.w0 = np.array([r["world"]["weight"] for r in rows], float)
        self.dyn = np.array([bool(r["world_stats"].get("dyn_rich", False)) for r in rows])
        self.srich = np.array([r["world_stats"]["block_entropy"] >= RICH_BLOCK_ENTROPY for r in rows])
        self.rows = rows
        p1 = class_weights("phase1")
        self.wp = {}
        for P in PRIORS:
            cw = class_weights(P)
            f = np.array([cw[k] / p1[k] if k >= 0 else 1.0 for k in self.rule])
            self.wp[P] = self.w0 * f
        self._cache = {}

    def col(self, filt, metric):
        key = (filt, metric)
        if key not in self._cache:
            v = np.full(self.n, np.nan)
            for i, r in enumerate(self.rows):
                x = r["filters"][filt].get(metric)
                if x is not None:
                    v[i] = x
            self._cache[key] = v
        return self._cache[key]

    def world_col(self, fn):
        v = np.full(self.n, np.nan)
        for i, r in enumerate(self.rows):
            x = fn(r)
            if x is not None:
                v[i] = x
        return v

    def arm_spec(self, name):
        """(mask, weights) of a virtual arm."""
        if name in ("uniform", "noise"):
            return self.arm == name, self.w0
        kind, P = name.split("_", 1)
        src = "simplicity" if kind == "simp" else "uniform"
        return self.arm == src, self.wp[P]


def term(tab, arm, values, keep, mode="SSA", filt=None):
    """(idx, w, v) for one arm: rows in keep with finite values and positive weight."""
    m, w = tab.arm_spec(arm)
    if mode == "SIA" and filt is not None:
        w = w * np.nan_to_num(tab.col(filt, "mass"))
    ok = m & keep & np.isfinite(values) & (w > 0)
    idx = np.where(ok)[0]
    return idx, w[idx], values[idx]


def est(t):
    _, w, v = t
    return float(np.sum(w * v) / np.sum(w)) if w.size else float("nan")


def contrast(terms, boot):
    """Estimate and 95% interval of sum(sign * weighted mean); all terms on the same replicates."""
    if not all(t[1].size for _, t in terms):
        return (float("nan"),) * 3
    e = sum(s * est(t) for s, t in terms)
    if any(t[1].size < 3 for _, t in terms):
        return e, float("nan"), float("nan")
    d = sum(s * boot.means(*t) for s, t in terms)
    d = d[np.isfinite(d)]
    if d.size < 10:
        return e, float("nan"), float("nan")
    return float(e), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def ci(d):
    return f"{fmt(d[0])} [{fmt(d[1], 2)}, {fmt(d[2], 2)}]"


def hyp_verdict(n):
    return "SUPPORTED" if n >= 3 else "not supported"


# ----------------------------------------------------------------------------- hypotheses

def pair_test(tab, boot, a, b, keep, mode="SSA", filters=FILTERS):
    """Per-filter lawful_nt(a) - lawful_nt(b) on the rows in keep."""
    out = {}
    for f in filters:
        v = tab.col(f, "lawful_nt")
        d = contrast([(1, term(tab, a, v, keep, mode, f)), (-1, term(tab, b, v, keep, mode, f))], boot)
        out[f] = (d, verdict(d[1], d[2]))
    return out


def tally(res):
    return sum(res[f][1] == "SUPPORTED" for f in COUNTED)


def fragility(tab, boot, a, b, keep, base_n):
    """A4: classes whose removal changes the hypothesis verdict, plus per-filter flips."""
    flips_h, flips_f = [], {}
    base = pair_test(tab, boot, a, b, keep)
    for k in sorted(set(tab.rule[tab.rule >= 0])):
        res = pair_test(tab, boot, a, b, keep & (tab.rule != k), filters=COUNTED)
        n = tally(res)
        if hyp_verdict(n) != hyp_verdict(base_n):
            flips_h.append((int(k), n))
        for f in COUNTED:
            if res[f][1] != base[f][1]:
                flips_f.setdefault(f, []).append(int(k))
    return flips_h, flips_f


def h3(tab, boot, arm, keep, names):
    pairs = sorted({k for r in tab.rows for k in r["jaccard"] if all(x in names for x in k.split("|"))})
    ts = []
    for k in pairs:
        t = term(tab, arm, tab.world_col(lambda r, k=k: r["jaccard"].get(k)), keep)
        if t[1].size >= 3:
            ts.append(t)
    if not ts:
        return (float("nan"),) * 3
    e = float(np.mean([est(t) for t in ts]))
    d = np.mean([boot.means(*t) for t in ts], axis=0)
    d = d[np.isfinite(d)]
    return e, float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def h5(tab, boot, arm, keep, filt, base="xmi_base", adm="xmi_adm"):
    sign = -1 if base.startswith("ncd") else 1
    v = sign * (tab.col(filt, adm) - tab.col(filt, base))
    a, c = term(tab, arm, v, keep), term(tab, "noise", v, keep)
    if c[1].size == 0 and a[1].size:     # README 3b.4: no admitted pairs in noise -> noise gap 0
        return contrast([(1, a)], boot), "0 (none admitted)"
    return contrast([(1, a), (-1, c)], boot), fmt(est(c))


# ----------------------------------------------------------------------------- report

def analyze2a(out_dir, phase1_dir=None, do_fragility=True):
    rows = load(out_dir)
    cfg = json.load(open(os.path.join(out_dir, "config.json")))["config"]
    rows = ensure_dyn(rows, cfg, out_dir)
    nested = cfg.get("seed_mode") == "per_rule"
    ws = [r["world"] for r in rows]
    exact = int(cfg.get("ic_exact_len", 0))
    boot = ClusterBootstrap(ws, exact, np.random.default_rng(0), nested=nested)
    boot_fixed = ClusterBootstrap(ws, exact, np.random.default_rng(0), resample_rules=False, nested=nested)
    tab = Table(rows)
    ALL = np.ones(tab.n, bool)
    noise = tab.arm == "noise"
    DYN, SR = tab.dyn | noise, tab.srich | noise
    co = (("all worlds", ALL), ("dynamically rich", DYN))
    analyses = co + (("spatially rich (secondary)", SR),)

    L = [f"# Phase 2a report — `{os.path.basename(os.path.abspath(out_dir))}`\n"]
    n_arm = {a: int((tab.arm == a).sum()) for a in ("simplicity", "uniform", "noise")}
    L.append(f"Worlds: simplicity {n_arm['simplicity']}, uniform {n_arm['uniform']}, noise {n_arm['noise']}. "
             f"Seeds {'nested within rules' if nested else 'crossed with rules'}; "
             f"complement approximation {'removed' if cfg.get('complement_embed') else 'in use'}. "
             "Intervals: 95% cluster bootstrap with rules resampled (decides verdicts); "
             "fixed-rule intervals shown for the headline (README §3b.7 A5). Verdicts use SSA. "
             "Counted filters: " + ", ".join(COUNTED) + "; \"most\" = at least 3 of 4. † = deprecated.\n")
    if boot.single_seed_strata:
        L.append(f"*{boot.n_single} of {boot.n_strata} seed strata hold a single sampled seed (lengths "
                 f"{boot.single_seed_strata}), usually because the other draw (e.g. 0000) produced the same world as "
                 "a shorter seed and was merged into it. Their seed variance is missing from the intervals.*\n")

    head_at = len(L)

    # ---- disjointness from Phase 1
    excl_dirs = tuple(cfg.get("exclude_programs_from") or ())
    if excl_dirs and all(os.path.exists(os.path.join(d, "results.jsonl")) for d in excl_dirs):
        old = {program_key(r["world"]) for d in excl_dirs for r in load(d) if r["world"]["arm"] == "simplicity"}
        simp = np.where(tab.arm == "simplicity")[0]
        shared = np.array([program_key(rows[i]["world"]) in old for i in simp])
        L.append("## Fresh data check (README §5.1)\n")
        L.append(f"Sampled seeds were drawn only from programs absent from {', '.join(f'`{d}`' for d in excl_dirs)}. "
                 "The enumerated short seeds are the whole population and are necessarily shared. "
                 "Share of simplicity weight in programs that also appear in Phase 1:\n")
        L.append("| prior | shared with Phase 1 | fresh |")
        L.append("|---|---|---|")
        for P in PRIORS:
            w = tab.wp[P][simp]
            s = float(w[shared].sum() / w.sum())
            L.append(f"| {P} | {s:.1%} | {1 - s:.1%} |")
        L.append("\nUnder A8 a headline verdict counts as **confirmed** only if it holds on all programs "
                 "and on fresh programs alone (the headline table shows both).\n")
        fresh_keep = np.ones(tab.n, bool)
        fresh_keep[simp[shared]] = False
    else:
        fresh_keep = None

    # ---- headline
    L.append("## Headline (README §3b.7 A6): M_A, SSA\n")
    L.append("| hypothesis | analysis | all programs | fresh programs only | confirmed? (A8) | fragile? (A4) |")
    L.append("|---|---|---|---|---|---|")
    detail = []
    summary = {}
    for h, a, b in HEADLINE:
        for lab, keep in co:
            res = pair_test(tab, boot, a, b, keep)
            resf = pair_test(tab, boot_fixed, a, b, keep)
            n = tally(res)
            frag = "—"
            if do_fragility:
                fh, ff = fragility(tab, boot, a, b, keep, n)
                frag = ("**fragile**: dropping class " + ", ".join(f"{k} → {m}/4" for k, m in fh[:6])
                        + (" …" if len(fh) > 6 else "")) if fh else "robust"
                if ff:
                    frag += "; per-filter flips: " + "; ".join(f"{f}: {len(v)} classes ({', '.join(map(str, v[:5]))}{' …' if len(v) > 5 else ''})"
                                                               for f, v in ff.items())
            summary[(h, "M_A", lab)] = f"{n}/4 {hyp_verdict(n)}"
            if fresh_keep is not None:
                nf = tally(pair_test(tab, boot, a, b, keep & fresh_keep, filters=COUNTED))
                conf = "**CONFIRMED**" if (n >= 3 and nf >= 3) else (
                    "not confirmed (fresh programs disagree)" if n >= 3 else "no")
                fcell = f"{nf}/4 {hyp_verdict(nf)}"
                summary[(h, "M_A fresh", lab)] = fcell
            else:
                conf, fcell = "n/a (no exclusion directories)", "—"
            summary[(h, "M_A confirmed", lab)] = conf.strip("*")
            L.append(f"| {h} | {lab} | {n}/4 {hyp_verdict(n)} | {fcell} | {conf} | {frag} |")
            detail.append((h, lab, a, b, res, resf))
    L.append("\nH7b uses only uniform-arm worlds, which are all fresh random ICs, so its two columns "
             "agree by construction.")
    L.append("\nH1 = simplicity(M_A) − uniform-program; H7a = simplicity(M_A) − simplicity(Phase 1 rule weights), "
             "same worlds; H7b = uniform-IC(M_A) − uniform-program. Per-filter detail, with fixed-rule intervals:\n")
    for h, lab, a, b, res, resf in detail:
        L.append(f"\n**{h}, {lab}** ({a} − {b})\n")
        L.append("| filter | difference [rules resampled] | verdict | [rules fixed] | verdict |")
        L.append("|---|---|---|---|---|")
        for f in FILTERS:
            (d, v), (df, vf) = res[f], resf[f]
            L.append(f"| {_mark(f)} | {ci(d)} | {v} | [{fmt(df[1], 2)}, {fmt(df[2], 2)}] | {vf} |")

    # ---- weight in rich worlds (A1 thin-share flag, H6, H6d)
    L.append("\n## Where the weight sits\n")
    L.append("| arm | dynamically rich | spatially rich | flag |")
    L.append("|---|---|---|---|")
    thin = set()
    for P in PRIORS:
        for kind in ("simp", "uic"):
            arm = f"{kind}_{P}"
            m, w = tab.arm_spec(arm)
            dshare = float(w[m & tab.dyn].sum() / w[m].sum())
            sshare = float(w[m & tab.srich].sum() / w[m].sum())
            flag = f"< {THIN_SHARE:.0%}: dynamically rich analysis rests on a sliver" if dshare < THIN_SHARE else ""
            if flag:
                thin.add(arm)
            L.append(f"| {arm} | {dshare:.3f} | {sshare:.3f} | {flag} |")

    L.append("\n## H6 and H6d — dead worlds grow\n")
    nonrich_s = (~tab.srich).astype(float)
    nonrich_d = (~tab.dyn).astype(float)
    e6 = contrast([(1, term(tab, "simp_M_A", nonrich_s, ALL))], boot)
    v6 = "falsified" if e6[0] <= PHASE1_NONRICH_SHARE else "holds"
    L.append(f"- **H6** (pre-registered, spatial): M_A non-rich share {ci(e6)}; falsified if ≤ {PHASE1_NONRICH_SHARE}. "
             f"Point estimate: **{v6}**; interval vs threshold: {verdict(e6[1] - PHASE1_NONRICH_SHARE, e6[2] - PHASE1_NONRICH_SHARE)}.")
    d6 = contrast([(1, term(tab, "simp_M_A", nonrich_d, ALL)), (-1, term(tab, "simp_phase1", nonrich_d, ALL))], boot)
    v6d = verdict(d6[1], d6[2])
    L.append(f"- **H6d** (A2, paired, dynamical): non-dynamically-rich share under M_A − under Phase 1 weights = {ci(d6)}: "
             f"**{v6d}**{' (falsified)' if v6d == 'CONTRADICTED' else ''}.")
    summary[("H6", "M_A", "—")] = v6
    summary[("H6d", "M_A", "—")] = v6d

    # ---- secondary: the same three hypotheses under every prior and analysis
    L.append("\n## Secondary: H1 / H7a / H7b under every prior and analysis (SSA, then SIA)\n")
    L.append("| hypothesis | prior | " + " | ".join(lab for lab, _ in analyses) + " |")
    L.append("|---|---|" + "---|" * len(analyses))
    for h, a0, b0 in HEADLINE:
        for P in ("M_A", "M_B", "M_A*"):
            a = a0.replace("M_A", P)
            cells = []
            for lab, keep in analyses:
                n = tally(pair_test(tab, boot, a, b0, keep))
                ns = tally(pair_test(tab, boot, a, b0, keep, "SIA"))
                t = " ⚠ thin" if (lab == "dynamically rich" and a in thin) else ""
                cells.append(f"{n}/4 (SIA {ns}/4){t}")
                summary.setdefault((h, P, lab), f"{n}/4 {hyp_verdict(n)}")
            L.append(f"| {h} | {P} | " + " | ".join(cells) + " |")
    if fresh_keep is not None:
        L.append("\n*Fresh programs only (simplicity programs shared with Phase 1 dropped), M_A:* " + "; ".join(
            f"{h} {lab}: {tally(pair_test(tab, boot, a, b, keep & fresh_keep))}/4" for h, a, b in HEADLINE for lab, keep in co))

    # ---- lawfulness table
    arms_all = [f"simp_{P}" for P in PRIORS] + ["uniform"] + [f"uic_{P}" for P in PRIORS if P != "phase1"] + ["noise"]
    for lab, keep in analyses[:2]:
        L.append(f"\n## lawful_nt by arm (SSA, {lab})\n")
        L.append("| filter | " + " | ".join(arms_all) + " |")
        L.append("|---|" + "---|" * len(arms_all))
        for f in FILTERS:
            v = tab.col(f, "lawful_nt")
            L.append(f"| {_mark(f)} | " + " | ".join(fmt(est(term(tab, a, v, keep))) for a in arms_all) + " |")

    # ---- H2
    L.append("\n## H2 — trivial-observer catch (simplicity arm, all worlds)\n")
    L.append("| filter | " + " | ".join(f"null {P} | trivial {P}" for P in ("M_A", "M_B")) + " |")
    L.append("|---|" + "---|" * 4)
    for f in FILTERS:
        cells = [fmt(est(term(tab, f"simp_{P}", tab.col(f, m), ALL)))
                 for P in ("M_A", "M_B") for m in ("famshare_null", "frac_trivial")]
        L.append(f"| {_mark(f)} | " + " | ".join(cells) + " |")

    # ---- H3
    L.append(f"\n## H3 — convergence of criteria (mean pairwise Jaccard vs chance {CHANCE_JACCARD:.3f})\n")
    L.append("| prior | analysis | counted filters | verdict | all filters | verdict |")
    L.append("|---|---|---|---|---|---|")
    for P in ("M_A", "M_B"):
        for lab, keep in analyses:
            cells = []
            for names in (COUNTED, SCORED):
                e, lo, hi = h3(tab, boot, f"simp_{P}", keep, names)
                v = verdict(lo - CHANCE_JACCARD, hi - CHANCE_JACCARD)
                cells += [f"{fmt(e)} [{fmt(lo, 2)}, {fmt(hi, 2)}]", v]
            summary[("H3", P, lab)] = cells[1]
            L.append(f"| {P} | {lab} | " + " | ".join(cells) + " |")

    # ---- H4
    L.append("\n## H4 — facts change, laws hold (all worlds; descriptive, as in §3.6)\n")
    L.append("| filter | facts changed M_A | …laws held M_A | facts changed noise | …laws held noise |")
    L.append("|---|---|---|---|---|")
    for f in FILTERS:
        cells = [fmt(est(term(tab, a, tab.col(f, m), ALL))) for a in ("simp_M_A", "noise")
                 for m in ("facts_change_nt", "facts_change_laws_hold_nt")]
        L.append(f"| {_mark(f)} | " + " | ".join(cells) + " |")

    # ---- H5
    have_xmi = any(r["filters"]["none"].get("xmi_adm") is not None for r in rows)
    L.append("\n## H5 — intersubjectivity (aligned MI, pre-registered §3b.4; verdict on simplicity gap − noise gap)\n")
    for P in ("M_A", "M_B"):
        L.append(f"\n**{P}**\n")
        L.append("| filter | " + " | ".join(f"{lab}: simp − noise" for lab, _ in analyses[:2]) + " | NCD (secondary), all worlds |")
        L.append("|---|---|---|---|")
        n5 = {lab: 0 for lab, _ in analyses[:2]}
        for f in FILTERS:
            cells = []
            for lab, keep in analyses[:2]:
                d, _ = h5(tab, boot, f"simp_{P}", keep, f)
                v = verdict(d[1], d[2])
                n5[lab] += (f in COUNTED and v == "SUPPORTED")
                cells.append(f"{ci(d)} {v}")
            dn, _ = h5(tab, boot, f"simp_{P}", ALL, f, "ncd_base", "ncd_adm")
            cells.append(f"{ci(dn)} {verdict(dn[1], dn[2])}")
            L.append(f"| {_mark(f)} | " + " | ".join(cells) + " |")
        for lab, _ in analyses[:2]:
            summary[("H5", P, lab)] = (f"{n5[lab]}/4 {hyp_verdict(n5[lab])}" if have_xmi
                                       else "n/a (aligned MI not in these results)")

    # ---- where lawful observers live
    L.append("\n## Where the lawful observers live (simplicity, M_A, SSA)\n")
    _, wA = tab.arm_spec("simp_M_A")
    for f in ("individuality", "integration_excl", "causal_complexity"):
        fl = np.nan_to_num(tab.col(f, "frac_lawful"))
        m = tab.arm == "simplicity"
        contrib = {}
        for k in set(tab.rule[m]):
            sel = m & (tab.rule == k)
            contrib[int(k)] = float((wA[sel] * fl[sel]).sum())
        tot = sum(contrib.values())
        if tot > 0:
            top = sorted(contrib.items(), key=lambda kv: -kv[1])[:10]
            L.append(f"- **{f}**: " + ", ".join(f"rule {k} ({v / tot:.0%})" for k, v in top))

    # ---- vs Phase 1
    if phase1_dir:
        L.append(f"\n## What changed relative to Phase 1 (`{phase1_dir}`, same analysis code)\n")
        p1 = _phase1_summary(phase1_dir)
        L.append("| hypothesis | prior | analysis | Phase 1 data | this run |")
        L.append("|---|---|---|---|---|")
        for key in sorted(set(summary) | set(p1), key=str):
            if key in summary or key in p1:
                L.append(f"| {key[0]} | {key[1]} | {key[2]} | {p1.get(key, '—')} | {summary.get(key, '—')} |")
        L.append("\nPhase 1 column: the Phase 1 worlds under the same priors and analyses (the 2a preview, "
                 "extended to the A1 analysis by re-simulating world properties). It inherits the crossed "
                 "seeds and the complement approximation.")

    # ---- summary up top
    S = ["\n## Summary of verdicts\n", "| hypothesis | prior | " + " | ".join(lab for lab, _ in analyses) + " |",
         "|---|---|" + "---|" * len(analyses)]
    for h in ("H1", "H7a", "H7b", "H3", "H5"):
        for P in ("M_A", "M_B", "M_A*"):
            cells = [summary.get((h, P, lab), "—") for lab, _ in analyses]
            if any(c != "—" for c in cells):
                tag = " (headline)" if h in ("H1", "H7a", "H7b") and P == "M_A" else ""
                S.append(f"| {h}{tag} | {P}{' (robustness row, A3)' if P == 'M_A*' else ''} | " + " | ".join(cells) + " |")
    S.append(f"| H6 | M_A | {summary[('H6', 'M_A', '—')]} | | |")
    S.append(f"| H6d | M_A | {summary[('H6d', 'M_A', '—')]} | | |")
    S.append("")
    L[head_at:head_at] = S

    secs = [r.get("seconds", 0) for r in rows]
    L.append(f"\n---\nCompute: {sum(secs) / 3600:.2f} CPU-hours over {len(rows)} worlds.\n")
    report = "\n".join(L) + "\n"
    path = os.path.join(out_dir, "report_2a.md")
    with open(path, "w") as f:
        f.write(report)
    with open(os.path.join(out_dir, "summary_2a.json"), "w") as f:
        json.dump({"|".join(k): v for k, v in summary.items()}, f, indent=1)
    print(f"[analyze2a] wrote {path}")
    return summary


def _phase1_summary(phase1_dir):
    path = os.path.join(phase1_dir, "summary_2a.json")
    if not os.path.exists(path):
        analyze2a(phase1_dir, phase1_dir=None, do_fragility=False)
        os.replace(os.path.join(phase1_dir, "report_2a.md"), os.path.join(phase1_dir, "report_2a_phase1-worlds.md"))
    return {tuple(k.split("|")): v for k, v in json.load(open(path)).items()}


def _mark(f):
    return f"{f} †" if f in DEPRECATED_FILTERS else f


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 2a analysis (README 3b, 3b.7).")
    ap.add_argument("out_dir")
    ap.add_argument("--phase1", default=None, metavar="DIR", help="Phase 1 directory for the comparison table")
    ap.add_argument("--no-fragility", action="store_true")
    a = ap.parse_args(argv)
    analyze2a(a.out_dir, phase1_dir=a.phase1, do_fragility=not a.no_fragility)


if __name__ == "__main__":
    main()
