"""Aggregate Experiment 1 results and test the pre-registered hypotheses.

    python -m lawlike.analyze results/quick

Two anthropic weightings are reported for every number, because they are the two standard,
defensible choices and they can disagree:

  SSA-style : each world counts by its prior weight P(world), and within a world the
              admissible observers share that weight (worlds with few observers count as
              much as worlds with many).
  SIA-style : each (world, observer) pair counts by P(world) * P(observer), restricted to
              admissible observers, so worlds that host more observer-mass count more.

Uncertainty: a cluster bootstrap matched to the sampling design (bootstrap.py): rules and
sampled seeds are resampled as crossed factors, uniform ICs within rule, noise worlds i.i.d.
With a simplicity prior a handful of very simple worlds carry most of the weight, so
intervals can be wide; that is information, not a bug.

    python -m lawlike.analyze results/full --exclude-programs-from results/quick

drops every simplicity-arm world program (rule, constructor, seed) that also appears in the
other directory and writes report_disjoint-<name>.md. A confirmation run must be disjoint
from the run that motivated the change (README section 5.1); this is how to check it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

from .bootstrap import ClusterBootstrap
from .experiment1 import DEPRECATED_FILTERS, FILTERS

ARMS = ("simplicity", "uniform", "noise")
METRICS = ("lawful_nt", "frac_trivial", "famshare_null", "mean_Lambda", "mean_hnorm_nt",
           "facts_change_nt", "facts_change_laws_hold_nt", "ncd_adm", "ncd_base", "mass",
           "famshare_window", "famshare_coarse", "famshare_causal", "frac_lawful",
           "xmi_adm", "xmi_base")
RICH_BLOCK_ENTROPY = 0.5   # bits/cell; below this a world is (nearly) homogeneous
CHANCE_JACCARD = 0.2 * 0.2 / (0.2 + 0.2 - 0.2 * 0.2)   # two independent top-20% sets


def load(out_dir: str) -> list[dict]:
    rows = []
    with open(os.path.join(out_dir, "results.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def program_key(world: dict):
    return (world["rule"], world["ic_kind"], world["seed_bits"])


def _arrays(rows, arm, filt, metric, mode):
    """(row indices, weights, values) for one arm/filter/metric; indices refer to the bootstrap."""
    i, w, v = [], [], []
    for r in rows:
        if r["world"]["arm"] != arm:
            continue
        fs = r["filters"][filt]
        val = fs.get(metric, float("nan"))
        if val is None or not np.isfinite(val):
            continue
        wt = r["world"]["weight"]
        if mode == "SIA":
            wt *= fs.get("mass", 0.0)
        if wt <= 0:
            continue
        i.append(r["_i"])
        w.append(wt)
        v.append(val)
    return np.array(i, dtype=int), np.array(w), np.array(v)


def _world_arrays(rows, arm, fn):
    """Like _arrays, for a per-world quantity fn(row) (NaN/None skipped)."""
    i, w, v = [], [], []
    for r in rows:
        if r["world"]["arm"] != arm:
            continue
        val = fn(r)
        if val is None or not np.isfinite(val) or r["world"]["weight"] <= 0:
            continue
        i.append(r["_i"])
        w.append(r["world"]["weight"])
        v.append(val)
    return np.array(i, dtype=int), np.array(w), np.array(v)


def _est(a):
    _, w, v = a
    return float(np.sum(w * v) / np.sum(w)) if w.size else float("nan")


def contrast_ci(terms, boot):
    """Estimate and 95% CI of sum(sign * weighted-mean(a)) over terms [(sign, a), ...].

    Every term is evaluated on the same bootstrap replicates, so shared rules and seeds are
    paired across arms.
    """
    if any(a[1].size < 3 for _, a in terms):
        if all(a[1].size for _, a in terms):
            return sum(s * _est(a) for s, a in terms), float("nan"), float("nan")
        return (float("nan"),) * 3
    est = sum(s * _est(a) for s, a in terms)
    d = sum(s * boot.means(*a) for s, a in terms)
    d = d[np.isfinite(d)]
    if d.size < 10:
        return float(est), float("nan"), float("nan")
    return float(est), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def wmean_ci(a, boot):
    return contrast_ci([(1, a)], boot)


def diff_ci(a, b, boot):
    return contrast_ci([(1, a), (-1, b)], boot)


def verdict(lo, hi, sign=+1):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "INSUFFICIENT DATA"
    if sign * lo > 0 and sign * hi > 0:
        return "SUPPORTED"
    if sign * hi < 0 and sign * lo < 0:
        return "CONTRADICTED"
    return "INCONCLUSIVE"


def fmt(x, nd=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _mark(filt):
    return f"{filt} †" if filt in DEPRECATED_FILTERS else filt


def analyze(out_dir: str, exclude_from: str | None = None, resample_rules: bool = True) -> str:
    rows = load(out_dir)
    if not rows:
        print("[analyze] no results yet")
        return ""
    cfg = {}
    cfg_path = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path)).get("config", {})
    ic_exact_len = int(cfg.get("ic_exact_len", 0))

    suffix, excl_note = "", ""
    if exclude_from:
        other = {program_key(r["world"]) for r in load(exclude_from) if r["world"]["arm"] == "simplicity"}
        simp_w = sum(r["world"]["weight"] for r in rows if r["world"]["arm"] == "simplicity")
        drop = [r for r in rows if r["world"]["arm"] == "simplicity" and program_key(r["world"]) in other]
        drop_w = sum(r["world"]["weight"] for r in drop)
        ids = {id(r) for r in drop}
        rows = [r for r in rows if id(r) not in ids]
        name = os.path.basename(os.path.abspath(exclude_from))
        suffix = f"_disjoint-{name}"
        excl_note = (f"**Disjoint-program analysis.** {len(drop)} simplicity-arm world programs that also "
                     f"appear in `{exclude_from}` were dropped ({drop_w / simp_w:.0%} of simplicity-arm "
                     "weight). Uniform and noise worlds are random draws and are kept.\n")
    for i, r in enumerate(rows):
        r["_i"] = i
    boot = ClusterBootstrap([r["world"] for r in rows], ic_exact_len, np.random.default_rng(0),
                            resample_rules=resample_rules, nested=cfg.get("seed_mode") == "per_rule")
    counts = {a: sum(r["world"]["arm"] == a for r in rows) for a in ARMS}

    # ---- main table
    table = []
    for mode in ("SSA", "SIA"):
        for filt in FILTERS:
            for arm in ARMS:
                for m in METRICS:
                    a = _arrays(rows, arm, filt, m, "SSA" if m == "mass" else mode)
                    est, lo, hi = wmean_ci(a, boot)
                    table.append(dict(weighting=mode, filter=filt, arm=arm, metric=m,
                                      mean=est, lo=lo, hi=hi, n_worlds=int(a[1].size)))
    with open(os.path.join(out_dir, f"summary{suffix}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        wr.writeheader()
        wr.writerows(table)
    index = {(t["weighting"], t["filter"], t["arm"], t["metric"]): t for t in table}

    def get(mode, filt, arm, m):
        return index.get((mode, filt, arm, m))

    # ---- jaccard (simplicity arm, prior-weighted over worlds)
    jac = {}
    for r in rows:
        if r["world"]["arm"] != "simplicity":
            continue
        for k, val in r["jaccard"].items():
            if val is not None and np.isfinite(val):
                jac.setdefault(k, [0.0, 0.0])
                jac[k][0] += r["world"]["weight"] * val
                jac[k][1] += r["world"]["weight"]
    jac = {k: s / z for k, (s, z) in jac.items() if z > 0}
    names = [f for f in FILTERS if f != "none"]
    active = [f for f in names if f not in DEPRECATED_FILTERS]

    L = []
    L.append(f"# Experiment 1 report — {os.path.basename(os.path.abspath(out_dir))}{suffix}\n")
    if excl_note:
        L.append(excl_note)
    L.append(f"Worlds analysed: simplicity {counts['simplicity']}, uniform {counts['uniform']}, "
             f"noise {counts['noise']}. Intervals are 95% cluster bootstrap "
             f"({'rules × sampled seeds, crossed' if resample_rules else 'sampled seeds; rule set held fixed'}; "
             "uniform ICs within rule; noise i.i.d.). Simplicity − uniform contrasts are paired on rules.\n")
    if boot.single_seed_strata:
        L.append(f"*Seed lengths {boot.single_seed_strata} have a single sampled seed, so their "
                 "seed-sampling variance cannot be estimated and is missing from the intervals.*\n")
    L.append("Reading guide: *lawful_nt* = share of non-trivial admitted observer-mass whose "
             "experience a simplicity-prior predictor finds lawful. *trivial* = share of admitted "
             "mass whose stream has essentially no variety. *null* = share of admitted mass that "
             "ignores the world entirely.\n")
    L.append(f"† = deprecated filter ({', '.join(DEPRECATED_FILTERS)}): failed a filter-health "
             "criterion on Phase 1 `full` (NOTES.md). Still computed and reported; not counted "
             "toward hypothesis verdicts.\n")

    for mode in ("SSA", "SIA"):
        L.append(f"\n## Lawfulness by arm and filter ({mode} weighting)\n")
        L.append("| filter | simplicity | uniform | noise | trivial (simp) | null share (simp) |")
        L.append("|---|---|---|---|---|---|")
        for filt in FILTERS:
            cells = []
            for arm in ARMS:
                t = get(mode, filt, arm, "lawful_nt")
                cells.append(f"{fmt(t['mean'])} [{fmt(t['lo'], 2)}, {fmt(t['hi'], 2)}]")
            tr = get(mode, filt, "simplicity", "frac_trivial")
            nl = get(mode, filt, "simplicity", "famshare_null")
            L.append(f"| {_mark(filt)} | " + " | ".join(cells) + f" | {fmt(tr['mean'])} | {fmt(nl['mean'])} |")

    L.append("\n## Admitted observer mass (SSA)\n")
    L.append("Share of the prior observer-mass each filter admits. — in the tables above means the "
             "filter admitted no observers in any world of that arm (e.g. no structure to find).\n")
    L.append("| filter | simplicity | uniform | noise |")
    L.append("|---|---|---|---|")
    for filt in FILTERS:
        L.append(f"| {_mark(filt)} | " + " | ".join(fmt(get("SSA", filt, a, "mass")["mean"]) for a in ARMS) + " |")

    # ---- hypotheses
    L.append("\n## Pre-registered hypotheses (SSA weighting unless noted)\n")
    L.append("**H1. A simplicity prior makes lawful experience typical; a uniform (Boltzmann) "
             "prior and pure noise do not.** Prediction: lawful_nt(simplicity) > lawful_nt(uniform) "
             "> lawful_nt(noise), for most filters.\n")

    def h1_table(sub, label):
        L.append("| filter | simp | unif | simp − unif | verdict | simp − noise | verdict |")
        L.append("|---|---|---|---|---|---|---|")
        n_sup = 0
        for filt in FILTERS:
            a = _arrays(sub, "simplicity", filt, "lawful_nt", "SSA")
            b = _arrays(sub, "uniform", filt, "lawful_nt", "SSA")
            c = _arrays(sub, "noise", filt, "lawful_nt", "SSA")
            d1, d2 = diff_ci(a, b, boot), diff_ci(a, c, boot)
            v1, v2 = verdict(d1[1], d1[2]), verdict(d2[1], d2[2])
            if filt in active:
                n_sup += (v1 == "SUPPORTED")
            L.append(f"| {_mark(filt)} | {fmt(_est(a))} | {fmt(_est(b))} | "
                     f"{fmt(d1[0])} [{fmt(d1[1], 2)}, {fmt(d1[2], 2)}] | {v1} | "
                     f"{fmt(d2[0])} [{fmt(d2[1], 2)}, {fmt(d2[2], 2)}] | {v2} |")
        L.append(f"\n{label}: simplicity > uniform SUPPORTED for {n_sup} of {len(active)} "
                 "non-deprecated world-reading filters.\n")

    h1_table(rows, "All worlds")

    rich = [r for r in rows if r["world_stats"]["block_entropy"] >= RICH_BLOCK_ENTROPY
            or r["world"]["arm"] == "noise"]
    rich_w = {}
    for a in ARMS:
        tot = sum(r["world"]["weight"] for r in rows if r["world"]["arm"] == a)
        rw = sum(r["world"]["weight"] for r in rows if r["world"]["arm"] == a
                 and r["world_stats"]["block_entropy"] >= RICH_BLOCK_ENTROPY)
        rich_w[a] = rw / tot if tot > 0 else float("nan")
    L.append(f"\n*H1 restricted to rich worlds* (spatial 3-block entropy ≥ {RICH_BLOCK_ENTROPY} bits/cell; "
             f"these carry {rich_w['simplicity']:.0%} of simplicity-arm weight and "
             f"{rich_w['uniform']:.0%} of uniform-arm weight). If the verdicts flip here, the "
             "full-arm result is driven by homogeneous or blinking worlds.\n")
    h1_table(rich, "Rich worlds")

    L.append("\n**H2. The trivial-observer catch.** Under the bare 2^-L measure (filter *none*), "
             "observers that ignore the world, or see nothing change, take a large share of the "
             "mass. Every world-reading filter should drive the null share to zero; a filter that "
             "still admits mostly trivial streams is not doing its job.\n")
    L.append("| filter | null share | trivial share |")
    L.append("|---|---|---|")
    for filt in FILTERS:
        nl = get("SSA", filt, "simplicity", "famshare_null")
        tr = get("SSA", filt, "simplicity", "frac_trivial")
        L.append(f"| {_mark(filt)} | {fmt(nl['mean'])} | {fmt(tr['mean'])} |")

    L.append("\n**H3. Convergence of criteria.** If 'observer' picks out something real, "
             "independently motivated filters should admit overlapping observers. Prior-weighted "
             "Jaccard overlap of admitted sets (simplicity arm; 1 = identical, 0 = disjoint):\n")
    L.append("| | " + " | ".join(_mark(n) for n in names) + " |")
    L.append("|---|" + "---|" * len(names))
    for a in names:
        cells = []
        for b in names:
            if a == b:
                cells.append("1")
            else:
                k = f"{a}|{b}" if f"{a}|{b}" in jac else f"{b}|{a}"
                cells.append(fmt(jac.get(k, float("nan")), 2))
        L.append(f"| {_mark(a)} | " + " | ".join(cells) + " |")

    def mean_overlap_ci(keep):
        """Mean over filter pairs of each pair's prior-weighted overlap (the statistic above)."""
        pairs = [k for k in jac if all(x in keep for x in k.split("|"))]
        arrs = [_world_arrays(rows, "simplicity", lambda r, k=k: r["jaccard"].get(k)) for k in pairs]
        arrs = [a for a in arrs if a[1].size >= 3]
        if not arrs:
            return (float("nan"),) * 3
        est = float(np.mean([_est(a) for a in arrs]))
        d = np.mean([boot.means(*a) for a in arrs], axis=0)
        d = d[np.isfinite(d)]
        return est, float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))

    L.append(f"\nChance level: two independent top-20% subsets overlap {CHANCE_JACCARD:.3f}. "
             "Verdict SUPPORTED if the mean overlap's interval lies above chance, CONTRADICTED if below.\n")
    L.append("| filters | mean pairwise overlap | verdict |")
    L.append("|---|---|---|")
    for label, keep in (("all (as pre-registered)", names), ("non-deprecated only", active)):
        e, lo, hi = mean_overlap_ci(keep)
        L.append(f"| {label} | {fmt(e)} [{fmt(lo, 2)}, {fmt(hi, 2)}] | "
                 f"{verdict(lo - CHANCE_JACCARD, hi - CHANCE_JACCARD)} |")

    L.append("\n**H4. Facts change, laws hold.** A one-cell perturbation mid-history should often "
             "change what admitted observers later experience (facts) without breaking the "
             "regularities learned before it (laws). Shown: share of non-trivial admitted mass "
             "whose facts changed, and share whose facts changed while laws held.\n")
    L.append("| filter | facts changed (simp) | …and laws held (simp) | facts changed (noise) | …and laws held (noise) |")
    L.append("|---|---|---|---|---|")
    for filt in FILTERS:
        fc = get("SSA", filt, "simplicity", "facts_change_nt")
        fl = get("SSA", filt, "simplicity", "facts_change_laws_hold_nt")
        fcn = get("SSA", filt, "noise", "facts_change_nt")
        fln = get("SSA", filt, "noise", "facts_change_laws_hold_nt")
        L.append(f"| {_mark(filt)} | {fmt(fc['mean'])} | {fmt(fl['mean'])} | {fmt(fcn['mean'])} | {fmt(fln['mean'])} |")

    L.append("\n**H5. Intersubjectivity.** Admitted observers in the same world should share "
             "more information with each other than random windows do — *and more so than in a "
             "noise world*, where there is no common world to agree about.\n")

    def gap(base, adm, filt):
        def fn(r):
            fs = r["filters"][filt]
            x, y = fs.get(base), fs.get(adm)
            if x is None or y is None:
                return None
            return (x - y) if base.startswith("ncd") else (y - x)
        return fn

    have_xmi = any("xmi_adm" in r["filters"]["none"] for r in rows)
    variants = [("ncd_base", "ncd_adm", "NCD (Phase 1 statistic): gap = NCD(random windows) − NCD(admitted pairs)")]
    if have_xmi:
        variants.insert(0, ("xmi_base", "xmi_adm", "Aligned mutual information (pre-registered for Phase 2): "
                            "gap = MI(admitted pairs) − MI(random windows), normalised, footprints disjoint"))
    for base, adm, desc in variants:
        L.append(f"*{desc}.* Verdict on (simplicity gap − noise gap).\n")
        L.append("| filter | simplicity gap | noise gap | simp − noise | verdict |")
        L.append("|---|---|---|---|---|")
        for filt in FILTERS:
            a = _world_arrays(rows, "simplicity", gap(base, adm, filt))
            c = _world_arrays(rows, "noise", gap(base, adm, filt))
            if c[1].size == 0 and a[1].size:
                # pre-registered (README 3b.4): no admitted pairs in noise -> noise gap counts as 0
                d, ctxt = wmean_ci(a, boot), "0 (none admitted)"
            else:
                d, ctxt = diff_ci(a, c, boot), fmt(_est(c))
            L.append(f"| {_mark(filt)} | {fmt(_est(a))} | {ctxt} | "
                     f"{fmt(d[0])} [{fmt(d[1], 2)}, {fmt(d[2], 2)}] | {verdict(d[1], d[2])} |")
        L.append("")
    if not have_xmi:
        L.append("These results predate the aligned-MI statistic, so only the NCD version is available. "
                 "Testing against the noise arm was added after seeing Phase 1 `full` (NOTES.md): "
                 "treat this table as a post-hoc re-analysis.\n")

    # ---- filter health (pruning guidance)
    L.append("\n## Filter health (inputs to pruning decisions — see README before pruning)\n")
    L.append("A filter earns its keep if it (a) excludes world-ignoring observers, (b) does not "
             "mostly admit frozen streams, (c) tells a structured world from noise. Agreement "
             "with other filters is reported but is **not** a pruning criterion by itself.\n")
    L.append("| filter | (a) null share | (b) trivial share | (c) lawful simp − noise | mean overlap with others |")
    L.append("|---|---|---|---|---|")
    for filt in names:
        nl = get("SSA", filt, "simplicity", "famshare_null")["mean"]
        tr = get("SSA", filt, "simplicity", "frac_trivial")["mean"]
        a = _arrays(rows, "simplicity", filt, "lawful_nt", "SSA")
        c = _arrays(rows, "noise", filt, "lawful_nt", "SSA")
        d = diff_ci(a, c, boot)
        ov = [v for k, v in jac.items() if filt in k.split("|")]
        L.append(f"| {_mark(filt)} | {fmt(nl)} | {fmt(tr)} | {fmt(d[0])} [{fmt(d[1], 2)}, {fmt(d[2], 2)}] | "
                 f"{fmt(np.mean(ov) if ov else float('nan'))} |")

    # ---- which rules host lawful observers
    L.append("\n## Where the lawful observers live (simplicity arm)\n")
    L.append("Share of all lawful, non-trivial, admitted observer-mass contributed by each rule "
             "(SSA). Top 10 per filter.\n")
    for filt in ("none", "individuality", "integration_excl", "causal_complexity"):
        contrib: dict[int, float] = {}
        for r in rows:
            if r["world"]["arm"] != "simplicity":
                continue
            fs = r["filters"][filt]
            val = fs.get("frac_lawful", 0.0) or 0.0
            contrib[r["world"]["rule"]] = contrib.get(r["world"]["rule"], 0.0) + r["world"]["weight"] * val
        tot = sum(contrib.values())
        if tot <= 0:
            continue
        top = sorted(contrib.items(), key=lambda kv: -kv[1])[:10]
        L.append(f"- **{filt}**: " + ", ".join(f"rule {k} ({v / tot:.0%})" for k, v in top))

    secs = [r.get("seconds", 0) for r in rows]
    L.append(f"\n---\nCompute: {sum(secs) / 3600:.2f} CPU-hours over {len(rows)} worlds "
             f"({np.mean(secs):.1f} s/world).\n")
    report = "\n".join(L)
    with open(os.path.join(out_dir, f"report{suffix}.md"), "w") as f:
        f.write(report)
    try:
        _plots(os.path.join(out_dir, f"plots{suffix}"), get, jac, names)
    except Exception as e:  # plotting is optional
        print(f"[analyze] plots skipped: {e}")
    print(f"[analyze] wrote {os.path.join(out_dir, f'report{suffix}.md')} and summary{suffix}.csv")
    return report


# ----------------------------------------------------------------------------- plots

ARM_COLORS = {"simplicity": "#2a78d6", "uniform": "#eb6834", "noise": "#1baf7a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _plots(pdir, get, jac, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(pdir, exist_ok=True)

    # 1. lawfulness by filter and arm
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURF)
    _style(ax)
    x = np.arange(len(FILTERS))
    bw = 0.26
    for j, arm in enumerate(ARMS):
        m = [get("SSA", f, arm, "lawful_nt")["mean"] for f in FILTERS]
        lo = [get("SSA", f, arm, "lawful_nt")["lo"] for f in FILTERS]
        hi = [get("SSA", f, arm, "lawful_nt")["hi"] for f in FILTERS]
        m = np.nan_to_num(np.array(m, float))
        err = np.array([np.nan_to_num(m - np.array(lo, float)), np.nan_to_num(np.array(hi, float) - m)])
        err = np.clip(err, 0, None)
        ax.bar(x + (j - 1) * bw, m, bw - 0.03, color=ARM_COLORS[arm], label=arm, edgecolor=SURF, linewidth=2)
        ax.errorbar(x + (j - 1) * bw, m, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax.set_xticks(x, [f.replace("_", "\n") for f in FILTERS], color=INK2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("lawful share of non-trivial admitted mass", color=INK2, fontsize=9)
    ax.set_title("How often is admitted experience lawful? (SSA, 95% CI)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "lawfulness_by_filter.png"), dpi=150)
    plt.close(fig)

    # 2. filter agreement heatmap (single-hue sequential)
    M = np.eye(len(names))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                k = f"{a}|{b}" if f"{a}|{b}" in jac else f"{b}|{a}"
                M[i, j] = jac.get(k, np.nan)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f1f6fd", "#2a78d6", "#0d3a72"])
    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor=SURF)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names], fontsize=8, color=INK2)
    ax.set_yticks(range(len(names)), names, fontsize=8, color=INK2)
    for i in range(len(names)):
        for j in range(len(names)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.55 else INK)
    ax.set_title("Do the observer criteria agree? (weighted Jaccard)", color=INK, fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "filter_agreement.png"), dpi=150)
    plt.close(fig)

    # 3. who gets admitted: family shares per filter (simplicity arm)
    fams = ("null", "window", "coarse", "causal")
    fcol = ["#6250d6", "#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=SURF)
    _style(ax)
    bottom = np.zeros(len(FILTERS))
    for fam, c in zip(fams, fcol):
        vals = np.nan_to_num(np.array([get("SSA", f, "simplicity", f"famshare_{fam}")["mean"]
                                       if get("SSA", f, "simplicity", f"famshare_{fam}") else np.nan
                                       for f in FILTERS], float))
        ax.bar(x, vals, 0.6, bottom=bottom, color=c, label=fam, edgecolor=SURF, linewidth=2)
        bottom += vals
    ax.set_xticks(x, [f.replace("_", "\n") for f in FILTERS], color=INK2)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("share of admitted observer mass", color=INK2, fontsize=9)
    ax.set_title("Which kinds of observer does each criterion admit? (simplicity arm)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=4, labelcolor=INK2, loc="upper right", bbox_to_anchor=(1, 1.12))
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "family_shares.png"), dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Aggregate results and test the pre-registered hypotheses.")
    ap.add_argument("out_dir", nargs="?", default="results/smoke")
    ap.add_argument("--exclude-programs-from", default=None, metavar="DIR",
                    help="drop simplicity-arm world programs that also appear in DIR")
    ap.add_argument("--fixed-rules", action="store_true",
                    help="treat the enumerated rule set as the whole population (do not resample rules)")
    args = ap.parse_args(argv)
    analyze(args.out_dir, exclude_from=args.exclude_programs_from, resample_rules=not args.fixed_rules)


if __name__ == "__main__":
    main()
