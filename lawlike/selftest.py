"""Sanity checks. Run before trusting any result:   python -m lawlike.selftest

Each check is a case where we know the right answer in advance. If one fails, the
instrument is broken and no experiment result means anything.
"""
from __future__ import annotations

import sys

import numpy as np

from . import info
from .causal import CausalStates
from .bootstrap import ClusterBootstrap
from .config import get_preset
from .experiment1 import law_metrics
from .worlds import build_ic, rule_representatives, simulate

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def main():
    rng = np.random.default_rng(0)
    print("worlds")
    reps = rule_representatives()
    check("88 ECA equivalence classes covering 256 rules", len(reps) == 88 and sum(reps.values()) == 256)
    st = simulate(90, build_ic("embed", "1", 31), 8)
    c = 15
    check("rule 90 grows the Sierpinski pattern", st[1, c - 1] == 1 and st[1, c] == 0 and st[1, c + 1] == 1
          and st[2, c - 2] == 1 and st[2, c] == 0 and st[2, c + 2] == 1)
    ic = rng.integers(0, 2, 40, dtype=np.uint8)
    check("rule 204 is the identity", np.all(simulate(204, ic, 5) == ic))
    ic = rng.integers(0, 2, 50, dtype=np.uint8)
    a, b = simulate(30, ic, 40), simulate(30, ic, 40, flip=(20, 10))
    check("perturbation leaves the past untouched and changes the future",
          np.all(a[:20] == b[:20]) and np.any(a[21:] != b[21:]))

    print("estimators")
    x = rng.integers(0, 4, 4000)
    y = rng.integers(0, 4, 4000)
    check("entropy of a fair 4-sided die ~ 2 bits", abs(info.entropy(x) - 2) < 0.02, f"{info.entropy(x):.3f}")
    check("debiased MI of independent variables ~ 0", abs(info.mi_debiased(x, y, rng)) < 0.02)
    check("MI of a copy ~ its entropy", abs(info.mi(x, x) - info.entropy(x)) < 0.02)
    z = rng.integers(0, 2, 4000)
    check("debiased CMI of independent variables ~ 0", abs(info.cmi_debiased(x, y, z, rng)) < 0.03)
    xor = (x + z) % 2
    check("CMI detects dependence only visible given Z", info.cmi(x % 2, xor, z) > 0.9)

    # checks added in Phase 2 prep use their own generator so the original checks see the same draws
    rng2 = np.random.default_rng(2)
    check("normalised aligned MI: a copy ~ 1, independent streams ~ 0",
          info.mi_norm_debiased(x, x, rng2) > 0.95 and abs(info.mi_norm_debiased(x, y, rng2)) < 0.02)

    print("cluster bootstrap")
    # crossed design: 30 rules all run on the same 2 sampled seeds; the seed effect is large,
    # so worlds sharing a seed are correlated and an i.i.d. bootstrap is overconfident
    worlds, vals = [], []
    seed_eff = {"10110": 0.0, "01101": 1.0}
    for r in range(30):
        for bits, e in seed_eff.items():
            worlds.append(dict(arm="simplicity", rule=r, seed_bits=bits))
            vals.append(e + 0.1 * rng2.standard_normal())
    v = np.array(vals)
    w = np.ones(v.size)
    idx = np.arange(v.size)
    cb = ClusterBootstrap(worlds, 3, np.random.default_rng(1)).means(idx, w, v)
    iid = np.array([v[rng2.integers(0, v.size, v.size)].mean() for _ in range(400)])
    wc, wi = np.quantile(cb, 0.975) - np.quantile(cb, 0.025), np.quantile(iid, 0.975) - np.quantile(iid, 0.025)
    check("crossed seeds widen the interval relative to i.i.d. worlds", wc > 3 * wi, f"cluster {wc:.2f} vs iid {wi:.2f}")
    enum = [dict(arm="simplicity", rule=r, seed_bits="01") for r in range(30)]
    cb = ClusterBootstrap(enum, 3, np.random.default_rng(1), resample_rules=False).means(
        np.arange(30), np.ones(30), rng2.standard_normal(30))
    check("fully enumerated design with fixed rules has no sampling variance", np.ptp(cb) < 1e-12)

    print("integration proxy")
    T = 3000
    a0 = rng.integers(0, 2, T)
    b0 = rng.integers(0, 2, T)
    indep = np.stack([a0, b0], 1)                     # each bit copies itself: no integration
    check("two self-copying bits are not integrated", info.phi_whole_minus_parts(indep[:-1], indep[1:], 2) < 0.02)
    swap = np.stack([a0, b0], 1)
    swapped_next = swap[:, ::-1]                       # each bit becomes the other: fully integrated
    check("two bits that swap each step are integrated",
          info.phi_whole_minus_parts(swap, swapped_next, 2) > 1.5)

    print("coding (computable stand-in for the universal prior)")
    per = np.tile(np.array([0, 1, 1, 0, 1]), 400)
    rnd = rng.integers(0, 2, 2000)
    mp = law_metrics(per, per, 8)
    mr = law_metrics(rnd, rnd, 8)
    check("periodic stream is lawful", mp["klass"] == 1 and mp["hnorm"] < 0.1, f"hnorm={mp['hnorm']:.3f}")
    check("random stream is noisy", mr["klass"] == 2 and mr["hnorm"] > 0.95, f"hnorm={mr['hnorm']:.3f}")
    check("constant stream is trivial", law_metrics(np.zeros(500, np.int64), np.zeros(500, np.int64), 8)["klass"] == 0)
    alt = per.copy()
    alt[1000:] = rng.integers(0, 2, 1000)
    m = law_metrics(per, alt, 8)
    check("perturbation test sees broken laws", m["rgap"] > 0.5 and m["div"] > 0.3)
    shifted = per.copy()
    shifted[1000:] = np.roll(per, 2)[1000:]
    m = law_metrics(per, shifted, 8)
    check("perturbation test sees changed facts with intact laws", m["div"] > 0.3 and m["rgap"] < 0.2,
          f"div={m['div']:.2f} rgap={m['rgap']:.3f}")
    check("NCD: a stream vs itself < vs unrelated noise",
          info.ncd(rnd, rnd) < info.ncd(rnd, rng.integers(0, 2, 2000)))

    print("local causal states")
    cfg = get_preset("smoke")
    st0 = simulate(0, np.zeros(64, np.uint8), 80)
    cs = CausalStates(cfg.hp, cfg.hf).fit(st0, 10, 70)
    check("rule 0 (nothing happens) has one causal state", cs.n_states == 1)
    st18 = simulate(18, rng.integers(0, 2, 128, dtype=np.uint8), 300)
    cs = CausalStates(cfg.hp, cfg.hf).fit(st18, 50, 290)
    cpx = cs.complexity(cs.label(st18))[50:290]
    check("rule 18 has several causal states", cs.n_states >= 3, f"{cs.n_states} states")
    check("rule 18 complexity is concentrated (defects stand out from the domain)",
          np.quantile(cpx, 0.95) > 2 * np.median(cpx) + 0.5,
          f"median={np.median(cpx):.2f} q95={np.quantile(cpx, 0.95):.2f}")
    stn = rng.integers(0, 2, (300, 128), dtype=np.uint8)
    cs = CausalStates(cfg.hp, cfg.hf).fit(stn, 10, 290)
    check("pure noise collapses to ~one causal state", cs.n_states <= 2, f"{cs.n_states} states")

    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) FAILED: {FAILS}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
