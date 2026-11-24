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
from .worlds import (build_ic, complement, make_world_list_2a, reflect, rule_representatives, simulate,
                     translation_period)

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

    print("dynamical richness (README 3b.7 A1)")
    rng3 = np.random.default_rng(3)   # own generator: earlier checks keep their draws
    ic = rng3.integers(0, 2, 128, dtype=np.uint8)
    check("identity and shift on a random IC are translation-periodic",
          translation_period(simulate(204, ic, 200), 16) == (1, 0)
          and translation_period(simulate(170, ic, 200), 16) == (1, -1)
          and translation_period(simulate(240, ic, 200), 16) == (1, 1))
    check("a homogeneous blinker is translation-periodic", translation_period(simulate(1, np.zeros(64, np.uint8), 50), 16) == (2, 0))
    check("rule 30 and rule 110 on a random IC are not", translation_period(simulate(30, ic, 400)[100:], 16) is None
          and translation_period(simulate(110, ic, 400)[100:], 16) is None)
    glider = simulate(170, build_ic("embed", "1101", 64), 50)
    check("a lone translating pattern is translation-periodic", translation_period(glider, 16) is not None)

    print("Phase 2a world list (README 3b.6)")
    x0 = rng3.integers(0, 2, 64, dtype=np.uint8)
    check("complement symmetry: comp(r) on ~x is ~(r on x)",
          all(np.array_equal(simulate(complement(r), 1 - x0, 40), 1 - simulate(r, x0, 40)) for r in (18, 30, 54, 110)))
    check("complement of an embed-in-zeros seed is not an embed-in-zeros seed (why README 3b.6 runs it)",
          not np.array_equal(1 - build_ic("embed", "101", 16), build_ic("embed", "010", 16)))
    from dataclasses import replace as _replace
    c2 = _replace(get_preset("smoke2a"), exclude_programs_from=(), rules="all", ic_max_len=6)
    w2 = make_world_list_2a(c2)
    simp = [w for w in w2 if w.arm == "simplicity"]
    reps = rule_representatives()
    tot = {}
    for w in simp:
        tot[w.rule] = tot.get(w.rule, 0.0) + w.weight
    check("2a simplicity weights sum to 1", abs(sum(tot.values()) - 1) < 1e-9)
    check("2a class weight is proportional to class size (seed prior identical per rule)",
          max(abs(tot[r] - reps[r] / 256) for r in reps) < 1e-9)
    check("2a runs complement-orbit embed worlds exactly for non-self-complementary classes",
          {w.rule for w in simp if w.sim_rule >= 0} == {r for r in reps if complement(r) not in {r, reflect(r)}})
    long_seeds = {}
    for w in simp:
        if len(w.seed_bits) > c2.ic_exact_len:
            long_seeds.setdefault(len(w.seed_bits), {}).setdefault(w.rule, set()).add(w.seed_bits)
    differs = any(len({frozenset(v) for v in d.values()}) > 1 for d in long_seeds.values())
    check("2a sampled seeds differ between rules (nested, not crossed)", differs)

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

    print("seam-free tiling (README 3b.8 A9)")
    from types import SimpleNamespace
    from .experiment1 import anticipation, anticipation_gaps
    from .observers import Extractor
    from .worlds import World, realise, ring_size
    base = get_preset("smoke")
    tw = World("t", "simplicity", 126, 1, "tile", "010", 5.0, 1.0, 1)
    sf = _replace(base, seam_free=True)
    st_seam, _, _ = realise(tw, base)
    st_sf, _, fl = realise(tw, sf)
    check("a 3-cell tile on a 64-cell ring runs on 63 cells when seam-free", ring_size(tw, sf) == 63
          and st_sf.shape[1] == 63 and st_seam.shape[1] == 64 and fl[1] < 63)
    check("the seam-free world keeps the tile's spatial period; the seamed one does not",
          np.array_equal(st_sf, np.roll(st_sf, 3, axis=1)) and not np.array_equal(st_seam, np.roll(st_seam, 3, axis=1)))
    full = get_preset("full")      # at smoke size the seamed world's defect dies out on its own
    reg = slice(full.T0, full.T0 + full.T)
    check("at N = 256 the seam-free world is translation-periodic; the seamed one is kept alive by its defect",
          translation_period(realise(tw, _replace(full, seam_free=True))[0][reg], 16) is not None
          and translation_period(realise(tw, full)[0][reg], 16) is None)
    ew = World("e", "simplicity", 126, 1, "embed", "011", 5.0, 1.0, 1)
    check("embed worlds are untouched", np.array_equal(realise(ew, base)[0], realise(ew, sf)[0]))

    print("memory horizons, depth, learning progress (README 3c.2, 3c.4)")
    rng4 = np.random.default_rng(4)
    c = info.mixture_codes(per, None, 8, 2)
    check("the horizon-8 code length is the existing code length", abs(c["by_horizon"][8] - c["total"]) < 1e-9)
    m = law_metrics(rnd, rnd, 8)
    check("h^(8) is the existing h", m["hm8"] == m["h"])
    check("noise: no depth, no learning progress", abs(m["depth"]) < 0.02 and abs(m["gap"]) < 0.02,
          f"depth={m['depth']:.3f} gap={m['gap']:.3f}")
    check("a learned periodic stream: no depth beyond its period, no learning progress",
          abs(mp["gap"]) < 0.02, f"gap={mp['gap']:.3f}")
    s3 = rng4.integers(0, 2, 4000)
    flipn = rng4.random(4000) < 0.05
    for t in range(3, 4000):
        s3[t] = s3[t - 3] ^ int(flipn[t])
    m3 = law_metrics(s3, s3, 8)
    check("a stream with only third-order dependence: h^(2) high, h^(4) low, positive depth",
          m3["hm2"] > 0.9 and m3["hm4"] < 0.4 and m3["depth"] > 0.5,
          f"h2={m3['hm2']:.2f} h4={m3['hm4']:.2f} depth={m3['depth']:.2f}")
    check("...and it is lawful only for an observer who remembers at least three steps",
          m3["hm1"] / m3["H1"] > 0.5 and m3["hm2"] / m3["H1"] > 0.5 and m3["hm4"] / m3["H1"] <= 0.5)

    print("anticipation (README 3c.3)")
    big = SimpleNamespace(N=8192, T=4000, T0=10, velocities=((0, 1),))
    ic = rng4.integers(0, 2, big.N, dtype=np.uint8)
    st170 = simulate(170, ic, big.T0 + big.T + 4)
    ex = Extractor(big, st170, None, None, 1)
    for w in (2, 5):
        rd = ex.read(1, 1000, w, 0, 1)
        n = rd["sym"].size
        ak = {k: info.cmi_debiased(rd["env"][k:], rd["sym"][:n - k], rd["env"][:n - k], rng4)
              / (info.entropy(info.combine(rd["env"][k:], rd["env"][:n - k])) - info.entropy(rd["env"][:n - k]))
              for k in (1, 2, 4, 8)}
        check(f"shift rule, window of {w}: a_k ~ 1/2 for k <= {w} and ~ 0 beyond",
              all(abs(ak[k] - (0.5 if k <= w else 0.0)) < 0.06 for k in ak),
              " ".join(f"a{k}={v:.2f}" for k, v in ak.items()))
    an = anticipation(rd["sym"], rd["env"], rng4)
    # diary: the downstream boundary at t+k is the upstream boundary at t-(w+1-k), so a two-step
    # record of inputs holds it only for k = 4 (half of the two boundary cells): mean 1/6
    check("shift rule, window of 5: anticipation = mean of a_2, a_4, a_8 ~ 1/3; a two-step diary ~ 1/6",
          abs(an["antic"] - 1 / 3) < 0.05 and abs(an["a1"] - 0.5) < 0.06 and abs(an["diary"] - 1 / 6) < 0.03,
          f"A={an['antic']:.3f} diary={an['diary']:.3f}")
    stn = rng4.integers(0, 2, (big.T0 + big.T + 4, 64), dtype=np.uint8)
    exn = Extractor(SimpleNamespace(N=64, T=1000, T0=10, velocities=((0, 1),)), stn, None, None, 1)
    A = np.array([anticipation(r_["sym"], r_["env"], rng4)["antic"] for r_ in (exn.read(1, x, 5, 0, 1) for x in range(0, 64, 8))])
    check("noise world, 5-cell windows over 1000 steps: anticipation ~ 0", abs(A.mean()) < 0.02 and np.abs(A).max() < 0.08,
          f"mean={A.mean():.3f} max|A|={np.abs(A).max():.3f}")
    fz = anticipation(np.arange(500) % 7, np.zeros(500, np.int64), rng4)
    check("a frozen environment offers nothing to anticipate (A = 0)", fz["antic"] == 0.0 and fz["ret4"] == 0.0)
    obs_ = dict(fam=np.array([1, 1, 1, 1, 2]), w=np.array([2, 2, 2, 3, 2]), s=np.ones(5, int), v=np.zeros(5, int))
    g = anticipation_gaps(obs_, np.array([0.3, 0.1, 0.2, 0.5, np.nan]))
    check("the control is the same program at the other positions",
          np.allclose(g[:3], [0.15, -0.15, 0.0]) and np.isnan(g[3]) and np.isnan(g[4]))

    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) FAILED: {FAILS}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
