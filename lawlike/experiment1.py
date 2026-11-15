"""Experiment 1: world-first, elementary cellular automata.

For one world: simulate it (and a single-bit-perturbed copy), infer its local causal
states, read every enumerated observer's experience stream, score each observer under each
admission criterion ("filter"), measure how law-like each stream is, and summarise.

Everything that matters for the hypotheses is summarised per (world, filter); the full
per-observer table is optionally saved so later analysis does not require re-running.
"""
from __future__ import annotations

import math
import os

import numpy as np

from . import info
from .causal import CausalStates
from .observers import FAMILIES, Extractor, enumerate_observers, null_stream
from .worlds import World, realise, translation_period

FILTERS = ("none", "persistence", "individuality", "blanket", "integration",
           "integration_excl", "causal_complexity")
SCORE_FILTERS = ("persistence", "individuality", "blanket", "integration", "causal_complexity")
# Failed a filter-health criterion on Phase 1 `full` (NOTES.md, 2026-10-23): persistence admits
# world-ignoring observers (a), blanket admits mostly frozen streams (b). Still computed and
# reported (README section 5.4), but not counted toward hypothesis verdicts.
DEPRECATED_FILTERS = ("persistence", "blanket")
METRICS_VERSION = 3   # 2: adds aligned-MI intersubjectivity (xmi_adm, xmi_base)
                      # 3: world_stats adds translation period / dynamical richness (README 3b.7 A1)
TPERIOD_SCAN = 512    # translation periods are recorded up to this; dyn_rich uses cfg.dyn_pmax

# classification conventions (fixed before seeing results; see README)
TRIVIAL_H1 = 0.1        # bits: a stream with less marginal entropy than this has no variety
LAWFUL_HNORM = 0.5      # stream is "lawful" if the predictor's surprise <= half its variety
DIVERGE = 0.05          # perturbation changed > 5% of later experiences ("facts changed")
ROBUST = 0.1            # bits/symbol: old laws still predict the perturbed future ("laws held")

_OBS_CACHE: dict = {}
_NULL_CACHE: dict = {}


def _observers(cfg):
    key = cfg.hash()
    if key not in _OBS_CACHE:
        _OBS_CACHE[key] = enumerate_observers(cfg)
    return _OBS_CACHE[key]


# ----------------------------------------------------------------------------- per observer

def law_metrics(sym: np.ndarray, alt: np.ndarray, K: int) -> dict:
    T = sym.size
    T2 = T // 2
    A = int(np.unique(np.concatenate([sym, alt])).size)
    c = info.mixture_codes(sym, alt, K, A)
    H1 = info.entropy(sym)
    h = c["total"] / T
    hnorm = h / H1 if H1 > 1e-9 else 0.0
    gap = (c["second_frozen"] - c["second_online"]) / (T - T2)
    rgap = (c["alt_frozen"] - c["second_frozen"]) / (T - T2)
    div = float(np.mean(sym[T2:] != alt[T2:]))
    if H1 < TRIVIAL_H1:
        klass = 0
    elif hnorm <= LAWFUL_HNORM:
        klass = 1
    else:
        klass = 2
    return dict(H1=H1, h=h, Lambda=max(0.0, H1 - h), hnorm=hnorm, gap=gap, rgap=rgap,
                div=div, klass=klass)


def observer_scores(rd: dict, rng: np.random.Generator) -> dict:
    s = rd["sym"]
    S0, S1, S2 = s[:-2], s[1:-1], s[2:]
    E0 = rd["env"][:-2]
    X0 = rd["ext"][:-2]
    H = info.entropy(s)
    out = {}
    if H < 1e-6:
        out["persistence"] = 0.0
        out["individuality"] = 0.0
    else:
        out["persistence"] = info.mi_debiased(S0, S1, rng) / H
        H1 = info.entropy(S1)
        out["individuality"] = info.cmi_debiased(S1, S0, E0, rng) / max(H1, 1e-9)
    z = info.combine(S0, E0)
    hc = info.entropy(info.combine(S2, z)) - info.entropy(z)
    if hc < 1e-3:
        out["blanket"] = 1.0
    else:
        leak = info.cmi_debiased(S2, X0, z, rng)
        out["blanket"] = float(np.clip(1.0 - leak / hc, 0.0, 1.0))
    p = rd["parts"]
    out["integration"] = info.phi_whole_minus_parts(p[:-1], p[1:], rd["radix"])
    out["causal_complexity"] = rd["cplx"]
    return out


# ----------------------------------------------------------------------------- per world

def process_world(world: World, cfg, out_dir: str | None = None) -> dict:
    rng = np.random.default_rng(world.rng_seed + 7)
    obs = _observers(cfg)
    n_obs = obs["fam"].size
    st, pert, flip = realise(world, cfg)

    cs = CausalStates(cfg.hp, cfg.hf, cfg.causal_min_count, cfg.causal_eps, cfg.causal_max_states)
    cs.fit(st, cfg.T0, cfg.T0 + cfg.T)
    lab, lab_p = cs.label(st), cs.label(pert)
    cplx = cs.complexity(lab)
    n_lab = cs.n_states + 1
    ex = Extractor(cfg, st, lab, cplx, n_lab)
    exp_ = Extractor(cfg, pert, lab_p, None, n_lab)

    cols = {k: np.full(n_obs, np.nan) for k in
            ("H1", "h", "Lambda", "hnorm", "gap", "rgap", "div", "klass", *SCORE_FILTERS)}
    streams: list[np.ndarray] = [None] * n_obs  # type: ignore

    for i in range(n_obs):
        fam = int(obs["fam"][i])
        if fam == 0:
            key = (int(obs["period"][i]), int(obs["pattern"][i]), cfg.T, cfg.K)
            if key not in _NULL_CACHE:
                s = null_stream(key[0], key[1], cfg.T)
                m = law_metrics(s, s, cfg.K)
                H = info.entropy(s)
                pers = 0.0 if H < 1e-6 else info.mi_debiased(s[:-1], s[1:], np.random.default_rng(0)) / H
                _NULL_CACHE[key] = (s, m, pers)
            s, m, pers = _NULL_CACHE[key]
            streams[i] = s
            for k, v in m.items():
                cols[k][i] = v
            cols["persistence"][i] = pers
            continue
        args = (fam, int(obs["x0"][i]), int(obs["w"][i]), int(obs["v"][i]), int(obs["s"][i]))
        rd = ex.read(*args)
        rdp = exp_.read(*args)
        streams[i] = rd["sym"]
        m = law_metrics(rd["sym"], rdp["sym"], cfg.K)
        for k, v in m.items():
            cols[k][i] = v
        for k, v in observer_scores(rd, rng).items():
            cols[k][i] = v

    masks = admissible_masks(obs, cols, cfg.top_q)
    xrng = np.random.default_rng(world.rng_seed + 11)   # own stream: older statistics stay reproducible
    summary = dict(world=world.to_dict(), world_stats=world_stats(st, cfg, cs),
                   metrics_version=METRICS_VERSION,
                   filters={f: summarise(obs, cols, masks[f], streams, rng, cfg, ex, xrng) for f in FILTERS},
                   jaccard=jaccard(obs["prior"], masks))

    if out_dir is not None and cfg.save_observers:
        d = os.path.join(out_dir, "observers")
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, f"{world.world_id}.npz"),
                            **{k: v for k, v in obs.items()},
                            **{f"m_{k}": v.astype(np.float32) for k, v in cols.items()},
                            **{f"adm_{f}": masks[f] for f in FILTERS})
    return summary


def admissible_masks(obs, cols, top_q: float) -> dict[str, np.ndarray]:
    n = obs["fam"].size
    masks = {"none": np.ones(n, bool)}
    for f in SCORE_FILTERS:
        sc = cols[f]
        ok = np.isfinite(sc) & (sc > 1e-3)
        if ok.sum() == 0:
            masks[f] = ok
            continue
        thr = np.quantile(sc[ok], 1.0 - top_q)
        masks[f] = ok & (sc >= thr)
    # IIT exclusion postulate: among overlapping candidates (same start & drift), only the
    # grain with maximal integration exists as an observer.
    phi = cols["integration"]
    m = np.zeros(n, bool)
    world_readers = np.where(obs["fam"] > 0)[0]
    groups: dict = {}
    for i in world_readers:
        groups.setdefault((int(obs["x0"][i]), int(obs["v"][i])), []).append(i)
    for idx in groups.values():
        idx = np.array(idx)
        best = idx[np.argmax(phi[idx])]
        if phi[best] > 1e-3:
            m[best] = True
    masks["integration_excl"] = m
    return masks


def summarise(obs, cols, mask, streams, rng, cfg, ex=None, xrng=None) -> dict:
    prior = obs["prior"]
    n_adm = int(mask.sum())
    out = dict(n_adm=n_adm, mass=float(prior[mask].sum()))
    if n_adm == 0:
        return out
    w = prior[mask] / prior[mask].sum()
    k = cols["klass"][mask]
    out["frac_trivial"] = float(w[k == 0].sum())
    out["frac_lawful"] = float(w[k == 1].sum())
    out["frac_noisy"] = float(w[k == 2].sum())
    nt = k > 0
    out["n_nontrivial"] = int(nt.sum())
    if nt.any():
        wn = w[nt] / w[nt].sum()
        out["lawful_nt"] = float(wn[k[nt] == 1].sum())
        out["lawful_nt_unweighted"] = float(np.mean(k[nt] == 1))
        for name in ("Lambda", "hnorm", "gap", "rgap", "div"):
            out[f"mean_{name}_nt"] = float(np.sum(wn * cols[name][mask][nt]))
        fl = (cols["div"][mask][nt] > DIVERGE) & (cols["rgap"][mask][nt] < ROBUST)
        out["facts_change_laws_hold_nt"] = float(wn[fl].sum())
        out["facts_change_nt"] = float(wn[cols["div"][mask][nt] > DIVERGE].sum())
    out["mean_Lambda"] = float(np.sum(w * cols["Lambda"][mask]))
    fam = obs["fam"][mask]
    for j, name in enumerate(FAMILIES):
        out[f"famshare_{name}"] = float(w[fam == j].sum())
    # intersubjectivity: compression distance between experience streams of admissible,
    # non-trivial observers in the same world, versus random non-trivial raw windows.
    idx = np.where(mask)[0][nt]
    out["ncd_adm"] = _ncd_pairs(idx, prior[idx], streams, rng, cfg.ncd_pairs)
    base = np.where((obs["fam"] == 1) & (cols["klass"] > 0))[0]
    out["ncd_base"] = _ncd_pairs(base, None, streams, rng, cfg.ncd_pairs)
    # aligned mutual information between world-reading observers whose footprints never
    # overlap (so they cannot agree just by reading the same cells), versus random windows.
    # In a noise world this is ~0 by construction; H5 is tested against that.
    if ex is not None and xrng is not None:
        wr = idx[obs["fam"][idx] > 0]
        out["xmi_adm"] = _xmi_pairs(wr, prior[wr], obs, streams, ex, xrng, cfg.ncd_pairs)
        out["xmi_base"] = _xmi_pairs(base, None, obs, streams, ex, xrng, cfg.ncd_pairs)
    return out


def _footprint(obs, i, ex) -> np.ndarray:
    fam, w, s = int(obs["fam"][i]), int(obs["w"][i]), int(obs["s"][i])
    foot = w if fam == 3 else w * s
    return ex._cols(int(obs["x0"][i]), int(obs["v"][i]), np.arange(foot))


def _xmi_pairs(idx, p, obs, streams, ex, rng, n_pairs) -> float:
    if idx.size < 2:
        return float("nan")
    if p is not None:
        p = p / p.sum()
    vals = []
    for _ in range(4 * n_pairs):
        if len(vals) == n_pairs:
            break
        a, b = rng.choice(idx, size=2, replace=False, p=p) if p is not None else rng.choice(idx, 2, replace=False)
        fa, fb = _footprint(obs, a, ex), _footprint(obs, b, ex)
        if (fa[:, :, None] == fb[:, None, :]).any():
            continue
        vals.append(info.mi_norm_debiased(streams[a], streams[b], rng))
    return float(np.mean(vals)) if vals else float("nan")


_ZCACHE: dict = {}


def _ncd_pairs(idx, p, streams, rng, n_pairs) -> float:
    if idx.size < 2:
        return float("nan")
    if p is not None:
        p = p / p.sum()
    vals = []
    for _ in range(n_pairs):
        a, b = rng.choice(idx, size=2, replace=False, p=p) if p is not None else rng.choice(idx, 2, replace=False)
        vals.append(info.ncd(streams[a], streams[b]))
    return float(np.mean(vals))


def jaccard(prior, masks) -> dict:
    names = [f for f in FILTERS if f != "none"]
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            u = prior[masks[a] | masks[b]].sum()
            out[f"{a}|{b}"] = float(prior[masks[a] & masks[b]].sum() / u) if u > 0 else float("nan")
    return out


def world_stats(st, cfg, cs) -> dict:
    reg = st[cfg.T0:cfg.T0 + cfg.T].astype(np.int64)
    blocks = (np.roll(reg, 1, axis=1) * 4 + reg * 2 + np.roll(reg, -1, axis=1)).ravel()
    p2 = float(np.mean(np.all(reg[2:] == reg[:-2], axis=1)))
    tp = translation_period(st[cfg.T0:cfg.T0 + cfg.T], TPERIOD_SCAN)
    pmax = getattr(cfg, "dyn_pmax", 16)
    return dict(density=float(reg.mean()), block_entropy=float(info.entropy(blocks) / 3.0),
                period2_fraction=p2, n_causal_states=int(cs.n_states),
                tperiod=None if tp is None else tp[0], tshift=None if tp is None else tp[1],
                dyn_rich=bool(tp is None or tp[0] > pmax))
