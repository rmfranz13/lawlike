"""Cluster bootstrap matched to how worlds were actually sampled.

The world list is not an i.i.d. sample. It is a crossed design: in Phase 1 every rule is run
on the *same* sampled seeds, so two worlds sharing a seed are correlated even when their
rules differ. Bootstrapping worlds as if they were independent ignores that and gives
intervals that are too narrow (NOTES.md, 2026-10-23).

Each bootstrap replicate draws, independently:

  rules         : resample the rules with replacement (superpopulation view: "would this
                  hold for other rules like these?"). Shared by the simplicity and uniform
                  arms, so simplicity - uniform contrasts are paired. Pass resample_rules=False
                  to treat the enumerated rule set as the whole population.
  sampled seeds : within each seed-length stratum, resample the sampled seeds with
                  replacement (nested=True, Phase 2a: within each (rule, length) stratum). Enumerated seeds (length <= ic_exact_len) are the whole
                  population and stay fixed.
  uniform ICs   : within each rule, resample its uniform-arm replicates.
  noise worlds  : resample i.i.d.

A world's multiplier in a replicate is the product of its counts, and every weighted mean is
recomputed with weights (prior x multiplier). Strata holding a single sampled seed contribute
no variance (one draw cannot estimate its own spread); the report notes this.
"""
from __future__ import annotations

import numpy as np

B = 400


def seed_cluster(world: dict, ic_exact_len: int, nested: bool = False):
    """(stratum, key) of the sampled seed a simplicity world was built from, or None if enumerated.

    crossed (Phase 1): one cluster per seed, shared by every rule, stratified by length.
    nested (Phase 2a): seeds were drawn per rule, so the cluster is (rule, seed) within the
    stratum (rule, length).
    """
    bits = world["seed_bits"]
    if len(bits) <= ic_exact_len:
        return None
    if nested:
        return (world["rule"], len(bits)), (world["rule"], bits)
    return len(bits), bits


class ClusterBootstrap:
    def __init__(self, worlds: list[dict], ic_exact_len: int, rng: np.random.Generator,
                 resample_rules: bool = True, n_boot: int = B, nested: bool = False):
        n = len(worlds)
        M = np.ones((n_boot, n))
        arm = np.array([w["arm"] for w in worlds])

        # rules: one draw per replicate, shared across simplicity and uniform arms
        rule_ids = sorted({w["rule"] for w in worlds if w["arm"] != "noise"})
        if resample_rules and rule_ids:
            rpos = {r: j for j, r in enumerate(rule_ids)}
            counts = _counts(rng, n_boot, len(rule_ids))
            cols = np.array([rpos.get(w["rule"], -1) for w in worlds])
            ok = arm != "noise"
            M[:, ok] *= counts[:, cols[ok]]

        # sampled seeds: crossed with rules, resampled within length strata
        strata: dict = {}
        for w in worlds:
            if w["arm"] == "simplicity":
                c = seed_cluster(w, ic_exact_len, nested)
                if c is not None and c[1] not in strata.setdefault(c[0], []):
                    strata[c[0]].append(c[1])
        single = [k for k, v in strata.items() if len(v) < 2]
        self.single_seed_strata = sorted({k[1] for k in single} if nested else single)
        self.n_single, self.n_strata = len(single), len(strata)
        seed_count = {}
        for ell, keys in strata.items():
            cnt = _counts(rng, n_boot, len(keys))
            for j, k in enumerate(keys):
                seed_count[k] = cnt[:, j]
        for i, w in enumerate(worlds):
            if w["arm"] == "simplicity":
                c = seed_cluster(w, ic_exact_len, nested)
                if c is not None:
                    M[:, i] *= seed_count[c[1]]

        # uniform-arm replicates, nested within rule
        by_rule: dict = {}
        for i, w in enumerate(worlds):
            if w["arm"] == "uniform":
                by_rule.setdefault(w["rule"], []).append(i)
        for idx in by_rule.values():
            M[:, idx] *= _counts(rng, n_boot, len(idx))

        # noise worlds: i.i.d.
        idx = np.where(arm == "noise")[0]
        if idx.size:
            M[:, idx] *= _counts(rng, n_boot, idx.size)

        self.M = M
        self.n_boot = n_boot

    def means(self, idx: np.ndarray, w: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Bootstrap distribution of the weighted mean of v (NaN where a replicate has no weight)."""
        mw = self.M[:, idx] * w[None, :]
        den = mw.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, (mw * v[None, :]).sum(axis=1) / den, np.nan)


def _counts(rng, n_boot, k):
    """Multinomial resampling counts: (n_boot, k), each row sums to k."""
    if k == 0:
        return np.zeros((n_boot, 0))
    draws = rng.integers(0, k, size=(n_boot, k))
    out = np.zeros((n_boot, k))
    np.add.at(out, (np.repeat(np.arange(n_boot), k), draws.ravel()), 1.0)
    return out
