"""Experiment presets.

Every number here is a *convention* chosen before seeing results. If you change one
after looking at results, write down why in NOTES.md (see README, "Guarding against
the wrong mountain").
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace


@dataclass
class Exp1Config:
    name: str = "quick"

    # --- world geometry ---
    N: int = 128            # lattice width (periodic)
    T: int = 512            # experience length (time steps an observer sees)
    T0: int = 64            # transient discarded before observers start watching

    # --- rule set ---
    # "all" = the 88 minimal equivalence classes of elementary CAs (weighted by class size),
    # or an explicit list of rule numbers (mapped to their class representatives).
    rules: object = "all"

    # --- world priors ---
    ic_max_len: int = 6     # longest seed in the simplicity prior over initial conditions
    ic_exact_len: int = 3   # every seed up to this length is enumerated exactly
    ic_per_len: int = 1     # seeds sampled per (length, constructor) beyond ic_exact_len (importance-weighted)
    n_uniform: int = 2      # uniformly random initial conditions per rule (uniform arm)
    n_noise: int = 16       # pure-noise worlds (no rule at all)

    # --- observers ---
    pos_stride: int = 32    # observer start positions every pos_stride cells (importance-weighted)
    window_widths: tuple = (1, 2, 3, 4, 5)
    coarse_scales: tuple = (2, 3)
    coarse_widths: tuple = (1, 2, 3)
    causal_widths: tuple = (1, 2)
    # drift speeds p/q (cells per step) an observer can follow; lets observers ride gliders
    velocities: tuple = ((0, 1), (1, 1), (-1, 1), (1, 2), (-1, 2), (1, 3), (-1, 3), (2, 3), (-2, 3))
    null_max_period: int = 6

    # --- local causal states (Rupe & Crutchfield style light cones) ---
    hp: int = 3             # past light-cone depth
    hf: int = 2             # future light-cone depth
    causal_min_count: int = 16
    causal_eps: float = 0.15   # total-variation merge threshold
    causal_max_states: int = 48

    # --- coding / metrics ---
    K: int = 8              # max Markov order in the KT mixture (computable stand-in for the universal prior)
    top_q: float = 0.2      # fraction of observers admitted by each score-based filter
    ncd_pairs: int = 24     # observer pairs sampled per world for intersubjectivity

    # --- bookkeeping ---
    seed: int = 12345
    save_observers: bool = True

    # --- Phase 2a confirmation design (README 3b.6, 3b.7). Defaults reproduce Phase 1. ---
    # "crossed": one set of sampled seeds shared by every rule (Phase 1).
    # "per_rule": sampled seeds drawn independently for each rule (seed clusters nest in rules).
    seed_mode: str = "crossed"
    # run embed seeds under the complement orbit too, instead of the complement approximation
    complement_embed: bool = False
    # sampled seeds per (rule, length) for long seeds; 0 = use ic_per_len
    ic_per_len_long: int = 0
    ic_long_from: int = 4          # lengths >= this count as "long"
    # simplicity-arm programs in these result directories are never sampled (fresh data)
    exclude_programs_from: tuple = ()
    # translation period up to which a world counts as dynamically trivial (README 3b.7 A1)
    dyn_pmax: int = 16

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def hash(self) -> str:
        d = self.to_dict()
        # fields added for Phase 2 are left out of the hash while at their defaults, so
        # Phase 1 configs keep their hashes (and their result directories stay resumable)
        for k, v in PHASE2_DEFAULTS.items():
            if d.get(k) == v:
                d.pop(k, None)
        blob = json.dumps(d, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:10]


PHASE2_DEFAULTS = dict(seed_mode="crossed", complement_embed=False, ic_per_len_long=0, ic_long_from=4,
                       exclude_programs_from=(), dyn_pmax=16)


PRESETS: dict[str, Exp1Config] = {
    # ~1-2 minutes. Checks that every piece runs end to end. Results are NOT meaningful.
    "smoke": Exp1Config(
        name="smoke", N=64, T=256, T0=32,
        rules=[0, 4, 18, 30, 54, 90, 110, 184, 232],
        ic_max_len=3, ic_exact_len=2, ic_per_len=1, n_uniform=1, n_noise=3,
        pos_stride=16, window_widths=(1, 2, 3), coarse_widths=(1, 2),
        causal_widths=(1,), null_max_period=4, ncd_pairs=8,
        velocities=((0, 1), (1, 1), (-1, 1)),
    ),
    # ~10-25 minutes on a 4-8 core laptop. First real look; good for iterating on code.
    "quick": Exp1Config(name="quick", ic_max_len=5, ic_exact_len=2),
    # ~1.5-3 hours on a 4-8 core laptop. Press go, walk away.
    "full": Exp1Config(
        name="full", N=256, T=1024, T0=128,
        ic_max_len=10, ic_per_len=2, n_uniform=6, n_noise=48,
        pos_stride=16, ncd_pairs=32,
    ),
    # ~8-16 hours (overnight). Use once quick/full say the question is worth the electricity.
    "deep": Exp1Config(
        name="deep", N=256, T=2048, T0=128,
        ic_max_len=14, ic_exact_len=4, ic_per_len=2, n_uniform=12, n_noise=96,
        pos_stride=16, ncd_pairs=48,
    ),
}

# Phase 2a confirmation run (README 3b.6): new RNG seed, seeds nested in rules, complement
# approximation removed, long seeds oversampled, programs disjoint from Phase 1.
_P1_DIRS = ("results/quick", "results/full")
_C2A = dict(seed=20261110, seed_mode="per_rule", complement_embed=True, exclude_programs_from=_P1_DIRS)
PRESETS["smoke2a"] = replace(PRESETS["smoke"], name="smoke2a", ic_max_len=5, ic_exact_len=2,
                             ic_per_len_long=2, **_C2A)
PRESETS["quick2a"] = replace(PRESETS["quick"], name="quick2a", ic_max_len=7, ic_per_len_long=2, **_C2A)
PRESETS["full2a"] = replace(PRESETS["full"], name="full2a", ic_per_len_long=3, **_C2A)


def scaled_for_size_check(cfg: Exp1Config) -> Exp1Config:
    """README 3b.7 A7: 2N and 2T, transient and observer start grid scaled to match."""
    return replace(cfg, name=f"{cfg.name}_size", N=2 * cfg.N, T=2 * cfg.T, T0=2 * cfg.T0,
                   pos_stride=2 * cfg.pos_stride)


def get_preset(name: str) -> Exp1Config:
    if name not in PRESETS:
        raise SystemExit(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]
