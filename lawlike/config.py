"""Experiment presets.

Every number here is a *convention* chosen before seeing results. If you change one
after looking at results, write down why in NOTES.md (see README, "Guarding against
the wrong mountain").
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


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

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:10]


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


def get_preset(name: str) -> Exp1Config:
    if name not in PRESETS:
        raise SystemExit(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]
