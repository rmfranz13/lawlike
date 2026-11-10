"""Observers: programs that read an experience stream out of a world.

Every observer is itself a short program, so it gets a prior weight 2^-L just like a world.
Summing over (world, observer) pairs is then marginalisation, not an arbitrary choice
(README, "The observer is a program too"). Families (2 bits to pick one):

  null    : ignores the world entirely; emits a fixed periodic bit pattern.
            L = 2 + EliasGamma(period) + period.  The "trivial observer" control. It is the
            cheapest thing that produces an experience stream, so under a bare 2^-L measure
            it should dominate. Any criterion worth having must exclude it for a reason.
  window  : reads w raw cells at a start position, drifting at one of the configured speeds
            p/q cells per step (so it can ride a glider). L = 2 + EG(w) + log2(#speeds) + log2(N).
  coarse  : reads w*s raw cells, majority-votes each block of s (a computationally bounded
            observer in Wolfram's sense). L = window cost + EG(s).
  causal  : reads w local causal-state labels (Crutchfield). L = window cost + EG(hp)+EG(hf).

Start positions are sampled on a stride grid; each sampled position stands in for
pos_stride positions (importance weight), so total family mass does not depend on N.
"""
from __future__ import annotations

import math

import numpy as np

from .worlds import elias_gamma_len

FAMILIES = ("null", "window", "coarse", "causal")
FAM_BITS = 2.0


def enumerate_observers(cfg) -> dict[str, np.ndarray]:
    rows = []  # (fam, x0, w, v, s, period, pattern, L, log2_importance)
    pos = list(range(0, cfg.N, cfg.pos_stride))
    log_imp = math.log2(cfg.N / len(pos))
    pos_bits = math.log2(cfg.N)
    v_bits = math.log2(len(cfg.velocities))
    for p in range(1, cfg.null_max_period + 1):
        for pat in range(2 ** p):
            rows.append((0, -1, 0, 0, 0, p, pat, FAM_BITS + elias_gamma_len(p) + p, 0.0))
    for w in cfg.window_widths:
        for v in range(len(cfg.velocities)):
            for x0 in pos:
                L = FAM_BITS + elias_gamma_len(w) + v_bits + pos_bits
                rows.append((1, x0, w, v, 1, 0, 0, L, log_imp))
    for s in cfg.coarse_scales:
        for w in cfg.coarse_widths:
            for v in range(len(cfg.velocities)):
                for x0 in pos:
                    L = FAM_BITS + elias_gamma_len(w) + elias_gamma_len(s) + v_bits + pos_bits
                    rows.append((2, x0, w, v, s, 0, 0, L, log_imp))
    hp_bits = elias_gamma_len(cfg.hp) + elias_gamma_len(cfg.hf)
    for w in cfg.causal_widths:
        for v in range(len(cfg.velocities)):
            for x0 in pos:
                L = FAM_BITS + elias_gamma_len(w) + hp_bits + v_bits + pos_bits
                rows.append((3, x0, w, v, 1, 0, 0, L, log_imp))
    arr = np.array(rows, dtype=float)
    out = dict(fam=arr[:, 0].astype(int), x0=arr[:, 1].astype(int), w=arr[:, 2].astype(int),
               v=arr[:, 3].astype(int), s=arr[:, 4].astype(int), period=arr[:, 5].astype(int),
               pattern=arr[:, 6].astype(int), L=arr[:, 7], log_imp=arr[:, 8])
    logw = -out["L"] + out["log_imp"]
    logw -= logw.max()
    pw = np.exp2(logw)
    out["prior"] = pw / pw.sum()     # P(observer), normalised over the whole enumerated set
    return out


def null_stream(period: int, pattern: int, T: int) -> np.ndarray:
    bits = np.array([(pattern >> i) & 1 for i in range(period)], dtype=np.int64)
    return np.tile(bits, -(-T // period))[:T]


class Extractor:
    """Reads observer streams out of a (spacetime, causal-label) pair."""

    def __init__(self, cfg, st: np.ndarray, labels: np.ndarray | None, cplx: np.ndarray | None,
                 n_labels: int):
        self.cfg = cfg
        self.st = st
        self.labels = labels
        self.cplx = cplx
        self.radix_causal = n_labels
        self.rows = np.arange(cfg.T0, cfg.T0 + cfg.T)[:, None]
        self.tt = np.arange(cfg.T)[:, None]

    def _cols(self, x0, v, offsets):
        p, q = self.cfg.velocities[v]
        return (x0 + (p * self.tt) // q + offsets[None, :]) % self.cfg.N

    def read(self, fam: int, x0: int, w: int, v: int, s: int) -> dict:
        """Returns parts (T,w), radix, sym (T,), env (T,), ext (T,), cplx_mean."""
        if fam == 3:
            foot = w
            cols = self._cols(x0, v, np.arange(w))
            parts = self.labels[self.rows, cols].astype(np.int64)
            radix = self.radix_causal
        else:
            foot = w * s
            cols = self._cols(x0, v, np.arange(foot))
            raw = self.st[self.rows, cols].astype(np.int64)
            if fam == 2:
                parts = (raw.reshape(self.cfg.T, w, s).sum(axis=2) * 2 >= s).astype(np.int64)
            else:
                parts = raw
            radix = 2
        weights = radix ** np.arange(parts.shape[1], dtype=np.int64)
        sym = (parts * weights).sum(axis=1)
        # boundary (blanket) cells immediately outside the footprint, and the next ones out
        b = self._cols(x0, v, np.array([-1, foot]))
        e = self._cols(x0, v, np.array([-3, -2, foot + 1, foot + 2]))
        B = self.st[self.rows, b].astype(np.int64)
        X = self.st[self.rows, e].astype(np.int64)
        env = B[:, 0] * 2 + B[:, 1]
        ext = X[:, 0] * 8 + X[:, 1] * 4 + X[:, 2] * 2 + X[:, 3]
        cm = float(self.cplx[self.rows, cols].mean()) if self.cplx is not None else float("nan")
        return dict(parts=parts, radix=radix, sym=sym, env=env, ext=ext, cplx=cm)
