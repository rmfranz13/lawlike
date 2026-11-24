"""Information-theoretic estimators and the computable stand-in for a universal prior.

Estimators are plug-in entropies with the Miller-Madow bias correction. Mutual-information
style scores are additionally debiased by subtracting the same quantity computed on a
surrogate in which one variable is permuted *within strata of the conditioning variable*
(so the surrogate keeps the marginals and the conditioning structure, and destroys only the
dependence being measured). These are the standard cheap fixes; they are not perfect,
especially for large alphabets at short T. See README "Known limitations".

Coding: a Bayesian mixture over Markov orders 0..K, each order using the Krichevsky-Trofimov
(KT) estimator. The mixture code length is an online (sequential) prediction code, i.e. a
computable approximation of "how surprised would a simplicity-prior predictor be".
"""
from __future__ import annotations

import math
import zlib

import numpy as np

LN2 = math.log(2.0)


# ----------------------------------------------------------------------------- entropy

def _counts(ids: np.ndarray) -> np.ndarray:
    n = ids.size
    if n == 0:
        return np.zeros(0, np.int64)
    mx = int(ids.max())
    if mx < 8 * n + 64 and int(ids.min()) >= 0:
        c = np.bincount(ids)
        return c[c > 0]
    s = np.sort(ids)
    edges = np.flatnonzero(s[1:] != s[:-1]) + 1
    return np.diff(np.concatenate(([0], edges, [n])))


def entropy(ids: np.ndarray) -> float:
    """Miller-Madow corrected plug-in entropy (bits) of a 1-D array of integer labels."""
    if ids.size == 0:
        return 0.0
    c = _counts(ids)
    n = c.sum()
    p = c / n
    h = float(-(p * np.log2(p)).sum())
    return h + (len(c) - 1) / (2.0 * n * LN2)


def _compact(c: np.ndarray) -> np.ndarray:
    _, ci = np.unique(c, return_inverse=True)
    return ci.astype(np.int64)


def combine(*cols: np.ndarray) -> np.ndarray:
    """Combine several non-negative integer label arrays into one joint label array."""
    out = cols[0].astype(np.int64)
    if out.size and int(out.min()) < 0:
        out = _compact(out)
    for c in cols[1:]:
        c = c.astype(np.int64)
        if c.size and int(c.min()) < 0:
            c = _compact(c)
        base = int(c.max()) + 1 if c.size else 1
        if out.size and int(out.max()) * base + base > 2**62:
            out, c = _compact(out), _compact(c)
            base = int(c.max()) + 1
        out = out * base + c
    return out


def mi(x: np.ndarray, y: np.ndarray) -> float:
    return entropy(x) + entropy(y) - entropy(combine(x, y))


def cmi(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """I(X;Y|Z)."""
    return (entropy(combine(x, z)) + entropy(combine(y, z))
            - entropy(combine(x, y, z)) - entropy(z))


def permute_within(y: np.ndarray, z: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    """Shuffle y within groups of equal z (all of y if z is None)."""
    if z is None:
        return y[rng.permutation(y.size)]
    order_a = np.argsort(z, kind="stable")
    order_b = np.lexsort((rng.random(z.size), z))
    out = np.empty_like(y)
    out[order_a] = y[order_b]
    return out


def mi_debiased(x, y, rng) -> float:
    return mi(x, y) - mi(x, permute_within(y, None, rng))


def mi_norm_debiased(x, y, rng) -> float:
    """Permutation-debiased I(x;y) / min(H(x), H(y)): 0 for independent streams, 1 for copies."""
    h = min(entropy(x), entropy(y))
    return 0.0 if h < 1e-9 else mi_debiased(x, y, rng) / h


def cmi_debiased(x, y, z, rng) -> float:
    return cmi(x, y, z) - cmi(x, permute_within(y, z, rng), z)


# ----------------------------------------------------------------------------- integration

def phi_whole_minus_parts(parts_t: np.ndarray, parts_t1: np.ndarray, radix: int) -> float:
    """A cheap IIT-flavoured integration proxy ("stochastic interaction", Ay 2001/2015).

    phi = min over bipartitions (A,B) of  I(S_t;S_t+1) - I(A_t;A_t+1) - I(B_t;B_t+1)

    i.e. how much the whole's own temporal predictive information exceeds what the two
    halves carry separately, at the weakest cut. It is NOT IIT 3.0/4.0 phi (that is
    intractable here); it is in the same family as the measures used to sanity-check IIT
    on small systems. Returns 0 for a single part (no cut exists).
    """
    w = parts_t.shape[1]
    if w < 2:
        return 0.0
    weights = radix ** np.arange(w, dtype=np.int64)

    def sub_id(arr, mask):
        cols = [j for j in range(w) if (mask >> j) & 1]
        return (arr[:, cols] * weights[: len(cols)]).sum(axis=1)

    full = (1 << w) - 1
    cache: dict[int, float] = {}

    def mi_mask(mask):
        if mask not in cache:
            cache[mask] = mi(sub_id(parts_t, mask), sub_id(parts_t1, mask))
        return cache[mask]

    whole = mi_mask(full)
    best = math.inf
    for mask in range(1, 1 << (w - 1)):   # each bipartition once (top part never in A)
        val = whole - mi_mask(mask) - mi_mask(full ^ mask)
        best = min(best, val)
    return max(0.0, best)


# ----------------------------------------------------------------------------- coding

_P1 = np.uint64(0x9E3779B97F4A7C15)


def _context_hashes(seq: np.ndarray, K: int) -> list[np.ndarray]:
    T = seq.size
    s1 = seq.astype(np.uint64) + np.uint64(1)   # 0 is reserved for "before the start"
    ctx = [np.zeros(T, np.uint64)]
    h = np.zeros(T, np.uint64)
    with np.errstate(over="ignore"):
        for k in range(1, K + 1):
            prev = np.zeros(T, np.uint64)
            if k < T:
                prev[k:] = s1[:-k]
            h = h * _P1 + prev
            ctx.append(h.copy())
    return ctx


def _cumcount(keys: np.ndarray) -> np.ndarray:
    """For each position, how many earlier positions share its key."""
    n = keys.size
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    start = np.ones(n, bool)
    start[1:] = sk[1:] != sk[:-1]
    idx = np.arange(n)
    grp = np.maximum.accumulate(np.where(start, idx, 0))
    out = np.empty(n, np.int64)
    out[order] = idx - grp
    return out


class _Table:
    """Counts of a training key set, for frozen-model lookups."""

    def __init__(self, train: np.ndarray):
        if train.size:
            self.u, self.c = np.unique(train, return_counts=True)
        else:
            self.u = self.c = None

    def __call__(self, query: np.ndarray) -> np.ndarray:
        if self.u is None:
            return np.zeros(query.size, np.int64)
        i = np.clip(np.searchsorted(self.u, query), 0, self.u.size - 1)
        return np.where(self.u[i] == query, self.c[i], 0)


def _mix(cls: np.ndarray, log_prior: np.ndarray) -> float:
    a = log_prior - cls
    m = a.max()
    return float(-(m + math.log2(np.sum(np.exp2(a - m)))))


def mixture_codes(seq: np.ndarray, alt: np.ndarray | None, K: int, alphabet: int) -> dict:
    """Code lengths (bits) for seq under a KT mixture over Markov orders 0..K.

    Returns:
      total        : online code length of the whole sequence
      first        : online code length of the first half
      second_online: online code length of the second half (keeps learning)
      second_frozen: second half coded by the model frozen after the first half
      alt_frozen   : second half of `alt` coded by the same frozen model (perturbation test)
      by_horizon   : {m: online code length of the whole sequence under orders 0..m only}, the
                     experience of an observer whose state is its last m readings (README 3c.2)
    """
    T = seq.size
    T2 = T // 2
    A = max(2, alphabet)
    K = min(K, max(0, T2 - 1))
    log_prior = -(np.arange(K + 1) + 1.0)
    log_prior -= math.log2(np.sum(np.exp2(log_prior)))

    if alt is not None and np.array_equal(alt, seq):
        alt = None
        same = True
    else:
        same = False
    # The order-(k+1) context at t+1 encodes exactly (symbol at t, order-k context at t), so it
    # doubles as the joint key for order k. Extend by one position so the last joint exists.
    ext = np.append(seq, 0)
    ctxs = _context_hashes(ext, K + 1)
    cc = [_cumcount(c) for c in ctxs]
    if alt is not None:
        actxs = _context_hashes(np.append(alt, 0), K + 1)
    cl_total = np.empty(K + 1)
    cl_first = np.empty(K + 1)
    cl_frozen = np.empty(K + 1)
    cl_alt = np.empty(K + 1)
    for k in range(K + 1):
        n_c = cc[k][:T]
        n_j = cc[k + 1][1:T + 1]
        bits = -np.log2((n_j + 0.5) / (n_c + 0.5 * A))
        cl_total[k] = bits.sum()
        cl_first[k] = bits[:T2].sum()
        # frozen model = counts from the first half only
        tj, tc = _Table(ctxs[k + 1][1:T2 + 1]), _Table(ctxs[k][:T2])
        fj = tj(ctxs[k + 1][T2 + 1:T + 1])
        fc = tc(ctxs[k][T2:T])
        cl_frozen[k] = (-np.log2((fj + 0.5) / (fc + 0.5 * A))).sum()
        if alt is not None:
            fj = tj(actxs[k + 1][T2 + 1:T + 1])
            fc = tc(actxs[k][T2:T])
            cl_alt[k] = (-np.log2((fj + 0.5) / (fc + 0.5 * A))).sum()

    total = _mix(cl_total, log_prior)
    first = _mix(cl_first, log_prior)
    post = log_prior - cl_first
    post -= np.log2(np.sum(np.exp2(post - post.max()))) + post.max()
    frozen = _mix(cl_frozen, post)
    alt_frozen = frozen if same else (_mix(cl_alt, post) if alt is not None else float("nan"))
    by_h = {}
    for m in range(K + 1):
        lp = log_prior[:m + 1] - math.log2(np.sum(np.exp2(log_prior[:m + 1])))
        by_h[m] = _mix(cl_total[:m + 1], lp)
    return dict(total=total, first=first, second_online=total - first,
                second_frozen=frozen, alt_frozen=alt_frozen, T=T, T2=T2, by_horizon=by_h)


# ----------------------------------------------------------------------------- compression distance

def zlen(seq: np.ndarray) -> int:
    return len(zlib.compress(seq.astype(np.uint16).tobytes(), 9))


def ncd(a: np.ndarray, b: np.ndarray, ca: int | None = None, cb: int | None = None) -> float:
    """Normalised compression distance (Cilibrasi & Vitanyi). ~0 shared, ~1 unrelated."""
    ca = zlen(a) if ca is None else ca
    cb = zlen(b) if cb is None else cb
    cab = zlen(np.concatenate([a, b]))
    return (cab - min(ca, cb)) / max(ca, cb)
