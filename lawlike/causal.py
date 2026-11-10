"""Local causal states (after Shalizi; Rupe & Crutchfield 2018).

At each spacetime point, the past light cone (cells that could have influenced it) and the
future light cone (cells it could influence) are read off. Two pasts are "causally
equivalent" if they predict the same distribution over futures. Clustering pasts by that
criterion gives the local causal-state field. It is the least arbitrary coarse-graining we
know of: it is defined purely by prediction, with no hand-chosen features.

Local statistical complexity C(t,x) = -log2 P(state at (t,x)): domains (background
patterns) get low C, coherent structures (particles, gliders, defects) get high C.

Approximations here: finite cone depths, a total-variation merge threshold, likelihood
assignment for rarely seen pasts, and a cap on the number of states.
"""
from __future__ import annotations

import numpy as np


def _cone_keys(st: np.ndarray, depths, sign: int) -> np.ndarray:
    """Pack a light cone into an integer per spacetime point. sign=-1 past, +1 future."""
    key = np.zeros(st.shape, np.int64)
    bit = 0
    for d in depths:
        row = np.roll(st, -sign * d, axis=0)   # row[t] = st[t + sign*d]
        for j in range(-d, d + 1):
            key |= np.roll(row, -j, axis=1).astype(np.int64) << bit
            bit += 1
    return key


class CausalStates:
    def __init__(self, hp: int = 3, hf: int = 2, min_count: int = 8, eps: float = 0.15,
                 max_states: int = 48):
        self.hp, self.hf = hp, hf
        self.min_count, self.eps, self.max_states = min_count, eps, max_states
        self.n_future = 2 ** sum(2 * d + 1 for d in range(1, hf + 1))

    # -- keys
    def past_keys(self, st):
        return _cone_keys(st, range(0, self.hp + 1), -1)

    def future_keys(self, st):
        return _cone_keys(st, range(1, self.hf + 1), +1)

    def fit(self, st: np.ndarray, t_lo: int, t_hi: int) -> "CausalStates":
        """Infer the past -> state map from rows [t_lo, t_hi) of spacetime st."""
        assert t_lo >= self.hp and t_hi <= st.shape[0] - self.hf
        P = self.past_keys(st)[t_lo:t_hi].ravel()
        F = self.future_keys(st)[t_lo:t_hi].ravel()
        nF = self.n_future
        joint, jc = np.unique(P * nF + F, return_counts=True)
        jp, jf = joint // nF, joint % nF
        pasts, pinv = np.unique(jp, return_inverse=True)
        ptot = np.bincount(pinv, weights=jc).astype(np.int64)

        common = np.where(ptot >= self.min_count)[0]
        common = common[np.argsort(-ptot[common], kind="stable")]
        # dense conditional future distributions for common pasts
        row_of = -np.ones(pasts.size, np.int64)
        row_of[common] = np.arange(common.size)
        dense = np.zeros((common.size, nF))
        m = row_of[pinv] >= 0
        np.add.at(dense, (row_of[pinv[m]], jf[m]), jc[m])

        centroids: list[np.ndarray] = []  # summed counts
        labels_common = np.empty(common.size, np.int64)
        for i in range(common.size):
            row = dense[i]
            p = row / row.sum()
            if centroids:
                C = np.array(centroids)
                Cn = C / C.sum(axis=1, keepdims=True)
                tv = 0.5 * np.abs(Cn - p).sum(axis=1)
                j = int(np.argmin(tv))
                # An empirical distribution from n samples over k outcomes sits ~sqrt(k/n)/2 (TV)
                # from its own truth, so the merge threshold widens for rarely seen pasts.
                n, k = row.sum(), max(1, int((row > 0).sum()))
                thr = self.eps + 0.5 * np.sqrt(k / n)
                if tv[j] < thr or len(centroids) >= self.max_states:
                    centroids[j] = centroids[j] + row
                    labels_common[i] = j
                    continue
            centroids.append(row.copy())
            labels_common[i] = len(centroids) - 1
        if not centroids:  # degenerate (tiny input): one state
            centroids = [np.ones(nF)]
        C = np.array(centroids)
        logPc = np.log((C + 0.5) / (C + 0.5).sum(axis=1, keepdims=True))   # (S, nF)

        labels = np.empty(pasts.size, np.int64)
        labels[common] = labels_common
        rare = np.where(row_of < 0)[0]
        if rare.size:
            mr = row_of[pinv] < 0
            rp = pinv[mr]                    # past index for each rare joint entry
            ll = jc[mr][:, None] * logPc[:, jf[mr]].T     # (entries, S)
            rare_rows = -np.ones(pasts.size, np.int64)
            rare_rows[rare] = np.arange(rare.size)
            acc = np.zeros((rare.size, C.shape[0]))
            np.add.at(acc, rare_rows[rp], ll)
            labels[rare] = np.argmax(acc, axis=1)

        self.past_values = pasts
        self.past_labels = labels
        self.n_states = C.shape[0]
        # state frequencies on the fitting region -> local statistical complexity
        lab = labels[pinv]
        freq = np.bincount(lab, weights=jc, minlength=self.n_states)
        freq = freq / freq.sum()
        self.state_freq = freq
        self.unknown = self.n_states          # label for pasts never seen while fitting
        return self

    def label(self, st: np.ndarray) -> np.ndarray:
        """Causal-state field for every spacetime point (rows < hp are garbage; don't read them)."""
        P = self.past_keys(st)
        i = np.clip(np.searchsorted(self.past_values, P), 0, self.past_values.size - 1)
        found = self.past_values[i] == P
        return np.where(found, self.past_labels[i], self.unknown)

    def complexity(self, labels: np.ndarray) -> np.ndarray:
        f = np.append(self.state_freq, 0.0)
        floor = 1.0 / max(1, labels.size)
        return -np.log2(np.maximum(f[labels], floor))
