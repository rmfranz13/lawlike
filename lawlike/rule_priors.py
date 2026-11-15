"""Phase 2 priors over rules (README section 3b.2).

Each function returns a normalised weight for every one of the 256 ECAs (Wolfram numbering,
neighbourhood index 4*l + 2*c + r).

  phase1  : every rule 8 bits, i.e. uniform. The Phase 1 reference point.
  M_A     : family table (primary). program = 2-bit family tag + EliasGamma(r) + table.
            At r = 1: totalistic 7 bits, outer-totalistic 9, general 11. A rule's weight is
            the SUM over every r = 1 program that produces it (totalistic is a subset of
            outer-totalistic is a subset of general).
  M_B     : shortest Boolean formula (sensitivity). Prefix formula over the 10 tokens
            {NOT, AND, OR, XOR, XNOR, l, c, r, 0, 1}; weight proportional to 10^-(min size).

    python -m lawlike.rule_priors      # prints the checks quoted in README 3b.2
"""
from __future__ import annotations

import math

import numpy as np

from .worlds import elias_gamma_len, rule_class, rule_representatives

N_RULES = 256
FAM_TAG_BITS = 2


def _neigh(idx: int) -> tuple[int, int, int]:
    return (idx >> 2) & 1, (idx >> 1) & 1, idx & 1


def is_totalistic(rule: int) -> bool:
    out = {}
    for idx in range(8):
        s = sum(_neigh(idx))
        b = (rule >> idx) & 1
        if out.setdefault(s, b) != b:
            return False
    return True


def is_outer_totalistic(rule: int) -> bool:
    out = {}
    for idx in range(8):
        l, c, r = _neigh(idx)
        key = (c, l + r)
        b = (rule >> idx) & 1
        if out.setdefault(key, b) != b:
            return False
    return True


def phase1_weights() -> np.ndarray:
    return np.full(N_RULES, 1.0 / N_RULES)


def m_a_lengths(radius: int = 1) -> dict[str, int]:
    n = 2 * radius + 1
    head = FAM_TAG_BITS + elias_gamma_len(radius)
    return {"totalistic": head + (n + 1), "outer_totalistic": head + 2 * n, "general": head + 2 ** n}


def m_a_weights() -> np.ndarray:
    L = m_a_lengths(1)
    w = np.zeros(N_RULES)
    for rule in range(N_RULES):
        w[rule] = 2.0 ** -L["general"]
        if is_outer_totalistic(rule):
            w[rule] += 2.0 ** -L["outer_totalistic"]
        if is_totalistic(rule):
            w[rule] += 2.0 ** -L["totalistic"]
    return w / w.sum()


def m_a_star_weights(r_max: int = 40) -> np.ndarray:
    """M_A summed over programs at EVERY radius that compute the ECA (README 3b.7 A3).

    At r >= 2 a program computes an ECA only if it ignores the outer cells:
      totalistic table that is constant           -> rules 0, 255
      outer-totalistic table reading only centre  -> rules 0, 255, 204 (identity), 51 (NOT c)
      general table (one per ECA)                 -> every rule
    The series converges fast; r_max = 40 is far past double precision.
    """
    L1 = m_a_lengths(1)
    w = np.zeros(N_RULES)
    for rule in range(N_RULES):
        w[rule] = 2.0 ** -L1["general"]
        if is_outer_totalistic(rule):
            w[rule] += 2.0 ** -L1["outer_totalistic"]
        if is_totalistic(rule):
            w[rule] += 2.0 ** -L1["totalistic"]
    for r in range(2, r_max + 1):
        L = m_a_lengths(r)
        for rule in (0, 255):
            w[rule] += 2.0 ** -L["totalistic"]
        for rule in (0, 255, 204, 51):
            w[rule] += 2.0 ** -L["outer_totalistic"]
        if L["general"] < 1000:
            w += 2.0 ** -L["general"]
    return w / w.sum()


def min_formula_sizes() -> np.ndarray:
    """Minimal prefix-formula size (token count) of every 3-input Boolean function."""
    INF = 10 ** 9
    best = np.full(N_RULES, INF, dtype=np.int64)
    leaves = {"l": 240, "c": 204, "r": 170, "0": 0, "1": 255}
    by_size: dict[int, set[int]] = {1: set()}
    for tt in leaves.values():
        best[tt] = 1
        by_size[1].add(tt)
    ops = (lambda a, b: a & b, lambda a, b: a | b, lambda a, b: a ^ b, lambda a, b: 0xFF ^ (a ^ b))
    size = 1
    while (best >= INF).any():
        size += 1
        new = set()
        for f in by_size.get(size - 1, ()):  # NOT f
            new.add(0xFF ^ f)
        for a in range(1, size - 1):           # BIN f g, sizes a + b + 1 = size
            b = size - 1 - a
            for f in by_size.get(a, ()):
                for g in by_size.get(b, ()):
                    for op in ops:
                        new.add(op(f, g))
        fresh = {t for t in new if best[t] >= INF}
        for t in fresh:
            best[t] = size
        by_size[size] = fresh   # only functions first reached at this size are needed
    return best


def m_b_weights() -> np.ndarray:
    s = min_formula_sizes().astype(float)
    w = 10.0 ** -(s - s.min())
    return w / w.sum()


PRIORS = {"phase1": phase1_weights, "M_A": m_a_weights, "M_B": m_b_weights, "M_A*": m_a_star_weights}


def class_weights(name: str) -> dict[int, float]:
    """Total weight of each ECA symmetry class, keyed by its representative (sums to 1)."""
    w = PRIORS[name]()
    return {rep: float(sum(w[x] for x in rule_class(rep))) for rep in rule_representatives()}


def check_class_invariance(name: str) -> bool:
    w = PRIORS[name]()
    return all(np.allclose([w[x] for x in rule_class(rep)], w[rep]) for rep in rule_representatives())


def entropy_bits(w: np.ndarray) -> float:
    w = w[w > 0]
    return float(-(w * np.log2(w)).sum())


def main():
    for name in PRIORS:
        w = PRIORS[name]()
        top = np.argsort(-w)[:8]
        print(f"{name}: entropy {entropy_bits(w):.2f} bits; class-invariant {check_class_invariance(name)}; "
              f"top rules " + ", ".join(f"{t} ({w[t]:.3f})" for t in top))
    wb = m_b_weights()
    print(f"M_B weight on rules 0, 255, 204, 170, 240: {wb[[0, 255, 204, 170, 240]].sum():.3f}")
    print("M_A totalistic rules:", [r for r in range(256) if is_totalistic(r)])
    print("M_A r=1 program lengths:", m_a_lengths(1))
    a, st = m_a_weights(), m_a_star_weights()
    ch = sorted(((abs(st[r] / a[r] - 1), r) for r in range(256)), reverse=True)[:6]
    print("M_A* vs M_A, largest relative changes:", ", ".join(f"rule {r}: {st[r] / a[r] - 1:+.1%}" for _, r in ch))


if __name__ == "__main__":
    main()
