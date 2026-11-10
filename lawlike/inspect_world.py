"""Look at one world: its spacetime, its causal-complexity field, and where each filter's
admitted observers sit (coloured by whether their experience is lawful, noisy or trivial).

    python -m lawlike.inspect_world --rule 110 --seed 1 --kind embed --filter individuality
    python -m lawlike.inspect_world --rule 54 --kind uniform --filter integration_excl
    python -m lawlike.inspect_world --kind noise --filter causal_complexity

This is the "microscope" for the biological-experiment loop: when a table says a filter
loves rule 54, come here and see what it is actually picking.
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

from .config import get_preset
from .experiment1 import FILTERS, process_world
from .causal import CausalStates
from .worlds import World, realise, rep_of

KLASS_COLORS = {0: "#9a9993", 1: "#2a78d6", 2: "#eb6834"}   # trivial, lawful, noisy
KLASS_NAMES = {0: "trivial", 1: "lawful", 2: "noisy"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="quick")
    ap.add_argument("--rule", type=int, default=110)
    ap.add_argument("--kind", default="embed", choices=["embed", "tile", "uniform", "noise"])
    ap.add_argument("--seed", default="1", help="seed bits for embed/tile, e.g. 1011")
    ap.add_argument("--filter", default="individuality", choices=FILTERS)
    ap.add_argument("--out", default="inspect.png")
    args = ap.parse_args(argv)

    cfg = get_preset(args.preset)
    arm = {"embed": "simplicity", "tile": "simplicity", "uniform": "uniform", "noise": "noise"}[args.kind]
    rule = -1 if arm == "noise" else rep_of(args.rule)
    w = World(f"inspect-{arm}-{rule}-{args.kind}-{args.seed}", arm, rule, 1, args.kind,
              args.seed if arm == "simplicity" else "", float("nan"), 1.0, 1234)
    with tempfile.TemporaryDirectory() as tmp:
        summ = process_world(w, cfg, tmp)
        d = dict(np.load(os.path.join(tmp, "observers", f"{w.world_id}.npz")))
    st, _, flip = realise(w, cfg)
    cs = CausalStates(cfg.hp, cfg.hf, cfg.causal_min_count, cfg.causal_eps, cfg.causal_max_states)
    cs.fit(st, cfg.T0, cfg.T0 + cfg.T)
    cpx = cs.complexity(cs.label(st))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reg = slice(cfg.T0, cfg.T0 + cfg.T)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor="#fcfcfb", sharey=True)
    axes[0].imshow(st[reg], cmap="gray_r", aspect="auto", interpolation="nearest")
    axes[0].set_title("spacetime (time runs down)", loc="left", fontsize=10)
    im = axes[1].imshow(cpx[reg], cmap="magma", aspect="auto", interpolation="nearest")
    axes[1].set_title(f"local statistical complexity ({cs.n_states} causal states)", loc="left", fontsize=10)
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="bits")
    axes[2].imshow(st[reg], cmap="gray_r", aspect="auto", interpolation="nearest", alpha=0.35)
    mask = d[f"adm_{args.filter}"] & (d["fam"] > 0)
    tt = np.arange(cfg.T)
    for i in np.where(mask)[0]:
        k = int(d["m_klass"][i])
        foot = d["w"][i] * (d["s"][i] if d["fam"][i] == 2 else 1)
        vp, vq = cfg.velocities[int(d["v"][i])]
        x = (d["x0"][i] + (vp * tt) // vq + foot / 2.0) % cfg.N
        jumps = np.where(np.abs(np.diff(x)) > cfg.N / 2)[0] + 1
        for seg_t, seg_x in zip(np.split(tt, jumps), np.split(x, jumps)):
            axes[2].plot(seg_x, seg_t, color=KLASS_COLORS[k], linewidth=1.2, alpha=0.9)
    for k, c in KLASS_COLORS.items():
        axes[2].plot([], [], color=c, label=KLASS_NAMES[k])
    axes[2].axhline(flip[0] - cfg.T0, color="#e34948", linewidth=0.8, linestyle="--", label="perturbation")
    axes[2].legend(frameon=False, fontsize=8, loc="lower right")
    fs = summ["filters"][args.filter]
    axes[2].set_title(f"admitted by '{args.filter}': {int(mask.sum())} world-reading observers, "
                      f"lawful share {fs.get('lawful_nt', float('nan')):.2f}", loc="left", fontsize=10)
    for ax in axes:
        ax.set_xlabel("cell")
    axes[0].set_ylabel("time step")
    label = "noise world" if arm == "noise" else f"rule {rule}, {args.kind} {args.seed if arm == 'simplicity' else ''}"
    fig.suptitle(label, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    for f in FILTERS:
        s = summ["filters"][f]
        print(f"  {f:18s} admitted={s['n_adm']:4d}  lawful_nt={s.get('lawful_nt', float('nan')):.2f}  "
              f"trivial={s.get('frac_trivial', float('nan')):.2f}")


if __name__ == "__main__":
    main()
