"""
The "why this design works" figure.

Assembles the four pieces of evidence that justify the
[composition x charge x spin x expert] tensor:

  A  Expert disjointness across tasks   -> why task CANNOT be mode 2
  B  Per-expert charge response         -> charge is a real, smooth router input
  C  CP charge (x) spin outer products  -> what each component selects
  D  Expert loadings per component      -> which experts each component owns

Inputs (from earlier runs):
  omat24_results/routing_coefficients.npz
  omol_neutral_results/routing_coefficients.npz
  charge_smoothness.npz
  routing_tensor_4way_cp.npz

Usage:
    python design_evidence_figure.py \
        --omat omat24_results/routing_coefficients.npz \
        --omol omol_neutral_results/routing_coefficients.npz \
        --smooth charge_smoothness.npz \
        --cp routing_tensor_4way_cp.npz
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

BLUE, RED, GREY = "#1f77b4", "#d62728", "#888888"


def panel_disjointness(ax, omat_path, omol_path):
    """Mean alpha per expert for both tasks. The punchline: no shared experts."""
    a1 = np.load(omat_path, allow_pickle=True)["alpha"].mean(0)
    a2 = np.load(omol_path, allow_pickle=True)["alpha"].mean(0)
    k = np.arange(len(a1))
    ax.bar(k - 0.2, a1, width=0.4, color=BLUE, label="OMat24")
    ax.bar(k + 0.2, a2, width=0.4, color=RED, label="OMol25-neutral")

    top1, top2 = set(np.argsort(a1)[::-1][:8]), set(np.argsort(a2)[::-1][:8])
    shared = top1 & top2
    ax.set_xlabel("expert")
    ax.set_ylabel("mean $\\alpha$")
    ax.set_title(f"A. Expert usage by task — top-8 overlap: {len(shared)}/8", fontsize=10)
    ax.legend(fontsize=8)
    return shared


def panel_charge_response(ax, smooth_path, experts=(2, 23, 14)):
    """Per-expert response to charge at fixed multiplicity 1."""
    d = np.load(smooth_path)
    even, ec = d["even"], list(d["even_charges"])
    for k, style in zip(experts, ["o-", "s-", "^-"]):
        ax.plot(ec, even[:, :, k].mean(1), style, label=f"expert {k}", ms=5)
    ax.axvline(0, ls=":", c=GREY, lw=1)
    ax.set_xlabel("charge (multiplicity 1)")
    ax.set_ylabel("mean $\\alpha$")
    ax.set_title("B. Charge response — mirrored specialists", fontsize=10)
    ax.legend(fontsize=8)


def multiplicity_grid(charges, n_k):
    """Which (charge, k) cells share a multiplicity, for an even-electron system."""
    M = np.zeros((len(charges), n_k), dtype=int)
    for i, c in enumerate(charges):
        lo = 1 if (0 - c) % 2 == 0 else 2      # neutral molecule -> even n_elec
        for k in range(n_k):
            M[i, k] = lo + 2 * k
    return M


def panel_cp_outer(axs, cp_path, charges):
    """
    charge (x) spin outer product per component. Because CP is multiplicative,
    this product IS the component's selection pattern over conditions.
    """
    d = np.load(cp_path)
    fc, fs, fe = d["factor_charge"], d["factor_spin"], d["factor_expert"]
    R = fc.shape[1]
    mult = multiplicity_grid(charges, fs.shape[0])

    for r in range(min(R, len(axs))):
        outer = np.outer(fc[:, r], fs[:, r])
        im = axs[r].imshow(outer, cmap="viridis", aspect="auto")
        axs[r].set_xticks(range(fs.shape[0]))
        axs[r].set_xticklabels([f"k={k}" for k in range(fs.shape[0])], fontsize=7)
        axs[r].set_yticks(range(len(charges)))
        axs[r].set_yticklabels([f"{c:+d}" for c in charges], fontsize=7)
        # annotate multiplicity so the selection pattern is readable
        for i in range(len(charges)):
            for j in range(fs.shape[0]):
                axs[r].text(j, i, mult[i, j], ha="center", va="center",
                            fontsize=6, color="white")
        top = int(np.argmax(fe[:, r]))
        axs[r].set_title(f"comp {r} — e{top}", fontsize=9)
        if r == 0:
            axs[r].set_ylabel("charge", fontsize=8)
    return fe


def panel_expert_loadings(ax, fe):
    """Which experts each component owns."""
    R = fe.shape[1]
    width = 0.8 / R
    k = np.arange(fe.shape[0])
    for r in range(R):
        ax.bar(k + (r - (R - 1) / 2) * width, fe[:, r], width=width,
               label=f"comp {r}")
    ax.set_xlabel("expert")
    ax.set_ylabel("loading")
    ax.set_title("D. Expert loadings", fontsize=10)
    ax.legend(fontsize=7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--omat", required=True)
    p.add_argument("--omol", required=True)
    p.add_argument("--smooth", required=True)
    p.add_argument("--cp", required=True)
    p.add_argument("--charges", type=int, nargs="+", default=[-2, -1, 0, 1, 2])
    p.add_argument("--out", default="design_evidence.png")
    args = p.parse_args()

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1], hspace=0.38, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2:])
    shared = panel_disjointness(ax_a, args.omat, args.omol)
    panel_charge_response(ax_b, args.smooth)

    axs_c = [fig.add_subplot(gs[1, i]) for i in range(3)]
    fe = panel_cp_outer(axs_c, args.cp, args.charges)
    axs_c[1].set_title(axs_c[1].get_title() + "\n(cells labelled by multiplicity)",
                       fontsize=8)
    fig.text(0.13, 0.46, "C. CP charge $\\otimes$ spin selection per component",
             fontsize=10)

    ax_d = fig.add_subplot(gs[1, 3])
    panel_expert_loadings(ax_d, fe)

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"top-8 expert overlap between tasks: {sorted(shared)} ({len(shared)}/8)")


if __name__ == "__main__":
    main()
