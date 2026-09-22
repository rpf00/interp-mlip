"""
Compare expert-mass concentration and element-mode low-rank structure
across two routing_coefficients.npz files (e.g. OMat24 vs OMol25).

Panel 1: cumulative mass in top-k experts -- how concentrated is routing?
Panel 2: centered SVD scree of the element x expert matrix -- how much
         low-rank structure exists AFTER removing the shared mean profile?
Panel 3: element-row pairwise cosine distribution -- how distinguishable
         are elements by their routing signature? (1.0 = indistinguishable)

Usage:
    python compare_routing.py omat24_results/routing_coefficients.npz \
                              omol_neutral_results/routing_coefficients.npz \
                              --labels OMat24 OMol25-neutral
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


def load(path, min_count=5):
    d = np.load(path, allow_pickle=True)
    alpha = d["alpha"]
    me, ec = d["element_expert"], d["element_counts"]
    keep = np.where(ec >= min_count)[0]
    M = me[keep]
    M = M / M.sum(axis=1, keepdims=True)      # each row = routing profile
    return alpha, M, keep, ec[keep]


def cumulative_mass(alpha):
    m = alpha.mean(axis=0)
    m = np.sort(m)[::-1]
    return np.cumsum(m) / m.sum()


def centered_scree(M):
    """Explained variance after removing the shared mean profile."""
    _, S, _ = np.linalg.svd(M - M.mean(axis=0), full_matrices=False)
    return S**2 / (S**2).sum()


def pairwise_cosine(M):
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    C = Mn @ Mn.T
    iu = np.triu_indices(len(M), 1)
    return C[iu]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs=2, help="two routing_coefficients.npz files")
    p.add_argument("--labels", nargs=2, default=["dataset A", "dataset B"])
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--out", default="routing_comparison.png")
    args = p.parse_args()

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = ["#1f77b4", "#d62728"]

    for path, lab, col in zip(args.npz, args.labels, colors):
        alpha, M, keep, counts = load(path, args.min_count)
        n_exp = alpha.shape[1]

        # --- panel 1: cumulative expert mass
        cum = cumulative_mass(alpha)
        axs[0].plot(np.arange(1, n_exp + 1), cum, "o-", color=col, ms=3, label=lab)
        k80 = int(np.searchsorted(cum, 0.8) + 1)

        # --- panel 2: centered scree
        var = centered_scree(M)
        n = min(10, len(var))
        axs[1].plot(np.arange(1, n + 1), np.cumsum(var[:n]), "o-", color=col,
                    ms=4, label=lab)

        # --- panel 3: cosine distribution
        cos = pairwise_cosine(M)
        axs[2].hist(cos, bins=40, alpha=0.55, color=col,
                    label=f"{lab} (mean {cos.mean():.2f})", density=True)

        print(f"{lab}:")
        print(f"  elements (count >= {args.min_count}) : {len(keep)}")
        print(f"  experts for 80% of mass          : {k80} of {n_exp}")
        print(f"  top-5 mass                       : {cum[4]:.1%}")
        print(f"  centered SVD, 3 comps            : {np.cumsum(var)[2]:.1%}")
        print(f"  mean pairwise cosine             : {cos.mean():.4f}")
        print()

    axs[0].axhline(0.8, ls=":", c="grey", lw=1)
    axs[0].set_xlabel("k (experts, sorted by mean mass)")
    axs[0].set_ylabel("cumulative fraction of mass")
    axs[0].set_title("Expert-mass concentration")
    axs[0].legend(fontsize=8)

    axs[1].set_xlabel("number of components")
    axs[1].set_ylabel("cumulative explained variance")
    axs[1].set_title("Centered SVD of element x expert")
    axs[1].legend(fontsize=8)

    axs[2].set_xlabel("pairwise cosine between element rows")
    axs[2].set_ylabel("density")
    axs[2].set_title("Element distinguishability (1.0 = identical)")
    axs[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
