"""
Build a COMPLETE 4-way routing tensor [composition x charge x spin x expert]
by sweeping charge/spin over real OMol25 compositions.

Why this works: UMA's router is a pure function of (composition, charge, spin, task).
No positions, no structures. So we don't have to FIND charged systems in a dataset --
we evaluate the router on a grid we choose. Every cell exists by construction:
no structural missingness, no collapse ratio, no uneven cell counts.

Only valid for task='omol'. The fairchem docs state UMA has not seen varying
charge/spin for omat, omc, oc20, or odac -- sweeping those probes untrained
embeddings and returns artifacts, not chemistry.

Requires uma_routing_extract_v2.py in the same directory.

Usage:
    python uma_charge_spin_sweep.py --asedb ./neutral_val --n-comps 500
    python uma_charge_spin_sweep.py --asedb ./neutral_val --n-comps 500 --fit-rank 4
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    from uma_routing_extract_v2 import (
        coefficient_floor,
        dedupe,
        load_backbone,
        routing_coefficients,
        strip_floor,
        systems_from_asedb,
    )
except ImportError:
    sys.exit("Need uma_routing_extract_v2.py in the same directory as this script.")


# -----------------------------------------------------------------------------
# Spin bookkeeping
# -----------------------------------------------------------------------------
def multiplicity(numbers: np.ndarray, charge: int, k: int) -> int:
    """
    The k-th allowed spin multiplicity for this composition at this charge.

    Electron count parity fixes which multiplicities are physical: an even
    electron count allows 1, 3, 5, ...; an odd count allows 2, 4, 6, ....
    Changing charge flips the parity, so 'spin = 3' is meaningful for some
    charges and nonsense for others.

    Indexing by k (0 = lowest allowed, 1 = next, ...) instead of by absolute
    multiplicity keeps every cell of the tensor physical AND keeps it complete.
    Using absolute spin instead would give a checkerboard of impossible cells --
    systematic missingness, which is the worst kind for a CP fit.
    """
    n_elec = int(numbers.sum()) - charge
    lowest = 1 if n_elec % 2 == 0 else 2
    return lowest + 2 * k


def build_sweep(compositions, charges, n_spin_levels, task="omol"):
    """Cartesian product of compositions x charges x spin levels."""
    systems = []
    for numbers in compositions:
        for c in charges:
            for k in range(n_spin_levels):
                systems.append(
                    {
                        "numbers": numbers,
                        "charge": int(c),
                        "spin": int(multiplicity(numbers, c, k)),
                        "task": task,
                    }
                )
    return systems


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
def variance_split(X: np.ndarray) -> dict:
    """
    THE decisive check: does charge/spin actually move the router?

    X has shape [n_comp, n_charge, n_spin, n_expert]. Compare variance ACROSS
    compositions against variance WITHIN a composition (across the charge/spin
    grid). If the within term is negligible, charge and spin are not viable
    tensor modes and this whole approach fails -- better to learn that in one
    number than after fitting a decomposition.
    """
    n_comp = X.shape[0]
    flat = X.reshape(n_comp, -1, X.shape[-1])       # [comp, cell, expert]
    per_comp_mean = flat.mean(axis=1)               # [comp, expert]

    between = per_comp_mean.var(axis=0).mean()
    within = (flat - per_comp_mean[:, None, :]).var(axis=(0, 1)).mean()
    return {
        "between_composition": float(between),
        "within_composition": float(within),
        "within_fraction": float(within / (between + within)),
    }


def report(X, comps, charges, n_spin_levels, expert_names=None):
    print(f"\ntensor shape {X.shape}  [composition x charge x spin x expert]")
    print(f"  compositions : {X.shape[0]}")
    print(f"  charges      : {list(charges)}")
    print(f"  spin levels  : {n_spin_levels} (k-th allowed multiplicity)")
    print(f"  experts      : {X.shape[3]}")
    print(f"  fill         : 100% by construction, {X.size} cells")

    v = variance_split(X)
    print("\nvariance decomposition of alpha:")
    print(f"  between compositions : {v['between_composition']:.3e}")
    print(f"  within  (charge/spin): {v['within_composition']:.3e}")
    print(f"  within fraction      : {v['within_fraction']:.1%}")
    if v["within_fraction"] < 0.02:
        print("  >> charge/spin barely move the router. Mode 2/3 are near-degenerate;")
        print("     a CP will spend all its rank on composition. Reconsider before fitting.")
    elif v["within_fraction"] < 0.15:
        print("  >> modest but real charge/spin signal. Usable, expect composition to dominate.")
    else:
        print("  >> substantial charge/spin signal. Good -- modes 2/3 carry information.")

    # which experts are live at all
    m = X.reshape(-1, X.shape[-1]).mean(0)
    order = np.argsort(m)[::-1]
    live = int((m > 2 * m.max() / 100).sum())
    print(f"\nlive experts: {live} of {X.shape[3]}")
    print("  top 8:", [(int(k), round(float(m[k]), 3)) for k in order[:8]])


# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asedb", required=True, help="OMol25 aselmdb dir")
    p.add_argument("--model", default="uma-s-1p1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-comps", type=int, default=500, help="unique compositions to sweep")
    p.add_argument("--n-read", type=int, default=20000, help="structures to read before dedup")
    p.add_argument("--charges", type=int, nargs="+", default=[-2, -1, 0, 1, 2])
    p.add_argument("--spin-levels", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fit-rank", type=int, default=None, help="optional NNCP rank to fit")
    p.add_argument("--out", default="routing_tensor_4way.npz")
    args = p.parse_args()

    # 1. real compositions from the dataset
    systems = systems_from_asedb(args.asedb, task="omol", n=args.n_read, seed=args.seed)
    unique, _ = dedupe(systems)
    print(f"{len(unique)} unique (composition, charge, spin) tuples")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(unique), size=min(args.n_comps, len(unique)), replace=False)
    comps = [unique[i]["numbers"] for i in idx]
    print(f"sweeping {len(comps)} compositions")

    # 2. sweep the router over the grid
    sweep = build_sweep(comps, args.charges, args.spin_levels)
    print(f"{len(sweep)} router evaluations")

    backbone = load_backbone(args.model, device=args.device)
    alpha = routing_coefficients(backbone, sweep, device=args.device)
    alpha = strip_floor(alpha, coefficient_floor(backbone))

    # 3. reshape -- build_sweep iterates comp, then charge, then spin
    X = alpha.reshape(len(comps), len(args.charges), args.spin_levels, alpha.shape[1])
    report(X, comps, args.charges, args.spin_levels)

    np.savez(
        args.out,
        tensor=X,
        charges=np.array(args.charges),
        spin_levels=args.spin_levels,
        compositions=np.array([np.bincount(c, minlength=100) for c in comps]),
    )
    print(f"\nwrote {args.out}")

    # 4. optional NNCP
    if args.fit_rank:
        try:
            import tensorly as tl
            from tensorly.decomposition import non_negative_parafac
        except ImportError:
            print("pip install tensorly to fit the decomposition")
            return
        print(f"\nfitting non-negative CP, rank {args.fit_rank} ...")
        cp = non_negative_parafac(tl.tensor(X), rank=args.fit_rank, init="random",
                                  n_iter_max=500, tol=1e-8, random_state=args.seed)
        recon = tl.cp_to_tensor(cp)
        err = float(np.linalg.norm(X - recon) / np.linalg.norm(X))
        print(f"  relative reconstruction error: {err:.4f}")
        print("  factor shapes:", [f.shape for f in cp.factors])
        print("  NOTE: rank chosen by hand. Before trusting it, run CORCONDIA and")
        print("        split-half, and check whether component 1 is just the mean.")


if __name__ == "__main__":
    main()
