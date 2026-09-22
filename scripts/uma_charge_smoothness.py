"""
Is UMA's routing response to charge SMOOTH or ERRATIC?

Trained embeddings should change gradually across adjacent charges. Untrained
ones can jump arbitrarily. This distinguishes "UMA learned something about
charge state" from "UMA does something arbitrary with charges it never saw."

PARITY CONSTRAINT: neutral molecules have an even electron count, so charge
parity fixes multiplicity parity. Even charges -> mult 1, 3, 5...; odd charges
-> mult 2, 4, 6.... You therefore CANNOT sweep -3..+3 at a single fixed
multiplicity. We run two interleaved series instead, each at constant
multiplicity and constant step size of 2:

    even series: -4 -2 0 +2 +4   all at multiplicity 1
    odd  series: -3 -1 +1 +3     all at multiplicity 2

Requires uma_routing_extract_v2.py in the same directory.

Usage:
    python uma_charge_smoothness.py --asedb ./neutral_val --n-comps 300
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


EVEN_CHARGES = [-4, -2, 0, 2, 4]   # multiplicity 1
ODD_CHARGES = [-3, -1, 1, 3]       # multiplicity 2


def sweep_series(backbone, comps, charges, mult, device="cpu"):
    """Evaluate the router over one parity series at constant multiplicity."""
    systems = [
        {"numbers": n, "charge": int(c), "spin": int(mult), "task": "omol"}
        for c in charges
        for n in comps
    ]
    alpha = routing_coefficients(backbone, systems, device=device)
    alpha = strip_floor(alpha, coefficient_floor(backbone))
    # reshape: charge-major -> [charge, comp, expert]
    return alpha.reshape(len(charges), len(comps), alpha.shape[1])


def analyse(A, charges, label, ref_idx):
    """Distance from reference charge, and successive-step distances."""
    print(f"\n=== {label} ===")
    ref = A[ref_idx]
    print("  distance from reference charge:")
    dists = []
    for i, c in enumerate(charges):
        d = float(np.linalg.norm(A[i] - ref, axis=-1).mean())
        dists.append(d)
        print(f"    charge {c:+d}: {d:.4f}")

    print("  successive-step distance (should be gradual and comparable):")
    steps = []
    for i in range(len(charges) - 1):
        s = float(np.linalg.norm(A[i + 1] - A[i], axis=-1).mean())
        steps.append(s)
        print(f"    {charges[i]:+d} -> {charges[i+1]:+d}: {s:.4f}")

    steps = np.array(steps)
    ratio = steps.max() / max(steps.min(), 1e-12)
    print(f"  step size max/min ratio: {ratio:.2f}")
    if ratio < 2:
        print("    >> steps are comparable: consistent with a smooth learned response")
    elif ratio < 5:
        print("    >> moderately uneven steps")
    else:
        print("    >> highly uneven steps: consistent with erratic/untrained embeddings")
    return np.array(dists), steps


def plot(even, odd, ec, oc, top_experts, fname="charge_smoothness.png"):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].plot(ec, np.linalg.norm(even - even[ec.index(0)], axis=-1).mean(1),
                "o-", label="even charges (mult 1)")
    axs[0].plot(oc, np.linalg.norm(odd - odd[0], axis=-1).mean(1),
                "s-", label="odd charges (mult 2)")
    axs[0].set_xlabel("charge")
    axs[0].set_ylabel("mean L2 from series reference")
    axs[0].set_title("Distance from reference")
    axs[0].legend(fontsize=8)

    for series, chg, marker, lab in [(even, ec, "o", "mult 1"), (odd, oc, "s", "mult 2")]:
        steps = [float(np.linalg.norm(series[i + 1] - series[i], axis=-1).mean())
                 for i in range(len(chg) - 1)]
        mids = [(chg[i] + chg[i + 1]) / 2 for i in range(len(chg) - 1)]
        axs[1].plot(mids, steps, marker + "-", label=lab)
    axs[1].set_xlabel("charge (midpoint of step)")
    axs[1].set_ylabel("mean L2 per step of 2")
    axs[1].set_title("Successive-step size")
    axs[1].legend(fontsize=8)

    for k in top_experts:
        axs[2].plot(ec, even[:, :, k].mean(1), "o-", label=f"expert {k}")
    axs[2].set_xlabel("charge")
    axs[2].set_ylabel("mean alpha (mult 1)")
    axs[2].set_title("Per-expert response")
    axs[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"\nwrote {fname}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asedb", required=True)
    p.add_argument("--model", default="uma-s-1p1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-comps", type=int, default=300)
    p.add_argument("--n-read", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    systems = systems_from_asedb(args.asedb, task="omol", n=args.n_read, seed=args.seed)
    unique, _ = dedupe(systems)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(unique), size=min(args.n_comps, len(unique)), replace=False)
    comps = [unique[i]["numbers"] for i in idx]
    print(f"sweeping {len(comps)} compositions")

    backbone = load_backbone(args.model, device=args.device)
    even = sweep_series(backbone, comps, EVEN_CHARGES, mult=1, device=args.device)
    odd = sweep_series(backbone, comps, ODD_CHARGES, mult=2, device=args.device)

    analyse(even, EVEN_CHARGES, "even charges, multiplicity 1", EVEN_CHARGES.index(0))
    analyse(odd, ODD_CHARGES, "odd charges, multiplicity 2", 0)

    top = np.argsort(even.reshape(-1, even.shape[-1]).mean(0))[::-1][:4]
    print("\nper-expert curves plotted for experts:", top.tolist())
    plot(even, odd, EVEN_CHARGES, ODD_CHARGES, top)

    np.savez("charge_smoothness.npz", even=even, odd=odd,
             even_charges=EVEN_CHARGES, odd_charges=ODD_CHARGES)
    print("wrote charge_smoothness.npz")
    print("\nNOTE: only charge 0 / multiplicity 1 is in-distribution for neutral_val.")
    print("Everything else is extrapolation -- smoothness is evidence about the")
    print("embeddings, not proof that the response is chemically correct.")


if __name__ == "__main__":
    main()
