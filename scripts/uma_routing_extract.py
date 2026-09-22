"""
Step 0: extract UMA MoLE routing coefficients and reproduce Figure 5 / Figure 6 style plots.

Key idea (verified against fairchem source, escn_moe.py):
    The router sees ONLY composition, charge, spin, task -- never atomic positions.
    So we can call csd_embedding() + set_MOLE_coefficients() directly and skip the
    energy/force forward pass entirely. Extracting alpha for 50k systems is cheap.

Figure 5 recipe (UMA paper, Appendix F.1):
    "mean expert coefficient for each element-expert pair across all systems where
     the pair appears" -- i.e. PRESENCE-based, not composition-weighted. Displayed
     on a log scale, but aggregate on RAW alpha.

Requires:  pip install fairchem-core matplotlib numpy
           huggingface-cli login   (facebook/UMA is a gated repo)

Usage:
    python uma_routing_extract.py --demo                    # synthetic sanity check
    python uma_routing_extract.py --xyz-dir ./omol_subset   # real structures
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

# ----------------------------------------------------------------------------- 
# Periodic table layout (Z -> (row, col), 1-indexed), main block + lanthanides
# -----------------------------------------------------------------------------
_PT_ROWS = [
    "H                                                  He",
    "Li Be                               B  C  N  O  F  Ne",
    "Na Mg                               Al Si P  S  Cl Ar",
    "K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr",
    "Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe",
    "Cs Ba La Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn",
    "Fr Ra Ac Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og",
]
_LANTH = "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu"
_ACT = "Th Pa U  Np Pu Am Cm Bk Cf Es Fm Md No Lr"


def periodic_table_layout() -> dict[str, tuple[int, int]]:
    """Map element symbol -> (row, col) grid position for plotting."""
    layout = {}
    for r, row in enumerate(_PT_ROWS):
        # fixed-width 3-char columns
        for c in range(18):
            sym = row[c * 3 : c * 3 + 3].strip()
            if sym:
                layout[sym] = (r, c)
    for c, sym in enumerate(_LANTH.split()):
        layout[sym] = (7, c + 3)
    for c, sym in enumerate(_ACT.split()):
        layout[sym] = (8, c + 3)
    return layout


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_backbone(model_name: str = "uma-s-1p1", device: str = "cpu"):
    """Return the MoLE backbone module, which owns the router."""
    from fairchem.core import pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
    backbone = predictor.model.module.backbone
    backbone.eval()

    n_exp = backbone.num_experts
    if n_exp != 32:
        print(
            f"WARNING: {model_name} has {n_exp} experts, not 32. "
            "Figure 5 used a 32-expert UMA-S; uma-s-1p2* has 64 and will not match."
        )
    if getattr(backbone, "model_id", None) == "UMA-S-1.2":
        print(
            "WARNING: UMA-S-1.2 includes a zero-init self value in the composition "
            "mean (comp_break_extensivity), breaking extensivity. Prefer 1.1."
        )
    print(f"Loaded {model_name}: {n_exp} experts")
    return backbone


# -----------------------------------------------------------------------------
# Routing coefficient extraction -- the cheap path
# -----------------------------------------------------------------------------
@torch.no_grad()
def routing_coefficients(
    backbone,
    systems: list[dict],
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """
    systems: list of dicts with keys
        'numbers' (np.ndarray of Z), 'charge' (int), 'spin' (int), 'task' (str)
    Returns alpha of shape [n_systems, n_experts].

    NOTE: rows do NOT sum to 1. UMA's _softmax is `softmax(x) + 0.005`, so rows
    sum to 1 + 0.005 * n_experts (= 1.16 for 32 experts). See coefficient_floor().
    """
    out = []
    for start in range(0, len(systems), batch_size):
        chunk = systems[start : start + batch_size]

        atomic_numbers = torch.tensor(
            np.concatenate([s["numbers"] for s in chunk]), dtype=torch.long, device=device
        )
        batch_idx = torch.tensor(
            np.concatenate([np.full(len(s["numbers"]), i) for i, s in enumerate(chunk)]),
            dtype=torch.long,
            device=device,
        )
        charge = torch.tensor([s["charge"] for s in chunk], dtype=torch.long, device=device)
        spin = torch.tensor([s["spin"] for s in chunk], dtype=torch.long, device=device)
        dataset = [s["task"] for s in chunk]  # list of strings, e.g. "omol"

        csd = backbone.csd_embedding(charge=charge, spin=spin, dataset=dataset)
        backbone.set_MOLE_coefficients(
            atomic_numbers_full=atomic_numbers,
            batch_full=batch_idx,
            csd_mixed_emb=csd,
        )
        alpha = backbone.global_mole_tensors.expert_mixing_coefficients
        out.append(alpha.detach().cpu().numpy())

    alpha = np.concatenate(out, axis=0)

    floor = coefficient_floor(backbone)
    expected = 1.0 + floor * alpha.shape[1]
    obs = alpha.sum(axis=1)
    if not np.allclose(obs, expected, atol=1e-3):
        raise AssertionError(
            f"row sums {obs.min():.4f}..{obs.max():.4f}, expected {expected:.4f}. "
            "Router normalization is not what this script assumes."
        )
    print(f"row sums = {expected:.4f} (softmax + floor {floor}); min alpha = {alpha.min():.4f}")
    return alpha


def coefficient_floor(backbone) -> float:
    """
    UMA's _softmax is `torch.softmax(x, dim=1) + 0.005` -- a constant additive
    floor on EVERY expert, not a plain softmax. Consequences:

      * rows sum to 1 + 0.005 * n_experts, not 1
      * no coefficient is ever truly zero; "expert unused" reads as ~0.005
      * that uniform background is a rank-1 all-ones structure in the tensor.
        An NNCP will spend a component reproducing it unless you strip it first.
    """
    fn = getattr(backbone, "mole_expert_coefficient_norm", None)
    name = getattr(fn, "__name__", "")
    if name == "_softmax":
        return 0.005
    if name == "_pnorm":
        return 0.0
    print(f"WARNING: unrecognized router norm {name!r}; assuming no floor")
    return 0.0


def strip_floor(alpha: np.ndarray, floor: float, renormalize: bool = True) -> np.ndarray:
    """Remove the constant background so the tensor carries only routing signal."""
    if floor == 0.0:
        return alpha
    out = np.clip(alpha - floor, 0.0, None)
    if renormalize:
        out = out / out.sum(axis=1, keepdims=True)
    return out


def dedupe(systems: list[dict]) -> tuple[list[dict], np.ndarray]:
    """
    Routing is position-independent, so systems sharing
    (reduced composition, charge, spin, task) have IDENTICAL alpha.
    Returns unique systems + multiplicity counts. Check the collapse ratio:
    if it is large, your effective sample size is far below your structure count.
    """
    seen: dict[tuple, int] = {}
    unique, counts = [], []
    for s in systems:
        key = (
            tuple(sorted(np.unique(s["numbers"], return_counts=True)[0].tolist())),
            tuple(np.unique(s["numbers"], return_counts=True)[1].tolist()),
            s["charge"],
            s["spin"],
            s["task"],
        )
        if key in seen:
            counts[seen[key]] += 1
        else:
            seen[key] = len(unique)
            unique.append(s)
            counts.append(1)
    return unique, np.array(counts)


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------
def element_expert_matrix(
    systems: list[dict], alpha: np.ndarray, weights: np.ndarray | None = None
):
    """
    Figure 5 aggregation: for element e and expert k, mean of alpha_k over all
    systems CONTAINING e. Returns (matrix [n_Z+1, n_experts], counts [n_Z+1]).

    Note this is NOT the composition-weighted average of proposal eq. (4).
    Both are defensible; they are different objects. Keep them distinct.
    """
    n_exp = alpha.shape[1]
    if weights is None:
        weights = np.ones(len(systems))

    acc = defaultdict(lambda: np.zeros(n_exp))
    cnt = defaultdict(float)
    for s, a, w in zip(systems, alpha, weights):
        for z in np.unique(s["numbers"]):
            acc[int(z)] += a * w
            cnt[int(z)] += w

    max_z = max(acc) if acc else 0
    mat = np.full((max_z + 1, n_exp), np.nan)
    counts = np.zeros(max_z + 1)
    for z in acc:
        mat[z] = acc[z] / cnt[z]
        counts[z] = cnt[z]
    return mat, counts


def task_expert_stats(systems: list[dict], alpha: np.ndarray):
    """Figure 6 aggregation: per-task mean and variance of alpha."""
    tasks = sorted({s["task"] for s in systems})
    means = np.zeros((len(tasks), alpha.shape[1]))
    vars_ = np.zeros_like(means)
    for i, t in enumerate(tasks):
        mask = np.array([s["task"] == t for s in systems])
        means[i] = alpha[mask].mean(axis=0)
        vars_[i] = alpha[mask].var(axis=0)
    return tasks, means, vars_


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_expert_periodic_tables(mat, counts, experts=None, min_count=1, fname="fig5.png"):
    """Figure 5 style: one periodic table heatmap per expert, log colour scale."""
    import matplotlib.pyplot as plt
    from ase.data import chemical_symbols

    layout = periodic_table_layout()
    experts = list(range(mat.shape[1])) if experts is None else experts
    ncols = min(4, len(experts))
    nrows = int(np.ceil(len(experts) / ncols))

    with np.errstate(divide="ignore", invalid="ignore"):
        logmat = np.log10(mat)
    vmin, vmax = np.nanpercentile(logmat, [2, 98])

    fig, axs = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.2 * nrows), squeeze=False)
    for ax, k in zip(axs.flat, experts):
        grid = np.full((9, 18), np.nan)
        for z in range(1, len(mat)):
            if counts[z] < min_count or np.isnan(mat[z, k]):
                continue
            sym = chemical_symbols[z]
            if sym in layout:
                r, c = layout[sym]
                grid[r, c] = logmat[z, k]
        ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"expert {k}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axs.flat[len(experts) :]:
        ax.axis("off")

    fig.suptitle("Log mean expert coefficient across element-expert pairs", fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"wrote {fname}")


def plot_task_expert(tasks, means, vars_, fname="fig6.png"):
    """Figure 6 style: task x expert mean and variance heatmaps."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(12, 3 + 0.4 * len(tasks)))
    for ax, data, label in zip(axs, [means, vars_], ["mean", "variance"]):
        im = ax.imshow(data, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(tasks)
        ax.set_xlabel("expert")
        ax.set_title(f"expert coefficient {label} across tasks", fontsize=10)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"wrote {fname}")


# -----------------------------------------------------------------------------
# Data sources
# -----------------------------------------------------------------------------
def demo_systems(n: int = 2000, seed: int = 0) -> list[dict]:
    """Synthetic systems to validate the extraction path without dataset access."""
    rng = np.random.default_rng(seed)
    pools = {
        "omol": [1, 6, 7, 8, 9, 15, 16, 17, 26, 29, 30],
        "omat": [3, 8, 11, 12, 13, 14, 22, 26, 28, 29, 30, 42],
        "oc20": [1, 6, 7, 8, 26, 27, 28, 29, 44, 45, 46, 78],
        "omc": [1, 6, 7, 8, 16, 17],
        "odac": [1, 6, 7, 8, 12, 13, 26, 30, 40],
    }
    systems = []
    for _ in range(n):
        task = rng.choice(list(pools))
        pool = pools[task]
        k = rng.integers(2, min(6, len(pool)) + 1)
        elems = rng.choice(pool, size=k, replace=False)
        numbers = np.repeat(elems, rng.integers(1, 12, size=k))
        charge = int(rng.choice([-1, 0, 0, 0, 1])) if task == "omol" else 0
        spin = int(rng.choice([1, 1, 1, 2, 3])) if task == "omol" else 0
        systems.append(
            {"numbers": numbers, "charge": charge, "spin": spin, "task": task}
        )
    return systems


def _default_spin(task: str) -> int:
    """OMol defaults to spin multiplicity 1; every other task expects 0."""
    return 1 if task == "omol" else 0


def systems_from_asedb(
    src: str, task: str = "omol", n: int | None = None, seed: int = 0
) -> list[dict]:
    """
    Load structures from OMol25 .aselmdb files (or any ASE DB).

    src may be a single .aselmdb, a folder of them, or a glob string.
    Charge and spin live in atoms.info because ASE has no native slot for them.
    """
    from fairchem.core.datasets import AseDBDataset

    ds = AseDBDataset({"src": src})
    total = len(ds)
    print(f"{total} structures in {src}")

    if n is not None and n < total:
        rng = np.random.default_rng(seed)
        idxs = rng.choice(total, size=n, replace=False)
    else:
        idxs = np.arange(total)

    systems, missing = [], 0
    for i in idxs:
        atoms = ds.get_atoms(int(i))
        if "charge" not in atoms.info or "spin" not in atoms.info:
            missing += 1
        systems.append(
            {
                "numbers": atoms.get_atomic_numbers(),
                "charge": int(atoms.info.get("charge", 0)),
                "spin": int(atoms.info.get("spin", _default_spin(task))),
                "task": task,
                "sid": atoms.info.get("sid", int(i)),
            }
        )
    if missing:
        print(
            f"WARNING: {missing}/{len(systems)} structures lacked charge/spin in "
            f"atoms.info; defaulted to charge=0, spin={_default_spin(task)}. "
            "For omol these drive the router directly -- verify before trusting results."
        )
    return systems


def systems_from_dir(path: str, task: str = "omol") -> list[dict]:
    """Fallback loader for plain xyz/cif/traj files."""
    from pathlib import Path

    from ase.io import read

    systems = []
    for f in sorted(Path(path).iterdir()):
        if f.suffix.lower() not in {".xyz", ".extxyz", ".cif", ".traj", ".json"}:
            continue
        for atoms in read(f, index=":"):
            systems.append(
                {
                    "numbers": atoms.get_atomic_numbers(),
                    "charge": int(atoms.info.get("charge", 0)),
                    "spin": int(atoms.info.get("spin", _default_spin(task))),
                    "task": task,
                }
            )
    return systems


# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="uma-s-1p1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--demo", action="store_true", help="use synthetic systems")
    p.add_argument(
        "--asedb",
        default=None,
        help="OMol25 .aselmdb file, folder of them, or glob string",
    )
    p.add_argument("--n-samples", type=int, default=20000, help="structures to sample")
    p.add_argument("--xyz-dir", default=None, help="directory of ASE-readable structures")
    p.add_argument("--task", default="omol", help="task label for --xyz-dir structures")
    p.add_argument("--min-count", type=int, default=5, help="hide elements seen < N times")
    p.add_argument("--experts", type=int, default=8, help="how many experts to plot")
    p.add_argument(
        "--strip-floor",
        action="store_true",
        help="subtract the constant 0.005 softmax floor and renormalize",
    )
    args = p.parse_args()

    if args.asedb:
        systems = systems_from_asedb(
            args.asedb, task=args.task, n=args.n_samples
        )
    elif args.xyz_dir:
        systems = systems_from_dir(args.xyz_dir, task=args.task)
    else:
        systems = demo_systems()
        if not args.demo:
            print("No --asedb given; falling back to --demo synthetic systems.\n")

    print(f"{len(systems)} structures")
    unique, counts = dedupe(systems)
    ratio = len(systems) / max(len(unique), 1)
    print(
        f"{len(unique)} unique (composition, charge, spin, task) tuples "
        f"-- collapse ratio {ratio:.1f}x"
    )
    if ratio > 3:
        print(
            f"NOTE: effective sample size is {len(unique)}, not {len(systems)}. "
            "Routing ignores positions, so duplicate tuples give identical alpha."
        )

    backbone = load_backbone(args.model, device=args.device)
    alpha_raw = routing_coefficients(backbone, unique, device=args.device)
    print(f"alpha shape {alpha_raw.shape}")

    floor = coefficient_floor(backbone)
    alpha = strip_floor(alpha_raw, floor) if args.strip_floor else alpha_raw
    if args.strip_floor:
        frac = floor * alpha_raw.shape[1] / (1.0 + floor * alpha_raw.shape[1])
        print(f"stripped floor: it was {frac:.1%} of total coefficient mass")

    # weight unique tuples by multiplicity so aggregation matches the raw sample set
    mat, el_counts = element_expert_matrix(unique, alpha, weights=counts)
    n_seen = int((el_counts >= args.min_count).sum())
    print(f"{n_seen} elements with >= {args.min_count} systems")
    print("NOTE: element cell reliability is very non-uniform -- keep el_counts.")

    plot_expert_periodic_tables(
        mat, el_counts, experts=range(min(args.experts, alpha.shape[1])),
        min_count=args.min_count,
    )

    tasks, means, vars_ = task_expert_stats(unique, alpha)
    if len(tasks) > 1:
        plot_task_expert(tasks, means, vars_)
    else:
        print(f"only one task ({tasks[0]}); skipping Figure 6 plot")

    np.savez(
        "routing_coefficients.npz",
        alpha=alpha,
        alpha_raw=alpha_raw,
        floor=floor,
        multiplicity=counts,
        element_expert=mat,
        element_counts=el_counts,
        tasks=np.array(tasks),
    )
    print("wrote routing_coefficients.npz")


if __name__ == "__main__":
    main()
