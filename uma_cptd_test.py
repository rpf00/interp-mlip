import numpy as np, torch, tensorly as tl
from math import gcd
from functools import reduce
from fairchem.core import pretrained_mlip
from fairchem.core.datasets import AseDBDataset
from tensorly.decomposition import non_negative_parafac


rank_list = [3]

#rank_list = [1,2,3]
#rank_list = range(1,21)

rel_error_list = []

for R in rank_list:

    print(f"\n###### CP Tensor Decomposiiton with rank {R}\n")

    charges, n_spin, n_comps, rank = [-2, -1, 0, 1, 2], 4, 500, R

    # 1. compositions: one molecule per unique element ratio (C2H4 == C3H6 == CH2)
    ds = AseDBDataset({"src": "data/neutral_val"})

    def key(z):
        c = np.bincount(z, minlength=100)
        return tuple(c // reduce(gcd, c[c > 0]))

    seen = {}
    for i in range(len(ds)):
        z = ds.get_atoms(i).get_atomic_numbers()
        seen.setdefault(key(z), (i, z))
    unique = list(seen.values())
    pick = np.random.default_rng(0).choice(len(unique), min(n_comps, len(unique)), replace=False)
    ids, mols = zip(*[unique[p] for p in pick])      # ids[i] -> ds.get_atoms(ids[i]) gives the structure
    print(f"{len(ds)} structures, {len(unique)} unique compositions, using {len(mols)}")

    # 2. grid: every molecule x charge x spin level (k-th allowed multiplicity)
    def mult(z, q, k):
        return (1 if (z.sum() - q) % 2 == 0 else 2) + 2 * k

    grid = [(z, q, mult(z, q, k)) for z in mols for q in charges for k in range(n_spin)]

    # 3. router only: composition + charge + spin -> 32 expert weights
    backbone = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu").model.module.backbone.eval()
    with torch.no_grad():
        Z = torch.tensor(np.concatenate([z for z, _, _ in grid]))
        batch = torch.tensor(np.concatenate([[i] * len(z) for i, (z, _, _) in enumerate(grid)]))
        csd = backbone.csd_embedding(charge=torch.tensor([q for _, q, _ in grid]),
                                    spin=torch.tensor([s for _, _, s in grid]),
                                    dataset=["omol"] * len(grid))
        backbone.set_MOLE_coefficients(atomic_numbers_full=Z, batch_full=batch, csd_mixed_emb=csd)
        alpha = backbone.global_mole_tensors.expert_mixing_coefficients.numpy()

    # 4. tensor: strip UMA's constant 0.005 floor, reshape in grid order
    alpha = np.clip(alpha - 0.005, 0, None)
    X = alpha.reshape(len(mols), len(charges), n_spin, -1)   # [molecule x charge x spin x expert]
    print("tensor", X.shape)

    # 5. non-negative CP
    cp = non_negative_parafac(tl.tensor(X), rank=R, init="random", n_iter_max=500, random_state=0)
    
    rel_error = np.linalg.norm(X - tl.cp_to_tensor(cp)) / np.linalg.norm(X)
    print("relative error:", rel_error)

    M, Q, S, E = [np.asarray(f) for f in cp.factors]
    for r in range(R):
        print(f"\ncomponent {r}")
        print("  charge   ", dict(zip(charges, Q[:, r].round(2))))
        print("  spin k   ", S[:, r].round(2))
        print("  experts  ", np.argsort(E[:, r])[::-1][:5])
        print("  molecules", [ds.get_atoms(ids[i]).get_chemical_formula()
                            for i in np.argsort(M[:, r])[::-1][:5]])

    np.savez(f"cp_rank_{R}.npz", X=X, ids=ids, charges=charges,
            mol=M, charge=Q, spin=S, expert=E, weights=np.asarray(cp.weights))

    if len(rank_list) > 1:
        rel_error_list.append(rel_error) 

if len(rank_list) > 1:
    np.savetxt(f"relative_errors_to_rank_{len(rel_error_list)}.txt", np.column_stack((rank_list,rel_error_list)))