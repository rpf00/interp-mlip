import sys, numpy as np, matplotlib.pyplot as plt

R = int(sys.argv[1]) if len(sys.argv) > 1 else 3          # rank to inspect

# 1. relative error vs rank
rank, err = np.loadtxt("relative_errors_to_rank_20.txt", unpack=True)
plt.figure(figsize=(4, 3))
plt.plot(rank, err, "o-")
plt.xlabel("rank"); plt.ylabel("relative error")
plt.tight_layout(); plt.savefig("pngs/error_vs_rank.png")

# 2. each component's loading vector, one panel per mode (scaled to max 1 so components compare)
d = np.load(f"cp_rank_{R}.npz")
modes = {"charge": (d["charges"], d["charge"]),
         "spin level k": (np.arange(len(d["spin"])), d["spin"]),
         "expert": (np.arange(len(d["expert"])), d["expert"]),
         "molecule": (np.arange(len(d["mol"])), d["mol"])}

fig, axs = plt.subplots(2, 2, figsize=(12, 9))
axs = axs.flat
for ax, (name, (x, F)) in zip(axs, modes.items()):
    F = F / (F.max(axis=0) + 1e-12)
    for r in range(R):
        ax.plot(x, F[:, r], "o-" if len(x) < 40 else ".", ms=4, label=f"comp {r}")
    ax.set_xlabel(name)
    ax.set_ylabel("loading (scaled to max 1)")
axs[1].legend()
fig.tight_layout(); fig.savefig(f"pngs/factors_rank_{R}.png", dpi=150)