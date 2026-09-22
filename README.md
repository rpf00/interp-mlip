# Interpreting MLIP Expert Routing

Tensor-decomposition analysis of the mixture-of-linear-experts (MoLE) router in
Meta's UMA universal machine-learning interatomic potential.

## Idea

UMA routes each system to a mixture of 32 expert weight sets via a small router.
The router is a **pure function of composition, charge, spin and task** — it never
sees atomic positions. That has two consequences this project exploits:

1. Routing coefficients can be extracted without running the energy/force forward
   pass, so 20k systems cost seconds rather than GPU-hours.
2. The routing function can be **swept on a chosen grid** rather than sampled from a
   dataset, giving a tensor that is complete by construction.

We build a 4-way tensor of routing coefficients and factor it with non-negative CP
to ask what the router has learned to distinguish.

## Final tensor design

```
X ∈ ℝ≥0^(500 × 5 × 4 × 32)   [composition × charge × spin level × expert]
```

- **composition** — distinct OMol25 molecules (not elements; see below)
- **charge** — q ∈ {−2, −1, 0, +1, +2}
- **spin level** — the *k*-th *allowed* multiplicity given electron-count parity,
  not an absolute multiplicity. Parity couples charge and spin, so indexing this way
  keeps every cell both physical and populated.
- **expert** — all 32, with UMA's constant softmax floor removed

100% fill, no structural missingness, uniform cell reliability.

## Main result

Non-negative CP at rank 3 recovers a **singlet detector**: a component whose
charge ⊗ spin outer product is 1.08–1.25 on multiplicity-1 cells and ≤0.065 on all
others (17× contrast), dominated by a single expert and essentially
composition-independent (loading CV 0.15).

This component was **predicted from the per-expert charge sweep before the
decomposition was run**, which is what makes it a result rather than an
after-the-fact reading.

The parity coupling is not representable in either mode alone — it emerges as a
product of two loading vectors, which is the multilinear structure doing real work.

## Negative results that shaped the design

These mattered more than the positive one:

| Finding | Consequence |
|---|---|
| OMat24 and OMol25 share **0 of 8** top experts | Task cannot be mode 2 — CP would return task labels, not chemistry |
| Element-row pairwise cosine 0.78 (OMat24), 0.88 (OMol25) | Element is a poor mode; presence-based averaging washes rows out |
| UMA's `_softmax` adds a constant 5e-3 floor (13.8% of mass) | A rank-1 all-ones nuisance; must be stripped before fitting |
| Uncentered element rows dominated by a shared mean | NNCP cannot center, so the mean competes for components |

## Repository layout

```
scripts/
  uma_routing_extract.py      extract routing coefficients; reproduce UMA Fig. 5/6
  uma_charge_spin_sweep.py    build the 4-way tensor; fit non-negative CP
  uma_charge_smoothness.py    test whether the charge response is smooth or arbitrary
  compare_routing.py          expert concentration, centered SVD, element cosine
  design_evidence_figure.py   assemble the four-panel evidence figure
  download_data.sh            fetch the datasets (not committed)
docs/
  analysis_log.md             step-by-step record with conclusions per step
```

## Setup

```bash
pip install fairchem-core tensorly matplotlib numpy ase
hf auth login          # facebook/UMA is gated
bash scripts/download_data.sh
```

## Reproduce

```bash
# validate extraction against the published figure's construction
python scripts/uma_routing_extract.py --asedb data/rattled-300-subsampled \
    --task omat --n-samples 20000 --strip-floor

# build the 4-way tensor and decompose
python scripts/uma_charge_spin_sweep.py --asedb data/neutral_val \
    --n-comps 500 --fit-rank 3
```

## Data

| Dataset | Subset | Size | Gated |
|---|---|---|---|
| OMat24 | `rattled-300-subsampled` (35,579 structures) | 68 MB | no |
| OMol25 | `val_neutral` (27,697 structures) | 118 MB | yes — request access |

Neither is redistributed here. `download_data.sh` fetches them from FAIR's servers.
OMol25 requires accepting the terms at <https://huggingface.co/facebook/OMol25>.

Model: **UMA-S-1.1** (32 experts). The original UMA-S from the paper is no longer in
the fairchem registry; same architecture, different training run, so expect
qualitative rather than numerical agreement with the published figures.

## Limitations

- **Extrapolation ceiling.** `val_neutral` is entirely charge 0 / multiplicity 1, so
  only one cell of the swept grid is in-distribution. "UMA has a singlet expert" is
  well supported; "UMA routes on charge state" is not yet — that needs compositions
  actually trained with varied charge and spin (the metal complexes and electrolytes
  in the 20 GB full validation split).
- **Rank not formally selected.** Reconstruction error declines smoothly with no
  elbow (0.43 → 0.34 → 0.28 for R = 2, 3, 4). CORCONDIA, split-half and degeneracy
  diagnostics are outstanding; rank 3 is a working choice justified by component
  stability, not by fit.
- **Narrow chemistry.** 17 elements, no transition metals.
- **Element cosine needs a null.** Cosine between non-negative vectors is floored well
  above zero, so the raw values are not interpretable without a permutation baseline.
- **Error prediction untouched.** No MLIP-vs-DFT error labels have been extracted or
  regressed against the routing-derived latents.

## References

- Wood et al., *UMA: A Family of Universal Models for Atoms* (arXiv:2506.23971) — the
  routing-coefficient analysis is in Appendix F.1.
- Bro, *PARAFAC: Tutorial and applications*, Chemom. Intell. Lab. Syst. 38 (1997).
- Kolda & Bader, *Tensor Decompositions and Applications*, SIAM Rev. 51 (2009).
