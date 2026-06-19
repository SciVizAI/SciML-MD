# msm/

**Stages 4 & 5 of the SciML-MD pipeline — tICA and Markov State Model.**

## Files

| File | Purpose |
|---|---|
| `select_lag_and_dim.py` | VAMP-2 grid search to select optimal tICA lag and dimension |
| `validation.py` | Chapman-Kolmogorov test and implied timescale analysis |
| `bootstrap_msm.py` | Bootstrap uncertainty quantification for π and P |
| `soft_states.py` | HMM soft state assignments |
| `reproducibility.py` | Deterministic seed management |
| `input_validation.py` | Input integrity checks |

## What happens here

1. **tICA** — finds the slowest collective motions in the 8D feature space
2. **KMeans** — discretises the tICA space into 20 conformational states
3. **MSM** — estimates transition matrix P and stationary distribution π

**Outputs:** `tica_coords.npy`, `dtraj.npy`, `P.npy`, `pi.npy`

The MSM is validated with Chapman-Kolmogorov tests before anomaly scoring
proceeds. Invalid models are rejected rather than accepted by default.
