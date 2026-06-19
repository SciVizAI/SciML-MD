# scoring/

**Stage 6 of the SciML-MD pipeline — Anomaly Scoring.**

This is the core of SciML-MD. Three independent signals are computed,
rank-normalised, fused by median, and smoothed to produce hotspot scores.

## Signals

| Signal | Formula | High score means |
|---|---|---|
| Rarity | `1 − π[s_t]` | Frame is in a rarely-visited state |
| Transition surprise | `−log P[s_t → s_{t+lag}]` | Unexpected state transition |
| Local density | `−mean k-NN distance in tICA space` | Frame is isolated in conformation space |

## Why median fusion?
Mean fusion lets one dominant signal skew the result. Median fusion
means all three signals must agree — more robust, less overfit.

## Residue score formula
```
residue_score = (0.5 × RMSF_norm[i] + 0.5 × mean_frame_score) × 100
```

**Outputs:** `frame_scores_dynamic.csv`, `residue_scores_dynamic.json`
