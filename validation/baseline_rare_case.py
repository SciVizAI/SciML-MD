#!/usr/bin/env python3
"""
Baseline rare-case detection accuracy on REAL proteins (1VII, 8H0R, 1CRN).

Ground truth is operational (no synthetic data), defined from the correctly
estimated MSM (labels remapped through the active set):

  L_rare_state : frame is in a state whose stationary probability pi is in the
                 bottom quartile of states, OR in a disconnected (dropped) state.
  L_rare_trans : the transition (s_t -> s_{t+lag}) was observed <= 1 time in the
                 lag-count matrix (rare/unique kinetic event), or leaves the
                 connected set.

We then ask: does the score the pipeline ACTUALLY produced (buggy indexing)
rank those frames highly?  And how much does correct indexing recover?
Density channel is kept in the fused score but labels are purely kinetic, to
avoid circularity.

AUROC / AUPRC via sklearn.  Also: channel cross-correlations (is transition
surprise anti-correlated with rarity?).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scoring.anomaly_v2 import (  # noqa
    compute_kinetic_signals, compute_local_density_signal, fuse_signals, moving_median,
)


def refit(dtraj, lag):
    from deeptime.markov.msm import MaximumLikelihoodMSM
    lag = max(1, min(lag, len(dtraj) // 4))
    msm = MaximumLikelihoodMSM(lagtime=lag, reversible=True).fit(dtraj).fetch_model()
    symbols = np.asarray(msm.count_model.state_symbols)
    lookup = -np.ones(int(dtraj.max()) + 1, dtype=np.int64)
    for i, s in enumerate(symbols):
        lookup[s] = i
    return msm, lookup[dtraj], lag


def correct_signals(msm, da, lag):
    n = len(da)
    pi, P, ns = msm.stationary_distribution, msm.transition_matrix, msm.n_states
    r = np.ones(n)
    ok = (da >= 0) & (da < ns)
    r[ok] = 1.0 - pi[da[ok]]
    s = np.zeros(n)
    for t in range(n - lag):
        a, b = da[t], da[t + lag]
        if 0 <= a < ns and 0 <= b < ns:
            s[t] = -np.log(max(P[a, b], 1e-12))
    return r, s


def fuse(r, s, Y, k, window):
    d = compute_local_density_signal(Y, k=min(k, len(Y) - 1))
    raw, comps = fuse_signals({"rarity": r, "transition_surprise": s, "local_density": -d},
                              method="median", normalize_method="rank")
    return moving_median(raw * 100.0, window=window), comps


def labels(msm, da, dtraj_raw, lag):
    """Operational kinetic ground truth from the correct MSM."""
    pi, ns = msm.stationary_distribution, msm.n_states
    n = len(da)
    # bottom-quartile-pi states (at least 1 state)
    q25 = np.quantile(pi, 0.25)
    rare_states = set(np.where(pi <= q25)[0].tolist())
    L_state = np.zeros(n, dtype=int)
    for t in range(n):
        if da[t] == -1 or da[t] in rare_states:
            L_state[t] = 1
    # rare transitions from lag-count matrix on the raw labels
    n_labels = int(dtraj_raw.max()) + 1
    C = np.zeros((n_labels, n_labels))
    for t in range(n - lag):
        C[dtraj_raw[t], dtraj_raw[t + lag]] += 1
    L_trans = np.zeros(n, dtype=int)
    for t in range(n - lag):
        a, b = da[t], da[t + lag]
        if a == -1 or b == -1 or C[dtraj_raw[t], dtraj_raw[t + lag]] <= 1:
            L_trans[t] = 1
    return L_state, L_trans


def evaluate(y, score):
    score = np.asarray(score, dtype=float)
    if np.any(~np.isfinite(score)):  # undefined tail (NaN surprise) -> lowest
        fill = np.nanmin(score) if np.any(np.isfinite(score)) else 0.0
        score = np.nan_to_num(score, nan=fill)
    if y.sum() == 0 or y.sum() == len(y):
        return {"auroc": None, "auprc": None, "n_pos": int(y.sum())}
    return {"auroc": round(float(roc_auc_score(y, score)), 4),
            "auprc": round(float(average_precision_score(y, score)), 4),
            "n_pos": int(y.sum()), "prevalence": round(float(y.mean()), 3)}


def main():
    cfg = {
        "1VII": {"art": ROOT / "baseline_artifacts/1VII", "lag": 5, "k": 5, "w": 3},
        "8H0R": {"art": ROOT / "baseline_artifacts/8H0R", "lag": 5, "k": 5, "w": 3},
        "1CRN": {"art": ROOT / "artifacts/1CRN", "lag": 5, "k": 5, "w": 3},
    }
    out = {}
    for pid, c in cfg.items():
        dtraj = np.load(c["art"] / "dtraj.npy")
        Y = np.load(c["art"] / "tica_coords.npy")
        msm, da, lag = refit(dtraj, c["lag"])

        r_bug, s_bug = compute_kinetic_signals(msm, dtraj, lag)   # production
        r_ok, s_ok = correct_signals(msm, da, lag)                 # corrected

        f_bug, comps_bug = fuse(r_bug, s_bug, Y, c["k"], c["w"])
        f_ok, comps_ok = fuse(r_ok, s_ok, Y, c["k"], c["w"])

        L_state, L_trans = labels(msm, da, dtraj, lag)

        out[pid] = {
            "n_frames": len(dtraj),
            "rare_state_frames": evaluate(L_state, f_bug) | {"_": "fused_buggy"},
            "detection": {
                "rare_state | fused (as-shipped)": evaluate(L_state, f_bug),
                "rare_state | rarity channel (as-shipped)": evaluate(L_state, r_bug),
                "rare_state | fused (corrected indexing)": evaluate(L_state, f_ok),
                "rare_state | rarity channel (corrected)": evaluate(L_state, r_ok),
                "rare_trans | fused (as-shipped)": evaluate(L_trans, f_bug),
                "rare_trans | surprise channel (as-shipped)": evaluate(L_trans, s_bug),
                "rare_trans | fused (corrected indexing)": evaluate(L_trans, f_ok),
                "rare_trans | surprise channel (corrected)": evaluate(L_trans, s_ok),
            },
            "channel_correlations_corrected": {
                "rarity_vs_surprise": round(float(spearmanr(r_ok, s_ok).statistic), 3),
                "rarity_vs_density": round(float(spearmanr(
                    r_ok, comps_ok["local_density"]).statistic), 3),
                "surprise_vs_density": round(float(spearmanr(
                    s_ok, comps_ok["local_density"]).statistic), 3),
            },
        }
        del out[pid]["rare_state_frames"]

    p = ROOT / "validation/baseline_rare_case_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
