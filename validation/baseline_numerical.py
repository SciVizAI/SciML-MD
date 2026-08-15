#!/usr/bin/env python3
"""
Baseline numerical validation of the SciML-MD scoring path (AS-IS, no fixes).

Checks, per protein (1VII, 8H0R from fresh baseline run; 1CRN from laptop artifacts):
  A. Active-set mapping bug: production compute_kinetic_signals(raw kmeans dtraj)
     vs correct computation (dtraj remapped through msm.count_model.state_symbols).
     Metrics: n frames in dropped states, n frames misindexed, Spearman rho,
     top-10% overlap for rarity/surprise and for the fused score.
  B. Hand-computed unit checks on a tiny analytic MSM (exact expected values).
  C. Tie-handling in rank_normalize: spread assigned to tied values.
  D. Residue scores vs RMSF: Pearson r (expected exactly 1.0 -> pure RMSF).
  E. Mean-score ~= 50 artifact across proteins.

Outputs validation/baseline_numerical_results.json + printed summary.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scoring.anomaly_v2 import (  # noqa: E402
    compute_kinetic_signals,
    compute_local_density_signal,
    fuse_signals,
    moving_median,
    rank_normalize,
)

RESULTS = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class MSMShim:
    """Duck-typed MSM carrying P/pi/n_states (mirrors what production uses)."""

    def __init__(self, P, pi):
        self.transition_matrix = np.asarray(P)
        self.stationary_distribution = np.asarray(pi)
        self.n_states = len(pi)


def refit_msm(dtraj, lag):
    """Re-estimate the MSM exactly as run_all_proteins.build_msm does,
    but also return the active-set symbols (original kmeans labels kept)."""
    from deeptime.markov.msm import MaximumLikelihoodMSM

    lag = max(1, min(lag, len(dtraj) // 4))
    msm = MaximumLikelihoodMSM(lagtime=lag, reversible=True).fit(dtraj).fetch_model()
    symbols = np.asarray(msm.count_model.state_symbols)  # active-set -> original label
    return msm, symbols, lag


def remap_dtraj(dtraj, symbols):
    """Map original kmeans labels -> active-set indices; dropped states -> -1."""
    lookup = -np.ones(int(dtraj.max()) + 1, dtype=np.int64)
    for active_idx, orig_label in enumerate(symbols):
        lookup[orig_label] = active_idx
    return lookup[dtraj]


def correct_kinetic_signals(msm, dtraj_active, lag):
    """Reference implementation: same math as production but on a correctly
    remapped dtraj; frames in dropped states (-1) get rarity=1, surprise skipped."""
    n = len(dtraj_active)
    pi = msm.stationary_distribution
    P = msm.transition_matrix
    ns = msm.n_states
    rarity = np.ones(n)
    ok = (dtraj_active >= 0) & (dtraj_active < ns)
    rarity[ok] = 1.0 - pi[dtraj_active[ok]]
    surprise = np.zeros(n)
    eps = 1e-12
    for t in range(n - lag):
        s1, s2 = dtraj_active[t], dtraj_active[t + lag]
        if 0 <= s1 < ns and 0 <= s2 < ns:
            surprise[t] = -np.log(max(P[s1, s2], eps))
    return rarity, surprise


def fused_score(rarity, surprise, Y, k, window):
    density = compute_local_density_signal(Y, k=min(k, len(Y) - 1))
    signals = {
        "rarity": rarity,
        "transition_surprise": surprise,
        "local_density": -density,
    }
    raw, comps = fuse_signals(signals, method="median", normalize_method="rank")
    return moving_median(raw * 100.0, window=window), comps


def topk_overlap(a, b, frac=0.10):
    k = max(1, int(len(a) * frac))
    ta = set(np.argsort(a)[-k:])
    tb = set(np.argsort(b)[-k:])
    return len(ta & tb) / k


# ---------------------------------------------------------------------------
# A. Active-set mapping bug on real proteins
# ---------------------------------------------------------------------------
def check_active_set(pid, dtraj, Y, lag, k, window):
    out = {}
    msm, symbols, lag_eff = refit_msm(dtraj, lag)
    ns = msm.n_states
    n_clusters = int(dtraj.max()) + 1
    dropped = sorted(set(range(n_clusters)) - set(symbols.tolist()))
    out["n_frames"] = int(len(dtraj))
    out["n_clusters_labels"] = n_clusters
    out["n_msm_states"] = int(ns)
    out["dropped_original_labels"] = dropped
    out["identity_mapping"] = bool(np.array_equal(symbols, np.arange(ns)))

    dtraj_active = remap_dtraj(dtraj, symbols)
    in_dropped = dtraj_active == -1
    # frames whose label is a *valid index* into pi but maps to the WRONG state
    wrong_row = (~in_dropped) & (dtraj < ns) & (dtraj != dtraj_active)
    silently_defaulted = in_dropped & (dtraj >= ns)   # get rarity=1.0 default
    wrongly_indexed_dropped = in_dropped & (dtraj < ns)  # dropped but pi[label] exists!

    out["frames_in_dropped_states"] = int(in_dropped.sum())
    out["frames_wrong_pi_row (survivors misindexed)"] = int(wrong_row.sum())
    out["dropped_frames_reading_valid_but_wrong_pi"] = int(wrongly_indexed_dropped.sum())
    out["dropped_frames_hitting_default"] = int(silently_defaulted.sum())

    # Production (buggy) signals
    r_bug, s_bug = compute_kinetic_signals(msm, dtraj, lag_eff)
    # Correct signals
    r_ok, s_ok = correct_kinetic_signals(msm, dtraj_active, lag_eff)

    out["rarity_spearman"] = float(spearmanr(r_bug, r_ok).statistic)
    out["surprise_spearman"] = float(spearmanr(s_bug, s_ok).statistic)
    out["rarity_top10_overlap"] = topk_overlap(r_bug, r_ok)
    out["surprise_top10_overlap"] = topk_overlap(s_bug, s_ok)

    f_bug, _ = fused_score(r_bug, s_bug, Y, k, window)
    f_ok, _ = fused_score(r_ok, s_ok, Y, k, window)
    out["fused_spearman"] = float(spearmanr(f_bug, f_ok).statistic)
    out["fused_top10_overlap"] = topk_overlap(f_bug, f_ok)
    out["fused_mean_bug"] = float(f_bug.mean())
    out["fused_mean_correct"] = float(f_ok.mean())
    return out


# ---------------------------------------------------------------------------
# B. Hand-computed unit check (tiny analytic chain)
# ---------------------------------------------------------------------------
def unit_check():
    """3-state chain with known P, pi; dtraj visits states 0,1,2 plus a bogus
    label 5 (simulating a dropped state). Expected values computed by hand."""
    P = np.array([[0.9, 0.1, 0.0],
                  [0.1, 0.8, 0.1],
                  [0.0, 0.2, 0.8]])
    # stationary: solve piP = pi -> pi = (1/4, 1/4... ) compute numerically
    evals, evecs = np.linalg.eig(P.T)
    pi = np.real(evecs[:, np.argmax(np.real(evals))])
    pi = pi / pi.sum()
    msm = MSMShim(P, pi)
    dtraj = np.array([0, 1, 2, 2, 1, 0, 5, 1])
    lag = 1
    r, s = compute_kinetic_signals(msm, dtraj, lag)
    checks = {}
    checks["rarity_state0_expected"] = float(1 - pi[0])
    checks["rarity_state0_actual"] = float(r[0])
    checks["rarity_bogus_label5_actual"] = float(r[6])  # default 1.0
    checks["surprise_0to1_expected"] = float(-np.log(P[0, 1]))
    checks["surprise_0to1_actual"] = float(s[0])
    checks["surprise_2to2_expected"] = float(-np.log(P[2, 2]))
    checks["surprise_2to2_actual"] = float(s[2])
    # transition into bogus state (frame 5: 0 -> 5): production SKIPS it -> 0
    checks["surprise_into_bogus_actual(should_flag_not_zero)"] = float(s[5])
    # last-lag frame forced to 0
    checks["surprise_last_frame"] = float(s[-1])
    checks["rarity_math_ok"] = bool(np.isclose(r[0], 1 - pi[0]))
    checks["surprise_math_ok"] = bool(
        np.isclose(s[0], -np.log(P[0, 1])) and np.isclose(s[2], -np.log(P[2, 2]))
    )
    return checks


# ---------------------------------------------------------------------------
# C. Tie handling in rank_normalize
# ---------------------------------------------------------------------------
def tie_check(pid, dtraj, lag):
    msm, symbols, lag_eff = refit_msm(dtraj, lag)
    r_bug, s_bug = compute_kinetic_signals(msm, dtraj, lag_eff)
    out = {}
    # tied raw values -> spread of normalized scores
    rn = rank_normalize(r_bug) * 100
    sn = rank_normalize(s_bug) * 100
    for name, raw, norm in [("rarity", r_bug, rn), ("surprise", s_bug, sn)]:
        vals, counts = np.unique(np.round(raw, 12), return_counts=True)
        biggest_tie = vals[np.argmax(counts)]
        mask = np.round(raw, 12) == biggest_tie
        out[f"{name}_largest_tie_group_size"] = int(mask.sum())
        out[f"{name}_tied_raw_value"] = float(biggest_tie)
        out[f"{name}_score_spread_within_tie_group"] = float(norm[mask].max() - norm[mask].min())
        # correct tie-aware version for comparison
        fair = (rankdata(raw, method="average") - 1) / (len(raw) - 1) * 100
        out[f"{name}_fair_spread_within_tie_group"] = float(fair[mask].max() - fair[mask].min())
    return out


# ---------------------------------------------------------------------------
# D. Residue score vs RMSF
# ---------------------------------------------------------------------------
def residue_vs_rmsf(pid, results_dir):
    import mdtraj as md

    top = {"1VII": ROOT / "data/1VII/topology.pdb",
           "8H0R": ROOT / "data/8H0R/canonical_topology.pdb"}[pid]
    trj = {"1VII": ROOT / "data/1VII/traj.xtc",
           "8H0R": ROOT / "data/8H0R/traj.xtc"}[pid]
    traj = md.load(str(trj), top=str(top))
    ca = traj.topology.select("name CA")
    ct = traj.atom_slice(ca).superpose(traj.atom_slice(ca))
    rmsf = np.sqrt(((ct.xyz - ct.xyz.mean(0)) ** 2).sum(-1).mean(0))
    scores = json.load(open(results_dir / pid / "residue_scores_dynamic.json"))
    vals = np.array(list(scores.values()))
    r = pearsonr(rmsf[: len(vals)], vals).statistic
    return {"pearson_r_residue_score_vs_rmsf": float(r),
            "n_residues": len(vals),
            "score_min": float(vals.min()), "score_max": float(vals.max())}


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------
def main():
    cfg = {
        # fresh baseline run (batch_runner defaults, same as laptop 1VII/8H0R runs)
        "1VII": {"art": ROOT / "baseline_artifacts/1VII", "lag": 5, "k": 5, "window": 3,
                 "res": ROOT / "baseline_results"},
        "8H0R": {"art": ROOT / "baseline_artifacts/8H0R", "lag": 5, "k": 5, "window": 3,
                 "res": ROOT / "baseline_results"},
        # laptop artifacts (1001 frames, solvated 1CRN)
        "1CRN": {"art": ROOT / "artifacts/1CRN", "lag": 5, "k": 5, "window": 3, "res": None},
    }
    for pid, c in cfg.items():
        dtraj = np.load(c["art"] / "dtraj.npy")
        Y = np.load(c["art"] / "tica_coords.npy")
        RESULTS.setdefault(pid, {})
        RESULTS[pid]["active_set"] = check_active_set(pid, dtraj, Y, c["lag"], c["k"], c["window"])
        RESULTS[pid]["ties"] = tie_check(pid, dtraj, c["lag"])
        if c["res"] is not None:
            RESULTS[pid]["residue_vs_rmsf"] = residue_vs_rmsf(pid, c["res"])
        csvp = (c["res"] or ROOT / "results") / pid / "frame_scores_dynamic.csv"
        if csvp.exists():
            import pandas as pd
            RESULTS[pid]["mean_fused_score"] = float(pd.read_csv(csvp)["score_dynamic"].mean())

    RESULTS["unit_checks_tiny_msm"] = unit_check()

    out = ROOT / "validation/baseline_numerical_results.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print(json.dumps(RESULTS, indent=2))


if __name__ == "__main__":
    main()
