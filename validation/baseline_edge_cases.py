#!/usr/bin/env python3
"""
Baseline edge-case testing of the PRODUCTION code path (run_all_proteins.py
functions + scoring/anomaly_v2.py), as-is, no fixes.

Unlike tests/test_pipeline_edge_cases.py (which tests scoring/signals.py — a
parallel implementation the batch pipeline never calls), this exercises the
functions the pipeline actually runs, and records behavior (pass / crash /
silent wrong output) instead of only asserting "no crash".
"""
import json
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_all_proteins import run_tica, cluster_states, build_msm, compute_anomaly_signals  # noqa
from scoring.anomaly_v2 import (  # noqa
    compute_kinetic_signals, compute_local_density_signal,
    rank_normalize, quantile_normalize, moving_median, fuse_signals,
)

RESULTS = {}


def record(name, fn):
    entry = {"status": None, "detail": None}
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            detail = fn()
        entry["status"] = "ok"
        entry["detail"] = detail
        if w:
            entry["warnings"] = list({str(x.message)[:120] for x in w})[:5]
    except Exception as e:
        entry["status"] = "CRASH"
        entry["detail"] = f"{type(e).__name__}: {e}"
        entry["trace_tail"] = traceback.format_exc().splitlines()[-3:]
    RESULTS[name] = entry
    print(f"[{entry['status']:>5}] {name}: {str(entry['detail'])[:140]}")


def synth_Y(n, dim=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim))


# ---------------------------------------------------------------------------
# 1. Short trajectories through the full production chain
# ---------------------------------------------------------------------------
def full_chain(n_frames, lag_tica=5, dim=3, n_clusters=10, lag_msm=5, k=5, window=3, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_frames, 7)) + np.sin(np.arange(n_frames))[:, None]
    Y, _ = run_tica(X, lag=lag_tica, dim=dim)
    dtraj, _ = cluster_states(Y, n_clusters=n_clusters, seed=seed)
    msm, P, pi = build_msm(dtraj, lag=lag_msm)
    scores, comps = compute_anomaly_signals(msm, dtraj, Y, lag_msm=lag_msm,
                                            k_neighbors=k, window=window)
    n_labels = int(dtraj.max()) + 1
    return {
        "n_frames": n_frames, "n_labels": n_labels, "n_msm_states": int(msm.n_states),
        "state_space_mismatch": bool(n_labels != msm.n_states),
        "score_mean": float(np.mean(scores)), "score_std": float(np.std(scores)),
        "finite": bool(np.all(np.isfinite(scores))),
        "surprise_tail_frames_zero_raw": int(lag_msm),
    }


for n in (10, 12, 20, 50):
    record(f"short_trajectory_{n}_frames_full_chain", lambda n=n: full_chain(n))

# 9 frames: pipeline gate rejects <10 in run_pipeline; the functions themselves:
record("9_frames_functions_only", lambda: full_chain(9))

# ---------------------------------------------------------------------------
# 2. lag >= trajectory length (clamps) and lag equal to length in scorer
# ---------------------------------------------------------------------------
record("tica_lag_larger_than_T", lambda: full_chain(24, lag_tica=100))
record("msm_lag_larger_than_T", lambda: full_chain(24, lag_msm=100))


def scorer_lag_equal_T():
    n = 20
    dtraj = np.arange(n) % 3
    msm, P, pi = build_msm(dtraj, lag=1)
    r, s = compute_kinetic_signals(msm, dtraj, lag_msm=n)  # no clamp inside scorer!
    return {"rarity_finite": bool(np.all(np.isfinite(r))),
            "surprise_all_zero": bool(np.all(s == 0)),
            "note": "scorer itself does NOT clamp lag; caller must"}


record("scorer_called_with_lag_equal_T", scorer_lag_equal_T)

# ---------------------------------------------------------------------------
# 3. Constant / near-constant features
# ---------------------------------------------------------------------------
def constant_features():
    X = np.ones((50, 7))
    Y, _ = run_tica(X, lag=5, dim=3)
    return {"tica_output_finite": bool(np.all(np.isfinite(Y))),
            "tica_output_norm": float(np.abs(Y).max())}


record("constant_features_tica", constant_features)


def near_zero_variance_chain():
    rng = np.random.default_rng(1)
    X = np.ones((60, 7)) + rng.normal(scale=1e-10, size=(60, 7))
    return full_chain_from_X(X)


def full_chain_from_X(X, n_clusters=10, lag_msm=5, k=5, window=3):
    Y, _ = run_tica(X, lag=5, dim=3)
    dtraj, _ = cluster_states(Y, n_clusters=n_clusters, seed=42)
    msm, P, pi = build_msm(dtraj, lag=lag_msm)
    scores, _ = compute_anomaly_signals(msm, dtraj, Y, lag_msm=lag_msm,
                                        k_neighbors=k, window=window)
    return {"n_msm_states": int(msm.n_states), "n_labels": int(dtraj.max()) + 1,
            "finite": bool(np.all(np.isfinite(scores))),
            "score_std": float(np.std(scores))}


record("near_zero_variance_full_chain", near_zero_variance_chain)

# ---------------------------------------------------------------------------
# 4. Single-state and two-block disconnected trajectories
# ---------------------------------------------------------------------------
def single_state():
    dtraj = np.zeros(50, dtype=np.int64)
    msm, P, pi = build_msm(dtraj, lag=5)
    r, s = compute_kinetic_signals(msm, dtraj, 5)
    return {"n_states": int(msm.n_states), "rarity_const": bool(np.all(r == r[0])),
            "rarity_value": float(r[0]), "surprise_max": float(s.max())}


record("single_state_dtraj", single_state)


def disconnected_blocks():
    # frames 0-24 in states {0,1}, frames 25-49 in states {2,3}: no crossing
    a = np.array([0, 1] * 13)[:25]
    b = np.array([2, 3] * 13)[:25]
    dtraj = np.concatenate([a, b])
    msm, P, pi = build_msm(dtraj, lag=1)
    r, s = compute_kinetic_signals(msm, dtraj, 1)
    n_labels = int(dtraj.max()) + 1
    return {"n_labels": n_labels, "n_msm_states": int(msm.n_states),
            "which_half_survives": "first" if msm.n_states == 2 else "both/other",
            "rarity_second_half_default_1.0": bool(np.all(r[26:] == 1.0)),
            "rarity_first_half_example": float(r[0]),
            "note": "half the trajectory silently gets default rarity + zero surprise"
            if msm.n_states == 2 else ""}


record("disconnected_two_blocks", disconnected_blocks)

# ---------------------------------------------------------------------------
# 5. Cluster/kNN parameter edges
# ---------------------------------------------------------------------------
record("clusters_gt_frames", lambda: full_chain(14, n_clusters=50))
record("k_neighbors_gt_frames", lambda: full_chain(20, k=500))


def knn_k0():
    Y = synth_Y(1)
    d = compute_local_density_signal(Y, k=0)
    return {"len": len(d), "all_zero": bool(np.all(d == 0))}


record("density_single_frame_k0", knn_k0)

# ---------------------------------------------------------------------------
# 6. Normalization / smoothing primitives
# ---------------------------------------------------------------------------
record("rank_normalize_empty", lambda: {"len": len(rank_normalize(np.array([])))})
record("rank_normalize_single", lambda: {"val": rank_normalize(np.array([3.0])).tolist()})
record("rank_normalize_constant", lambda: {"vals": rank_normalize(np.ones(5)).tolist()})
record("rank_normalize_two_ties_of_two", lambda: {
    "vals": (rank_normalize(np.array([1.0, 1.0, 2.0, 2.0])) * 100).tolist(),
    "note": "tied values should get equal scores — they don't"})
record("quantile_normalize_constant", lambda: {"vals": quantile_normalize(np.ones(5)).tolist()})
record("moving_median_window_gt_len", lambda: {
    "vals_len": len(moving_median(np.arange(5.0), window=99))})
record("moving_median_even_window", lambda: {
    "vals": moving_median(np.arange(6.0), window=4).tolist(), "note": "window bumped to 5"})


def fuse_single_signal():
    raw, comps = fuse_signals({"only": np.arange(10.0)})
    return {"score_range": [float(raw.min()), float(raw.max())]}


record("fuse_single_signal", fuse_single_signal)

# ---------------------------------------------------------------------------
# 7. Invalid state labels straight into the scorer (production guard check)
# ---------------------------------------------------------------------------
def invalid_labels():
    dtraj = np.array([0, 1, 2, 1, 0, 1, 2, 1, 0, 1] * 5)
    msm, P, pi = build_msm(dtraj, lag=1)
    bad = dtraj.copy()
    bad[3] = -7
    bad[7] = 99
    r, s = compute_kinetic_signals(msm, bad, 1)
    return {"finite": bool(np.all(np.isfinite(r)) and np.all(np.isfinite(s))),
            "rarity_at_bad": [float(r[3]), float(r[7])],
            "note": "bad labels silently become rarity=1.0 (max-anomalous default)"}


record("invalid_labels_into_scorer", invalid_labels)

# ---------------------------------------------------------------------------
# 8. Real-protein regression: does a re-run reproduce the laptop artifacts?
# ---------------------------------------------------------------------------
def reproducibility_1vii():
    old = np.load(ROOT / "artifacts/1VII/dtraj.npy")
    new = np.load(ROOT / "baseline_artifacts/1VII/dtraj.npy")
    if len(old) != len(new):
        return {"comparable": False}
    return {"comparable": True, "identical_dtraj": bool(np.array_equal(old, new)),
            "n_differing": int((old != new).sum()),
            "note": "same seed, same data, cross-machine determinism check"}


record("cross_machine_reproducibility_1VII_dtraj", reproducibility_1vii)

out = ROOT / "validation/baseline_edge_results.json"
out.write_text(json.dumps(RESULTS, indent=2))
print(f"\nSaved -> {out}")
