#!/usr/bin/env python3
"""
Layer 2 — MSM statistical validity battery, executed on the project's actual
systems (1VII, 8H0R, 1UBQ, 1CRN), with figures.

Runs, per protein:
  1. Implied-timescale convergence (msm/validation.py::implied_timescales_convergence)
  2. Chapman-Kolmogorov test  — implemented here with correct active-set
     embedding (the repo's chapman_kolmogorov_test compares P matrices of
     different state spaces without remapping — same bug class as the scoring
     path; flagged in VALIDATION_REPORT.md)
  3. VAMP-2 grid search (msm/select_lag_and_dim.py::compute_vamp2_score)
     on recomputed features (proteins with trajectory available in workspace)
  4. Block-bootstrap 95% CIs on the stationary distribution pi
     (block resampling preserves temporal correlation; the repo's 'frames'
     mode breaks transitions and is not used)

Figures -> validation/figures/{PID}_its.png, _ck.png, _pi_ci.png, _vamp2.png
Numbers -> validation/layer2_results.json

Colors: Okabe-Ito colorblind-safe palette (standard for scientific figures).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIG = ROOT / "validation/figures"
FIG.mkdir(exist_ok=True)

from msm.validation import implied_timescales_convergence  # noqa: E402
from msm.select_lag_and_dim import compute_vamp2_score  # noqa: E402
from deeptime.markov.msm import MaximumLikelihoodMSM  # noqa: E402

# Okabe-Ito (CVD-safe)
C_BLUE, C_ORANGE, C_GREEN, C_VERM, C_GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#999999")
SERIES = [C_BLUE, C_ORANGE, C_GREEN, C_VERM]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
})

RESULTS = {}


def fit_msm(dtraj, lag):
    return MaximumLikelihoodMSM(lagtime=lag, reversible=True).fit(dtraj).fetch_model()


def embed_P(msm, n_labels):
    """Embed active-set transition matrix into the full label space (rows/cols
    of dropped labels = identity), plus the symbols mapping."""
    sym = np.asarray(msm.count_model.state_symbols)
    P_full = np.eye(n_labels)
    P_full[np.ix_(sym, sym)] = msm.transition_matrix
    return P_full, sym


def empirical_P(dtraj, lag, n_labels):
    C = np.zeros((n_labels, n_labels))
    for t in range(len(dtraj) - lag):
        C[dtraj[t], dtraj[t + lag]] += 1
    rows = C.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1.0
    return C / rows, C


# ---------------------------------------------------------------------------
# 1. Implied timescales
# ---------------------------------------------------------------------------
def run_its(pid, dtraj, base_lag):
    T = len(dtraj)
    lag_range = [l for l in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50] if l <= T // 5]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lags, its = implied_timescales_convergence(dtraj, lag_range=lag_range, n_its=3)

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for i in range(its.shape[1]):
        ax.plot(lags, its[:, i], "-o", ms=3.5, lw=1.5, color=SERIES[i],
                label=f"t{i + 2}")
    ax.plot(lags, lags, "--", lw=1, color=C_GREY)
    ax.text(lags[-1], lags[-1], " τ = lag", color=C_GREY, fontsize=8, va="bottom", ha="right")
    ax.axvline(base_lag, color=C_VERM, lw=1, ls=":", alpha=0.8)
    ax.text(base_lag, ax.get_ylim()[1], f" chosen lag={base_lag}", color=C_VERM,
            fontsize=8, va="top")
    ax.set(xlabel="lag time (frames)", ylabel="implied timescale (frames)",
           title=f"{pid} — implied timescales vs lag", yscale="log")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG / f"{pid}_its.png")
    plt.close(fig)

    # Convergence heuristic: relative change of slowest ITS across last two lags <= 20%?
    finite = np.isfinite(its[:, 0])
    conv = None
    if finite.sum() >= 2:
        last = its[finite, 0][-2:]
        conv = round(float(abs(last[1] - last[0]) / max(last[0], 1e-9)), 3)
    return {"lags_tested": [int(l) for l in lags],
            "slowest_its_by_lag": [None if not np.isfinite(v) else round(float(v), 1)
                                   for v in its[:, 0]],
            "slowest_its_rel_change_last_step": conv}


# ---------------------------------------------------------------------------
# 2. Chapman-Kolmogorov (active-set-correct)
# ---------------------------------------------------------------------------
def run_ck(pid, dtraj, base_lag, n_mult=4):
    n_labels = int(dtraj.max()) + 1
    msm = fit_msm(dtraj, base_lag)
    P1, sym = embed_P(msm, n_labels)

    # top-4 most-populated surviving labels
    pi_full = np.zeros(n_labels)
    pi_full[sym] = msm.stationary_distribution
    states = list(np.argsort(pi_full)[::-1][:4])

    ks = np.arange(1, n_mult + 1)
    pred = {s: [] for s in states}
    est = {s: [] for s in states}
    est_err = {s: [] for s in states}
    for k in ks:
        Pk = np.linalg.matrix_power(P1, k)
        Pe, C = empirical_P(dtraj, base_lag * k, n_labels)
        for s in states:
            pred[s].append(Pk[s, s])
            est[s].append(Pe[s, s])
            n_s = C[s].sum()
            p = Pe[s, s]
            est_err[s].append(1.96 * np.sqrt(max(p * (1 - p), 0) / max(n_s, 1)))

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.8), sharex=True)
    mismatch = []
    for ax, s in zip(axes.flat, states):
        ax.plot(ks * base_lag, pred[s], "-o", ms=3.5, lw=1.5, color=C_BLUE,
                label="MSM predicted  P(τ)^k")
        ax.errorbar(ks * base_lag, est[s], yerr=est_err[s], fmt="s--", ms=3.5,
                    lw=1.2, color=C_ORANGE, capsize=2, label="estimated at kτ")
        ax.set(title=f"state {s}  (π={pi_full[s]:.2f})", ylim=(0, 1.05))
        mismatch.append(float(np.max(np.abs(np.array(pred[s]) - np.array(est[s])))))
    for ax in axes[-1]:
        ax.set_xlabel("lag (frames)")
    for ax in axes[:, 0]:
        ax.set_ylabel("P(state → state)")
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"{pid} — Chapman-Kolmogorov test (base lag {base_lag})", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / f"{pid}_ck.png")
    plt.close(fig)

    inside = 0
    total = 0
    for s in states:
        for k in range(len(ks)):
            total += 1
            if abs(pred[s][k] - est[s][k]) <= est_err[s][k] + 1e-9:
                inside += 1
    return {"states_tested": [int(s) for s in states],
            "max_abs_deviation_per_state": [round(m, 3) for m in mismatch],
            "fraction_within_95CI": round(inside / total, 3)}


# ---------------------------------------------------------------------------
# 3. VAMP-2 grid (needs features -> trajectory present in workspace)
# ---------------------------------------------------------------------------
TRAJ = {
    "1VII": ("data/1VII/topology.pdb", "data/1VII/traj.xtc"),
    "8H0R": ("data/8H0R/canonical_topology.pdb", "data/8H0R/traj.xtc"),
    "1UBQ": ("data/1UBQ/canonical_topology.pdb", "data/1UBQ/traj.xtc"),
}


def run_vamp2(pid):
    if pid not in TRAJ:
        return {"skipped": "trajectory not in workspace"}
    from run_all_proteins import compute_features
    X, _ = compute_features(ROOT / TRAJ[pid][0], ROOT / TRAJ[pid][1])
    lags = [2, 3, 5, 8, 10, 15]
    dims = [2, 3, 5]
    grid = np.full((len(dims), len(lags)), np.nan)
    grid_repo = np.full((len(dims), len(lags)), np.nan)

    # NOTE: msm/select_lag_and_dim.py::compute_vamp2_score is numerically
    # unstable on real (ill-conditioned) MD features — its Cholesky-inverse
    # whitening with reg=1e-6 blows up, returning values like 2.5e5 where a
    # valid VAMP-2 score is bounded by `dim`. We therefore score with
    # deeptime's own cross-validated VAMP-2 (train/test split) and record the
    # repo values alongside for the record. See VALIDATION_REPORT.md §5c.
    from deeptime.decomposition import VAMP
    n_train = int(len(X) * 0.8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, d in enumerate(dims):
            for j, l in enumerate(lags):
                s_repo = compute_vamp2_score(X, l, d)
                grid_repo[i, j] = s_repo if np.isfinite(s_repo) else np.nan
                try:
                    train = VAMP(lagtime=l, dim=d).fit(X[:n_train]).fetch_model()
                    test = VAMP(lagtime=l, dim=d).fit(X[n_train:]).fetch_model()
                    grid[i, j] = float(train.score(test_model=test, r=2))
                except Exception:
                    grid[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    im = ax.imshow(grid, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(lags)), lags)
    ax.set_yticks(range(len(dims)), dims)
    ax.set(xlabel="tICA lag (frames)", ylabel="dimensions",
           title=f"{pid} — VAMP-2 cross-validated score (deeptime)")
    ax.grid(False)
    for i in range(len(dims)):
        for j in range(len(lags)):
            if np.isfinite(grid[i, j]):
                dark = grid[i, j] > np.nanmax(grid) * 0.7
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if dark else "#333333")
    fig.colorbar(im, ax=ax, shrink=0.85, label="VAMP-2")
    fig.savefig(FIG / f"{pid}_vamp2.png")
    plt.close(fig)

    best = np.unravel_index(np.nanargmax(grid), grid.shape)
    saturated = bool(np.nanmax(grid) >= dims[best[0]] - 1e-6)
    return {"grid_lags": lags, "grid_dims": dims,
            "scorer": "deeptime VAMP.score(r=2), 80/20 train/test split",
            "best": {"lag": lags[best[1]], "dim": dims[best[0]],
                     "score": round(float(np.nanmax(grid)), 4),
                     "saturated_at_dim (overfit flag)": saturated},
            "score_at_pipeline_default_lag5_dim3":
                round(float(grid[dims.index(3), lags.index(5)]), 4),
            "repo_scorer_values_UNSTABLE":
                [[None if not np.isfinite(v) else round(float(v), 3) for v in row]
                 for row in grid_repo]}


# ---------------------------------------------------------------------------
# 4. Block-bootstrap CIs on pi
# ---------------------------------------------------------------------------
def run_bootstrap(pid, dtraj, base_lag, n_boot=200, block=10, seed=123):
    rng = np.random.RandomState(seed)
    n_labels = int(dtraj.max()) + 1
    T = len(dtraj)
    n_blocks = T // block
    samples = np.full((n_boot, n_labels), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for b in range(n_boot):
            idx = []
            for blk in rng.choice(n_blocks, size=n_blocks, replace=True):
                idx.extend(range(blk * block, min((blk + 1) * block, T)))
            boot = dtraj[np.array(idx[:T])]
            try:
                m = fit_msm(boot, base_lag)
                sym = np.asarray(m.count_model.state_symbols)
                samples[b, sym] = m.stationary_distribution
            except Exception:
                continue

    ref = fit_msm(dtraj, base_lag)
    sym = np.asarray(ref.count_model.state_symbols)
    pi_ref = np.zeros(n_labels)
    pi_ref[sym] = ref.stationary_distribution

    lo = np.nanpercentile(samples, 2.5, axis=0)
    hi = np.nanpercentile(samples, 97.5, axis=0)
    obs_frac = np.mean(np.isfinite(samples), axis=0)  # how often state survives

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x = np.arange(n_labels)
    yerr = np.vstack([np.clip(pi_ref - lo, 0, None), np.clip(hi - pi_ref, 0, None)])
    ax.bar(x, pi_ref, width=0.62, color=C_BLUE, edgecolor="none", label="π (full data)")
    ax.errorbar(x, pi_ref, yerr=yerr, fmt="none", ecolor="#333333",
                elinewidth=1, capsize=2.5, label="95% CI (block bootstrap)")
    dropped = [i for i in range(n_labels) if i not in sym]
    for i in dropped:
        ax.plot(i, 0.004, marker="x", color=C_VERM, ms=6)
    if dropped:
        ax.plot([], [], marker="x", ls="none", color=C_VERM, label="disconnected state")
    ax.set(xlabel="cluster state", ylabel="stationary probability π",
           title=f"{pid} — π with bootstrap 95% CIs (n={n_boot}, block={block})")
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(FIG / f"{pid}_pi_ci.png")
    plt.close(fig)

    width = hi - lo
    return {"pi": [round(float(v), 4) for v in pi_ref],
            "ci_low": [None if not np.isfinite(v) else round(float(v), 4) for v in lo],
            "ci_high": [None if not np.isfinite(v) else round(float(v), 4) for v in hi],
            "state_survival_fraction": [round(float(v), 3) for v in obs_frac],
            "median_ci_width": round(float(np.nanmedian(width)), 4),
            "median_ci_width_over_pi": round(float(np.nanmedian(
                width[sym] / np.maximum(pi_ref[sym], 1e-9))), 3)}


# ---------------------------------------------------------------------------
def main():
    cfg = {
        "1VII": (ROOT / "fixed_artifacts/1VII/dtraj.npy", 5),
        "8H0R": (ROOT / "fixed_artifacts/8H0R/dtraj.npy", 5),
        "1UBQ": (ROOT / "fixed_artifacts/1UBQ/dtraj.npy", 5),
        "1CRN": (ROOT / "artifacts/1CRN/dtraj.npy", 5),
    }
    for pid, (path, lag) in cfg.items():
        if not path.exists():
            continue
        dtraj = np.load(path)
        print(f"[{pid}] T={len(dtraj)}")
        RESULTS[pid] = {
            "implied_timescales": run_its(pid, dtraj, lag),
            "chapman_kolmogorov": run_ck(pid, dtraj, lag),
            "vamp2_grid": run_vamp2(pid),
            "bootstrap_pi": run_bootstrap(pid, dtraj, lag),
        }
    out = ROOT / "validation/layer2_results.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
