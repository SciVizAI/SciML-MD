#!/usr/bin/env python3
"""
Layer 3 — external scientific validation.

Part A: Experimental B-factor comparison.
  Per-residue CA B-factors are parsed from the RAW RCSB PDB (experimental
  temperature factors) and correlated (Spearman) against:
    - the pipeline's per-residue dynamic score (fixed code),
    - trajectory RMSF (residue_scores_rmsf.json).
  Only X-ray / cryo-EM entries carry meaningful B-factors (1VII is NMR ->
  excluded by design).

Part B: Baseline method comparison (frame level).
  The pipeline's fused score vs off-the-shelf anomaly detectors run in the
  SAME tICA space: IsolationForest, LocalOutlierFactor, and the raw k-NN
  distance (the pipeline's own density channel, as ablation). Evaluated
  against the operational kinetic ground truth of baseline_rare_case.py
  (bottom-quartile-pi states + disconnected; transitions observed <= 1 time).

Outputs: validation/layer3_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scoring.anomaly_v2 import (  # noqa: E402
    compute_kinetic_signals, compute_local_density_signal, fuse_signals, moving_median,
)

RAW_PDB = {
    # protein -> raw RCSB pdb with experimental B-factors
    "8H0R": ROOT / "raw_pdb/8H0R.pdb",
    "1UBQ": ROOT / "raw_pdb/1UBQ.pdb",
    "9O6O": ROOT / "raw_pdb/9O6O.pdb",
    "9UNN": ROOT / "raw_pdb/9UNN.pdb",
}
METHOD = {"8H0R": "X-ray 1.20 A", "1UBQ": "X-ray 1.80 A",
          "9O6O": "X-ray 2.70 A", "9UNN": "cryo-EM 3.29 A"}


# ---------------------------------------------------------------------------
# Part A helpers
# ---------------------------------------------------------------------------
def parse_ca_bfactors(pdb_path):
    """(chain_order_idx, resSeq, resName) -> CA B-factor, first model only."""
    out = {}
    chain_ids = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            ch = line[21]
            if ch not in chain_ids:
                chain_ids.append(ch)
            resseq = int(line[22:26])
            resname = line[17:20].strip()
            b = float(line[60:66])
            out[(chain_ids.index(ch), resseq, resname)] = b
    return out


def score_key_to_tuple(key):
    """'MET41' or 'MET41_chain1' -> (chain_idx or None, resSeq, resName)."""
    chain = None
    if "_chain" in key:
        key, c = key.split("_chain")
        chain = int(c)
    name = "".join(ch for ch in key if ch.isalpha())
    seq = int("".join(ch for ch in key if ch.isdigit()))
    return chain, seq, name


def bfactor_comparison(pid, results_dir):
    res = {"method": METHOD.get(pid, "?")}
    pdb = RAW_PDB[pid]
    if not pdb.exists():
        return {"skipped": f"raw pdb not present: {pdb.name}"}
    bfac = parse_ca_bfactors(pdb)

    dyn_p = results_dir / pid / "residue_scores_dynamic.json"
    rmsf_p = results_dir / pid / "residue_scores_rmsf.json"
    if not dyn_p.exists():
        return {"skipped": "no pipeline results yet"}
    dyn = json.load(open(dyn_p))
    rmsf = json.load(open(rmsf_p)) if rmsf_p.exists() else {}

    pairs_dyn, pairs_rmsf, missed = [], [], 0
    for key, score in dyn.items():
        ch, seq, name = score_key_to_tuple(key)
        cand = [(c, seq, name) for c in ([ch] if ch is not None else range(8))]
        b = next((bfac[c] for c in cand if c in bfac), None)
        if b is None:
            missed += 1
            continue
        pairs_dyn.append((b, float(score)))
        if key in rmsf:
            pairs_rmsf.append((b, float(rmsf[key])))

    res["n_matched"] = len(pairs_dyn)
    res["n_unmatched"] = missed
    if len(pairs_dyn) >= 10:
        b, s = np.array(pairs_dyn).T
        res["spearman_B_vs_dynamic_score"] = round(float(spearmanr(b, s).statistic), 3)
    if len(pairs_rmsf) >= 10:
        b, s = np.array(pairs_rmsf).T
        res["spearman_B_vs_traj_RMSF"] = round(float(spearmanr(b, s).statistic), 3)
    return res


# ---------------------------------------------------------------------------
# Part B helpers (reuse operational labels from baseline_rare_case)
# ---------------------------------------------------------------------------
def refit(dtraj, lag):
    from deeptime.markov.msm import MaximumLikelihoodMSM
    lag = max(1, min(lag, len(dtraj) // 4))
    msm = MaximumLikelihoodMSM(lagtime=lag, reversible=True).fit(dtraj).fetch_model()
    symbols = np.asarray(msm.count_model.state_symbols)
    lookup = -np.ones(int(dtraj.max()) + 1, dtype=np.int64)
    for i, s in enumerate(symbols):
        lookup[s] = i
    return msm, lookup[dtraj], lag


def labels(msm, da, dtraj_raw, lag):
    pi, ns = msm.stationary_distribution, msm.n_states
    n = len(da)
    q25 = np.quantile(pi, 0.25)
    rare_states = set(np.where(pi <= q25)[0].tolist())
    L_state = np.array([1 if (da[t] == -1 or da[t] in rare_states) else 0
                        for t in range(n)])
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


def auroc(y, s):
    s = np.asarray(s, dtype=float)
    if np.any(~np.isfinite(s)):
        s = np.nan_to_num(s, nan=np.nanmin(s[np.isfinite(s)]))
    if y.sum() in (0, len(y)):
        return None
    return round(float(roc_auc_score(y, s)), 3)


def auprc(y, s):
    s = np.asarray(s, dtype=float)
    if np.any(~np.isfinite(s)):
        s = np.nan_to_num(s, nan=np.nanmin(s[np.isfinite(s)]))
    if y.sum() in (0, len(y)):
        return None
    return round(float(average_precision_score(y, s)), 3)


def baseline_comparison(pid, art_dir, lag=5, k=5, window=3, seed=42):
    dtraj = np.load(art_dir / "dtraj.npy")
    Y = np.load(art_dir / "tica_coords.npy")
    msm, da, lag_eff = refit(dtraj, lag)
    L_state, L_trans = labels(msm, da, dtraj, lag_eff)

    # Pipeline (fixed production path)
    r, s = compute_kinetic_signals(msm, dtraj, lag_eff)
    d = compute_local_density_signal(Y, k=min(k, len(Y) - 1))
    raw, _ = fuse_signals({"rarity": r, "transition_surprise": s, "local_density": -d})
    fused = moving_median(raw * 100.0, window=window)

    # Baselines in the same tICA space
    iso = IsolationForest(random_state=seed, n_estimators=200).fit(Y)
    iso_score = -iso.score_samples(Y)  # higher = more anomalous
    lof = LocalOutlierFactor(n_neighbors=min(20, len(Y) - 1))
    lof.fit(Y)
    lof_score = -lof.negative_outlier_factor_
    knn_dist = -d  # pipeline's own density channel (ablation)

    methods = {
        "pipeline_fused (ours)": fused,
        "IsolationForest": iso_score,
        "LocalOutlierFactor": lof_score,
        "kNN_distance_only (density ablation)": knn_dist,
        "rarity_channel_only": r,
        "surprise_channel_only": s,
    }
    out = {"n_frames": len(dtraj)}
    for name, score in methods.items():
        out[name] = {
            "rare_state_AUROC": auroc(L_state, score),
            "rare_state_AUPRC": auprc(L_state, score),
            "rare_trans_AUROC": auroc(L_trans, score),
            "rare_trans_AUPRC": auprc(L_trans, score),
        }
    return out


# ---------------------------------------------------------------------------
def main():
    results = {"part_A_bfactor": {}, "part_B_baselines": {}}

    for pid in ["8H0R", "1UBQ", "9O6O", "9UNN"]:
        results["part_A_bfactor"][pid] = bfactor_comparison(pid, ROOT / "fixed_results")

    for pid, art in {
        "1VII": ROOT / "fixed_artifacts/1VII",
        "8H0R": ROOT / "fixed_artifacts/8H0R",
        "1UBQ": ROOT / "fixed_artifacts/1UBQ",
        "1CRN": ROOT / "artifacts/1CRN",
        "9O6O": ROOT / "fixed_artifacts/9O6O",
        "9UNN": ROOT / "fixed_artifacts/9UNN",
    }.items():
        if (art / "dtraj.npy").exists():
            results["part_B_baselines"][pid] = baseline_comparison(pid, art)

    out = ROOT / "validation/layer3_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
