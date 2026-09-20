#!/usr/bin/env python3
"""
Phase 2 — temporal-scrambling test: does the kinetic channel detect anomalies
that are INVISIBLE to geometry?

MOTIVATION
----------
Phase 1 seeded a THERMODYNAMIC anomaly (expanded 450 K conformations). A trivial
RMSD-from-mean baseline scored AUROC 1.000 and the pipeline 0.831 - the method
lost, because that is a case geometry owns.

This test is the mirror image, and targets the only claim the method can uniquely
support: a KINETICALLY improbable transition between STRUCTURALLY ORDINARY states.

Construction: every frame is a real 300 K frame from the same trajectory. Nothing
is out of distribution. We only alter WHICH FRAME FOLLOWS WHICH, by splicing the
trajectory at chosen junctions. Ground truth = the junction positions, defined by
the splice operation and never exposed to the model.

By construction, every time-blind detector (RMSD-from-mean, feature z-score,
Isolation Forest, LOF, kNN density, state rarity) must sit at chance: it sees a
bag of perfectly normal frames. Only a method that models DYNAMICS can score.

TWO SCRAMBLE MODES
------------------
far   : blocks shuffled arbitrarily. Junctions have LARGE geometric jumps, so a
        naive frame-to-frame RMSD detector should also find them. Included to
        show the method is not the only option here - honesty requires it.
near  : junctions chosen between frame pairs that are geometrically CLOSE
        (bottom decile of pairwise RMSD) but temporally distant. The geometric
        jump is small, so the naive jump detector should struggle. This is the
        discriminating case: if the kinetic channel wins here, it is detecting
        something no geometric method can.

BASELINES include `frame_to_frame_rmsd` - the naive temporal detector. If that
beats the MSM, the honest conclusion is that an MSM is unnecessary for this task.

PRE-REGISTERED ACCEPTANCE CRITERIA (fixed before running)
  B1  surprise AUROC >= 0.75 on NEAR-mode junctions
  B2  surprise > the BEST trivial temporal detector on NEAR mode, on ROC-AUC
      AND on PR-AUC. The family is TRIVIAL_TEMPORAL = {abs_diff_oneliner,
      frame_to_frame_rmsd}.  [widened 2026-08-30, OI-27 - strictly stricter]
  B3  every time-blind baseline within 0.40-0.60 on NEAR mode
      (sanity check: confirms the frames really are in-distribution)

Outputs: validation/phase2_results.json
"""
import json
import sys
import zlib
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

DATA = ROOT / "validation/phase1_data"
TOP = ROOT / "data/1VII/topology.pdb"

N_JUNCTIONS = 8
LAG_MSM = 5
N_REPLICATES = 12
TIME_BLIND = ["rmsd_from_mean", "feature_zscore", "isolation_forest",
              "local_outlier_factor", "density_only", "rarity_only"]
# Reported and gated, but NOT part of the time-blind family used for C2/C3:
#   random_score      is a null, not a baseline - it defines chance
#   abs_diff_oneliner is a TEMPORAL detector (it reads frame order), so it
#                     belongs with frame_to_frame_rmsd, not with the static set
NULL_CHANNELS = ["random_score"]
TRIVIAL_TEMPORAL = ["abs_diff_oneliner", "frame_to_frame_rmsd"]
ACCEPT = {"B1_min_surprise_auroc": 0.75, "B3_band": (0.40, 0.60)}


# ---------------------------------------------------------------------------
def scramble(traj, mode, rng, rmsd_matrix=None):
    """Reorder frames; return (new_order, junction_positions).

    A junction at position p means new_order[p] -> new_order[p+1] is a splice,
    i.e. those two frames were NOT adjacent in the original trajectory.
    """
    import mdtraj as md

    n = len(traj)
    n_blocks = N_JUNCTIONS + 1
    # Cut points are RANDOMISED per replicate (min block length 12) so the
    # labelled frames are not the same fixed subset every time - otherwise a
    # time-blind detector can score merely because those particular frames
    # happen to be distinctive.
    min_len = 12
    while True:
        cuts = np.sort(rng.choice(np.arange(min_len, n - min_len), n_blocks - 1,
                                  replace=False))
        edges = np.concatenate([[0], cuts, [n]])
        if np.all(np.diff(edges) >= min_len):
            break
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_blocks)]

    # CRITICAL: both modes are PERMUTATIONS of the blocks, so every frame is used
    # exactly once. The multiset of frames — and therefore the entire static
    # ensemble — is identical to the original. Any time-blind detector is at
    # chance BY CONSTRUCTION; if one scores, the test is broken, not the method.
    if mode == "far":
        perm = rng.permutation(n_blocks)

    elif mode == "near":
        # Greedy nearest-endpoint ordering: join each block to the unused block
        # whose FIRST frame is structurally closest to the current block's LAST
        # frame. Junctions are therefore geometrically SMOOTH — a naive
        # frame-to-frame jump detector has almost nothing to see, and only a
        # model of the dynamics can flag them.
        assert rmsd_matrix is not None
        R = rmsd_matrix
        start = int(rng.integers(n_blocks))
        perm, remaining = [start], set(range(n_blocks)) - {start}
        while remaining:
            tail = blocks[perm[-1]][-1]
            nxt = min(remaining, key=lambda b: R[tail, blocks[b][0]])
            perm.append(nxt)
            remaining.discard(nxt)
        perm = np.array(perm)
    elif mode == "identity_null":
        # NULL CONTROL: blocks kept in original order, so there are NO real
        # junctions. We still label the same positions. Every method must be at
        # chance; anything that scores is responding to block position, not to
        # any anomaly.
        perm = np.arange(n_blocks)
    else:
        raise ValueError(mode)

    order = np.concatenate([blocks[i] for i in perm])

    if mode == "identity_null":
        junctions = list(np.cumsum([len(blocks[i]) for i in perm])[:-1] - 1)
    else:
        junctions = [p for p in range(len(order) - 1)
                     if order[p + 1] != order[p] + 1]
    return order, junctions


def junction_labels(n_frames, junctions, lag):
    """surprise[t] compares frames t and t+lag, so a junction at p is 'visible'
    to frames t in [p-lag+1, p]. Label exactly that window."""
    y = np.zeros(n_frames, dtype=int)
    for p in junctions:
        lo, hi = max(0, p - lag + 1), min(n_frames - 1, p)
        y[lo:hi + 1] = 1
    return y


# ---------------------------------------------------------------------------
def score_all(traj, lag_tica=5, dim=3, n_clusters=10, lag_msm=LAG_MSM, k=5, seed=42):
    import mdtraj as md
    from features.compute_md_features import features_to_matrix
    from run_all_proteins import run_tica, cluster_states, build_msm
    from scoring.anomaly_v2 import (compute_kinetic_signals,
                                    compute_local_density_signal, fuse_signals)

    feats = {}
    feats["rmsd"] = md.rmsd(traj, traj, 0)
    feats["rg"] = md.compute_rg(traj)
    ca = traj.topology.select("name CA")
    ii, jj = np.triu_indices(len(ca), k=1)
    d = md.compute_distances(traj, np.stack([ca[ii], ca[jj]], axis=1))
    feats["contacts"] = (d < 0.8).sum(axis=1).astype(float)
    _, phi = md.compute_phi(traj)
    _, psi = md.compute_psi(traj)
    feats["phi_sin"] = np.sin(phi).mean(axis=1)
    feats["phi_cos"] = np.cos(phi).mean(axis=1)
    feats["psi_sin"] = np.sin(psi).mean(axis=1)
    feats["psi_cos"] = np.cos(psi).mean(axis=1)
    X, _ = features_to_matrix(feats)

    Y, _ = run_tica(X, lag=lag_tica, dim=dim)
    dtraj, _ = cluster_states(Y, n_clusters=n_clusters, seed=seed)
    msm, _, _ = build_msm(dtraj, lag=lag_msm)
    rar, sur = compute_kinetic_signals(msm, dtraj, lag_msm)
    dens = compute_local_density_signal(Y, k=min(k, len(Y) - 1))
    fused, _ = fuse_signals({"rarity": rar, "transition_surprise": sur,
                             "local_density": -dens})

    sup = traj.superpose(traj, 0)
    mean_xyz = sup.xyz.mean(axis=0)
    rmsd_mean = np.sqrt(((sup.xyz - mean_xyz) ** 2).sum(-1).mean(-1))
    z = np.abs((X - X.mean(0)) / (X.std(0) + 1e-12)).max(axis=1)

    # naive TEMPORAL baseline: geometric jump between consecutive frames,
    # aligned to the same lag window the surprise channel uses
    step = np.zeros(len(traj))
    step[:-1] = md.rmsd(traj[1:], traj[:-1]) if len(traj) > 1 else 0
    jump_lagged = np.zeros(len(traj))
    for t in range(len(traj) - lag_msm):
        jump_lagged[t] = step[t:t + lag_msm].max()

    # literal abs(diff) one-liner on the standardised feature matrix, max over
    # features, aligned to the same lag window the surprise channel uses
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
    step_feat = np.zeros(len(traj))
    step_feat[:-1] = np.abs(np.diff(Xz, axis=0)).max(axis=1)
    abs_diff_oneliner = np.zeros(len(traj))
    for t in range(len(traj) - lag_msm):
        abs_diff_oneliner[t] = step_feat[t:t + lag_msm].max()

    iso = -IsolationForest(random_state=seed, n_estimators=200).fit(Y).score_samples(Y)
    lof = -LocalOutlierFactor(n_neighbors=min(20, len(Y) - 1)).fit(Y).negative_outlier_factor_

    return {
        "surprise_only": sur,
        "pipeline_fused": fused * 100,
        "rarity_only": rar,
        "density_only": -dens,
        "frame_to_frame_rmsd": jump_lagged,
        "rmsd_from_mean": rmsd_mean,
        "feature_zscore": z,
        "isolation_forest": iso,
        "local_outlier_factor": lof,
        # --- baselines demanded by the anomaly-detection benchmarking
        # --- literature (added 2026-08-29 after the standards audit)
        #
        # NULL. Seeded from the COORDINATES, not from `seed` alone. The first
        # version used default_rng(seed) with seed fixed at 42, so every
        # replicate and every protein received the IDENTICAL vector - one
        # arbitrary fixed ranking, not a random baseline. It duly picked up
        # positional structure (0.614 at Phase 5 D=1). crc32 of a coordinate
        # subsample varies with the permutation and stays reproducible.
        #
        # Kim et al. (AAAI 2022) proved that under common evaluation
        # protocols uniformly random scores can reach near-perfect F1. Reporting
        # what chance looks like under OUR exact protocol is now expected, and
        # its absence reads as an oversight. This must sit at ~0.500; if it does
        # not, the harness is broken and no other number means anything.
        "random_score": np.random.default_rng(
            seed ^ zlib.crc32(traj.xyz[::max(1, len(traj) // 8)].tobytes())
        ).random(len(traj)),
        # TRIVIAL. Wu & Keogh (IEEE TKDE 2023) showed 86% of the Yahoo benchmark
        # and over half of several others are solved by one-line expressions,
        # `abs(diff(x)) > b` chief among them. frame_to_frame_rmsd is close but
        # is RMSD-based and lag-windowed; this is the literal one-liner applied
        # to the feature series the pipeline itself consumes, so it is the
        # cheapest possible competitor on exactly our inputs.
        "abs_diff_oneliner": abs_diff_oneliner,
    }


def metrics(y, s):
    s = np.asarray(s, float)
    if np.any(~np.isfinite(s)):
        s = np.nan_to_num(s, nan=np.nanmin(s[np.isfinite(s)]))
    return (round(float(roc_auc_score(y, s)), 3),
            round(float(average_precision_score(y, s)), 3))


# ---------------------------------------------------------------------------
def main():
    import mdtraj as md

    traj = md.load(str(DATA / "normal_300K.xtc"), top=str(TOP))
    print(f"source: {len(traj)} frames, single 300 K ensemble (nothing out of distribution)")

    # pairwise RMSD for the 'near' construction
    R = np.zeros((len(traj), len(traj)))
    for i in range(len(traj)):
        R[i] = md.rmsd(traj, traj, i)

    results = {"n_frames": len(traj), "n_junctions": N_JUNCTIONS,
               "replicates": N_REPLICATES, "modes": {}}

    for mode in ("far", "near", "identity_null"):
        per = {}
        jump_sizes = []
        for rep in range(N_REPLICATES):
            rng = np.random.default_rng(500 + rep)
            order, junc = scramble(traj, mode, rng, R)
            new = traj[order]
            y = junction_labels(len(new), junc, LAG_MSM)
            if y.sum() == 0:
                continue
            jump_sizes += [float(R[order[p], order[p + 1]]) for p in junc]
            sc = score_all(new)
            for name, s in sc.items():
                au, ap = metrics(y, s)
                per.setdefault(name, {"auroc": [], "auprc": []})
                per[name]["auroc"].append(au)
                per[name]["auprc"].append(ap)
        summary = {n: {"auroc_mean": round(float(np.mean(v["auroc"])), 3),
                       "auroc_std": round(float(np.std(v["auroc"])), 3),
                       "auprc_mean": round(float(np.mean(v["auprc"])), 3)}
                   for n, v in per.items()}
        results["modes"][mode] = {
            "median_junction_rmsd_nm": round(float(np.median(jump_sizes)), 4),
            "methods": summary}
        print(f"\n[{mode}]  median geometric jump at junctions: "
              f"{np.median(jump_sizes):.4f} nm")
        for n in sorted(summary, key=lambda k: -summary[k]["auroc_mean"]):
            tag = "  (time-blind)" if n in TIME_BLIND else ""
            print(f"   {n:22s} AUROC {summary[n]['auroc_mean']:.3f} "
                  f"+/- {summary[n]['auroc_std']:.3f}{tag}")

    near = results["modes"]["near"]["methods"]
    b1 = near["surprise_only"]["auroc_mean"] >= ACCEPT["B1_min_surprise_auroc"]
    # OI-27: B2 used to race only frame_to_frame_rmsd. It now races the best of
    # the whole trivial temporal family, on ROC AND on PR-AUC. Strictly
    # stricter (max over a superset), so it cannot let a weaker result through.
    triv = {n: near[n] for n in TRIVIAL_TEMPORAL if n in near}
    best_triv = max(triv, key=lambda n: triv[n]["auroc_mean"]) if triv else None
    best_triv_ap = (max(triv, key=lambda n: triv[n].get("auprc_mean", 0.0))
                    if triv else None)
    b2 = bool(best_triv is not None
              and near["surprise_only"]["auroc_mean"] > triv[best_triv]["auroc_mean"]
              and near["surprise_only"].get("auprc_mean", 0.0)
              > triv[best_triv_ap].get("auprc_mean", 0.0))
    lo, hi = ACCEPT["B3_band"]
    blind = {n: near[n]["auroc_mean"] for n in TIME_BLIND if n in near}
    b3 = all(lo <= v <= hi for v in blind.values())
    nullm = results["modes"]["identity_null"]["methods"]
    b4 = 0.40 <= nullm["surprise_only"]["auroc_mean"] <= 0.60
    results["acceptance"] = {
        "B4_identity_null_at_chance": {"pass": bool(b4),
                                       "surprise": nullm["surprise_only"]["auroc_mean"]},
        "B1_surprise_near_auroc_ge_0.75": {"pass": bool(b1),
                                           "value": near["surprise_only"]["auroc_mean"]},
        "B2_beats_best_trivial_temporal": {
            "pass": bool(b2),
            "surprise_auroc": near["surprise_only"]["auroc_mean"],
            "surprise_prauc": near["surprise_only"].get("auprc_mean"),
            "trivial_family": {n: {"auroc": triv[n]["auroc_mean"],
                                   "prauc": triv[n].get("auprc_mean")}
                               for n in triv},
            "note": ("OI-27, 2026-08-30: widened from frame_to_frame_rmsd alone "
                     "to the whole trivial temporal family, and now requires a "
                     "win on PR-AUC as well as ROC-AUC.")},
        "B3_time_blind_at_chance": {"pass": bool(b3), "values": blind, "band": [lo, hi]},
        "overall_pass": bool(b1 and b2 and b3 and b4),
    }
    print("\nACCEPTANCE:", json.dumps(results["acceptance"], indent=2))
    (ROOT / "validation/phase2_results.json").write_text(json.dumps(results, indent=2))
    print("\nsaved -> validation/phase2_results.json")


if __name__ == "__main__":
    main()
