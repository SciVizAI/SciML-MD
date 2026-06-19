#!/usr/bin/env python3
"""
Batch ML pipeline runner for all protein datasets in data/.

For every sub-directory under data/ that contains a topology.pdb **and** a
trajectory file (traj.xtc / traj.dcd / trajectory.xtc / trajectory.dcd),
this script runs the full anomaly-detection pipeline:

    1. Load trajectory & compute MD features
    2. Run tICA (TICA dimensionality reduction)
    3. Cluster conformational states (KMeans)
    4. Build a Markov State Model (MSM)
    5. Compute per-frame and per-residue anomaly signals

Outputs per protein
-------------------
    artifacts/{PDB_ID}/tica_coords.npy
    artifacts/{PDB_ID}/dtraj.npy
    artifacts/{PDB_ID}/P.npy
    artifacts/{PDB_ID}/pi.npy
    results/{PDB_ID}/frame_scores_dynamic.csv
    results/{PDB_ID}/residue_scores_dynamic.json

Usage
-----
    python run_all_proteins.py
    python run_all_proteins.py --data_dir data --lag_tica 10 --n_clusters 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Accepted trajectory file names (searched in order)
# ---------------------------------------------------------------------------
TRAJ_NAMES = ("traj.xtc", "traj.dcd", "trajectory.xtc", "trajectory.dcd")


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def find_protein_dirs(data_dir):
    """
    Return all sub-directories of *data_dir* that are valid protein datasets.

    A valid dataset directory must contain:
    - topology.pdb
    - at least one trajectory file matching TRAJ_NAMES

    Args:
        data_dir: Path to the root data directory.

    Returns:
        List of (pdb_id, topology_path, trajectory_path) tuples.
    """
    data_dir = Path(data_dir)
    datasets = []

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue

        topology = subdir / "topology.pdb"
        if not topology.exists():
            continue

        traj = None
        for name in TRAJ_NAMES:
            candidate = subdir / name
            if candidate.exists():
                traj = candidate
                break

        if traj is None:
            log.debug("[%s] No trajectory found — skipping.", subdir.name)
            continue

        datasets.append((subdir.name, topology, traj))

    return datasets


# ---------------------------------------------------------------------------
# ML pipeline steps
# ---------------------------------------------------------------------------

def compute_features(topology_path, trajectory_path, stride=1):
    """
    Step 1 — Extract MD features from a trajectory.

    Delegates to the project's existing feature-extraction module.

    Args:
        topology_path: Path to topology.pdb.
        trajectory_path: Path to the trajectory file.
        stride: Load every *stride*-th frame.

    Returns:
        X: Feature matrix (n_frames × n_features).
        traj: MDTraj trajectory object.
    """
    from features.compute_md_features import (
        compute_features as _compute_features,
        features_to_matrix,
    )

    feats, traj = _compute_features(str(topology_path), str(trajectory_path), stride=stride)
    X, _ = features_to_matrix(feats)
    return X, traj


def run_tica(X, lag=10, dim=5):
    """
    Step 2 — Time-lagged Independent Component Analysis.

    Args:
        X: Feature matrix (T × F).
        lag: TICA lag time in frames.
        dim: Number of tICA components to keep.

    Returns:
        Y: Projected coordinates (T × dim).
        tica_model: Fitted deeptime TICA model.
    """
    from deeptime.decomposition import TICA

    # Clamp lag to avoid exceeding trajectory length
    lag = min(lag, len(X) // 4)
    lag = max(lag, 1)

    tica_model = TICA(lagtime=lag, dim=dim).fit(X).fetch_model()
    Y = tica_model.transform(X)
    return Y, tica_model


def cluster_states(Y, n_clusters=20, seed=42):
    """
    Step 3 — Cluster tICA coordinates into discrete conformational states.

    Args:
        Y: tICA coordinate matrix (T × dim).
        n_clusters: Number of KMeans clusters.
        seed: Random seed for reproducibility.

    Returns:
        dtraj: Discrete trajectory as integer array (T,).
        kmeans_model: Fitted deeptime KMeans model.
    """
    from deeptime.clustering import KMeans

    # Clamp cluster count to number of unique frames
    n_clusters = min(n_clusters, len(Y) // 2)
    n_clusters = max(n_clusters, 2)

    kmeans_model = (
        KMeans(n_clusters=n_clusters, max_iter=100, n_jobs=1, fixed_seed=seed)
        .fit(Y)
        .fetch_model()
    )
    dtraj = kmeans_model.transform(Y).astype(np.int64)
    return dtraj, kmeans_model


def build_msm(dtraj, lag=10):
    """
    Step 4 — Build a reversible Maximum Likelihood MSM.

    Args:
        dtraj: Discrete trajectory (T,).
        lag: MSM lag time in frames.

    Returns:
        msm: Fitted deeptime MarkovStateModel.
        P: Transition matrix (n_states × n_states).
        pi: Stationary distribution (n_states,).
    """
    from deeptime.markov.msm import MaximumLikelihoodMSM

    # Clamp lag
    lag = min(lag, len(dtraj) // 4)
    lag = max(lag, 1)

    msm = MaximumLikelihoodMSM(lagtime=lag, reversible=True).fit(dtraj).fetch_model()
    P = msm.transition_matrix
    pi = msm.stationary_distribution
    return msm, P, pi


def compute_anomaly_signals(msm, dtraj, Y, lag_msm=10, k_neighbors=10, window=5):
    """
    Step 5 — Compute per-frame anomaly scores using kinetic + density signals.

    Args:
        msm: Fitted MSM.
        dtraj: Discrete trajectory.
        Y: tICA coordinate matrix.
        lag_msm: MSM lag for transition surprise.
        k_neighbors: Neighbours for local density.
        window: Window size for smoothing.

    Returns:
        frame_scores: Per-frame anomaly score in [0, 100].
        components: Dict of normalised individual signal arrays.
    """
    from scoring.anomaly_v2 import (
        compute_kinetic_signals,
        compute_local_density_signal,
        fuse_signals,
        moving_median,
    )

    # Clamp lag
    lag_msm = min(lag_msm, len(dtraj) // 4)
    lag_msm = max(lag_msm, 1)

    rarity, surprise = compute_kinetic_signals(msm, dtraj, lag_msm)
    density = compute_local_density_signal(Y, k=min(k_neighbors, len(Y) - 1))

    signals = {
        "rarity": rarity,
        "transition_surprise": surprise,
        "local_density": -density,  # invert so low density → high score
    }

    score_raw, components = fuse_signals(signals, method="median", normalize_method="rank")
    score_100 = score_raw * 100.0
    frame_scores = moving_median(score_100, window=window)

    return frame_scores, components


def compute_residue_scores(traj, frame_scores):
    """
    Aggregate per-frame anomaly scores to per-residue scores.

    Uses RMSF (root-mean-square fluctuation) as a proxy for structural
    flexibility and combines it with the mean per-frame anomaly score to
    produce a per-residue anomaly estimate.

    Args:
        traj: MDTraj trajectory object.
        frame_scores: Per-frame anomaly scores (T,).

    Returns:
        residue_scores: Dict mapping residue index → score.
    """
    try:
        import mdtraj as md

        ca_idx = traj.topology.select("name CA")
        if len(ca_idx) == 0:
            ca_idx = traj.topology.select("protein")

        if len(ca_idx) == 0:
            return {}

        # RMSF of CA atoms (one per residue, selected via "name CA")
        ca_traj = traj.atom_slice(ca_idx)
        ca_traj = ca_traj.superpose(ca_traj)
        mean_xyz = ca_traj.xyz.mean(axis=0)
        rmsf = np.sqrt(((ca_traj.xyz - mean_xyz) ** 2).sum(axis=-1).mean(axis=0))
        rmsf_norm = rmsf / (rmsf.max() + 1e-12)

        # Mean frame score (already in [0,100])
        mean_frame_score = frame_scores.mean() / 100.0

        residue_scores = {}
        for i, atom_idx in enumerate(ca_idx):
            atom = traj.topology.atom(atom_idx)
            res = atom.residue
            # Blend RMSF weight with mean frame anomaly
            combined = float(0.5 * rmsf_norm[i] + 0.5 * mean_frame_score) * 100.0
            residue_scores[str(res)] = round(combined, 4)

        return residue_scores

    except Exception as exc:
        log.warning("Residue score computation failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Per-protein pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    pdb_id,
    topology_path,
    trajectory_path,
    artifacts_dir,
    results_dir,
    stride=1,
    lag_tica=10,
    dim_tica=5,
    n_clusters=20,
    lag_msm=10,
    k_neighbors=10,
    window=5,
    seed=42,
):
    """
    Run the complete ML pipeline for a single protein.

    Args:
        pdb_id: PDB identifier string (used for logging and directory names).
        topology_path: Path to topology.pdb.
        trajectory_path: Path to the trajectory file.
        artifacts_dir: Root directory for numpy artifacts.
        results_dir: Root directory for CSV/JSON results.
        stride: Trajectory stride.
        lag_tica: tICA lag time.
        dim_tica: tICA dimensions.
        n_clusters: KMeans clusters.
        lag_msm: MSM lag time.
        k_neighbors: k-NN for local density.
        window: Smoothing window.
        seed: Random seed.

    Returns:
        True on success, False on failure.
    """
    art_dir = Path(artifacts_dir) / pdb_id
    res_dir = Path(results_dir) / pdb_id
    art_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    log.info("[%s] ── Step 1/5: Feature extraction", pdb_id)
    try:
        X, traj = compute_features(topology_path, trajectory_path, stride=stride)
    except Exception as exc:
        log.error("[%s] Feature extraction failed: %s", pdb_id, exc)
        return False

    n_frames, n_feats = X.shape
    log.info("[%s]   %d frames × %d features", pdb_id, n_frames, n_feats)

    if n_frames < 10:
        log.warning("[%s] Too few frames (%d) — skipping.", pdb_id, n_frames)
        return False

    log.info("[%s] ── Step 2/5: tICA (lag=%d, dim=%d)", pdb_id, lag_tica, dim_tica)
    try:
        Y, _ = run_tica(X, lag=lag_tica, dim=dim_tica)
    except Exception as exc:
        log.error("[%s] tICA failed: %s", pdb_id, exc)
        return False

    np.save(art_dir / "tica_coords.npy", Y)
    log.info("[%s]   tICA shape: %s", pdb_id, Y.shape)

    log.info("[%s] ── Step 3/5: Clustering (n_clusters=%d)", pdb_id, n_clusters)
    try:
        dtraj, _ = cluster_states(Y, n_clusters=n_clusters, seed=seed)
    except Exception as exc:
        log.error("[%s] Clustering failed: %s", pdb_id, exc)
        return False

    np.save(art_dir / "dtraj.npy", dtraj)
    log.info("[%s]   Unique states: %d", pdb_id, len(np.unique(dtraj)))

    log.info("[%s] ── Step 4/5: MSM (lag=%d)", pdb_id, lag_msm)
    try:
        msm, P, pi = build_msm(dtraj, lag=lag_msm)
    except Exception as exc:
        log.error("[%s] MSM failed: %s", pdb_id, exc)
        return False

    np.save(art_dir / "P.npy", P)
    np.save(art_dir / "pi.npy", pi)
    log.info("[%s]   MSM states: %d", pdb_id, msm.n_states)

    log.info("[%s] ── Step 5/5: Anomaly scoring", pdb_id)
    try:
        frame_scores, components = compute_anomaly_signals(
            msm, dtraj, Y, lag_msm=lag_msm, k_neighbors=k_neighbors, window=window
        )
    except Exception as exc:
        log.error("[%s] Anomaly scoring failed: %s", pdb_id, exc)
        return False

    # --- Save frame scores ---
    import pandas as pd

    scores_df = pd.DataFrame(
        {
            "frame": np.arange(len(frame_scores)),
            "score_dynamic": frame_scores,
            **{f"component_{k}": v * 100.0 for k, v in components.items()},
        }
    )
    frame_csv = res_dir / "frame_scores_dynamic.csv"
    scores_df.to_csv(frame_csv, index=False)
    log.info("[%s]   Frame scores → %s", pdb_id, frame_csv)

    # --- Save residue scores ---
    residue_scores = compute_residue_scores(traj, frame_scores)
    residue_json = res_dir / "residue_scores_dynamic.json"
    with open(residue_json, "w") as fh:
        json.dump(residue_scores, fh, indent=2)
    log.info("[%s]   Residue scores → %s", pdb_id, residue_json)

    log.info(
        "[%s] ✓ Pipeline complete. Mean score: %.1f",
        pdb_id,
        float(frame_scores.mean()),
    )
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the full ML anomaly-detection pipeline for all proteins in data/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Root data directory containing {PDB_ID}/topology.pdb subdirectories",
    )
    parser.add_argument(
        "--artifacts_dir",
        default="artifacts",
        help="Directory for numpy artifacts (tica_coords, dtraj, P, pi)",
    )
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Directory for CSV/JSON results",
    )
    parser.add_argument("--stride", type=int, default=1, help="Trajectory stride")
    parser.add_argument(
        "--lag_tica", type=int, default=10, help="tICA lag time (frames)"
    )
    parser.add_argument(
        "--dim_tica", type=int, default=5, help="Number of tICA components"
    )
    parser.add_argument(
        "--n_clusters", type=int, default=20, help="Number of KMeans clusters"
    )
    parser.add_argument(
        "--lag_msm", type=int, default=10, help="MSM lag time (frames)"
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=10,
        help="k for k-NN local density signal",
    )
    parser.add_argument(
        "--window", type=int, default=5, help="Smoothing window for anomaly scores"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--protein",
        default=None,
        help="Process only this PDB ID (useful for debugging)",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Batch ML Pipeline — Ensemble Anomaly Maps")
    log.info("=" * 60)

    datasets = find_protein_dirs(args.data_dir)

    if not datasets:
        log.error(
            "No valid protein datasets found under '%s'. "
            "Each dataset must have topology.pdb and a trajectory file.",
            args.data_dir,
        )
        return 1

    if args.protein:
        datasets = [(pid, top, traj) for pid, top, traj in datasets if pid == args.protein]
        if not datasets:
            log.error("Protein '%s' not found in '%s'.", args.protein, args.data_dir)
            return 1

    log.info("Found %d protein dataset(s) to process.", len(datasets))

    n_ok = 0
    n_fail = 0
    failed_ids = []

    for i, (pdb_id, topology, trajectory) in enumerate(datasets, 1):
        log.info("")
        log.info("─" * 60)
        log.info("[%d/%d] Processing: %s", i, len(datasets), pdb_id)
        log.info("─" * 60)

        ok = run_pipeline(
            pdb_id=pdb_id,
            topology_path=topology,
            trajectory_path=trajectory,
            artifacts_dir=args.artifacts_dir,
            results_dir=args.results_dir,
            stride=args.stride,
            lag_tica=args.lag_tica,
            dim_tica=args.dim_tica,
            n_clusters=args.n_clusters,
            lag_msm=args.lag_msm,
            k_neighbors=args.k_neighbors,
            window=args.window,
            seed=args.seed,
        )

        if ok:
            n_ok += 1
        else:
            n_fail += 1
            failed_ids.append(pdb_id)

    log.info("")
    log.info("=" * 60)
    log.info("Batch run complete: %d succeeded, %d failed.", n_ok, n_fail)
    if failed_ids:
        log.warning("Failed proteins: %s", ", ".join(failed_ids))
    log.info("Artifacts → %s/", args.artifacts_dir)
    log.info("Results   → %s/", args.results_dir)
    log.info("=" * 60)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
