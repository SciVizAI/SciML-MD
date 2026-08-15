# ============================================================== #
#  Module:      batch_runner.py
#  Description: End-to-end runner — download, MD generation, ML pipeline, ASVS export
#  Author:      Siya Jethliya
#  Copyright (c) 2026 SciVizAI — All rights reserved.
# ============================================================== #

#!/usr/bin/env python3
"""
End-to-end batch runner for ensemble-anomaly-maps.

This beginner-friendly script automates the full pipeline for multiple PDB targets:

  1. Downloads PDB structures from RCSB for the given PDB IDs
     (default: 9UNN, 9O6O, 1CRN).
  2. Generates short "toy" MD trajectories (.xtc) from each PDB using OpenMM
     so the pipeline can run even when no real trajectories are available.
  3. Runs the ensemble-anomaly-maps ML pipeline (tICA → clustering ��� MSM →
     anomaly scoring) per target in isolated output directories.
  4. Exports results into ASVS-compatible JSON via tools/export_for_asvs.py.
  5. Writes everything under a predictable directory tree::

        <work_dir>/
          <PDB_ID>/
            topology.pdb       ← downloaded from RCSB
            traj.xtc           ← toy trajectory from OpenMM
        artifacts/
          <PDB_ID>/            ← numpy intermediates (tica_coords, dtraj, …)
        results/
          <PDB_ID>/            ← CSV / JSON pipeline outputs
        exports/
          <PDB_ID>/            ← ASVS-compatible JSON + topology + trajectory

Usage
-----
    # Run all three default proteins
    python batch_runner.py

    # Override which proteins and where outputs land
    python batch_runner.py --pdb_ids 1CRN 9O6O --work_dir /tmp/batch

    # Skip steps you've already done
    python batch_runner.py --skip_download --skip_md

    # Tune the toy MD (more frames = richer signal but slower)
    python batch_runner.py --md_frames 200 --md_steps_per_frame 500

Dependencies
------------
    pip install numpy scipy pandas mdtraj deeptime scikit-learn
    conda install -c conda-forge openmm pdbfixer
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PDB_IDS = ["9UNN", "9O6O", "1CRN"]
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

# Toy-MD defaults — small enough to finish in < 1 min per protein on CPU
DEFAULT_MD_FRAMES = 100           # number of trajectory frames to save
DEFAULT_MD_STEPS_PER_FRAME = 250  # integration steps between saved frames

# Canonical topology filename — the single source of truth for atom identity.
# Written once after PDBFixer + addHydrogens and used by EVERY downstream stage
# (simulation, trajectory conversion, feature extraction). Nothing downstream
# may reload the raw PDB again.
CANONICAL_TOPOLOGY_NAME = "canonical_topology.pdb"

# Pipeline hyper-parameters (kept small for fast toy runs)
DEFAULT_LAG_TICA = 5
DEFAULT_DIM_TICA = 3
DEFAULT_N_CLUSTERS = 10
DEFAULT_LAG_MSM = 5
DEFAULT_K_NEIGHBORS = 5
DEFAULT_WINDOW = 3


# ---------------------------------------------------------------------------
# Step 1 — Download PDB
# ---------------------------------------------------------------------------


# -------------------------------------------------------------- #
# Function: download_pdb
# -------------------------------------------------------------- #
def download_pdb(pdb_id: str, dest_dir: Path) -> Path:
    """Download a PDB structure from RCSB and return the local path.

    If the file already exists it is reused without re-downloading.

    Args:
        pdb_id:   Four-character PDB identifier (case-insensitive).
        dest_dir: Directory where the .pdb file is written.

    Returns:
        Path to the downloaded (or cached) PDB file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = dest_dir / "topology.pdb"

    if pdb_path.exists():
        log.info("[%s] topology.pdb already present — skipping download.", pdb_id)
        return pdb_path

    url = RCSB_PDB_URL.format(pdb_id=pdb_id.upper())
    log.info("[%s] Downloading from %s …", pdb_id, url)

    try:
        urllib.request.urlretrieve(url, pdb_path)
    except Exception as exc:
        raise RuntimeError(
            f"[{pdb_id}] Failed to download PDB: {exc}\n"
            f"  URL tried: {url}\n"
            "  Check your internet connection or provide topology.pdb manually."
        ) from exc

    log.info("[%s] Saved to %s", pdb_id, pdb_path)
    return pdb_path


# ---------------------------------------------------------------------------
# Step 2 — Generate toy MD trajectory with OpenMM
# ---------------------------------------------------------------------------


# -------------------------------------------------------------- #
# Function: _fix_pdb_with_pdbfixer
# -------------------------------------------------------------- #
def _fix_pdb_with_pdbfixer(pdb_id: str, pdb_path: Path, out_dir: Path) -> Path:
    """
    Use PDBFixer to repair common PDB issues (missing atoms/residues, nonstandard residues,
    missing hydrogens) and write a fixed PDB for OpenMM.

    Returns: Path to fixed PDB.
    """
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError as exc:
        raise RuntimeError(
            "PDBFixer is required to robustly prepare PDBs for OpenMM.\n"
            "Install with: conda install -c conda-forge pdbfixer\n"
            f"Original error: {exc}"
        ) from exc

    fixed_path = out_dir / "topology_fixed.pdb"
    if fixed_path.exists():
        log.info("[%s] topology_fixed.pdb already present — reusing.", pdb_id)
        return fixed_path

    log.info("[%s] Fixing PDB with PDBFixer …", pdb_id)
    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    with open(fixed_path, "w") as f:
        # keepIds=True helps preserve residue numbering where possible
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    log.info("[%s] Wrote fixed topology: %s", pdb_id, fixed_path)
    return fixed_path



# -------------------------------------------------------------- #
# Function: generate_toy_trajectory
# -------------------------------------------------------------- #
def generate_toy_trajectory(
    pdb_id: str,
    pdb_path: Path,
    out_dir: Path,
    n_frames: int = DEFAULT_MD_FRAMES,
    steps_per_frame: int = DEFAULT_MD_STEPS_PER_FRAME,
) -> Path:
    """Generate a short MD trajectory from a PDB file using OpenMM.

    The simulation uses the AMBER14 force field with implicit solvent (GBn2)
    so no periodic box or explicit water is needed — ideal for quick demos.

    A 500-step energy minimization is run before dynamics to remove clashes
    that are common in raw PDB files.

    Args:
        pdb_id:          Protein identifier (used only for logging).
        pdb_path:        Path to the topology PDB file.
        out_dir:         Directory where traj.xtc is written.
        n_frames:        Number of frames to record.
        steps_per_frame: MD steps between consecutive saved frames.

    Returns:
        Path to the generated trajectory (out_dir/traj.xtc).

    Raises:
        RuntimeError: If OpenMM or MDTraj is not installed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    xtc_path = out_dir / "traj.xtc"

    if xtc_path.exists():
        log.info("[%s] traj.xtc already present — skipping MD.", pdb_id)
        return xtc_path

    # ------------------------------------------------------------------
    # Lazy imports so the script is importable even without OpenMM
    # ------------------------------------------------------------------
    try:
        from openmm import app as omm_app
        from openmm import unit
        import openmm as omm
    except ImportError as exc:
        raise RuntimeError(
            "OpenMM is required for trajectory generation.\n"
            "Install it with:  conda install -c conda-forge openmm\n"
            f"Original error: {exc}"
        ) from exc

    try:
        import mdtraj as md
    except ImportError as exc:
        raise RuntimeError(
            "MDTraj is required for trajectory conversion.\n"
            "Install it with:  pip install mdtraj\n"
            f"Original error: {exc}"
        ) from exc

    log.info(
        "[%s] Running toy MD: %d frames × %d steps/frame …",
        pdb_id, n_frames, steps_per_frame,
    )

    # --- Load PDB ---
    pdb = omm_app.PDBFile(str(pdb_path))

    # --- Force field (AMBER14 + implicit GBn2 solvent) ---
    forcefield = omm_app.ForceField("amber14-all.xml", "implicit/gbn2.xml")

    # Build modeller
    modeller = omm_app.Modeller(pdb.topology, pdb.positions)

    # Try to add hydrogens; if it fails, fallback to PDBFixer and retry.
    try:
        modeller.addHydrogens(forcefield)
    except Exception as exc:
        log.warning("[%s] addHydrogens failed (%s) — trying PDBFixer fallback.", pdb_id, exc)
        fixed_pdb = _fix_pdb_with_pdbfixer(pdb_id, pdb_path, out_dir)
        pdb_fixed = omm_app.PDBFile(str(fixed_pdb))
        modeller = omm_app.Modeller(pdb_fixed.topology, pdb_fixed.positions)
        try:
            modeller.addHydrogens(forcefield)
        except Exception as exc2:
            # At this point PDBFixer already added hydrogens; this is non-fatal.
            log.warning("[%s] addHydrogens still failed after PDBFixer (%s) — continuing.", pdb_id, exc2)

    # ── Canonical topology: the ONLY topology valid downstream ──────────────
    # Persist the exact atom set that is about to be simulated (post Fixer +
    # hydrogens). The trajectory is written in this same atom order, and
    # feature extraction loads THIS file — so their atoms are identical by
    # construction. Never pair the trajectory with the raw H-less PDB again.
    canonical_path = out_dir / CANONICAL_TOPOLOGY_NAME
    with open(canonical_path, "w") as fh:
        omm_app.PDBFile.writeFile(
            modeller.topology, modeller.positions, fh, keepIds=True
        )
    log.info(
        "[%s] Canonical topology written: %s (%d atoms)",
        pdb_id, canonical_path, modeller.topology.getNumAtoms(),
    )

    # Create system (IMPORTANT: use modeller.topology; don't proceed with broken raw PDB)
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=omm_app.NoCutoff,
        constraints=omm_app.HBonds,

    )

    # --- Integrator (Langevin, 300 K, 2 fs) ---
    integrator = omm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )
    integrator.setRandomNumberSeed(42)

    # --- Simulation ---
    platform = _best_openmm_platform()
    simulation = omm_app.Simulation(
        modeller.topology, system, integrator, platform
    )
    simulation.context.setPositions(modeller.positions)

    # Energy minimization (removes clashes from raw PDB files)
    log.info("[%s] Minimizing energy …", pdb_id)
    simulation.minimizeEnergy(maxIterations=500)

    # Collect frames into a DCD temp file then convert to XTC with MDTraj
    tmp_fd, tmp_dcd_name = tempfile.mkstemp(suffix=".dcd")
    os.close(tmp_fd)
    dcd_path = Path(tmp_dcd_name)

    try:
        dcd_reporter = omm_app.DCDReporter(str(dcd_path), steps_per_frame)
        simulation.reporters.append(dcd_reporter)

        log.info("[%s] Integrating %d steps …", pdb_id, n_frames * steps_per_frame)
        simulation.step(n_frames * steps_per_frame)

        # Flush reporter
        simulation.reporters.clear()

        # Convert DCD → XTC using MDTraj against the CANONICAL topology only.
        log.info("[%s] Converting DCD → XTC …", pdb_id)
        traj = md.load(str(dcd_path), top=str(canonical_path))

        # Fail fast on any topology/trajectory atom drift.
        n_top = modeller.topology.getNumAtoms()
        if traj.n_atoms != n_top:
            raise RuntimeError(
                f"[{pdb_id}] Topology drift after MD: canonical topology has "
                f"{n_top} atoms but trajectory has {traj.n_atoms}."
            )
        traj.save_xtc(str(xtc_path))
    finally:
        dcd_path.unlink(missing_ok=True)

    log.info("[%s] traj.xtc written (%d frames).", pdb_id, len(traj))
    return xtc_path



# -------------------------------------------------------------- #
# Function: _best_openmm_platform
# -------------------------------------------------------------- #
def _best_openmm_platform():
    """Return the fastest OpenMM Platform available (CUDA > OpenCL > CPU)."""
    try:
        import openmm as omm

        for name in ("CUDA", "OpenCL", "CPU"):
            try:
                platform = omm.Platform.getPlatformByName(name)
                log.debug("OpenMM platform: %s", name)
                return platform
            except Exception:
                continue
        return None  # OpenMM will choose automatically
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step 3 — Run the anomaly-detection pipeline
# ---------------------------------------------------------------------------


# -------------------------------------------------------------- #
# Function: run_pipeline
# -------------------------------------------------------------- #
def run_pipeline(
    pdb_id: str,
    topology_path: Path,
    trajectory_path: Path,
    artifacts_dir: Path,
    results_dir: Path,
    lag_tica: int = DEFAULT_LAG_TICA,
    dim_tica: int = DEFAULT_DIM_TICA,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    lag_msm: int = DEFAULT_LAG_MSM,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    window: int = DEFAULT_WINDOW,
    seed: int = 42,
) -> bool:
    """Run the full ML pipeline for a single protein.

    Delegates to the functions in ``run_all_proteins.py`` so there is no
    code duplication.

    Args:
        pdb_id:          Protein identifier.
        topology_path:   Path to topology.pdb.
        trajectory_path: Path to traj.xtc.
        artifacts_dir:   Root directory for numpy artifacts.
        results_dir:     Root directory for CSV/JSON pipeline outputs.
        lag_tica:        tICA lag time (frames).
        dim_tica:        Number of tICA dimensions.
        n_clusters:      KMeans cluster count.
        lag_msm:         MSM lag time (frames).
        k_neighbors:     k for local-density signal.
        window:          Smoothing window size.
        seed:            Random seed.

    Returns:
        True on success, False on failure.
    """
    # Import pipeline functions from the sibling module
    try:
        from run_all_proteins import (
            compute_features,
            run_tica,
            cluster_states,
            build_msm,
            compute_anomaly_signals,
            compute_residue_scores,
            compute_rmsf_residue_json,
        )
    except ImportError as exc:
        log.error("[%s] Cannot import run_all_proteins: %s", pdb_id, exc)
        return False

    art_dir = artifacts_dir / pdb_id
    res_dir = results_dir / pdb_id
    art_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Feature extraction ---
    log.info("[%s] ── Step 1/5: Feature extraction", pdb_id)
    try:
        X, traj = compute_features(topology_path, trajectory_path, stride=1)
    except Exception as exc:
        log.error("[%s] Feature extraction failed: %s", pdb_id, exc)
        return False

    n_frames, n_feats = X.shape
    log.info("[%s]   %d frames × %d features", pdb_id, n_frames, n_feats)

    if n_frames < 10:
        log.warning("[%s] Too few frames (%d) — skipping.", pdb_id, n_frames)
        return False

    # --- Step 2: tICA ---
    log.info("[%s] ── Step 2/5: tICA (lag=%d, dim=%d)", pdb_id, lag_tica, dim_tica)
    try:
        Y, _ = run_tica(X, lag=lag_tica, dim=dim_tica)
    except Exception as exc:
        log.error("[%s] tICA failed: %s", pdb_id, exc)
        return False

    np.save(art_dir / "tica_coords.npy", Y)

    # --- Step 3: Clustering ---
    log.info("[%s] ── Step 3/5: Clustering (n_clusters=%d)", pdb_id, n_clusters)
    try:
        dtraj, _ = cluster_states(Y, n_clusters=n_clusters, seed=seed)
    except Exception as exc:
        log.error("[%s] Clustering failed: %s", pdb_id, exc)
        return False

    np.save(art_dir / "dtraj.npy", dtraj)

    # --- Step 4: MSM ---
    log.info("[%s] ── Step 4/5: MSM (lag=%d)", pdb_id, lag_msm)
    try:
        msm, P, pi = build_msm(dtraj, lag=lag_msm)
    except Exception as exc:
        log.error("[%s] MSM failed: %s", pdb_id, exc)
        return False

    np.save(art_dir / "P.npy", P)
    np.save(art_dir / "pi.npy", pi)

    # --- Step 5: Anomaly scoring ---
    log.info("[%s] ── Step 5/5: Anomaly scoring", pdb_id)
    try:
        frame_scores, components = compute_anomaly_signals(
            msm, dtraj, Y,
            lag_msm=lag_msm,
            k_neighbors=k_neighbors,
            window=window,
        )
    except Exception as exc:
        log.error("[%s] Anomaly scoring failed: %s", pdb_id, exc)
        return False

    # Save frame scores
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

    # Save residue scores
    residue_scores = compute_residue_scores(traj, frame_scores)
    residue_json = res_dir / "residue_scores_dynamic.json"
    with open(residue_json, "w") as fh:
        json.dump(residue_scores, fh, indent=2)
    log.info("[%s]   Residue scores → %s", pdb_id, residue_json)

    # RMSF residue scores (consumed by tools/export_for_asvs.py)
    rmsf_scores = compute_rmsf_residue_json(traj)
    if rmsf_scores:
        with open(res_dir / "residue_scores_rmsf.json", "w") as fh:
            json.dump(rmsf_scores, fh, indent=2)

    log.info(
        "[%s] ✓ Pipeline complete — mean frame score: %.1f",
        pdb_id, float(frame_scores.mean()),
    )
    return True


# ---------------------------------------------------------------------------
# Step 4 — Export ASVS-compatible JSON
# ---------------------------------------------------------------------------


# -------------------------------------------------------------- #
# Function: export_for_asvs
# -------------------------------------------------------------- #
def export_for_asvs(
    pdb_id: str,
    topology_path: Path,
    trajectory_path: Path,
    results_dir: Path,
    exports_dir: Path,
) -> bool:
    """Export pipeline outputs as ASVS-compatible JSON files.

    Calls ``tools/export_for_asvs.py`` as a subprocess so it runs in exactly
    the same way a user would invoke it from the command line.  The topology
    and trajectory files are also copied into the export directory so the ASVS
    viewer can find them alongside the JSON files.

    Output layout::

        exports/<PDB_ID>/
          hotspots_residue.json
          anomaly_residue.json
          rmsf_residue.json
          tica_importance.json
          topology.pdb
          trajectory.xtc

    Args:
        pdb_id:          Protein identifier.
        topology_path:   Path to topology.pdb.
        trajectory_path: Path to traj.xtc.
        results_dir:     Root results directory (contains <PDB_ID>/ sub-dir).
        exports_dir:     Root exports directory.

    Returns:
        True on success, False on failure.
    """
    metrics_dir = results_dir / pdb_id
    out_dir = exports_dir / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate export_for_asvs.py relative to this script
    script = Path(__file__).parent / "tools" / "export_for_asvs.py"
    if not script.exists():
        log.error("[%s] export_for_asvs.py not found at %s", pdb_id, script)
        return False

    cmd = [
        sys.executable, str(script),
        "--topology", str(topology_path),
        "--trajectory", str(trajectory_path),
        "--msm_dir", str(metrics_dir),  # export_for_asvs accepts this but does not require it
        "--metrics_dir", str(metrics_dir),
        "--output_dir", str(out_dir),
    ]

    log.info("[%s] Exporting ASVS JSON → %s", pdb_id, out_dir)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            log.error(
                "[%s] export_for_asvs.py failed (exit %d):\n%s",
                pdb_id, result.returncode, result.stderr,
            )
            return False
    except Exception as exc:
        log.error("[%s] export_for_asvs.py subprocess error: %s", pdb_id, exc)
        return False

    # Copy topology and trajectory so the ASVS viewer finds them locally
    shutil.copy2(topology_path, out_dir / "topology.pdb")
    shutil.copy2(trajectory_path, out_dir / "trajectory.xtc")

    log.info("[%s] ✓ ASVS export complete → %s", pdb_id, out_dir)
    return True


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


# -------------------------------------------------------------- #
# Function: main
# -------------------------------------------------------------- #
def main() -> int:
    """Parse arguments, then run all four pipeline stages for each PDB ID."""
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end batch runner: downloads PDB structures, generates toy "
            "MD trajectories with OpenMM, runs the anomaly-detection pipeline, "
            "and exports ASVS-compatible JSON files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pdb_ids",
        nargs="+",
        default=DEFAULT_PDB_IDS,
        metavar="ID",
        help="One or more PDB IDs to process.",
    )
    parser.add_argument(
        "--work_dir",
        default="data",
        help="Root directory for downloaded PDB files and toy trajectories.",
    )
    parser.add_argument(
        "--artifacts_dir",
        default="artifacts",
        help="Root directory for numpy pipeline artifacts.",
    )
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Root directory for CSV/JSON pipeline results.",
    )
    parser.add_argument(
        "--exports_dir",
        default="exports",
        help="Root directory for ASVS-compatible JSON exports.",
    )

    # MD options
    parser.add_argument(
        "--md_frames",
        type=int,
        default=DEFAULT_MD_FRAMES,
        help="Number of frames in the toy trajectory.",
    )
    parser.add_argument(
        "--md_steps_per_frame",
        type=int,
        default=DEFAULT_MD_STEPS_PER_FRAME,
        help="MD integration steps between saved frames.",
    )

    # Pipeline hyper-parameters
    parser.add_argument("--lag_tica", type=int, default=DEFAULT_LAG_TICA)
    parser.add_argument("--dim_tica", type=int, default=DEFAULT_DIM_TICA)
    parser.add_argument("--n_clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--lag_msm", type=int, default=DEFAULT_LAG_MSM)
    parser.add_argument("--k_neighbors", type=int, default=DEFAULT_K_NEIGHBORS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--seed", type=int, default=42)

    # Skip flags (useful for iterative development)
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip RCSB download (topology.pdb must already exist in <work_dir>/<PDB_ID>/).",
    )
    parser.add_argument(
        "--skip_md",
        action="store_true",
        help="Skip OpenMM MD generation (traj.xtc must already exist).",
    )
    parser.add_argument(
        "--skip_pipeline",
        action="store_true",
        help="Skip ML pipeline (results must already exist in <results_dir>/<PDB_ID>/).",
    )
    parser.add_argument(
        "--skip_export",
        action="store_true",
        help="Skip ASVS export.",
    )

    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    artifacts_dir = Path(args.artifacts_dir)
    results_dir = Path(args.results_dir)
    exports_dir = Path(args.exports_dir)

    log.info("=" * 60)
    log.info("Batch Runner — Ensemble Anomaly Maps")
    log.info("=" * 60)
    log.info("PDB IDs   : %s", ", ".join(args.pdb_ids))
    log.info("Work dir  : %s", work_dir)
    log.info("Exports   : %s", exports_dir)
    log.info("=" * 60)

    succeeded = []
    failed = []

    for pdb_id in args.pdb_ids:
        pdb_id = pdb_id.upper()
        pdb_dir = work_dir / pdb_id
        topology_path = pdb_dir / "topology.pdb"
        canonical_path = pdb_dir / CANONICAL_TOPOLOGY_NAME
        trajectory_path = pdb_dir / "traj.xtc"

        log.info("")
        log.info("─" * 60)
        log.info("Processing: %s", pdb_id)
        log.info("─" * 60)

        ok = True

        # ── Stage 1: Download PDB ──────────────────────────────────────────
        if not args.skip_download:
            try:
                download_pdb(pdb_id, pdb_dir)
            except RuntimeError as exc:
                log.error("%s", exc)
                ok = False

        if ok and not topology_path.exists():
            log.error(
                "[%s] topology.pdb not found at %s and download was skipped.",
                pdb_id, topology_path,
            )
            ok = False

        # ── Stage 2: Generate toy trajectory ──────────────────────────────
        if ok and not args.skip_md:
            try:
                generate_toy_trajectory(
                    pdb_id,
                    topology_path,
                    pdb_dir,
                    n_frames=args.md_frames,
                    steps_per_frame=args.md_steps_per_frame,
                )
            except RuntimeError as exc:
                log.error("%s", exc)
                ok = False

        if ok and not trajectory_path.exists():
            log.error(
                "[%s] traj.xtc not found at %s and MD generation was skipped.",
                pdb_id, trajectory_path,
            )
            ok = False

        # ── Topology-consistency contract ──────────────────────────────────
        # Everything after MD uses the canonical topology written during
        # simulation. Fall back to the raw PDB only if canonical is absent
        # (e.g. legacy trajectory reuse) — and warn, because that risks drift.
        downstream_topology = canonical_path if canonical_path.exists() else topology_path
        if ok and downstream_topology is topology_path:
            log.warning(
                "[%s] %s not found — using raw topology.pdb, which may not match "
                "the trajectory. Re-run MD to regenerate the canonical topology.",
                pdb_id, CANONICAL_TOPOLOGY_NAME,
            )

        # ── Stage 3: ML pipeline ───────────────────────────────────────────
        if ok and not args.skip_pipeline:
            ok = run_pipeline(
                pdb_id=pdb_id,
                topology_path=downstream_topology,
                trajectory_path=trajectory_path,
                artifacts_dir=artifacts_dir,
                results_dir=results_dir,
                lag_tica=args.lag_tica,
                dim_tica=args.dim_tica,
                n_clusters=args.n_clusters,
                lag_msm=args.lag_msm,
                k_neighbors=args.k_neighbors,
                window=args.window,
                seed=args.seed,
            )

        # ── Stage 4: ASVS export ───────────────────────────────────────────
        if ok and not args.skip_export:
            ok = export_for_asvs(
                pdb_id=pdb_id,
                topology_path=downstream_topology,
                trajectory_path=trajectory_path,
                results_dir=results_dir,
                exports_dir=exports_dir,
            )

        if ok:
            succeeded.append(pdb_id)
        else:
            failed.append(pdb_id)

    # ── Summary ────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("Batch run complete.")
    log.info("  Succeeded : %s", ", ".join(succeeded) if succeeded else "(none)")
    log.info("  Failed    : %s", ", ".join(failed) if failed else "(none)")
    if succeeded:
        log.info("")
        log.info("ASVS exports written to:")
        for pid in succeeded:
            log.info("  %s/", exports_dir / pid)
        log.info("")
        log.info("To visualize, copy an export directory to your ASVS viewer:")
        log.info("  cp -r %s/1CRN /path/to/asvs/viewer/data/", exports_dir)
        log.info("  cd /path/to/asvs && python app.py")
    log.info("=" * 60)

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
