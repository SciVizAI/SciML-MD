#!/usr/bin/env python3
"""
Generate short MD trajectories for protein structures using OpenMM.

For each protein directory under data/ that contains a topology.pdb but
no trajectory file, this script:

1. Loads the PDB structure.
2. Adds missing hydrogens.
3. Solvates the system in a TIP3P water box.
4. Minimizes the potential energy.
5. Runs a short NVT MD simulation (50,000–100,000 steps).
6. Saves the trajectory as data/{PDB_ID}/traj.xtc (mdtraj) or
   data/{PDB_ID}/traj.dcd (OpenMM native) when mdtraj is unavailable.

Requirements:
    pip install openmm mdtraj

Usage:
    python scripts/generate_md_trajectories.py
    python scripts/generate_md_trajectories.py --data_dir data --steps 50000
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trajectory file names accepted as "already present"
# ---------------------------------------------------------------------------
TRAJ_NAMES = ("traj.xtc", "traj.dcd", "trajectory.xtc", "trajectory.dcd")


def _check_openmm():
    """Raise ImportError with a helpful message if OpenMM is not installed."""
    try:
        import openmm  # noqa: F401
    except ImportError:
        raise ImportError(
            "OpenMM is required to generate trajectories.\n"
            "Install it with:  conda install -c conda-forge openmm  or\n"
            "                  pip install openmm"
        )


def find_protein_dirs(data_dir):
    """
    Return all sub-directories of *data_dir* that contain topology.pdb.

    Args:
        data_dir: Path to the root data directory.

    Returns:
        List of Path objects.
    """
    data_dir = Path(data_dir)
    dirs = []
    for subdir in sorted(data_dir.iterdir()):
        if subdir.is_dir() and (subdir / "topology.pdb").exists():
            dirs.append(subdir)
    return dirs


def has_trajectory(protein_dir):
    """Return True if a trajectory file already exists in *protein_dir*."""
    protein_dir = Path(protein_dir)
    return any((protein_dir / name).exists() for name in TRAJ_NAMES)


def generate_trajectory(
    protein_dir,
    n_steps=50_000,
    step_size_ps=0.002,
    temperature_K=300,
    padding_nm=1.0,
    report_interval=500,
    use_xtc=True,
):
    """
    Run a short OpenMM MD simulation for a single protein.

    Args:
        protein_dir: Directory containing topology.pdb.
        n_steps: Number of integration steps.
        step_size_ps: Integration time step in picoseconds.
        temperature_K: Simulation temperature in Kelvin.
        padding_nm: Water box padding around the solute in nanometres.
        report_interval: How often (steps) to write trajectory frames.
        use_xtc: If True and mdtraj is available, save as traj.xtc;
                 otherwise save as traj.dcd.

    Returns:
        Path to the written trajectory file.
    """
    _check_openmm()

    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    protein_dir = Path(protein_dir)
    pdb_path = protein_dir / "topology.pdb"
    pdb_id = protein_dir.name

    log.info("[%s] Loading structure: %s", pdb_id, pdb_path)
    pdb = app.PDBFile(str(pdb_path))

    # Force field
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    # Add hydrogens and solvate
    log.info("[%s] Adding hydrogens and solvating...", pdb_id)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield)
    modeller.addSolvent(
        forcefield,
        model="tip3p",
        padding=padding_nm * unit.nanometer,
    )

    log.info(
        "[%s] System: %d atoms after solvation.",
        pdb_id,
        modeller.topology.getNumAtoms(),
    )

    # Build system
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
    )

    # Integrator (Langevin)
    integrator = mm.LangevinMiddleIntegrator(
        temperature_K * unit.kelvin,
        1.0 / unit.picosecond,
        step_size_ps * unit.picoseconds,
    )

    # Simulation
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    # Energy minimization
    log.info("[%s] Minimising energy...", pdb_id)
    simulation.minimizeEnergy()

    # Velocity initialization
    simulation.context.setVelocitiesToTemperature(temperature_K * unit.kelvin)

    # Trajectory reporter
    dcd_path = protein_dir / "traj.dcd"
    simulation.reporters.append(
        app.DCDReporter(str(dcd_path), report_interval)
    )
    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout,
            report_interval * 10,
            step=True,
            potentialEnergy=True,
            temperature=True,
            progress=True,
            totalSteps=n_steps,
        )
    )

    # Run
    log.info("[%s] Running %d steps...", pdb_id, n_steps)
    simulation.step(n_steps)
    log.info("[%s] Simulation complete.", pdb_id)

    # Convert to XTC if mdtraj is available
    if use_xtc:
        try:
            import mdtraj as md

            traj = md.load_dcd(str(dcd_path), top=str(pdb_path))
            xtc_path = protein_dir / "traj.xtc"
            traj.save_xtc(str(xtc_path))
            dcd_path.unlink()  # Remove intermediate DCD
            log.info("[%s] Saved trajectory: %s", pdb_id, xtc_path)
            return xtc_path
        except ImportError:
            log.warning(
                "[%s] mdtraj not available; trajectory saved as DCD: %s",
                pdb_id,
                dcd_path,
            )

    log.info("[%s] Saved trajectory: %s", pdb_id, dcd_path)
    return dcd_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate MD trajectories for all proteins in data/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Root data directory containing {PDB_ID}/topology.pdb subdirectories",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50_000,
        help="Number of MD integration steps (50,000–100,000 recommended)",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.002,
        help="Integration time step in picoseconds",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Simulation temperature in Kelvin",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=1.0,
        help="Water box padding around the solute in nanometres",
    )
    parser.add_argument(
        "--report_interval",
        type=int,
        default=500,
        help="Steps between trajectory frames",
    )
    parser.add_argument(
        "--no_xtc",
        action="store_true",
        help="Keep DCD format even when mdtraj is available",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate trajectory even if one already exists",
    )
    args = parser.parse_args()

    # Validate OpenMM early
    try:
        _check_openmm()
    except ImportError as exc:
        log.error(str(exc))
        return 1

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        log.error("Data directory not found: %s", data_dir)
        return 1

    protein_dirs = find_protein_dirs(data_dir)
    if not protein_dirs:
        log.error(
            "No protein directories (containing topology.pdb) found under %s",
            data_dir,
        )
        return 1

    log.info("Found %d protein directories.", len(protein_dirs))

    n_done = 0
    n_skipped = 0
    n_failed = 0

    for protein_dir in protein_dirs:
        pdb_id = protein_dir.name

        if not args.overwrite and has_trajectory(protein_dir):
            log.info("[%s] Trajectory already exists — skipping.", pdb_id)
            n_skipped += 1
            continue

        try:
            generate_trajectory(
                protein_dir,
                n_steps=args.steps,
                step_size_ps=args.step_size,
                temperature_K=args.temperature,
                padding_nm=args.padding,
                report_interval=args.report_interval,
                use_xtc=not args.no_xtc,
            )
            n_done += 1
        except Exception as exc:
            log.error("[%s] Failed: %s", pdb_id, exc)
            n_failed += 1

    print("=" * 60)
    print(
        f"[summary] generated={n_done}, skipped={n_skipped}, failed={n_failed}"
    )
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
