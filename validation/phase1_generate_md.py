#!/usr/bin/env python3
"""
Phase 1, step 1 — generate the two trajectory pools for the seeded-anomaly test.

Both pools use the IDENTICAL topology, force field, solvent model, integrator and
timestep. The ONLY difference is temperature:

    normal pool : 300 K  — the "background" ensemble
    hot pool    : 450 K  — the anomaly source. Conformations abundant at 450 K are
                  genuinely rare at 300 K, so frames drawn from this pool constitute
                  rare states whose identity is known WITHOUT reference to any MSM.

This is the same logic mdCATH exposes at scale (multi-temperature replicates), so
Phase 1 prefigures the Phase 3 design on a system small enough to run locally.

Outputs: validation/phase1_data/{normal_300K,hot_450K}.xtc
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "validation/phase1_data"
OUT.mkdir(parents=True, exist_ok=True)

TOP = ROOT / "data/1VII/topology.pdb"
STEPS_PER_FRAME = 100
POOLS = {"normal_300K": (300.0, 250, 42), "hot_450K": (450.0, 120, 7)}
EQUIL_STEPS = 2000  # discarded before recording


def run(name, temp_K, n_frames, seed):
    import mdtraj as md
    from openmm import app as omm_app, unit
    import openmm as omm

    xtc = OUT / f"{name}.xtc"
    if xtc.exists():
        print(f"[{name}] exists — skipping", flush=True)
        return

    t0 = time.time()
    pdb = omm_app.PDBFile(str(TOP))
    ff = omm_app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    mod = omm_app.Modeller(pdb.topology, pdb.positions)
    mod.addHydrogens(ff)
    system = ff.createSystem(mod.topology, nonbondedMethod=omm_app.NoCutoff,
                             constraints=omm_app.HBonds)
    integ = omm.LangevinMiddleIntegrator(temp_K * unit.kelvin,
                                         1.0 / unit.picosecond,
                                         0.002 * unit.picoseconds)
    integ.setRandomNumberSeed(seed)
    sim = omm_app.Simulation(mod.topology, system, integ)
    sim.context.setPositions(mod.positions)
    sim.minimizeEnergy(maxIterations=500)
    sim.context.setVelocitiesToTemperature(temp_K * unit.kelvin, seed)
    sim.step(EQUIL_STEPS)  # equilibrate at target temperature, not recorded

    dcd = OUT / f"{name}.dcd"
    sim.reporters.append(omm_app.DCDReporter(str(dcd), STEPS_PER_FRAME))
    sim.step(n_frames * STEPS_PER_FRAME)
    sim.reporters.clear()

    traj = md.load(str(dcd), top=str(TOP))
    traj.save_xtc(str(xtc))
    dcd.unlink(missing_ok=True)
    print(f"[{name}] {len(traj)} frames @ {temp_K} K in {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    for name, (T, n, seed) in POOLS.items():
        run(name, T, n, seed)
    print("done", flush=True)
