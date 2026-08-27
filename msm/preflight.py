# ============================================================== #
#  Module:      msm/preflight.py
#  Description: Pre-analysis guards — equilibration discard and
#               system-suitability screening
#  Author:      Siya Jethliya
#  Copyright (c) 2026 SciVizAI — All rights reserved.
# ============================================================== #
"""
Two checks that must run BEFORE anomaly scoring, both added after external
audit found the pipeline reporting artifacts as discoveries.

1. EQUILIBRATION DISCARD (defect D-10)
   An independent 5 ns explicit-solvent audit of ubiquitin found frames 1-3
   scoring 99.7/100 - the highest anomaly scores in the whole trajectory - and
   traced them to unrelaxed Cartesian coordinates relaxing out of the starting
   crystal lattice. Our pipeline recorded from step 0 and discarded nothing, so
   simulation startup was being reported as rare-event biology. Any frame before
   the structure has equilibrated is not a rare conformational state; it is the
   model falling downhill from an experimental structure.

2. SYSTEM SUITABILITY (audit finding, Crambin)
   Crambin is a 46-residue plant thionin locked by three invariant disulfide
   bonds (Cys3-Cys40, Cys4-Cys32, Cys16-Cys26). It has no large-scale macrostate
   switching at 300 K. Running a multi-state MSM anomaly detector on it produces
   microstates that are noise partitions of a single basin - which is exactly
   what happened. A pipeline that cannot say "this system is unsuitable" will
   always return an answer, including when no answer exists.
"""
import numpy as np


# -------------------------------------------------------------- #
# Function: detect_equilibration
# -------------------------------------------------------------- #
def detect_equilibration(traj, max_fraction=0.20, n_sigma=2.0):
    """Return the index of the first equilibrated frame.

    Method: RMSD to the initial frame rises as the structure relaxes away from
    the deposited coordinates, then fluctuates about a plateau. We estimate the
    plateau from the last half of the trajectory and return the first frame that
    reaches its lower band (plateau_mean - n_sigma * plateau_std). Frames before
    that are still relaxing.

    Args:
        traj: mdtraj Trajectory.
        max_fraction: never discard more than this fraction of the trajectory,
            so a pathological detection cannot silently eat the data.
        n_sigma: tolerance band on the plateau.

    Returns:
        (start_index, info_dict)
    """
    import mdtraj as md

    n = len(traj)
    if n < 20:
        return 0, {"skipped": f"trajectory too short ({n} frames)"}

    r = md.rmsd(traj, traj, 0) * 10.0                      # Angstrom
    half = n // 2
    plateau_mean = float(np.mean(r[half:]))
    plateau_std = float(np.std(r[half:]))
    band = plateau_mean - n_sigma * plateau_std

    reached = np.where(r >= band)[0]
    start = int(reached[0]) if len(reached) else 0

    cap = int(n * max_fraction)
    capped = start > cap
    if capped:
        start = cap

    return start, {
        "n_frames": int(n),
        "equilibration_frames_discarded": int(start),
        "fraction_discarded": round(start / n, 4),
        "rmsd_initial_ang": round(float(r[0]), 4),
        "rmsd_plateau_ang": round(plateau_mean, 4),
        "rmsd_plateau_std_ang": round(plateau_std, 4),
        "detection_capped_at_max_fraction": bool(capped),
    }


# -------------------------------------------------------------- #
# Function: count_disulfides
# -------------------------------------------------------------- #
def count_disulfides(traj, cutoff_nm=0.25):
    """Count SG-SG pairs within `cutoff_nm` in the first frame."""
    sg = traj.topology.select("name SG")
    if len(sg) < 2:
        return 0
    xyz = traj.xyz[0][sg]
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return int((d < cutoff_nm).sum() // 2)


# -------------------------------------------------------------- #
# Function: assess_suitability
# -------------------------------------------------------------- #
def assess_suitability(traj, rmsd_cluster_cutoff_ang=2.0):
    """Judge whether a trajectory can support multi-state MSM anomaly detection.

    Heuristics, each reported with its value so the verdict can be argued with:
      - conformational spread: max pairwise CA RMSD
      - single-basin test: fraction of frames within `cutoff` of the medoid
      - covalent constraint: disulfide count relative to chain length
      - sampling: number of frames

    Returns a dict with `verdict` in {suitable, marginal, unsuitable} and the
    reasons behind it. This is advisory: it warns, it does not block.
    """
    import mdtraj as md

    ca = traj.topology.select("name CA")
    ct = traj.atom_slice(ca).superpose(traj.atom_slice(ca))
    n = len(ct)
    n_res = len(ca)

    step = max(1, n // 200)                                 # cap the O(n^2) work
    sub = ct[::step]
    m = len(sub)
    R = np.zeros((m, m))
    for i in range(m):
        R[i] = md.rmsd(sub, sub, i) * 10.0

    max_rmsd = float(R.max())
    medoid = int(np.argmin(R.sum(axis=1)))
    frac_in_basin = float((R[medoid] < rmsd_cluster_cutoff_ang).mean())
    ss = count_disulfides(traj)

    reasons, verdict = [], "suitable"
    if frac_in_basin > 0.95:
        verdict = "unsuitable"
        reasons.append(
            f"{frac_in_basin*100:.0f}% of frames lie within {rmsd_cluster_cutoff_ang} A "
            "of a single medoid - the ensemble is one basin, so MSM microstates "
            "will partition noise rather than conformational states")
    elif frac_in_basin > 0.85:
        verdict = "marginal"
        reasons.append(f"{frac_in_basin*100:.0f}% of frames in a single basin")

    if max_rmsd < 1.5:
        verdict = "unsuitable"
        reasons.append(f"max pairwise CA RMSD is only {max_rmsd:.2f} A - "
                       "no resolvable conformational change")

    if ss >= 2 and n_res < 80:
        if verdict == "suitable":
            verdict = "marginal"
        reasons.append(f"{ss} disulfide bonds in a {n_res}-residue chain - "
                       "covalently constrained scaffold, large-scale state "
                       "switching is unlikely at 300 K")

    if n < 500:
        if verdict == "suitable":
            verdict = "marginal"
        reasons.append(f"only {n} frames - MSM statistics will be thin "
                       "(see O-11: 100-1000 frames cannot support CK or ITS tests)")

    return {"verdict": verdict, "reasons": reasons, "n_frames": int(n),
            "n_residues": int(n_res), "max_pairwise_ca_rmsd_ang": round(max_rmsd, 3),
            "fraction_in_single_basin": round(frac_in_basin, 3),
            "n_disulfides": ss}
