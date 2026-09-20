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

    # ---- basin census -------------------------------------------------- #
    # The single-basin test below is one-sided: it only catches an ensemble
    # that is too TIGHT. An ATLAS screen of 63 trajectories returned 60
    # "suitable" at 15-44 A spread with 0.5% medoid occupancy - i.e. one frame.
    # Those systems are not multi-basin, they are diffusive: no structure is
    # revisited, so there is no metastable state to be in. An MSM has as little
    # to find there as in a single basin, for the opposite reason.
    #
    # A leader-clustering census distinguishes the two. Frames are assigned in
    # order to the first cluster whose representative lies within the cutoff.
    # A multi-basin ensemble yields a few well-populated clusters; a single
    # basin yields one holding nearly everything; a diffusive chain yields many
    # clusters none of which holds anything. Order dependence is accepted here -
    # this is an advisory screen, not an estimator.
    leaders, assign = [], np.empty(m, dtype=np.int64)
    for i in range(m):
        placed = False
        for c, lead in enumerate(leaders):
            if R[i, lead] < rmsd_cluster_cutoff_ang:
                assign[i] = c
                placed = True
                break
        if not placed:
            leaders.append(i)
            assign[i] = len(leaders) - 1
    occ = np.bincount(assign, minlength=len(leaders)) / m
    occ_sorted = np.sort(occ)[::-1]
    n_basins = int(len(leaders))
    largest_occ = float(occ_sorted[0])
    n_pop = int((occ >= 0.05).sum())

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

    # Diffusive means NO cluster is populated - not merely that the largest one
    # is small. An earlier version vetoed on largest_occ < 0.10, which rejected
    # 3dso_A_R2 (33 clusters, top 9.0%, SIX clusters over 5%) as diffusive while
    # accepting its own replicate R1 at top 18.4%. A system spreading population
    # over six basins is the multi-basin case, not the structureless one. Keying
    # the veto on the populated-cluster count removes that knife edge.
    if n_pop == 0:
        verdict = "unsuitable"
        reasons.append(
            f"no conformational cluster holds even 5% of frames (largest "
            f"{largest_occ*100:.1f}%, {n_basins} clusters over {m} sampled frames) - "
            "the ensemble is diffusive, not metastable. Nothing is revisited, so "
            "there are no states for an MSM to resolve and transition "
            "probabilities cannot be estimated from repeat visits")
    elif n_pop < 2:
        if verdict == "suitable":
            verdict = "marginal"
        reasons.append(
            f"only {n_pop} cluster(s) hold >=5% of frames - the ensemble is close "
            "to unimodal, so a multi-state model is unlikely to be identifiable")

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
            "n_basins": n_basins,
            "largest_basin_occupancy": round(largest_occ, 3),
            "n_basins_ge_5pct": n_pop,
            "n_frames_sampled_for_clustering": int(m),
            "n_disulfides": ss}


# ---------------------------------------------------------------------------
# S-1b - recurrence.  Added 2026-08-31 (OI-34 / Phase 8b, report sections 18-19)
# ---------------------------------------------------------------------------
# The basin census above asks whether conformational states EXIST. It cannot
# tell you whether they are ever REVISITED, and an MSM needs both: transition
# probabilities are estimated from repeat visits, and a stationary distribution
# is meaningless without them. Phase 7 found ATLAS replicates ANTI-occupying
# each other's states (rho = -0.515 against a +0.620 control); Phase 8 found the
# reason - at 100 ns these trajectories are too short for recurrence to be
# usable. This check surfaces that before any model is fitted.
RECURRENCE_BANDS = {"recurrent": 25.0, "weak": 60.0}   # see docstring: DESCRIPTIVE
SINGLETON_MASS_LIMIT = 0.50
# Sampling required before a stationary distribution means anything. Kozlowski &
# Grubmuller put the convergence "tipping point" at 4-8 us for 50-112 residue
# proteins; we take the LOWER bound and still fail almost everything, which is
# the honest reading of report section 15 (ATLAS replicates ANTI-occupy at
# rho = -0.515 against a +0.620 control). See the note in assess_estimability
# about why singleton mass alone cannot catch this.
PI_CONVERGENCE_NS = 4000.0
# The figure above is Kozlowski & Grubmuller's LOWER bound (they report 4-8 us)
# and it was measured on proteins of 50-112 residues. Applying it unchanged to a
# 300-residue protein is an extrapolation, and a generous one - larger proteins
# converge more slowly, not less. OI-40: the threshold is therefore (a) a module
# constant any caller may override, and (b) flagged as extrapolated whenever the
# chain falls outside the range it was measured on, so the caveat travels with
# the number instead of living in a docstring.
PI_CONVERGENCE_RESIDUE_RANGE = (50, 112)


def assess_recurrence(traj, min_gap_ns=10.0, max_frames=400, selection="name CA",
                      also=("all",)):
    """Does this trajectory ever return to a conformation it already visited?

    Model-free: no clustering, no tICA, no MSM. Compares the closest pair of
    frames that are at least `min_gap_ns` apart in time against the distribution
    of ordinary consecutive-frame steps, and reports where it falls as a
    percentile.

      low percentile  -> the trajectory revisits conformations
      high percentile -> it never comes back; every frame is new territory

    ALWAYS REPORTED WITH ITS STRIDE. The percentile is confounded by the frame
    save interval - a coarser stride inflates ordinary steps and flatters the
    number by up to 8x (report section 19.1). A percentile quoted without a
    stride is uninterpretable.

    The bands in RECURRENCE_BANDS are DESCRIPTIVE, anchored to the measured
    ATLAS distribution (median 64.3 at 100 ps delivery, n=25). They are NOT
    calibrated against a ground-truth sampling-adequacy label, because no such
    label exists. Treat the number as the finding and the band as a label on it.
    """
    import mdtraj as md

    # OI-38, settled 2026-09-12 by measurement across all 25 ATLAS systems
    # (report section 22, validation/phase8_selection.json):
    #
    #   selection   median pct   recurrent(<=25)   non_recurrent(>60)
    #   CA               4.0          0.88               0.04
    #   backbone         4.2          0.88               0.04
    #   heavy           43.3          0.12               0.32
    #   all             64.3          0.08               0.60
    #
    # The division is NOT CA-versus-all-atom, it is BACKBONE-versus-SIDE-CHAIN:
    # CA and full backbone agree to within 0.2 percentile, while heavy atoms and
    # all atoms sit an order of magnitude higher. Spearman(CA, all) = +0.476,
    # and the two disagree on verdict band in 20 of 25 systems - they are
    # different measurements, not noisy copies of one.
    #
    # They also answer different questions, which is the whole reconciliation:
    #   backbone recurrence -> "does the protein revisit conformational STATES?"
    #                          the right question for whether an MSM's states
    #                          can be revisited.                    YES, 88%
    #   all-atom recurrence -> "can a splice be hidden from a detector reading
    #                          full coordinates?" the right question for
    #                          benchmark constructibility.           NO, 8%
    #
    # Section 18 used all atoms and was right to, for the question it asked.
    # This function uses CA and is right to, for the question it asks. The error
    # was ever letting one number answer both. Both are now reported, each
    # labelled with the question it answers.
    sel = traj.topology.select(selection)
    ct = traj.atom_slice(sel).superpose(traj.atom_slice(sel))
    n_full = len(ct)

    # physical stride, from the trajectory's own clock where available
    stride_ns = None
    if getattr(traj, "time", None) is not None and n_full > 1:
        dt = float(np.median(np.diff(np.asarray(traj.time, dtype=float))))
        if np.isfinite(dt) and dt > 0:
            stride_ns = dt / 1000.0                      # mdtraj time is in ps

    step = max(1, n_full // max_frames)
    sub = ct[::step]
    m = len(sub)
    eff_stride = stride_ns * step if stride_ns else None

    out = {"n_frames_sampled": int(m),
           "atom_selection": selection,
           "n_atoms": int(len(sel)),
           "subsample_step": int(step),
           "stride_ns": round(eff_stride, 4) if eff_stride else None,
           "total_ns": round(n_full * stride_ns, 1) if stride_ns else None}

    if eff_stride is None:
        return {**out, "verdict": "unknown",
                "reason": ("trajectory carries no usable time axis, so the "
                           "percentile cannot be reported with its stride and "
                           "is therefore uninterpretable")}

    gap = max(1, int(round(min_gap_ns / eff_stride)))
    if m <= gap + 2:
        return {**out, "verdict": "unknown",
                "reason": (f"only {m} sampled frames at {eff_stride:.3f} ns; "
                           f"need more than {gap + 2} for a {min_gap_ns} ns gap")}

    R = np.zeros((m, m))
    for i in range(m):
        R[i] = md.rmsd(sub, sub, i) * 10.0

    ii, jj = np.triu_indices(m, k=gap)
    closest = float(R[ii, jj].min())
    ref = np.array([R[t, t + 1] for t in range(m - 1)])
    pct = float((ref < closest).mean() * 100)

    if pct <= RECURRENCE_BANDS["recurrent"]:
        verdict, why = "recurrent", "the trajectory returns to visited conformations"
    elif pct <= RECURRENCE_BANDS["weak"]:
        verdict, why = "weak", "it returns only loosely; repeat visits will be thin"
    else:
        verdict, why = "non_recurrent", (
            "it never returns - the closest pair separated by "
            f"{min_gap_ns:g} ns is further apart than {pct:.0f}% of ordinary steps, "
            "so there are no repeat visits for transition probabilities or a "
            "stationary distribution to be estimated from")

    # the other selections, reported but never used for the verdict
    comparison = {}
    for alt in (also or ()):
        try:
            asel = traj.topology.select(alt)
            at = traj.atom_slice(asel).superpose(traj.atom_slice(asel))[::step]
            am = len(at)
            if am <= gap + 2:
                continue
            AR = np.zeros((am, am))
            for i in range(am):
                AR[i] = md.rmsd(at, at, i) * 10.0
            ai, aj = np.triu_indices(am, k=gap)
            acl = float(AR[ai, aj].min())
            aref = np.array([AR[t, t + 1] for t in range(am - 1)])
            comparison[alt] = {
                "n_atoms": int(len(asel)),
                "closest_pair_percentile": round(
                    float((aref < acl).mean() * 100), 1)}
        except Exception:
            continue

    return {**out,
            "verdict": verdict,
            "question": ("does the protein revisit conformational STATES? "
                         "backbone geometry, because state identity is a "
                         "backbone property (OI-38, report section 22)"),
            "comparison_selections": comparison,
            "comparison_note": ("all-atom answers a DIFFERENT question - whether "
                                "a splice could be hidden from a detector reading "
                                "full coordinates - and is expected to be much "
                                "higher. Do not read it as this verdict."),
            "closest_distant_pair_ang": round(closest, 3),
            "ordinary_step_median_ang": round(float(np.median(ref)), 3),
            "closest_pair_percentile": round(pct, 1),
            "min_gap_ns": min_gap_ns,
            "reason": why,
            "bands_are_descriptive": True}


def assess_estimability(dtraj, lag, total_ns=None, n_residues=None,
                        convergence_ns=None):
    """Is there enough REPEATED evidence to estimate P and pi?

    TWO conditions, and the second is the one that actually bites.

    (1) Within-trajectory repetition: are transition counts dominated by cells
        seen exactly once? This catches a trajectory chopped into so many
        microstates that nothing repeats.

    (2) Total sampling: is the trajectory long enough for a stationary
        distribution to have converged at all?

    Condition (1) alone is NOT sufficient and must never be used alone. With 20
    states over ~1000 frames every state is revisited ~50 times and singleton
    mass sits near 4%, so (1) passes comfortably on exactly the ATLAS data where
    Phase 7 measured pi FAILING to generalise across replicates (rho = -0.515,
    control +0.620). Plenty of within-trajectory repetition and no reproducible
    stationary distribution are entirely compatible: coarse states are revisited
    inside one run while their populations mean nothing across runs. Only (2)
    reflects that finding, because cross-replicate generalisation cannot be
    measured from a single trajectory.

    This is the direct, non-arbitrary form of the recurrence question, asked of
    the discrete trajectory the MSM is actually fitted to. If most observed
    transitions occurred exactly once, then P is fitted to unrepeated events and
    pi is a normalisation of noise - which is precisely what Phase 7 measured
    when ATLAS replicates anti-occupied each other's states.

    `singleton_mass` is the fraction of all transition counts sitting in cells
    observed exactly once. Above SINGLETON_MASS_LIMIT the stationary
    distribution is reported as UNRESOLVED and every pi-derived quantity is
    withheld downstream (CLAIMS.md W-3).
    """
    d = np.asarray(dtraj, dtype=int)
    n_obs = len(d) - lag
    if n_obs < 2:
        return {"verdict": "unknown", "reason": f"only {max(n_obs,0)} transitions"}

    k = int(d.max()) + 1
    C = np.zeros((k, k), dtype=np.int64)
    np.add.at(C, (d[:-lag], d[lag:]), 1)
    total = int(C.sum())
    singleton_mass = float(C[C == 1].sum()) / total if total else 1.0

    visits = np.bincount(d, minlength=k)
    occupied = int((visits > 0).sum())
    revisited = int((visits > 1).sum())

    reasons = []
    if singleton_mass > SINGLETON_MASS_LIMIT:
        reasons.append(
            f"{singleton_mass*100:.0f}% of transition counts were observed exactly "
            "once, so the transition matrix is fitted to unrepeated events")
    threshold = float(convergence_ns if convergence_ns else PI_CONVERGENCE_NS)
    lo, hi = PI_CONVERGENCE_RESIDUE_RANGE
    extrapolated = bool(n_residues is not None and not (lo <= n_residues <= hi))
    ratio = None
    if total_ns is None:
        reasons.append(
            "trajectory length unknown, so convergence of pi cannot be checked - "
            "failing closed")
    else:
        ratio = threshold / float(total_ns) if total_ns > 0 else float("inf")
        if float(total_ns) < threshold:
            reasons.append(
                f"{total_ns:.0f} ns of sampling is {ratio:.0f}x below the "
                f"{threshold:.0f} ns convergence threshold (Kozlowski & "
                "Grubmuller, lower bound), so the stationary distribution has not "
                "converged - measured directly in report section 15, where ATLAS "
                "replicates anti-occupy each other's states")
    unresolved = bool(reasons)
    return {
        "verdict": "unresolved" if unresolved else "estimable",
        "total_ns": round(float(total_ns), 1) if total_ns else None,
        "pi_convergence_ns": threshold,
        "pi_convergence_source": ("Kozlowski & Grubmuller, lower bound of a "
                                  f"4000-8000 ns range measured on {lo}-{hi} "
                                  "residue proteins"),
        "n_residues": n_residues,
        "threshold_extrapolated": extrapolated,
        "threshold_caveat": (
            f"chain has {n_residues} residues, outside the {lo}-{hi} range the "
            "threshold was measured on; larger proteins converge more slowly, so "
            "this threshold is if anything too generous"
            if extrapolated else None),
        "shortfall_factor": round(ratio, 1) if ratio else None,
        "n_transitions": total,
        "n_states_occupied": occupied,
        "n_states_revisited": revisited,
        "singleton_mass": round(singleton_mass, 3),
        "singleton_mass_limit": SINGLETON_MASS_LIMIT,
        "reason": ("; ".join(reasons) + " (CLAIMS.md W-3)") if unresolved else
                  (f"{singleton_mass*100:.0f}% singleton mass; {revisited}/{occupied} "
                   f"states revisited; {total_ns:.0f} ns sampling"),
    }
