# ============================================================== #
#  Module:      scoring/residue_attribution.py
#  Description: RMSF-orthogonal per-residue attribution (KSD-RA+)
#  Author:      Siya Jethliya
#  Copyright (c) 2026 SciVizAI — All rights reserved.
# ============================================================== #
"""
Kinetic State-Discriminant Residue Attribution, corrected and orthogonalised.

WHY THIS EXISTS
---------------
The original per-residue score was `0.5*rmsf_norm + 0.5*mean_frame_score`. The
second term is a single constant added to every residue, so the ranking WAS
RMSF (Pearson r = 1.0000000000 on 1VII; defect D-04). The first repair replaced
the constant with a displacement-weighted participation term, which removed the
offset but not the collinearity: r stayed at 0.990-0.996 (erratum E-1).

Independent external audits reached the same conclusion and proposed KSD-RA,
which conditions on ANOMALOUS vs BASELINE ensembles instead of measuring
displacement magnitude. That idea is sound. This module implements it with four
corrections to the reference implementation:

  1. TIE-AWARE RANKING. The reference used argsort(argsort(x)) - the exact
     defect (D-02) already fixed elsewhere in this codebase. Residues with equal
     contact flux were spread across the score range by array index. We use
     scipy.stats.rankdata(method="average").

  2. MEMORY-SAFE CONTACT FLUX. The reference materialised an
     (n_frames, n_res, n_res) float64 array: 2.5 GB for a 178-residue protein at
     10k frames, ~80 GB for a 3176-residue complex. We accumulate the two group
     means in a streaming pass, so memory is O(n_res^2) regardless of length.

  3. CORRECTLY ALIGNED DIHEDRALS. mdtraj's compute_phi returns angles for
     residues 2..N and compute_psi for residues 1..N-1 - offset by one. Pairing
     them column-wise silently associates phi(i+1) with psi(i). We map each
     angle back to its residue through the returned atom-index arrays.

  4. EXPLICIT RMSF ORTHOGONALISATION (optional, ON by default). Even a
     well-designed kinetic metric can correlate with flexibility simply because
     flexible residues move more. We regress the fused score on RMSF and keep
     the residual, so the reported attribution is by construction the part NOT
     explained by flexibility. The raw (non-orthogonalised) score is returned
     alongside, and both collinearity figures are reported.

WHAT THIS DOES NOT DO
---------------------
Producing a score that is not RMSF is necessary but NOT sufficient. This module
makes no claim that its output identifies functionally important residues. That
requires external validation against annotations independent of the model
(see OI-12). `diagnostics` is returned so the collinearity claim can be checked
on every run rather than assumed.
"""
import warnings

import numpy as np
from scipy.stats import rankdata


# -------------------------------------------------------------- #
# Function: _rank_norm
# -------------------------------------------------------------- #
def _rank_norm(x):
    """Tie-aware rank normalisation to [0, 100]. Equal inputs -> equal outputs."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    out = np.zeros_like(x)
    if finite.sum() <= 1 or np.ptp(x[finite]) == 0:
        return out
    r = rankdata(x[finite], method="average") - 1.0
    out[finite] = r / (finite.sum() - 1) * 100.0
    return out


# -------------------------------------------------------------- #
# Function: split_ensembles
# -------------------------------------------------------------- #
def split_ensembles(frame_scores, percentile=85.0):
    """Return (anomalous_mask, baseline_mask) from per-frame anomaly scores."""
    s = np.asarray(frame_scores, dtype=np.float64)
    finite = np.isfinite(s)
    if finite.sum() < 10:
        raise ValueError(f"Need >=10 finite frame scores, got {finite.sum()}")
    thresh = np.percentile(s[finite], percentile)
    anom = finite & (s >= thresh)
    base = finite & (s < thresh)
    if anom.sum() < 3 or base.sum() < 3:
        raise ValueError(
            f"Ensemble split too small at percentile={percentile}: "
            f"{anom.sum()} anomalous / {base.sum()} baseline frames"
        )
    return anom, base


# -------------------------------------------------------------- #
# Function: contact_flux
# -------------------------------------------------------------- #
def contact_flux(ca_xyz, anom_mask, base_mask, cutoff_nm=0.8, chunk=200):
    """Anomaly-conditioned contact flux, streamed to bound memory.

    S_contact(r) = sum_j |<C_rj>_anom - <C_rj>_base|

    Thermal breathing that does not change the contact map contributes ~0, which
    is what makes this channel structurally different from RMSF.

    Args:
        ca_xyz: (n_frames, n_res, 3) CA coordinates in nm.
        anom_mask / base_mask: boolean frame masks.
        cutoff_nm: contact distance threshold (0.8 nm = 8 A, matching the
            pipeline's native-contact feature).
        chunk: frames processed per batch; memory is O(chunk * n_res^2).

    Returns:
        (n_res,) contact flux per residue.
    """
    n_frames, n_res, _ = ca_xyz.shape
    acc_a = np.zeros((n_res, n_res), dtype=np.float64)
    acc_b = np.zeros((n_res, n_res), dtype=np.float64)
    n_a = n_b = 0

    for start in range(0, n_frames, chunk):
        stop = min(start + chunk, n_frames)
        block = ca_xyz[start:stop]                      # (c, n_res, 3)
        d = np.linalg.norm(block[:, :, None, :] - block[:, None, :, :], axis=-1)
        c = (d < cutoff_nm).astype(np.float64)
        ma = anom_mask[start:stop]
        mb = base_mask[start:stop]
        if ma.any():
            acc_a += c[ma].sum(axis=0); n_a += int(ma.sum())
        if mb.any():
            acc_b += c[mb].sum(axis=0); n_b += int(mb.sum())

    if n_a == 0 or n_b == 0:
        raise ValueError("One ensemble is empty after masking")
    delta = np.abs(acc_a / n_a - acc_b / n_b)
    np.fill_diagonal(delta, 0.0)
    return delta.sum(axis=1)


# -------------------------------------------------------------- #
# Function: dihedral_shift
# -------------------------------------------------------------- #
def dihedral_shift(traj, anom_mask, base_mask):
    """Anomaly-conditioned circular divergence of backbone phi/psi, per residue.

    phi and psi are mapped back to their own residues through mdtraj's returned
    atom-index arrays, so no off-by-one pairing can occur. Residues lacking an
    angle contribute 0 for that channel.

    Returns:
        (n_res,) combined angular divergence in degrees.
    """
    import mdtraj as md

    n_res = traj.topology.n_residues
    total = np.zeros(n_res, dtype=np.float64)

    for compute in (md.compute_phi, md.compute_psi):
        idx, ang = compute(traj)                         # idx (n_ang, 4), ang (T, n_ang)
        if ang.size == 0:
            continue
        # The dihedral belongs to the residue of its THIRD atom for phi
        # (C-N-CA-C) and its SECOND atom for psi (N-CA-C-N); in both cases the
        # CA-bearing residue. Resolve it from the topology rather than assuming.
        res_of = np.array([traj.topology.atom(q[2]).residue.index for q in idx])
        ca, sa = np.cos(ang[anom_mask]).mean(0), np.sin(ang[anom_mask]).mean(0)
        cb, sb = np.cos(ang[base_mask]).mean(0), np.sin(ang[base_mask]).mean(0)
        cos_diff = np.clip(ca * cb + sa * sb, -1.0, 1.0)
        d = np.degrees(np.arccos(cos_diff))
        for r, v in zip(res_of, d):
            if 0 <= r < n_res and np.isfinite(v):
                total[r] += v
    return total


# -------------------------------------------------------------- #
# Function: orthogonalise
# -------------------------------------------------------------- #
def orthogonalise(score, rmsf):
    """Remove the component of `score` linearly explained by `rmsf`.

    Returns the residual, re-ranked to [0, 100]. This makes 'not explained by
    flexibility' a property of the output rather than a hope about it.
    """
    s = np.asarray(score, float)
    f = np.asarray(rmsf, float)
    ok = np.isfinite(s) & np.isfinite(f)
    if ok.sum() < 3 or np.ptp(f[ok]) == 0:
        return s.copy()
    # rank-space regression: robust to the heavy tails RMSF usually has
    rs = rankdata(s[ok]).astype(float)
    rf = rankdata(f[ok]).astype(float)
    beta = np.polyfit(rf, rs, 1)
    resid = rs - np.polyval(beta, rf)
    out = np.zeros_like(s)
    out[ok] = resid
    return _rank_norm(out)


# -------------------------------------------------------------- #
# Function: compute_residue_attribution
# -------------------------------------------------------------- #
def compute_residue_attribution(traj, frame_scores, percentile=85.0,
                                w_contact=0.5, w_dihedral=0.5,
                                cutoff_nm=0.8, orthogonalize=True):
    """RMSF-orthogonal per-residue attribution.

    Args:
        traj: mdtraj Trajectory.
        frame_scores: (T,) per-frame anomaly scores from the pipeline.
        percentile: threshold defining the anomalous ensemble (default 85).
        w_contact / w_dihedral: channel weights (must sum to 1).
        cutoff_nm: contact cutoff in nm.
        orthogonalize: regress out RMSF and report the residual as the score.

    Returns:
        dict with 'score' (the reported attribution), 'score_raw' (pre-
        orthogonalisation), the two channels, 'rmsf_ang', and 'diagnostics'
        carrying the collinearity figures that must be checked, not assumed.
    """
    import mdtraj as md

    frame_scores = np.asarray(frame_scores, dtype=np.float64)
    if len(frame_scores) != len(traj):
        raise ValueError(f"frame_scores length {len(frame_scores)} != "
                         f"trajectory length {len(traj)}")

    anom, base = split_ensembles(frame_scores, percentile)

    ca_idx = traj.topology.select("name CA")
    if len(ca_idx) == 0:
        raise ValueError("No CA atoms found")
    ca_traj = traj.atom_slice(ca_idx).superpose(traj.atom_slice(ca_idx))
    ca_xyz = ca_traj.xyz                                  # nm

    # --- channels ---
    c_flux = contact_flux(ca_xyz, anom, base, cutoff_nm=cutoff_nm)
    d_shift_all = dihedral_shift(traj, anom, base)
    res_of_ca = np.array([traj.topology.atom(a).residue.index for a in ca_idx])
    d_shift = d_shift_all[res_of_ca]

    # --- RMSF, for orthogonalisation and reporting ---
    mean_xyz = ca_xyz.mean(axis=0)
    rmsf_ang = np.sqrt(((ca_xyz - mean_xyz) ** 2).sum(-1).mean(0)) * 10.0

    raw = w_contact * _rank_norm(c_flux) + w_dihedral * _rank_norm(d_shift)
    score = orthogonalise(raw, rmsf_ang) if orthogonalize else raw

    def _sp(a, b):
        from scipy.stats import spearmanr
        if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
            return float("nan")
        return float(spearmanr(a, b).statistic)

    diagnostics = {
        "n_residues": int(len(ca_idx)),
        "n_frames_anomalous": int(anom.sum()),
        "n_frames_baseline": int(base.sum()),
        "percentile": percentile,
        "spearman_raw_vs_rmsf": round(_sp(raw, rmsf_ang), 4),
        "spearman_final_vs_rmsf": round(_sp(score, rmsf_ang), 4),
        "spearman_contact_vs_rmsf": round(_sp(c_flux, rmsf_ang), 4),
        "spearman_dihedral_vs_rmsf": round(_sp(d_shift, rmsf_ang), 4),
        "orthogonalized": bool(orthogonalize),
    }
    if abs(diagnostics["spearman_final_vs_rmsf"]) > 0.7:
        warnings.warn(
            f"Residue attribution is still strongly collinear with RMSF "
            f"(Spearman {diagnostics['spearman_final_vs_rmsf']:.3f}). The score "
            "may be reporting flexibility rather than kinetic anomaly."
        )

    return {"score": score, "score_raw": raw, "contact_flux": c_flux,
            "dihedral_shift_deg": d_shift, "rmsf_ang": rmsf_ang,
            "diagnostics": diagnostics}
