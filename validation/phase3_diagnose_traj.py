#!/usr/bin/env python3
"""
Sanity-check downloaded trajectories BEFORE trusting the suitability screen.

WHY THIS EXISTS
    The first ATLAS screen returned 60/63 "suitable" with max pairwise CA RMSD
    of 15-44 A and 0.5% of frames inside the medoid basin. 0.5% is 1/200, i.e.
    the medoid frame and nothing else - every other frame further than 2 A from
    every other frame. A folded 90-residue protein at 300 K does not do that in
    100 ns. Either

      (a) the coordinates are broken across the periodic boundary, so RMSD is
          measuring box-jumps rather than conformational change, or
      (b) the systems really are that disordered - which is possible, because
          we selected the TOP of the ATLAS avg_RMSF distribution (up to 14.4 A
          against a database median of 1.33 A).

    (a) invalidates everything downstream. (b) is real but means the sample is
    the disordered tail of ATLAS, not a representative one.

HOW IT TELLS THEM APART
    Periodic-boundary breakage moves whole fragments by roughly one box vector
    in a single frame. Real unfolding does not teleport. So:

      Rg and Rmax (max CA distance from the centroid) get plotted over time.
      Genuine disorder drifts. PBC breakage steps discontinuously by ~box/2.
      `max_step` reports the largest single-frame change in Rmax.

    And an external cross-check that does not depend on any of our own code:
    ATLAS publishes avg_RMSF per system in the release TSV. We recompute RMSF
    from the downloaded coordinates. If our value is many times theirs, we are
    mishandling the file, not observing biology.

    Finally, where a unit cell is present, the trajectory is re-imaged with
    mdtraj's image_molecules() and the spread is recomputed. A large collapse
    means the raw numbers were PBC artefacts.

USAGE
    python validation/phase3_diagnose_traj.py --data_root ~/atlas
    python validation/phase3_diagnose_traj.py --data_root ~/atlas --only 1ef1_D 5w82_E
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MAX_FRAMES = 2000


def load_info_tsv(root: Path):
    """ATLAS release table saved by phase3_fetch_atlas.py -> {pdb_chain: row}."""
    import csv
    path = root / "atlas_info.tsv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8", errors="ignore") as fh:
        rows = list(csv.reader(fh, delimiter="\t", quotechar='"'))
    if not rows:
        return {}
    header = [h.strip() for h in rows[0]]
    out = {}
    for r in rows[1:]:
        if not r:
            continue
        rec = {header[i]: (r[i] if i < len(r) else "") for i in range(len(header))}
        out[rec.get("PDB", "").strip()] = rec
    return out


def _f(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def geometry(ct):
    """ct is a CA-only trajectory. Returns spread descriptors in Angstrom."""
    import mdtraj as md

    xyz = ct.xyz                                   # (n_frames, n_ca, 3), nm
    cen = xyz.mean(axis=1, keepdims=True)
    d = np.linalg.norm(xyz - cen, axis=2) * 10.0   # A, distance from centroid
    rmax = d.max(axis=1)
    rg = md.compute_rg(ct) * 10.0                  # A

    step = max(1, len(ct) // 200)
    sub = ct[::step]
    m = len(sub)
    R = np.zeros((m, m))
    for i in range(m):
        R[i] = md.rmsd(sub, sub, i) * 10.0
    medoid = int(np.argmin(R.sum(axis=1)))

    # RMSF about the mean structure, two-pass alignment
    c2 = ct[:]
    c2.superpose(c2, 0)
    ref = c2[0]
    ref.xyz = c2.xyz.mean(axis=0, keepdims=True)
    c2.superpose(ref, 0)
    mean_xyz = c2.xyz.mean(axis=0)
    rmsf = np.sqrt(((c2.xyz - mean_xyz) ** 2).sum(axis=2).mean(axis=0)) * 10.0

    return {
        "rg_median": float(np.median(rg)),
        "rg_max": float(rg.max()),
        "rmax_median": float(np.median(rmax)),
        "rmax_max": float(rmax.max()),
        "rmax_max_single_frame_step": float(np.abs(np.diff(rmax)).max()),
        "max_pairwise_ca_rmsd_ang": float(R.max()),
        "fraction_in_single_basin": float((R[medoid] < 2.0).mean()),
        "rmsf_mean_ang": float(rmsf.mean()),
        "rmsf_max_ang": float(rmsf.max()),
    }


def diagnose_one(top, trj, max_frames):
    import mdtraj as md

    t = md.load(str(trj), top=str(top))
    sel = t.topology.select("protein")
    if len(sel) and len(sel) != t.n_atoms:
        t = t.atom_slice(sel)
    if len(t) > max_frames:
        t = t[:: max(1, len(t) // max_frames)][:max_frames]

    ca = t.topology.select("name CA")
    # A box only counts if it is physically plausible. Some files carry a dummy
    # CRYST1 or a zero box; a 1 A "box" would make every jump test fire.
    box = None
    has_cell = False
    if t.unitcell_lengths is not None:
        b = float(np.median(t.unitcell_lengths)) * 10.0
        if np.isfinite(b) and b >= 20.0:
            box, has_cell = b, True

    out = {"n_frames": int(len(t)), "n_residues": int(len(ca)),
           "has_unitcell": bool(has_cell), "box_median_ang": box}
    out["raw"] = geometry(t.atom_slice(ca))

    if has_cell:
        try:
            tw = t[:]
            tw.image_molecules(inplace=True)
            out["imaged"] = geometry(tw.atom_slice(ca))
        except Exception as e:                     # noqa: BLE001
            out["imaged_error"] = f"{type(e).__name__}: {e}"[:120]
    return out


def verdict(d, atlas_rmsf):
    raw = d["raw"]
    img = d.get("imaged")
    notes = []

    collapsed = False
    if img:
        a, b = raw["max_pairwise_ca_rmsd_ang"], img["max_pairwise_ca_rmsd_ang"]
        if a > 0 and b < 0.6 * a:
            collapsed = True
            notes.append(f"re-imaging drops max RMSD {a:.1f} -> {b:.1f} A")

    # a single-frame jump comparable to half the box is a teleport, not motion
    jump = raw["rmax_max_single_frame_step"]
    if d["box_median_ang"] and jump > 0.25 * d["box_median_ang"]:
        notes.append(f"Rmax jumps {jump:.1f} A in one frame "
                     f"(box {d['box_median_ang']:.0f} A)")

    ratio = None
    if atlas_rmsf:
        ratio = raw["rmsf_mean_ang"] / atlas_rmsf
        notes.append(f"our RMSF {raw['rmsf_mean_ang']:.2f} A vs ATLAS "
                     f"{atlas_rmsf:.2f} A (x{ratio:.1f})")

    if collapsed or (d["box_median_ang"] and jump > 0.25 * d["box_median_ang"]):
        v = "PBC_ARTEFACT"
    elif ratio is not None and (ratio > 2.5 or ratio < 0.4):
        v = "INCONSISTENT_WITH_ATLAS"
    elif ratio is not None:
        v = "GENUINE_DISORDER" if atlas_rmsf > 3.0 else "GENUINE"
    else:
        v = "UNKNOWN_NO_REFERENCE"
    return v, notes


def cross_check(results):
    """External validation of our geometry pipeline against ATLAS's published
    per-system avg_RMSF.

    This is the one number in the whole campaign that is not self-referential:
    ATLAS computed its RMSF independently, with its own code, from its own
    simulations. If our RMSF - computed by our loader, our alignment and our
    fluctuation code from the same coordinates - tracks theirs, our geometry
    handling is correct. If it does not, every downstream metric is suspect.

    Replicates of the same system are averaged first, because ATLAS reports one
    value per system aggregated over its three runs. Comparing a single replicate
    to a three-replicate mean adds sampling noise that is ours, not the
    pipeline's.
    """
    from scipy.stats import pearsonr, spearmanr

    per_system = {}
    for key, d in results.items():
        if d.get("atlas_avg_rmsf"):
            per_system.setdefault(key.split(":")[0], (d["atlas_avg_rmsf"], []))
            per_system[key.split(":")[0]][1].append(d["raw"]["rmsf_mean_ang"])
    if len(per_system) < 5:
        return None

    names = sorted(per_system)
    db = np.array([per_system[k][0] for k in names])
    ours = np.array([float(np.mean(per_system[k][1])) for k in names])
    nrep = int(np.median([len(per_system[k][1]) for k in names]))
    err = 100.0 * (ours - db) / db

    pr, pp = pearsonr(db, ours)
    sr, _ = spearmanr(db, ours)
    slope, inter = np.polyfit(db, ours, 1)

    print(f"\n=== EXTERNAL CROSS-CHECK vs ATLAS avg_RMSF ===")
    print(f"  {len(names)} systems, {nrep} replicate(s) averaged each, "
          f"reference range {db.min():.2f}-{db.max():.2f} A")
    print(f"  Pearson r = {pr:.3f} (p={pp:.1e})   Spearman rho = {sr:.3f}")
    print(f"  OLS slope = {slope:.3f}  intercept = {inter:+.3f}   "
          f"(ideal 1.000 / 0.000)")
    print(f"  signed error: median {np.median(err):+.1f}%  mean {err.mean():+.1f}%")
    print(f"  |error|:      median {np.median(abs(err)):.1f}%  "
          f"max {abs(err).max():.1f}%")
    print(f"  within 10%: {(abs(err) <= 10).sum()}/{len(names)}   "
          f"within 20%: {(abs(err) <= 20).sum()}/{len(names)}")
    worst = np.argsort(-abs(err))[:3]
    print("  largest deviations: " + ", ".join(
        f"{names[i]} {err[i]:+.0f}%" for i in worst))

    if pr >= 0.85 and abs(slope - 1.0) <= 0.15:
        print("  VERDICT: our geometry pipeline reproduces an independent "
              "published measurement.")
        print("  Scatter at this level is sampling and convention "
              "(CA-only vs all-atom, frame stride),")
        print("  not a coordinate-handling fault - that would inflate errors by "
              "hundreds of percent.")
    else:
        print("  VERDICT: agreement is NOT adequate. Do not proceed until this "
              "is explained.")

    return {"n_systems": len(names), "n_replicates_averaged": nrep,
            "pearson_r": float(pr), "spearman_rho": float(sr),
            "ols_slope": float(slope), "ols_intercept": float(inter),
            "median_signed_error_pct": float(np.median(err)),
            "median_abs_error_pct": float(np.median(abs(err))),
            "max_abs_error_pct": float(abs(err).max()),
            "per_system": {k: {"atlas": per_system[k][0],
                               "ours_mean": float(np.mean(per_system[k][1])),
                               "ours_replicates": per_system[k][1]}
                           for k in names}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--only", nargs="+", default=None,
                    help="Diagnose only these system directories")
    ap.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--replicates", type=int, default=1,
                    help="Trajectories per system to check (default 1)")
    ap.add_argument("--out", default="validation/phase3_traj_diagnosis.json")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser()
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 1

    info = load_info_tsv(root)
    if info:
        print(f"cross-reference: atlas_info.tsv ({len(info)} systems)\n")
    else:
        print("! atlas_info.tsv not found in the data root - the external RMSF\n"
              "  cross-check will be skipped. Re-run phase3_fetch_atlas.py to "
              "save it.\n")

    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if args.only:
        want = set(args.only)
        dirs = [p for p in dirs if p.name in want]
    if not dirs:
        print(f"No system directories under {root}")
        return 1

    results, tally = {}, {}
    print(f"{'system':10s} {'res':>4s} {'box':>6s} {'maxRMSD':>8s} {'imaged':>8s} "
          f"{'Rg':>6s} {'jump':>6s} {'RMSFus':>7s} {'RMSFdb':>7s}  verdict")
    print("-" * 96)
    for p in dirs:
        tops = sorted(p.glob("*.pdb"))
        trjs = sorted(list(p.glob("*.xtc")) + list(p.glob("*.dcd")))
        if not tops or not trjs:
            continue
        for trj in trjs[: args.replicates]:
            key = f"{p.name}:{trj.stem}"
            try:
                d = diagnose_one(tops[0], trj, args.max_frames)
            except Exception as e:                 # noqa: BLE001
                print(f"{p.name:10s}  FAILED {type(e).__name__}: {e}"[:96])
                continue
            atlas_rmsf = _f(info.get(p.name, {}).get("avg_RMSF"))
            v, notes = verdict(d, atlas_rmsf)
            d["verdict"], d["notes"], d["atlas_avg_rmsf"] = v, notes, atlas_rmsf
            results[key] = d
            tally[v] = tally.get(v, 0) + 1

            raw = d["raw"]
            img = d.get("imaged", {})
            print(f"{p.name:10s} {d['n_residues']:4d} "
                  f"{(d['box_median_ang'] or 0):6.0f} "
                  f"{raw['max_pairwise_ca_rmsd_ang']:8.2f} "
                  f"{img.get('max_pairwise_ca_rmsd_ang', float('nan')):8.2f} "
                  f"{raw['rg_median']:6.1f} "
                  f"{raw['rmax_max_single_frame_step']:6.1f} "
                  f"{raw['rmsf_mean_ang']:7.2f} "
                  f"{(atlas_rmsf if atlas_rmsf else float('nan')):7.2f}  {v}",
                  flush=True)

    xc = cross_check(results)

    print("\n=== VERDICTS ===")
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {k:26s} {v}")

    if tally.get("PBC_ARTEFACT"):
        print("\n*** Periodic-boundary breakage detected. ***")
        print("The suitability screen and anything computed from these")
        print("coordinates is measuring box jumps, not conformational change.")
        print("Re-run the screen with --make-whole before reading any result.")
    elif tally.get("GENUINE_DISORDER"):
        print("\nThe spread is real, but these are the disordered tail of ATLAS.")
        print("A method validated only here does not generalise to folded")
        print("proteins. Stratify the sample across the avg_RMSF distribution")
        print("rather than taking the top of it.")

    Path(args.out).write_text(json.dumps(
        {"per_trajectory": results, "cross_check_vs_atlas": xc}, indent=2))
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
