"""Phase 8, step 0 - IS A GEOMETRICALLY INVISIBLE SEAM CONSTRUCTIBLE AT ALL?

Section 12 retracted the Phase 3 headline because the spliced junctions sat at
the 100th percentile of ordinary frame-to-frame steps: 3.43 A against a 1.65 A
median. A four-line abs(diff) detector won because it was handed exactly the
signal it reads.

The proposed fix (Phase 8) is a reject-and-recut benchmark: only accept a splice
whose junction step falls INSIDE the ordinary step distribution, so a difference
detector is at chance BY CONSTRUCTION and only a model of the dynamics can flag
the seam.

That fix has a precondition nobody has checked: such splice pairs must EXIST.
Consecutive frames in a 100 ns ATLAS trajectory are 100 ps apart and barely
move. If no two temporally distant frames are ever as close as two consecutive
frames, then a geometrically invisible seam is IMPOSSIBLE to construct at this
sampling - and Phase 3 did not fail because of a design mistake, it failed
because the experiment it was trying to run cannot be run on this data.

This script answers that question and nothing else. It builds no benchmark and
scores no detector.

WHAT IS MEASURED, per trajectory
  ordinary step   : RMSD between consecutive frames, t -> t+1  (the thing a
                    difference detector reads as "normal")
  candidate splice: RMSD between frames i, j with |i - j| >= MIN_GAP frames,
                    i.e. a genuine temporal discontinuity
  A candidate is ADMISSIBLE at percentile P if its step is <= the Pth percentile
  of ordinary steps in BOTH the RMSD space and the standardised feature space
  used by abs_diff_oneliner. Both, because the benchmark must be invisible to
  both trivial detectors, not just one.

DECISION RULE (fixed before running)
  A trajectory is CONSTRUCTIBLE at P if it admits >= MIN_CANDIDATES splice
  pairs that are also mutually usable, i.e. spread over >= N_JUNCTIONS distinct
  source regions. Phase 8 proceeds only if a majority of trajectories are
  constructible at P = 50. If they are constructible only at P = 90+, the seams
  are not invisible and Phase 8 is abandoned - that outcome is reported as a
  finding, not worked around.

Usage
  python validation/phase8_feasibility.py --data_root /path/to/atlas [--limit N]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

MIN_GAP = 100          # frames; a splice closer than this in time is not an anomaly
PERCENTILES = [10, 25, 50, 75, 90, 99]
MIN_CANDIDATES = 50
N_JUNCTIONS = 8
MAJORITY = 0.50


def feature_matrix(traj):
    """The same feature space score_all() builds, so the probe measures the
    space abs_diff_oneliner actually reads."""
    import mdtraj as md
    from features.compute_md_features import features_to_matrix

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
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


def probe(top, xtc, max_frames=1200):
    import mdtraj as md

    traj = md.load(str(xtc), top=str(top))
    if len(traj) > max_frames:
        traj = traj[:: int(np.ceil(len(traj) / max_frames))]
    traj = traj.superpose(traj, 0)
    n = len(traj)

    # full pairwise RMSD, in Angstrom
    R = np.stack([md.rmsd(traj, traj, i) for i in range(n)]) * 10.0
    Xz = feature_matrix(traj)
    # pairwise feature step = max abs difference over features (what the
    # one-liner reads, but between arbitrary pairs rather than consecutive ones)
    F = np.zeros((n, n))
    for i in range(n):
        F[i] = np.abs(Xz - Xz[i]).max(axis=1)

    ordinary_r = np.array([R[t, t + 1] for t in range(n - 1)])
    ordinary_f = np.array([F[t, t + 1] for t in range(n - 1)])

    # candidate splices: temporally distant pairs
    ii, jj = np.triu_indices(n, k=MIN_GAP)
    cand_r, cand_f = R[ii, jj], F[ii, jj]

    out = {
        "n_frames": n,
        "n_candidate_pairs": int(len(ii)),
        "ordinary_rmsd_median_ang": round(float(np.median(ordinary_r)), 3),
        "ordinary_feat_median": round(float(np.median(ordinary_f)), 3),
        "min_candidate_rmsd_ang": round(float(cand_r.min()), 3),
        "candidate_rmsd_median_ang": round(float(np.median(cand_r)), 3),
        "levels": {},
    }
    # where does the CLOSEST temporally distant pair sit inside the ordinary
    # step distribution? This single number decides the whole question.
    out["min_candidate_percentile_in_ordinary"] = round(
        float((ordinary_r < cand_r.min()).mean() * 100), 1)

    for P in PERCENTILES:
        thr_r = float(np.percentile(ordinary_r, P))
        thr_f = float(np.percentile(ordinary_f, P))
        ok = (cand_r <= thr_r) & (cand_f <= thr_f)
        n_ok = int(ok.sum())
        # how spread out are the admissible sources? A benchmark needs
        # N_JUNCTIONS splices from distinct regions, not 50 variants of one.
        regions = len(np.unique(ii[ok] // 50)) if n_ok else 0
        out["levels"][str(P)] = {
            "rmsd_threshold_ang": round(thr_r, 3),
            "n_admissible_pairs": n_ok,
            "n_distinct_source_regions": int(regions),
            "constructible": bool(n_ok >= MIN_CANDIDATES
                                  and regions >= N_JUNCTIONS),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="validation/phase8_feasibility.json")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser()
    systems = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        pdb = next(iter(d.glob("*.pdb")), None)
        xtc = sorted(d.glob("*_R1.xtc")) or sorted(d.glob("*.xtc"))
        if pdb and xtc:
            systems.append((d.name, pdb, xtc[0]))
    if args.limit:
        systems = systems[: args.limit]

    print(f"probing {len(systems)} systems  (MIN_GAP={MIN_GAP} frames)\n")
    results, failed = {}, []
    for sid, top, xtc in systems:
        try:
            r = probe(top, xtc)
            results[sid] = r
            lv = r["levels"]["50"]
            print(f"  {sid:10s} ordinary median {r['ordinary_rmsd_median_ang']:6.3f} A | "
                  f"closest distant pair {r['min_candidate_rmsd_ang']:6.3f} A "
                  f"(pct {r['min_candidate_percentile_in_ordinary']:5.1f}) | "
                  f"P50 admissible {lv['n_admissible_pairs']:6d} "
                  f"regions {lv['n_distinct_source_regions']:3d} "
                  f"{'OK' if lv['constructible'] else '--'}")
        except Exception as e:
            failed.append((sid, str(e)))
            print(f"  {sid:10s} FAILED {e}")

    summary = {}
    for P in PERCENTILES:
        frac = float(np.mean([r["levels"][str(P)]["constructible"]
                              for r in results.values()])) if results else 0.0
        summary[str(P)] = {
            "frac_constructible": round(frac, 3),
            "median_admissible_pairs": float(np.median(
                [r["levels"][str(P)]["n_admissible_pairs"]
                 for r in results.values()])) if results else 0.0,
        }

    verdict = ("PROCEED" if summary.get("50", {}).get("frac_constructible", 0)
               >= MAJORITY else "ABANDON")
    lowest_ok = next((P for P in PERCENTILES
                      if summary[str(P)]["frac_constructible"] >= MAJORITY), None)

    print("\n" + "=" * 70)
    print("CONSTRUCTIBILITY (fraction of trajectories admitting a valid benchmark)")
    for P in PERCENTILES:
        print(f"  P{P:<3d} {summary[str(P)]['frac_constructible']:.2f}   "
              f"median admissible pairs {summary[str(P)]['median_admissible_pairs']:.0f}")
    print(f"\n  lowest percentile with a majority constructible: "
          f"{('P' + str(lowest_ok)) if lowest_ok else 'NONE'}")
    print(f"  DECISION (pre-registered, P50 majority): {verdict}")
    if verdict == "ABANDON":
        print("\n  A geometrically invisible seam cannot be built at this sampling.")
        print("  Phase 3 did not fail on design. The experiment is not runnable")
        print("  on 100 ns trajectories, and that is the finding.")
    print("=" * 70)

    out = {
        "min_gap_frames": MIN_GAP,
        "decision_rule": (f"proceed iff >= {MAJORITY:.0%} of trajectories are "
                          f"constructible at P50, where constructible means "
                          f">= {MIN_CANDIDATES} admissible pairs spread over "
                          f">= {N_JUNCTIONS} distinct source regions, admissible "
                          f"in BOTH RMSD and feature space"),
        "verdict": verdict,
        "lowest_majority_percentile": lowest_ok,
        "summary": summary,
        "per_system": results,
        "failed": failed,
    }
    Path(ROOT / args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
