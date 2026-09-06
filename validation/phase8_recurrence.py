"""Phase 8b - recurrence, measured so two datasets can actually be compared.

WHY THIS EXISTS (read before changing anything)
-----------------------------------------------
phase8_feasibility.py asked whether a geometrically invisible splice can be
built, by comparing candidate splices against the CONSECUTIVE-FRAME step
distribution. On ATLAS (100 ps between saved frames) that is the right
reference: a difference detector reads the trajectory as delivered, so a splice
must hide among the steps the detector actually sees.

It becomes a trap the moment two datasets are compared.

  ATLAS  saves a frame every 100 ps
  mdCATH saves a frame every   1 ns      (10x coarser)

A coarser stride means consecutive frames are further apart geometrically, so
the "ordinary step" bar rises, so almost any temporally distant pair slips under
it. Run the frame-based probe on mdCATH and it will report abundant recurrence
that is an artifact of the save interval, not a property of the protein. That is
the same triviality trap that produced the Phase 3 retraction (section 12),
arriving from a different direction.

So this script reports TWO numbers and refuses to collapse them:

  (A) NATIVE constructibility - reference = consecutive frames AS DELIVERED.
      Answers: "can a benchmark be built on this dataset?"
      Legitimately stride-dependent, because the detector is too.

  (B) MATCHED constructibility - the trajectory RE-DELIVERED at REF_LAG_NS
      (default 1 ns), with candidates AND reference both taken from the
      subsampled result. Answers: "could a benchmark be built if every dataset
      were delivered at the same physical stride?" Comparable across datasets.

  (C) PHYSICAL recurrence - full-resolution candidate pool against a 1 ns-lag
      reference. Answers: "does the protein ever revisit a conformation?"
      Reported for interpretation, DELIBERATELY EXCLUDED from the decision,
      because a benchmark cannot exploit recurrence it has to subsample away.

DECISION RULE (fixed before any mdCATH data is downloaded)
  Download the bulk of mdCATH only if it passes BOTH.
    - passes (A) only  -> the benchmark would be constructible only because the
                          frames are far apart in time. The anomaly hides behind
                          the save interval, not behind the physics. Weak test;
                          do not spend the bandwidth.
    - passes (B) only  -> real recurrence exists but the delivered stride is too
                          fine to hide a splice; re-save at a coarser stride.
    - passes both      -> download, and Phase 8's recut benchmark is back on.

Usage
  python validation/phase8_recurrence.py --atlas /path/to/atlas [--limit N]
  python validation/phase8_recurrence.py --mdcath /path/to/*.h5 --temps 320 450
"""
import argparse
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

REF_LAG_NS = 1.0        # physical lag defining a "matched ordinary step"
MIN_GAP_NS = 10.0       # a splice spanning less than this is not a temporal anomaly
PERCENTILES = [10, 25, 50, 75, 90, 99]
MIN_CANDIDATES = 50
N_JUNCTIONS = 8
MAJORITY = 0.50


def _pairwise_rmsd_ang(traj):
    import mdtraj as md
    traj = traj.superpose(traj, 0)
    n = len(traj)
    return np.stack([md.rmsd(traj, traj, i) for i in range(n)]) * 10.0


def recurrence(R, stride_ns):
    """Both measurements from one pairwise-RMSD matrix.

    R          pairwise RMSD in Angstrom, shape (n, n)
    stride_ns  physical time between consecutive rows of R
    """
    n = len(R)
    out = {"n_frames": n, "stride_ns": stride_ns,
           "total_ns": round(n * stride_ns, 1)}

    gap = max(1, int(round(MIN_GAP_NS / stride_ns)))
    if n <= gap + 2:
        return {**out, "error": f"too few frames for a {MIN_GAP_NS} ns gap"}
    ii, jj = np.triu_indices(n, k=gap)
    cand = R[ii, jj]

    # ---- (A) native: reference is consecutive frames as delivered ----------
    native = np.array([R[t, t + 1] for t in range(n - 1)])
    out["A_native"] = _levels(cand, native, ii, "native (consecutive frames)", n)

    # ---- (B) matched: the trajectory RE-DELIVERED at REF_LAG_NS ------------
    # CORRECTION 2026-08-31. The first version of this block compared candidate
    # splices drawn from the FULL-resolution pool against a coarse reference.
    # That is internally inconsistent and it is optimistic: if the trajectory is
    # delivered at 1 ns, only the retained frames can be spliced, and throwing
    # away 9 frames in 10 destroys the candidate pool at the same time as it
    # relaxes the bar. The inconsistent version reported ATLAS as 0.84
    # constructible; the self-consistent version below reports 0.00.
    # Subsample FIRST, then take both candidates and reference from the result.
    k = max(1, int(round(REF_LAG_NS / stride_ns)))
    Rs, s_ns = R[::k, ::k], stride_ns * k
    ns_ = len(Rs)
    gap_s = max(1, int(round(MIN_GAP_NS / s_ns)))
    if ns_ > gap_s + 2:
        i2, j2 = np.triu_indices(ns_, k=gap_s)
        ref2 = np.array([Rs[t, t + 1] for t in range(ns_ - 1)])
        out["B_matched"] = _levels(Rs[i2, j2], ref2, i2,
                                   f"re-delivered at {REF_LAG_NS} ns "
                                   f"({ns_} frames retained)", ns_)
        out["B_matched"]["n_frames_retained"] = ns_

    # ---- (C) physical recurrence: NOT a constructibility claim -------------
    # Does the protein ever come back to within a typical 1 ns displacement?
    # Full-resolution candidate pool against a 1 ns-lag reference. This is a
    # statement about the PHYSICS and it is deliberately NOT used in the
    # download decision, because a benchmark cannot exploit it (see B).
    ref_k = max(1, int(round(REF_LAG_NS / stride_ns)))
    phys = np.array([R[t, t + ref_k] for t in range(n - ref_k)])
    out["C_physical_recurrence"] = {
        "note": ("closest full-resolution distant pair vs a 1 ns-lag step "
                 "distribution. Physical recurrence only - does NOT imply a "
                 "benchmark can be built. See B."),
        "closest_pair_percentile": round(
            float((phys < cand.min()).mean() * 100), 1),
        "ref_step_median_ang": round(float(np.median(phys)), 3),
    }
    return out


def _levels(cand, ref, ii, label, n_frames=None):
    d = {"reference": label,
         "ref_step_median_ang": round(float(np.median(ref)), 3),
         "closest_distant_pair_ang": round(float(cand.min()), 3),
         # THE decisive number: where the closest temporally distant pair sits
         # inside the reference step distribution.
         "closest_pair_percentile": round(float((ref < cand.min()).mean() * 100), 1),
         "levels": {}}
    for P in PERCENTILES:
        thr = float(np.percentile(ref, P))
        ok = cand <= thr
        n_ok = int(ok.sum())
        # BUGFIX 2026-08-31: the bucket was hard-coded at 50 frames, which makes
        # ">= 8 distinct source regions" unsatisfiable on a 101-frame subsample
        # no matter what the data says - the coarse-stride arm of this probe was
        # failing by construction. The bucket now scales with trajectory length,
        # so "spread over the trajectory" means the same thing at every stride.
        nb = n_frames if n_frames else (int(ii.max()) + 2)
        bucket = max(1, nb // 20)
        regions = len(np.unique(ii[ok] // bucket)) if n_ok else 0
        d["levels"][str(P)] = {
            "threshold_ang": round(thr, 3),
            "n_admissible": n_ok,
            "n_regions": int(regions),
            "constructible": bool(n_ok >= MIN_CANDIDATES and regions >= N_JUNCTIONS),
        }
    return d


# ---------------------------------------------------------------------------
def load_atlas(d, max_frames=1200):
    import mdtraj as md
    pdb = next(iter(d.glob("*.pdb")), None)
    xtc = sorted(d.glob("*_R1.xtc")) or sorted(d.glob("*.xtc"))
    if not (pdb and xtc):
        return None
    traj = md.load(str(xtc[0]), top=str(pdb))
    stride = 0.1                                    # ATLAS: 100 ps per frame
    if len(traj) > max_frames:
        k = int(np.ceil(len(traj) / max_frames))
        traj, stride = traj[::k], stride * k
    return traj, stride


def load_mdcath(path, temp, repl=0, max_frames=1200):
    """mdCATH per-domain HDF5: /<temp>/<replica>/coords, plus root topology."""
    import h5py
    import mdtraj as md

    with h5py.File(path, "r") as f:
        dom = list(f.keys())[0] if len(f.keys()) == 1 else None
        g = f[dom] if dom else f
        tkey = str(temp)
        if tkey not in g:
            avail = [k for k in g.keys()]
            raise KeyError(f"temperature {tkey} not in {avail}")
        reps = sorted(g[tkey].keys())
        rg = g[tkey][reps[repl]]
        xyz = np.asarray(rg["coords"], dtype=np.float32)
        pdb_txt = g["pdbProteinAtoms"][()] if "pdbProteinAtoms" in g else g["pdb"][()]

    if isinstance(pdb_txt, bytes):
        pdb_txt = pdb_txt.decode()
    tmp = Path("/tmp/_mdcath_top.pdb")
    tmp.write_text(pdb_txt)
    top = md.load(str(tmp)).topology
    if xyz.shape[1] != top.n_atoms:
        raise ValueError(f"atom mismatch: coords {xyz.shape[1]} vs top {top.n_atoms}")
    traj = md.Trajectory(xyz / 10.0, top)           # mdCATH coords are Angstrom
    stride = 1.0                                     # mdCATH: 1 ns per frame
    if len(traj) > max_frames:
        k = int(np.ceil(len(traj) / max_frames))
        traj, stride = traj[::k], stride * k
    return traj, stride


# ---------------------------------------------------------------------------
def summarise(results, key):
    lv = {}
    for P in PERCENTILES:
        vals = [r[key]["levels"][str(P)]["constructible"]
                for r in results.values() if key in r]
        lv[str(P)] = round(float(np.mean(vals)), 3) if vals else 0.0
    pcts = [r[key]["closest_pair_percentile"] for r in results.values() if key in r]
    return {"frac_constructible": lv,
            "closest_pair_percentile": {
                "min": round(float(np.min(pcts)), 1),
                "median": round(float(np.median(pcts)), 1),
                "max": round(float(np.max(pcts)), 1)} if pcts else {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas")
    ap.add_argument("--mdcath", help="glob for per-domain .h5 files")
    ap.add_argument("--temps", nargs="*", type=int, default=[320, 450])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="validation/phase8_recurrence.json")
    args = ap.parse_args()

    results = {}
    if args.atlas:
        dirs = sorted(p for p in Path(args.atlas).expanduser().iterdir() if p.is_dir())
        if args.limit:
            dirs = dirs[: args.limit]
        for d in dirs:
            got = load_atlas(d)
            if not got:
                continue
            traj, stride = got
            results[f"ATLAS:{d.name}"] = recurrence(_pairwise_rmsd_ang(traj), stride)
            _line(f"ATLAS:{d.name}", results[f"ATLAS:{d.name}"])

    if args.mdcath:
        files = sorted(glob.glob(str(Path(args.mdcath).expanduser())))
        if args.limit:
            files = files[: args.limit]
        for fp in files:
            for T in args.temps:
                sid = f"mdCATH:{Path(fp).stem}@{T}K"
                try:
                    traj, stride = load_mdcath(fp, T)
                    results[sid] = recurrence(_pairwise_rmsd_ang(traj), stride)
                    _line(sid, results[sid])
                except Exception as e:
                    print(f"  {sid:34s} FAILED {e}")

    groups = {}
    for sid, r in results.items():
        groups.setdefault(sid.split(":")[0].split("@")[0], {})[sid] = r
    for sid in list(results):
        if sid.startswith("mdCATH"):
            T = sid.split("@")[1]
            groups.setdefault(f"mdCATH@{T}", {})[sid] = results[sid]
    groups.pop("mdCATH", None)

    out = {"ref_lag_ns": REF_LAG_NS, "min_gap_ns": MIN_GAP_NS,
           "decision_rule": ("download mdCATH in bulk only if it is constructible "
                             "at P50 on BOTH the native and the matched reference; "
                             "native-only means the splice hides behind the 1 ns "
                             "save interval rather than behind the physics"),
           "groups": {g: {"n": len(rs),
                          "A_native": summarise(rs, "A_native"),
                          "B_matched": summarise(rs, "B_matched")}
                      for g, rs in groups.items()},
           "per_system": results}
    Path(ROOT / args.out).write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 74)
    for g, s in out["groups"].items():
        a50 = s["A_native"]["frac_constructible"]["50"]
        b50 = s["B_matched"]["frac_constructible"]["50"]
        print(f"{g:22s} n={s['n']:3d}   A native P50 {a50:.2f}   B matched P50 {b50:.2f}"
              f"   closest-pair pct: native {s['A_native']['closest_pair_percentile'].get('median')}"
              f" / matched {s['B_matched']['closest_pair_percentile'].get('median')}")
    print("=" * 74)
    print(f"saved -> {args.out}")


def _line(sid, r):
    if "error" in r:
        print(f"  {sid:34s} {r['error']}")
        return
    a, b = r["A_native"], r["B_matched"]
    print(f"  {sid:34s} {r['total_ns']:6.0f} ns @ {r['stride_ns']:.2f} ns/frame | "
          f"native pct {a['closest_pair_percentile']:5.1f} "
          f"P50 {'OK' if a['levels']['50']['constructible'] else '--'} | "
          f"matched pct {b['closest_pair_percentile']:5.1f} "
          f"P50 {'OK' if b['levels']['50']['constructible'] else '--'}")


if __name__ == "__main__":
    main()
