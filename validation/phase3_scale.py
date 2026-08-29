#!/usr/bin/env python3
"""
Phase 3 — scale the Phase 2 test to many proteins on public long trajectories.

WHY
---
Phase 2 established the method's distinctive claim on ONE system (villin, 250
frames): transition surprise reaches AUROC 0.741 on geometrically invisible
junctions while every geometric detector sits at chance. It missed the
pre-registered 0.75 bar by 0.009, with a replicate spread of +/-0.113 - a
sampling-limited near-miss, not a refutation.

Phase 3 resolves it the only legitimate way: MORE DATA, SAME BAR. It runs the
identical Phase 2 harness across many proteins from ATLAS or mdCATH, where
trajectories are 2-3 orders of magnitude longer, and reports a DISTRIBUTION
rather than a single number.

This simultaneously fixes the Layer-2 sampling deficit (OI-1): with real
sampling, Chapman-Kolmogorov intervals narrow, implied timescales can plateau,
and VAMP-2 stops saturating on a degenerate split.

  PART A  Phase 2 temporal scrambling, per protein, aggregated.
  PART B  mdCATH temperature-replicate rare-event test: conformations abundant
          at high T but rare at low T are rare states labelled BY THE DATASET,
          not by our model. Only runs if multi-temperature data is present.

DATA
----
This script does NOT download anything. Fetch trajectories yourself and point
it at the directory:

  ATLAS   https://www.dsimb.inserm.fr/ATLAS   (~1500 proteins, 3 x 100 ns)
  mdCATH  https://huggingface.co/datasets/compsciencelab/mdCATH

Accepted layouts (auto-detected):
  <root>/<SYSTEM>/*.pdb + *.xtc|*.dcd            (ATLAS-style, one dir per system)
  <root>/<SYSTEM>_<TEMP>/*.pdb + *.xtc           (temperature replicates -> PART B)
  <root>/*.h5                                     (mdCATH HDF5, needs h5py)

PRE-REGISTERED CRITERIA (identical bar to Phase 2 - do NOT change these)
  C1  median surprise AUROC across systems >= 0.75
  C2  surprise > best time-blind baseline on >= 80% of systems
  C3  median of every time-blind baseline within 0.40-0.60
  C4  Wilcoxon signed-rank p < 0.05, surprise vs best time-blind baseline

Usage
-----
  python validation/phase3_scale.py --data_root /path/to/atlas --limit 10   # pilot
  python validation/phase3_scale.py --data_root /path/to/atlas              # full
  python validation/phase3_scale.py --data_root /path/to/mdcath --temperature-test
"""
import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from validation.phase2_temporal_scramble import (  # noqa: E402
    scramble, junction_labels, score_all, metrics, TIME_BLIND)

MAX_FRAMES = 4000      # subsample long trajectories to keep MSM estimation sane
N_REPLICATES = 5       # per system (Phase 2 used 12 on a single system)
BAR = 0.75             # SAME bar as Phase 2 - never move this


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover(root: Path):
    """Return [(system_id, topology_path, trajectory_path, temperature_or_None)]."""
    out = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        tops = sorted(list(sub.glob("*.pdb")))
        trjs = sorted(list(sub.glob("*.xtc")) + list(sub.glob("*.dcd")))
        if not tops or not trjs:
            continue
        # temperature suffix, e.g. 1abcA00_413  -> ("1abcA00", 413)
        name, temp = sub.name, None
        if "_" in sub.name:
            head, tail = sub.name.rsplit("_", 1)
            if tail.isdigit():
                name, temp = head, int(tail)
        for trj in trjs:
            sid = f"{name}:{trj.stem}" if len(trjs) > 1 else name
            out.append((sid, tops[0], trj, temp))
    return out


def load_traj(top, trj, max_frames=MAX_FRAMES, make_whole=False):
    import mdtraj as md
    t = md.load(str(trj), top=str(top))
    # protein heavy atoms only - ATLAS/mdCATH may carry solvent
    sel = t.topology.select("protein")
    if len(sel) and len(sel) != t.n_atoms:
        t = t.atom_slice(sel)
    if make_whole:
        # Reassemble molecules split across the periodic boundary. Without this
        # a fragment that wraps round the box registers as a ~box-length
        # displacement, which RMSD reads as a huge conformational change. See
        # validation/phase3_diagnose_traj.py for how to tell the two apart.
        box_ok = (t.unitcell_lengths is not None
                  and float(np.median(t.unitcell_lengths)) * 10.0 >= 20.0)
        if box_ok:
            try:
                t.image_molecules(inplace=True)
            except Exception as exc:                       # noqa: BLE001
                print(f"    ! image_molecules failed ({type(exc).__name__}); "
                      f"using raw coordinates")
        else:
            print("    ! --make-whole requested but no usable unit cell; "
                  "using raw coordinates")
    if len(t) > max_frames:
        t = t[:: max(1, len(t) // max_frames)][:max_frames]
    return t


def scale_hyperparams(n_frames):
    """Hyper-parameters must grow with sampling; the toy defaults (10 clusters,
    lag 5) were tuned for 100-frame trajectories and under-resolve long ones."""
    n_clusters = int(np.clip(np.sqrt(n_frames) * 1.5, 10, 200))
    lag = int(np.clip(n_frames // 200, 5, 50))
    return {"n_clusters": n_clusters, "lag_msm": lag, "lag_tica": lag,
            "dim": 4, "k": max(5, int(np.sqrt(n_frames) / 2))}


# ---------------------------------------------------------------------------
# PART A — Phase 2 harness per system
# ---------------------------------------------------------------------------
def run_system(traj, seed_base=500):
    """Run BOTH the near-mode test and the identity-null control per system.

    The null control is not optional. It keeps blocks in their original order,
    so there are no real junctions, and labels the same positions. Every method
    must be at chance. If the null is elevated, the near-mode numbers for that
    system are measuring block-position bias (e.g. slow conformational drift in
    long trajectories) rather than anomaly detection, and must not be read as a
    method result.
    """
    import mdtraj as md

    hp = scale_hyperparams(len(traj))
    R = np.zeros((len(traj), len(traj)), dtype=np.float32)
    for i in range(len(traj)):
        R[i] = md.rmsd(traj, traj, i)

    out = {"hyperparams": hp}
    for mode in ("near", "identity_null"):
        per = {}
        for rep in range(N_REPLICATES):
            rng = np.random.default_rng(seed_base + rep)
            order, junc = scramble(traj, mode, rng, R)
            new = traj[order]
            y = junction_labels(len(new), junc, hp["lag_msm"])
            if y.sum() == 0 or y.sum() == len(y):
                continue
            sc = score_all(new, lag_tica=hp["lag_tica"], dim=hp["dim"],
                           n_clusters=hp["n_clusters"], lag_msm=hp["lag_msm"],
                           k=hp["k"])
            for name, s in sc.items():
                au, _ = metrics(y, s)
                per.setdefault(name, []).append(au)
        if not per:
            return None
        key = "auroc" if mode == "near" else "auroc_null"
        out[key] = {n: round(float(np.mean(v)), 3) for n, v in per.items()}

    # drift diagnostic: how much does the structure change over the trajectory?
    n = len(traj)
    out["drift_rmsd_first_vs_last_decile"] = round(
        float(np.mean(R[: n // 10, -n // 10:])), 4)
    return out


# ---------------------------------------------------------------------------
# PART B — temperature-replicate rare-event test (mdCATH)
# ---------------------------------------------------------------------------
def temperature_test(systems, low_T=320, high_T=413, n_seed=40):
    """Splice high-T frames into a low-T background. Unlike Phase 1 this uses
    PUBLISHED trajectories at both temperatures, so the rare-state labels come
    from the dataset rather than from a simulation we ran ourselves."""
    import mdtraj as md
    from sklearn.metrics import roc_auc_score

    by_name = {}
    for sid, top, trj, temp in systems:
        if temp is None:
            continue
        by_name.setdefault(sid.split(":")[0], {})[temp] = (top, trj)
    usable = {k: v for k, v in by_name.items() if low_T in v and high_T in v}
    if not usable:
        return {"skipped": f"no systems with both {low_T}K and {high_T}K replicates"}

    rows = {}
    for name, v in usable.items():
        try:
            cold = load_traj(*v[low_T])
            hot = load_traj(*v[high_T])
            if cold.n_atoms != hot.n_atoms:
                continue
            rng = np.random.default_rng(11)
            start = int(rng.integers(0, max(1, len(hot) - n_seed)))
            ins = len(cold) // 2
            new = md.join([cold[:ins], hot[start:start + n_seed], cold[ins:]])
            y = np.zeros(len(new), dtype=int)
            y[ins:ins + n_seed] = 1
            hp = scale_hyperparams(len(new))
            sc = score_all(new, lag_tica=hp["lag_tica"], dim=hp["dim"],
                           n_clusters=hp["n_clusters"], lag_msm=hp["lag_msm"], k=hp["k"])
            rows[name] = {n: round(float(roc_auc_score(y, np.nan_to_num(
                np.asarray(s, float), nan=0.0))), 3) for n, s in sc.items()}
        except Exception as exc:
            rows[name] = {"error": str(exc)[:120]}
    return rows


# ---------------------------------------------------------------------------
def aggregate(per_system):
    """Distribution across systems + the four pre-registered criteria.

    IMPORTANT: replicates of the same protein (ATLAS ships R1/R2/R3) are NOT
    independent samples. They are averaged into one value per PROTEIN before any
    statistic is computed, otherwise the Wilcoxon test is inflated ~3x by
    pseudo-replication.
    """
    names = sorted({n for v in per_system.values() for n in v["auroc"]})

    by_protein, by_protein_null = {}, {}
    for sid, v in per_system.items():
        prot = sid.split(":")[0]
        by_protein.setdefault(prot, []).append(v["auroc"])
        if "auroc_null" in v:
            by_protein_null.setdefault(prot, []).append(v["auroc_null"])
    collapsed = {prot: {n: float(np.mean([a[n] for a in lst if n in a]))
                        for n in names if any(n in a for a in lst)}
                 for prot, lst in by_protein.items()}

    dist = {n: np.array([v[n] for v in collapsed.values() if n in v])
            for n in names}

    collapsed_null = {prot: {n: float(np.mean([a[n] for a in lst if n in a]))
                             for n in names if any(n in a for a in lst)}
                      for prot, lst in by_protein_null.items()}
    dist_null = {n: np.array([v[n] for v in collapsed_null.values() if n in v])
                 for n in names} if collapsed_null else {}

    sur = dist["surprise_only"]
    blind_names = [n for n in TIME_BLIND if n in dist and n != "rarity_only"]
    # rarity excluded from "time-blind": pi is estimated from transition counts,
    # so it inherits temporal information (Phase 2 finding P2-3).
    best_blind = np.max(np.vstack([dist[n] for n in blind_names]), axis=0)

    c1 = float(np.median(sur)) >= BAR
    win = float(np.mean(sur > best_blind))
    c2 = win >= 0.80
    c3 = all(0.40 <= float(np.median(dist[n])) <= 0.60 for n in blind_names)
    try:
        p = float(wilcoxon(sur, best_blind, alternative="greater").pvalue)
    except Exception:
        p = float("nan")
    c4 = p < 0.05

    return {
        "n_trajectories": len(per_system),
        "n_independent_proteins": len(collapsed),
        "note": ("statistics computed per PROTEIN; replicates averaged first to "
                 "avoid pseudo-replication"),
        "distribution": {n: {"median": round(float(np.median(v)), 3),
                             "mean": round(float(v.mean()), 3),
                             "q25": round(float(np.percentile(v, 25)), 3),
                             "q75": round(float(np.percentile(v, 75)), 3),
                             "frac_ge_bar": round(float(np.mean(v >= BAR)), 3)}
                         for n, v in dist.items()},
        "null_control_median": {n: round(float(np.median(v)), 3)
                                for n, v in dist_null.items()} if dist_null else {},
        "C5_null_control_at_chance": {
            "pass": bool(dist_null and all(
                0.40 <= float(np.median(v)) <= 0.60 for v in dist_null.values())),
            "note": ("MUST pass for any other criterion to be interpretable. An "
                     "elevated null means block-position bias (e.g. slow drift), "
                     "not anomaly detection.")} if dist_null else {},
        "acceptance": {
            "C1_median_surprise_ge_0.75": {"pass": bool(c1),
                                           "value": round(float(np.median(sur)), 3)},
            "C2_beats_best_blind_on_80pct": {"pass": bool(c2),
                                             "win_fraction": round(win, 3)},
            "C3_blind_medians_in_band": {"pass": bool(c3)},
            "C4_wilcoxon_p_lt_0.05": {"pass": bool(c4), "p_value": p},
            "overall_pass": bool(c1 and c2 and c3 and c4),
        },
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Suitability screen  (--screen-only)
# ---------------------------------------------------------------------------
def screen(systems, args):
    """Preflight every discovered system and report which ones can carry an MSM.

    This exists because every system validated so far (1UBQ, villin, the 9UNN /
    9O6O pair) screened UNSUITABLE: >95% of frames sit within 2 A of a single
    medoid. A single-basin ensemble has no conformational states to detect, so
    disconnected microstates, an unresolved stationary distribution and a
    powerless CK test are the expected consequence, not a bug in the estimator.

    Nothing downstream of this is worth running until systems pass. Verdicts
    come from msm.preflight.assess_suitability - the same advisory used inside
    the production pipeline, so the screen and the run agree by construction.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from msm.preflight import assess_suitability

    rows, failed = [], {}
    for i, (sid, top, trj, temp) in enumerate(systems, 1):
        try:
            traj = load_traj(top, trj, args.max_frames, args.make_whole)
            a = assess_suitability(traj)
            a["system"] = sid
            a["temperature"] = temp
            rows.append(a)
            print(f"[{i}/{len(systems)}] {sid:28s} {a['verdict']:11s} "
                  f"n={a['n_frames']:5d} res={a['n_residues']:4d} "
                  f"maxRMSD={a['max_pairwise_ca_rmsd_ang']:6.2f} A  "
                  f"basin={a['fraction_in_single_basin']*100:5.1f}%  "
                  f"clusters={a.get('n_basins', 0):3d} "
                  f"top={a.get('largest_basin_occupancy', 0)*100:5.1f}% "
                  f"pop={a.get('n_basins_ge_5pct', 0):2d}", flush=True)
        except Exception as exc:
            failed[sid] = f"{type(exc).__name__}: {exc}"[:150]
            print(f"[{i}/{len(systems)}] {sid:28s} FAILED: {failed[sid]}", flush=True)

    if not rows:
        print("\nNo system could be screened.")
        return 1

    by = {v: [r for r in rows if r["verdict"] == v]
          for v in ("suitable", "marginal", "unsuitable")}
    print("\n=== SCREEN ===")
    for v in ("suitable", "marginal", "unsuitable"):
        print(f"  {v:11s} {len(by[v]):4d}  ({len(by[v])/len(rows)*100:.0f}%)")

    # ---- aggregate replicates to the protein ------------------------------
    # ATLAS ships three independent runs per protein and the verdict disagrees
    # between them often enough that a per-trajectory list is misleading:
    # 2hnu_A is single-basin in R1 and eight-basin in R2. The decision unit for
    # the sweep is the PROTEIN, and treating replicates as independent systems
    # is the same pseudo-replication that inflated the earlier Wilcoxon test.
    # A protein is usable if a majority of its replicates are.
    prot = {}
    for r in rows:
        prot.setdefault(r["system"].split(":")[0], []).append(r)

    protein_rows = []
    for name, reps in sorted(prot.items()):
        n_ok = sum(1 for r in reps if r["verdict"] in ("suitable", "marginal"))
        protein_rows.append({
            "protein": name,
            "n_replicates": len(reps),
            "n_usable_replicates": n_ok,
            "verdicts": [r["verdict"] for r in reps],
            "median_n_basins_ge_5pct": float(np.median(
                [r.get("n_basins_ge_5pct", 0) for r in reps])),
            "median_largest_basin_occupancy": float(np.median(
                [r.get("largest_basin_occupancy", 0) for r in reps])),
            "median_max_rmsd_ang": float(np.median(
                [r["max_pairwise_ca_rmsd_ang"] for r in reps])),
            "usable": n_ok * 2 >= len(reps),
        })

    n_usable_prot = sum(1 for p in protein_rows if p["usable"])
    print(f"\n=== BY PROTEIN ({len(protein_rows)} proteins) ===")
    print(f"  {n_usable_prot} usable (majority of replicates suitable/marginal), "
          f"{len(protein_rows) - n_usable_prot} not")
    disagree = [p for p in protein_rows
                if 0 < p["n_usable_replicates"] < p["n_replicates"]]
    if disagree:
        print(f"  {len(disagree)} protein(s) DISAGREE across replicates - the "
              f"ensemble is not converged at 100 ns:")
        for p in disagree[:12]:
            print(f"    {p['protein']:9s} {p['n_usable_replicates']}/"
                  f"{p['n_replicates']} usable   {', '.join(p['verdicts'])}")

    print(f"\n{'protein':9s} {'usable':>7s} {'basins':>7s} {'top%':>6s} "
          f"{'maxRMSD':>8s}")
    print("-" * 44)
    for p in sorted(protein_rows, key=lambda p: (-p["usable"],
                                                 -p["median_n_basins_ge_5pct"])):
        print(f"{p['protein']:9s} {p['n_usable_replicates']}/{p['n_replicates']:<5d} "
              f"{p['median_n_basins_ge_5pct']:7.1f} "
              f"{p['median_largest_basin_occupancy']*100:6.1f} "
              f"{p['median_max_rmsd_ang']:8.2f}")

    usable = by["suitable"] + by["marginal"]
    # rank by how MULTI-BASIN a system is, not by how far it travels. Widest
    # spread was the old key and it promoted diffusive chains to the top.
    usable.sort(key=lambda r: (-r.get("n_basins_ge_5pct", 0),
                               -r.get("largest_basin_occupancy", 0)))
    if usable:
        print(f"\n  best candidates (most populated conformational basins first):")
        for r in usable[:20]:
            print(f"    {r['system']:28s} {r['verdict']:11s} "
                  f"maxRMSD={r['max_pairwise_ca_rmsd_ang']:6.2f} A  "
                  f"top cluster={r.get('largest_basin_occupancy', 0)*100:5.1f}%  "
                  f"populated={r.get('n_basins_ge_5pct', 0)}")
    else:
        print("\n  *** No system is suitable or marginal. ***")
        print("  The pipeline's premise - that the ensemble contains more than one")
        print("  conformational basin - does not hold on this sample. Widen the")
        print("  fetch (--select rmsf without --max-len, or --kind protein for the")
        print("  10000-frame trajectories) before concluding anything about the")
        print("  method itself.")

    out = {"n_screened": len(rows),
           "counts": {v: len(by[v]) for v in by},
           "usable_systems": [r["system"] for r in usable],
           "n_proteins": len(protein_rows),
           "n_usable_proteins": n_usable_prot,
           "usable_proteins": [p["protein"] for p in protein_rows if p["usable"]],
           "proteins": protein_rows,
           "rows": rows, "failed": failed}
    Path(args.screen_out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {args.screen_out}")

    if n_usable_prot:
        ranked = sorted((p for p in protein_rows if p["usable"]),
                        key=lambda p: (-p["n_usable_replicates"],
                                       -p["median_n_basins_ge_5pct"]))
        ids = " ".join(p["protein"] for p in ranked)
        print(f"\nNext - run the full harness on the {len(ranked)} usable protein(s):")
        print(f"  python validation/phase3_scale.py --data_root {args.data_root} \\")
        print(f"      --only {ids}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True,
                    help="Directory of downloaded ATLAS / mdCATH systems")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N systems (pilot run)")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Process only these system ids (as printed by --screen-only)")
    ap.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--make-whole", action="store_true",
                    help="Reassemble molecules split across the periodic boundary "
                         "before any analysis. Run phase3_diagnose_traj.py first "
                         "to find out whether you need it.")
    ap.add_argument("--temperature-test", action="store_true",
                    help="Also run PART B (needs multi-temperature replicates)")
    ap.add_argument("--screen-only", action="store_true",
                    help="Run the suitability preflight on every discovered system "
                         "and stop. Cheap. Use this to find multi-basin systems "
                         "BEFORE spending hours on the full harness.")
    ap.add_argument("--screen-out", default="validation/phase3_screen.json")
    ap.add_argument("--out", default="validation/phase3_results.json")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser()
    if not root.exists():
        print(f"Data directory not found: {root}\n\n"
              "Phase 3 needs trajectories you have already downloaded. Nothing is\n"
              "fetched automatically. Get them from:\n"
              "  ATLAS   https://www.dsimb.inserm.fr/ATLAS\n"
              "  mdCATH  https://huggingface.co/datasets/compsciencelab/mdCATH\n\n"
              "Then arrange them as:\n"
              f"  {root}/<SYSTEM>/<something>.pdb\n"
              f"  {root}/<SYSTEM>/<something>.xtc   (or .dcd)\n\n"
              "For temperature replicates (enables --temperature-test), name the\n"
              "directories <SYSTEM>_<TEMP>, e.g. 1abcA00_320 and 1abcA00_413.")
        return 1
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 1
    systems = discover(root)
    if not systems:
        print(f"No systems found under {root}.\n"
              "Expected <root>/<SYSTEM>/ containing a .pdb and a .xtc/.dcd.")
        return 1
    if args.only:
        want = set(args.only)
        systems = [s for s in systems
                   if s[0] in want or s[0].split(":")[0] in want]
        if not systems:
            print(f"None of {sorted(want)} were found under {root}.")
            return 1
    if args.limit:
        systems = systems[: args.limit]
    print(f"discovered {len(systems)} system(s)\n")

    if args.screen_only:
        return screen(systems, args)

    per_system, failed = {}, {}
    for i, (sid, top, trj, temp) in enumerate(systems, 1):
        try:
            traj = load_traj(top, trj, args.max_frames, args.make_whole)
            if len(traj) < 200:
                failed[sid] = f"too short ({len(traj)} frames)"
                continue
            res = run_system(traj)
            if res is None:
                failed[sid] = "no usable replicates"
                continue
            per_system[sid] = res
            s = res["auroc"]
            print(f"[{i}/{len(systems)}] {sid:28s} n={len(traj):5d} "
                  f"surprise {s.get('surprise_only'):.3f}  "
                  f"jump {s.get('frame_to_frame_rmsd'):.3f}  "
                  f"rmsd_mean {s.get('rmsd_from_mean'):.3f}", flush=True)
        except Exception as exc:
            failed[sid] = f"{type(exc).__name__}: {exc}"[:150]
            print(f"[{i}/{len(systems)}] {sid:28s} FAILED: {failed[sid]}", flush=True)

    if not per_system:
        print("\nNo system produced results.")
        return 1

    out = {"bar": BAR, "n_replicates_per_system": N_REPLICATES,
           "per_system": per_system, "failed": failed}
    out.update(aggregate(per_system))

    if args.temperature_test:
        print("\n[PART B] temperature-replicate rare-event test ...")
        out["temperature_test"] = temperature_test(systems)

    n_prot = out.get("n_independent_proteins", 0)
    print(f"\n=== AGGREGATE ===")
    print(f"  {out.get('n_trajectories')} trajectories -> {n_prot} INDEPENDENT protein(s)")
    if n_prot < 10:
        print(f"  *** WARNING: n={n_prot} proteins. C1/C3/C4 are NOT interpretable ***")
        print(f"  *** below ~10 proteins. Per-replicate spread is large (observed  ***")
        print(f"  *** 0.4-0.9 for geometric detectors on the SAME protein).        ***")

    nc = out.get("null_control_median", {})
    if nc:
        bad = [f"{k}={v:.3f}" for k, v in nc.items() if not 0.40 <= v <= 0.60]
        print("\n--- NULL CONTROL (must all be ~0.50) ---")
        for k, v in sorted(nc.items(), key=lambda kv: -kv[1]):
            mark = "  <-- ELEVATED" if not 0.40 <= v <= 0.60 else ""
            print(f"  {k:22s} {v:.3f}{mark}")
        print("  VERDICT: " + ("construction valid - junctions carry no positional bias"
                               if not bad else "INVALID - " + ", ".join(bad)))

    print("\n=== NEAR MODE (median AUROC across proteins) ===")
    for n, v in sorted(out["distribution"].items(),
                       key=lambda kv: -kv[1]["median"]):
        if n == "rarity_only":
            tag = "  (excluded from C3: pi inherits temporal info, P2-3)"
        elif n in TIME_BLIND:
            tag = "  (time-blind)"
        else:
            tag = ""
        print(f"  {n:22s} {v['median']:.3f}  [IQR {v['q25']:.3f}-{v['q75']:.3f}]"
              f"  {v['frac_ge_bar']*100:.0f}% >= bar{tag}")
    print("\nACCEPTANCE:", json.dumps(out["acceptance"], indent=2))

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
