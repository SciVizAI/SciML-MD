#!/usr/bin/env python3
"""
Phase 5 — do the channels dissociate, and does fusion ever help?

WHY
---
Two "open defects" have sat in the register for weeks:

  P1-1  the pipeline loses to a trivial baseline on seeded STRUCTURAL anomalies
  P1-2  transition surprise is BELOW CHANCE on sustained rare states (0.351)

Phase 3 then produced the opposite ordering on kinetic transitions:

  channel              P1 sustained    P3 transition
  surprise_only            0.351           0.816
  rarity_only              0.803           0.606
  density_only             0.826           0.572
  rmsd_from_mean           1.000           0.487
  pipeline_fused           0.831           0.666

Read together these are not two failures and one success. They are a DOUBLE
DISSOCIATION: surprise scores the MOVE, rarity and density score the DWELL, and
each is near-useless in the other's regime. That is what -log P and 1-pi
respectively mean, so the behaviour is the mechanism working correctly, not a
defect.

But the table above is not yet evidence. Phase 1 is villin at 100 frames with
seeded excursions; Phase 3 is 25 ATLAS proteins at 1,001 frames with spliced
junctions. Different systems, different scales, different constructions. A
between-experiment comparison cannot support an interaction claim.

This phase makes it WITHIN-SUBJECT: the same proteins, the same hyper-
parameters, the same construction, with ONE knob - the DURATION of the
anomaly - interpolating between the two regimes.

CONSTRUCTION
    A contiguous window of D frames is sampled from the structurally distant
    quartile of the trajectory (distance from the CA medoid - a GEOMETRIC
    criterion, never the MSM, so the label cannot be circular) and relocated to
    a random interior position. The D relocated frames are the label.

      D = 1   the label is one frame between two strangers -> a pure transition
      D = 64  the label is a long coherent run in an unusual region; its
              internal transitions are ordinary -> a pure sustained rare state

    Every frame appears exactly once, so the static ensemble is unchanged by
    construction and a detector reading only the marginal distribution of
    structures is at chance for reasons of arithmetic, not luck.

CRITERIA, v2 (restated 2026-08-29 after the first run; see CORRECTION below).
Everything is computed on the RELOCATION EFFECT = AUROC(relocated) - AUROC(null),
per protein, which cancels the frame-identity confound.
    G1  Spearman rho(duration, surprise relocation effect) < 0, p < 0.05
    G2  surprise relocation effect > 0 at D=1 on >= 70% of proteins
    G3  surprise relocation effect exceeds EVERY other channel's at D=1
        (including abs_diff_oneliner, which beat surprise outright in Phase 3)
    G4  pipeline_fused does not exceed the best single channel at ANY duration
    G5  random_score within 0.40-0.60 in BOTH conditions at every duration
        [gates G1-G4]

WHAT A PASS BUYS
    A routing rule instead of a fusion rule, and a published operating envelope:
    "surprise for transitions, rarity/density for dwells, and here is the
    crossover". P1-1 and P1-2 stop being open defects and become the two ends of
    a characterised axis.

USAGE
    python validation/phase5_channel_dissociation.py --data_root ~/atlas --limit 12
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from validation.phase2_temporal_scramble import score_all, TIME_BLIND  # noqa: E402
from validation.phase3_scale import (  # noqa: E402
    discover, load_traj, scale_hyperparams)

DURATIONS = [1, 4, 16, 64]
N_REPLICATES = 3
CHANNELS = ["surprise_only", "rarity_only", "density_only",
            "frame_to_frame_rmsd", "rmsd_from_mean", "feature_zscore",
            "isolation_forest", "local_outlier_factor", "pipeline_fused",
            "abs_diff_oneliner", "random_score"]
SINGLE = ["surprise_only", "rarity_only", "density_only",
          "frame_to_frame_rmsd", "rmsd_from_mean", "feature_zscore",
          "isolation_forest", "local_outlier_factor", "abs_diff_oneliner"]

# ---------------------------------------------------------------------------
# CORRECTION, 2026-08-29, after the first run
#
# The first version scored the RELOCATED condition directly and required the
# null to sit at chance (old G5). That criterion cannot be met by this design
# and the run correctly failed it: the source window is deliberately drawn from
# the structurally distant quartile, so its frames are unusual REGARDLESS of
# whether they were moved. In the null, rmsd_from_mean scored 0.837 - it was
# reading frame identity, exactly as it should.
#
# Two effects are confounded in the relocated condition:
#     (a) IDENTITY  - these frames are structurally unusual
#     (b) RELOCATION - these frames are in the wrong temporal position
# The null isolates (a). The quantity of interest is therefore the DIFFERENCE,
#     relocation effect = AUROC(relocated) - AUROC(null),
# computed per protein, which cancels (a) and leaves (b).
#
# This is repair of a criterion that was unsatisfiable by construction, not a
# relaxation: the difference measure is STRICTER than the original, because a
# channel now has to beat its own identity-only score rather than beat chance.
# The first run's numbers are reported in the reportregardless.
# ---------------------------------------------------------------------------


def relocate_excursion(n_frames, dist_from_medoid, duration, rng, null=False):
    """Move a distant window of `duration` frames to a random interior slot.

    Returns (order, label_positions). The source window is chosen from the top
    quartile of distance-from-medoid so it is a genuine excursion, but sampled
    at random within that quartile rather than taken as the single extreme, so
    the construction is not tuned to whichever window a geometric detector
    would find easiest.

    With null=True the window is put back where it came from: same frames, same
    label positions, no relocation. Every channel must then be at chance.
    """
    d = np.asarray(dist_from_medoid, dtype=float)
    win = np.array([d[i:i + duration].mean()
                    for i in range(n_frames - duration + 1)])
    thresh = np.quantile(win, 0.75)
    cand = np.flatnonzero(win >= thresh)
    start = int(rng.choice(cand))
    block = list(range(start, start + duration))

    bset = set(block)
    rest = [i for i in range(n_frames) if i not in bset]
    if null:
        insert = start
    else:
        lo, hi = duration + 2, len(rest) - duration - 2
        if hi <= lo:
            return None, None
        # Re-insert far from where the window came from, so the relocation is a
        # real discontinuity rather than a near-identity shuffle. The required
        # separation must stay satisfiable: on a short trajectory 4*duration can
        # exceed the whole insertable range, which previously spun forever.
        sep = min(4 * duration, max(1, (hi - lo) // 3))
        insert = int(rng.integers(lo, hi))
        for _ in range(200):
            if abs(insert - start) >= sep:
                break
            insert = int(rng.integers(lo, hi))

    order = rest[:insert] + block + rest[insert:]
    labels = np.zeros(n_frames, dtype=bool)
    labels[insert:insert + duration] = True
    return np.asarray(order), labels


def medoid_distance(traj):
    import mdtraj as md
    ca = traj.atom_slice(traj.topology.select("name CA"))
    step = max(1, len(ca) // 200)
    sub = ca[::step]
    R = np.zeros((len(sub), len(sub)))
    for i in range(len(sub)):
        R[i] = md.rmsd(sub, sub, i)
    medoid_sub = int(np.argmin(R.sum(axis=1)))
    return md.rmsd(ca, sub, medoid_sub) * 10.0


def run_protein(traj, seed_base=1300):
    hp = scale_hyperparams(len(traj))
    dist = medoid_distance(traj)
    out = {}
    for mode in ("relocated", "null"):
        for D in DURATIONS:
            per = {}
            for rep in range(N_REPLICATES):
                rng = np.random.default_rng(seed_base + 17 * D + rep)
                order, y = relocate_excursion(len(traj), dist, D, rng,
                                              null=(mode == "null"))
                if order is None or y.sum() == 0 or y.sum() == len(y):
                    continue
                sc = score_all(traj[order], lag_tica=hp["lag_tica"],
                               dim=hp["dim"], n_clusters=hp["n_clusters"],
                               lag_msm=hp["lag_msm"], k=hp["k"])
                for name in CHANNELS:
                    if name not in sc:
                        continue
                    s = np.asarray(sc[name], float)
                    finite = np.isfinite(s)
                    if not finite.any():
                        continue
                    s = np.nan_to_num(s, nan=float(np.nanmin(s[finite])))
                    per.setdefault(name, []).append(float(roc_auc_score(y, s)))
            if per:
                out[f"{mode}_D{D}"] = {n: round(float(np.mean(v)), 3)
                                       for n, v in per.items()}
    return out or None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--screen", default="validation/phase3_screen.json")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--max_frames", type=int, default=4000)
    ap.add_argument("--out", default="validation/phase5_results.json")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser()
    want = set(args.only) if args.only else None
    if want is None and Path(args.screen).exists():
        want = set(json.loads(Path(args.screen).read_text())
                   .get("usable_proteins") or [])

    systems, seen, picked = discover(root), set(), []
    for sid, top, trj, _ in systems:
        base = sid.split(":")[0]
        if want and base not in want:
            continue
        if base in seen:
            continue
        seen.add(base)
        picked.append((base, top, trj))
    picked = picked[: args.limit]
    if not picked:
        print(f"No systems under {root}")
        return 1

    print(f"{len(picked)} protein(s) x {len(DURATIONS)} durations "
          f"x {N_REPLICATES} replicates, relocated + null\n")
    print(f"{'protein':10s} " + " ".join(f"D{D:<3d}sur D{D:<3d}rar" for D in DURATIONS))
    print("-" * (11 + 20 * len(DURATIONS)))

    per_protein = {}
    for name, top, trj in picked:
        try:
            traj = load_traj(top, trj, args.max_frames)
            r = run_protein(traj)
        except Exception as e:                                # noqa: BLE001
            print(f"{name:10s} FAILED {type(e).__name__}: {e}"[:70])
            continue
        if not r:
            continue
        per_protein[name] = r
        cells = []
        for D in DURATIONS:
            k = f"relocated_D{D}"
            cells.append(f"{r.get(k,{}).get('surprise_only', float('nan')):8.3f}"
                         f"{r.get(k,{}).get('rarity_only', float('nan')):8.3f}")
        print(f"{name:10s} " + " ".join(cells), flush=True)

    if len(per_protein) < 5:
        print("\nToo few proteins to judge.")
        return 1

    def col(mode, D, ch):
        return np.array([v[f"{mode}_D{D}"][ch] for v in per_protein.values()
                         if f"{mode}_D{D}" in v and ch in v[f"{mode}_D{D}"]])

    print(f"\n=== MEDIAN AUROC BY DURATION ({len(per_protein)} proteins) ===")
    print(f"{'channel':22s} " + "".join(f"{'D=' + str(D):>9s}" for D in DURATIONS))
    print("-" * (23 + 9 * len(DURATIONS)))
    med = {}
    for ch in CHANNELS:
        vals = [np.median(col("relocated", D, ch)) if len(col("relocated", D, ch))
                else np.nan for D in DURATIONS]
        med[ch] = vals
        print(f"{ch:22s} " + "".join(f"{v:9.3f}" for v in vals))

    # ---- relocation effect: cancels the frame-identity confound ----------
    def effect(ch, dd):
        a, b = col("relocated", dd, ch), col("null", dd, ch)
        n = min(len(a), len(b))
        return a[:n] - b[:n] if n else np.array([])

    print(f"\n=== RELOCATION EFFECT  (relocated - null) ===")
    print(f"{'channel':22s} " + "".join(f"{'D=' + str(dd):>9s}" for dd in DURATIONS))
    print("-" * (23 + 9 * len(DURATIONS)))
    eff = {}
    for ch in CHANNELS:
        vals = [float(np.median(effect(ch, dd))) if len(effect(ch, dd)) else np.nan
                for dd in DURATIONS]
        eff[ch] = vals
        print(f"{ch:22s} " + "".join(f"{v:+9.3f}" for v in vals))

    xs, ys = [], []
    for dd in DURATIONS:
        v = effect("surprise_only", dd)
        xs += [dd] * len(v); ys += list(v)
    rho_s, p_s = spearmanr(xs, ys) if len(set(xs)) > 1 else (float("nan"), 1.0)
    g1 = rho_s < 0 and p_s < 0.05

    e1 = effect("surprise_only", DURATIONS[0])
    g2 = bool(len(e1) and (e1 > 0).mean() >= 0.70)

    others = [c for c in CHANNELS if c not in ("surprise_only", "pipeline_fused")]
    g3 = bool(len(e1) and all(
        eff["surprise_only"][0] > eff[c][0] for c in others
        if not np.isnan(eff[c][0])))

    fused_wins = []
    for i in range(len(DURATIONS)):
        best = max(eff[c][i] for c in SINGLE if not np.isnan(eff[c][i]))
        fused_wins.append(eff["pipeline_fused"][i] > best + 1e-9)
    g4 = not any(fused_wins)

    rnd_bad = []
    for dd in DURATIONS:
        for mode in ("relocated", "null"):
            v = col(mode, dd, "random_score")
            if len(v) and not 0.40 <= float(np.median(v)) <= 0.60:
                rnd_bad.append(f"{mode}@D{dd}={np.median(v):.3f}")
    g5 = not rnd_bad

    acc = {
        "G1_surprise_effect_falls_with_duration": {"rho": round(float(rho_s), 3),
                                                   "p": float(p_s), "pass": bool(g1)},
        "G2_surprise_effect_positive_at_Dmin": {
            "frac": round(float((e1 > 0).mean()), 3) if len(e1) else None,
            "pass": bool(g2)},
        "G3_surprise_effect_beats_every_channel_at_Dmin": {
            "surprise": round(eff["surprise_only"][0], 3),
            "runner_up": max(((round(eff[c][0], 3), c) for c in others
                              if not np.isnan(eff[c][0])), default=None),
            "pass": bool(g3)},
        "G4_fusion_never_best": {"pass": bool(g4)},
        "G5_random_at_chance": {"pass": bool(g5), "violations": rnd_bad[:8]},
        "overall_pass": bool(g1 and g2 and g3 and g4 and g5),
    }
    print(f"\nsurprise relocation effect vs duration: rho={rho_s:+.3f} (p={p_s:.1e})")
    print("\nACCEPTANCE (v2, on the relocation effect):")
    for k, v in acc.items():
        if k != "overall_pass":
            print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}")
    if not g5:
        print("\n  G5 FAILED - harness broken. Ignore G1-G4.")
    elif acc["overall_pass"]:
        print("\n  Surprise is the only channel that responds to temporal")
        print("  displacement once frame identity is cancelled, and the response")
        print("  decays with dwell length. P1-1/P1-2 are the ends of an axis,")
        print("  not defects. Fusion never wins: route, do not fuse.")
    else:
        print("\n  Not established as specified. Report as measured.")

    Path(args.out).write_text(json.dumps(
        {"durations": DURATIONS, "n_proteins": len(per_protein),
         "median_auroc_by_duration": {k: [None if np.isnan(x) else round(x, 3)
                                          for x in v] for k, v in med.items()},
         "relocation_effect_by_duration": {
             k: [None if np.isnan(x) else round(x, 3) for x in v]
             for k, v in eff.items()},
         "acceptance": acc, "per_protein": per_protein}, indent=2))
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
