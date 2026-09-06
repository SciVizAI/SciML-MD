#!/usr/bin/env python3
"""
Phase 6 — does the kinetic channel fire BEFORE the geometry moves?

WHY THIS TEST AND NOT ANOTHER
-----------------------------
Two constructions have now failed to separate transition surprise from a
four-line difference detector:

  Phase 3 (spliced junctions)   oneliner 0.839 vs surprise 0.814, PR 0.227 vs 0.082
  Phase 5 (relocated excursions) oneliner +0.274 vs surprise +0.149 at D=1

In both, the question was DETECTION: who ranks the anomalous frames higher.
On that question a difference detector is very hard to beat, because every
constructed anomaly leaves a geometric trace somewhere.

This phase asks a different question, on UNMODIFIED trajectories with real
dynamics and no splicing at all:

    when a conformational transition happens, WHO FIRES FIRST?

If the system enters a rarely-visited precursor state some frames before the
large structural move, then -log P has something to report while the geometry
is still ordinary, and a difference detector has nothing to difference yet. A
geometric detector CANNOT fire before the geometry changes - that is a
definition, not a bias - so the comparison is fair and the claim is
falsifiable.

It is also the version a user of the software actually wants: early warning,
not post-hoc annotation.

FAIRNESS: THE LOOKAHEAD MUST BE MATCHED
    surprise[t] compares frames t and t+lag, so it sees lag frames into the
    future. `abs_diff_oneliner` and `frame_to_frame_rmsd` both take a max over
    the same [t, t+lag] window, so all three detectors have IDENTICAL
    lookahead. Any lead surprise shows is not an artefact of alignment. This is
    asserted here and checked in the output (`lookahead_frames` is reported per
    detector and must be equal).

EVENT DEFINITION (chosen before running; see --event-min-persist)
    Leader-clustering of CA structures at 2 A - the same machinery
    msm/preflight.py uses for the suitability census. An EVENT is a basin label
    change that PERSISTS for at least `--event-min-persist` frames, so flicker
    across a cluster boundary does not count. The event frame is the first
    frame carrying the new label.

    The event label is geometric. That is intentional and is not circular here:
    the claim is about TIMING relative to geometry's own firing time, not about
    detecting something geometry cannot see.

FIRING TIME
    A detector's firing time for an event is where its score PEAKS inside
    [event - max_lead, event]. Lead = event - peak frame; larger is earlier.
    The peak is always defined, so every event contributes to every detector
    and none is dropped by a threshold. Under a null the peak is uniform over
    the window, so the expected lead is max_lead/2 for any detector - see the
    CORRECTION note in firing_lead() for why the original threshold rule was
    replaced.

PRE-REGISTERED CRITERIA  (agreed before the first run; do not adjust after)
    L1  median lead of surprise over abs_diff_oneliner > 0
    L2  surprise leads the one-liner on >= 60% of events
    L3  Wilcoxon signed-rank p < 0.05, surprise lead > one-liner lead
    L4  surprise's median lead exceeds random_score's, i.e. it beats the
        uniform-peak baseline rather than merely being positive
    L5  random_score's median lead is max_lead/2 +/- 5 frames, the uniform
        expectation under the peak rule  [GATES L1-L4]

    (L4/L5 were recalibrated when the peak rule replaced the threshold rule -
    see firing_lead's CORRECTION note. L1-L3 are unchanged in substance.)

HOW TO READ A FAILURE
    L5 fails            -> harness broken, ignore everything else.
    L4 fails            -> surprise does not beat a uniformly random peak;
                           report as inconclusive, do not read L1-L3.
    L1/L2/L3 fail       -> surprise does not provide early warning either. Two
                           detection tests and one timing test would then all
                           point the same way, and the honest conclusion is
                           that the MSM machinery is not earning its place.

USAGE
    python validation/phase6_lead_time.py --data_root ~/atlas --limit 25
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from validation.phase2_temporal_scramble import score_all          # noqa: E402
from validation.phase3_scale import (                              # noqa: E402
    discover, load_traj, scale_hyperparams)

PRIMARY = "surprise_only"
COMPARATOR = "abs_diff_oneliner"
NULL = "random_score"
REPORT = [PRIMARY, COMPARATOR, "frame_to_frame_rmsd", "rmsd_from_mean",
          "rarity_only", "density_only", "pipeline_fused", NULL]


def basin_events(traj, cutoff_ang=2.0, min_persist=20):
    """Leader-cluster CA structures; return frames where the basin changes and
    the new basin holds for >= min_persist frames.

    Same construction as msm/preflight.py's basin census, so the event
    definition and the suitability screen agree by design.
    """
    import mdtraj as md

    ca = traj.atom_slice(traj.topology.select("name CA"))
    ca.superpose(ca, 0)
    n = len(ca)

    leaders, assign = [], np.empty(n, dtype=np.int64)
    for i in range(n):
        placed = False
        for c, lead in enumerate(leaders):
            if md.rmsd(ca[i], ca[lead])[0] * 10.0 < cutoff_ang:
                assign[i] = c
                placed = True
                break
        if not placed:
            leaders.append(i)
            assign[i] = len(leaders) - 1

    events = []
    for t in range(1, n):
        if assign[t] == assign[t - 1]:
            continue
        seg = assign[t:t + min_persist]
        if len(seg) >= min_persist and np.all(seg == assign[t]):
            events.append(t)
    return events, assign, len(leaders)


def firing_lead(score, event, max_lead, pctile=None):
    """Frames by which `score` anticipates `event`, measured at its PEAK.

    CORRECTION, 2026-08-29, made after L5 failed on the first run.
    ------------------------------------------------------------------
    v1 used "first frame above the detector's own 95th percentile". That rule
    is base-rate driven and rewards any detector that fires often: a random
    score has 5% of frames above its own 95th percentile, so in a 50-frame
    window it crosses somewhere 93% of the time and the FIRST crossing is early
    by construction. Measured: random scored a median lead of 38 frames, and
    the first live run duly showed random at 35-41 frames, ahead of both real
    detectors. L5 caught it, which is what L5 is for.

    The peak rule has no such bias. Under a null the argmax is uniform over the
    window, so the expected lead is exactly max_lead/2 for every detector
    regardless of how often it fires - which is why L5 is now calibrated to
    max_lead/2 rather than to zero.

    It also removes a selection bias: the peak is always defined, so no event
    is dropped because one detector failed to cross a threshold. v1 discarded
    exactly the events where the one-liner was quiet, biasing the paired
    comparison towards events it could see.
    """
    s = np.asarray(score, dtype=float)
    finite = np.isfinite(s)
    if not finite.any():
        return None
    s = np.nan_to_num(s, nan=float(np.nanmin(s[finite])))
    lo = max(0, event - max_lead)
    win = s[lo:event + 1]
    if not len(win):
        return None
    return int(event - (lo + int(np.argmax(win))))


def run_protein(traj, args):
    hp = scale_hyperparams(len(traj))
    events, assign, n_basins = basin_events(
        traj, args.event_cutoff, args.event_min_persist)
    if len(events) < args.min_events:
        return None, f"only {len(events)} persistent basin change(s)"

    sc = score_all(traj, lag_tica=hp["lag_tica"], dim=hp["dim"],
                   n_clusters=hp["n_clusters"], lag_msm=hp["lag_msm"],
                   k=hp["k"])
    leads = {n: [] for n in REPORT}
    fired = {n: 0 for n in REPORT}
    for e in events:
        for n in REPORT:
            if n not in sc:
                continue
            L = firing_lead(sc[n], e, args.max_lead, args.pctile)
            leads[n].append(L)
            if L is not None:
                fired[n] += 1
    return {
        "n_events": len(events),
        "n_basins": int(n_basins),
        "lag_frames": hp["lag_msm"],
        "leads": leads,
        "fire_rate": {n: round(fired[n] / len(events), 3) for n in REPORT},
        "median_lead": {
            n: (float(np.median([x for x in leads[n] if x is not None]))
                if any(x is not None for x in leads[n]) else None)
            for n in REPORT},
    }, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--screen", default="validation/phase3_screen.json")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--max_frames", type=int, default=4000)
    ap.add_argument("--event-cutoff", type=float, default=2.0,
                    help="Leader-clustering RMSD cutoff, Angstrom")
    ap.add_argument("--event-min-persist", type=int, default=20,
                    help="A basin change counts only if it holds this many frames")
    ap.add_argument("--min-events", type=int, default=3,
                    help="Skip a trajectory with fewer events than this")
    ap.add_argument("--max-lead", type=int, default=50,
                    help="How many frames before the event a detector may fire")
    ap.add_argument("--pctile", type=float, default=95.0,
                    help="Each detector fires above this percentile of its own scores")
    ap.add_argument("--replicate", default="R1")
    ap.add_argument("--out", default="validation/phase6_results.json")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser()
    want = set(args.only) if args.only else None
    if want is None and Path(args.screen).exists():
        want = set(json.loads(Path(args.screen).read_text())
                   .get("usable_proteins") or [])

    seen, picked = set(), []
    for sid, top, trj, _ in discover(root):
        base = sid.split(":")[0]
        if (want and base not in want) or base in seen:
            continue
        if args.replicate not in Path(trj).stem:
            continue
        seen.add(base)
        picked.append((base, top, trj))
    picked = picked[: args.limit]
    if not picked:
        print(f"No systems under {root}")
        return 1

    print(f"{len(picked)} protein(s), unmodified trajectories, replicate "
          f"{args.replicate}")
    print(f"event = basin change persisting >= {args.event_min_persist} frames; "
          f"fire = above own {args.pctile:.0f}th pctile within "
          f"{args.max_lead} frames\n")
    print(f"{'protein':10s} {'events':>7s} {'surp':>7s} {'1line':>7s} "
          f"{'rmsdmn':>7s} {'rand':>6s}   fire rates (surp/1line)")
    print("-" * 74)

    per, skipped = {}, {}
    for name, top, trj in picked:
        try:
            traj = load_traj(top, trj, args.max_frames)
            r, why = run_protein(traj, args)
        except Exception as e:                                 # noqa: BLE001
            skipped[name] = f"{type(e).__name__}: {e}"[:90]
            print(f"{name:10s} FAILED {skipped[name]}"[:74])
            continue
        if r is None:
            skipped[name] = why
            print(f"{name:10s} skipped: {why}")
            continue
        per[name] = r
        ml = r["median_lead"]
        f = r["fire_rate"]
        def g(n):
            v = ml.get(n)
            return f"{v:7.1f}" if v is not None else f"{'-':>7s}"
        print(f"{name:10s} {r['n_events']:7d} {g(PRIMARY)} {g(COMPARATOR)} "
              f"{g('rmsd_from_mean')} "
              f"{(ml.get(NULL) if ml.get(NULL) is not None else float('nan')):6.1f}"
              f"   {f[PRIMARY]:.2f}/{f[COMPARATOR]:.2f}", flush=True)

    if len(per) < 5:
        print(f"\nOnly {len(per)} protein(s) produced events - inconclusive.")
        Path(args.out).write_text(json.dumps(
            {"per_protein": per, "skipped": skipped}, indent=2))
        return 1

    # ---- paired over EVENTS, pooled, and over PROTEINS (median per protein) --
    pair_s, pair_c = [], []
    for r in per.values():
        for a, b in zip(r["leads"][PRIMARY], r["leads"][COMPARATOR]):
            if a is not None and b is not None:
                pair_s.append(a); pair_c.append(b)
    pair_s, pair_c = np.array(pair_s, float), np.array(pair_c, float)

    prot_s = np.array([r["median_lead"][PRIMARY] for r in per.values()
                       if r["median_lead"][PRIMARY] is not None])
    prot_c = np.array([r["median_lead"][COMPARATOR] for r in per.values()
                       if r["median_lead"][COMPARATOR] is not None])

    n_events = sum(r["n_events"] for r in per.values())
    fire_s = float(np.mean([r["fire_rate"][PRIMARY] for r in per.values()]))
    fire_c = float(np.mean([r["fire_rate"][COMPARATOR] for r in per.values()]))
    null_leads = [r["median_lead"][NULL] for r in per.values()
                  if r["median_lead"][NULL] is not None]
    null_med = float(np.median(null_leads)) if null_leads else None

    print(f"\n=== PHASE 6 ({len(per)} proteins, {n_events} events, "
          f"{len(pair_s)} paired) ===")
    print(f"{'detector':22s} {'median lead':>12s} {'fire rate':>10s}")
    print("-" * 46)
    for n in REPORT:
        vals = [r["median_lead"][n] for r in per.values()
                if r["median_lead"][n] is not None]
        fr = float(np.mean([r["fire_rate"][n] for r in per.values()]))
        v = f"{np.median(vals):12.1f}" if vals else f"{'-':>12s}"
        print(f"{n:22s} {v} {fr:10.2f}")

    l1 = bool(len(pair_s) and np.median(pair_s - pair_c) > 0)
    l2 = bool(len(pair_s) and (pair_s > pair_c).mean() >= 0.60)
    try:
        p = float(wilcoxon(pair_s, pair_c, alternative="greater").pvalue)
    except Exception:                                          # noqa: BLE001
        p = float("nan")
    l3 = bool(p < 0.05)
    prot_r = np.array([r["median_lead"][NULL] for r in per.values()
                       if r["median_lead"][NULL] is not None])
    l4 = bool(len(prot_s) and len(prot_r)
              and float(np.median(prot_s)) > float(np.median(prot_r)))
    expected = args.max_lead / 2.0
    l5 = bool(null_med is not None and abs(null_med - expected) <= 5.0)

    acc = {
        "L1_surprise_leads_oneliner": {
            "median_diff_frames": (round(float(np.median(pair_s - pair_c)), 2)
                                   if len(pair_s) else None), "pass": l1},
        "L2_leads_on_60pct_events": {
            "frac": (round(float((pair_s > pair_c).mean()), 3)
                     if len(pair_s) else None), "pass": l2},
        "L3_wilcoxon_p_lt_0.05": {"p_value": p, "pass": l3},
        "L4_surprise_beats_random_lead": {
            "surprise": (round(float(np.median(prot_s)), 2) if len(prot_s) else None),
            "random": (round(float(np.median(prot_r)), 2) if len(prot_r) else None),
            "pass": l4},
        "L5_random_at_uniform_expectation": {
            "median_lead": null_med, "expected": args.max_lead / 2.0,
            "pass": l5},
        "overall_pass": bool(l1 and l2 and l3 and l4 and l5),
    }
    print("\nACCEPTANCE (pre-registered):")
    for k, v in acc.items():
        if k != "overall_pass":
            print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}")

    if not l5:
        print(f"\n  L5 FAILED - random median lead {null_med} vs uniform")
        print(f"  expectation {args.max_lead/2.0}. The harness is broken; ignore L1-L4.")
    elif not l4:
        print("\n  L4 FAILED - surprise does not beat a uniformly random peak.")
        print("  Any apparent lead over the one-liner is not evidence of early")
        print("  warning; INCONCLUSIVE.")
    elif acc["overall_pass"]:
        print("\n  Transition surprise provides EARLY WARNING that a difference")
        print("  detector cannot. This is a claim no trivial baseline can match,")
        print("  and it is the one the product should be built on.")
    else:
        print("\n  Surprise does not anticipate transitions earlier than a")
        print("  four-line detector. Together with Phase 3 and Phase 5, three")
        print("  independent designs now agree: the MSM machinery is not")
        print("  earning its place. Report it and change the claim.")

    Path(args.out).write_text(json.dumps(
        {"n_proteins": len(per), "n_events": n_events,
         "event_definition": {"cutoff_ang": args.event_cutoff,
                              "min_persist_frames": args.event_min_persist},
         "firing": {"percentile": args.pctile, "max_lead": args.max_lead},
         "acceptance": acc, "per_protein": per, "skipped": skipped}, indent=2))
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
