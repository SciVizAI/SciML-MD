#!/usr/bin/env python3
"""
Is transition surprise detecting IMPROBABLE transitions, or merely UNSEEN ones?

WHY THIS MATTERS
----------------
Phase 3 passed all four criteria: surprise 0.816 median across 25 proteins,
beating the best time-blind detector on 25/25, Wilcoxon p = 3e-08. Before that
becomes the headline, one confound has to be excluded.

compute_kinetic_signals treats a transition that touches a state the MSM
dropped from its active set as MAXIMALLY surprising - it assigns the largest
finite surprise observed (defect D-03; the previous behaviour was a silent
zero, which ranked such transitions as the LEAST anomalous). That is the right
default. But it means some frames receive a high score by IMPUTATION rather
than from -log P.

Temporal scrambling splices together frames that were never adjacent. If those
splices systematically produce transitions the count matrix never saw, then
junction frames get the cap by construction, and the 0.816 would partly measure
our own imputation rule rather than kinetic improbability. The run log is full
of "Skipping state set ... zero counts", so states ARE being dropped.

Three measurements settle it:

  1. cap rate at junction frames vs everywhere else. If they are equal, the cap
     is not tracking the label and there is no confound.
  2. AUROC of the CAP INDICATOR ALONE - a binary feature that is 1 when the
     transition touches a dropped state. If that alone reaches ~0.8, the
     headline claim is about disconnection, not about -log P.
  3. AUROC of surprise with every capped frame REMOVED from the evaluation.
     This is the conservative number: what surprise achieves using only
     genuinely estimated transition probabilities. If it stays near 0.8, the
     claim survives intact.

WHAT EACH OUTCOME MEANS
    (2) low and (3) high   -> the claim holds as stated. Report 0.816.
    (2) high and (3) high  -> both mechanisms carry signal; report the
                              uncapped AUROC as the conservative headline and
                              describe disconnection as a second channel.
    (3) collapses to ~0.5  -> the result is an artefact of the imputation rule.
                              The claim must be restated entirely.

USAGE
    python validation/phase3_confound_check.py --data_root ~/atlas
    python validation/phase3_confound_check.py --data_root ~/atlas --limit 8
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from validation.phase2_temporal_scramble import (  # noqa: E402
    scramble, junction_labels)
from validation.phase3_scale import (  # noqa: E402
    discover, load_traj, scale_hyperparams)

N_REPLICATES = 3


def surprise_with_cap_mask(traj, lag_tica, dim, n_clusters, lag_msm, seed=42):
    """Recompute the surprise channel and return which frames were CAPPED.

    Mirrors validation.phase2_temporal_scramble.score_all's front half exactly,
    then reproduces the disconnection test from
    scoring.anomaly_v2.compute_kinetic_signals so the mask corresponds to the
    same frames the production path caps.
    """
    import mdtraj as md
    from features.compute_md_features import features_to_matrix
    from run_all_proteins import run_tica, cluster_states, build_msm
    from scoring.anomaly_v2 import (compute_kinetic_signals,
                                    remap_dtraj_to_active_set)

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

    Y, _ = run_tica(X, lag=lag_tica, dim=dim)
    dtraj, _ = cluster_states(Y, n_clusters=n_clusters, seed=seed)
    msm, _, _ = build_msm(dtraj, lag=lag_msm)
    rarity, surprise = compute_kinetic_signals(msm, dtraj, lag_msm)

    dtraj_active, _ = remap_dtraj_to_active_set(dtraj, msm)
    valid = (dtraj_active >= 0) & (dtraj_active < msm.n_states)
    n = len(dtraj)
    capped = np.zeros(n, dtype=bool)
    if n > lag_msm:
        capped[:-lag_msm] = ~(valid[:-lag_msm] & valid[lag_msm:])
    return surprise, capped, rarity


def run_protein(sid, top, trj, max_frames, seed_base=900):
    import mdtraj as md

    traj = load_traj(top, trj, max_frames)
    hp = scale_hyperparams(len(traj))
    R = np.zeros((len(traj), len(traj)), dtype=np.float32)
    for i in range(len(traj)):
        R[i] = md.rmsd(traj, traj, i)

    rows = []
    for rep in range(N_REPLICATES):
        rng = np.random.default_rng(seed_base + rep)
        order, junc = scramble(traj, "near", rng, R)
        new = traj[order]
        y = junction_labels(len(new), junc, hp["lag_msm"]).astype(bool)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        sur, capped, _ = surprise_with_cap_mask(
            new, hp["lag_tica"], hp["dim"], hp["n_clusters"], hp["lag_msm"])

        finite = np.isfinite(sur)
        s = np.nan_to_num(sur, nan=np.nanmin(sur[finite]) if finite.any() else 0.0)

        auroc_full = roc_auc_score(y, s)
        auroc_cap_only = (roc_auc_score(y, capped.astype(float))
                          if 0 < capped.sum() < len(capped) else np.nan)

        keep = finite & ~capped
        if keep.sum() > 20 and 0 < y[keep].sum() < keep.sum():
            auroc_uncapped = roc_auc_score(y[keep], s[keep])
        else:
            auroc_uncapped = np.nan

        rows.append({
            "auroc_full": float(auroc_full),
            "auroc_cap_indicator_only": float(auroc_cap_only),
            "auroc_excluding_capped": float(auroc_uncapped),
            "cap_rate_at_junctions": float(capped[y].mean()),
            "cap_rate_elsewhere": float(capped[~y].mean()),
            "cap_rate_overall": float(capped.mean()),
            "n_junction_frames": int(y.sum()),
            "n_frames_kept": int(keep.sum()),
        })
    if not rows:
        return None
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    out["n_replicates"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--screen", default="validation/phase3_screen.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_frames", type=int, default=4000)
    ap.add_argument("--out", default="validation/phase3_confound.json")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser()
    want = set(args.only) if args.only else None
    if want is None:
        sp = Path(args.screen)
        if sp.exists():
            want = set(json.loads(sp.read_text()).get("usable_proteins") or [])

    systems = discover(root)
    if want:
        systems = [s for s in systems if s[0].split(":")[0] in want]
    seen, picked = set(), []
    for s in systems:
        base = s[0].split(":")[0]
        if base not in seen:
            seen.add(base)
            picked.append(s)
    if args.limit:
        picked = picked[: args.limit]
    if not picked:
        print(f"No systems found under {root}")
        return 1

    print(f"{len(picked)} protein(s), first replicate each, "
          f"{N_REPLICATES} scramble repeats\n")
    print(f"{'protein':10s} {'full':>6s} {'caponly':>8s} {'nocap':>6s} "
          f"{'cap@junc':>9s} {'cap@else':>9s}")
    print("-" * 56)

    per = {}
    for sid, top, trj, _ in picked:
        name = sid.split(":")[0]
        try:
            r = run_protein(sid, top, trj, args.max_frames)
        except Exception as e:                                # noqa: BLE001
            print(f"{name:10s} FAILED {type(e).__name__}: {e}"[:70])
            continue
        if r is None:
            continue
        per[name] = r
        print(f"{name:10s} {r['auroc_full']:6.3f} "
              f"{r['auroc_cap_indicator_only']:8.3f} "
              f"{r['auroc_excluding_capped']:6.3f} "
              f"{r['cap_rate_at_junctions']*100:8.1f}% "
              f"{r['cap_rate_elsewhere']*100:8.1f}%", flush=True)

    if len(per) < 3:
        print("\nToo few proteins to judge.")
        return 1

    full = np.array([v["auroc_full"] for v in per.values()])
    capo = np.array([v["auroc_cap_indicator_only"] for v in per.values()])
    nocap = np.array([v["auroc_excluding_capped"] for v in per.values()])
    cj = np.array([v["cap_rate_at_junctions"] for v in per.values()])
    ce = np.array([v["cap_rate_elsewhere"] for v in per.values()])

    print(f"\n=== CONFOUND CHECK ({len(per)} proteins) ===")
    print(f"  surprise, all frames          median {np.nanmedian(full):.3f}")
    print(f"  cap indicator alone           median {np.nanmedian(capo):.3f}")
    print(f"  surprise, capped frames out   median {np.nanmedian(nocap):.3f}")
    print(f"  cap rate at junctions         {np.nanmedian(cj)*100:.1f}%")
    print(f"  cap rate elsewhere            {np.nanmedian(ce)*100:.1f}%")

    enriched = np.nanmedian(cj) > 1.5 * max(np.nanmedian(ce), 1e-6)
    cap_carries = np.nanmedian(capo) >= 0.60
    survives = np.nanmedian(nocap) >= 0.70

    print("\nVERDICT:")
    if not enriched and not cap_carries:
        print("  No confound. Capping is not enriched at junctions and the cap")
        print("  indicator alone is at chance. The 0.816 is -log P, as claimed.")
    elif survives:
        print("  Capping IS enriched at junctions, but surprise still reaches")
        print(f"  {np.nanmedian(nocap):.3f} on genuinely estimated transitions alone.")
        print("  Report the uncapped figure as the conservative headline and")
        print("  describe disconnection as a second, honest channel.")
    else:
        print("  *** The result depends on the imputation rule. ***")
        print(f"  Excluding capped frames drops surprise to "
              f"{np.nanmedian(nocap):.3f}.")
        print("  The claim must be restated: the detector is largely flagging")
        print("  transitions the MSM never observed, not improbable ones.")

    Path(args.out).write_text(json.dumps(
        {"n_proteins": len(per),
         "median_auroc_full": float(np.nanmedian(full)),
         "median_auroc_cap_indicator_only": float(np.nanmedian(capo)),
         "median_auroc_excluding_capped": float(np.nanmedian(nocap)),
         "median_cap_rate_at_junctions": float(np.nanmedian(cj)),
         "median_cap_rate_elsewhere": float(np.nanmedian(ce)),
         "per_protein": per}, indent=2))
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
