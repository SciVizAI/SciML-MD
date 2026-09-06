#!/usr/bin/env python3
"""
Phase 7 — is the rarity channel CALIBRATED? Does pi predict held-out populations?

WHY THIS AND NOT ANOTHER DETECTION TEST
---------------------------------------
Three designs have now shown that transition surprise does not beat a four-line
detector (Phase 3 detection, Phase 5 identity-cancelled detection, Phase 6
timing). OI-30 says: stop running variations on that theme.

Rarity is the one channel with a live positive signal - in Phase 6 it was the
only detector trending earlier than random (31.0 vs 25.0, p = 0.132, n = 9).
But rarity's real claim is not a detection claim at all. `1 - pi(s)` asserts
that state s is thermodynamically improbable. That is a CALIBRATION claim, and
calibration is testable directly, without constructing a single anomaly.

THE TEST
    ATLAS ships three independent replicates per protein. So:

      1. Discretise the POOLED ensemble of all three replicates into microstates,
         so every replicate is scored on the same state space.
      2. Fit the MSM on replicates {0,1}. Take pi_model.
      3. Compute pi_true as the empirical state frequency in the HELD-OUT
         replicate {2}.
      4. Ask whether pi_model predicts pi_true.

    Nothing is spliced, relocated or seeded. The ground truth is the protein's
    own behaviour in a run the model never saw. There is no geometric detector
    to be circular against, because "which states are rarely occupied" is not a
    quantity a difference detector computes at all.

THE BASELINE THAT MATTERS
    `empirical_train` - the raw fraction of training frames in each state, with
    no MSM, no transition matrix, no reversibility, no eigenvector. If pi does
    not beat counting, then everything the MSM adds to the rarity channel is
    decoration. This is the rarity-channel analogue of `abs_diff_oneliner`, and
    it is the comparison the whole phase exists to make.

    Also reported: `uniform` (every state equally likely) as the floor.

PRE-REGISTERED CRITERIA  (fixed before the first run)
    (v2: all statistics are computed on SHARED SUPPORT - states visited in both
     the training and held-out replicates - and proteins below 50% held-out
     coverage are excluded. See the CORRECTION block for why.)

    R1  Spearman(pi_model, pi_true) > 0 with p < 0.05, pooled over proteins
    R2  pi_model beats `uniform` at identifying the bottom-decile-rare states
        (AUROC of 1 - pi_model against "in the bottom decile of pi_true")
    R3  pi_model beats `empirical_train` on R1's correlation, on >= 60% of
        proteins  <- THE ONE THAT DECIDES WHETHER THE MSM EARNS ITS PLACE
    R4  Wilcoxon p < 0.05, pi_model vs empirical_train correlation
    R5  a SHUFFLED pi (same values, permuted across states) scores at chance:
        |Spearman| <= 0.15 and AUROC in 0.40-0.60   [GATES R1-R6]
    R6  MISSED MASS: the share of held-out frames landing in states the
        predictor ranks in its bottom decile. Reported for pi and for counting;
        lower is better. This is the failure the shared-support restriction
        would otherwise hide - states the model calls near-impossible that the
        protein actually occupies.

HOW TO READ IT
    R5 fails             -> harness broken, ignore the rest.
    R1/R2 pass, R3 fails -> pi is calibrated but adds nothing over counting.
                            The rarity channel is real and the MSM is not
                            needed to compute it. Ship the count.
    R3 passes            -> the MSM's stationary distribution carries
                            information raw frequencies do not. That is a
                            genuine, defensible claim and the first one in this
                            campaign that survives its own baseline.
    R1 fails             -> pi does not predict held-out populations at all.
                            The rarity channel is not calibrated and 1 - pi
                            should not be reported to users as improbability.

USAGE
    python validation/phase7_rarity_calibration.py --data_root ~/atlas
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from validation.phase3_scale import (  # noqa: E402
    discover, load_traj, scale_hyperparams)

# --------------------------------------------------------------------------
# CORRECTION v2, 2026-08-29, after v1 proved uninterpretable.
#
# v1 defined "rare" as the bottom decile of held-out population. But a median
# 26 of 76 training states are NEVER visited by the held-out replicate, so the
# 10th percentile IS zero and the label silently became "every unvisited state"
# - 12 to 57 of 82 depending on protein. Spearman then ran on one enormous tie
# block, and raw COUNTING (no MSM at all) scored rho = -0.508 on 24 of 24
# proteins. A no-model baseline cannot genuinely anti-predict held-out
# frequency that consistently: the run measured arithmetic, not biology.
#
# Three changes, all fixed before this run:
#   1. SHARED SUPPORT. The correlation runs only on states visited in BOTH the
#      training and the held-out replicates. No tie block.
#   2. RANK-BASED RARE LABEL. "Rare" is the bottom RARE_FRACTION by rank within
#      shared support, so the split is always non-degenerate.
#   3. COVERAGE GATE. A protein is excluded when less than MIN_COVERAGE of its
#      held-out frames land in states the training replicates ever visited.
#      Below that there is no shared support worth correlating on.
#
# Restricting to shared support answers a WEAKER question than full
# calibration: it cannot see states the model calls impossible that the protein
# actually visits. That failure mode is therefore measured separately and
# reported as R6 (missed mass), so the restriction hides nothing.
# --------------------------------------------------------------------------
RARE_FRACTION = 0.25       # bottom quarter BY RANK within shared support
MIN_COVERAGE = 0.50        # held-out frames that must fall in training states
MIN_SHARED = 12            # shared-support states needed to correlate at all
MISSED_DECILE = 0.10       # R6 looks at the bottom decile of the predictor


def featurise(traj):
    """Same feature set the production path uses, so the state space is the
    pipeline's own - not one invented for this test."""
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
    return X


def run_protein(trajs, seed=42):
    """trajs: list of >=3 replicate trajectories of the same protein."""
    import mdtraj as md
    from run_all_proteins import run_tica, cluster_states, build_msm
    from scoring.anomaly_v2 import remap_dtraj_to_active_set

    if len(trajs) < 3:
        return None, f"needs 3 replicates, got {len(trajs)}"

    # --- pooled discretisation: one state space for every replicate ---------
    lengths = [len(t) for t in trajs]
    pooled = md.join(trajs)
    hp = scale_hyperparams(len(pooled))
    X = featurise(pooled)
    Y, _ = run_tica(X, lag=hp["lag_tica"], dim=hp["dim"])
    dtraj_all, _ = cluster_states(Y, n_clusters=hp["n_clusters"], seed=seed)

    bounds = np.cumsum([0] + lengths)
    parts = [dtraj_all[bounds[i]:bounds[i + 1]] for i in range(len(trajs))]
    train = np.concatenate(parts[:2])
    test = parts[2]
    n_states = int(dtraj_all.max()) + 1
    if n_states < MIN_SHARED:
        return None, f"only {n_states} states"

    # --- pi_model: MSM stationary distribution from the training replicates -
    msm, _, _ = build_msm(train, lag=hp["lag_msm"])
    symbols = np.asarray(msm.count_model.state_symbols)
    pi_model = np.zeros(n_states, dtype=float)
    pi_model[symbols] = msm.stationary_distribution

    # --- baselines ---------------------------------------------------------
    emp_train = np.bincount(train, minlength=n_states).astype(float)
    emp_train /= emp_train.sum()
    uniform = np.full(n_states, 1.0 / n_states)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(pi_model)

    # --- ground truth: held-out replicate frequency -------------------------
    pi_true = np.bincount(test, minlength=n_states).astype(float)
    pi_true /= max(pi_true.sum(), 1.0)

    # --- support sets -------------------------------------------------------
    seen = emp_train > 0                       # visited during training
    coverage = float(pi_true[seen].sum())      # held-out mass the model can reach
    shared = seen & (pi_true > 0)              # visited in BOTH -> correlatable

    out = {"n_states": n_states, "n_states_seen_in_train": int(seen.sum()),
           "n_shared_support": int(shared.sum()),
           "heldout_coverage": round(coverage, 4),
           "lag": hp["lag_msm"], "frames": lengths}

    if coverage < MIN_COVERAGE:
        return None, (f"held-out coverage {coverage:.3f} < {MIN_COVERAGE} "
                      f"- no shared support worth correlating")
    if shared.sum() < MIN_SHARED:
        return None, f"only {int(shared.sum())} shared-support states"

    # --- R7 CONTROL: split that ignores replicate identity -----------------
    # The decisive control. Same clustering, same counting, but the train/test
    # split carries no information about WHICH replicate a frame came from. If
    # the replicate split anti-correlates while this one does not, the negative
    # correlation is a property of the protein's replicates, not of the harness.
    rng_c = np.random.default_rng(seed)
    allidx = np.arange(len(dtraj_all))
    n_tr = len(train)
    ctrl = []
    for _ in range(20):
        perm = rng_c.permutation(allidx)
        a = np.bincount(dtraj_all[perm[:n_tr]], minlength=n_states).astype(float)
        b = np.bincount(dtraj_all[perm[n_tr:]], minlength=n_states).astype(float)
        m = (a > 0) & (b > 0)
        if m.sum() >= MIN_SHARED:
            rr = spearmanr(a[m], b[m]).statistic
            if np.isfinite(rr):
                ctrl.append(float(rr))
    out["random_split_control_rho"] = (round(float(np.median(ctrl)), 4)
                                       if ctrl else None)

    # rank-based rare label inside shared support: always non-degenerate
    pt = pi_true[shared]
    order = np.argsort(pt)
    k = max(3, int(round(RARE_FRACTION * len(pt))))
    if k >= len(pt) - 3:
        return None, f"cannot split {len(pt)} shared states into rare/common"
    y_rare = np.zeros(len(pt), dtype=bool)
    y_rare[order[:k]] = True
    out["n_rare_states"] = int(y_rare.sum())

    for name, p in (("pi_model", pi_model), ("empirical_train", emp_train),
                    ("uniform", uniform), ("shuffled_pi", shuffled)):
        rho, pv = spearmanr(p[shared], pt)
        rho = 0.0 if not np.isfinite(rho) else float(rho)
        try:
            au = float(roc_auc_score(y_rare, -p[shared]))   # low p => rare
        except ValueError:
            au = float("nan")
        # R6 MISSED MASS: held-out probability sitting in states this predictor
        # ranks in its bottom decile, over the states it could reach at all.
        ps = p[seen]
        thr = np.quantile(ps, MISSED_DECILE)
        missed = float(pi_true[seen][ps <= thr].sum() / max(pi_true[seen].sum(), 1e-12))
        out[name] = {"spearman": round(rho, 4),
                     "spearman_p": (None if not np.isfinite(pv) else float(pv)),
                     "auroc_rare": round(au, 4),
                     "missed_mass": round(missed, 4)}
    return out, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--screen", default="validation/phase3_screen.json")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--max_frames", type=int, default=4000)
    ap.add_argument("--out", default="validation/phase7_results.json")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser()
    want = set(args.only) if args.only else None
    if want is None and Path(args.screen).exists():
        want = set(json.loads(Path(args.screen).read_text())
                   .get("usable_proteins") or [])

    by_prot = {}
    for sid, top, trj, _ in discover(root):
        base = sid.split(":")[0]
        if want and base not in want:
            continue
        by_prot.setdefault(base, []).append((top, trj))
    names = sorted(by_prot)[: args.limit]
    if not names:
        print(f"No systems under {root}")
        return 1

    print(f"{len(names)} protein(s); MSM fit on replicates 1-2, "
          f"held out replicate 3\n")
    print(f"{'protein':10s} {'shared':>7s} {'rare':>5s} {'cover':>6s} "
          f"{'pi rho':>7s} {'count rho':>10s} {'pi AUROC':>9s} "
          f"{'pi miss':>8s} {'cnt miss':>9s} {'ctrl rho':>8s}")
    print("-" * 92)

    per, skipped = {}, {}
    for name in names:
        files = sorted(by_prot[name])[:3]
        try:
            trajs = [load_traj(t, x, args.max_frames) for t, x in files]
            r, why = run_protein(trajs)
        except Exception as e:                                 # noqa: BLE001
            skipped[name] = f"{type(e).__name__}: {e}"[:90]
            print(f"{name:10s} FAILED {skipped[name]}"[:70])
            continue
        if r is None:
            skipped[name] = why
            print(f"{name:10s} skipped: {why}")
            continue
        per[name] = r
        print(f"{name:10s} {r['n_shared_support']:7d} {r['n_rare_states']:5d} "
              f"{r['heldout_coverage']:6.3f} {r['pi_model']['spearman']:7.3f} "
              f"{r['empirical_train']['spearman']:10.3f} "
              f"{r['pi_model']['auroc_rare']:9.3f} "
              f"{r['pi_model']['missed_mass']:8.3f} "
              f"{r['empirical_train']['missed_mass']:9.3f} "
              f"{(r.get('random_split_control_rho') or float('nan')):8.3f}",
              flush=True)

    if len(per) < 5:
        print(f"\nOnly {len(per)} protein(s) usable - inconclusive.")
        Path(args.out).write_text(json.dumps(
            {"per_protein": per, "skipped": skipped}, indent=2))
        return 1

    g = lambda k, f: np.array([r[k][f] for r in per.values()], dtype=float)
    pi_rho, emp_rho = g("pi_model", "spearman"), g("empirical_train", "spearman")
    pi_au = g("pi_model", "auroc_rare")
    uni_au = g("uniform", "auroc_rare")
    sh_rho, sh_au = g("shuffled_pi", "spearman"), g("shuffled_pi", "auroc_rare")

    print(f"\n=== PHASE 7 ({len(per)} proteins) ===")
    print(f"{'predictor':18s} {'median rho':>11s} {'AUROC(rare)':>12s} "
          f"{'missed mass':>12s}")
    print("-" * 56)
    for k, lbl in (("pi_model", "pi (MSM)"), ("empirical_train", "count (no MSM)"),
                   ("uniform", "uniform"), ("shuffled_pi", "shuffled pi [null]")):
        print(f"{lbl:18s} {np.median(g(k,'spearman')):11.3f} "
              f"{np.median(g(k,'auroc_rare')):12.3f} "
              f"{np.median(g(k,'missed_mass')):12.3f}")
    ctrl_v = np.array([r["random_split_control_rho"] for r in per.values()
                       if r.get("random_split_control_rho") is not None])
    if len(ctrl_v):
        print(f"\nR7 CONTROL - split ignoring replicate identity: "
              f"median rho {np.median(ctrl_v):+.3f}")
        print(f"     replicate split: median rho {np.median(pi_rho):+.3f} (pi), "
              f"{np.median(emp_rho):+.3f} (count)")
        print("     A positive control with a negative replicate split means the")
        print("     harness is sound and the replicates genuinely anti-occupy.")

    excluded = [k for k, v in skipped.items() if "coverage" in str(v)]
    print(f"\ncoverage gate ({MIN_COVERAGE}): excluded {len(excluded)} protein(s)"
          + (f" - {', '.join(excluded)}" if excluded else ""))

    try:
        _, p_r1 = wilcoxon(pi_rho, alternative="greater")
    except ValueError:
        p_r1 = float("nan")
    r1 = bool(np.median(pi_rho) > 0 and p_r1 < 0.05)
    r2 = bool(np.median(pi_au) > np.median(uni_au))
    r3 = bool((pi_rho > emp_rho).mean() >= 0.60)
    try:
        p_r4 = float(wilcoxon(pi_rho, emp_rho, alternative="greater").pvalue)
    except ValueError:
        p_r4 = float("nan")
    r4 = bool(p_r4 < 0.05)
    r5 = bool(abs(float(np.median(sh_rho))) <= 0.15
              and 0.40 <= float(np.median(sh_au)) <= 0.60)

    acc = {
        "R1_pi_predicts_heldout": {"median_rho": round(float(np.median(pi_rho)), 3),
                                   "p": p_r1, "pass": r1},
        "R2_pi_beats_uniform_on_rare": {
            "pi": round(float(np.median(pi_au)), 3),
            "uniform": round(float(np.median(uni_au)), 3), "pass": r2},
        "R3_pi_beats_counting_on_60pct": {
            "frac": round(float((pi_rho > emp_rho).mean()), 3), "pass": r3},
        "R4_wilcoxon_pi_vs_counting": {"p": p_r4, "pass": r4},
        "R5_shuffled_at_chance": {"rho": round(float(np.median(sh_rho)), 3),
                                  "auroc": round(float(np.median(sh_au)), 3),
                                  "pass": r5},
        "R7_random_split_control": {
            "median_rho": (round(float(np.median(ctrl_v)), 3)
                           if len(ctrl_v) else None),
            "pass": bool(len(ctrl_v) and float(np.median(ctrl_v)) > 0.2),
            "note": ("Same clustering and counting, split ignoring replicate "
                     "identity. Must be clearly positive; if it is, a negative "
                     "replicate split is a property of the data, not the test.")},
        "R6_missed_mass": {
            "pi": round(float(np.median(g("pi_model", "missed_mass"))), 4),
            "empirical_train": round(float(np.median(
                g("empirical_train", "missed_mass"))), 4),
            "note": ("Held-out probability landing in the predictor's "
                     "bottom decile. Reported, not gated - it measures the "
                     "failure the shared-support restriction would hide.")},
        "overall_pass": bool(r1 and r2 and r3 and r4 and r5),
    }
    print("\nACCEPTANCE (pre-registered):")
    for k, v in acc.items():
        if k == "overall_pass":
            continue
        if isinstance(v, dict) and "pass" in v:
            print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}")
        elif isinstance(v, dict):
            print(f"  ----  {k}: " + ", ".join(
                f"{kk}={vv}" for kk, vv in v.items() if kk != "note"))

    if not r5:
        print("\n  R5 FAILED - shuffled pi is not at chance. Harness broken.")
    elif not r1:
        print("\n  pi does NOT predict held-out populations. The rarity channel")
        print("  is not calibrated; 1 - pi must not be shown to users as")
        print("  improbability.")
    elif not r3:
        print("\n  pi IS calibrated, but does not beat simply counting training")
        print("  frames. The rarity signal is real; the MSM is not needed to")
        print("  compute it. Ship the count and drop the machinery.")
    elif acc["overall_pass"]:
        print("\n  pi carries information raw state frequencies do not. This is")
        print("  the first claim in the campaign to survive its own strongest")
        print("  baseline - report it as the method's contribution.")
    else:
        print("\n  Mixed. Report exactly as measured.")

    Path(args.out).write_text(json.dumps(
        {"n_proteins": len(per), "rare_fraction": RARE_FRACTION,
         "min_coverage": MIN_COVERAGE,
         "acceptance": acc, "per_protein": per, "skipped": skipped}, indent=2))
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
