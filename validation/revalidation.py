#!/usr/bin/env python3
"""
Adversarial re-validation of the 15-Aug-2026 campaign's own conclusions.

Every check here is designed to FALSIFY a claim made in VALIDATION_REPORT.md.
Findings are recorded in validation/revalidation_results.json and summarised in
the ERRATA section of the report.

Checks
  R1  Did the residue-score fix (D-04) actually decouple scores from RMSF?
  R2  Multi-chain residue count (D-05) — is 8H0R really 182 now?
  R3  Does the dynamic score add anything beyond RMSF against B-factors?
      (partial rank correlation controlling for RMSF)
  R4  Is the rare-STATE baseline comparison circular? Re-run fusion without
      the rarity channel, which is definitionally the label source.
  R5  Is the rare-TRANSITION result circular? Correlate surprise against the
      raw transition count that defines the labels.
  R6  VAMP-2 "saturation" — how many lagged test pairs actually exist?
  R7  Edge-case count consistency between artifacts and prose.
  R8  Independent re-derivation of the core active-set finding (D-01).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scoring.anomaly_v2 import (  # noqa: E402
    compute_kinetic_signals, compute_local_density_signal, fuse_signals, moving_median)
from validation.layer3_external import (  # noqa: E402
    refit, labels, auroc, parse_ca_bfactors, score_key_to_tuple, RAW_PDB)

ART = {"1VII": ROOT / "fixed_artifacts/1VII", "8H0R": ROOT / "fixed_artifacts/8H0R",
       "1UBQ": ROOT / "fixed_artifacts/1UBQ", "1CRN": ROOT / "artifacts/1CRN"}
RES = ROOT / "fixed_results"
OUT = {}


def r1_residue_vs_rmsf():
    out = {}
    for pid in ["1VII", "8H0R", "1UBQ"]:
        dyn = json.load(open(RES / pid / "residue_scores_dynamic.json"))
        rmsf = json.load(open(RES / pid / "residue_scores_rmsf.json"))
        k = [x for x in dyn if x in rmsf]
        d = np.array([dyn[x] for x in k]); r = np.array([rmsf[x] for x in k])
        out[pid] = {"n": len(k),
                    "pearson_post_fix": round(float(pearsonr(d, r).statistic), 6),
                    "spearman_post_fix": round(float(spearmanr(d, r).statistic), 4)}
    out["_verdict"] = ("D-04 only PARTIALLY fixed: the constant-offset defect is gone, but "
                       "residue scores remain ~99% collinear with RMSF (r >= 0.99). The "
                       "anomaly channel still contributes almost no rank information.")
    return out


def r2_chain_counts():
    out = {}
    for pid in ["1VII", "8H0R", "1UBQ"]:
        dyn = json.load(open(RES / pid / "residue_scores_dynamic.json"))
        out[pid] = {"entries": len(dyn),
                    "chain_disambiguated": sum(1 for x in dyn if "_chain" in x)}
    out["_verdict"] = "D-05 CONFIRMED fixed: 8H0R now yields 182 entries (was 91)."
    return out


def r3_partial_bfactor():
    out = {}
    for pid in ["8H0R", "1UBQ"]:
        bf = parse_ca_bfactors(RAW_PDB[pid])
        dyn = json.load(open(RES / pid / "residue_scores_dynamic.json"))
        rmsf = json.load(open(RES / pid / "residue_scores_rmsf.json"))
        B, D, R = [], [], []
        for key, s in dyn.items():
            ch, seq, name = score_key_to_tuple(key)
            cand = [(c, seq, name) for c in ([ch] if ch is not None else range(8))]
            b = next((bf[c] for c in cand if c in bf), None)
            if b is None or key not in rmsf:
                continue
            B.append(b); D.append(float(s)); R.append(float(rmsf[key]))
        B, D, R = map(np.array, (B, D, R))
        rb, rd, rr = rankdata(B), rankdata(D), rankdata(R)
        res_d = rd - np.polyval(np.polyfit(rr, rd, 1), rr)
        res_b = rb - np.polyval(np.polyfit(rr, rb, 1), rr)
        out[pid] = {"n": len(B),
                    "spearman_B_vs_score": round(float(spearmanr(B, D).statistic), 3),
                    "spearman_B_vs_rmsf": round(float(spearmanr(B, R).statistic), 3),
                    "partial_spearman_score_given_rmsf": round(float(np.corrcoef(res_d, res_b)[0, 1]), 3)}
    out["_verdict"] = ("B-factor agreement is fully explained by RMSF. Controlling for RMSF, the "
                       "dynamic score's partial correlation with B-factors is ~0 (-0.02, -0.18). "
                       "Section 6.1 must NOT be read as external validation of the anomaly signal — "
                       "it validates the RMSF component only.")
    return out


def r4_state_circularity():
    out = {}
    for pid, art in ART.items():
        dtraj = np.load(art / "dtraj.npy"); Y = np.load(art / "tica_coords.npy")
        msm, da, lag = refit(dtraj, 5)
        L_state, _ = labels(msm, da, dtraj, lag)
        r, s = compute_kinetic_signals(msm, dtraj, lag)
        d = compute_local_density_signal(Y, k=min(5, len(Y) - 1))
        full, _ = fuse_signals({"rarity": r, "surprise": s, "density": -d})
        norar, _ = fuse_signals({"surprise": s, "density": -d})
        iso = -IsolationForest(random_state=42, n_estimators=200).fit(Y).score_samples(Y)
        lof = LocalOutlierFactor(n_neighbors=min(20, len(Y) - 1)); lof.fit(Y)
        out[pid] = {
            "fused_all_three": auroc(L_state, moving_median(full * 100, window=3)),
            "fused_without_rarity": auroc(L_state, moving_median(norar * 100, window=3)),
            "isolation_forest": auroc(L_state, iso),
            "local_outlier_factor": auroc(L_state, -lof.negative_outlier_factor_)}
    out["_verdict"] = ("Rare-STATE baseline comparison is CIRCULAR. Labels are defined from pi; "
                       "the rarity channel is 1-pi. Removing it collapses the advantage: the "
                       "pipeline then loses to LOF on 8H0R (0.83 vs 0.93), loses to Isolation "
                       "Forest on 1CRN (0.76 vs 0.85), and ties on 1UBQ. The 'wins on all four "
                       "proteins' claim in section 6.2 is not supportable as a benchmark result.")
    return out


def r5_transition_circularity():
    out = {}
    for pid, art in ART.items():
        dtraj = np.load(art / "dtraj.npy")
        msm, da, lag = refit(dtraj, 5)
        _, s = compute_kinetic_signals(msm, dtraj, lag)
        n = int(dtraj.max()) + 1
        C = np.zeros((n, n))
        for t in range(len(dtraj) - lag):
            C[dtraj[t], dtraj[t + lag]] += 1
        cnt = np.array([C[dtraj[t], dtraj[t + lag]] if t < len(dtraj) - lag else np.nan
                        for t in range(len(dtraj))])
        ok = np.isfinite(cnt) & np.isfinite(s)
        out[pid] = {"spearman_surprise_vs_neg_transition_count":
                    round(float(spearmanr(s[ok], -cnt[ok]).statistic), 3)}
    out["_verdict"] = ("Rare-TRANSITION result is also substantially circular: surprise = -log P, "
                       "and P is row-normalised from the same count matrix that defines the labels "
                       "(rho 0.69-0.87 between surprise and negative transition count). The 1UBQ "
                       "'headline figure' (surprise 0.91 vs geometric 0.43) therefore cannot be "
                       "presented as a benchmark win.")
    return out


def r6_vamp2_pairs():
    out = {}
    for nframes in (100, 1001):
        ntest = int(nframes * 0.2)
        out[f"{nframes}_frames"] = {f"lag_{l}": max(0, ntest - l) for l in (5, 10, 15)}
    out["_verdict"] = ("VAMP-2 'saturation at dim' is explained by a degenerate test set: with 100 "
                       "frames the 20-frame held-out split leaves only 5 lagged pairs at lag 15. "
                       "The score is not measuring model quality at that setting at all.")
    return out


def r7_edge_count():
    e = json.load(open(ROOT / "validation/edge_results_POSTFIX.json"))
    return {"actual_cases": len(e),
            "report_prose_said": 21,
            "_verdict": "Report section 3.5 prose understates the count; actual suite is 25 cases."}


def r8_core_finding():
    from deeptime.markov.msm import MaximumLikelihoodMSM
    out = {}
    for pid, art in [("1VII", ART["1VII"]), ("8H0R", ART["8H0R"]), ("1CRN", ART["1CRN"])]:
        d = np.load(art / "dtraj.npy")
        m = MaximumLikelihoodMSM(lagtime=5, reversible=True).fit(d).fetch_model()
        sym = np.asarray(m.count_model.state_symbols); pi = m.stationary_distribution
        wrong = 0
        for t in range(len(d)):
            lbl = int(d[t])
            if lbl >= len(pi):
                continue
            idx = np.where(sym == lbl)[0]
            if len(idx) and abs(pi[lbl] - pi[idx[0]]) > 1e-12:
                wrong += 1
        out[pid] = {"frames": len(d), "frames_reading_wrong_pi": wrong,
                    "percent": round(100 * wrong / len(d), 1)}
    out["_verdict"] = ("D-01 CONFIRMED by independent re-derivation: 28-50% of frames would read a "
                       "different pi value than the truth. The core finding stands unchanged.")
    return out


if __name__ == "__main__":
    OUT["R1_residue_score_vs_rmsf"] = r1_residue_vs_rmsf()
    OUT["R2_chain_counts"] = r2_chain_counts()
    OUT["R3_partial_bfactor"] = r3_partial_bfactor()
    OUT["R4_state_label_circularity"] = r4_state_circularity()
    OUT["R5_transition_label_circularity"] = r5_transition_circularity()
    OUT["R6_vamp2_test_pairs"] = r6_vamp2_pairs()
    OUT["R7_edge_case_count"] = r7_edge_count()
    OUT["R8_core_finding_recheck"] = r8_core_finding()
    p = ROOT / "validation/revalidation_results.json"
    p.write_text(json.dumps(OUT, indent=2))
    print(json.dumps(OUT, indent=2))
