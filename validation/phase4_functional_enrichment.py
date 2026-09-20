#!/usr/bin/env python3
"""
Phase 4 — validate what the PRODUCT actually shows: the residue hotspot map.

WHY THIS IS THE MISSING PIECE
-----------------------------
Every validated claim so far is FRAME-level. Phase 2 showed that transition
surprise separates geometrically invisible junction FRAMES from ordinary ones
(AUROC 0.741, all time-blind detectors at chance). Phase 3 scales that same
frame-level claim across proteins.

The visualisation software does not show frames. It paints RESIDUES. That is a
different claim, and it currently has no external evidence behind it:

  - The original residue score correlated with RMSF at rho 0.94-0.98, i.e. it
    was a flexibility map wearing a different label (erratum E-1).
  - The fix orthogonalises against RMSF and drives that to ~0.00.
  - But "not RMSF" is not "correct". Nothing has yet shown the residues it
    highlights mean anything.

Shipping a colour map that a user reads as "these residues matter" requires
evidence that they do. This is that test.

THE TEST
--------
Ground truth comes from the deposited crystal structure, NOT from our
trajectory and NOT from any quantity our pipeline consumes:

    a residue is FUNCTIONAL if any of its atoms lies within `cutoff` of a
    ligand, metal ion, nucleic acid, or partner protein chain in the deposited
    PDB entry.

The MD our pipeline sees is the isolated apo chain - ATLAS strips everything
else before simulating - so the label cannot leak into the score. This is the
non-circularity that the retracted rare-state and rare-transition comparisons
lacked (errata E-3, E-4).

We then ask whether the hotspot ranking recovers those residues, and compare
against the baseline that matters:

    RMSF. If plain flexibility scores as well as the attribution, the kinetic
    machinery adds nothing a user could not get from a B-factor colouring, and
    the product should say so rather than imply otherwise.

PRE-REGISTERED CRITERIA  (fix before running; never adjust after seeing results)
    F1  median AUROC (attribution vs functional label) >= 0.65 across proteins
    F2  attribution > RMSF baseline on >= 70% of proteins
    F3  shuffled-score null median within 0.45-0.55        [gates F1/F2/F4]
    F4  Wilcoxon signed-rank p < 0.05, attribution vs RMSF

HOW TO READ A FAILURE
    F1 fails, F2 passes  -> the map carries real signal that plain flexibility
        does not, but functional sites are not what it finds. Ligand pockets are
        often the RIGID part of a protein, so a dynamics-driven detector can be
        genuinely informative and still score near chance here. Report it as a
        negative result against THIS label, not against the method.
    F2 fails             -> the attribution is not adding value over RMSF. The
        product should colour by RMSF and drop the claim.
    F3 fails             -> the harness is broken; ignore every other number.

USAGE
    python validation/phase4_functional_enrichment.py --data_root ~/atlas
    python validation/phase4_functional_enrichment.py --data_root ~/atlas \\
        --only 5z42_A 6kty_A --cutoff 4.5
"""
import argparse
import json
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

RCSB = "https://files.rcsb.org/download/{pdb}.pdb"

# Solvent, cryoprotectants and buffer components are crystallisation artefacts,
# not function. Labelling residues by their proximity would add noise that has
# nothing to do with what the protein does.
IGNORE_HET = {
    "HOH", "DOD", "WAT", "SO4", "PO4", "GOL", "EDO", "PEG", "PGE", "MPD",
    "TRS", "MES", "EPE", "ACT", "ACY", "FMT", "CL", "NA", "K", "BR", "IOD",
    "DMS", "IPA", "MRD", "BME", "NH4", "AZI", "NO3", "CIT", "TLA", "SCN",
}
NUCLEIC = {"DA", "DT", "DG", "DC", "DU", "A", "U", "G", "C", "I", "N"}

# VALIDITY PRECONDITIONS on the label, applied identically to both arms.
# An AUROC computed on 3 positives is noise: 3dso_A scored 0.831 on three
# labelled residues in the first run, the single best number in the table and
# entirely meaningless. 4v1g_B had 77 of 86 residues labelled - a ranking
# cannot be evaluated when 90% of the answer is "yes".
MIN_SITE_RESIDUES = 5
MIN_PREVALENCE = 0.05
MAX_PREVALENCE = 0.80

# HIS protonation variants and terminal forms differ between force field and
# crystal; normalise before comparing sequences.
RESNAME_ALIAS = {"HIE": "HIS", "HID": "HIS", "HIP": "HIS", "HSD": "HIS",
                 "HSE": "HIS", "HSP": "HIS", "CYX": "CYS", "CYM": "CYS",
                 "ASH": "ASP", "GLH": "GLU", "LYN": "LYS", "MSE": "MET"}


def _norm(name):
    n = str(name).strip().upper()
    return RESNAME_ALIAS.get(n, n)


def _json_safe(o):
    """numpy scalars are not JSON serializable and killed the first run after
    every number had already been computed."""
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


# --------------------------------------------------------------- ground truth
def fetch_pdb(pdb_id, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{pdb_id.lower()}.pdb"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    url = RCSB.format(pdb=pdb_id.upper())
    req = urllib.request.Request(url, headers={"User-Agent": "SciML-MD/phase4"})
    with urllib.request.urlopen(req, timeout=120) as r:
        dest.write_bytes(r.read())
    return dest


def parse_pdb_atoms(path: Path):
    """Minimal fixed-column PDB reader. Returns (target_by_chain, partners).

    target_by_chain[chain] = list of (resseq, icode, resname, xyz)
    partners               = list of (kind, xyz) for every non-water HETATM,
                             nucleic-acid atom and other-chain protein atom.
    Only the first MODEL is read - NMR ensembles would otherwise be counted
    many times over.
    """
    prot, het, nuc = {}, [], []
    for raw in path.read_text(errors="ignore").splitlines():
        rec = raw[:6]
        if rec == "ENDMDL":
            break
        if rec not in ("ATOM  ", "HETATM"):
            continue
        alt = raw[16]
        if alt not in (" ", "A"):
            continue
        resname = raw[17:20].strip().upper()
        chain = raw[21].strip() or " "
        try:
            resseq = int(raw[22:26])
            xyz = (float(raw[30:38]), float(raw[38:46]), float(raw[46:54]))
        except ValueError:
            continue
        icode = raw[26].strip()

        if rec == "HETATM":
            if resname not in IGNORE_HET:
                het.append(xyz)
            continue
        if resname in NUCLEIC:
            nuc.append(xyz)
            continue
        prot.setdefault(chain, []).append((resseq, icode, resname, xyz))
    return prot, het, nuc


def functional_labels(pdb_path: Path, chain_id, cutoff_ang=4.5):
    """Label residues of `chain_id` that contact a functional partner.

    Returns (resseq_array, label_array, breakdown) where breakdown counts how
    many residues each partner class contributed."""
    prot, het, nuc = parse_pdb_atoms(pdb_path)
    if chain_id not in prot:
        avail = ",".join(sorted(prot))
        raise KeyError(f"chain {chain_id} absent from {pdb_path.name} (have {avail})")

    target = prot[chain_id]
    others = [xyz for c, atoms in prot.items() if c != chain_id
              for (_, _, _, xyz) in atoms]

    keys, coords, resname_of = [], [], {}
    for resseq, icode, resname, xyz in target:
        keys.append((resseq, icode))
        coords.append(xyz)
        resname_of.setdefault((resseq, icode), _norm(resname))
    coords = np.asarray(coords, dtype=np.float64)

    uniq, inverse = [], np.empty(len(keys), dtype=np.int64)
    seen = {}
    for i, k in enumerate(keys):
        if k not in seen:
            seen[k] = len(uniq)
            uniq.append(k)
        inverse[i] = seen[k]

    lab = np.zeros(len(uniq), dtype=bool)
    lab_ligand = np.zeros(len(uniq), dtype=bool)
    breakdown = {}
    for name, pts in (("ligand", het), ("nucleic", nuc), ("chain", others)):
        if not pts:
            breakdown[name] = 0
            continue
        P = np.asarray(pts, dtype=np.float64)
        hit = np.zeros(len(uniq), dtype=bool)
        # chunk to keep the distance matrix bounded
        for s in range(0, len(P), 4000):
            d = np.linalg.norm(coords[:, None, :] - P[None, s:s + 4000, :], axis=2)
            close = (d < cutoff_ang).any(axis=1)
            np.logical_or.at(hit, inverse, close)
        breakdown[name] = int(hit.sum())
        lab |= hit
        if name in ("ligand", "nucleic"):
            lab_ligand |= hit

    return {"resseq": np.array([u[0] for u in uniq]),
            "resname": np.array([resname_of[u] for u in uniq]),
            "label_all": lab,
            "label_ligand": lab_ligand,
            "breakdown": breakdown}


def align_md_to_crystal(md_resnames, cry_resnames, cry_resseq, md_resseq):
    """Map each MD residue onto a crystal residue.

    ATLAS topologies do not always carry the author numbering of the deposited
    entry. The first run matched on resSeq alone and lost FIVE proteins to
    "0 residues matched" - 1upt_D, 3bpj_B, 3bzl_D, 4eo1_A, 6l4p_B - which is a
    defect in this harness, not a property of the data.

    Strategy: try resSeq first, since it is exact when it works. If that covers
    less than half the chain, slide the two residue-NAME sequences against each
    other and take the offset with the most agreement. Returns an index array
    into the crystal arrays (-1 where unmatched) plus a short description.
    """
    md_n, cry_n = len(md_resnames), len(cry_resnames)
    by_seq = {int(s): i for i, s in enumerate(cry_resseq)}
    direct = np.array([by_seq.get(int(s), -1) for s in md_resseq])
    agree = sum(1 for i, j in enumerate(direct)
                if j >= 0 and md_resnames[i] == cry_resnames[j])
    if agree >= 0.5 * md_n:
        return direct, f"resSeq ({agree}/{md_n} names agree)"

    best, best_off = -1, None
    for off in range(-md_n + 1, cry_n):
        hit = 0
        for i in range(md_n):
            j = i + off
            if 0 <= j < cry_n and md_resnames[i] == cry_resnames[j]:
                hit += 1
        if hit > best:
            best, best_off = hit, off
    if best_off is None or best < 0.5 * md_n:
        return None, (f"sequence alignment failed (best {best}/{md_n} at "
                      f"offset {best_off})")
    idx = np.array([i + best_off if 0 <= i + best_off < cry_n else -1
                    for i in range(md_n)])
    # only keep positions whose residue name actually agrees
    idx = np.array([j if (j >= 0 and md_resnames[i] == cry_resnames[j]) else -1
                    for i, j in enumerate(idx)])
    return idx, f"sequence offset {best_off:+d} ({best}/{md_n} names agree)"


# ------------------------------------------------------------------- scoring
def score_residues(traj, seed=42):
    """Run the production path and return (resseq, attribution, rmsf)."""
    from validation.phase2_temporal_scramble import score_all
    from scoring.residue_attribution import compute_residue_attribution
    from validation.phase3_scale import scale_hyperparams

    hp = scale_hyperparams(len(traj))
    sc = score_all(traj, lag_tica=hp["lag_tica"], dim=hp["dim"],
                   n_clusters=hp["n_clusters"], lag_msm=hp["lag_msm"],
                   k=hp["k"], seed=seed)
    frame_scores = np.nan_to_num(sc["pipeline_fused"], nan=50.0)

    att = compute_residue_attribution(traj, frame_scores)
    ca = traj.topology.select("name CA")
    resseq = np.array([traj.topology.atom(i).residue.resSeq for i in ca])
    resname = np.array([_norm(traj.topology.atom(i).residue.name) for i in ca])
    return resseq, resname, np.asarray(att["score"]), np.asarray(att["rmsf_ang"]), att


def label_is_valid(label):
    """Reject labels on which no AUROC is interpretable, before scoring."""
    n, k = len(label), int(label.sum())
    if k < MIN_SITE_RESIDUES:
        return f"only {k} functional residue(s) (need {MIN_SITE_RESIDUES})"
    if n - k < MIN_SITE_RESIDUES:
        return f"only {n-k} non-functional residue(s)"
    prev = k / n
    if not MIN_PREVALENCE <= prev <= MAX_PREVALENCE:
        return (f"prevalence {prev:.0%} outside "
                f"{MIN_PREVALENCE:.0%}-{MAX_PREVALENCE:.0%}")
    return None


def evaluate(score, label, rng):
    """AUROC/AUPRC of one residue score against the functional label."""
    if label_is_valid(label) is not None:
        return None
    shuffled = rng.permutation(score)
    return {
        "auroc": float(roc_auc_score(label, score)),
        "auprc": float(average_precision_score(label, score)),
        "auroc_shuffled": float(roc_auc_score(label, shuffled)),
        "prevalence": float(label.mean()),
    }


# ---------------------------------------------------------------------- main
def run_one(name, top, trjs, cache, cutoff, max_frames, rng):
    """Score every available replicate and average. ATLAS verdicts disagree
    between replicates (2hnu_A is single-basin in R1 and eight-basin in R2), so
    a one-replicate number carries that variance straight into the result."""
    from validation.phase3_scale import load_traj

    pdb_id, chain = name.split("_")[0], name.split("_")[-1]
    gt = functional_labels(fetch_pdb(pdb_id, cache), chain, cutoff)

    per_rep, notes = [], []
    for trj in trjs:
        traj = load_traj(top, trj, max_frames)
        md_seq, md_name, att, rmsf, raw = score_residues(traj)
        idx, how = align_md_to_crystal(md_name, gt["resname"], gt["resseq"],
                                       md_seq)
        if idx is None:
            raise ValueError(how)
        notes.append(how)
        keep = idx >= 0
        if keep.sum() < 3 * MIN_SITE_RESIDUES:
            raise ValueError(f"only {int(keep.sum())} residues aligned; {how}")
        per_rep.append({
            "att": att[keep], "rmsf": rmsf[keep],
            "y_all": gt["label_all"][idx[keep]],
            "y_lig": gt["label_ligand"][idx[keep]],
            "diag": raw.get("diagnostics", {}),
        })

    out = {"protein": name, "pdb": pdb_id, "chain": chain,
           "n_replicates": len(per_rep),
           "n_residues_compared": int(len(per_rep[0]["att"])),
           "alignment": notes[0],
           "label_breakdown": gt["breakdown"],
           "attribution_diagnostics": per_rep[0]["diag"]}

    # PRIMARY: the pre-registered composite label (ligand/ion/nucleic/chain).
    # SECONDARY, exploratory: ligand/ion/nucleic only. Interface residues can
    # cover most of a small chain - 4v1g_B was 77/86 - and that is a different
    # biological question from a binding site. The secondary is reported but
    # the acceptance criteria are judged on the primary, as pre-registered.
    for key, ykey in (("primary", "y_all"), ("ligand_only", "y_lig")):
        y = per_rep[0][ykey]
        bad = label_is_valid(y)
        if bad:
            out[key] = {"skipped": bad, "n_functional": int(y.sum())}
            continue
        a = [evaluate(r["att"], r[ykey], rng) for r in per_rep]
        f = [evaluate(r["rmsf"], r[ykey], rng) for r in per_rep]
        a = [x for x in a if x]
        f = [x for x in f if x]
        if not a or not f:
            out[key] = {"skipped": "no replicate produced a score"}
            continue
        out[key] = {
            "n_functional": int(y.sum()),
            "prevalence": round(float(y.mean()), 3),
            "auroc": round(float(np.mean([x["auroc"] for x in a])), 3),
            "auroc_per_replicate": [round(x["auroc"], 3) for x in a],
            "auprc": round(float(np.mean([x["auprc"] for x in a])), 3),
            "rmsf_auroc": round(float(np.mean([x["auroc"] for x in f])), 3),
            "auroc_shuffled": round(
                float(np.mean([x["auroc_shuffled"] for x in a])), 3),
        }
        out[key]["delta_auroc"] = round(
            out[key]["auroc"] - out[key]["rmsf_auroc"], 3)

    if "skipped" in out.get("primary", {}):
        raise ValueError(out["primary"]["skipped"])
    rho, _ = spearmanr(per_rep[0]["att"], per_rep[0]["rmsf"])
    out["spearman_attribution_vs_rmsf"] = round(float(rho), 3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--screen", default="validation/phase3_screen.json",
                    help="Restrict to proteins the screen marked usable")
    ap.add_argument("--cutoff", type=float, default=4.5,
                    help="Contact cutoff in Angstrom (default 4.5)")
    ap.add_argument("--max_frames", type=int, default=2000)
    ap.add_argument("--max-replicates", type=int, default=3,
                    help="Replicates to score and average per protein")
    ap.add_argument("--cache", default="~/atlas/_rcsb",
                    help="Where to cache downloaded crystal structures")
    ap.add_argument("--out", default="validation/phase4_results.json")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser()
    cache = Path(args.cache).expanduser()
    rng = np.random.default_rng(0)

    names = args.only
    if not names:
        sp = Path(args.screen)
        if sp.exists():
            scr = json.loads(sp.read_text())
            names = scr.get("usable_proteins") or []
            print(f"using {len(names)} usable protein(s) from {sp}")
        if not names:
            names = sorted(p.name for p in root.iterdir()
                           if p.is_dir() and not p.name.startswith("_"))
    if not names:
        print("nothing to evaluate")
        return 1

    rows, failed = [], {}
    print(f"\nfunctional label: residue within {args.cutoff} A of a ligand, ion, "
          f"nucleic acid or partner chain in the deposited entry\n")
    print(f"{'protein':10s} {'res':>4s} {'func':>5s} {'AUROC':>6s} {'RMSF':>6s} "
          f"{'delta':>6s} {'null':>6s} {'lig-only':>8s}  breakdown")
    print("-" * 92)
    for name in names:
        d = root / name
        tops = sorted(d.glob("*.pdb"))
        trjs = sorted(d.glob("*.xtc"))[: args.max_replicates]
        if not tops or not trjs:
            failed[name] = "no pdb/xtc"
            continue
        try:
            r = run_one(name, tops[0], trjs, cache, args.cutoff,
                        args.max_frames, rng)
        except Exception as e:                              # noqa: BLE001
            failed[name] = f"{type(e).__name__}: {e}"[:120]
            print(f"{name:10s} FAILED {failed[name]}"[:78], flush=True)
            continue
        rows.append(r)
        bd = ",".join(f"{k}:{v}" for k, v in r["label_breakdown"].items() if v)
        pr = r["primary"]
        lig = r.get("ligand_only", {})
        ligs = f"{lig['auroc']:8.3f}" if "auroc" in lig else f"{'-':>8s}"
        print(f"{r['protein']:10s} {r['n_residues_compared']:4d} "
              f"{pr['n_functional']:5d} {pr['auroc']:6.3f} "
              f"{pr['rmsf_auroc']:6.3f} {pr['delta_auroc']:+6.3f} "
              f"{pr['auroc_shuffled']:6.3f} {ligs}  {bd}", flush=True)

    if len(rows) < 3:
        print(f"\nonly {len(rows)} protein(s) evaluated - not enough to judge.")
        Path(args.out).write_text(json.dumps({"rows": rows, "failed": failed},
                                             indent=2, default=_json_safe))
        return 1

    att = np.array([r["primary"]["auroc"] for r in rows])
    rms = np.array([r["primary"]["rmsf_auroc"] for r in rows])
    null = np.array([r["primary"]["auroc_shuffled"] for r in rows])

    f1 = float(np.median(att))
    f2 = float((att > rms).mean())
    f3 = float(np.median(null))
    try:
        _, p = wilcoxon(att, rms)
    except ValueError:
        p = 1.0

    acc = {
        "F1_median_auroc_ge_0.65": {"value": round(f1, 3), "pass": f1 >= 0.65},
        "F2_beats_rmsf_on_70pct": {"value": round(f2, 3), "pass": f2 >= 0.70},
        "F3_null_within_0.45_0.55": {"value": round(f3, 3),
                                     "pass": 0.45 <= f3 <= 0.55},
        "F4_wilcoxon_p_lt_0.05": {"value": round(float(p), 5), "pass": p < 0.05},
    }

    print(f"\n=== PHASE 4 ({len(rows)} proteins) ===")
    print(f"  attribution AUROC  median {f1:.3f}  "
          f"IQR {np.percentile(att,25):.3f}-{np.percentile(att,75):.3f}")
    print(f"  RMSF baseline      median {np.median(rms):.3f}")
    print(f"  shuffled null      median {f3:.3f}")
    print(f"  beats RMSF on      {f2*100:.0f}% of proteins")
    print("\nACCEPTANCE:")
    for k, v in acc.items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k} = {v['value']}")

    if not acc["F3_null_within_0.45_0.55"]["pass"]:
        print("\n  F3 FAILED - the harness is broken. Ignore F1/F2/F4 entirely.")
    elif not acc["F2_beats_rmsf_on_70pct"]["pass"]:
        print("\n  The attribution does not beat plain RMSF. On this evidence the")
        print("  product should colour residues by flexibility and NOT claim the")
        print("  kinetic channel localises anything RMSF does not.")
    elif not acc["F1_median_auroc_ge_0.65"]["pass"]:
        print("\n  Signal beyond RMSF, but functional sites are not what it finds.")
        print("  Ligand pockets are often the rigid core, so this is a negative")
        print("  result against THIS label, not against the method. Do not report")
        print("  functional relevance; report the Phase 2/3 kinetic claim instead.")
    else:
        print("\n  The residue map recovers externally-defined functional sites")
        print("  better than flexibility does. This is the claim the product can make.")

    lig = [r["ligand_only"] for r in rows if "auroc" in r.get("ligand_only", {})]
    if len(lig) >= 3:
        la = np.array([x["auroc"] for x in lig])
        lr = np.array([x["rmsf_auroc"] for x in lig])
        print(f"\nSECONDARY (exploratory, ligand/ion/nucleic only, "
              f"n={len(lig)}): attribution median {np.median(la):.3f}, "
              f"RMSF {np.median(lr):.3f}, beats RMSF on {100*(la>lr).mean():.0f}%")
        print("  Not a pre-registered criterion. Reported so the composite "
              "label cannot hide a difference between binding sites and "
              "interfaces.")

    Path(args.out).write_text(json.dumps(
        {"cutoff_ang": args.cutoff, "n_proteins": len(rows),
         "acceptance": acc, "rows": rows, "failed": failed},
        indent=2, default=_json_safe))
    print(f"\nsaved -> {args.out}")
    if failed:
        print(f"{len(failed)} failed: {json.dumps(failed, indent=2)[:500]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
