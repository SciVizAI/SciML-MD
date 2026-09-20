"""The one export the visualiser should consume.

WHY THIS REPLACES export_unified.py
-----------------------------------
`tools/export_unified.py` builds `hotspots_unified.json` with fields called
`anomaly_score` and `hotspot`. `PIPELINE_OVERVIEW.md` labels that file "combined
for viewer", and `exports/` ships built examples of it. So it is the file an
engineer will naturally wire up - and it carries no trust information at all.
Every gate in `msm/trust.py` is bypassed, and four withdrawn claims (CLAIMS.md
W-1, W-2, W-5, W-6) ship straight into the interface under their original names.

This module is the governed replacement. One system in, one JSON out, and the
trust contract decides what is in it.

THE RULE THAT MAKES THIS SAFE
-----------------------------
A withheld channel is not exported with a flag, or as nulls, or greyed out.
**It is absent from the file.** A viewer cannot render data it was never given,
which is the only form of enforcement that survives contact with a deadline.
What each omission was, and why, is listed under `omitted` so the interface can
explain the gap honestly instead of showing an unexplained blank.

Usage
  python tools/export_for_viewer.py --system 1g2r_A
  python tools/export_for_viewer.py --all --out viewer/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCHEMA = "sciml-md/viewer@1"


def _read_trust(results_dir: Path) -> dict:
    """Load the contract. A missing or unparseable contract is fatal.

    Failing closed is the entire point: if we cannot prove a number is safe to
    display, we do not ship it. The old export had no such check and therefore
    no such guarantee.
    """
    p = results_dir / "trust.json"
    if not p.exists():
        raise SystemExit(
            f"ERROR: no trust contract at {p}\n"
            "       Run run_all_proteins.py first. Exporting scores without a\n"
            "       contract is exactly the failure this module exists to prevent."
        )
    trust = json.loads(p.read_text())
    if trust.get("schema") != "sciml-md/trust@1":
        raise SystemExit(f"ERROR: unexpected contract schema {trust.get('schema')!r}")
    return trust


def build_bundle(system: str, results_dir: Path, artifacts_dir: Path) -> dict:
    import pandas as pd

    trust = _read_trust(results_dir)
    channels = trust.get("channels", {})
    diag = trust.get("diagnostics", {}) or {}
    rec = diag.get("recurrence") or {}
    est = diag.get("estimability") or {}
    basin = diag.get("basin_census") or {}

    bundle = {
        "schema": SCHEMA,
        "claims_version": trust.get("claims_version"),
        "system": system,
        # ---- what the interface must show FIRST (INTEGRATION.md s3) --------
        "verdict": {
            "suitability": trust.get("verdict"),
            "recurrence": trust.get("recurrence_verdict"),
            "estimability": trust.get("estimability_verdict"),
            "displayable": trust.get("displayable"),
            "headline": trust.get("headline"),
            "reasons": basin.get("reasons", []),
        },
        # ---- sampling context; without this every number below is over-read -
        "sampling": {
            "n_frames": trust.get("n_frames"),
            "total_ns": rec.get("total_ns"),
            "stride_ns": rec.get("stride_ns"),
            "convergence_threshold_ns": est.get("pi_convergence_ns"),
            "shortfall_factor": est.get("shortfall_factor"),
            "threshold_extrapolated": est.get("threshold_extrapolated"),
            "threshold_caveat": est.get("threshold_caveat"),
            "caption": _sampling_caption(rec, est),
        },
        "diagnostics": {
            "basin_census": {k: basin.get(k) for k in
                             ("n_basins", "n_basins_ge_5pct",
                              "largest_basin_occupancy", "max_pairwise_ca_rmsd_ang")},
            "recurrence": {k: rec.get(k) for k in
                           ("closest_pair_percentile", "atom_selection",
                            "stride_ns", "question", "comparison_selections")},
            "estimability": {k: est.get(k) for k in
                             ("singleton_mass", "n_states_occupied",
                              "n_states_revisited", "reason")},
        },
        "frames": {"index": [], "channels": {}},   # channels absent = do not plot
        "residues": {},
        "omitted": {},
        "display_rules": trust.get("display_rules", []),
    }

    # An unsuitable trajectory cannot support ANY model-derived quantity, so
    # none is exported. Handing over data the contract says must not be rendered
    # and trusting the consumer to check a boolean is how gates get lost.
    # residue_rmsf is deliberately still exported below: it is measured straight
    # from the coordinates, needs no model, and is unaffected by suitability.
    displayable = bool(trust.get("displayable"))
    if not displayable:
        for name in ("score_dynamic", "rarity", "transition_surprise",
                     "local_density", "residue_scores"):
            bundle["omitted"][name] = (
                f"system verdict is {trust.get('verdict')!r}: "
                + (trust.get("headline") or "no model-derived quantity is exported"))

    # ---- frame channels: withheld ones are ABSENT, not null ---------------
    csv = results_dir / "frame_scores_dynamic.csv"
    if displayable and csv.exists():
        df = pd.read_csv(csv)
        bundle["frames"]["index"] = df["frame"].astype(int).tolist()
        for name in ("score_dynamic", "rarity", "transition_surprise",
                     "local_density"):
            col = name if name == "score_dynamic" else f"component_{name}"
            status = channels.get(name, {}).get("status")
            if status == "withheld" or col not in df.columns:
                bundle["omitted"][name] = channels.get(name, {}).get(
                    "reason", "not produced by this run")
                continue
            v = df[col].to_numpy(dtype=float)
            if np.isnan(v).all():
                bundle["omitted"][name] = (
                    "every value is NaN, which means it was gated upstream")
                continue
            bundle["frames"]["channels"][name] = {
                "status": status,
                "reason": channels.get(name, {}).get("reason"),
                "claims": channels.get(name, {}).get("claims"),
                "values": [None if np.isnan(x) else round(float(x), 4) for x in v],
            }

    # ---- residue level ----------------------------------------------------
    rmsf_p = results_dir / "residue_scores_rmsf.json"
    score_p = results_dir / "residue_scores_dynamic.json"
    if rmsf_p.exists():
        rmsf = json.loads(rmsf_p.read_text())
        bundle["residues"]["ids"] = list(rmsf.keys())
        bundle["residues"]["rmsf_ang"] = {
            "status": channels.get("residue_rmsf", {}).get("status"),
            "reason": channels.get("residue_rmsf", {}).get("reason"),
            "claims": channels.get("residue_rmsf", {}).get("claims"),
            "values": [round(float(v), 4) for v in rmsf.values()],
        }
    if score_p.exists() and displayable:
        st = channels.get("residue_scores", {}).get("status")
        if st == "withheld":
            bundle["omitted"]["residue_scores"] = channels["residue_scores"]["reason"]
        else:
            sc = json.loads(score_p.read_text())
            bundle["residues"].setdefault("ids", list(sc.keys()))
            bundle["residues"]["scores"] = {
                "status": st,
                "reason": channels.get("residue_scores", {}).get("reason"),
                "claims": channels.get("residue_scores", {}).get("claims"),
                "values": [round(float(v), 4) for v in sc.values()],
            }
    return bundle


def _sampling_caption(rec, est):
    """One sentence the interface can print verbatim under the verdict."""
    ns, stride = rec.get("total_ns"), rec.get("stride_ns")
    short = est.get("shortfall_factor")
    if ns is None:
        return "Sampling length unknown."
    base = f"{ns:.0f} ns at {stride:g} ns per frame"
    if short:
        return (f"{base} - {short:.0f}x below the "
                f"{est.get('pi_convergence_ns', 0):.0f} ns threshold at which "
                "state populations are considered converged.")
    return base + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--out", default="viewer")
    args = ap.parse_args()

    rroot, aroot = ROOT / args.results_dir, ROOT / args.artifacts_dir
    if args.all:
        systems = sorted(p.name for p in rroot.iterdir()
                         if p.is_dir() and (p / "trust.json").exists())
        if not systems:
            raise SystemExit(f"ERROR: no system under {rroot} has a trust.json")
    elif args.system:
        systems = [args.system]
    else:
        raise SystemExit("ERROR: pass --system SYS or --all")

    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    for sid in systems:
        b = build_bundle(sid, rroot / sid, aroot / sid)
        (outdir / f"{sid}.json").write_text(json.dumps(b, indent=2, default=str))
        shown = ", ".join(b["frames"]["channels"]) or "none"
        held = ", ".join(b["omitted"]) or "none"
        print(f"  {sid:12s} -> {args.out}/{sid}.json   shown: {shown} | omitted: {held}")
    print(f"\n{len(systems)} bundle(s) written to {args.out}/")


if __name__ == "__main__":
    main()
