# ================================================================ #
#  Module:      tools/export_unified.py
#  Description: Generates hotspots_unified.json — single file with
#               all residue-level signals for SciViz consumption
#  Author:      Siya Jethliya
#  Copyright (c) 2026 SciVizAI — All rights reserved.
# ================================================================ #

import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# ---------------------------------------------------------------- #
# Load all signal sources from results and artifacts directories
# ---------------------------------------------------------------- #
def load_signals(results_dir: Path, artifacts_dir: Path) -> dict:
    signals = {}

    # ── Dynamic residue scores (blended score)
    dynamic_path = results_dir / "residue_scores_dynamic.json"
    if dynamic_path.exists():
        with open(dynamic_path) as f:
            signals["dynamic_score"] = json.load(f)

    # ── Frame scores — extract per-component means per residue
    frame_path = results_dir / "frame_scores_dynamic.csv"
    if frame_path.exists():
        df = pd.read_csv(frame_path)
        signals["_frame_df"] = df

    # ── RMSF residue scores
    rmsf_path = results_dir / "residue_scores_rmsf.json"
    if rmsf_path.exists():
        with open(rmsf_path) as f:
            signals["rmsf"] = json.load(f)

    # ── tICA importance scores
    tica_path = results_dir / "residue_scores_tica_importance.json"
    if tica_path.exists():
        with open(tica_path) as f:
            signals["tica_importance"] = json.load(f)

    # ── ASVS exports if they exist
    exports_dir = results_dir.parent.parent / "exports" / results_dir.parent.name
    
    anomaly_path = exports_dir / "anomaly_residue.json"
    if anomaly_path.exists():
        with open(anomaly_path) as f:
            signals["anomaly"] = json.load(f)

    rmsf_export_path = exports_dir / "rmsf_residue.json"
    if rmsf_export_path.exists():
        with open(rmsf_export_path) as f:
            signals["rmsf_export"] = json.load(f)

    tica_export_path = exports_dir / "tica_importance.json"
    if tica_export_path.exists():
        with open(tica_export_path) as f:
            signals["tica_export"] = json.load(f)

    hotspot_path = exports_dir / "hotspots_residue.json"
    if hotspot_path.exists():
        with open(hotspot_path) as f:
            signals["hotspot"] = json.load(f)

    return signals

# ---------------------------------------------------------------- #
# Normalise a dict of residue → float values to [0, 1]
# ---------------------------------------------------------------- #
def normalise(values: dict) -> dict:
    if not values:
        return {}
    vals = list(values.values())
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {k: 0.5 for k in values}
    return {k: round((v - mn) / (mx - mn), 4) for k, v in values.items()}

# ---------------------------------------------------------------- #
# Build the unified per-residue signal dictionary
# ---------------------------------------------------------------- #
def build_unified(signals: dict) -> dict:
    unified = {}

    # Get residue list from dynamic scores (always present)
    dynamic = signals.get("dynamic_score", {})
    if not dynamic:
        raise ValueError("residue_scores_dynamic.json not found or empty")

    # Frame-level component means — rarity, transition_surprise, local_density
    frame_df = signals.get("_frame_df")
    component_means = {}
    if frame_df is not None:
        for col in ["component_rarity", "component_transition_surprise", "component_local_density"]:
            if col in frame_df.columns:
                # These are frame-level signals — mean across all frames
                component_means[col] = float(frame_df[col].mean())

    # Pull from available sources, fall back to None if not present
    rmsf        = signals.get("rmsf") or signals.get("rmsf_export", {})
    tica        = signals.get("tica_importance") or signals.get("tica_export", {})
    anomaly     = signals.get("anomaly", {})
    hotspot     = signals.get("hotspot", {})

    # Normalise RMSF and tICA if present
    rmsf_norm   = normalise(rmsf) if rmsf else {}
    tica_norm   = normalise(tica) if tica else {}

    for residue, score in dynamic.items():
        entry = {
            # ── Primary fused score [0-100]
            "dynamic_score": round(float(score), 4),

            # ── Component signals — frame-level means [0-100]
            "rarity":               round(component_means.get("component_rarity", 0.0), 4),
            "transition_surprise":  round(component_means.get("component_transition_surprise", 0.0), 4),
            "local_density":        round(component_means.get("component_local_density", 0.0), 4),

            # ── Structural signals — normalised [0-1]
            "rmsf_norm":            rmsf_norm.get(residue, None),
            "tica_importance":      tica_norm.get(residue, None),

            # ── Raw anomaly score if separately exported
            "anomaly_score":        round(float(anomaly[residue]), 4) if residue in anomaly else None,

            # ── Hotspot flag
            "is_hotspot":           bool(hotspot.get(residue, False)) if hotspot else bool(score >= 60),

            # ── Metadata
            "residue":              residue,
        }
        unified[residue] = entry

    return unified

# ---------------------------------------------------------------- #
# Write unified JSON to output path
# ---------------------------------------------------------------- #
def write_unified(unified: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "meta": {
                "version":      "1.0",
                "producer":     "SciML-MD",
                "description":  "Unified per-residue dynamical signals for SciViz",
                "signals": [
                    "dynamic_score",
                    "rarity",
                    "transition_surprise",
                    "local_density",
                    "rmsf_norm",
                    "tica_importance",
                    "anomaly_score",
                    "is_hotspot"
                ]
            },
            "residues": unified
        }, f, indent=2)
    print(f"[ok] hotspots_unified.json → {output_path}")
    print(f"     {len(unified)} residues, {sum(1 for r in unified.values() if r['is_hotspot'])} hotspots")

# ---------------------------------------------------------------- #
# Main entry point
# ---------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Generate hotspots_unified.json for SciViz"
    )
    parser.add_argument("--pdb_id",       required=True,  help="PDB ID e.g. 1CRN")
    parser.add_argument("--results_dir",  default="results",  help="Results directory")
    parser.add_argument("--artifacts_dir",default="artifacts", help="Artifacts directory")
    parser.add_argument("--output_dir",   default="exports",  help="Output directory")
    args = parser.parse_args()

    results_dir   = Path(args.results_dir)  / args.pdb_id
    artifacts_dir = Path(args.artifacts_dir) / args.pdb_id
    output_path   = Path(args.output_dir)   / args.pdb_id / "hotspots_unified.json"

    print(f"\nGenerating unified hotspot file for {args.pdb_id}")
    print(f"Results:   {results_dir}")
    print(f"Artifacts: {artifacts_dir}")
    print(f"Output:    {output_path}\n")

    signals = load_signals(results_dir, artifacts_dir)
    unified = build_unified(signals)
    write_unified(unified, output_path)

if __name__ == "__main__":
    main()
