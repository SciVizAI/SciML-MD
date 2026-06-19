#!/usr/bin/env python3
"""
Export ML pipeline outputs for the ASVS visualization tool.

This script converts the ensemble-anomaly-maps pipeline outputs into the
JSON formats expected by the ASVS viewer (https://github.com/chiranjibsur/asvs).

Usage:
    python tools/export_for_asvs.py \
        --topology data/raw_trajectory/align_topol.pdb \
        --trajectory data/raw_trajectory/trajectory_0.xtc \
        --msm_dir outputs/msm \
        --metrics_dir outputs/metrics \
        --output_dir /path/to/asvs/viewer
"""
import argparse
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Default values when data is not available
DEFAULT_N_FRAMES = 100
DEFAULT_N_RESIDUES = 374


def load_frame_scores(metrics_dir: Path) -> pd.DataFrame:
    """Load per-frame scores from CSV."""
    csv_path = metrics_dir / 'frame_scores_dynamic.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def load_residue_scores(metrics_dir: Path) -> dict:
    """Load per-residue scores from JSON files."""
    scores = {}
    
    for name in ['dynamic', 'rmsf', 'tica_importance']:
        json_path = metrics_dir / f'residue_scores_{name}.json'
        if json_path.exists():
            with open(json_path) as f:
                scores[name] = json.load(f)
    
    return scores


def create_per_frame_residue_json(
    frame_scores: pd.DataFrame, 
    residue_scores: dict, 
    n_residues: int,
    description: str = None
) -> dict:
    """
    Create per-frame, per-residue scores in ASVS format.
    
    Format:
    {
      "0": {"0": 0.05, "1": 0.10, ...},
      "1": {"0": 0.03, "1": 0.12, ...},
      ...
    }
    
    Args:
        frame_scores: DataFrame with per-frame scores
        residue_scores: Dictionary of per-residue score dictionaries
        n_residues: Number of residues
        description: Optional description to include in output
        
    Returns:
        Dictionary in ASVS format
    """
    output = {}
    
    if description:
        output["description"] = description
    
    if frame_scores is None:
        # Use static residue scores for all frames
        for frame in range(DEFAULT_N_FRAMES):
            output[str(frame)] = {str(i): 0.5 for i in range(n_residues)}
        return output
    
    n_frames = len(frame_scores)
    
    # Get dynamic residue scores if available
    dynamic_scores = residue_scores.get('dynamic', {})
    
    for frame_idx in range(n_frames):
        frame_data = {}
        # Get frame-level score
        frame_score = float(frame_scores.iloc[frame_idx]['score']) if 'score' in frame_scores.columns else 0.5
        
        for res_idx in range(n_residues):
            # Combine frame score with residue importance
            res_score = float(dynamic_scores.get(str(res_idx), 0.5))
            # Weight by frame score
            combined = res_score * frame_score
            frame_data[str(res_idx)] = round(combined, 6)
        
        output[str(frame_idx)] = frame_data
    
    return output


def create_hotspots_residue_json(frame_scores: pd.DataFrame, residue_scores: dict, n_residues: int) -> dict:
    """Create per-frame, per-residue hotspot scores in ASVS format."""
    return create_per_frame_residue_json(frame_scores, residue_scores, n_residues)


def create_anomaly_residue_json(frame_scores: pd.DataFrame, residue_scores: dict, n_residues: int) -> dict:
    """Create per-frame, per-residue anomaly scores in ASVS format."""
    return create_per_frame_residue_json(
        frame_scores, 
        residue_scores, 
        n_residues,
        description="Dynamic anomaly scores per residue per frame from ensemble-anomaly-maps pipeline"
    )


def create_rmsf_json(residue_scores: dict) -> dict:
    """
    Create RMSF data in ASVS format.
    
    Format:
    {
      "min": 0.45,
      "max": 8.68,
      "normalized": {"0": 0.05, "1": 0.04, ...}
    }
    """
    rmsf_scores = residue_scores.get('rmsf', {})
    
    if not rmsf_scores:
        return {"min": 0.0, "max": 1.0, "normalized": {}}
    
    values = [float(v) for v in rmsf_scores.values()]
    min_val = min(values) if values else 0.0
    max_val = max(values) if values else 1.0
    
    # Normalize to [0, 1]
    range_val = max_val - min_val if max_val > min_val else 1.0
    normalized = {
        str(k): round((float(v) - min_val) / range_val, 4)
        for k, v in rmsf_scores.items()
    }
    
    return {
        "min": round(min_val, 6),
        "max": round(max_val, 6),
        "normalized": normalized
    }


def create_tica_importance_json(residue_scores: dict) -> dict:
    """
    Create tICA importance data in ASVS format.
    
    Format:
    {
      "description": "tICA importance scores...",
      "min": 0.0,
      "max": 0.89,
      "normalized": {"0": 0.02, "1": 0.05, ...}
    }
    """
    tica_scores = residue_scores.get('tica_importance', {})
    
    if not tica_scores:
        return {
            "description": "tICA importance scores showing residue contribution to slow collective motions",
            "min": 0.0,
            "max": 1.0,
            "normalized": {}
        }
    
    values = [float(v) for v in tica_scores.values()]
    min_val = min(values) if values else 0.0
    max_val = max(values) if values else 1.0
    
    # Normalize to [0, 1]
    range_val = max_val - min_val if max_val > min_val else 1.0
    normalized = {
        str(k): round((float(v) - min_val) / range_val, 4)
        for k, v in tica_scores.items()
    }
    
    return {
        "description": "tICA importance scores showing residue contribution to slow collective motions",
        "min": round(min_val, 6),
        "max": round(max_val, 6),
        "normalized": normalized
    }


def get_n_residues(topology_path: str) -> int:
    """Get number of residues from topology file."""
    try:
        import mdtraj as md
        traj = md.load(topology_path)
        return traj.n_residues
    except ImportError:
        print("  Warning: mdtraj not installed, using default residue count")
        return DEFAULT_N_RESIDUES
    except FileNotFoundError:
        print(f"  Warning: Topology file not found: {topology_path}")
        return DEFAULT_N_RESIDUES
    except (ValueError, OSError) as e:
        print(f"  Warning: Could not load topology: {e}")
        return DEFAULT_N_RESIDUES


def main():
    parser = argparse.ArgumentParser(
        description='Export ML pipeline outputs for ASVS viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic export
    python tools/export_for_asvs.py \\
        --topology data/raw_trajectory/align_topol.pdb \\
        --trajectory data/raw_trajectory/trajectory_0.xtc \\
        --msm_dir outputs/msm \\
        --metrics_dir outputs/metrics \\
        --output_dir /path/to/asvs/viewer

Output Files:
    - hotspots_residue.json: Per-frame, per-residue hotspot scores
    - anomaly_residue.json: Per-frame, per-residue anomaly scores
    - rmsf_residue.json: Per-residue RMSF (flexibility) scores
    - tica_importance.json: Per-residue tICA importance scores
        """
    )
    
    parser.add_argument('--topology', required=True,
                       help='Path to topology file (PDB)')
    parser.add_argument('--trajectory', required=True,
                       help='Path to trajectory file (XTC)')
    parser.add_argument('--msm_dir', required=True,
                       help='Directory with MSM outputs')
    parser.add_argument('--metrics_dir', required=True,
                       help='Directory with metric outputs from compute_all_metrics.py')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory (typically /path/to/asvs/viewer)')
    
    args = parser.parse_args()
    
    # Validate inputs
    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    
    if not metrics_dir.exists():
        print(f"Error: Metrics directory not found: {metrics_dir}")
        print("Run 'python tools/compute_all_metrics.py' first")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EXPORT FOR ASVS VIEWER")
    print("=" * 60)
    print(f"Metrics dir: {metrics_dir}")
    print(f"Output dir:  {output_dir}")
    print()
    
    # Get number of residues
    n_residues = get_n_residues(args.topology)
    print(f"[1/5] Detected {n_residues} residues")
    
    # Load data
    print("[2/5] Loading frame scores...")
    frame_scores = load_frame_scores(metrics_dir)
    n_frames = len(frame_scores) if frame_scores is not None else 0
    print(f"  ✓ Loaded {n_frames} frames")
    
    print("[3/5] Loading residue scores...")
    residue_scores = load_residue_scores(metrics_dir)
    print(f"  ✓ Loaded: {list(residue_scores.keys())}")
    
    # Create output files
    print("[4/5] Creating ASVS-compatible JSON files...")
    
    # hotspots_residue.json
    hotspots_data = create_hotspots_residue_json(frame_scores, residue_scores, n_residues)
    with open(output_dir / 'hotspots_residue.json', 'w') as f:
        json.dump(hotspots_data, f, indent=2)
    print(f"  ✓ hotspots_residue.json ({len(hotspots_data)} frames)")
    
    # anomaly_residue.json
    anomaly_data = create_anomaly_residue_json(frame_scores, residue_scores, n_residues)
    with open(output_dir / 'anomaly_residue.json', 'w') as f:
        json.dump(anomaly_data, f, indent=2)
    print(f"  ✓ anomaly_residue.json")
    
    # rmsf_residue.json
    rmsf_data = create_rmsf_json(residue_scores)
    with open(output_dir / 'rmsf_residue.json', 'w') as f:
        json.dump(rmsf_data, f, indent=2)
    print(f"  ✓ rmsf_residue.json ({len(rmsf_data.get('normalized', {}))} residues)")
    
    # tica_importance.json
    tica_data = create_tica_importance_json(residue_scores)
    with open(output_dir / 'tica_importance.json', 'w') as f:
        json.dump(tica_data, f, indent=2)
    print(f"  ✓ tica_importance.json ({len(tica_data.get('normalized', {}))} residues)")
    
    print()
    print("[5/5] Export complete!")
    print()
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"1. Copy your trajectory files to the ASVS viewer directory:")
    print(f"   cp {args.topology} {output_dir}/topology.pdb")
    print(f"   cp {args.trajectory} {output_dir}/trajectory.xtc")
    print()
    print(f"2. Start the ASVS viewer:")
    print(f"   cd {output_dir.parent}")
    print(f"   python app.py")
    print()
    print(f"3. Open in browser: http://localhost:5000/viewer")
    

if __name__ == '__main__':
    main()
