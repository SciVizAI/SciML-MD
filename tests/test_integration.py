#!/usr/bin/env python3
"""
Integration test for the new metrics computation pipeline.

This demonstrates end-to-end functionality without requiring full MD simulation.
Creates synthetic data to test all components.
"""
import numpy as np
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring.signals import (
    compute_dynamic_anomaly_scores,
    normalize_scores,
    _rank_normalize
)


def create_synthetic_msm(n_states=5, seed=42):
    """Create a synthetic MSM for testing."""
    np.random.seed(seed)
    
    class SyntheticMSM:
        def __init__(self, n_states):
            self.n_states = n_states
            
            # Create stationary distribution (some states are rarer)
            pi_raw = np.random.exponential(1.0, n_states)
            self.stationary_distribution = pi_raw / pi_raw.sum()
            
            # Create transition matrix (reversible)
            P = np.random.exponential(0.5, (n_states, n_states))
            # Make it reversible: π_i P_ij = π_j P_ji
            for i in range(n_states):
                for j in range(i+1, n_states):
                    # Symmetrize
                    avg = (P[i,j] + P[j,i]) / 2
                    P[i,j] = avg
                    P[j,i] = avg
            
            # Normalize rows
            self.transition_matrix = P / P.sum(axis=1, keepdims=True)
        
        def generate_trajectory(self, n_frames):
            """Generate a synthetic trajectory."""
            dtraj = np.zeros(n_frames, dtype=int)
            dtraj[0] = np.random.choice(self.n_states, p=self.stationary_distribution)
            
            for t in range(1, n_frames):
                current_state = dtraj[t-1]
                dtraj[t] = np.random.choice(
                    self.n_states, 
                    p=self.transition_matrix[current_state]
                )
            
            return dtraj
    
    return SyntheticMSM(n_states)


def create_synthetic_tica_coords(n_frames, n_dims=3, seed=42):
    """Create synthetic tICA coordinates."""
    np.random.seed(seed)
    
    # Create coordinates with some structure
    # Most points cluster, some are outliers
    coords = np.random.randn(n_frames, n_dims) * 0.5
    
    # Add some outlier frames
    n_outliers = max(1, n_frames // 20)
    outlier_indices = np.random.choice(n_frames, n_outliers, replace=False)
    coords[outlier_indices] += np.random.randn(n_outliers, n_dims) * 3.0
    
    return coords


def test_integration():
    """Test full integration of metrics computation."""
    print("="*70)
    print("INTEGRATION TEST: Metrics Computation Pipeline")
    print("="*70)
    
    # Setup
    n_frames = 500
    n_residues = 50
    n_states = 10
    
    print(f"\nSetup:")
    print(f"  Frames: {n_frames}")
    print(f"  Residues: {n_residues}")
    print(f"  States: {n_states}")
    
    # ========================================================================
    # 1. Create synthetic MSM and trajectory
    # ========================================================================
    print("\n[1/5] Creating synthetic MSM...")
    msm = create_synthetic_msm(n_states=n_states, seed=42)
    dtraj = msm.generate_trajectory(n_frames)
    
    print(f"  ✓ MSM with {msm.n_states} states")
    print(f"  ✓ Stationary distribution: min={msm.stationary_distribution.min():.3f}, "
          f"max={msm.stationary_distribution.max():.3f}")
    
    # ========================================================================
    # 2. Create synthetic tICA coordinates
    # ========================================================================
    print("\n[2/5] Creating synthetic tICA coordinates...")
    tica_coords = create_synthetic_tica_coords(n_frames, n_dims=3, seed=42)
    
    print(f"  ✓ Coordinates shape: {tica_coords.shape}")
    print(f"  ✓ Mean: {tica_coords.mean(axis=0)}")
    print(f"  ✓ Std: {tica_coords.std(axis=0)}")
    
    # ========================================================================
    # 3. Compute dynamic anomaly signals
    # ========================================================================
    print("\n[3/5] Computing dynamic anomaly signals...")
    signals = compute_dynamic_anomaly_scores(
        msm=msm,
        dtraj=dtraj,
        tica_coords=tica_coords,
        lag_msm=10,
        k_neighbors=20,
        normalize=True
    )
    
    for signal_name, signal_values in signals.items():
        print(f"  ✓ {signal_name}: "
              f"range=[{signal_values.min():.3f}, {signal_values.max():.3f}], "
              f"mean={signal_values.mean():.3f}")
    
    # ========================================================================
    # 4. Test normalization strategies
    # ========================================================================
    print("\n[4/5] Testing normalization strategies...")
    
    # Create test scores
    test_scores = np.random.exponential(1.0, 100)
    
    # Rank normalization
    norm_rank = normalize_scores(test_scores, method='rank')
    print(f"  ✓ Rank: range=[{norm_rank.min():.3f}, {norm_rank.max():.3f}]")
    
    # Percentile normalization
    norm_perc = normalize_scores(
        test_scores, 
        method='percentile',
        low_percentile=0.05,
        high_percentile=0.95
    )
    print(f"  ✓ Percentile: range=[{norm_perc.min():.3f}, {norm_perc.max():.3f}]")
    
    # Global vs per-frame
    test_2d = np.random.exponential(1.0, (10, 20))
    norm_global = normalize_scores(test_2d, method='rank', per_frame=False)
    norm_per_frame = normalize_scores(test_2d, method='rank', per_frame=True)
    
    print(f"  ✓ Global 2D: shape={norm_global.shape}")
    print(f"  ✓ Per-frame 2D: shape={norm_per_frame.shape}")
    
    # ========================================================================
    # 5. Create synthetic RMSF and tICA importance
    # ========================================================================
    print("\n[5/5] Creating synthetic residue metrics...")
    
    # Synthetic RMSF (some residues more flexible)
    rmsf_raw = np.random.exponential(2.0, n_residues)
    rmsf_norm = normalize_scores(rmsf_raw, method='percentile')
    rmsf_scores = {i: float(rmsf_norm[i]) for i in range(n_residues)}
    
    print(f"  ✓ RMSF: {len(rmsf_scores)} residues")
    
    # Synthetic tICA importance (some residues drive slow modes)
    importance_raw = np.random.exponential(1.5, n_residues)
    importance_norm = normalize_scores(importance_raw, method='percentile')
    importance_scores = {i: float(importance_norm[i]) for i in range(n_residues)}
    
    print(f"  ✓ tICA importance: {len(importance_scores)} residues")
    
    # ========================================================================
    # 6. Create unified output
    # ========================================================================
    print("\n[6/6] Creating unified output...")
    
    # Aggregate dynamic anomaly to residues (simplified)
    from scoring.anomaly_v2 import fuse_signals
    frame_scores_raw, _ = fuse_signals(signals, method='median', normalize_method='rank')
    frame_scores = frame_scores_raw
    
    # Simple aggregation: use top 10% frames
    threshold = np.percentile(frame_scores, 90)
    high_frames = frame_scores >= threshold
    
    dynamic_residue_scores = {}
    for res_id in range(n_residues):
        # Weight by RMSF (proxy for contribution)
        contribution = rmsf_scores.get(res_id, 0.5)
        mean_anomaly = frame_scores[high_frames].mean() if high_frames.any() else 0.5
        dynamic_residue_scores[res_id] = float(contribution * mean_anomaly)
    
    # Normalize
    max_val = max(dynamic_residue_scores.values())
    if max_val > 0:
        dynamic_residue_scores = {k: v/max_val for k, v in dynamic_residue_scores.items()}
    
    # Create unified output
    unified_output = {
        "meta": {
            "n_frames": n_frames,
            "n_residues": n_residues,
            "metrics": ["dynamic_anomaly", "rmsf", "tica_importance"],
            "normalization": "percentile",
            "percentile_range": [0.05, 0.95],
            "description": {
                "dynamic_anomaly": "Involvement in rare/unexpected dynamics",
                "rmsf": "Root Mean Square Fluctuation - flexibility metric",
                "tica_importance": "Contribution to slow collective motions"
            }
        },
        "per_residue": {
            "dynamic_anomaly": {str(k): v for k, v in dynamic_residue_scores.items()},
            "rmsf": {str(k): v for k, v in rmsf_scores.items()},
            "tica_importance": {str(k): v for k, v in importance_scores.items()}
        }
    }
    
    print(f"  ✓ Unified output with {len(unified_output['meta']['metrics'])} metrics")
    
    # ========================================================================
    # 7. Validate output format
    # ========================================================================
    print("\n[7/7] Validating output format...")
    
    assert "meta" in unified_output, "Missing meta section"
    assert "per_residue" in unified_output, "Missing per_residue section"
    assert len(unified_output["per_residue"]) == 3, "Should have 3 metric types"
    
    for metric in unified_output["meta"]["metrics"]:
        assert metric in unified_output["per_residue"], f"Missing metric: {metric}"
        scores = unified_output["per_residue"][metric]
        assert len(scores) == n_residues, f"{metric}: wrong number of residues"
        
        # Check all scores are in [0, 1]
        for res_id, score in scores.items():
            assert 0 <= score <= 1, f"{metric} res {res_id}: score {score} not in [0,1]"
    
    print(f"  ✓ Output format valid")
    print(f"  ✓ All scores in [0, 1] range")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Dynamic anomaly statistics
    print("\nDynamic Anomaly (per frame):")
    print(f"  Range: [{frame_scores.min():.3f}, {frame_scores.max():.3f}]")
    print(f"  Mean: {frame_scores.mean():.3f} ± {frame_scores.std():.3f}")
    print(f"  High anomaly frames (>90th percentile): {high_frames.sum()}")
    
    # Per-residue statistics
    print("\nPer-Residue Scores:")
    for metric in ["dynamic_anomaly", "rmsf", "tica_importance"]:
        scores = list(unified_output["per_residue"][metric].values())
        print(f"  {metric:20s}: range=[{min(scores):.3f}, {max(scores):.3f}], "
              f"mean={np.mean(scores):.3f}")
    
    # Top hotspots
    print("\nTop 5 Dynamic Hotspots:")
    dynamic_vals = unified_output["per_residue"]["dynamic_anomaly"]
    top_hotspots = sorted(dynamic_vals.items(), key=lambda x: x[1], reverse=True)[:5]
    for res_id, score in top_hotspots:
        rmsf = unified_output["per_residue"]["rmsf"][res_id]
        importance = unified_output["per_residue"]["tica_importance"][res_id]
        print(f"  Res {res_id:3s}: dynamic={score:.3f}, rmsf={rmsf:.3f}, "
              f"tica_importance={importance:.3f}")
    
    print("\n" + "="*70)
    print("✓ INTEGRATION TEST PASSED")
    print("="*70)
    
    return True


def test_tica_model_persistence():
    """
    Regression test: run_msm_tica must save tica_model.npz so that
    compute_all_metrics.py can load real tICA importance scores instead
    of falling back to uniform (0.5) scores.
    """
    import io
    import contextlib
    import tempfile
    import warnings
    from pathlib import Path

    # Add tools to path for imports
    tools_dir = Path(__file__).parent.parent / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))

    np.random.seed(0)
    X = np.random.randn(150, 7)  # small synthetic trajectory

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        feat_path = tmp / "features.npy"
        np.save(feat_path, X)
        msm_dir = tmp / "msm"

        # Run the MSM+TICA pipeline
        from run_msm_tica import main as run_msm
        with contextlib.redirect_stdout(io.StringIO()):
            run_msm(str(feat_path), str(msm_dir),
                    lag_tica=5, lag_msm=5, n_clusters=10, use_cache=False)

        # 1. tica_model.npz must exist
        tica_npz = msm_dir / "tica_model.npz"
        assert tica_npz.exists(), (
            "tica_model.npz was not saved by run_msm_tica.py. "
            "compute_all_metrics.py will fall back to uniform importance scores."
        )

        # 2. The file must contain the 'eigenvectors' key with valid shape
        data = np.load(tica_npz)
        assert "eigenvectors" in data.files, \
            "tica_model.npz missing 'eigenvectors' key"
        eigvecs = data["eigenvectors"]
        assert eigvecs.ndim == 2, \
            f"eigenvectors should be 2-D, got shape {eigvecs.shape}"
        n_features, n_components = eigvecs.shape
        assert n_features == X.shape[1], \
            f"eigenvectors n_features ({n_features}) != feature dim ({X.shape[1]})"

        # 3. load_msm_and_tica must return a non-None tICA model
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        from tools.compute_all_metrics import load_msm_and_tica
        _, _, _, tica_model, _, _ = load_msm_and_tica(msm_dir)
        assert tica_model is not None, (
            "load_msm_and_tica returned tica_model=None even though "
            "tica_model.npz was saved correctly."
        )

        # 4. compute_tica_importance_scores must return non-uniform scores
        from scoring.signals import compute_tica_importance_scores
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            importance = compute_tica_importance_scores(
                tica_model=tica_model,
                feature_names=None,
                aggregate_by_residue=True
            )

        assert len(importance) > 0, "No residue importance scores returned"
        vals = list(importance.values())
        assert not all(v == 0.5 for v in vals), (
            "All importance scores are 0.5 (uniform fallback). "
            "tICA eigenvectors are not being used."
        )

    print("  ✓ test_tica_model_persistence passed")


def main():
    """Run integration test."""
    try:
        success = test_integration()
        return 0 if success else 1
    except Exception as e:
        print(f"\n✗ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
