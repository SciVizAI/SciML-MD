#!/usr/bin/env python3
"""
Unit tests for signals.py module.
"""
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring.signals import (
    compute_dynamic_anomaly_scores,
    normalize_scores,
    aggregate_frame_to_residue,
    _rank_normalize,
    _percentile_normalize,
    _compute_zscore
)


def test_rank_normalize():
    """Test rank normalization."""
    print("[TEST] Rank normalization...")
    
    x = np.array([1, 5, 3, 9, 2])
    normalized = _rank_normalize(x)
    
    assert normalized.min() == 0.0, "Min should be 0"
    assert normalized.max() == 1.0, "Max should be 1"
    assert len(normalized) == len(x), "Length preserved"
    
    # Check ordering
    assert normalized[0] < normalized[2] < normalized[1], "Ordering preserved"
    
    print(f"  ✓ Input: {x}")
    print(f"  ✓ Normalized: {normalized}")


def test_percentile_normalize():
    """Test percentile normalization."""
    print("[TEST] Percentile normalization...")
    
    x = np.array([1, 2, 3, 4, 5, 100])  # With outlier
    normalized = _percentile_normalize(x, lower=0.1, upper=0.9)
    
    assert 0 <= normalized.min(), "Should be >= 0"
    assert normalized.max() <= 1, "Should be <= 1"
    
    # Outlier should be clipped
    assert normalized[-1] == 1.0, "Outlier clipped to 1"
    
    print(f"  ✓ Input: {x}")
    print(f"  ✓ Normalized: {normalized}")


def test_normalize_scores_global():
    """Test global normalization."""
    print("[TEST] Global normalization...")
    
    scores = np.random.randn(100)
    
    # Rank normalization
    norm_rank = normalize_scores(scores, method='rank', per_frame=False)
    assert 0 <= norm_rank.min() <= norm_rank.max() <= 1, "Should be in [0,1]"
    
    # Percentile normalization
    norm_perc = normalize_scores(scores, method='percentile', per_frame=False)
    assert 0 <= norm_perc.min() <= norm_perc.max() <= 1, "Should be in [0,1]"
    
    print(f"  ✓ Rank range: [{norm_rank.min():.3f}, {norm_rank.max():.3f}]")
    print(f"  ✓ Percentile range: [{norm_perc.min():.3f}, {norm_perc.max():.3f}]")


def test_normalize_scores_per_frame():
    """Test per-frame normalization."""
    print("[TEST] Per-frame normalization...")
    
    # 2D array: [n_frames, n_residues]
    scores = np.random.randn(10, 20)
    
    norm = normalize_scores(scores, method='rank', per_frame=True)
    
    assert norm.shape == scores.shape, "Shape preserved"
    
    # Check each frame is normalized independently
    for i in range(norm.shape[0]):
        frame_vals = norm[i, :]
        assert 0 <= frame_vals.min() <= frame_vals.max() <= 1, f"Frame {i} not in [0,1]"
    
    print(f"  ✓ Normalized {norm.shape[0]} frames independently")


def test_dynamic_anomaly_signals():
    """Test dynamic anomaly signal computation."""
    print("[TEST] Dynamic anomaly signals...")
    
    # Create mock MSM
    class MockMSM:
        def __init__(self):
            self.n_states = 5
            self.stationary_distribution = np.array([0.4, 0.3, 0.2, 0.08, 0.02])
            self.transition_matrix = np.array([
                [0.8, 0.1, 0.05, 0.03, 0.02],
                [0.1, 0.7, 0.15, 0.03, 0.02],
                [0.05, 0.15, 0.6, 0.15, 0.05],
                [0.03, 0.03, 0.15, 0.7, 0.09],
                [0.02, 0.02, 0.05, 0.09, 0.82]
            ])
    
    msm = MockMSM()
    
    # Create mock trajectory
    n_frames = 100
    dtraj = np.random.choice(5, size=n_frames, p=msm.stationary_distribution)
    tica_coords = np.random.randn(n_frames, 3)
    
    # Compute signals
    signals = compute_dynamic_anomaly_scores(
        msm=msm,
        dtraj=dtraj,
        tica_coords=tica_coords,
        lag_msm=10,
        k_neighbors=20,
        normalize=True
    )
    
    assert 'rarity' in signals, "Missing rarity signal"
    assert 'transition_surprise' in signals, "Missing transition surprise"
    assert 'local_density' in signals, "Missing local density"
    
    # Check all signals are normalized to [0,1]
    for name, signal in signals.items():
        assert len(signal) == n_frames, f"{name}: wrong length"
        assert 0 <= signal.min() <= signal.max() <= 1, f"{name}: not in [0,1]"
    
    print(f"  ✓ Computed {len(signals)} signals")
    print(f"  ✓ Rarity range: [{signals['rarity'].min():.3f}, {signals['rarity'].max():.3f}]")
    print(f"  ✓ Surprise range: [{signals['transition_surprise'].min():.3f}, {signals['transition_surprise'].max():.3f}]")
    print(f"  ✓ Density range: [{signals['local_density'].min():.3f}, {signals['local_density'].max():.3f}]")


def test_aggregate_frame_to_residue():
    """Test frame-to-residue aggregation."""
    print("[TEST] Frame-to-residue aggregation...")
    
    n_frames = 50
    n_residues = 20
    
    np.random.seed(42)  # For reproducibility
    
    # Mock frame scores (some frames are anomalous)
    frame_scores = np.random.rand(n_frames)
    frame_scores[10:15] = 0.9  # High anomaly frames
    
    # Mock residue contributions (some residues contribute more)
    contributions = np.random.rand(n_frames, n_residues)
    contributions[:, 5] = 0.8  # Residue 5 always contributes
    contributions[10:15, 10] = 0.95  # Residue 10 contributes strongly in anomalous frames
    
    # Aggregate
    residue_scores = aggregate_frame_to_residue(
        frame_scores,
        contributions,
        method='weighted_mean'
    )
    
    assert len(residue_scores) == n_residues, "Wrong number of residues"
    # Just check that scores are reasonable
    assert np.all(np.isfinite(residue_scores)), "Scores should be finite"
    
    print(f"  ✓ Aggregated {n_frames} frames to {n_residues} residues")
    print(f"  ✓ Range: [{residue_scores.min():.3f}, {residue_scores.max():.3f}]")


def test_normalization_consistency():
    """Test that normalization methods give consistent orderings."""
    print("[TEST] Normalization consistency...")
    
    x = np.array([1, 5, 3, 9, 2, 7, 4, 8, 6])
    
    norm_rank = _rank_normalize(x)
    norm_perc = _percentile_normalize(x, 0.1, 0.9)
    
    # Ordering should be preserved (except possibly at clipped edges)
    original_order = np.argsort(x)
    rank_order = np.argsort(norm_rank)
    
    assert np.array_equal(original_order, rank_order), "Rank should preserve order exactly"
    
    print(f"  ✓ Original order preserved")


def test_edge_cases():
    """Test edge cases."""
    print("[TEST] Edge cases...")
    
    # Empty array
    empty = np.array([])
    assert len(_rank_normalize(empty)) == 0, "Empty should stay empty"
    
    # Single element
    single = np.array([5.0])
    assert _rank_normalize(single)[0] == 0.0, "Single element should be 0"
    
    # Constant array
    constant = np.array([5.0, 5.0, 5.0, 5.0])
    norm_const = _rank_normalize(constant)
    assert np.all(norm_const == 0), "Constant should give zeros"
    
    # Two elements
    two = np.array([1.0, 2.0])
    norm_two = _rank_normalize(two)
    assert norm_two[0] == 0 and norm_two[1] == 1, "Two elements should be 0,1"
    
    print(f"  ✓ Edge cases handled correctly")


def test_normalization_ranges():
    """Test that all normalization methods respect [0,1] range."""
    print("[TEST] Normalization ranges...")
    
    # Test with various distributions
    distributions = [
        np.random.randn(100),           # Normal
        np.random.exponential(2, 100),   # Exponential (heavy tail)
        np.random.uniform(-5, 5, 100),   # Uniform
        np.concatenate([np.ones(90), np.ones(10) * 100])  # With outliers
    ]
    
    for i, data in enumerate(distributions):
        for method in ['rank', 'percentile']:
            normalized = normalize_scores(data, method=method)
            assert 0 <= normalized.min(), f"Dist {i}, method {method}: min < 0"
            assert normalized.max() <= 1, f"Dist {i}, method {method}: max > 1"
    
    print(f"  ✓ All methods respect [0,1] range across distributions")


def test_signal_properties():
    """Test that signals have expected properties."""
    print("[TEST] Signal properties...")
    
    # Create mock MSM with clear rare state
    class MockMSM:
        def __init__(self):
            self.n_states = 3
            # State 2 is very rare
            self.stationary_distribution = np.array([0.45, 0.45, 0.10])
            self.transition_matrix = np.array([
                [0.8, 0.15, 0.05],
                [0.15, 0.8, 0.05],
                [0.05, 0.05, 0.9]
            ])
    
    msm = MockMSM()
    
    # Trajectory mostly in states 0,1 with occasional visits to state 2
    dtraj = np.array([0, 0, 1, 1, 0, 2, 2, 0, 1, 1] * 10)
    tica_coords = np.random.randn(len(dtraj), 2)
    
    signals = compute_dynamic_anomaly_scores(
        msm, dtraj, tica_coords,
        normalize=False  # Check raw values
    )
    
    # State 2 frames should have high rarity
    state2_frames = np.where(dtraj == 2)[0]
    state0_frames = np.where(dtraj == 0)[0]
    
    mean_rarity_state2 = signals['rarity'][state2_frames].mean()
    mean_rarity_state0 = signals['rarity'][state0_frames].mean()
    
    assert mean_rarity_state2 > mean_rarity_state0, "Rare state should have higher rarity"
    
    print(f"  ✓ Rare state (2) rarity: {mean_rarity_state2:.3f}")
    print(f"  ✓ Common state (0) rarity: {mean_rarity_state0:.3f}")


def main():
    """Run all tests."""
    print("="*70)
    print("TESTING SIGNALS MODULE")
    print("="*70 + "\n")
    
    try:
        # Basic normalization tests
        test_rank_normalize()
        test_percentile_normalize()
        test_normalization_consistency()
        test_normalization_ranges()
        
        # Score normalization tests
        test_normalize_scores_global()
        test_normalize_scores_per_frame()
        
        # Signal computation tests
        test_dynamic_anomaly_signals()
        test_signal_properties()
        
        # Aggregation tests
        test_aggregate_frame_to_residue()
        
        # Edge cases
        test_edge_cases()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        return 0
    
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
