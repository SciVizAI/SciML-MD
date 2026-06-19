#!/usr/bin/env python3
"""
Edge case tests for the pipeline.

Tests robust handling of:
- Very short trajectories (few frames)
- Low/zero variance features
- Disconnected MSM states
- Extreme outliers
- Missing or invalid data
"""
import sys
import traceback
import numpy as np
import warnings
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring.signals import (
    compute_dynamic_anomaly_scores,
    normalize_scores,
    _rank_normalize,
    _percentile_normalize,
)
from msm.select_lag_and_dim import compute_vamp2_score
from msm.bootstrap_msm import bootstrap_resample


class MockMSM:
    """Mock MSM for testing."""
    def __init__(self, n_states=5, pi=None, P=None):
        self.n_states = n_states
        if pi is None:
            pi_raw = np.random.exponential(1.0, n_states)
            self.stationary_distribution = pi_raw / pi_raw.sum()
        else:
            self.stationary_distribution = pi
        
        if P is None:
            P = np.random.exponential(0.5, (n_states, n_states))
            self.transition_matrix = P / P.sum(axis=1, keepdims=True)
        else:
            self.transition_matrix = P


# ============================================================================
# Test: Very Short Trajectories
# ============================================================================

def test_very_short_trajectory_10_frames():
    """Test with only 10 frames - should handle gracefully."""
    print("[TEST] Very short trajectory (10 frames)...")
    
    n_frames = 10
    n_states = 3
    
    np.random.seed(42)
    msm = MockMSM(n_states=n_states)
    dtraj = np.random.choice(n_states, size=n_frames)
    tica_coords = np.random.randn(n_frames, 2)
    
    # Should not crash
    try:
        signals = compute_dynamic_anomaly_scores(
            msm=msm,
            dtraj=dtraj,
            tica_coords=tica_coords,
            lag_msm=2,  # Very short lag for short trajectory
            k_neighbors=min(5, n_frames - 1),
            normalize=True
        )
        
        assert len(signals['rarity']) == n_frames
        assert len(signals['transition_surprise']) == n_frames
        assert len(signals['local_density']) == n_frames
        
        # Check no NaN or inf
        for name, signal in signals.items():
            assert np.all(np.isfinite(signal)), f"{name} has non-finite values"
        
        print(f"  ✓ Computed signals for {n_frames} frames successfully")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_very_short_trajectory_50_frames():
    """Test with 50 frames - minimal viable trajectory."""
    print("[TEST] Short trajectory (50 frames)...")
    
    n_frames = 50
    n_states = 5
    
    np.random.seed(42)
    msm = MockMSM(n_states=n_states)
    dtraj = np.random.choice(n_states, size=n_frames, 
                             p=msm.stationary_distribution)
    tica_coords = np.random.randn(n_frames, 3)
    
    signals = compute_dynamic_anomaly_scores(
        msm=msm,
        dtraj=dtraj,
        tica_coords=tica_coords,
        lag_msm=5,
        k_neighbors=10,
        normalize=True
    )
    
    assert len(signals['rarity']) == n_frames
    for name, signal in signals.items():
        assert np.all(np.isfinite(signal)), f"{name} has non-finite values"
        assert 0 <= signal.min() <= signal.max() <= 1
    
    print(f"  ✓ Computed signals for {n_frames} frames successfully")


def test_vamp2_short_trajectory():
    """Test VAMP-2 with insufficient data."""
    print("[TEST] VAMP-2 with short trajectory...")
    
    np.random.seed(42)
    X_short = np.random.randn(30, 5)  # Only 30 frames
    
    # Should return -inf or valid finite score
    score = compute_vamp2_score(X_short, lag=20, dim=3)
    
    assert score == -np.inf or np.isfinite(score), \
        f"VAMP-2 should handle short trajectory gracefully, got {score}"
    
    print(f"  ✓ VAMP-2 returns {score} for short trajectory (expected -inf or finite)")


# ============================================================================
# Test: Low/Zero Variance Features
# ============================================================================

def test_constant_features():
    """Test with constant (zero variance) features."""
    print("[TEST] Constant features...")
    
    # All same values
    constant = np.ones((100, 5)) * 42.0
    
    # Normalization should handle gracefully
    normalized = _rank_normalize(constant[:, 0])
    assert np.all(normalized == 0), "Constant array should normalize to zeros"
    
    normalized_2d = normalize_scores(constant, method='rank')
    assert np.all(normalized_2d == 0), "Constant 2D array should normalize to zeros"
    
    print("  ✓ Constant features handled correctly")


def test_near_zero_variance():
    """Test with very low variance features."""
    print("[TEST] Near-zero variance features...")
    
    np.random.seed(42)
    # Small perturbations around constant
    low_var = 100.0 + np.random.randn(100, 3) * 1e-10
    
    normalized = normalize_scores(low_var, method='rank')
    
    assert np.all(np.isfinite(normalized)), "Should handle low variance"
    assert 0 <= normalized.min() <= normalized.max() <= 1
    
    print("  ✓ Low variance features handled correctly")


def test_single_unique_value():
    """Test with all identical values."""
    print("[TEST] Single unique value...")
    
    single = np.array([5.0] * 100)
    
    result = _rank_normalize(single)
    assert np.all(result == 0), "Should return zeros for constant"
    
    result = _percentile_normalize(single)
    assert np.all(result == 0), "Percentile should return zeros for constant"
    
    print("  ✓ Single unique value handled correctly")


# ============================================================================
# Test: Disconnected MSM States
# ============================================================================

def test_disconnected_states():
    """Test MSM with disconnected/absorbing states."""
    print("[TEST] Disconnected MSM states...")
    
    n_frames = 100
    n_states = 5
    
    # State 4 is absorbing (never leaves)
    P = np.array([
        [0.8, 0.2, 0.0, 0.0, 0.0],
        [0.2, 0.7, 0.1, 0.0, 0.0],
        [0.0, 0.1, 0.8, 0.1, 0.0],
        [0.0, 0.0, 0.1, 0.9, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],  # Absorbing state
    ])
    pi = np.array([0.3, 0.3, 0.2, 0.15, 0.05])
    
    np.random.seed(42)
    msm = MockMSM(n_states=n_states, pi=pi, P=P)
    dtraj = np.random.choice(n_states, size=n_frames)
    tica_coords = np.random.randn(n_frames, 3)
    
    # Should not crash even with absorbing state
    signals = compute_dynamic_anomaly_scores(
        msm=msm,
        dtraj=dtraj,
        tica_coords=tica_coords,
        lag_msm=10,
        k_neighbors=20,
        normalize=True
    )
    
    for name, signal in signals.items():
        assert np.all(np.isfinite(signal)), f"{name} has non-finite values"
    
    print("  ✓ Disconnected states handled correctly")


def test_single_state_trajectory():
    """Test trajectory stuck in one state."""
    print("[TEST] Single-state trajectory...")
    
    n_frames = 100
    n_states = 5
    
    np.random.seed(42)
    msm = MockMSM(n_states=n_states)
    dtraj = np.zeros(n_frames, dtype=int)  # All in state 0
    tica_coords = np.random.randn(n_frames, 3)
    
    signals = compute_dynamic_anomaly_scores(
        msm=msm,
        dtraj=dtraj,
        tica_coords=tica_coords,
        lag_msm=10,
        k_neighbors=20,
        normalize=True
    )
    
    # Rarity should be constant (all same state)
    # After normalization, all zeros (constant input)
    assert np.all(np.isfinite(signals['rarity']))
    
    # Transition surprise: same state -> same state
    # Should be finite
    assert np.all(np.isfinite(signals['transition_surprise']))
    
    print("  ✓ Single-state trajectory handled correctly")


# ============================================================================
# Test: Extreme Outliers
# ============================================================================

def test_extreme_outlier_coordinates():
    """Test with extreme outlier in tICA coordinates."""
    print("[TEST] Extreme outlier coordinates...")
    
    n_frames = 100
    n_states = 5
    
    np.random.seed(42)
    msm = MockMSM(n_states=n_states)
    dtraj = np.random.choice(n_states, size=n_frames)
    
    # Normal coordinates with one extreme outlier
    tica_coords = np.random.randn(n_frames, 3)
    tica_coords[50] = [1e6, 1e6, 1e6]  # Extreme outlier
    
    signals = compute_dynamic_anomaly_scores(
        msm=msm,
        dtraj=dtraj,
        tica_coords=tica_coords,
        lag_msm=10,
        k_neighbors=20,
        normalize=True
    )
    
    # Local density should still be finite
    assert np.all(np.isfinite(signals['local_density']))
    
    # Outlier frame should have high density score (more isolated)
    # After rank normalization, should be close to 1
    outlier_density = signals['local_density'][50]
    assert outlier_density > 0.9, f"Outlier should have high density score, got {outlier_density}"
    
    print(f"  ✓ Outlier detected with density score {outlier_density:.3f}")


def test_extreme_probability_values():
    """Test with extreme probability values in MSM."""
    print("[TEST] Extreme probability values...")
    
    n_frames = 100
    n_states = 3
    
    # One state has very low probability
    pi = np.array([0.499, 0.5, 1e-10])
    P = np.array([
        [0.99, 0.01, 1e-12],
        [0.01, 0.99, 1e-12],
        [1e-12, 1e-12, 1.0 - 2e-12],
    ])
    
    np.random.seed(42)
    msm = MockMSM(n_states=n_states, pi=pi, P=P)
    dtraj = np.random.choice(n_states, size=n_frames, p=[0.45, 0.45, 0.1])
    tica_coords = np.random.randn(n_frames, 3)
    
    signals = compute_dynamic_anomaly_scores(
        msm=msm,
        dtraj=dtraj,
        tica_coords=tica_coords,
        lag_msm=10,
        k_neighbors=20,
        normalize=True
    )
    
    # Should handle very small probabilities without overflow
    assert np.all(np.isfinite(signals['rarity']))
    assert np.all(np.isfinite(signals['transition_surprise']))
    
    print("  ✓ Extreme probabilities handled correctly")


# ============================================================================
# Test: Empty and Edge Arrays
# ============================================================================

def test_empty_array_normalization():
    """Test normalization of empty arrays."""
    print("[TEST] Empty array normalization...")
    
    empty = np.array([])
    
    result = _rank_normalize(empty)
    assert len(result) == 0, "Empty input should give empty output"
    
    result = _percentile_normalize(empty)
    assert len(result) == 0, "Empty input should give empty output"
    
    print("  ✓ Empty arrays handled correctly")


def test_single_element_array():
    """Test with single-element arrays."""
    print("[TEST] Single element array...")
    
    single = np.array([42.0])
    
    result = _rank_normalize(single)
    assert result[0] == 0, "Single element should normalize to 0"
    
    print("  ✓ Single element arrays handled correctly")


def test_two_element_array():
    """Test with two-element arrays."""
    print("[TEST] Two element array...")
    
    two = np.array([1.0, 2.0])
    
    result = _rank_normalize(two)
    assert result[0] == 0 and result[1] == 1, "Two elements should be [0, 1]"
    
    print("  ✓ Two element arrays handled correctly")


# ============================================================================
# Test: Bootstrap Edge Cases
# ============================================================================

def test_bootstrap_preserves_length():
    """Test bootstrap resampling preserves trajectory length."""
    print("[TEST] Bootstrap preserves length...")
    
    X = np.random.randn(100, 5)
    
    X_boot = bootstrap_resample(X, method='frames', seed=42)
    assert X_boot.shape == X.shape
    
    X_boot = bootstrap_resample(X, method='blocks', block_size=10, seed=42)
    assert X_boot.shape == X.shape
    
    print("  ✓ Bootstrap preserves trajectory length")


def test_bootstrap_short_trajectory():
    """Test bootstrap with very short trajectory."""
    print("[TEST] Bootstrap short trajectory...")
    
    X_short = np.random.randn(20, 3)
    
    X_boot = bootstrap_resample(X_short, method='frames', seed=42)
    assert X_boot.shape == X_short.shape
    
    # Block bootstrap with blocks larger than trajectory
    X_boot = bootstrap_resample(X_short, method='blocks', block_size=5, seed=42)
    assert X_boot.shape == X_short.shape
    
    print("  ✓ Bootstrap handles short trajectories")


# ============================================================================
# Test: Numerical Stability
# ============================================================================

def test_log_zero_protection():
    """Test protection against log(0) in transition surprise."""
    print("[TEST] Log(0) protection...")
    
    n_frames = 100
    n_states = 3
    
    # Transition matrix with zeros
    P = np.array([
        [0.9, 0.1, 0.0],  # State 0 never goes to state 2
        [0.1, 0.8, 0.1],
        [0.0, 0.1, 0.9],  # State 2 never goes to state 0
    ])
    pi = np.array([0.33, 0.34, 0.33])
    
    np.random.seed(42)
    msm = MockMSM(n_states=n_states, pi=pi, P=P)
    
    # Create trajectory with "forbidden" transitions
    dtraj = np.zeros(n_frames, dtype=int)
    dtraj[::2] = 0
    dtraj[1::2] = 2  # Alternating 0, 2, 0, 2... (forbidden transition)
    
    tica_coords = np.random.randn(n_frames, 3)
    
    signals = compute_dynamic_anomaly_scores(
        msm=msm,
        dtraj=dtraj,
        tica_coords=tica_coords,
        lag_msm=1,  # Direct transitions
        k_neighbors=20,
        normalize=True
    )
    
    # Should not have inf values (epsilon protection)
    assert np.all(np.isfinite(signals['transition_surprise']))
    
    print("  ✓ Log(0) protected with epsilon")


def test_divide_by_zero_protection():
    """Test protection against division by zero in normalization."""
    print("[TEST] Division by zero protection...")
    
    # All same values
    constant = np.array([5.0] * 100)
    
    # Percentile normalization with constant array
    result = _percentile_normalize(constant)
    assert np.all(np.isfinite(result))
    assert np.all(result == 0)  # Should be zeros
    
    print("  ✓ Division by zero protected")


# ============================================================================
# Test: Negative and Special Values
# ============================================================================

def test_negative_coordinates():
    """Test with negative tICA coordinates."""
    print("[TEST] Negative coordinates...")
    
    n_frames = 100
    n_states = 5
    
    np.random.seed(42)
    msm = MockMSM(n_states=n_states)
    dtraj = np.random.choice(n_states, size=n_frames)
    
    # Mostly negative coordinates
    tica_coords = np.random.randn(n_frames, 3) - 100
    
    signals = compute_dynamic_anomaly_scores(
        msm=msm,
        dtraj=dtraj,
        tica_coords=tica_coords,
        lag_msm=10,
        k_neighbors=20,
        normalize=True
    )
    
    for name, signal in signals.items():
        assert np.all(np.isfinite(signal))
        assert 0 <= signal.min() <= signal.max() <= 1
    
    print("  ✓ Negative coordinates handled correctly")


def test_invalid_state_indices():
    """Test handling of out-of-range state indices."""
    print("[TEST] Invalid state indices...")
    
    n_frames = 100
    n_states = 5
    
    np.random.seed(42)
    msm = MockMSM(n_states=n_states)
    
    # Some indices outside valid range
    dtraj = np.random.choice(n_states, size=n_frames)
    dtraj[10] = -1  # Invalid: negative
    dtraj[20] = 100  # Invalid: too large
    
    tica_coords = np.random.randn(n_frames, 3)
    
    # Should handle gracefully (skip invalid indices)
    signals = compute_dynamic_anomaly_scores(
        msm=msm,
        dtraj=dtraj,
        tica_coords=tica_coords,
        lag_msm=10,
        k_neighbors=20,
        normalize=True
    )
    
    # Rarity for invalid indices should be default (1.0 before normalization)
    assert np.all(np.isfinite(signals['rarity']))
    
    print("  ✓ Invalid state indices handled correctly")


def test_large_trajectory_optimization():
    """Test auto-optimization for large trajectories."""
    print("[TEST] Large trajectory optimization...")
    
    n_frames = 100_000  # Simulate a large trajectory
    n_states = 10
    
    np.random.seed(42)
    
    # Create mock MSM
    pi = np.random.dirichlet(np.ones(n_states))
    P = np.random.dirichlet(np.ones(n_states), size=n_states)
    
    class MockMSM:
        def __init__(self):
            self.n_states = n_states
            self.stationary_distribution = pi
            self.transition_matrix = P
    
    msm = MockMSM()
    dtraj = np.random.choice(n_states, size=n_frames)
    tica_coords = np.random.randn(n_frames, 3)
    
    # With auto_optimize=True (default), should use subsampling
    import time
    start = time.time()
    signals = compute_dynamic_anomaly_scores(
        msm=msm,
        dtraj=dtraj,
        tica_coords=tica_coords,
        lag_msm=30,
        k_neighbors=20,
        normalize=True,
        auto_optimize=True
    )
    elapsed = time.time() - start
    
    assert len(signals['rarity']) == n_frames
    assert len(signals['local_density']) == n_frames
    assert np.all(np.isfinite(signals['rarity']))
    assert np.all(np.isfinite(signals['local_density']))
    
    print(f"  ✓ Processed {n_frames:,} frames in {elapsed:.2f}s")
    print(f"  ✓ All signals computed correctly")


def test_adaptive_parameter_scaling():
    """Test that adaptive parameters scale correctly with trajectory size."""
    print("[TEST] Adaptive parameter scaling...")
    
    from scoring.signals import _get_adaptive_parameters
    
    # Test different trajectory sizes
    test_cases = [
        (100, 20, 30),      # Small
        (1000, 20, 30),     # Medium
        (20000, 20, 30),    # Large
        (100000, 20, 30),   # Very large
        (500000, 20, 30),   # Extremely large
    ]
    
    for n_frames, k, lag in test_cases:
        adj_k, adj_lag, use_sub, sub_size = _get_adaptive_parameters(n_frames, k, lag)
        
        # Check k is reasonable
        assert adj_k >= 1, f"k should be at least 1 for {n_frames} frames"
        assert adj_k <= n_frames - 1, f"k should be less than n_frames for {n_frames}"
        
        # Check lag is reasonable
        assert adj_lag >= 1, f"lag should be at least 1 for {n_frames} frames"
        assert adj_lag < n_frames, f"lag should be less than n_frames for {n_frames}"
        
        # Check subsampling is enabled for very large trajectories
        if n_frames > 200_000:
            assert use_sub, f"Subsampling should be enabled for {n_frames} frames"
            assert sub_size < n_frames, "Subsample size should be less than n_frames"
    
    print("  ✓ Adaptive parameters scale correctly")


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all edge case tests."""
    print("="*70)
    print("TESTING PIPELINE EDGE CASES")
    print("="*70 + "\n")
    
    tests = [
        # Short trajectories
        test_very_short_trajectory_10_frames,
        test_very_short_trajectory_50_frames,
        test_vamp2_short_trajectory,
        
        # Low/zero variance
        test_constant_features,
        test_near_zero_variance,
        test_single_unique_value,
        
        # Disconnected states
        test_disconnected_states,
        test_single_state_trajectory,
        
        # Extreme outliers
        test_extreme_outlier_coordinates,
        test_extreme_probability_values,
        
        # Empty and edge arrays
        test_empty_array_normalization,
        test_single_element_array,
        test_two_element_array,
        
        # Bootstrap
        test_bootstrap_preserves_length,
        test_bootstrap_short_trajectory,
        
        # Numerical stability
        test_log_zero_protection,
        test_divide_by_zero_protection,
        
        # Special values
        test_negative_coordinates,
        test_invalid_state_indices,
        
        # Scalability
        test_adaptive_parameter_scaling,
        test_large_trajectory_optimization,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
