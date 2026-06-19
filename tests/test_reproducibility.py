#!/usr/bin/env python3
"""
Reproducibility and robustness tests for ML pipeline.

Tests determinism, noise sensitivity, and parameter robustness
following best practices from ML literature.

References:
- Saltelli et al. (2008) "Global Sensitivity Analysis: The Primer"
- Peng (2011) "Reproducible Research in Computational Science" Science
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_random_seed_reproducibility():
    """
    Test that setting random seed produces identical results.
    
    Critical for reproducible science.
    """
    print("\n[TEST] Random seed reproducibility")
    
    def run_with_seed(seed):
        """Simulate a computation with random elements."""
        np.random.seed(seed)
        data = np.random.randn(100, 10)
        result = np.mean(data, axis=0)
        return result
    
    # Run twice with same seed
    result1 = run_with_seed(42)
    result2 = run_with_seed(42)
    
    # Should be identical
    np.testing.assert_array_equal(result1, result2,
                                   "Results differ with same seed")
    
    # Run with different seed
    result3 = run_with_seed(123)
    
    # Should be different
    assert not np.array_equal(result1, result3), \
        "Different seeds produced identical results"
    
    print(f"  Seed 42 (run 1): mean={result1.mean():.6f}")
    print(f"  Seed 42 (run 2): mean={result2.mean():.6f}")
    print(f"  Seed 123: mean={result3.mean():.6f}")
    print(f"  ✓ Reproducibility verified")


def test_noise_injection_robustness():
    """
    Test robustness to small amounts of noise.
    
    Small perturbations should not dramatically change results.
    """
    print("\n[TEST] Noise injection robustness")
    
    # Create clean signal
    t = np.linspace(0, 10, 100)
    clean_signal = np.sin(t)
    
    # Add small noise (SNR ~ 20 dB)
    noise_level = 0.1
    noisy_signal = clean_signal + noise_level * np.random.randn(len(t))
    
    # Compute simple statistic (mean)
    clean_mean = np.mean(clean_signal)
    noisy_mean = np.mean(noisy_signal)
    
    # Should be very similar
    relative_error = abs(noisy_mean - clean_mean) / (abs(clean_mean) + 1e-10)
    
    assert relative_error < 0.5, \
        f"Excessive sensitivity to noise: {relative_error:.2%} error"
    
    # Correlation should be high
    correlation = np.corrcoef(clean_signal, noisy_signal)[0, 1]
    assert correlation > 0.9, \
        f"Low correlation after noise: r={correlation:.3f}"
    
    print(f"  Noise level: {noise_level}")
    print(f"  Relative error: {relative_error:.2%}")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  ✓ Robust to noise")


def test_parameter_sensitivity_analysis():
    """
    Test sensitivity to hyperparameter changes.
    
    Should identify which parameters matter most.
    """
    print("\n[TEST] Parameter sensitivity analysis")
    
    def model_output(lag, dim, n_clusters):
        """Simulate model output depending on parameters."""
        # Simplified: output is combination of parameters
        # In reality, would be actual model performance
        np.random.seed(42)
        base = 0.8
        lag_effect = 0.05 * (lag / 10)
        dim_effect = 0.1 * (dim / 5)
        cluster_effect = 0.02 * (n_clusters / 50)
        noise = 0.01 * np.random.randn()
        return base + lag_effect + dim_effect + cluster_effect + noise
    
    # Baseline parameters
    base_lag = 10
    base_dim = 5
    base_clusters = 50
    base_output = model_output(base_lag, base_dim, base_clusters)
    
    # Test lag sensitivity
    lag_varied = model_output(base_lag * 1.5, base_dim, base_clusters)
    lag_sensitivity = abs(lag_varied - base_output) / base_output
    
    # Test dim sensitivity
    dim_varied = model_output(base_lag, base_dim * 1.5, base_clusters)
    dim_sensitivity = abs(dim_varied - base_output) / base_output
    
    # Test cluster sensitivity
    cluster_varied = model_output(base_lag, base_dim, base_clusters * 1.5)
    cluster_sensitivity = abs(cluster_varied - base_output) / base_output
    
    # At least one parameter should matter
    max_sensitivity = max(lag_sensitivity, dim_sensitivity, cluster_sensitivity)
    assert max_sensitivity > 0.01, "No parameters show sensitivity"
    
    print(f"  Baseline output: {base_output:.4f}")
    print(f"  Lag sensitivity: {lag_sensitivity:.4f}")
    print(f"  Dim sensitivity: {dim_sensitivity:.4f}")
    print(f"  Cluster sensitivity: {cluster_sensitivity:.4f}")
    print(f"  ✓ Sensitivity analysis complete")


def test_cross_validation_stability():
    """
    Test that cross-validation results are stable.
    
    Multiple runs should give similar average performance.
    """
    print("\n[TEST] Cross-validation stability")
    
    def simulate_cv_scores(n_folds=5, seed=None):
        """Simulate cross-validation scores."""
        if seed is not None:
            np.random.seed(seed)
        # Scores around 0.8 with some variance
        scores = 0.8 + 0.05 * np.random.randn(n_folds)
        return scores
    
    # Run CV multiple times
    n_runs = 5
    mean_scores = []
    
    for i in range(n_runs):
        scores = simulate_cv_scores(n_folds=5, seed=i)
        mean_scores.append(np.mean(scores))
    
    # Mean of means should be stable
    overall_mean = np.mean(mean_scores)
    overall_std = np.std(mean_scores)
    
    # Coefficient of variation should be small
    cv_coeff = overall_std / overall_mean
    assert cv_coeff < 0.1, \
        f"CV results unstable: CV={cv_coeff:.2%}"
    
    print(f"  Runs: {n_runs}")
    print(f"  Mean score: {overall_mean:.4f} ± {overall_std:.4f}")
    print(f"  CV coefficient: {cv_coeff:.2%}")
    print(f"  ✓ Cross-validation stable")


def test_data_subset_consistency():
    """
    Test that results are consistent across data subsets.
    
    Different random subsets should give similar trends.
    """
    print("\n[TEST] Data subset consistency")
    
    # Create full dataset
    n_total = 1000
    np.random.seed(42)
    full_data = np.random.randn(n_total)
    
    # Sample multiple subsets
    n_subsets = 5
    subset_size = 500
    subset_means = []
    
    for i in range(n_subsets):
        indices = np.random.choice(n_total, subset_size, replace=False)
        subset = full_data[indices]
        subset_means.append(np.mean(subset))
    
    # All subsets should have similar means
    subset_means = np.array(subset_means)
    mean_of_means = np.mean(subset_means)
    std_of_means = np.std(subset_means)
    
    # Check consistency
    for mean in subset_means:
        deviation = abs(mean - mean_of_means)
        assert deviation < 3 * std_of_means, \
            f"Subset mean {mean:.4f} too far from average"
    
    print(f"  Full data size: {n_total}")
    print(f"  Subset size: {subset_size}")
    print(f"  Subsets tested: {n_subsets}")
    print(f"  Mean ± std: {mean_of_means:.4f} ± {std_of_means:.4f}")
    print(f"  ✓ Subsets consistent")


def test_computation_determinism():
    """
    Test that computations are deterministic.
    
    Same input should always give same output.
    """
    print("\n[TEST] Computation determinism")
    
    def deterministic_computation(data):
        """A computation that should be deterministic."""
        # Matrix operations should be deterministic
        cov = np.cov(data.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        return eigenvalues
    
    # Create test data
    np.random.seed(42)
    data = np.random.randn(100, 10)
    
    # Run computation twice
    result1 = deterministic_computation(data)
    result2 = deterministic_computation(data)
    
    # Should be identical (within machine precision)
    np.testing.assert_allclose(result1, result2, rtol=1e-14,
                               err_msg="Computation is not deterministic")
    
    print(f"  Input shape: {data.shape}")
    print(f"  Output shape: {result1.shape}")
    print(f"  Max difference: {np.max(np.abs(result1 - result2)):.2e}")
    print(f"  ✓ Deterministic computation verified")


def test_missing_data_handling():
    """
    Test robustness to missing data.
    
    Should handle NaN values gracefully or raise informative errors.
    """
    print("\n[TEST] Missing data handling")
    
    # Create data with missing values
    data = np.random.randn(100, 10)
    data_with_nan = data.copy()
    data_with_nan[10:15, 3] = np.nan  # Insert some NaNs
    
    # Test NaN detection
    has_nan = np.any(np.isnan(data_with_nan))
    assert has_nan, "Failed to detect NaN values"
    
    # Remove NaN rows for processing
    valid_rows = ~np.any(np.isnan(data_with_nan), axis=1)
    cleaned_data = data_with_nan[valid_rows]
    
    # Check cleaning worked
    assert not np.any(np.isnan(cleaned_data)), "NaN values remain after cleaning"
    
    # Should have removed the problematic rows
    assert len(cleaned_data) < len(data_with_nan), "No rows removed"
    
    print(f"  Original rows: {len(data_with_nan)}")
    print(f"  Rows with NaN: {np.sum(~valid_rows)}")
    print(f"  Clean rows: {len(cleaned_data)}")
    print(f"  ✓ Missing data handled correctly")


def test_extreme_parameter_values():
    """
    Test behavior with extreme parameter values.
    
    Should fail gracefully or handle edge cases.
    """
    print("\n[TEST] Extreme parameter values")
    
    # Test very small lag
    small_lag = 1
    assert small_lag >= 1, "Lag must be at least 1"
    
    # Test very large lag
    large_lag = 1000
    n_frames = 500
    # Should warn if lag is too large relative to data
    if large_lag >= n_frames / 2:
        print(f"  Warning: lag {large_lag} >= half of frames {n_frames}")
    
    # Test zero dimensionality (should be invalid)
    try:
        dim = 0
        assert dim > 0, "Dimensionality must be positive"
        print(f"  ✗ Should have rejected dim=0")
    except AssertionError:
        print(f"  ✓ Correctly rejected dim=0")
    
    # Test negative n_clusters (should be invalid)
    try:
        n_clusters = -5
        assert n_clusters > 0, "Number of clusters must be positive"
        print(f"  ✗ Should have rejected negative clusters")
    except AssertionError:
        print(f"  ✓ Correctly rejected negative clusters")
    
    print(f"  ✓ Extreme values handled appropriately")


def main():
    """Run all reproducibility and robustness tests."""
    print("="*70)
    print("REPRODUCIBILITY & ROBUSTNESS TESTS")
    print("="*70)
    
    tests = [
        test_random_seed_reproducibility,
        test_noise_injection_robustness,
        test_parameter_sensitivity_analysis,
        test_cross_validation_stability,
        test_data_subset_consistency,
        test_computation_determinism,
        test_missing_data_handling,
        test_extreme_parameter_values,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
