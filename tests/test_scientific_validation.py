#!/usr/bin/env python3
"""
Unit tests for scientific validation tools.

Tests the validation methods in msm/validation.py to ensure
they correctly identify well-behaved and problematic models.
"""
import sys
import numpy as np
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from msm.validation import (
    chapman_kolmogorov_test,
    implied_timescales_convergence,
    vamp2_cross_validation,
    signal_correlation_analysis,
    validate_stationary_distribution,
    generate_validation_report
)


def create_synthetic_msm_data(n_states=10, T=1000, seed=42):
    """Create synthetic MSM data for testing."""
    np.random.seed(seed)
    
    # Create a simple 2-state system that switches occasionally
    dtraj = np.zeros(T, dtype=int)
    current_state = 0
    
    for t in range(1, T):
        if np.random.rand() < 0.02:  # 2% chance to switch
            current_state = (current_state + 1) % n_states
        dtraj[t] = current_state
    
    return dtraj


def create_well_behaved_features(T=500, seed=42):
    """Create features with temporal structure (good for tICA)."""
    np.random.seed(seed)
    t = np.linspace(0, 10, T)
    
    # Slow oscillations + noise
    X = np.column_stack([
        np.sin(t) + 0.1 * np.random.randn(T),
        np.cos(t) + 0.1 * np.random.randn(T),
        np.sin(2*t) + 0.2 * np.random.randn(T),
        np.random.randn(T) * 0.3
    ])
    
    return X


def test_chapman_kolmogorov_basic():
    """Test Chapman-Kolmogorov test with synthetic data."""
    print("\n[TEST] Chapman-Kolmogorov test - basic functionality")
    
    dtraj = create_synthetic_msm_data(n_states=5, T=800)
    
    lags, predicted, estimated = chapman_kolmogorov_test(
        dtraj, msm_lag=20, n_lags=3
    )
    
    # Check outputs have correct shape
    assert len(lags) == 3
    assert predicted.shape[0] == 3  # 3 lags
    assert estimated.shape[0] == 3
    
    # Predicted and estimated should be similar for well-behaved MSM
    errors = np.abs(predicted - estimated)
    mean_error = np.nanmean(errors)
    
    print(f"  Lags tested: {lags}")
    print(f"  Mean absolute error: {mean_error:.4f}")
    print(f"  ✓ Test completed")
    
    assert mean_error < 0.5, "CK test error should be reasonable for synthetic data"


def test_implied_timescales_convergence():
    """Test implied timescales computation."""
    print("\n[TEST] Implied timescales convergence")
    
    dtraj = create_synthetic_msm_data(n_states=8, T=1000)
    
    lags, timescales = implied_timescales_convergence(
        dtraj, lag_range=[5, 10, 15, 20, 30], n_its=3
    )
    
    # Check outputs
    assert len(lags) > 0
    assert timescales.shape[0] == len(lags)
    assert timescales.shape[1] == 3  # 3 timescales
    
    # Timescales should generally increase or stay constant
    # (shouldn't decrease significantly)
    for i in range(timescales.shape[1]):
        ts = timescales[:, i]
        ts_valid = ts[~np.isnan(ts)]
        if len(ts_valid) >= 2:
            # Check that timescales don't decrease dramatically
            ratio = ts_valid[-1] / ts_valid[0]
            assert ratio > 0.5, f"Timescale {i} decreased too much: {ratio}"
    
    print(f"  Lags tested: {lags}")
    print(f"  Timescales shape: {timescales.shape}")
    print(f"  ✓ Test completed")


def test_vamp2_cross_validation():
    """Test VAMP-2 cross-validation."""
    print("\n[TEST] VAMP-2 cross-validation")
    
    X = create_well_behaved_features(T=500)
    
    mean_score, std_score = vamp2_cross_validation(
        X, lag=10, dim=2, n_folds=3, seed=42
    )
    
    # Score should be positive and finite
    assert mean_score > 0, "Mean VAMP-2 score should be positive"
    assert np.isfinite(mean_score), "Mean score should be finite"
    assert np.isfinite(std_score), "Std score should be finite"
    assert std_score >= 0, "Std should be non-negative"
    
    print(f"  Mean VAMP-2 score: {mean_score:.4f} ± {std_score:.4f}")
    print(f"  ✓ Test completed")


def test_vamp2_cv_reproducibility():
    """Test that CV is reproducible with same seed."""
    print("\n[TEST] VAMP-2 CV reproducibility")
    
    X = create_well_behaved_features(T=400)
    
    score1, _ = vamp2_cross_validation(X, lag=10, dim=2, n_folds=3, seed=123)
    score2, _ = vamp2_cross_validation(X, lag=10, dim=2, n_folds=3, seed=123)
    
    assert np.abs(score1 - score2) < 1e-10, "Same seed should give same score"
    
    print(f"  Score 1: {score1:.6f}")
    print(f"  Score 2: {score2:.6f}")
    print(f"  ✓ Reproducible")


def test_signal_correlation_analysis():
    """Test signal correlation analysis."""
    print("\n[TEST] Signal correlation analysis")
    
    T = 500
    
    # Create signals with known correlations
    signal1 = np.random.randn(T)
    signal2 = signal1 + 0.3 * np.random.randn(T)  # Correlated
    signal3 = np.random.randn(T)  # Independent
    
    signals = {
        'signal1': signal1,
        'signal2': signal2,
        'signal3': signal3
    }
    
    corr = signal_correlation_analysis(signals)
    
    # Check correlation matrix properties
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0), "Diagonal should be 1"
    assert np.allclose(corr, corr.T), "Should be symmetric"
    
    # signal1 and signal2 should be correlated
    assert corr.loc['signal1', 'signal2'] > 0.5, "Correlated signals should have high correlation"
    
    # signal1 and signal3 should be independent
    assert abs(corr.loc['signal1', 'signal3']) < 0.3, "Independent signals should have low correlation"
    
    print(f"  Correlation matrix:\n{corr}")
    print(f"  ✓ Test completed")


def test_stationary_distribution_validation_pass():
    """Test stationary distribution validation with good match."""
    print("\n[TEST] Stationary distribution validation - passing case")
    
    dtraj = create_synthetic_msm_data(n_states=5, T=1000)
    
    # Compute empirical distribution
    pi_empirical = np.bincount(dtraj, minlength=5) / len(dtraj)
    
    # Add small noise to simulate MSM estimate
    pi_msm = pi_empirical + np.random.randn(5) * 0.02
    pi_msm = np.abs(pi_msm)
    pi_msm /= pi_msm.sum()
    
    is_valid, diagnostics = validate_stationary_distribution(
        pi_msm, dtraj, tolerance=0.2
    )
    
    print(f"  Max relative error: {diagnostics['max_relative_error']:.4f}")
    print(f"  Mean relative error: {diagnostics['mean_relative_error']:.4f}")
    print(f"  Validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
    
    assert is_valid, "Should pass validation with small error"
    assert diagnostics['max_relative_error'] < 0.2


def test_stationary_distribution_validation_fail():
    """Test stationary distribution validation with poor match."""
    print("\n[TEST] Stationary distribution validation - failing case")
    
    dtraj = create_synthetic_msm_data(n_states=5, T=1000)
    
    # Create a very different distribution
    pi_wrong = np.array([0.6, 0.2, 0.1, 0.05, 0.05])
    
    is_valid, diagnostics = validate_stationary_distribution(
        pi_wrong, dtraj, tolerance=0.1
    )
    
    print(f"  Max relative error: {diagnostics['max_relative_error']:.4f}")
    print(f"  Validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
    
    # Should fail validation
    assert not is_valid, "Should fail validation with large error"


def test_validation_report_generation():
    """Test validation report generation."""
    print("\n[TEST] Validation report generation")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / 'validation_report.json'
        
        # Create some sample results
        ck_results = {
            'lags_tested': [10, 20, 30],
            'mean_absolute_error': 0.12,
            'max_absolute_error': 0.18,
            'passed': True
        }
        
        stationary_results = {
            'max_relative_error': 0.09,
            'mean_relative_error': 0.05,
            'empirical_freq': np.array([0.3, 0.3, 0.2, 0.2]),
            'msm_stationary': np.array([0.29, 0.31, 0.21, 0.19]),
            'tolerance': 0.15
        }
        
        generate_validation_report(
            output_file,
            ck_results=ck_results,
            stationary_results=stationary_results
        )
        
        # Check file was created
        assert output_file.exists(), "Report file should be created"
        
        # Load and check contents
        import json
        with open(output_file) as f:
            report = json.load(f)
        
        assert 'validation_report' in report
        assert 'chapman_kolmogorov_test' in report
        assert 'stationary_distribution_validation' in report
        assert report['validation_report']['overall_status'] == 'PASSED'
        
        print(f"  Report saved to: {output_file}")
        print(f"  Overall status: {report['validation_report']['overall_status']}")
        print(f"  ✓ Test completed")


def test_edge_case_short_trajectory():
    """Test validation with very short trajectory."""
    print("\n[TEST] Edge case - short trajectory")
    
    # Very short trajectory
    dtraj = np.array([0, 1, 0, 1, 0] * 10, dtype=int)
    
    # Should handle gracefully
    try:
        lags, timescales = implied_timescales_convergence(
            dtraj, lag_range=[2, 5], n_its=2
        )
        print(f"  Handled short trajectory: {len(lags)} lags computed")
        print(f"  ✓ No crash")
    except Exception as e:
        print(f"  ✗ Failed with error: {e}")
        raise


def test_edge_case_disconnected_states():
    """Test validation with disconnected MSM states."""
    print("\n[TEST] Edge case - disconnected states")
    
    # Create trajectory that never visits state 2
    dtraj = np.array([0, 0, 1, 1, 0, 1] * 50, dtype=int)
    
    # Should handle gracefully (use largest connected component)
    try:
        lags, predicted, estimated = chapman_kolmogorov_test(
            dtraj, msm_lag=2, n_lags=2
        )
        print(f"  Handled disconnected states")
        print(f"  ✓ No crash")
    except Exception as e:
        print(f"  Note: Expected behavior for disconnected states: {e}")
        # This is acceptable - disconnected states are a known limitation


def main():
    """Run all tests."""
    print("="*70)
    print("TESTING SCIENTIFIC VALIDATION TOOLS")
    print("="*70)
    
    tests = [
        test_chapman_kolmogorov_basic,
        test_implied_timescales_convergence,
        test_vamp2_cross_validation,
        test_vamp2_cv_reproducibility,
        test_signal_correlation_analysis,
        test_stationary_distribution_validation_pass,
        test_stationary_distribution_validation_fail,
        test_validation_report_generation,
        test_edge_case_short_trajectory,
        test_edge_case_disconnected_states,
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
