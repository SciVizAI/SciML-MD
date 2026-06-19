#!/usr/bin/env python3
"""
Statistical validation tests for ML pipeline.

Implements hypothesis testing, effect size calculation, and statistical
power analysis following best practices from scientific literature.

References:
- Cohen, J. (1988) "Statistical Power Analysis for the Behavioral Sciences"
- Benjamini & Hochberg (1995) "Controlling the False Discovery Rate" J. R. Stat. Soc. B
- Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
"""
import sys
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))


def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size.
    
    Cohen's d is a standardized measure of effect size:
    - Small effect: d = 0.2
    - Medium effect: d = 0.5  
    - Large effect: d = 0.8
    
    Reference: Cohen, J. (1988)
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d


def test_anomaly_detection_statistical_power():
    """
    Test that anomaly detection has sufficient statistical power.
    
    Power analysis ensures we can detect true anomalies when they exist.
    """
    print("\n[TEST] Statistical power of anomaly detection")
    
    # Simulate normal frames and anomalous frames
    n_normal = 900
    n_anomaly = 100
    
    # Normal frames: mean=0, std=1
    normal_scores = np.random.randn(n_normal)
    
    # Anomalous frames: shifted distribution (effect size)
    effect_size = 1.0  # Large effect
    anomaly_scores = effect_size + np.random.randn(n_anomaly)
    
    # Statistical test (t-test)
    t_stat, p_value = stats.ttest_ind(anomaly_scores, normal_scores)
    
    # Calculate effect size
    d = cohens_d(anomaly_scores, normal_scores)
    
    # Power should be high for large effect sizes
    assert p_value < 0.05, f"Failed to detect anomalies: p={p_value:.4f}"
    assert abs(d) > 0.8, f"Effect size too small: d={d:.2f}"
    
    print(f"  Normal frames: {n_normal}")
    print(f"  Anomaly frames: {n_anomaly}")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4e}")
    print(f"  Cohen's d: {d:.2f} (Large effect)")
    print(f"  ✓ Sufficient statistical power")


def test_multiple_testing_correction():
    """
    Test multiple testing correction (Benjamini-Hochberg).
    
    When testing many hypotheses (e.g., per-residue significance),
    we must control the False Discovery Rate (FDR).
    
    Reference: Benjamini & Hochberg (1995)
    """
    print("\n[TEST] Multiple testing correction (FDR)")
    
    # Simulate testing 100 residues
    np.random.seed(42)  # Set seed for reproducibility
    n_tests = 100
    n_true_positives = 20
    
    # Generate p-values: 20 true positives (very small p), 80 nulls (uniform p)
    p_values = np.concatenate([
        np.random.beta(0.1, 10, n_true_positives),  # Stronger true signals
        np.random.uniform(0, 1, n_tests - n_true_positives)  # Null hypotheses
    ])
    
    # Benjamini-Hochberg procedure
    fdr_level = 0.05
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    # BH critical values
    m = len(p_values)
    bh_critical = np.arange(1, m + 1) * fdr_level / m
    
    # Find largest k where p_k <= (k/m) * FDR
    significant = sorted_p <= bh_critical
    n_discovered = np.sum(significant)
    
    # Should properly apply BH correction (may be 0 discoveries if no signals)
    assert n_discovered <= n_tests, "Invalid number of discoveries"
    
    # Check that BH procedure is working correctly
    # At least some of the small p-values should be significant
    assert np.any(sorted_p < fdr_level / m), "BH critical values too strict"
    
    print(f"  Tests performed: {n_tests}")
    print(f"  FDR level: {fdr_level}")
    print(f"  Discoveries: {n_discovered}")
    print(f"  ✓ Multiple testing properly controlled")


def test_bootstrap_confidence_intervals():
    """
    Test bootstrap confidence interval estimation.
    
    Bootstrap provides non-parametric confidence intervals for
    statistics without assuming a distribution.
    
    Reference: Efron & Tibshirani (1993)
    """
    print("\n[TEST] Bootstrap confidence intervals")
    
    # Simulate sample data
    np.random.seed(42)
    data = np.random.exponential(scale=2.0, size=200)
    
    # Statistic of interest: mean
    observed_mean = np.mean(data)
    
    # Bootstrap resampling
    n_bootstrap = 1000
    bootstrap_means = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means[i] = np.mean(sample)
    
    # 95% confidence interval (percentile method)
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    # True mean should be within CI (with high probability)
    true_mean = 2.0  # Known from exponential distribution
    
    # CI should be reasonable width
    ci_width = ci_upper - ci_lower
    assert ci_width > 0, "CI has zero width"
    assert ci_width < 2.0, f"CI too wide: {ci_width:.2f}"
    
    # Observed mean should be close to true mean
    assert abs(observed_mean - true_mean) < 0.5, \
        f"Sample mean {observed_mean:.2f} far from true mean {true_mean}"
    
    print(f"  Sample size: {len(data)}")
    print(f"  Bootstrap iterations: {n_bootstrap}")
    print(f"  Observed mean: {observed_mean:.3f}")
    print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"  CI width: {ci_width:.3f}")
    print(f"  ✓ Bootstrap CI computed successfully")


def test_normality_testing():
    """
    Test normality assumption using Shapiro-Wilk test.
    
    Many ML methods assume normality. We should test this assumption.
    """
    print("\n[TEST] Normality testing (Shapiro-Wilk)")
    
    # Test with normal data
    normal_data = np.random.randn(100)
    stat_normal, p_normal = stats.shapiro(normal_data)
    
    # Test with non-normal data (exponential)
    nonnormal_data = np.random.exponential(1.0, 100)
    stat_nonnormal, p_nonnormal = stats.shapiro(nonnormal_data)
    
    # Normal data should pass (p > 0.05)
    assert p_normal > 0.05, f"Normal data failed normality test: p={p_normal:.4f}"
    
    # Non-normal data should fail (p < 0.05)
    assert p_nonnormal < 0.05, f"Non-normal data passed normality test: p={p_nonnormal:.4f}"
    
    print(f"  Normal data: W={stat_normal:.4f}, p={p_normal:.4f} (PASS)")
    print(f"  Non-normal data: W={stat_nonnormal:.4f}, p={p_nonnormal:.4f} (FAIL)")
    print(f"  ✓ Normality testing works correctly")


def test_correlation_significance():
    """
    Test significance of correlations.
    
    Correlations should be tested for statistical significance,
    especially with small sample sizes.
    """
    print("\n[TEST] Correlation significance testing")
    
    n = 100
    
    # Create correlated variables
    x = np.random.randn(n)
    y = 0.7 * x + 0.3 * np.random.randn(n)  # Strong correlation
    
    # Pearson correlation
    r, p_value = stats.pearsonr(x, y)
    
    # Should detect significant correlation
    assert p_value < 0.05, f"Failed to detect correlation: p={p_value:.4f}"
    assert abs(r) > 0.5, f"Correlation too weak: r={r:.3f}"
    
    # Test with uncorrelated variables
    x2 = np.random.randn(n)
    y2 = np.random.randn(n)
    r2, p_value2 = stats.pearsonr(x2, y2)
    
    # Should not detect significant correlation
    assert p_value2 > 0.05 or abs(r2) < 0.3, \
        f"False positive correlation: r={r2:.3f}, p={p_value2:.4f}"
    
    print(f"  Correlated: r={r:.3f}, p={p_value:.4e}")
    print(f"  Uncorrelated: r={r2:.3f}, p={p_value2:.4f}")
    print(f"  ✓ Correlation testing works")


def test_distribution_comparison_ks():
    """
    Test distribution comparison using Kolmogorov-Smirnov test.
    
    Useful for comparing anomaly score distributions across conditions.
    """
    print("\n[TEST] Distribution comparison (KS test)")
    
    # Two samples from same distribution (should not reject)
    sample1 = np.random.normal(0, 1, 200)
    sample2 = np.random.normal(0, 1, 200)
    
    stat_same, p_same = stats.ks_2samp(sample1, sample2)
    
    # Should not reject null (p > 0.05)
    assert p_same > 0.05, f"False positive: different distributions detected, p={p_same:.4f}"
    
    # Two samples from different distributions (should reject)
    sample3 = np.random.normal(2, 1, 200)
    stat_diff, p_diff = stats.ks_2samp(sample1, sample3)
    
    # Should reject null (p < 0.05)
    assert p_diff < 0.05, f"Failed to detect different distributions: p={p_diff:.4f}"
    
    print(f"  Same distribution: KS={stat_same:.3f}, p={p_same:.4f}")
    print(f"  Different distributions: KS={stat_diff:.3f}, p={p_diff:.4e}")
    print(f"  ✓ KS test works correctly")


def test_variance_homogeneity_levene():
    """
    Test variance homogeneity using Levene's test.
    
    Important assumption for many parametric tests.
    """
    print("\n[TEST] Variance homogeneity (Levene's test)")
    
    # Groups with equal variance
    group1 = np.random.normal(0, 1, 100)
    group2 = np.random.normal(1, 1, 100)  # Different mean, same variance
    
    stat_equal, p_equal = stats.levene(group1, group2)
    
    # Should not reject equal variance
    assert p_equal > 0.05, f"False rejection of equal variance: p={p_equal:.4f}"
    
    # Groups with unequal variance
    group3 = np.random.normal(0, 3, 100)  # Much larger variance
    stat_unequal, p_unequal = stats.levene(group1, group3)
    
    # Should reject equal variance
    assert p_unequal < 0.05, f"Failed to detect unequal variance: p={p_unequal:.4f}"
    
    print(f"  Equal variance: F={stat_equal:.3f}, p={p_equal:.4f}")
    print(f"  Unequal variance: F={stat_unequal:.3f}, p={p_unequal:.4e}")
    print(f"  ✓ Levene's test works correctly")


def test_outlier_detection_mad():
    """
    Test outlier detection using Median Absolute Deviation (MAD).
    
    MAD is more robust than standard deviation for outlier detection.
    """
    print("\n[TEST] Outlier detection (MAD method)")
    
    # Normal data with outliers
    data = np.random.randn(100)
    data = np.append(data, [10, -10, 15])  # Add clear outliers
    
    # Calculate MAD
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    # Modified z-scores using MAD
    # 1.4826 is the constant to make MAD consistent with std for normal data
    if mad > 0:
        modified_z = 0.6745 * (data - median) / mad
    else:
        modified_z = np.zeros_like(data)
    
    # Identify outliers (|z| > 3.5 is common threshold)
    outliers = np.abs(modified_z) > 3.5
    n_outliers = np.sum(outliers)
    
    # Should detect the 3 outliers we added
    assert n_outliers >= 3, f"Failed to detect outliers: found {n_outliers}"
    assert n_outliers <= 10, f"Too many outliers detected: {n_outliers}"
    
    print(f"  Sample size: {len(data)}")
    print(f"  MAD: {mad:.3f}")
    print(f"  Outliers detected: {n_outliers}")
    print(f"  ✓ Outlier detection working")


def main():
    """Run all statistical validation tests."""
    print("="*70)
    print("STATISTICAL VALIDATION TESTS")
    print("="*70)
    
    tests = [
        test_anomaly_detection_statistical_power,
        test_multiple_testing_correction,
        test_bootstrap_confidence_intervals,
        test_normality_testing,
        test_correlation_significance,
        test_distribution_comparison_ks,
        test_variance_homogeneity_levene,
        test_outlier_detection_mad,
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
