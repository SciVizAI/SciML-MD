#!/usr/bin/env python3
"""
Scientific validation tools for MSM and TICA models.

This module implements various validation methods to ensure models are
scientifically sound and properly converged.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import warnings

from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.markov import TransitionCountEstimator
from deeptime.decomposition import TICA

# Constants for numerical stability
COVARIANCE_REGULARIZATION = 1e-6  # Small value added to diagonal for numerical stability


def chapman_kolmogorov_test(dtraj: np.ndarray, 
                            msm_lag: int, 
                            n_lags: int = 5,
                            n_states: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chapman-Kolmogorov test for MSM validation.
    
    Tests whether the MSM satisfies the Markov property by comparing
    predicted and estimated transition probabilities at multiple lag times.
    
    The Chapman-Kolmogorov equation states:
        P(k*tau) = P(tau)^k
    
    Args:
        dtraj: Discrete trajectory (state assignments)
        msm_lag: Lag time used to build the MSM
        n_lags: Number of lag times to test
        n_states: Number of states (if None, inferred from dtraj)
        
    Returns:
        lags: Array of lag times tested
        predicted: Predicted transition probabilities at each lag
        estimated: Estimated transition probabilities at each lag
        
    References:
        Prinz et al. (2011). "Markov models of molecular kinetics"
        J. Chem. Phys. 134: 174105
    """
    if n_states is None:
        n_states = int(dtraj.max() + 1)
    
    # Build MSM at base lag
    msm = MaximumLikelihoodMSM(lagtime=msm_lag, reversible=True).fit(dtraj).fetch_model()
    P_base = msm.transition_matrix
    
    # Test lags (multiples of base lag)
    test_lags = np.arange(1, n_lags + 1) * msm_lag
    
    predicted_P = []
    estimated_P = []
    
    for k, lag in enumerate(test_lags):
        # Predicted: P(k*tau) = P(tau)^k
        P_pred = np.linalg.matrix_power(P_base, k + 1)
        predicted_P.append(P_pred)
        
        # Estimated: Count transitions at lag k*tau
        try:
            counts = TransitionCountEstimator(lagtime=lag, count_mode='sliding').fit_fetch([dtraj])
            msm_test = MaximumLikelihoodMSM(reversible=True).fit_fetch(counts)
            P_est = msm_test.transition_matrix
            estimated_P.append(P_est)
        except Exception as e:
            warnings.warn(f"Failed to estimate transitions at lag {lag}: {e}")
            estimated_P.append(np.full_like(P_pred, np.nan))
    
    return test_lags, np.array(predicted_P), np.array(estimated_P)


def implied_timescales_convergence(dtraj: np.ndarray,
                                   lag_range: Optional[List[int]] = None,
                                   n_its: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute implied timescales as a function of lag time.
    
    Implied timescales should plateau at sufficiently long lag times,
    indicating that the Markov property is satisfied.
    
    Args:
        dtraj: Discrete trajectory
        lag_range: Range of lag times to test (if None, auto-determined)
        n_its: Number of implied timescales to compute
        
    Returns:
        lags: Array of lag times
        its: Implied timescales [n_lags x n_its]
        
    References:
        Prinz et al. (2011). "Markov models of molecular kinetics"
        J. Chem. Phys. 134: 174105
    """
    if lag_range is None:
        # Auto-determine lag range
        T = len(dtraj)
        lag_range = np.unique(np.logspace(
            np.log10(5), 
            np.log10(min(T // 10, 100)), 
            num=10
        ).astype(int))
    
    timescales_list = []
    valid_lags = []
    
    for lag in lag_range:
        try:
            counts = TransitionCountEstimator(lagtime=lag, count_mode='sliding').fit_fetch([dtraj])
            msm = MaximumLikelihoodMSM(reversible=True).fit_fetch(counts)
            ts = msm.timescales()[:n_its]
            
            # Pad with nan if fewer timescales available
            if len(ts) < n_its:
                ts = np.concatenate([ts, np.full(n_its - len(ts), np.nan)])
            
            timescales_list.append(ts)
            valid_lags.append(lag)
        except Exception as e:
            warnings.warn(f"Failed to compute timescales at lag {lag}: {e}")
            continue
    
    return np.array(valid_lags), np.array(timescales_list)


def vamp2_cross_validation(X: np.ndarray,
                           lag: int,
                           dim: int,
                           n_folds: int = 5,
                           seed: int = 42) -> Tuple[float, float]:
    """
    K-fold cross-validation for VAMP-2 score.
    
    Provides more robust estimate of model quality by testing on
    multiple held-out sets.
    
    Args:
        X: Feature trajectory [T x F]
        lag: TICA lag time
        dim: Number of dimensions
        n_folds: Number of cross-validation folds
        seed: Random seed
        
    Returns:
        mean_score: Mean VAMP-2 score across folds
        std_score: Standard deviation of VAMP-2 scores
    """
    rng = np.random.RandomState(seed)
    T = len(X)
    
    # Create fold indices
    indices = np.arange(T)
    rng.shuffle(indices)
    fold_size = T // n_folds
    
    scores = []
    
    for fold in range(n_folds):
        # Split into train/validation
        val_start = fold * fold_size
        val_end = min(val_start + fold_size, T)
        
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        # Ensure temporal order within each set
        train_indices = np.sort(train_indices)
        val_indices = np.sort(val_indices)
        
        X_train = X[train_indices]
        X_val = X[val_indices]
        
        try:
            # Fit on train
            vamp = TICA(lagtime=lag, dim=dim).fit(X_train).fetch_model()
            
            # Evaluate on validation
            if len(X_val) > lag + 10:
                Y_val = vamp.transform(X_val)
                
                # Compute VAMP-2 score
                C_0 = np.cov(Y_val[:-lag].T)
                C_1 = np.cov(Y_val[lag:].T)
                C_01 = np.cov(Y_val[:-lag].T, Y_val[lag:].T)[:dim, dim:]
                
                # Regularize covariances for numerical stability
                C_0 += COVARIANCE_REGULARIZATION * np.eye(dim)
                C_1 += COVARIANCE_REGULARIZATION * np.eye(dim)
                
                # VAMP-2 score
                C_0_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C_0)).T
                C_1_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C_1)).T
                
                K = C_0_inv_sqrt @ C_01 @ C_1_inv_sqrt
                s = np.linalg.svd(K, compute_uv=False)
                score = np.sum(s**2)
                
                scores.append(score)
        except Exception as e:
            warnings.warn(f"Fold {fold} failed: {e}")
            continue
    
    if len(scores) == 0:
        return -np.inf, np.inf
    
    return float(np.mean(scores)), float(np.std(scores))


def signal_correlation_analysis(signals: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Analyze correlation between anomaly detection signals.
    
    Signals should be approximately independent for effective fusion.
    High correlation (>0.7) suggests redundancy.
    
    Args:
        signals: Dictionary of signal name -> signal values
        
    Returns:
        correlation_matrix: DataFrame with pairwise correlations
    """
    # Create DataFrame from signals
    df = pd.DataFrame(signals)
    
    # Compute correlation matrix
    corr = df.corr(method='spearman')  # Spearman for robustness
    
    return corr


def validate_stationary_distribution(pi: np.ndarray,
                                     dtraj: np.ndarray,
                                     tolerance: float = 0.1) -> Tuple[bool, Dict]:
    """
    Validate that stationary distribution matches empirical frequencies.
    
    Args:
        pi: Stationary distribution from MSM
        dtraj: Discrete trajectory
        tolerance: Maximum allowed relative error
        
    Returns:
        is_valid: Whether validation passed
        diagnostics: Dictionary with validation metrics
    """
    # Compute empirical frequencies
    n_states = len(pi)
    empirical = np.bincount(dtraj, minlength=n_states) / len(dtraj)
    
    # Compute relative error
    # Only for states with sufficient sampling
    min_count = 10
    sampled_states = np.bincount(dtraj, minlength=n_states) >= min_count
    
    relative_error = np.abs(pi - empirical) / (empirical + 1e-10)
    
    max_error = np.max(relative_error[sampled_states]) if sampled_states.any() else np.inf
    mean_error = np.mean(relative_error[sampled_states]) if sampled_states.any() else np.inf
    
    is_valid = max_error < tolerance
    
    diagnostics = {
        'max_relative_error': float(max_error),
        'mean_relative_error': float(mean_error),
        'n_sampled_states': int(sampled_states.sum()),
        'empirical_freq': empirical,
        'msm_stationary': pi,
        'tolerance': tolerance
    }
    
    return is_valid, diagnostics


def plot_validation_summary(output_dir: Path,
                           ck_test_data: Optional[Tuple] = None,
                           its_data: Optional[Tuple] = None,
                           correlation_matrix: Optional[pd.DataFrame] = None):
    """
    Generate comprehensive validation plots.
    
    Args:
        output_dir: Directory to save plots
        ck_test_data: Chapman-Kolmogorov test results (lags, predicted, estimated)
        its_data: Implied timescales data (lags, timescales)
        correlation_matrix: Signal correlation matrix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Chapman-Kolmogorov test plot
    if ck_test_data is not None:
        lags, predicted, estimated = ck_test_data
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Plot first 4 most populated state pairs
        n_states = predicted.shape[1]
        for idx in range(min(4, n_states)):
            ax = axes[idx]
            for j in range(n_states):
                if predicted[:, idx, j].max() > 0.05:  # Only plot significant transitions
                    ax.plot(lags, predicted[:, idx, j], 'o-', label=f'Pred {idx}→{j}')
                    ax.plot(lags, estimated[:, idx, j], 's--', label=f'Est {idx}→{j}')
            
            ax.set_xlabel('Lag time')
            ax.set_ylabel('Transition probability')
            ax.set_title(f'Chapman-Kolmogorov: State {idx}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'chapman_kolmogorov.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved Chapman-Kolmogorov plot to {output_dir / 'chapman_kolmogorov.png'}")
    
    # Implied timescales plot
    if its_data is not None:
        lags, timescales = its_data
        
        plt.figure(figsize=(10, 6))
        for i in range(timescales.shape[1]):
            plt.plot(lags, timescales[:, i], 'o-', label=f'ITS {i+1}')
        
        plt.xlabel('Lag time (frames)')
        plt.ylabel('Implied timescale (frames)')
        plt.title('Implied Timescales Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'implied_timescales.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved implied timescales plot to {output_dir / 'implied_timescales.png'}")
    
    # Signal correlation heatmap
    if correlation_matrix is not None:
        plt.figure(figsize=(8, 7))
        im = plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, label='Spearman correlation')
        
        # Add text annotations
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix)):
                text = plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                              ha='center', va='center', color='black', fontsize=10)
        
        plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=45, ha='right')
        plt.yticks(range(len(correlation_matrix)), correlation_matrix.index)
        plt.title('Anomaly Signal Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / 'signal_correlations.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved signal correlation plot to {output_dir / 'signal_correlations.png'}")


def generate_validation_report(output_file: Path,
                               ck_results: Optional[Dict] = None,
                               its_results: Optional[Dict] = None,
                               cv_results: Optional[Dict] = None,
                               stationary_results: Optional[Dict] = None,
                               correlation_results: Optional[pd.DataFrame] = None):
    """
    Generate a comprehensive validation report in JSON format.
    
    Args:
        output_file: Path to save report
        ck_results: Chapman-Kolmogorov test results
        its_results: Implied timescales results
        cv_results: Cross-validation results
        stationary_results: Stationary distribution validation
        correlation_results: Signal correlation matrix
    """
    import json
    
    report = {
        'validation_report': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'tests_performed': []
        }
    }
    
    if ck_results is not None:
        report['chapman_kolmogorov_test'] = ck_results
        report['validation_report']['tests_performed'].append('chapman_kolmogorov')
    
    if its_results is not None:
        report['implied_timescales'] = its_results
        report['validation_report']['tests_performed'].append('implied_timescales')
    
    if cv_results is not None:
        report['cross_validation'] = cv_results
        report['validation_report']['tests_performed'].append('vamp2_cross_validation')
    
    if stationary_results is not None:
        # Convert numpy arrays to lists for JSON serialization
        stationary_results_json = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in stationary_results.items()
        }
        report['stationary_distribution_validation'] = stationary_results_json
        report['validation_report']['tests_performed'].append('stationary_distribution')
    
    if correlation_results is not None:
        report['signal_correlations'] = correlation_results.to_dict()
        report['validation_report']['tests_performed'].append('signal_correlation')
    
    # Determine overall validation status
    all_passed = True
    if stationary_results is not None:
        all_passed &= stationary_results.get('max_relative_error', 1.0) < stationary_results.get('tolerance', 0.1)
    
    report['validation_report']['overall_status'] = 'PASSED' if all_passed else 'NEEDS_REVIEW'
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Validation report saved to {output_file}")
    print(f"  Overall status: {report['validation_report']['overall_status']}")
    print(f"  Tests performed: {', '.join(report['validation_report']['tests_performed'])}")
