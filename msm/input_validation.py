#!/usr/bin/env python3
"""
Input data validation utilities.

This module provides functions to validate trajectory and feature data
before training ML models.
"""
import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, Dict, List, Optional


def check_trajectory_quality(traj_path: str, top_path: str) -> Dict[str, any]:
    """
    Check quality of MD trajectory.
    
    Args:
        traj_path: Path to trajectory file
        top_path: Path to topology file
        
    Returns:
        diagnostics: Dictionary with quality metrics
    """
    try:
        import mdtraj as md
    except ImportError:
        warnings.warn("mdtraj not available, skipping trajectory checks")
        return {'status': 'skipped', 'reason': 'mdtraj not installed'}
    
    traj = md.load(traj_path, top=top_path)
    
    diagnostics = {
        'n_frames': traj.n_frames,
        'n_atoms': traj.n_atoms,
        'n_residues': traj.n_residues,
        'time_span_ps': float(traj.time[-1] - traj.time[0]),
        'timestep_ps': float(np.median(np.diff(traj.time))),
        'issues': []
    }
    
    # Check 1: Sufficient frames
    if traj.n_frames < 100:
        diagnostics['issues'].append({
            'severity': 'ERROR',
            'message': f'Too few frames ({traj.n_frames}). Need at least 100, recommended 1000+.'
        })
    elif traj.n_frames < 1000:
        diagnostics['issues'].append({
            'severity': 'WARNING',
            'message': f'Limited frames ({traj.n_frames}). Recommended 1000+ for robust statistics.'
        })
    
    # Check 2: Clashes (minimum CA-CA distance)
    ca_atoms = traj.topology.select('name CA')
    if len(ca_atoms) > 1:
        distances = md.compute_distances(traj, [(ca_atoms[0], ca_atoms[1])])
        min_distance = distances.min()
        # Check for unrealistic close distances (0.2 nm = 2 Angstroms)
        if min_distance < 0.2:
            diagnostics['issues'].append({
                'severity': 'ERROR',
                'message': f'Possible clash detected: min CA-CA distance = {min_distance:.2f} nm (2 Å threshold)'
            })
    
    # Check 3: Unfolding
    rg = md.compute_rg(traj)
    rg_mean = rg.mean()
    rg_std = rg.std()
    rg_max = rg.max()
    
    if rg_max > rg_mean + 3 * rg_std:
        diagnostics['issues'].append({
            'severity': 'WARNING',
            'message': f'Possible unfolding: Rg outliers detected (max={rg_max:.2f} nm, mean={rg_mean:.2f} nm)'
        })
    
    diagnostics['rg_mean_nm'] = float(rg_mean)
    diagnostics['rg_std_nm'] = float(rg_std)
    diagnostics['rg_max_nm'] = float(rg_max)
    
    # Overall status
    has_errors = any(issue['severity'] == 'ERROR' for issue in diagnostics['issues'])
    diagnostics['status'] = 'FAILED' if has_errors else ('WARNING' if diagnostics['issues'] else 'PASSED')
    
    return diagnostics


def check_feature_quality(features: np.ndarray) -> Dict[str, any]:
    """
    Check quality of extracted features.
    
    Args:
        features: Feature matrix [T x F]
        
    Returns:
        diagnostics: Dictionary with quality metrics
    """
    T, F = features.shape
    
    diagnostics = {
        'n_frames': T,
        'n_features': F,
        'issues': []
    }
    
    # Check 1: NaN values
    nan_count = np.isnan(features).sum()
    if nan_count > 0:
        diagnostics['issues'].append({
            'severity': 'ERROR',
            'message': f'Found {nan_count} NaN values in features. Data is corrupted.'
        })
    
    # Check 2: Inf values
    inf_count = np.isinf(features).sum()
    if inf_count > 0:
        diagnostics['issues'].append({
            'severity': 'ERROR',
            'message': f'Found {inf_count} Inf values in features. Data is corrupted.'
        })
    
    # Check 3: Zero variance features
    variances = np.var(features, axis=0)
    zero_var_count = (variances == 0).sum()
    if zero_var_count > 0:
        diagnostics['issues'].append({
            'severity': 'ERROR',
            'message': f'Found {zero_var_count} zero-variance features. Will cause singular covariance matrix.'
        })
    
    # Check 4: Very low variance features
    low_var_count = (variances < 1e-6).sum()
    if low_var_count > zero_var_count:
        diagnostics['issues'].append({
            'severity': 'WARNING',
            'message': f'Found {low_var_count} very low variance features. May cause numerical issues.'
        })
    
    # Check 5: Extremely large values
    max_abs = np.abs(features).max()
    if max_abs > 1e6:
        diagnostics['issues'].append({
            'severity': 'WARNING',
            'message': f'Very large feature values (max={max_abs:.2e}). Consider normalization.'
        })
    
    # Statistics
    diagnostics['nan_count'] = int(nan_count)
    diagnostics['inf_count'] = int(inf_count)
    diagnostics['zero_var_features'] = int(zero_var_count)
    diagnostics['low_var_features'] = int(low_var_count)
    diagnostics['max_abs_value'] = float(max_abs)
    diagnostics['mean_variance'] = float(np.mean(variances))
    diagnostics['min_variance'] = float(np.min(variances))
    
    # Overall status
    has_errors = any(issue['severity'] == 'ERROR' for issue in diagnostics['issues'])
    diagnostics['status'] = 'FAILED' if has_errors else ('WARNING' if diagnostics['issues'] else 'PASSED')
    
    return diagnostics


def check_parameter_compatibility(n_frames: int,
                                  lag_tica: int,
                                  lag_msm: int,
                                  n_clusters: int) -> Dict[str, any]:
    """
    Check that parameters are compatible with data size.
    
    Args:
        n_frames: Number of frames in trajectory
        lag_tica: TICA lag time
        lag_msm: MSM lag time
        n_clusters: Number of clusters
        
    Returns:
        diagnostics: Dictionary with compatibility checks
    """
    diagnostics = {
        'n_frames': n_frames,
        'lag_tica': lag_tica,
        'lag_msm': lag_msm,
        'n_clusters': n_clusters,
        'issues': []
    }
    
    # Check 1: TICA lag vs trajectory length
    if lag_tica >= n_frames / 10:
        diagnostics['issues'].append({
            'severity': 'ERROR',
            'message': f'TICA lag ({lag_tica}) too large for trajectory ({n_frames} frames). Should be < {n_frames // 10}.'
        })
    elif lag_tica >= n_frames / 20:
        diagnostics['issues'].append({
            'severity': 'WARNING',
            'message': f'TICA lag ({lag_tica}) relatively large. Consider reducing if validation fails.'
        })
    
    # Check 2: MSM lag vs trajectory length
    if lag_msm >= n_frames / 5:
        diagnostics['issues'].append({
            'severity': 'ERROR',
            'message': f'MSM lag ({lag_msm}) too large for trajectory ({n_frames} frames). Should be < {n_frames // 5}.'
        })
    
    # Check 3: Frames per cluster
    frames_per_cluster = n_frames / n_clusters
    if frames_per_cluster < 20:
        diagnostics['issues'].append({
            'severity': 'ERROR',
            'message': f'Too few frames per cluster ({frames_per_cluster:.1f}). Need at least 20, recommended 100+.'
        })
    elif frames_per_cluster < 50:
        diagnostics['issues'].append({
            'severity': 'WARNING',
            'message': f'Limited frames per cluster ({frames_per_cluster:.1f}). Recommended 100+ for robust statistics.'
        })
    
    # Check 4: MSM lag should be >= TICA lag
    if lag_msm < lag_tica:
        diagnostics['issues'].append({
            'severity': 'WARNING',
            'message': f'MSM lag ({lag_msm}) < TICA lag ({lag_tica}). Typically MSM lag should be 2-5x TICA lag.'
        })
    
    diagnostics['frames_per_cluster'] = float(frames_per_cluster)
    diagnostics['tica_lag_fraction'] = float(lag_tica / n_frames)
    diagnostics['msm_lag_fraction'] = float(lag_msm / n_frames)
    
    # Overall status
    has_errors = any(issue['severity'] == 'ERROR' for issue in diagnostics['issues'])
    diagnostics['status'] = 'FAILED' if has_errors else ('WARNING' if diagnostics['issues'] else 'PASSED')
    
    return diagnostics


def validate_input_data(features_path: Optional[str] = None,
                       traj_path: Optional[str] = None,
                       top_path: Optional[str] = None,
                       lag_tica: Optional[int] = None,
                       lag_msm: Optional[int] = None,
                       n_clusters: Optional[int] = None) -> Tuple[bool, Dict]:
    """
    Comprehensive input data validation.
    
    Args:
        features_path: Path to features.npy (optional)
        traj_path: Path to trajectory (optional)
        top_path: Path to topology (optional)
        lag_tica: TICA lag time (optional)
        lag_msm: MSM lag time (optional)
        n_clusters: Number of clusters (optional)
        
    Returns:
        is_valid: Whether all checks passed
        report: Validation report
    """
    report = {
        'validation_date': np.datetime64('now').astype(str),
        'checks_performed': [],
        'overall_status': 'PASSED'
    }
    
    all_passed = True
    
    # Check trajectory if provided
    if traj_path is not None and top_path is not None:
        print("Checking trajectory quality...")
        traj_diag = check_trajectory_quality(traj_path, top_path)
        report['trajectory_quality'] = traj_diag
        report['checks_performed'].append('trajectory_quality')
        
        if traj_diag['status'] == 'FAILED':
            all_passed = False
            print(f"  ✗ Trajectory check FAILED")
        elif traj_diag['status'] == 'WARNING':
            print(f"  ⚠ Trajectory check has WARNINGS")
        else:
            print(f"  ✓ Trajectory check PASSED")
        
        for issue in traj_diag.get('issues', []):
            print(f"    [{issue['severity']}] {issue['message']}")
    
    # Check features if provided
    if features_path is not None:
        print("Checking feature quality...")
        features = np.load(features_path)
        feat_diag = check_feature_quality(features)
        report['feature_quality'] = feat_diag
        report['checks_performed'].append('feature_quality')
        
        if feat_diag['status'] == 'FAILED':
            all_passed = False
            print(f"  ✗ Feature check FAILED")
        elif feat_diag['status'] == 'WARNING':
            print(f"  ⚠ Feature check has WARNINGS")
        else:
            print(f"  ✓ Feature check PASSED")
        
        for issue in feat_diag.get('issues', []):
            print(f"    [{issue['severity']}] {issue['message']}")
        
        n_frames = features.shape[0]
    elif traj_path is not None and 'trajectory_quality' in report:
        n_frames = report['trajectory_quality']['n_frames']
    else:
        n_frames = None
    
    # Check parameter compatibility if provided
    if all([n_frames, lag_tica, lag_msm, n_clusters]):
        print("Checking parameter compatibility...")
        param_diag = check_parameter_compatibility(n_frames, lag_tica, lag_msm, n_clusters)
        report['parameter_compatibility'] = param_diag
        report['checks_performed'].append('parameter_compatibility')
        
        if param_diag['status'] == 'FAILED':
            all_passed = False
            print(f"  ✗ Parameter check FAILED")
        elif param_diag['status'] == 'WARNING':
            print(f"  ⚠ Parameter check has WARNINGS")
        else:
            print(f"  ✓ Parameter check PASSED")
        
        for issue in param_diag.get('issues', []):
            print(f"    [{issue['severity']}] {issue['message']}")
    
    report['overall_status'] = 'PASSED' if all_passed else 'FAILED'
    
    return all_passed, report


def print_validation_summary(report: Dict):
    """Print a human-readable validation summary."""
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    status_symbol = "✓" if report['overall_status'] == 'PASSED' else "✗"
    print(f"\nOverall Status: {status_symbol} {report['overall_status']}")
    print(f"Checks Performed: {', '.join(report['checks_performed'])}")
    
    # Count issues by severity
    total_errors = 0
    total_warnings = 0
    
    for check_name in report['checks_performed']:
        check_result = report[check_name]
        for issue in check_result.get('issues', []):
            if issue['severity'] == 'ERROR':
                total_errors += 1
            elif issue['severity'] == 'WARNING':
                total_warnings += 1
    
    print(f"\nIssues Found:")
    print(f"  Errors: {total_errors}")
    print(f"  Warnings: {total_warnings}")
    
    if total_errors > 0:
        print("\n⚠ CRITICAL: Fix all errors before training!")
    elif total_warnings > 0:
        print("\n⚠ WARNING: Review warnings before training.")
    else:
        print("\n✓ All checks passed. Data is ready for training.")
    
    print("="*70)
