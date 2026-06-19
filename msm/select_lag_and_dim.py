#!/usr/bin/env python3
"""
VAMP-2 based model selection for TICA lag time and dimensionality.

Performs a grid search over lag times and dimensions, evaluating each
combination using the VAMP-2 score to find optimal parameters.
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import yaml

from deeptime.decomposition import VAMP
from deeptime.util.data import timeshifted_split


def load_config(config_path='configs/pipeline.yaml'):
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_vamp2_score(X, lag, dim, validation_fraction=0.2, seed=42):
    """
    Compute VAMP-2 score for given lag and dimensionality.
    
    Args:
        X: Feature trajectory (T x F)
        lag: Lag time for VAMP
        dim: Number of dimensions
        validation_fraction: Fraction of data for validation
        seed: Random seed
        
    Returns:
        vamp2_score: VAMP-2 score (higher is better)
    """
    # Split into train and validation
    n_val = max(1, int(len(X) * validation_fraction))
    n_train = len(X) - n_val
    
    if n_train < lag + 10:  # Need enough data
        return -np.inf
    
    X_train = X[:n_train]
    X_val = X[n_train:]
    
    try:
        # Fit VAMP on training data
        vamp = VAMP(lagtime=lag, dim=dim).fit(X_train).fetch_model()
        
        # Evaluate on validation data
        if len(X_val) > lag + 1:
            # Transform validation data
            Y_val = vamp.transform(X_val)
            
            # Compute VAMP-2 score on validation
            # VAMP-2 is sum of squared singular values of C_01
            C_0 = np.cov(Y_val[:-lag].T)
            C_1 = np.cov(Y_val[lag:].T)
            C_01 = np.cov(Y_val[:-lag].T, Y_val[lag:].T)[:dim, dim:]
            
            # Regularize covariances
            reg = 1e-6
            C_0 += reg * np.eye(dim)
            C_1 += reg * np.eye(dim)
            
            # Compute VAMP-2 score
            C_0_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C_0)).T
            C_1_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C_1)).T
            
            K = C_0_inv_sqrt @ C_01 @ C_1_inv_sqrt
            s = np.linalg.svd(K, compute_uv=False)
            vamp2_score = np.sum(s**2)
            
            return float(vamp2_score)
        else:
            return -np.inf
            
    except Exception as e:
        print(f"  Warning: VAMP failed for lag={lag}, dim={dim}: {e}")
        return -np.inf


def select_lag_and_dim(features_path, output_dir, config_path=None):
    """
    Perform grid search to select optimal TICA lag and dimensionality.
    
    Args:
        features_path: Path to features.npy
        output_dir: Directory for output files
        config_path: Path to config YAML (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(config_path) if config_path else load_config()
    if config is None:
        # Default values
        lag_candidates = [5, 10, 15, 20, 30, 50]
        dim_candidates = [2, 3, 4, 5, 6, 8, 10]
        validation_fraction = 0.2
        seed = 42
    else:
        lag_candidates = config['tica']['lag_candidates']
        dim_candidates = config['tica']['dim_candidates']
        validation_fraction = config['model_selection']['validation_fraction']
        seed = config['seeds']['vamp']
    
    print(f"[1/3] Loading features from {features_path}")
    X = np.load(features_path)
    print(f"  Features shape: {X.shape}")
    
    print(f"[2/3] Grid search over {len(lag_candidates)} lags × {len(dim_candidates)} dims")
    
    results = []
    best_score = -np.inf
    best_params = None
    
    for lag, dim in product(lag_candidates, dim_candidates):
        print(f"  Testing lag={lag}, dim={dim}...", end=' ')
        
        score = compute_vamp2_score(X, lag, dim, validation_fraction, seed)
        
        results.append({
            'lag': lag,
            'dim': dim,
            'vamp2_score': score
        })
        
        print(f"score={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_params = (lag, dim)
    
    print(f"[3/3] Best parameters: lag={best_params[0]}, dim={best_params[1]}, score={best_score:.4f}")
    
    # Save results
    df = pd.DataFrame(results)
    df = df.sort_values('vamp2_score', ascending=False)
    
    grid_path = output_dir / 'vamp2_grid.csv'
    df.to_csv(grid_path, index=False)
    print(f"  Saved grid results to {grid_path}")
    
    # Save best parameters
    best_result = {
        'lag': int(best_params[0]),
        'dim': int(best_params[1]),
        'vamp2_score': float(best_score),
        'n_candidates': len(results),
        'features_shape': list(X.shape),
        'validation_fraction': validation_fraction,
        'seed': seed
    }
    
    best_path = output_dir / 'vamp2_best.json'
    with open(best_path, 'w') as f:
        json.dump(best_result, f, indent=2)
    print(f"  Saved best parameters to {best_path}")
    
    return best_params, best_score


def main():
    parser = argparse.ArgumentParser(
        description='Select optimal TICA lag and dimensionality using VAMP-2'
    )
    parser.add_argument('--features', required=True, 
                        help='Path to features.npy')
    parser.add_argument('--output_dir', default='reports',
                        help='Output directory for results')
    parser.add_argument('--config', default=None,
                        help='Path to pipeline.yaml config file')
    
    args = parser.parse_args()
    
    select_lag_and_dim(args.features, args.output_dir, args.config)


if __name__ == '__main__':
    main()
