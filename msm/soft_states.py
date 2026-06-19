#!/usr/bin/env python3
"""
Soft state assignments using Hidden Markov Models.

Produces per-frame soft assignments (state probabilities) and entropy.
"""
import argparse
import numpy as np
import json
from pathlib import Path
from hmmlearn import hmm


def fit_hmm_from_msm(dtraj, n_states, n_iter=100, seed=42):
    """
    Fit HMM initialized from MSM discrete trajectory.
    
    Args:
        dtraj: Discrete trajectory
        n_states: Number of states
        n_iter: Number of EM iterations
        seed: Random seed
        
    Returns:
        model: Fitted HMM
    """
    # Reshape for hmmlearn
    X = dtraj.reshape(-1, 1)
    
    # Initialize HMM
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type='diag',
        n_iter=n_iter,
        random_state=seed
    )
    
    # Fit
    model.fit(X)
    
    return model


def compute_soft_assignments(model, dtraj):
    """
    Compute soft state assignments (posterior probabilities).
    
    Args:
        model: Fitted HMM
        dtraj: Discrete trajectory
        
    Returns:
        soft_dtraj: Soft assignments (T x n_states)
    """
    X = dtraj.reshape(-1, 1)
    soft_dtraj = model.predict_proba(X)
    return soft_dtraj


def compute_state_entropy(soft_dtraj, epsilon=1e-12):
    """
    Compute per-frame state entropy.
    
    H_t = -Σ_i q_{t,i} log(q_{t,i})
    
    Args:
        soft_dtraj: Soft assignments (T x n_states)
        epsilon: Small constant for numerical stability
        
    Returns:
        entropy: Per-frame entropy
    """
    q = soft_dtraj + epsilon
    entropy = -np.sum(q * np.log(q), axis=1)
    return entropy


def main():
    parser = argparse.ArgumentParser(
        description='Compute soft state assignments using HMM'
    )
    parser.add_argument('--dtraj', required=True,
                       help='Path to discrete trajectory (dtraj.npy)')
    parser.add_argument('--n_states', type=int, default=None,
                       help='Number of states (auto-detect if not provided)')
    parser.add_argument('--output_soft', default='data/derived/soft_dtraj.npy',
                       help='Output file for soft assignments')
    parser.add_argument('--output_entropy', default='data/derived/state_entropy.npy',
                       help='Output file for state entropy')
    parser.add_argument('--output_meta', default='reports/soft_states_meta.json',
                       help='Output file for metadata')
    parser.add_argument('--n_iter', type=int, default=100,
                       help='Number of EM iterations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("[1/4] Loading discrete trajectory...")
    dtraj = np.load(args.dtraj)
    print(f"  Loaded: {len(dtraj)} frames")
    
    # Determine number of states
    if args.n_states is None:
        n_states = len(np.unique(dtraj))
        print(f"  Auto-detected: {n_states} states")
    else:
        n_states = args.n_states
        print(f"  Using: {n_states} states")
    
    print(f"[2/4] Fitting HMM...")
    print(f"  States: {n_states}")
    print(f"  EM iterations: {args.n_iter}")
    
    model = fit_hmm_from_msm(dtraj, n_states, args.n_iter, args.seed)
    print(f"  HMM fitted")
    
    print(f"[3/4] Computing soft assignments and entropy...")
    soft_dtraj = compute_soft_assignments(model, dtraj)
    entropy = compute_state_entropy(soft_dtraj)
    
    print(f"  Soft assignments shape: {soft_dtraj.shape}")
    print(f"  Entropy range: [{entropy.min():.3f}, {entropy.max():.3f}]")
    print(f"  Mean entropy: {entropy.mean():.3f} ± {entropy.std():.3f}")
    
    print(f"[4/4] Saving outputs...")
    
    # Save soft assignments
    output_soft = Path(args.output_soft)
    output_soft.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_soft, soft_dtraj)
    print(f"  Soft assignments saved to {output_soft}")
    
    # Save entropy
    output_entropy = Path(args.output_entropy)
    np.save(output_entropy, entropy)
    print(f"  State entropy saved to {output_entropy}")
    
    # Save metadata
    metadata = {
        'n_frames': len(dtraj),
        'n_states': n_states,
        'n_iter': args.n_iter,
        'seed': args.seed,
        'entropy_stats': {
            'mean': float(entropy.mean()),
            'std': float(entropy.std()),
            'min': float(entropy.min()),
            'max': float(entropy.max())
        }
    }
    
    output_meta = Path(args.output_meta)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    with open(output_meta, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {output_meta}")
    
    print("\nSoft states complete!")


if __name__ == '__main__':
    main()
