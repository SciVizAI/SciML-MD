#!/usr/bin/env python3
"""
Enhanced anomaly scoring v2 with multi-signal fusion.

Combines kinetic, structural, and energetic signals:
- Kinetic: rarity, transition surprise
- Structural: local density, soft entropy (optional)
- Energetic: energy stress, pocket volatility

Provides rank/quantile normalization and windowed smoothing.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy.ndimage import median_filter


def rank_normalize(x):
    """
    Normalize to [0,1] using rank-based scaling.
    Robust to outliers and heavy tails.
    
    Args:
        x: Input array
        
    Returns:
        Normalized array in [0,1]
    """
    x = np.asarray(x, dtype=np.float64)
    
    if len(x) == 0:
        return x
    
    # Handle constant arrays
    if np.all(x == x[0]):
        return np.zeros_like(x)
    
    # Rank-based normalization
    ranks = np.argsort(np.argsort(x))
    return ranks / (len(x) - 1)


def quantile_normalize(x, lower=0.01, upper=0.99):
    """
    Normalize using quantiles (robust to outliers).
    
    Args:
        x: Input array
        lower: Lower quantile
        upper: Upper quantile
        
    Returns:
        Normalized array in [0,1]
    """
    x = np.asarray(x, dtype=np.float64)
    
    if len(x) == 0:
        return x
    
    q_low = np.quantile(x, lower)
    q_high = np.quantile(x, upper)
    
    if q_high <= q_low:
        return np.zeros_like(x)
    
    return np.clip((x - q_low) / (q_high - q_low), 0, 1)


def compute_zscore(x):
    """Compute z-score with robust handling."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x)
    std = np.std(x)
    
    if std == 0:
        return np.zeros_like(x)
    
    return (x - mean) / std


def moving_median(x, window=5):
    """
    Apply moving median filter for smoothing.
    
    Args:
        x: Input array
        window: Window size (odd number preferred)
        
    Returns:
        Smoothed array
    """
    if window < 2:
        return x
    
    # Ensure window is odd
    if window % 2 == 0:
        window += 1
    
    return median_filter(x, size=window, mode='nearest')


def load_signal_data(features_path, vamp2_best_path, energy_path=None, 
                    pockets_path=None, soft_dtraj_path=None, 
                    state_entropy_path=None):
    """
    Load all signal data sources.
    
    Returns:
        dict with signal arrays
    """
    data = {}
    
    # Load basic features (for TICA projection)
    if Path(features_path).exists():
        X = np.load(features_path)
        data['features'] = X
        data['n_frames'] = len(X)
    else:
        raise FileNotFoundError(f"Features not found: {features_path}")
    
    # Load VAMP-2 selected parameters
    if Path(vamp2_best_path).exists():
        with open(vamp2_best_path) as f:
            vamp2_params = json.load(f)
        data['lag_tica'] = vamp2_params['lag']
        data['dim_tica'] = vamp2_params['dim']
    else:
        # Use defaults
        data['lag_tica'] = 10
        data['dim_tica'] = 5
    
    # Load energy features (optional)
    if energy_path and Path(energy_path).exists():
        df_energy = pd.read_parquet(energy_path)
        data['energy_df'] = df_energy
    
    # Load pocket features (optional)
    if pockets_path and Path(pockets_path).exists():
        df_pockets = pd.read_parquet(pockets_path)
        data['pockets_df'] = df_pockets
    
    # Load soft states (optional)
    if soft_dtraj_path and Path(soft_dtraj_path).exists():
        data['soft_dtraj'] = np.load(soft_dtraj_path)
    
    if state_entropy_path and Path(state_entropy_path).exists():
        data['state_entropy'] = np.load(state_entropy_path)
    
    return data


def compute_kinetic_signals(msm, dtraj, lag_msm):
    """
    Compute kinetic signals: rarity and transition surprise.
    
    Args:
        msm: Fitted MSM model
        dtraj: Discrete trajectory (mapped to active set)
        lag_msm: MSM lag time
        
    Returns:
        rarity: State rarity signal
        surprise: Transition surprise signal
    """
    n_frames = len(dtraj)
    pi = msm.stationary_distribution
    P = msm.transition_matrix
    n_states = msm.n_states
    
    # Rarity: 1 - π[state]
    rarity = np.ones(n_frames, dtype=np.float64)
    for t in range(n_frames):
        s = dtraj[t]
        if 0 <= s < n_states:
            rarity[t] = 1.0 - pi[s]
    
    # Transition surprise: -log(P[s_t -> s_{t+lag}])
    surprise = np.zeros(n_frames, dtype=np.float64)
    epsilon = 1e-12
    
    for t in range(n_frames - lag_msm):
        s1, s2 = dtraj[t], dtraj[t + lag_msm]
        if 0 <= s1 < n_states and 0 <= s2 < n_states:
            prob = max(P[s1, s2], epsilon)
            surprise[t] = -np.log(prob)
    
    return rarity, surprise


def compute_local_density_signal(Y, k=20):
    """
    Compute local density signal using k-NN distance.
    
    Args:
        Y: TICA coordinates (T x d)
        k: Number of neighbors
        
    Returns:
        density: Local density signal (lower = more anomalous)
    """
    from sklearn.neighbors import NearestNeighbors
    
    k = min(k, len(Y) - 1)
    if k < 1:
        return np.zeros(len(Y))
    
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(Y)
    distances, _ = nbrs.kneighbors(Y)
    
    # Return negative mean distance (inverted for consistency)
    return -distances.mean(axis=1)


def compute_energy_stress(energy_df, n_frames, method='sum', top_k=None):
    """
    Compute energetic stress signal from per-residue energies.
    
    Args:
        energy_df: DataFrame with columns [frame, res_id, energy]
        n_frames: Total number of frames
        method: 'sum', 'mean', or 'top_k'
        top_k: For top_k method, number of highest-energy residues
        
    Returns:
        energy_stress: Energy stress signal per frame
    """
    if energy_df is None or len(energy_df) == 0:
        return np.zeros(n_frames)
    
    stress = np.zeros(n_frames, dtype=np.float64)
    
    for frame in range(n_frames):
        frame_data = energy_df[energy_df['frame'] == frame]
        
        if len(frame_data) == 0:
            continue
        
        energies = frame_data['energy'].values
        
        if method == 'sum':
            stress[frame] = energies.sum()
        elif method == 'mean':
            stress[frame] = energies.mean()
        elif method == 'top_k' and top_k:
            # Use top-k highest (most unfavorable) energies
            top_energies = np.sort(energies)[-top_k:]
            stress[frame] = top_energies.mean()
        else:
            stress[frame] = energies.sum()
    
    return stress


def compute_pocket_volatility(pockets_df, n_frames, lag=1, metric='volume'):
    """
    Compute pocket volatility: frame-to-frame changes in pocket metrics.
    
    Args:
        pockets_df: DataFrame with pocket features
        n_frames: Total number of frames
        lag: Lag for computing changes
        metric: 'volume' or 'mouth_radius'
        
    Returns:
        volatility: Pocket volatility signal
    """
    if pockets_df is None or len(pockets_df) == 0:
        return np.zeros(n_frames)
    
    # Aggregate metric per frame (sum over all pockets)
    frame_metric = np.zeros(n_frames, dtype=np.float64)
    
    for frame in range(n_frames):
        frame_data = pockets_df[pockets_df['frame'] == frame]
        if len(frame_data) > 0:
            frame_metric[frame] = frame_data[metric].sum()
    
    # Compute absolute changes
    volatility = np.zeros(n_frames, dtype=np.float64)
    for t in range(lag, n_frames):
        volatility[t] = np.abs(frame_metric[t] - frame_metric[t - lag])
    
    return volatility


def fuse_signals(signals, method='median', normalize_method='rank'):
    """
    Fuse multiple normalized signals into final score.
    
    Args:
        signals: Dict of signal_name -> array
        method: 'median' or 'mean'
        normalize_method: 'rank' or 'quantile'
        
    Returns:
        score_raw: Raw fused score in [0,1]
        normalized_signals: Dict of normalized signals
    """
    # Normalize each signal
    normalized = {}
    
    for name, signal in signals.items():
        if normalize_method == 'rank':
            normalized[name] = rank_normalize(signal)
        elif normalize_method == 'quantile':
            normalized[name] = quantile_normalize(signal)
        else:
            # Fallback to rank
            normalized[name] = rank_normalize(signal)
    
    # Stack signals
    signal_matrix = np.column_stack([normalized[name] for name in signals.keys()])
    
    # Fuse
    if method == 'median':
        score_raw = np.median(signal_matrix, axis=1)
    elif method == 'mean':
        score_raw = np.mean(signal_matrix, axis=1)
    else:
        score_raw = np.median(signal_matrix, axis=1)
    
    return score_raw, normalized


def compute_anomaly_scores_v2(data, msm, dtraj, Y, 
                              config=None):
    """
    Compute enhanced anomaly scores with multi-signal fusion.
    
    Args:
        data: Dict with loaded data
        msm: Fitted MSM
        dtraj: Discrete trajectory
        Y: TICA coordinates
        config: Configuration dict
        
    Returns:
        scores_df: DataFrame with scores and components
        summary: Dict with scoring parameters
    """
    if config is None:
        config = {}
    
    n_frames = data['n_frames']
    lag_msm = config.get('lag_msm', 30)
    k_neighbors = config.get('k_neighbors', 20)
    window_size = config.get('window_size', 5)
    normalize_method = config.get('normalize_method', 'rank')
    fusion_method = config.get('fusion_method', 'median')
    
    # Collect signals
    signals = {}
    
    print("[1/7] Computing kinetic signals...")
    rarity, surprise = compute_kinetic_signals(msm, dtraj, lag_msm)
    signals['rarity'] = rarity
    signals['transition_surprise'] = surprise
    
    print("[2/7] Computing local density...")
    density = compute_local_density_signal(Y, k=k_neighbors)
    signals['local_density'] = -density  # Invert so low density = high score
    
    print("[3/7] Computing soft entropy (if available)...")
    if 'state_entropy' in data:
        signals['soft_entropy'] = data['state_entropy']
    
    print("[4/7] Computing energy stress (if available)...")
    if 'energy_df' in data:
        energy_stress = compute_energy_stress(data['energy_df'], n_frames, 
                                              method='sum')
        signals['energy_stress'] = energy_stress
    
    print("[5/7] Computing pocket volatility (if available)...")
    if 'pockets_df' in data:
        pocket_vol = compute_pocket_volatility(data['pockets_df'], n_frames,
                                              lag=1, metric='volume')
        signals['pocket_volatility'] = pocket_vol
    
    print("[6/7] Fusing signals...")
    score_raw, normalized_signals = fuse_signals(signals, 
                                                 method=fusion_method,
                                                 normalize_method=normalize_method)
    
    # Scale to [0, 100]
    score_raw_100 = score_raw * 100.0
    
    print("[7/7] Applying windowed smoothing...")
    score_windowed = moving_median(score_raw_100, window=window_size)
    
    # Create output DataFrame
    df_data = {
        'frame': np.arange(n_frames),
        'score_raw': score_raw_100,
        'score_windowed': score_windowed
    }
    
    # Add component columns (normalized)
    for name, signal in normalized_signals.items():
        df_data[f'component_{name}'] = signal * 100.0
    
    scores_df = pd.DataFrame(df_data)
    
    # Summary
    summary = {
        'n_frames': n_frames,
        'n_signals': len(signals),
        'signals': list(signals.keys()),
        'normalize_method': normalize_method,
        'fusion_method': fusion_method,
        'window_size': window_size,
        'lag_msm': lag_msm,
        'k_neighbors': k_neighbors,
        'score_stats': {
            'raw_mean': float(score_raw_100.mean()),
            'raw_std': float(score_raw_100.std()),
            'raw_min': float(score_raw_100.min()),
            'raw_max': float(score_raw_100.max()),
            'windowed_mean': float(score_windowed.mean()),
            'windowed_std': float(score_windowed.std())
        }
    }
    
    return scores_df, summary


def main():
    parser = argparse.ArgumentParser(
        description='Compute enhanced anomaly scores v2'
    )
    parser.add_argument('--features', required=True,
                       help='Path to features.npy')
    parser.add_argument('--msm_dir', required=True,
                       help='Directory with MSM outputs (dtraj.npy, etc.)')
    parser.add_argument('--vamp2_best', default='reports/vamp2_best.json',
                       help='Path to VAMP-2 best parameters')
    parser.add_argument('--energy', default=None,
                       help='Path to residue_energy.parquet (optional)')
    parser.add_argument('--pockets', default=None,
                       help='Path to pockets.parquet (optional)')
    parser.add_argument('--soft_dtraj', default=None,
                       help='Path to soft_dtraj.npy (optional)')
    parser.add_argument('--state_entropy', default=None,
                       help='Path to state_entropy.npy (optional)')
    parser.add_argument('--output_scores', default='data/derived/frame_scores_v2.csv',
                       help='Output CSV for scores')
    parser.add_argument('--output_summary', default='reports/scoring_v2_summary.json',
                       help='Output JSON for summary')
    parser.add_argument('--window', type=int, default=5,
                       help='Window size for smoothing')
    parser.add_argument('--normalize', choices=['rank', 'quantile'], default='rank',
                       help='Normalization method')
    parser.add_argument('--fusion', choices=['median', 'mean'], default='median',
                       help='Fusion method')
    
    args = parser.parse_args()
    
    print("[Phase 3] Enhanced Anomaly Scoring v2")
    print("="*70)
    
    # Load data
    print("\n[Loading data...]")
    data = load_signal_data(
        args.features,
        args.vamp2_best,
        args.energy,
        args.pockets,
        args.soft_dtraj,
        args.state_entropy
    )
    
    print(f"  Loaded {data['n_frames']} frames")
    print(f"  TICA params: lag={data['lag_tica']}, dim={data['dim_tica']}")
    
    # Load MSM outputs
    msm_dir = Path(args.msm_dir)
    dtraj = np.load(msm_dir / 'dtraj.npy')
    tica_coords = np.load(msm_dir / 'tica_coords.npy')
    
    # Reconstruct MSM (simplified - load saved parameters)
    from deeptime.markov.msm import MarkovStateModel
    P = np.load(msm_dir / 'P.npy')
    pi = np.load(msm_dir / 'pi.npy')
    msm = MarkovStateModel(P, stationary_distribution=pi)
    
    print(f"  MSM: {msm.n_states} states")
    print(f"  TICA: {tica_coords.shape}")
    
    # Configure scoring
    config = {
        'lag_msm': 30,
        'k_neighbors': 20,
        'window_size': args.window,
        'normalize_method': args.normalize,
        'fusion_method': args.fusion
    }
    
    # Compute scores
    print("\n[Computing anomaly scores v2...]")
    scores_df, summary = compute_anomaly_scores_v2(data, msm, dtraj, 
                                                   tica_coords, config)
    
    # Save outputs
    print("\n[Saving results...]")
    output_scores = Path(args.output_scores)
    output_scores.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(output_scores, index=False)
    print(f"  Scores saved to {output_scores}")
    
    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {output_summary}")
    
    print("\n[Summary]")
    print(f"  Signals used: {', '.join(summary['signals'])}")
    print(f"  Score range: [{summary['score_stats']['raw_min']:.1f}, "
          f"{summary['score_stats']['raw_max']:.1f}]")
    print(f"  Mean score: {summary['score_stats']['raw_mean']:.1f} ± "
          f"{summary['score_stats']['raw_std']:.1f}")
    
    print("\n" + "="*70)
    print("[Phase 3 Complete!]")


if __name__ == '__main__':
    main()
