#!/usr/bin/env python3
"""
Signal computation module for dynamic hotspot detection.

This module provides clean interfaces for computing separate metric channels:
1. Dynamic anomaly scores (rarity, transition surprise, local density)
2. RMSF/stability scores (per-residue flexibility)
3. tICA importance scores (slow-mode contribution)

Each function is self-contained and can be used independently.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import warnings


def compute_rmsf_scores(
    topology_path: Union[str, Path],
    trajectory_path: Union[str, Path],
    selection: str = "name CA",
    align_selection: str = "name CA"
) -> np.ndarray:
    """
    Compute per-residue RMSF (Root Mean Square Fluctuation) from trajectory.
    
    RMSF measures the average positional fluctuation of each residue around
    its mean position. High RMSF indicates flexible/floppy regions, while
    low RMSF indicates rigid/stable regions.
    
    Physical Interpretation:
    - RMSF is analogous to crystallographic B-factors
    - Typically: RMSF < 1 Å (rigid), 1-3 Å (moderate), > 3 Å (flexible)
    - Surface loops and termini usually have high RMSF
    - Buried core and secondary structure have low RMSF
    
    Args:
        topology_path: Path to topology file (PDB)
        trajectory_path: Path to trajectory file (XTC, DCD, etc.)
        selection: Atom selection for RMSF calculation (default: CA atoms)
        align_selection: Atom selection for alignment (default: CA atoms)
    
    Returns:
        rmsf: RMSF values per residue [n_residues] in Angstroms
        
    Note:
        Requires MDTraj. The trajectory is aligned to remove global
        rotation/translation before computing RMSF.
    """
    try:
        import mdtraj as md
    except ImportError:
        raise ImportError(
            "MDTraj is required for RMSF computation. "
            "Install with: pip install mdtraj"
        )
    
    # Load trajectory
    traj = md.load(str(trajectory_path), top=str(topology_path))
    
    # Select atoms for RMSF calculation
    atoms = traj.topology.select(selection)
    
    if len(atoms) == 0:
        raise ValueError(f"No atoms found with selection '{selection}'")
    
    # Align trajectory to remove global motion
    align_atoms = traj.topology.select(align_selection)
    traj.superpose(traj, frame=0, atom_indices=align_atoms)
    
    # Compute RMSF
    # RMSF = sqrt(mean((r_i(t) - <r_i>)^2))
    positions = traj.xyz[:, atoms, :]  # [n_frames, n_atoms, 3]
    mean_pos = positions.mean(axis=0)  # [n_atoms, 3]
    
    # Compute squared deviations
    squared_dev = np.sum((positions - mean_pos)**2, axis=2)  # [n_frames, n_atoms]
    
    # RMSF per atom
    rmsf_atoms = np.sqrt(squared_dev.mean(axis=0))  # [n_atoms]
    
    # Convert from nm to Angstroms (MDTraj uses nm)
    rmsf_atoms = rmsf_atoms * 10.0
    
    # Map atoms to residues (if multiple atoms per residue, take mean)
    topology = traj.topology
    residue_indices = [topology.atom(i).residue.index for i in atoms]
    
    n_residues = max(residue_indices) + 1
    rmsf_residues = np.zeros(n_residues)
    counts = np.zeros(n_residues)
    
    for atom_idx, res_idx in enumerate(residue_indices):
        rmsf_residues[res_idx] += rmsf_atoms[atom_idx]
        counts[res_idx] += 1
    
    # Average over atoms in each residue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rmsf_residues = np.divide(
            rmsf_residues, counts, 
            out=np.zeros_like(rmsf_residues),
            where=counts > 0
        )
    
    return rmsf_residues


def _get_adaptive_parameters(n_frames: int, k_neighbors: int, lag_msm: int) -> tuple:
    """
    Get adaptive parameters based on trajectory size for optimal performance.
    
    Automatically adjusts parameters for trajectories from 10 to 10,000,000+ frames.
    
    Args:
        n_frames: Number of frames in trajectory
        k_neighbors: User-specified k_neighbors (or default)
        lag_msm: User-specified lag_msm (or default)
        
    Returns:
        (adjusted_k_neighbors, adjusted_lag_msm, use_subsampling, subsample_size)
    """
    # Thresholds for adaptive behavior
    SMALL_TRAJ = 500          # Use robust mode
    MEDIUM_TRAJ = 10_000      # Standard processing
    LARGE_TRAJ = 50_000       # Use optimizations
    VERY_LARGE_TRAJ = 200_000 # Use subsampling for k-NN
    
    use_subsampling = False
    subsample_size = n_frames
    
    if n_frames < SMALL_TRAJ:
        # Small trajectory: use conservative settings
        adjusted_k = min(k_neighbors, max(5, n_frames // 20))
        adjusted_lag = min(lag_msm, max(1, n_frames // 10))
    elif n_frames < MEDIUM_TRAJ:
        # Medium trajectory: standard settings
        adjusted_k = min(k_neighbors, n_frames // 5)
        adjusted_lag = min(lag_msm, n_frames // 20)
    elif n_frames < LARGE_TRAJ:
        # Large trajectory: increase k for better statistics
        adjusted_k = min(max(k_neighbors, 30), n_frames // 10)
        adjusted_lag = min(max(lag_msm, 50), n_frames // 50)
    elif n_frames < VERY_LARGE_TRAJ:
        # Very large: start considering efficiency
        adjusted_k = min(max(k_neighbors, 50), 100)
        adjusted_lag = min(max(lag_msm, 50), 100)
    else:
        # Extremely large (>200K frames): use subsampling for k-NN
        adjusted_k = min(max(k_neighbors, 50), 100)
        adjusted_lag = min(max(lag_msm, 100), 200)
        use_subsampling = True
        # Subsample to ~50K points for k-NN fitting, but query all points
        subsample_size = min(50_000, n_frames // 4)
    
    return adjusted_k, adjusted_lag, use_subsampling, subsample_size


def compute_dynamic_anomaly_scores(
    msm,
    dtraj: np.ndarray,
    tica_coords: np.ndarray,
    lag_msm: int = 30,
    k_neighbors: int = 20,
    normalize: bool = True,
    auto_optimize: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute dynamic anomaly signals from MSM and tICA projection.
    
    This function computes three complementary signals that detect
    unusual or rare dynamics:
    
    1. State Rarity: How rare is the current state in equilibrium?
       - Based on MSM stationary distribution π
       - High values indicate rarely-visited states
       
    2. Transition Surprise: How unexpected is the observed transition?
       - Based on MSM transition matrix P
       - High values indicate rare or forbidden transitions
       
    3. Local Density: How isolated is this conformation structurally?
       - Based on k-NN distance in tICA space
       - High values (low density) indicate structural outliers
    
    Physical Interpretation:
    - State rarity: Thermodynamic perspective (equilibrium populations)
    - Transition surprise: Kinetic perspective (barrier heights)
    - Local density: Geometric perspective (structural uniqueness)
    
    Automatic Optimization:
    - For small trajectories (<500 frames): Uses conservative parameters
    - For medium trajectories (500-10K): Standard processing
    - For large trajectories (10K-50K): Optimized k-NN settings
    - For very large trajectories (>50K): Uses subsampling for O(n) k-NN
    
    Args:
        msm: Fitted Markov State Model (deeptime or PyEMMA)
        dtraj: Discrete trajectory (state assignments) [n_frames]
        tica_coords: tICA-projected coordinates [n_frames, n_dims]
        lag_msm: MSM lag time (frames)
        k_neighbors: Number of neighbors for density estimation
        normalize: If True, normalize each signal to [0,1] using rank
        auto_optimize: If True, automatically optimize parameters for trajectory size
        
    Returns:
        signals: Dictionary with keys:
            - 'rarity': State rarity signal [n_frames]
            - 'transition_surprise': Transition surprise signal [n_frames]
            - 'local_density': Local density signal (inverted) [n_frames]
            
    Note:
        All signals are oriented so that higher values = more anomalous.
        
    Raises:
        ValueError: If trajectory length is less than lag_msm + 1
    """
    from sklearn.neighbors import NearestNeighbors
    
    n_frames = len(dtraj)
    
    # Validate inputs
    if n_frames < 2:
        raise ValueError(f"Trajectory too short: {n_frames} frames (need at least 2)")
    
    # Auto-optimize parameters based on trajectory size
    if auto_optimize:
        k_neighbors, lag_msm, use_subsampling, subsample_size = _get_adaptive_parameters(
            n_frames, k_neighbors, lag_msm
        )
        if n_frames >= 10_000:
            print(f"  [Auto-optimize] n_frames={n_frames:,}, k={k_neighbors}, lag={lag_msm}"
                  f"{', subsampling=' + str(subsample_size) if use_subsampling else ''}")
    else:
        use_subsampling = False
        subsample_size = n_frames
    
    if n_frames <= lag_msm:
        warnings.warn(
            f"Trajectory length ({n_frames}) <= lag_msm ({lag_msm}). "
            f"Reducing lag_msm to {max(1, n_frames // 2)}."
        )
        lag_msm = max(1, n_frames // 2)
    
    signals = {}
    
    # Get MSM parameters with validation
    pi = msm.stationary_distribution
    P = msm.transition_matrix
    
    # Check for invalid probabilities
    if np.any(np.isnan(pi)) or np.any(np.isinf(pi)):
        warnings.warn("NaN or Inf in stationary distribution. Using uniform.")
        pi = np.ones(msm.n_states) / msm.n_states
    
    if np.any(np.isnan(P)) or np.any(np.isinf(P)):
        warnings.warn("NaN or Inf in transition matrix. Renormalizing.")
        P = np.nan_to_num(P, nan=0.0, posinf=1.0, neginf=0.0)
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        P = P / row_sums
    n_states = msm.n_states
    
    # --- Signal 1: State Rarity ---
    # rarity = 1 - π[state]
    # Vectorized for speed
    rarity = np.ones(n_frames, dtype=np.float64)
    valid_states = (dtraj >= 0) & (dtraj < n_states)
    rarity[valid_states] = 1.0 - pi[dtraj[valid_states]]
    
    signals['rarity'] = rarity
    
    # --- Signal 2: Transition Surprise ---
    # surprise = -log(P[s_t -> s_{t+lag}])
    # Vectorized for speed
    surprise = np.zeros(n_frames, dtype=np.float64)
    epsilon = 1e-12  # Avoid log(0)
    
    if n_frames > lag_msm:
        s1 = dtraj[:-lag_msm]
        s2 = dtraj[lag_msm:]
        valid = (s1 >= 0) & (s1 < n_states) & (s2 >= 0) & (s2 < n_states)
        
        # Clip indices to valid range before indexing
        s1_clipped = np.clip(s1, 0, n_states - 1)
        s2_clipped = np.clip(s2, 0, n_states - 1)
        
        probs = np.where(valid, np.maximum(P[s1_clipped, s2_clipped], epsilon), 1.0)
        surprise[:-lag_msm] = -np.log(probs)
    
    signals['transition_surprise'] = surprise
    
    # --- Signal 3: Local Density ---
    # Use k-NN distance as proxy for local density
    # Higher distance = lower density = more anomalous
    k = min(k_neighbors, len(tica_coords) - 1)
    if k < 1:
        signals['local_density'] = np.zeros(n_frames)
    elif use_subsampling and n_frames > subsample_size:
        # For very large trajectories, fit on subsample but query all points
        # This reduces complexity from O(n²) to O(n × subsample_size)
        np.random.seed(42)  # Reproducibility
        subsample_idx = np.random.choice(n_frames, subsample_size, replace=False)
        subsample_coords = tica_coords[subsample_idx]
        
        # Fit on subsample
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=-1).fit(subsample_coords)
        
        # Query all points against the subsample
        distances, _ = nbrs.kneighbors(tica_coords)
        mean_dist = distances.mean(axis=1)
        signals['local_density'] = mean_dist
    else:
        # Standard k-NN for smaller trajectories
        # Use ball_tree algorithm which is O(n log n) for queries
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=-1).fit(tica_coords)
        distances, _ = nbrs.kneighbors(tica_coords)
        
        # Mean distance to k nearest neighbors
        # Invert sign so high distance = high anomaly score
        mean_dist = distances.mean(axis=1)
        signals['local_density'] = mean_dist  # Already oriented correctly
    
    # Normalize signals to [0,1] if requested
    if normalize:
        for key in signals:
            signals[key] = _rank_normalize(signals[key])
    
    return signals


def compute_tica_importance_scores(
    tica_model,
    feature_names: Optional[list] = None,
    aggregate_by_residue: bool = True
) -> Union[np.ndarray, Dict[int, float]]:
    """
    Compute per-residue importance scores from tICA component loadings.
    
    tICA identifies slow collective motions. Residues with high loadings
    on the slowest components contribute most to these functionally-relevant
    slow modes.
    
    Physical Interpretation:
    - High scores indicate residues that drive slow collective motions
    - These are often hinge residues, allosteric nodes, or domain linkers
    - Complementary to RMSF: a residue can be rigid but still important
      for slow modes (e.g., a hinge with small but critical rotation)
    
    Args:
        tica_model: Fitted tICA model (deeptime or PyEMMA)
        feature_names: List of feature names (e.g., ['phi_1', 'psi_1', ...])
        aggregate_by_residue: If True, aggregate feature loadings by residue
        
    Returns:
        If aggregate_by_residue:
            importance: Dict mapping residue_id -> importance score
        Else:
            loadings: Raw feature loadings [n_features, n_components]
            
    Note:
        Importance is computed as the L2 norm of loadings across the
        top slow components (typically first 3-5 components).
    """
    # Get component loadings (eigenvectors)
    # Shape: [n_features, n_components]
    try:
        # deeptime format
        loadings = tica_model.eigenvectors
    except AttributeError:
        try:
            # PyEMMA format
            loadings = tica_model.eigenvectors_
        except AttributeError:
            raise AttributeError(
                "Could not find eigenvectors in tICA model. "
                "Ensure model is fitted."
            )
    
    n_features, n_components = loadings.shape
    
    # Use top components (slowest modes)
    # Typically first 3-5 components are most important
    n_top = min(5, n_components)
    top_loadings = loadings[:, :n_top]
    
    if not aggregate_by_residue:
        return top_loadings
    
    # Aggregate by residue
    # Assume feature names follow pattern: 'feature_type_residue_id'
    # e.g., 'phi_10', 'psi_10', 'contact_10_20'
    
    if feature_names is None:
        # If no feature names, assume features are ordered by residue
        # and each residue has same number of features
        warnings.warn(
            "No feature names provided. Assuming sequential residue ordering."
        )
        n_features_per_res = 2  # phi, psi
        n_residues = n_features // n_features_per_res
        
        importance_dict = {}
        for res_id in range(n_residues):
            start_idx = res_id * n_features_per_res
            end_idx = start_idx + n_features_per_res
            
            # L2 norm of loadings for this residue's features
            res_loadings = top_loadings[start_idx:end_idx, :]
            importance = np.sqrt(np.sum(res_loadings**2))
            importance_dict[res_id] = float(importance)
        
        return importance_dict
    
    # Parse feature names to extract residue IDs
    residue_contributions = {}
    
    for feat_idx, feat_name in enumerate(feature_names):
        # Extract residue ID(s) from feature name
        # Common patterns: 'phi_10', 'psi_25', 'dist_10_25'
        parts = feat_name.split('_')
        
        try:
            # Try to find numeric parts (residue IDs)
            res_ids = [int(p) for p in parts if p.isdigit()]
            
            if len(res_ids) == 0:
                continue
            
            # Get loading magnitude for this feature
            loading_magnitude = np.sqrt(np.sum(top_loadings[feat_idx, :]**2))
            
            # Distribute contribution to all residues involved in this feature
            for res_id in res_ids:
                if res_id not in residue_contributions:
                    residue_contributions[res_id] = 0.0
                residue_contributions[res_id] += loading_magnitude
        
        except (ValueError, IndexError):
            continue
    
    # Normalize to [0, 1]
    if len(residue_contributions) > 0:
        max_contrib = max(residue_contributions.values())
        if max_contrib > 0:
            residue_contributions = {
                k: v / max_contrib 
                for k, v in residue_contributions.items()
            }
    
    return residue_contributions


def normalize_scores(
    scores: np.ndarray,
    method: str = 'rank',
    low_percentile: float = 0.05,
    high_percentile: float = 0.95,
    per_frame: bool = False
) -> np.ndarray:
    """
    Normalize scores using various strategies.
    
    Normalization is critical for visualization. Different strategies
    offer different trade-offs:
    
    - **rank**: Robust, preserves ordering exactly, uniform output distribution
    - **percentile**: Focuses on bulk, clips extreme outliers
    - **zscore**: Gaussian assumption, sensitive to outliers
    
    Global vs. Per-Frame:
    - **global**: Compare across entire trajectory (default)
    - **per_frame**: Normalize within each frame independently
    
    Args:
        scores: Input scores [n_frames] or [n_frames, n_residues]
        method: Normalization method ('rank', 'percentile', 'zscore')
        low_percentile: Lower clip percentile (for 'percentile' method)
        high_percentile: Upper clip percentile (for 'percentile' method)
        per_frame: If True and scores are 2D, normalize each frame independently
        
    Returns:
        normalized: Normalized scores in [0, 1]
    """
    if method == 'rank':
        if per_frame and scores.ndim == 2:
            # Normalize each frame
            normalized = np.zeros_like(scores)
            for i in range(scores.shape[0]):
                normalized[i, :] = _rank_normalize(scores[i, :])
            return normalized
        else:
            return _rank_normalize(scores.flatten()).reshape(scores.shape)
    
    elif method == 'percentile':
        if per_frame and scores.ndim == 2:
            normalized = np.zeros_like(scores)
            for i in range(scores.shape[0]):
                normalized[i, :] = _percentile_normalize(
                    scores[i, :], low_percentile, high_percentile
                )
            return normalized
        else:
            return _percentile_normalize(
                scores.flatten(), low_percentile, high_percentile
            ).reshape(scores.shape)
    
    elif method == 'zscore':
        if per_frame and scores.ndim == 2:
            normalized = np.zeros_like(scores)
            for i in range(scores.shape[0]):
                z = _compute_zscore(scores[i, :])
                # Map to [0, 1] using sigmoid
                normalized[i, :] = 1.0 / (1.0 + np.exp(-z))
            return normalized
        else:
            z = _compute_zscore(scores.flatten())
            return (1.0 / (1.0 + np.exp(-z))).reshape(scores.shape)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def aggregate_frame_to_residue(
    frame_scores: np.ndarray,
    per_residue_contributions: np.ndarray,
    method: str = 'weighted_mean'
) -> np.ndarray:
    """
    Aggregate per-frame anomaly scores to per-residue scores.
    
    This mapping answers: "Which residues are most involved in anomalous frames?"
    
    Methods:
    - **weighted_mean**: Weight frame scores by residue's contribution in that frame
    - **max**: Maximum anomaly score this residue achieved
    - **mean**: Average anomaly across frames where residue is active
    
    Args:
        frame_scores: Anomaly scores per frame [n_frames]
        per_residue_contributions: Contribution matrix [n_frames, n_residues]
        method: Aggregation method
        
    Returns:
        residue_scores: Per-residue scores [n_residues]
    """
    n_frames, n_residues = per_residue_contributions.shape
    
    if method == 'weighted_mean':
        # Weight frame scores by residue contribution
        weighted = frame_scores[:, np.newaxis] * per_residue_contributions
        residue_scores = weighted.sum(axis=0) / (per_residue_contributions.sum(axis=0) + 1e-10)
    
    elif method == 'max':
        # Maximum score achieved
        weighted = frame_scores[:, np.newaxis] * per_residue_contributions
        residue_scores = weighted.max(axis=0)
    
    elif method == 'mean':
        # Mean score where residue contributes
        mask = per_residue_contributions > 0
        weighted = frame_scores[:, np.newaxis] * mask
        counts = mask.sum(axis=0)
        residue_scores = weighted.sum(axis=0) / (counts + 1e-10)
    
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    return residue_scores


# ============================================================================
# Helper functions (internal use)
# ============================================================================

def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Rank-based normalization to [0,1]."""
    x = np.asarray(x, dtype=np.float64)
    
    if len(x) == 0:
        return x
    
    if np.all(x == x[0]):
        return np.zeros_like(x)
    
    ranks = np.argsort(np.argsort(x))
    return ranks / (len(x) - 1)


def _percentile_normalize(
    x: np.ndarray,
    lower: float = 0.05,
    upper: float = 0.95
) -> np.ndarray:
    """Percentile-based normalization with clipping."""
    x = np.asarray(x, dtype=np.float64)
    
    if len(x) == 0:
        return x
    
    q_low = np.quantile(x, lower)
    q_high = np.quantile(x, upper)
    
    if q_high <= q_low:
        return np.zeros_like(x)
    
    return np.clip((x - q_low) / (q_high - q_low), 0, 1)


def _compute_zscore(x: np.ndarray) -> np.ndarray:
    """Compute z-scores."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x)
    std = np.std(x)
    
    if std == 0:
        return np.zeros_like(x)
    
    return (x - mean) / std
