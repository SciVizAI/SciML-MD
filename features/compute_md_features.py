# ============================================================== #
#  Module:      features/compute_md_features.py
#  Description: Physics feature extraction — RMSD, Rg, native contacts, φ/ψ dihedrals per frame
#  Author:      Siya Jethliya
#  Copyright (c) 2026 SciVizAI — All rights reserved.
# ============================================================== #

#!/usr/bin/env python3
"""
Compute MD features from trajectory files.

Features computed:
- RMSD from reference structure
- Radius of gyration
- Number of native contacts
- Phi/Psi backbone dihedral angles (sin/cos encoded)
"""
import numpy as np
import mdtraj as md



# -------------------------------------------------------------- #
# Function: compute_features
# -------------------------------------------------------------- #
def compute_features(topology_path, trajectory_path, stride=1, reference_frame=0):
    """
    Compute MD features from a trajectory.
    
    Args:
        topology_path: Path to topology file (PDB)
        trajectory_path: Path to trajectory file (XTC, DCD, etc.)
        stride: Load every stride-th frame
        reference_frame: Frame index for RMSD reference
        
    Returns:
        features: Dictionary of feature arrays
        traj: MDTraj trajectory object
    """
    # Load trajectory against the (canonical) topology. Fail fast with a clear,
    # located message if the atom sets disagree — this is exactly where topology
    # drift historically surfaced as the opaque MDTraj error
    # "topology and trajectory files might not contain the same atoms".
    try:
        traj = md.load(trajectory_path, top=topology_path, stride=stride)
    except (ValueError, IOError) as exc:
        raise ValueError(
            f"Topology/trajectory mismatch: could not load '{trajectory_path}' "
            f"with topology '{topology_path}'. Feature extraction must use the "
            f"SAME canonical topology produced during simulation "
            f"(canonical_topology.pdb). Original error: {exc}"
        ) from exc

    # Explicit atom-count assertion (defensive; md.load also enforces equality).
    top_only = md.load(topology_path)
    if traj.n_atoms != top_only.n_atoms:
        raise ValueError(
            f"Topology drift detected: trajectory has {traj.n_atoms} atoms but "
            f"topology '{topology_path}' has {top_only.n_atoms}. These must match."
        )

    n_frames = len(traj)
    
    features = {}
    
    # 1. RMSD from reference frame
    ref = traj[reference_frame]
    features['rmsd'] = md.rmsd(traj, ref)
    
    # 2. Radius of gyration
    features['rg'] = md.compute_rg(traj)
    
    # 3. Native contacts (CA-CA pairs within 8 Angstroms)
    ca_atoms = traj.topology.select("name CA")
    if len(ca_atoms) > 1:
        ii, jj = np.triu_indices(len(ca_atoms), k=1)
        pairs = np.stack([ca_atoms[ii], ca_atoms[jj]], axis=1)
        distances = md.compute_distances(traj, pairs)
        # Count contacts (distance < 0.8 nm = 8 Angstroms)
        features['contacts'] = (distances < 0.8).sum(axis=1).astype(float)
    else:
        features['contacts'] = np.zeros(n_frames)
    
    # 4. Backbone dihedrals (phi, psi)
    try:
        _, phi = md.compute_phi(traj)
        _, psi = md.compute_psi(traj)
    except (ValueError, RuntimeError, KeyError):
        # ValueError: topology issues, RuntimeError: computation failures
        # KeyError: missing atom types
        phi = np.zeros((n_frames, 1))
        psi = np.zeros((n_frames, 1))
    
    # Sin/cos encoding of dihedrals
    if phi.size > 0:
        features['phi_sin'] = np.sin(phi).mean(axis=1)
        features['phi_cos'] = np.cos(phi).mean(axis=1)
    else:
        features['phi_sin'] = np.zeros(n_frames)
        features['phi_cos'] = np.zeros(n_frames)
    
    if psi.size > 0:
        features['psi_sin'] = np.sin(psi).mean(axis=1)
        features['psi_cos'] = np.cos(psi).mean(axis=1)
    else:
        features['psi_sin'] = np.zeros(n_frames)
        features['psi_cos'] = np.zeros(n_frames)
    
    return features, traj



# -------------------------------------------------------------- #
# Function: features_to_matrix
# -------------------------------------------------------------- #
def features_to_matrix(features, keys=None):
    """
    Convert feature dictionary to a feature matrix.
    
    Args:
        features: Dictionary of feature arrays
        keys: List of feature keys to use (default: all)
        
    Returns:
        X: Feature matrix (n_frames x n_features)
        keys: List of feature keys used
    """
    if keys is None:
        keys = list(features.keys())
    
    X = np.column_stack([features[k] for k in keys])
    return X, keys
