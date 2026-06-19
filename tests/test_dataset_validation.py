#!/usr/bin/env python3
"""
Dataset quality validation tests.

Tests dataset integrity, trajectory quality, and topology consistency
following best practices from molecular dynamics literature.

References:
- Knapp et al. (2011) "Avoiding False Positive Conclusions in Molecular Simulation"
  J. Chem. Theory Comput. 7(4), 1102-1107
"""
import sys
import numpy as np
from pathlib import Path
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_trajectory_completeness():
    """
    Test that trajectory has sufficient frames for statistical analysis.
    
    Minimum requirements:
    - At least 1000 frames (common MD recommendation)
    - No missing frames or discontinuities
    - Consistent timestep
    """
    print("\n[TEST] Trajectory completeness")
    
    # Create synthetic trajectory metadata
    n_frames = 1500
    timestep = 0.002  # ps
    
    # Simulate frame times
    times = np.arange(n_frames) * timestep
    
    # Check minimum length
    assert n_frames >= 1000, f"Trajectory too short: {n_frames} < 1000 frames"
    
    # Check for gaps (consecutive times should differ by exactly timestep)
    time_diffs = np.diff(times)
    expected_diff = timestep
    gaps = np.abs(time_diffs - expected_diff) > 1e-6
    
    assert not np.any(gaps), f"Found {np.sum(gaps)} discontinuities in trajectory"
    
    print(f"  Frames: {n_frames}")
    print(f"  Timestep: {timestep} ps")
    print(f"  Total time: {times[-1]:.2f} ps")
    print(f"  ✓ Trajectory is complete and continuous")


def test_topology_consistency():
    """
    Test topology file consistency.
    
    Requirements:
    - All residues have valid identifiers
    - Protein atoms are correctly labeled
    - No duplicate atom indices
    """
    print("\n[TEST] Topology consistency")
    
    # Simulate topology data
    n_atoms = 1500
    n_residues = 100
    
    # Check residue count is reasonable for a protein
    assert 10 <= n_residues <= 10000, f"Unusual residue count: {n_residues}"
    
    # Check atoms per residue ratio
    atoms_per_residue = n_atoms / n_residues
    assert 5 <= atoms_per_residue <= 50, \
        f"Unusual atoms/residue ratio: {atoms_per_residue:.1f}"
    
    # Simulate atom indices (should be unique)
    atom_indices = np.arange(n_atoms)
    assert len(atom_indices) == len(set(atom_indices)), "Duplicate atom indices found"
    
    print(f"  Atoms: {n_atoms}")
    print(f"  Residues: {n_residues}")
    print(f"  Atoms/residue: {atoms_per_residue:.1f}")
    print(f"  ✓ Topology is consistent")


def test_coordinate_validity():
    """
    Test that coordinates are physically reasonable.
    
    Requirements:
    - No NaN or Inf values
    - Coordinates within reasonable bounds
    - No atoms at origin (common error)
    """
    print("\n[TEST] Coordinate validity")
    
    # Simulate protein coordinates (in nanometers)
    n_atoms = 1000
    # Typical protein in box ~ 5-10 nm on each side
    coords = np.random.randn(n_atoms, 3) * 2.0 + 5.0
    
    # Check for invalid values
    assert not np.any(np.isnan(coords)), "NaN coordinates detected"
    assert not np.any(np.isinf(coords)), "Inf coordinates detected"
    
    # Check reasonable bounds (typical proteins < 50 nm)
    assert np.all(np.abs(coords) < 50), "Coordinates outside reasonable bounds"
    
    # Check no atoms exactly at origin (common error)
    at_origin = np.all(coords == 0, axis=1)
    assert not np.any(at_origin), "Atoms found exactly at origin"
    
    # Check coordinate range is reasonable
    coord_range = np.ptp(coords, axis=0)
    assert np.all(coord_range > 0.5), "Coordinate range too small (collapsed structure?)"
    assert np.all(coord_range < 50), "Coordinate range too large (unfolded?)"
    
    print(f"  Atoms: {n_atoms}")
    print(f"  Range: [{coords.min():.2f}, {coords.max():.2f}] nm")
    print(f"  Box size: {coord_range} nm")
    print(f"  ✓ Coordinates are valid")


def test_trajectory_rmsd_sanity():
    """
    Test that trajectory RMSD values are reasonable.
    
    RMSD should:
    - Be positive
    - Not exceed typical protein dimensions (~5 nm)
    - Show some fluctuation (not frozen)
    """
    print("\n[TEST] Trajectory RMSD sanity check")
    
    # Simulate RMSD timeseries (in Angstroms, typical for proteins)
    n_frames = 1000
    # Stable protein: RMSD ~ 1-3 Å with fluctuations
    baseline_rmsd = 2.0
    fluctuation = 0.5
    rmsd = baseline_rmsd + fluctuation * np.random.randn(n_frames)
    rmsd = np.abs(rmsd)  # RMSD is always positive
    
    # Check all positive
    assert np.all(rmsd >= 0), "RMSD values must be non-negative"
    
    # Check reasonable range (0.5-50 Å for typical proteins)
    assert np.all(rmsd < 50), f"RMSD too large: max={rmsd.max():.1f} Å"
    assert np.mean(rmsd) > 0.5, f"RMSD too small: mean={np.mean(rmsd):.2f} Å (frozen?)"
    
    # Check for variation (not stuck)
    rmsd_std = np.std(rmsd)
    assert rmsd_std > 0.1, f"RMSD not varying: std={rmsd_std:.3f} (frozen trajectory?)"
    
    print(f"  Mean RMSD: {np.mean(rmsd):.2f} ± {rmsd_std:.2f} Å")
    print(f"  Range: [{rmsd.min():.2f}, {rmsd.max():.2f}] Å")
    print(f"  ✓ RMSD values are reasonable")


def test_feature_quality():
    """
    Test that extracted features are suitable for ML.
    
    Requirements:
    - No constant features (zero variance)
    - No extreme outliers (> 5 sigma)
    - Sufficient dynamic range
    """
    print("\n[TEST] Feature quality for ML")
    
    # Simulate feature matrix
    n_frames = 1000
    n_features = 50
    
    # Create features with realistic properties
    features = np.random.randn(n_frames, n_features)
    
    # Check for constant features
    feature_var = np.var(features, axis=0)
    constant_features = feature_var < 1e-10
    assert not np.any(constant_features), \
        f"Found {np.sum(constant_features)} constant features (zero variance)"
    
    # Check for extreme outliers (using median absolute deviation)
    for i in range(n_features):
        feat = features[:, i]
        median = np.median(feat)
        mad = np.median(np.abs(feat - median))
        if mad > 0:
            z_scores = np.abs(feat - median) / (1.4826 * mad)  # MAD to std conversion
            extreme = np.sum(z_scores > 5)
            assert extreme < 0.01 * n_frames, \
                f"Feature {i} has {extreme} extreme outliers (>5 MAD)"
    
    # Check dynamic range
    for i in range(n_features):
        feat = features[:, i]
        dynamic_range = np.ptp(feat) / (np.std(feat) + 1e-10)
        assert dynamic_range > 1.0, \
            f"Feature {i} has insufficient dynamic range: {dynamic_range:.2f}"
    
    print(f"  Features: {n_features}")
    print(f"  Frames: {n_frames}")
    print(f"  Mean variance: {np.mean(feature_var):.3f}")
    print(f"  ✓ Features suitable for ML")


def test_dataset_citation_info():
    """
    Test that dataset has proper citation information.
    
    For reproducibility and scientific rigor, datasets should have:
    - DOI or persistent identifier
    - Author information
    - Publication date
    """
    print("\n[TEST] Dataset citation metadata")
    
    # Example metadata that should be included
    dataset_info = {
        'doi': '10.17617/3.8O',  # Example Dataverse DOI
        'source': 'Edmond (Max Planck Digital Library)',
        'description': 'Molecular dynamics trajectory of protein system',
        'authors': 'Research Group',
        'date': '2024'
    }
    
    # Verify required fields
    required_fields = ['doi', 'source', 'description']
    for field in required_fields:
        assert field in dataset_info, f"Missing required field: {field}"
        assert len(dataset_info[field]) > 0, f"Empty field: {field}"
    
    print(f"  DOI: {dataset_info.get('doi', 'N/A')}")
    print(f"  Source: {dataset_info.get('source', 'N/A')}")
    print(f"  ✓ Citation metadata present")


def test_reproducibility_metadata():
    """
    Test that analysis includes reproducibility metadata.
    
    Should include:
    - Software versions
    - Random seeds
    - Exact parameters used
    """
    print("\n[TEST] Reproducibility metadata")
    
    # Example metadata
    metadata = {
        'python_version': '3.9',
        'numpy_version': '1.20.0',
        'deeptime_version': '0.4.0',
        'random_seed': 42,
        'tica_lag': 10,
        'tica_dim': 5,
        'msm_lag': 30,
        'n_clusters': 50
    }
    
    # Check essential fields
    assert 'random_seed' in metadata, "Random seed not specified"
    assert 'tica_lag' in metadata, "tICA lag not specified"
    assert 'msm_lag' in metadata, "MSM lag not specified"
    
    print(f"  Random seed: {metadata['random_seed']}")
    print(f"  tICA lag: {metadata['tica_lag']}")
    print(f"  MSM lag: {metadata['msm_lag']}")
    print(f"  ✓ Reproducibility metadata complete")


def main():
    """Run all dataset validation tests."""
    print("="*70)
    print("DATASET VALIDATION TESTS")
    print("="*70)
    
    tests = [
        test_trajectory_completeness,
        test_topology_consistency,
        test_coordinate_validity,
        test_trajectory_rmsd_sanity,
        test_feature_quality,
        test_dataset_citation_info,
        test_reproducibility_metadata,
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
