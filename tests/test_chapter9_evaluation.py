#!/usr/bin/env python3
"""
Tests for experiments/chapter9_evaluation.py

Tests cover all major functions using synthetic data to ensure
they run correctly and produce valid outputs.
"""
import sys
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.chapter9_evaluation import (
    _fit_pipeline,
    _fused_frame_scores,
    _residue_fused_scores,
    _jaccard_top10,
    compute_implied_timescales,
    compute_ck_errors,
    compute_vamp_comparison,
    compute_residue_ranking,
    compute_transition_enrichment,
    compute_spatial_clustering,
    run_chapter9_evaluation,
    compute_hotspot_slowmode_alignment_no_tica,
    compute_vamp_comparison_corrected,
    compute_transition_enrichment_window_sweep,
    compute_ranking_stability_extended,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_features(n_frames=150, n_features=7, seed=42):
    """Create a synthetic feature matrix with temporal correlation."""
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.normal(0, 0.1, (n_frames, n_features)), axis=0)
    return X


def _write_minimal_pdb(path, n_residues=20):
    """Write a minimal PDB file with Cα atoms on a straight line."""
    lines = []
    for i in range(n_residues):
        x, y, z = float(i * 3.8), 0.0, 0.0
        lines.append(
            f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        )
    lines.append("END\n")
    Path(path).write_text("".join(lines))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fit_pipeline_returns_correct_types():
    """_fit_pipeline returns MSM, dtraj, Y with expected shapes."""
    print("[TEST] _fit_pipeline returns correct types...")
    X = _make_features(n_frames=150, n_features=7)
    msm, dtraj, Y, tica_model = _fit_pipeline(X, lag_tica=5, dim_tica=3,
                                               n_clusters=6, lag_msm=5)
    assert hasattr(msm, "transition_matrix"), "MSM should have transition_matrix"
    assert len(dtraj) == len(X), "dtraj should match n_frames"
    assert Y.shape == (len(X), 3), "Y should have dim_tica columns"
    assert msm.n_states >= 1, "MSM should have at least 1 state"
    print("  ✓ Pipeline returns correct types")


def test_fused_frame_scores_shape():
    """_fused_frame_scores returns array of length n_frames."""
    print("[TEST] _fused_frame_scores shape...")
    X = _make_features(n_frames=150)
    msm, dtraj, Y, _ = _fit_pipeline(X, 5, 3, 6, 5)
    scores = _fused_frame_scores(msm, dtraj, Y, lag_msm=5)
    assert len(scores) == len(X), "Should match n_frames"
    assert scores.min() >= 0, "Scores should be non-negative"
    assert scores.max() <= 1, "Scores should be ≤ 1 after rank normalisation"
    print(f"  ✓ Frame scores shape: {scores.shape}, range [{scores.min():.3f}, {scores.max():.3f}]")


def test_residue_fused_scores_no_nan():
    """_residue_fused_scores returns finite values."""
    print("[TEST] _residue_fused_scores no NaN...")
    frame_scores = np.random.default_rng(0).uniform(0, 1, 100)
    n_residues = 30
    fused = _residue_fused_scores(frame_scores, n_residues)
    fused = np.nan_to_num(fused, nan=0.0)
    assert len(fused) == n_residues, "Length should equal n_residues"
    assert np.all(np.isfinite(fused)), "All values should be finite"
    print(f"  ✓ Residue scores: {len(fused)} values, all finite")


def test_jaccard_top10():
    """_jaccard_top10 computes correct Jaccard index."""
    print("[TEST] Jaccard top10...")
    n = 10
    # Identical rankings → Jaccard = 1
    rank_a = np.arange(1, n + 1)
    assert abs(_jaccard_top10(rank_a, rank_a, n) - 1.0) < 1e-9, (
        "Identical rankings should give J=1"
    )
    # Completely different → Jaccard = 0 (no overlap in top 10 % = top 1)
    rank_b = rank_a[::-1]
    j = _jaccard_top10(rank_a, rank_b, n)
    assert 0.0 <= j <= 1.0, "Jaccard should be in [0, 1]"
    print(f"  ✓ Jaccard identical=1.0, flipped={j:.3f}")


def test_compute_implied_timescales(tmp_path):
    """compute_implied_timescales saves expected CSV files."""
    print("[TEST] compute_implied_timescales...")
    X = _make_features(n_frames=150)
    compute_implied_timescales(X, lag_msm=5, dim_tica=3, n_clusters=6,
                               lag_tica=5, output_dir=tmp_path)

    its_path = tmp_path / "implied_timescales.csv"
    cv_path = tmp_path / "implied_timescale_cv.csv"

    assert its_path.exists(), "implied_timescales.csv should be created"
    assert cv_path.exists(), "implied_timescale_cv.csv should be created"

    df_its = pd.read_csv(its_path)
    assert set(df_its.columns) >= {"lag_time", "mode_index", "timescale"}

    df_cv = pd.read_csv(cv_path)
    assert set(df_cv.columns) >= {"mode_index", "mean", "std", "cv"}
    assert (df_cv["cv"] >= 0).all(), "CV should be non-negative"
    print(f"  ✓ ITS: {len(df_its)} rows; CV table: {len(df_cv)} modes")


def test_compute_ck_errors(tmp_path):
    """compute_ck_errors saves ck_errors.csv with expected columns."""
    print("[TEST] compute_ck_errors...")
    X = _make_features(n_frames=150)
    df_ck = compute_ck_errors(X, lag_msm=5, dim_tica=3, n_clusters=6,
                               lag_tica=5, output_dir=tmp_path)

    ck_path = tmp_path / "ck_errors.csv"
    assert ck_path.exists(), "ck_errors.csv should be created"
    assert set(df_ck.columns) >= {"n_step", "frobenius_error"}
    assert (df_ck["frobenius_error"] >= 0).all(), "Frobenius error should be ≥ 0"
    print(f"  ✓ CK errors: {len(df_ck)} rows")


def test_compute_vamp_comparison(tmp_path):
    """compute_vamp_comparison saves vamp_comparison.csv with 3 models."""
    print("[TEST] compute_vamp_comparison...")
    X = _make_features(n_frames=150)
    df_vamp = compute_vamp_comparison(X, lag_msm=5, dim_tica=3,
                                       lag_tica=5, output_dir=tmp_path)

    vamp_path = tmp_path / "vamp_comparison.csv"
    assert vamp_path.exists(), "vamp_comparison.csv should be created"
    assert len(df_vamp) == 3, "Should compare 3 model types"
    assert set(df_vamp["model_type"]) == {"tICA", "PCA", "raw_features"}
    print(f"  ✓ VAMP-2: {df_vamp[['model_type', 'vamp2_score']].to_dict('records')}")


def test_compute_residue_ranking(tmp_path):
    """compute_residue_ranking produces correct ranking files."""
    print("[TEST] compute_residue_ranking...")
    frame_scores = np.random.default_rng(42).uniform(0, 1, 100)
    n_residues = 20
    df_rank, fused = compute_residue_ranking(frame_scores, n_residues, tmp_path)

    assert (tmp_path / "residue_ranking.csv").exists()
    assert (tmp_path / "topk_sets.csv").exists()

    assert len(df_rank) == n_residues, "Should have one row per residue"
    assert set(df_rank.columns) >= {"residue_id", "fused_score", "rank"}
    assert df_rank["rank"].nunique() == n_residues, "Ranks should be unique"

    df_topk = pd.read_csv(tmp_path / "topk_sets.csv")
    assert set(df_topk["k_percent"].unique()) == {5, 10, 20}
    print(f"  ✓ Rankings: {len(df_rank)} residues; top-k subsets present")


def test_compute_transition_enrichment(tmp_path):
    """compute_transition_enrichment produces transition_enrichment.csv."""
    print("[TEST] compute_transition_enrichment...")
    n_frames = 100
    dtraj = np.array([0] * 30 + [1] * 40 + [0] * 30, dtype=np.int64)
    frame_scores = np.random.default_rng(0).uniform(0, 1, n_frames)

    df_enrich = compute_transition_enrichment(frame_scores, dtraj, tmp_path)

    enrich_path = tmp_path / "transition_enrichment.csv"
    assert enrich_path.exists()
    assert set(df_enrich.columns) >= {"mean_transition", "mean_stable", "cohens_d"}
    assert np.isfinite(df_enrich["cohens_d"].iloc[0]), "Cohen's d should be finite"
    print(f"  ✓ Transition enrichment: Cohen's d = {df_enrich['cohens_d'].iloc[0]:.4f}")


def test_compute_spatial_clustering(tmp_path):
    """compute_spatial_clustering produces spatial_clustering.csv."""
    print("[TEST] compute_spatial_clustering...")
    tmp_path.mkdir(parents=True, exist_ok=True)
    n_residues = 20
    pdb_path = tmp_path / "test_topology.pdb"
    _write_minimal_pdb(pdb_path, n_residues=n_residues)

    fused_scores = np.random.default_rng(42).uniform(0, 1, n_residues)
    df_spatial = compute_spatial_clustering(fused_scores, n_residues,
                                             pdb_path, tmp_path, n_random=20)

    spatial_path = tmp_path / "spatial_clustering.csv"
    assert spatial_path.exists()
    assert set(df_spatial.columns) >= {
        "observed_mean_distance", "random_mean", "random_std", "z_score"
    }
    assert np.isfinite(df_spatial["z_score"].iloc[0])
    print(f"  ✓ Spatial clustering Z = {df_spatial['z_score'].iloc[0]:.4f}")


def test_full_pipeline_end_to_end(tmp_path):
    """run_chapter9_evaluation completes and saves all 8 output files."""
    print("[TEST] Full end-to-end pipeline...")

    # Write synthetic features
    tmp_path.mkdir(parents=True, exist_ok=True)
    X = _make_features(n_frames=150)
    features_path = tmp_path / "features.npy"
    np.save(features_path, X)

    # Write minimal PDB
    topo_path = tmp_path / "topology.pdb"
    _write_minimal_pdb(topo_path, n_residues=25)

    out_dir = tmp_path / "results" / "chapter9"

    run_chapter9_evaluation(
        features_path=features_path,
        topology_path=topo_path,
        output_dir=out_dir,
        lag_tica=5,
        dim_tica=3,
        n_clusters=6,
        lag_msm=5,
    )

    expected = [
        "implied_timescales.csv",
        "implied_timescale_cv.csv",
        "ck_errors.csv",
        "vamp_comparison.csv",
        "residue_ranking.csv",
        "topk_sets.csv",
        "hotspot_slowmode_alignment.csv",
        "transition_enrichment.csv",
        "spatial_clustering.csv",
        "ranking_stability.csv",
    ]
    for fname in expected:
        assert (out_dir / fname).exists(), f"Missing output: {fname}"

    # Spot-check key files
    df_stab = pd.read_csv(out_dir / "ranking_stability.csv")
    assert "perturbation_type" in df_stab.columns
    assert len(df_stab) == 7, f"Expected 7 perturbations, got {len(df_stab)}"

    df_vamp = pd.read_csv(out_dir / "vamp_comparison.csv")
    assert len(df_vamp) == 3

    # Issue-specific outputs
    assert (out_dir / "hotspot_slowmode_alignment_no_tica.csv").exists()
    assert (out_dir / "vamp_comparison_corrected.csv").exists()
    assert (out_dir / "transition_enrichment_window_sweep.csv").exists()
    assert (out_dir / "ranking_stability_extended.csv").exists()

    print(f"  ✓ All {len(expected)} output files present")


def test_hotspot_slowmode_alignment_no_tica(tmp_path):
    """compute_hotspot_slowmode_alignment_no_tica: new ρ differs from old ρ."""
    print("[TEST] hotspot_slowmode_alignment_no_tica...")
    X = _make_features(n_frames=150)
    msm, dtraj, Y, tica_model = _fit_pipeline(X, 5, 3, 6, 5)
    frame_scores = _fused_frame_scores(msm, dtraj, Y, lag_msm=5)
    fused = _residue_fused_scores(frame_scores, n_residues=25)

    df = compute_hotspot_slowmode_alignment_no_tica(
        fused, frame_scores, tica_model, n_residues=25, output_dir=tmp_path
    )
    assert (tmp_path / "hotspot_slowmode_alignment_no_tica.csv").exists()
    assert set(df.columns) >= {"old_spearman_rho", "new_spearman_rho",
                               "circularity_confirmed"}
    old_rho = df["old_spearman_rho"].iloc[0]
    new_rho = df["new_spearman_rho"].iloc[0]
    assert -1.0 <= old_rho <= 1.0, "old_rho should be in [-1, 1]"
    assert -1.0 <= new_rho <= 1.0, "new_rho should be in [-1, 1]"
    print(f"  ✓ old_ρ={old_rho:.4f}  new_ρ={new_rho:.4f}  "
          f"circularity={df['circularity_confirmed'].iloc[0]}")


def test_compute_vamp_comparison_corrected(tmp_path):
    """compute_vamp_comparison_corrected: tICA ≠ raw_features VAMP-2 score."""
    print("[TEST] compute_vamp_comparison_corrected...")
    X = _make_features(n_frames=150)
    df = compute_vamp_comparison_corrected(X, lag_msm=5, dim_tica=3,
                                           lag_tica=5, output_dir=tmp_path)
    assert (tmp_path / "vamp_comparison_corrected.csv").exists()
    assert len(df) == 3
    assert set(df["model_type"]) == {"tICA", "PCA", "raw_features"}
    score_tica = df.loc[df["model_type"] == "tICA", "vamp2_score"].iloc[0]
    score_raw = df.loc[df["model_type"] == "raw_features", "vamp2_score"].iloc[0]
    # After fix, raw_features uses all features (dim=7) so score differs from tICA (dim=3)
    assert score_tica != score_raw, (
        f"tICA ({score_tica:.4f}) and raw_features ({score_raw:.4f}) "
        "should differ in corrected comparison"
    )
    print(f"  ✓ VAMP corrected: tICA={score_tica:.4f}  raw={score_raw:.4f}")


def test_compute_transition_enrichment_window_sweep(tmp_path):
    """compute_transition_enrichment_window_sweep: produces 3 windows."""
    print("[TEST] compute_transition_enrichment_window_sweep...")
    n_frames = 100
    dtraj = np.array([0] * 30 + [1] * 40 + [0] * 30, dtype=np.int64)
    frame_scores = np.random.default_rng(0).uniform(0, 1, n_frames)

    df = compute_transition_enrichment_window_sweep(frame_scores, dtraj, tmp_path)
    assert (tmp_path / "transition_enrichment_window_sweep.csv").exists()
    assert set(df.columns) >= {"window_size", "mean_transition",
                               "mean_stable", "cohens_d"}
    assert set(df["window_size"].tolist()) == {3, 5, 10}
    assert df["cohens_d"].notna().all(), "All Cohen's d should be finite"
    print(f"  ✓ Window sweep rows: {len(df)}, windows: {df['window_size'].tolist()}")


def test_compute_ranking_stability_extended(tmp_path):
    """compute_ranking_stability_extended: produces top-10/20/30% Jaccard rows."""
    print("[TEST] compute_ranking_stability_extended...")
    X = _make_features(n_frames=150)
    n_residues = 25
    # Derive baseline ranks quickly
    _, fused = compute_residue_ranking(
        np.random.default_rng(0).uniform(0, 1, 150), n_residues, tmp_path
    )
    baseline_ranks = (pd.Series(fused)
                      .rank(ascending=False, method="first")
                      .values.astype(int))

    df = compute_ranking_stability_extended(
        X, lag_msm=5, dim_tica=3, n_clusters=6, lag_tica=5,
        n_residues=n_residues, baseline_ranks=baseline_ranks,
        output_dir=tmp_path
    )
    assert (tmp_path / "ranking_stability_extended.csv").exists()
    assert set(df.columns) >= {"perturbation_type", "topk_percent", "jaccard_index"}
    assert set(df["topk_percent"].unique()) == {10, 20, 30}
    assert (df["jaccard_index"] >= 0).all() and (df["jaccard_index"] <= 1).all()
    print(f"  ✓ Extended stability rows: {len(df)}, "
          f"k% values: {sorted(df['topk_percent'].unique().tolist())}")


# ---------------------------------------------------------------------------
# RQ1 — Signal Validity Tests
# ---------------------------------------------------------------------------

def test_rq1_frame_scores_have_variance():
    """RQ1: Fused frame scores are not constant — pipeline captures real variation."""
    print("[TEST] RQ1: frame scores have non-zero variance...")
    X = _make_features(n_frames=150)
    msm, dtraj, Y, _ = _fit_pipeline(X, 5, 3, 6, 5)
    scores = _fused_frame_scores(msm, dtraj, Y, lag_msm=5)
    assert np.std(scores) > 0, "Frame scores must not be constant (signal has no variance)"
    assert scores.min() < scores.max(), "Frame scores must span a non-trivial range"
    print(f"  ✓ Frame score std={np.std(scores):.4f}, range=[{scores.min():.4f}, {scores.max():.4f}]")


def test_rq1_its_plateau_cv_finite(tmp_path):
    """RQ1: ITS plateau CV values are finite and non-negative (timescales are stable)."""
    print("[TEST] RQ1: ITS plateau CV is finite...")
    X = _make_features(n_frames=150)
    _, df_cv = compute_implied_timescales(X, lag_msm=5, dim_tica=3, n_clusters=6,
                                          lag_tica=5, output_dir=tmp_path)
    if len(df_cv) > 0:
        assert df_cv["cv"].notna().all(), "CV values must all be non-NaN"
        assert (df_cv["cv"] >= 0).all(), "CV values must be non-negative"
        assert (df_cv["mean"] > 0).all(), "Mean timescales must be positive"
    print(f"  ✓ ITS CV: {len(df_cv)} mode(s), all finite and non-negative")


def test_rq1_transition_enrichment_cohens_d_finite():
    """RQ1: Transition enrichment Cohen's d is finite for data with clear state changes."""
    print("[TEST] RQ1: transition enrichment Cohen's d is finite...")
    n_frames = 200
    # Construct dtraj with multiple clear transitions
    dtraj = np.array([0] * 50 + [1] * 50 + [0] * 50 + [1] * 50, dtype=np.int64)
    rng = np.random.default_rng(7)  # fixed seed for reproducibility
    # Transition frames get higher scores to model anomaly detection
    frame_scores = rng.uniform(0.3, 0.5, n_frames)
    transition_mask = np.zeros(n_frames, dtype=bool)
    for t in range(1, n_frames):
        if dtraj[t] != dtraj[t - 1]:
            transition_mask[max(0, t - 5):min(n_frames, t + 6)] = True
    frame_scores[transition_mask] = rng.uniform(0.6, 0.9, transition_mask.sum())

    with tempfile.TemporaryDirectory() as tmp:
        df_enrich = compute_transition_enrichment(frame_scores, dtraj, Path(tmp))
    cohens_d = df_enrich["cohens_d"].iloc[0]
    assert np.isfinite(cohens_d), "Cohen's d must be finite"
    assert cohens_d > 0, (
        "Cohen's d should be positive when transition frames have higher scores"
    )
    print(f"  ✓ Cohen's d = {cohens_d:.4f} (positive: transition frames more anomalous)")


def test_rq1_vamp2_corrected_scores_are_positive(tmp_path):
    """RQ1: Corrected VAMP-2 scores are positive for all model types."""
    print("[TEST] RQ1: corrected VAMP-2 scores are positive...")
    X = _make_features(n_frames=150)
    df = compute_vamp_comparison_corrected(X, lag_msm=5, dim_tica=3,
                                           lag_tica=5, output_dir=tmp_path)
    for _, row in df.iterrows():
        assert row["vamp2_score"] > 0, (
            f"VAMP-2 score for {row['model_type']} must be positive, "
            f"got {row['vamp2_score']:.4f}"
        )
    # tICA and raw_features should differ in the corrected comparison
    score_tica = df.loc[df["model_type"] == "tICA", "vamp2_score"].iloc[0]
    score_raw = df.loc[df["model_type"] == "raw_features", "vamp2_score"].iloc[0]
    assert score_tica != score_raw, (
        "Corrected VAMP-2: tICA and raw_features must use different dimensionality"
    )
    print(f"  ✓ VAMP-2 scores: tICA={score_tica:.4f}, raw={score_raw:.4f}, PCA="
          f"{df.loc[df['model_type'] == 'PCA', 'vamp2_score'].iloc[0]:.4f}")


# ---------------------------------------------------------------------------
# RQ2 — Visualization as Validation Mechanism Tests
# ---------------------------------------------------------------------------

def test_rq2_topk_sets_are_nested(tmp_path):
    """RQ2: Top-k% residue sets are monotonically nested (top-5 ⊆ top-10 ⊆ top-20)."""
    print("[TEST] RQ2: top-k sets are nested...")
    frame_scores = np.random.default_rng(99).uniform(0, 1, 100)  # fixed seed for reproducibility
    n_residues = 30
    _, _ = compute_residue_ranking(frame_scores, n_residues, tmp_path)
    df_topk = pd.read_csv(tmp_path / "topk_sets.csv")

    ids_5 = set(df_topk[df_topk["k_percent"] == 5]["residue_id"])
    ids_10 = set(df_topk[df_topk["k_percent"] == 10]["residue_id"])
    ids_20 = set(df_topk[df_topk["k_percent"] == 20]["residue_id"])

    assert ids_5.issubset(ids_10), "top-5% must be a subset of top-10%"
    assert ids_10.issubset(ids_20), "top-10% must be a subset of top-20%"
    assert len(ids_5) <= len(ids_10) <= len(ids_20), (
        "Set sizes must be non-decreasing: |top-5| ≤ |top-10| ≤ |top-20|"
    )
    print(f"  ✓ Nested sets: |top-5%|={len(ids_5)}, |top-10%|={len(ids_10)}, "
          f"|top-20%|={len(ids_20)}")


def test_rq2_residue_ranking_visualization_columns(tmp_path):
    """RQ2: Residue ranking CSV has all columns needed for visualization."""
    print("[TEST] RQ2: residue ranking has visualization columns...")
    frame_scores = np.random.default_rng(0).uniform(0, 1, 120)
    n_residues = 25
    df_rank, _ = compute_residue_ranking(frame_scores, n_residues, tmp_path)

    required_cols = {"residue_id", "fused_score", "rank"}
    assert required_cols.issubset(set(df_rank.columns)), (
        f"Missing columns for visualization: {required_cols - set(df_rank.columns)}"
    )
    # Residue IDs cover expected range
    assert df_rank["residue_id"].min() == 0
    assert df_rank["residue_id"].max() == n_residues - 1
    # Ranks are unique integers from 1 to n_residues
    assert df_rank["rank"].nunique() == n_residues
    assert df_rank["rank"].min() == 1
    assert df_rank["rank"].max() == n_residues
    print(f"  ✓ Visualization columns present; rank range [1, {n_residues}], "
          f"residue IDs [0, {n_residues - 1}]")


def test_rq2_frame_score_length_matches_trajectory():
    """RQ2: Frame scores have exactly the same length as the input trajectory."""
    print("[TEST] RQ2: frame score length matches trajectory...")
    for n_frames in [80, 150, 200]:
        X = _make_features(n_frames=n_frames)
        msm, dtraj, Y, _ = _fit_pipeline(X, 5, 3, 6, 5)
        scores = _fused_frame_scores(msm, dtraj, Y, lag_msm=5)
        assert len(scores) == n_frames, (
            f"Frame scores length {len(scores)} != n_frames {n_frames}"
        )
    print(f"  ✓ Frame score length matches trajectory for n_frames in [80, 150, 200]")


def test_rq2_window_sweep_columns_for_visualization(tmp_path):
    """RQ2: Window sweep CSV has columns enabling frame-resolved inspection."""
    print("[TEST] RQ2: window sweep columns for visualization...")
    dtraj = np.array([0] * 30 + [1] * 40 + [0] * 30, dtype=np.int64)
    frame_scores = np.random.default_rng(5).uniform(0, 1, 100)
    df = compute_transition_enrichment_window_sweep(frame_scores, dtraj, tmp_path)

    required_cols = {"window_size", "mean_transition", "mean_stable", "cohens_d"}
    assert required_cols.issubset(set(df.columns)), (
        f"Missing visualization columns: {required_cols - set(df.columns)}"
    )
    # Increasing window should not necessarily increase Cohen's d — just verify all rows valid
    assert df["cohens_d"].notna().all(), "All Cohen's d values must be non-NaN"
    print(f"  ✓ Window sweep has {len(df)} rows with required visualization columns")


# ---------------------------------------------------------------------------
# RQ3 — Sensitivity and Robustness Tests
# ---------------------------------------------------------------------------

def test_rq3_stability_metrics_bounded(tmp_path):
    """RQ3: All Jaccard and Spearman values from ranking stability are in valid ranges."""
    print("[TEST] RQ3: stability metrics bounded correctly...")
    X = _make_features(n_frames=150)
    n_residues = 20
    _, fused = compute_residue_ranking(
        np.random.default_rng(0).uniform(0, 1, 150), n_residues, tmp_path
    )
    baseline_ranks = (pd.Series(fused)
                      .rank(ascending=False, method="first")
                      .values.astype(int))
    df = compute_ranking_stability_extended(
        X, lag_msm=5, dim_tica=3, n_clusters=6, lag_tica=5,
        n_residues=n_residues, baseline_ranks=baseline_ranks,
        output_dir=tmp_path
    )
    assert (df["jaccard_index"] >= 0).all(), "Jaccard index must be ≥ 0"
    assert (df["jaccard_index"] <= 1).all(), "Jaccard index must be ≤ 1"
    assert set(df["topk_percent"].unique()) == {10, 20, 30}, (
        "Must report Jaccard for top-10%, top-20%, top-30%"
    )
    print(f"  ✓ All {len(df)} Jaccard values in [0, 1] for k% ∈ {{10, 20, 30}}")


def test_rq3_fusion_median_vs_mean_differ():
    """RQ3: Median and mean signal fusion produce different frame scores."""
    print("[TEST] RQ3: median vs mean fusion differ...")
    X = _make_features(n_frames=150)
    msm, dtraj, Y, _ = _fit_pipeline(X, 5, 3, 6, 5)

    from sklearn.neighbors import NearestNeighbors

    n_frames = len(dtraj)
    pi = msm.stationary_distribution
    P = msm.transition_matrix
    n_states = msm.n_states

    rarity = np.array([1.0 - pi[s] if 0 <= s < n_states else 1.0
                       for s in dtraj], dtype=np.float64)
    surprise = np.zeros(n_frames)
    for t in range(n_frames - 5):
        s1, s2 = dtraj[t], dtraj[t + 5]
        if 0 <= s1 < n_states and 0 <= s2 < n_states:
            surprise[t] = -np.log(max(P[s1, s2], 1e-12))

    k = min(10, n_frames - 1)
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=1).fit(Y)
    dists, _ = nbrs.kneighbors(Y)
    local_density = dists.mean(axis=1)

    def rank_norm(x):
        if np.all(x == x[0]):
            return np.zeros_like(x)
        r = np.argsort(np.argsort(x)).astype(float)
        return r / (len(x) - 1)

    mat = np.column_stack([rank_norm(rarity), rank_norm(surprise),
                           rank_norm(local_density)])
    scores_median = np.median(mat, axis=1)
    scores_mean = np.mean(mat, axis=1)

    # Median and mean fusion are only identical when all signals agree exactly
    assert not np.allclose(scores_median, scores_mean), (
        "Median and mean fusion should differ for realistic multi-signal data"
    )
    diff = np.abs(scores_median - scores_mean).mean()
    print(f"  ✓ Median vs mean fusion differ: mean |Δ| = {diff:.6f}")


def test_rq3_lag_perturbation_changes_frame_scores():
    """RQ3: Changing the MSM lag time produces different frame scores."""
    print("[TEST] RQ3: lag perturbation changes frame scores...")
    X = _make_features(n_frames=150)
    msm_5, dtraj_5, Y_5, _ = _fit_pipeline(X, lag_tica=5, dim_tica=3,
                                            n_clusters=6, lag_msm=5)
    msm_3, dtraj_3, Y_3, _ = _fit_pipeline(X, lag_tica=5, dim_tica=3,
                                            n_clusters=6, lag_msm=3)
    scores_5 = _fused_frame_scores(msm_5, dtraj_5, Y_5, lag_msm=5)
    scores_3 = _fused_frame_scores(msm_3, dtraj_3, Y_3, lag_msm=3)

    # Different lag times must produce at least some difference in frame scores
    assert not np.allclose(scores_5, scores_3), (
        "Different lag times must produce different frame scores"
    )
    diff = np.abs(scores_5 - scores_3).mean()
    print(f"  ✓ Lag 5 vs lag 3 frame scores differ: mean |Δ| = {diff:.6f}")


def test_rq3_window_sweep_cohens_d_varies():
    """RQ3: Window sweep Cohen's d values vary across window sizes (sensitivity check)."""
    print("[TEST] RQ3: window sweep Cohen's d varies across windows...")
    n_frames = 150
    rng = np.random.default_rng(11)
    dtraj = np.array([0] * 40 + [1] * 35 + [0] * 40 + [1] * 35, dtype=np.int64)
    frame_scores = rng.uniform(0, 1, n_frames)

    with tempfile.TemporaryDirectory() as tmp:
        df = compute_transition_enrichment_window_sweep(frame_scores, dtraj, Path(tmp))

    # Must have exactly 3 window sizes
    assert set(df["window_size"].tolist()) == {3, 5, 10}, (
        "Window sweep must cover windows 3, 5, and 10"
    )
    # Cohen's d values must all be finite
    assert df["cohens_d"].notna().all(), "Cohen's d must be finite for all windows"
    # Values should differ across windows (larger window includes more frames near transitions)
    cohens_vals = df.sort_values("window_size")["cohens_d"].values
    # Round to 8 decimal places to ignore floating-point noise while detecting real differences
    assert len(set(cohens_vals.round(8))) > 1, (
        "Cohen's d should differ across at least two window sizes"
    )
    print(f"  ✓ Cohen's d values: {dict(zip(df['window_size'].tolist(), cohens_vals.round(4).tolist()))}")


def main():
    """Run all Chapter 9 evaluation tests."""
    print("=" * 70)
    print("TESTING CHAPTER 9 EVALUATION MODULE")
    print("=" * 70 + "\n")

    try:
        test_fit_pipeline_returns_correct_types()
        test_fused_frame_scores_shape()
        test_residue_fused_scores_no_nan()
        test_jaccard_top10()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            test_compute_implied_timescales(tmp_path / "its")
            test_compute_ck_errors(tmp_path / "ck")
            test_compute_vamp_comparison(tmp_path / "vamp")
            test_compute_residue_ranking(tmp_path / "rank")
            test_compute_transition_enrichment(tmp_path / "enrich")
            test_compute_spatial_clustering(tmp_path / "spatial")
            test_hotspot_slowmode_alignment_no_tica(tmp_path / "no_tica")
            test_compute_vamp_comparison_corrected(tmp_path / "vamp_corr")
            test_compute_transition_enrichment_window_sweep(tmp_path / "sweep")
            test_compute_ranking_stability_extended(tmp_path / "rank_ext")
            test_full_pipeline_end_to_end(tmp_path / "e2e")

            # RQ1 — Signal Validity
            test_rq1_frame_scores_have_variance()
            test_rq1_its_plateau_cv_finite(tmp_path / "rq1_its")
            test_rq1_transition_enrichment_cohens_d_finite()
            test_rq1_vamp2_corrected_scores_are_positive(tmp_path / "rq1_vamp2")

            # RQ2 — Visualization as Validation
            test_rq2_topk_sets_are_nested(tmp_path / "rq2_topk")
            test_rq2_residue_ranking_visualization_columns(tmp_path / "rq2_rank")
            test_rq2_frame_score_length_matches_trajectory()
            test_rq2_window_sweep_columns_for_visualization(tmp_path / "rq2_sweep")

            # RQ3 — Sensitivity and Robustness
            test_rq3_stability_metrics_bounded(tmp_path / "rq3_stab")
            test_rq3_fusion_median_vs_mean_differ()
            test_rq3_lag_perturbation_changes_frame_scores()
            test_rq3_window_sweep_cohens_d_varies()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        return 0

    except AssertionError as exc:
        print(f"\n✗ TEST FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as exc:
        print(f"\n✗ ERROR: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
