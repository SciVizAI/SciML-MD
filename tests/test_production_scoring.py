# ============================================================== #
#  Module:      tests/test_production_scoring.py
#  Description: Correctness tests for the PRODUCTION scoring path
#               (scoring/anomaly_v2.py + run_all_proteins.py) —
#               active-set remapping, tie-aware ranks, NaN-aware
#               fusion, residue-key collisions, ASVS export.
# ============================================================== #
"""
These tests were added after the August 2026 validation campaign
(validation/VALIDATION_REPORT.md). Unlike tests/test_pipeline_edge_cases.py,
which exercises scoring/signals.py (a parallel implementation the batch
pipeline does not call), every test here targets the code path that
run_all_proteins.py / batch_runner.py actually execute, and asserts numeric
correctness, not just "finite and did not crash".

Run:  pytest tests/test_production_scoring.py -v
"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scoring.anomaly_v2 import (  # noqa: E402
    compute_kinetic_signals,
    remap_dtraj_to_active_set,
    rank_normalize,
    fuse_signals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_msm_with_dropped_state():
    """dtraj whose main component lives on labels {0,1,4} and whose final block
    sits in state 3 and never returns — the reversible MLE MSM drops state 3
    (and never-visited label 2), reproducing the state-space mismatch seen in
    every real run (10 KMeans labels vs 7 MSM states)."""
    from deeptime.markov.msm import MaximumLikelihoodMSM

    rng = np.random.default_rng(7)
    core = rng.choice([0, 1, 4], size=160)
    tail = np.full(40, 3)
    dtraj = np.concatenate([core, tail]).astype(np.int64)
    msm = MaximumLikelihoodMSM(lagtime=1, reversible=True).fit(dtraj).fetch_model()
    assert msm.n_states < int(dtraj.max()) + 1
    return msm, dtraj


# ---------------------------------------------------------------------------
# Active-set remapping
# ---------------------------------------------------------------------------
def test_remap_matches_state_symbols():
    msm, dtraj = make_msm_with_dropped_state()
    symbols = np.asarray(msm.count_model.state_symbols)
    assert msm.n_states < int(dtraj.max()) + 1, "fixture must drop a state"

    dtraj_active, remapped = remap_dtraj_to_active_set(dtraj, msm)
    assert remapped
    # every frame in a surviving state maps to the index of its label in symbols
    for t in range(len(dtraj)):
        if dtraj[t] in symbols:
            assert symbols[dtraj_active[t]] == dtraj[t]
        else:
            assert dtraj_active[t] == -1


def test_rarity_uses_correct_pi_row():
    """rarity[t] must equal 1 - pi[active_index(label_t)] — the historical bug
    indexed pi with the raw label instead."""
    msm, dtraj = make_msm_with_dropped_state()
    symbols = np.asarray(msm.count_model.state_symbols)
    pi = msm.stationary_distribution

    rarity, _ = compute_kinetic_signals(msm, dtraj, lag_msm=1)
    for t in range(len(dtraj)):
        if dtraj[t] in symbols:
            i = int(np.where(symbols == dtraj[t])[0][0])
            assert np.isclose(rarity[t], 1.0 - pi[i]), (
                f"frame {t}: label {dtraj[t]} should read pi[{i}]"
            )
        else:
            assert rarity[t] == 1.0  # disconnected state = maximally rare


def test_surprise_uses_correct_P_entries():
    msm, dtraj = make_msm_with_dropped_state()
    symbols = np.asarray(msm.count_model.state_symbols)
    P = msm.transition_matrix
    lag = 1
    _, surprise = compute_kinetic_signals(msm, dtraj, lag_msm=lag)
    idx = {int(s): i for i, s in enumerate(symbols)}
    for t in range(len(dtraj) - lag):
        a, b = int(dtraj[t]), int(dtraj[t + lag])
        if a in idx and b in idx:
            expected = -np.log(max(P[idx[a], idx[b]], 1e-12))
            assert np.isclose(surprise[t], expected)


def test_disconnected_transition_is_max_surprise_not_zero():
    """Transitions touching a dropped state were silently 0 (least anomalous);
    they must now rank at the top."""
    msm, dtraj = make_msm_with_dropped_state()
    symbols = set(np.asarray(msm.count_model.state_symbols).tolist())
    _, surprise = compute_kinetic_signals(msm, dtraj, lag_msm=1)
    touching = [
        t for t in range(len(dtraj) - 1)
        if dtraj[t] not in symbols or dtraj[t + 1] not in symbols
    ]
    assert touching, "fixture must contain transitions touching the dropped state"
    finite = surprise[np.isfinite(surprise)]
    for t in touching:
        assert surprise[t] == finite.max()


def test_negative_labels_do_not_wrap_around():
    msm, dtraj = make_msm_with_dropped_state()
    bad = dtraj.copy()
    bad[5] = -3
    rarity, _ = compute_kinetic_signals(msm, bad, lag_msm=1)
    assert rarity[5] == 1.0  # invalid, not lookup[-3]


# ---------------------------------------------------------------------------
# Tie-aware normalization + NaN-aware fusion
# ---------------------------------------------------------------------------
def test_tied_values_get_equal_scores():
    x = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    r = rank_normalize(x)
    assert r[0] == r[1] and r[2] == r[3] and r[3] < r[4]


def test_tie_spread_is_zero_for_large_tie_group():
    x = np.concatenate([np.zeros(50), np.arange(1, 11)])
    r = rank_normalize(x)
    assert np.ptp(r[:50]) == 0.0, "tied zeros must not be spread by frame order"


def test_surprise_tail_is_nan_and_fusion_stays_finite():
    msm, dtraj = make_msm_with_dropped_state()
    lag = 5
    rarity, surprise = compute_kinetic_signals(msm, dtraj, lag_msm=lag)
    assert np.all(np.isnan(surprise[-lag:])), "tail frames have no transition"
    assert np.all(np.isfinite(surprise[:-lag]))

    score, comps = fuse_signals(
        {"rarity": rarity, "transition_surprise": surprise,
         "local_density": np.random.default_rng(0).normal(size=len(dtraj))},
    )
    assert np.all(np.isfinite(score)), "fusion must be NaN-aware"
    assert np.all((score >= 0) & (score <= 1))


def test_rank_normalize_constant_and_empty():
    assert len(rank_normalize(np.array([]))) == 0
    assert np.all(rank_normalize(np.ones(4)) == 0.0)


# ---------------------------------------------------------------------------
# Residue scores: chain collisions + not-pure-RMSF
# ---------------------------------------------------------------------------
TWO_CHAIN_PDB = """\
MODEL        1
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00  0.00           C
ATOM      3  CA  ALA B   1       0.000  10.000   0.000  1.00  0.00           C
ATOM      4  CA  GLY B   2       3.800  10.000   0.000  1.00  0.00           C
TER
ENDMDL
MODEL        2
ATOM      1  CA  ALA A   1       0.100   0.000   0.000  1.00  0.00           C
ATOM      2  CA  GLY A   2       3.900   0.100   0.000  1.00  0.00           C
ATOM      3  CA  ALA B   1       0.000  10.200   0.000  1.00  0.00           C
ATOM      4  CA  GLY B   2       3.600  10.000   0.200  1.00  0.00           C
TER
ENDMDL
MODEL        3
ATOM      1  CA  ALA A   1       0.000   0.100   0.100  1.00  0.00           C
ATOM      2  CA  GLY A   2       3.700   0.000   0.100  1.00  0.00           C
ATOM      3  CA  ALA B   1       0.100  10.000   0.100  1.00  0.00           C
ATOM      4  CA  GLY B   2       3.800  10.300   0.000  1.00  0.00           C
TER
ENDMDL
END
"""


@pytest.fixture()
def two_chain_traj(tmp_path):
    md = pytest.importorskip("mdtraj")
    p = tmp_path / "two_chain.pdb"
    p.write_text(TWO_CHAIN_PDB)
    return md.load(str(p))


def test_residue_keys_do_not_collide_across_chains(two_chain_traj):
    from run_all_proteins import compute_residue_scores

    frame_scores = np.array([10.0, 90.0, 50.0])
    scores = compute_residue_scores(two_chain_traj, frame_scores)
    assert len(scores) == 4, (
        f"expected 4 residues (2 chains x 2), got {len(scores)} — "
        "chain collision regression"
    )


def test_residue_scores_not_pure_rmsf(two_chain_traj):
    """With two different frame-score vectors, residue scores must differ —
    the old implementation added the same constant to every residue, so the
    ranking was exactly RMSF regardless of frame scores."""
    from run_all_proteins import compute_residue_scores

    up = compute_residue_scores(two_chain_traj, np.array([0.0, 0.0, 100.0]))
    down = compute_residue_scores(two_chain_traj, np.array([100.0, 0.0, 0.0]))
    assert up.keys() == down.keys()
    diffs = [abs(up[k] - down[k]) for k in up]
    assert max(diffs) > 1e-6, "frame scores must influence residue scores"


# ---------------------------------------------------------------------------
# ASVS export regression (flat 0.25 bug)
# ---------------------------------------------------------------------------
def test_asvs_export_not_constant():
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from export_for_asvs import create_per_frame_residue_json

    frame_scores = pd.DataFrame({
        "frame": np.arange(4),
        "score_dynamic": [10.0, 90.0, 40.0, 70.0],
    })
    residue_scores = {"dynamic": {"MET41": 20.0, "LEU42": 80.0, "SER43": 50.0}}
    out = create_per_frame_residue_json(frame_scores, residue_scores, n_residues=3)

    vals = {v for f in ("0", "1", "2", "3") for v in out[f].values()}
    assert len(vals) > 1, "export must not be constant (0.25-everywhere regression)"
    # spot-check: frame 1 (score 0.9), residue 1 (0.8) -> 0.72
    assert abs(out["1"]["1"] - 0.72) < 1e-9
