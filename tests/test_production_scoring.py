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


# ---------------------------------------------------------------------------
# D-09: smoothing must be opt-in (it suppresses isolated rare frames)
# ---------------------------------------------------------------------------
def test_smoothing_is_off_by_default():
    """compute_anomaly_signals must not smooth unless explicitly asked.

    Phase 1 measured AUROC 0.839 unsmoothed vs 0.550 at window=3 on isolated
    single-frame anomalies (validation/phase1_results.json).
    """
    import inspect
    from run_all_proteins import compute_anomaly_signals

    sig = inspect.signature(compute_anomaly_signals)
    assert sig.parameters["window"].default == 1, "smoothing must default to OFF"


def test_isolated_spike_survives_default_scoring():
    """A single anomalous frame must remain the top-ranked frame by default."""
    from run_all_proteins import compute_anomaly_signals, cluster_states, build_msm

    rng = np.random.default_rng(3)
    Y = rng.normal(size=(120, 3))
    Y[60] += 12.0  # one isolated outlier in tICA space
    dtraj, _ = cluster_states(Y, n_clusters=8, seed=42)
    msm, _, _ = build_msm(dtraj, lag=3)

    scores, _ = compute_anomaly_signals(msm, dtraj, Y, lag_msm=3, k_neighbors=5)
    assert int(np.argmax(scores)) == 60, "isolated anomaly must survive default settings"

    smoothed, _ = compute_anomaly_signals(msm, dtraj, Y, lag_msm=3, k_neighbors=5,
                                          window=5)
    assert np.max(smoothed) <= np.max(scores), "smoothing should only attenuate"


# ---------------------------------------------------------------------------
# E-1 / D-10 / O-10: post-audit fixes
# ---------------------------------------------------------------------------
@pytest.fixture()
def synthetic_traj(tmp_path):
    """A 12-residue CA-only chain over 80 frames, with residue-dependent
    amplitude so RMSF genuinely varies across the chain."""
    md = pytest.importorskip("mdtraj")
    n_res, n_frames = 12, 80
    lines = ["MODEL        1"]
    for i in range(n_res):
        lines.append(f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                     f"{i*3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C")
    lines += ["TER", "ENDMDL", "END"]
    p = tmp_path / "chain.pdb"
    p.write_text("\n".join(lines))
    base = md.load(str(p))

    rng = np.random.default_rng(5)
    xyz = np.repeat(base.xyz, n_frames, axis=0)
    amp = np.linspace(0.01, 0.30, n_res)          # RMSF varies along the chain
    xyz += (rng.normal(size=xyz.shape) * amp[None, :, None]).astype("float32")
    return md.Trajectory(xyz, base.topology)


def test_residue_attribution_is_orthogonal_to_rmsf(synthetic_traj):
    """The reported residue score must not be a proxy for RMSF.

    Legacy score: Spearman vs RMSF 0.94-0.98 (erratum E-1).
    """
    from scoring.residue_attribution import compute_residue_attribution

    rng = np.random.default_rng(5)
    scores = rng.random(len(synthetic_traj)) * 100
    out = compute_residue_attribution(synthetic_traj, scores)
    d = out["diagnostics"]
    assert abs(d["spearman_final_vs_rmsf"]) < 0.7, (
        f"attribution still collinear with RMSF: {d['spearman_final_vs_rmsf']}")
    assert d["n_frames_anomalous"] >= 3 and d["n_frames_baseline"] >= 3


def test_residue_attribution_rank_is_tie_aware():
    """The external KSD-RA reference used argsort(argsort(x)) — defect D-02."""
    from scoring.residue_attribution import _rank_norm

    r = _rank_norm(np.array([5.0, 5.0, 5.0, 5.0, 9.0]))
    assert r[0] == r[1] == r[2] == r[3], "tied values must receive equal scores"
    assert r[4] > r[0]


def test_equilibration_detection_discards_startup(synthetic_traj):
    """A trajectory that relaxes away from its start must have those frames cut."""
    md = pytest.importorskip("mdtraj")
    from msm.preflight import detect_equilibration

    rng = np.random.default_rng(11)
    base = synthetic_traj
    n = len(base)
    xyz = base.xyz.copy()
    drift = np.concatenate([np.linspace(0, 0.6, 12), np.full(n - 12, 0.6)])
    xyz[:, :, 0] += drift[:, None].astype("float32")
    traj = md.Trajectory(xyz, base.topology)

    start, info = detect_equilibration(traj)
    assert start > 0, "startup relaxation must be detected"
    assert info["fraction_discarded"] <= 0.20


def test_suitability_flags_single_basin(synthetic_traj):
    """A single-basin ensemble must not be silently accepted."""
    from msm.preflight import assess_suitability

    s = assess_suitability(synthetic_traj)
    assert s["verdict"] in ("unsuitable", "marginal")
    assert s["reasons"]


def _basin_traj(synthetic_traj, kind, seed=3):
    """Build a single-basin, two-basin or diffusive ensemble on the same chain."""
    md = pytest.importorskip("mdtraj")
    rng = np.random.default_rng(seed)
    ref = synthetic_traj.xyz[0].copy()
    n_at = ref.shape[0]

    if kind == "two_basin":
        alt = ref.copy()
        alt[n_at // 2:] += np.array([0.8, 0.3, 0.0], dtype="float32")
        frames, state = [], 0
        for i in range(600):
            if i % 60 == 0:
                state ^= 1
            frames.append((alt if state else ref)
                          + rng.normal(0, 0.02, ref.shape))
        xyz = np.stack(frames)
    elif kind == "diffusive":
        # unbounded random walk: nothing is ever revisited
        xyz = ref[None] + np.cumsum(
            rng.normal(0, 0.06, (600,) + ref.shape), axis=0)
    else:
        xyz = ref[None] + rng.normal(0, 0.02, (600,) + ref.shape)
    return md.Trajectory(xyz.astype("float32"), synthetic_traj.topology)


def test_suitability_rejects_diffusive_ensemble(synthetic_traj):
    """An ensemble that never revisits a structure has no metastable states.

    The screen was one-sided: it only caught ensembles that were too TIGHT.
    An ATLAS screen of 63 trajectories returned 60 'suitable' at 15-44 A spread
    with 0.5% medoid occupancy - one frame. Those are diffusive chains, and an
    MSM has as little to find there as in a single basin.
    """
    from msm.preflight import assess_suitability

    s = assess_suitability(_basin_traj(synthetic_traj, "diffusive"))
    assert s["verdict"] == "unsuitable", s
    assert s["largest_basin_occupancy"] < 0.10
    assert s["n_basins_ge_5pct"] == 0
    assert any("diffusive" in r for r in s["reasons"]), s["reasons"]


def test_suitability_accepts_two_basin_ensemble(synthetic_traj):
    """The guard must not reject everything: a genuine two-state ensemble passes."""
    from msm.preflight import assess_suitability

    s = assess_suitability(_basin_traj(synthetic_traj, "two_basin"))
    assert s["verdict"] == "suitable", s
    assert s["n_basins_ge_5pct"] >= 2
    assert 0.20 <= s["largest_basin_occupancy"] <= 0.80


def test_suitability_accepts_many_small_basins(synthetic_traj):
    """Several modestly-populated basins is multi-basin, not diffusive.

    Regression for a knife-edge in the first version of the guard: it vetoed on
    largest_occupancy < 10%, which rejected 3dso_A_R2 (33 clusters, top 9.0%,
    SIX clusters above 5%) as diffusive while accepting its own replicate at
    18.4%. The veto now keys on the populated-cluster count instead.
    """
    md = pytest.importorskip("mdtraj")
    from msm.preflight import assess_suitability

    rng = np.random.default_rng(7)
    ref = synthetic_traj.xyz[0].copy()
    n_at = ref.shape[0]
    centres = []
    for k in range(8):                      # 8 basins -> none dominant
        c = ref.copy()
        c[n_at // 2:] += np.array([0.6 * np.cos(k), 0.6 * np.sin(k), 0.3 * k],
                                  dtype="float32")
        centres.append(c)
    frames = []
    for i in range(800):
        frames.append(centres[(i // 40) % 8] + rng.normal(0, 0.02, ref.shape))
    traj = md.Trajectory(np.stack(frames).astype("float32"),
                         synthetic_traj.topology)

    s = assess_suitability(traj)
    assert s["n_basins_ge_5pct"] >= 4, s
    assert s["largest_basin_occupancy"] < 0.60, s   # no basin dominates
    assert s["verdict"] != "unsuitable", s
    assert not any("diffusive" in r for r in s["reasons"]), s["reasons"]


def test_functional_labels_exclude_solvent_and_find_partners(tmp_path):
    """Phase 4 ground truth must come from the crystal, and must not count
    waters or cryoprotectants as function."""
    from validation.phase4_functional_enrichment import functional_labels

    def at(rec, serial, name, res, ch, seq, x, y, z):
        return (f"{rec:<6}{serial:5d} {name:^4} {res:>3} {ch}{seq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00")

    lines, n = [], 1
    for i in range(5):
        lines.append(at("ATOM  ", n, "CA", "ALA", "A", i + 1, i * 5.0, 0.0, 0.0))
        n += 1
    lines.append(at("HETATM", n, "C1", "ATP", "A", 101, 0.5, 2.0, 0.0)); n += 1
    lines.append(at("HETATM", n, "O", "HOH", "A", 201, 20.0, 2.0, 0.0)); n += 1
    lines.append(at("HETATM", n, "C1", "GOL", "A", 202, 10.0, 1.0, 0.0)); n += 1
    lines.append(at("ATOM  ", n, "CA", "GLY", "B", 1, 20.0, 1.0, 0.0)); n += 1
    lines.append("END")
    p = tmp_path / "t.pdb"
    p.write_text("\n".join(lines))

    gt = functional_labels(p, "A", cutoff_ang=4.5)
    lab, lig = gt["label_all"], gt["label_ligand"]
    assert list(gt["resseq"]) == [1, 2, 3, 4, 5]
    assert list(gt["resname"]) == ["ALA"] * 5
    assert lab[0], "residue beside the ATP ligand must be labelled"
    assert lab[4], "residue beside the partner chain must be labelled"
    assert not lab[2], "residue with no partner must not be labelled"
    assert not lab[1], "glycerol is a cryoprotectant, not function"
    assert gt["breakdown"] == {"ligand": 1, "nucleic": 0, "chain": 1}
    # the ligand-only label must exclude the interface contact
    assert lig[0] and not lig[4], lig


def test_phase4_alignment_survives_renumbering():
    """ATLAS topologies may not carry the crystal's author numbering.

    Matching on resSeq alone lost five proteins to '0 residues matched' -
    1upt_D, 3bpj_B, 3bzl_D, 4eo1_A, 6l4p_B - a harness defect, not data.
    """
    from validation.phase4_functional_enrichment import align_md_to_crystal

    cry_names = np.array(["MET", "ALA", "GLY", "SER", "LEU", "VAL", "THR"])
    cry_seq = np.array([101, 102, 103, 104, 105, 106, 107])
    md_names = np.array(["ALA", "GLY", "SER", "LEU", "VAL"])
    md_seq = np.array([1, 2, 3, 4, 5])

    idx, how = align_md_to_crystal(md_names, cry_names, cry_seq, md_seq)
    assert idx is not None, how
    assert list(idx) == [1, 2, 3, 4, 5], (idx, how)
    assert "offset" in how

    # exact numbering must still take the direct path
    idx2, how2 = align_md_to_crystal(cry_names, cry_names, cry_seq, cry_seq)
    assert how2.startswith("resSeq"), how2
    assert list(idx2) == list(range(len(cry_names)))


def test_phase4_label_validity_gate():
    """An AUROC on 3 positives, or on 90% positives, is not interpretable."""
    from validation.phase4_functional_enrichment import label_is_valid

    assert label_is_valid(np.array([1, 1, 1] + [0] * 60, dtype=bool))   # 3dso_A
    assert label_is_valid(np.array([1] * 77 + [0] * 9, dtype=bool))     # 4v1g_B
    assert label_is_valid(np.array([1] * 16 + [0] * 86, dtype=bool)) is None


def test_clustering_is_deterministic():
    """Same input + same seed must give identical labels, run to run."""
    from run_all_proteins import cluster_states

    rng = np.random.default_rng(9)
    Y = rng.normal(size=(200, 3))
    a, _ = cluster_states(Y, n_clusters=8, seed=42)
    b, _ = cluster_states(Y, n_clusters=8, seed=42)
    assert np.array_equal(a, b), "clustering must be reproducible"
    # canonical relabelling: centres are sorted, so label 0 is the lowest centre
    assert a.min() == 0 and a.max() == 7
