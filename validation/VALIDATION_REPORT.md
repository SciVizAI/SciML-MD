# SciML-MD Scientific Validation Report

**Date:** 2026-08-15 · **Branch:** `validation-fixes` (off `topology-contract-fix`)
**Scope:** rare-case detection validation, numerical-value validation, accuracy checks, edge-case testing of the production scoring path (`run_all_proteins.py` / `batch_runner.py` → `scoring/anomaly_v2.py`).
**Data:** the three real runs present in the repo — 1VII (100 frames), 8H0R (100 frames), 1CRN (1001 frames), re-executed end-to-end with `batch_runner` hyper-parameters (lag_tica=5, dim=3, n_clusters=10, lag_msm=5, k=5, window=3, seed=42). No synthetic trajectories were used for accuracy claims; tiny analytic MSMs were used only for unit-level math checks.

---

## 1. Executive summary

The mathematical formulas of all three anomaly signals were verified correct
(`1−π`, `−log P`, k-NN density match hand-computed values exactly). However,
**four implementation defects corrupted the scores the pipeline shipped**, the
worst being a state-indexing bug that made the kinetic channels read the wrong
rows of π and P for 30–43 % of frames. After the fixes, rare-state detection
AUROC (fused score) improved from ~0.83–0.86 to **0.95**, and rare-transition
detection from 0.65–0.79 to **0.81–0.999** on the three real proteins.

| Defect (validated, not hypothesized) | Impact measured at baseline | Fix |
|---|---|---|
| dtraj labels never remapped to MSM active set | rarity channel AUROC 0.45–0.80 (≈ coin flip on 8H0R/1CRN); buggy-vs-correct rarity Spearman as low as **−0.50**; top-10 % anomalous-frame overlap **0 %** (1VII, 8H0R) | `remap_dtraj_to_active_set()` applied inside `compute_kinetic_signals`; `state_symbols.npy` persisted |
| Ties broken by frame order in `rank_normalize` (`argsort(argsort)`) | tie groups of 21–410 frames spread across **20–42 score points** by frame index alone | tie-aware average ranks (`scipy.stats.rankdata`) |
| Last `lag` frames forced to surprise = 0, then ranked | artificial low scores for the trajectory tail | tail = NaN; NaN-aware fusion (`nanmedian` over available channels) |
| Transitions touching disconnected states silently 0 (= least anomalous) | corrected-surprise AUROC on 1CRN rare transitions was **0.02** (inverse ranking) | capped maximal surprise for such transitions |
| Residue scores were pure RMSF (constant offset from frame scores) | Pearson r(residue score, RMSF) = **1.000** on 1VII | displacement-weighted anomaly participation blended with RMSF |
| Residue keys collide across chains | 8H0R: 182 residues → 91 entries (half silently overwritten) | chain-disambiguated keys |
| ASVS export flat | `anomaly_residue.json` = 0.25 for every residue × frame (wrong CSV column name + wrong key type) | reads `score_dynamic`, positional residue mapping |
| `rmsf_residue.json` / `tica_importance.json` empty | `normalized: {}` in all exports | pipeline now writes `residue_scores_rmsf.json` |

---

## 2. Numerical validation

### 2.1 Formula-level checks (analytic 3-state MSM, hand-computed)
`compute_kinetic_signals` reproduces expected values exactly:
rarity(state 0) = 1−π₀ = 0.6 ✔ · surprise(0→1) = −ln 0.1 = 2.3026 ✔ ·
surprise(2→2) = −ln 0.8 = 0.2231 ✔. Conclusion: **the science is right; the
indexing was wrong.**

### 2.2 Active-set mismatch quantified (baseline, real proteins)

| Protein | frames | KMeans labels | MSM states | frames in dropped states | survivor frames reading wrong π row | rarity ρ (buggy vs correct) | fused top-10 % overlap |
|---|---|---|---|---|---|---|---|
| 1VII | 100 | 10 | 8 | 10 | ~30 | 0.68 | 30 % |
| 8H0R | 100 | 10 | 7 | 11 | 28 | **−0.50** | 40 % |
| 1CRN | 1001 | 10 | 7 | 53 | 375 | 0.26 | 25 % |

Because deeptime prunes *middle* states (e.g. labels 2, 4, 6), the surviving
labels shift: a frame in cluster 5 read `π[5]`, which after pruning belongs to
a different physical state. Dropped-state frames with labels < n_states read a
*valid but wrong* π entry — silent corruption, no warning, no NaN.

### 2.3 "Mean score ≈ 50" artifact
Rank normalization maps every signal to a uniform [0,1] distribution, so the
fused mean is ≈ 0.5 by construction: measured means 47.4–48.3 across all
proteins at baseline, 49.5–50.9 post-fix. **The mean frame score carries no
information about the protein and must not be reported as a health metric.**
Meaningful summaries: top-k frame identities, score of known-rare states,
AUROC against operational labels (below).

---

## 3. Rare-case detection accuracy (real proteins, operational ground truth)

Labels derive from the *correctly estimated* MSM only (kinetic channels; the
density channel is never used for labeling, avoiding circularity):
**rare state** = frame in a bottom-quartile-π state or a disconnected state;
**rare transition** = observed ≤ 1 time at lag, or leaves the connected set.

AUROC (AUPRC in parentheses), before → after fixes:

| Task / protein | 1VII | 8H0R | 1CRN |
|---|---|---|---|
| Rare states — fused score | 0.83 (0.50) → **0.95 (0.86)** | 0.86 (0.69) → **0.96 (0.93)** | 0.84 (0.48) → **0.96 (0.89)** |
| Rare states — rarity channel | 0.80 → 1.00* | 0.45 → 1.00* | 0.54 → 1.00* |
| Rare transitions — fused | 0.70 (0.34) → **0.81 (0.69)** | 0.79 (0.55) → **0.84 (0.76)** | 0.65 (0.21) → **1.00 (0.99)** |
| Rare transitions — surprise channel | 0.44 → 0.98 | 0.59 → 0.95 | 0.55 → 1.00 |

\* Rarity-channel = 1.00 against rare-*state* labels is expected by
construction (both derive from π) — it is a consistency check showing the
channel now reads the correct π, not an independent accuracy claim. The
headline numbers are the **fused** rows, whose density channel is independent
of the labels.

At baseline, detection was carried almost entirely by the density channel;
the kinetic channels were noise (0.45–0.59) or anti-signal.

**Channel structure (corrected signals):** rarity vs. surprise Spearman ρ =
−0.19 (1VII), −0.22 (8H0R), −0.58 (1CRN). Transition surprise measures rare
*events*, state rarity measures rare *occupancy*, and they are anti-correlated
in practice — under median fusion they can cancel. Recommendation for the
writeup: report the channels separately for "rare state" vs "rare event"
claims, or consider max-fusion for a combined detector.

---

## 4. Edge-case testing (production path)

`validation/baseline_edge_cases.py` exercises the exact functions the batch
pipeline calls (unlike `tests/test_pipeline_edge_cases.py`, which tests the
parallel implementation in `scoring/signals.py` that the pipeline never
invokes).

| Case | Behavior (post-fix) |
|---|---|
| 10/12/20/50-frame trajectories | run; state-space mismatch occurs in **every** regime and is now remapped correctly |
| < 10 frames | rejected by pipeline gate (functions themselves run at 9) |
| tICA/MSM lag > T | clamped by callers (`min(lag, T//4)`); scorer itself also clamps now |
| constant / near-zero-variance features | deeptime `ZeroRankError` → protein marked failed with logged error (fail-fast, no silent output) |
| single-state trajectory | rarity constant 0, surprise tail NaN, fused finite |
| disconnected two-block trajectory | dropped half now flagged maximally rare/surprising instead of receiving default scores silently; pipeline logs a WARNING with the frame count |
| clusters > frames, k > frames, k=0, empty/constant/single-element arrays, even smoothing windows | all handled, finite outputs |
| invalid labels (−7, 99) | treated as invalid (rarity 1.0), negative labels no longer wrap around via Python indexing |

**Reproducibility finding:** with identical inputs and `seed=42`, tICA
reproduces across machines (per-component r = 0.96–0.9996) but deeptime
KMeans does **not** (ARI 0.43–0.51 vs. the laptop artifacts; 90/100 frames
relabeled, and not by pure permutation). Cross-machine claims should therefore
be made at the level of scores/AUROC, not state assignments — or clustering
should be replaced with a deterministic method (e.g. regular-space clustering,
or sklearn KMeans with fixed `n_init` and documented library versions).

---

## 5. Changes on `validation-fixes`

Modified: `scoring/anomaly_v2.py` (remap + tie-aware ranks + NaN tail +
NaN-aware fusion), `run_all_proteins.py` (mismatch warning, `state_symbols.npy`
artifact, residue-score rewrite, RMSF export), `batch_runner.py` (RMSF export
hook), `tools/export_for_asvs.py` (column/key fixes).
Added: `tests/test_production_scoring.py` (12 correctness tests, all passing),
`validation/` (baseline + post-fix scripts and result JSONs, this report).
Existing test suites: 31/31 still pass. Pre-existing failure unrelated to
these changes: `test_integration.py::test_tica_model_persistence` imports
`run_msm_tica`, a module never migrated from the twin repo.

Interface notes (downstream consumers):
- `residue_scores_dynamic.json` keys unchanged for single-chain proteins;
  multi-chain duplicates now appear as `NAMEnnn_chainK` (previously silently
  dropped). Values now reflect anomaly participation, not pure RMSF.
- `frame_scores_dynamic.csv`: `component_transition_surprise` is empty (NaN)
  for the last `lag_msm` frames — by design.
- New artifact: `artifacts/{ID}/state_symbols.npy` when states were pruned.

## 6. Known remaining limitations

1. `anomaly_v2.py`'s CLI path reconstructs `MarkovStateModel(P, pi)` without
   count information; if fed a raw dtraj with pruned states it can only warn,
   not remap. The batch pipeline path is unaffected (passes the fitted MSM).
2. KMeans cross-machine non-determinism (Section 4).
3. Rare-case ground truth remains operational (MSM-derived), per project
   decision; literature-based hotspot labels for 1VII/8H0R would provide an
   external accuracy benchmark.
4. `requirements.txt` is alphabetically mangled (comments sorted into lines);
   docs reference ~15 files that exist only in the twin repo.
5. Root-level `abc.py` shadows Python's stdlib `abc` module for any script run
   from the repo root — recommend deleting or renaming it.
