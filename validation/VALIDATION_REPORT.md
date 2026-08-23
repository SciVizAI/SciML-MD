# SciML-MD Scientific Validation Report

**Date:** 2026-08-15 · **Branch:** `validation-fixes` (off `topology-contract-fix`)
**Scope:** rare-case detection validation, numerical-value validation, accuracy checks, edge-case testing of the production scoring path (`run_all_proteins.py` / `batch_runner.py` → `scoring/anomaly_v2.py`).
**Data:** the three real runs present in the repo — 1VII (100 frames), 8H0R (100 frames), 1CRN (1001 frames), re-executed end-to-end with `batch_runner` hyper-parameters (lag_tica=5, dim=3, n_clusters=10, lag_msm=5, k=5, window=3, seed=42). No synthetic trajectories were used for accuracy claims; tiny analytic MSMs were used only for unit-level math checks.

---

## 0. ERRATA — adversarial re-validation (added after first issue)

The campaign's own conclusions were re-tested with the explicit goal of
falsifying them (`validation/revalidation.py` → `revalidation_results.json`).
The core Layer-1 findings survived; **three downstream claims did not and are
corrected here. Read this section before citing §5b or §6.**

| # | Original claim | Re-validation verdict |
|---|---|---|
| E-1 | §3.4 / D-04: per-residue scores no longer reduce to RMSF | **Overstated.** The constant-offset defect is fixed, but post-fix residue scores remain ~99 % collinear with RMSF (Pearson r = 0.996 / 0.990 / 0.992 on 1VII / 8H0R / 1UBQ). The anomaly channel still contributes almost no rank information. D-04 is **partially fixed**. |
| E-2 | §5b.1: B-factor correlation (ρ ≈ 0.6) is external evidence for the pipeline's per-residue output | **Not supportable as stated.** Controlling for RMSF, the partial rank correlation between dynamic score and B-factors is −0.019 (8H0R) and −0.181 (1UBQ). The agreement is entirely attributable to the RMSF component; the anomaly component adds nothing. This validates RMSF, not the method. |
| E-3 | §5b.2: the pipeline beats Isolation Forest and LOF on rare-state detection across all four proteins | **Circular.** Labels are defined from π and the rarity channel *is* 1 − π. Removing that channel from the fusion collapses the advantage: the pipeline then loses to LOF on 8H0R (0.83 vs 0.93) and to Isolation Forest on 1CRN (0.76 vs 0.85), and ties on 1UBQ (0.69 vs 0.70). It retains an edge only on 1VII (0.85 vs 0.75). |
| E-4 | §5b.2: the 1UBQ rare-transition result is a candidate headline figure | **Also circular.** Surprise = −log P, and P is row-normalised from the same count matrix that defines the rare-transition labels (Spearman between surprise and negative transition count: 0.69 on 1UBQ, 0.87 on 1CRN). It cannot be presented as a benchmark win. |
| E-5 | §5c.4: VAMP-2 saturating at the dimension bound indicates overfitting | **Correct conclusion, wrong mechanism.** With 100 frames the 20-frame held-out split leaves only 5 lagged pairs at lag 15. The score is degenerate rather than merely overfit. |
| E-6 | §3.5: "twenty-one edge cases" | **Miscount.** The suite contains 25 cases. |

**What survives unchanged.** D-01 was independently re-derived from first
principles: 50 % (1VII), 28 % (8H0R) and 37.5 % (1CRN) of frames would read a
different π value under the old code — matching the original figures exactly.
Defects D-01, D-02, D-03, D-05, D-06, D-07 and D-08 are confirmed fixed, all
12 regression tests pass, and every Layer-2 conclusion (§5c) stands, including
the central finding that 100-frame trajectories cannot support MSM validity
claims.

**Consequence for publication.** The accuracy figures in §4 remain valid as
*internal consistency* measurements — they demonstrate that the corrected code
recovers the kinetic quantities it is designed to compute. They are **not**
evidence that the method finds biologically meaningful anomalies, and the
baseline comparison must not be presented as a benchmark win. Establishing
external validity now requires ground truth that is independent of the MSM:
literature-annotated functional sites, mutational or binding-site data, or
long-trajectory datasets with characterised rare events. What can still be
said honestly is narrower but true — geometric outlier detectors have no
access to kinetic information by construction, so the kinetic channels measure
something they cannot represent.

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

## 5b. Layer 3 — external scientific validation (added same day)

Terminology used with the team: Layer 1 = implementation correctness (§1–4),
Layer 2 = MSM statistical validity (CK / implied timescales / VAMP-2 /
bootstrap — code exists in `msm/` but has NOT yet been run on our systems),
Layer 3 = external validity (below). Script: `validation/layer3_external.py`,
raw numbers: `validation/layer3_results.json`.

### 5b.1 Experimental B-factor comparison (non-circular ground truth)

Per-residue CA B-factors parsed from the raw RCSB entries and compared
against pipeline outputs from the fixed code (Spearman ρ):

| Protein (method) | matched residues | ρ(B, dynamic score) | ρ(B, trajectory RMSF) |
|---|---|---|---|
| 8H0R — βB1-crystallin Y202X (X-ray 1.20 Å) | 178/182 | **0.595** | 0.642 |
| 1UBQ — ubiquitin (X-ray 1.80 Å) | 76/76 | **0.573** | 0.656 |

Reading: ρ ≈ 0.6 between crystallographic flexibility and scores derived from
100-frame toy trajectories is in the range typically reported for MD-vs-B-factor
comparisons, and is *external* evidence the per-residue outputs track real
structural flexibility. The dynamic score correlates slightly below raw RMSF —
expected and desirable: B-factors measure flexibility only, while the dynamic
score deliberately mixes in kinetic-anomaly participation. 1VII is NMR (no
meaningful B-factors) and is excluded by design. 9O6O (Siglec-10, X-ray 2.70 Å)
and 9UNN (NMDA receptor, cryo-EM 3.29 Å) will be added when their runs finish.

### 5b.2 Baseline method comparison (same tICA space, same labels)

Off-the-shelf anomaly detectors vs the pipeline, AUROC on the operational
kinetic labels (labels are MSM-derived, so the fair claim is: *geometric
detectors cannot recover kinetic rarity; the pipeline's kinetic channels add
real signal*):

| Rare-STATE AUROC | 1VII | 8H0R | 1UBQ | 1CRN |
|---|---|---|---|---|
| **Pipeline fused (ours)** | **0.947** | **0.958** | **0.904** | **0.955** |
| IsolationForest | 0.752 | 0.879 | 0.696 | 0.845 |
| LocalOutlierFactor | 0.835 | 0.934 | 0.694 | 0.490 |
| kNN distance only (density ablation) | 0.867 | 0.867 | 0.762 | 0.905 |

| Rare-TRANSITION AUROC | 1VII | 8H0R | 1UBQ | 1CRN |
|---|---|---|---|---|
| **Pipeline fused (ours)** | 0.811 | 0.837 | **0.808** | **0.999** |
| Surprise channel alone | **0.976** | **0.953** | **0.907** | **1.000** |
| IsolationForest | 0.771 | 0.826 | 0.432 | 0.959 |
| LocalOutlierFactor | 0.772 | 0.846 | 0.463 | 0.607 |

Highlights: on 1UBQ rare transitions, geometric detectors are at chance
(0.43–0.49) while the surprise channel reaches 0.91 — kinetic events are
invisible to purely geometric outlier detection. The density-only ablation
shows the fused score consistently beats its own geometric component, i.e.
the MSM machinery earns its keep. (Rarity-channel = 1.0 rows omitted:
circular with the state labels, see §3 footnote.)

## 5c. Layer 2 — MSM statistical validity battery (EXECUTED on our systems)

Script: `validation/layer2_msm_validity.py` · numbers: `layer2_results.json` ·
figures: `validation/figures/{PID}_{its,ck,pi_ci,vamp2}.png` (15 figures).
Run on 1VII, 8H0R, 1UBQ (100 frames each) and 1CRN (1001 frames), base lag 5.

### 5c.1 Chapman–Kolmogorov test
Correct active-set embedding was required: the repo's
`msm/validation.py::chapman_kolmogorov_test` compares `P(kτ)` matrices
estimated on *different* active sets without remapping — the same bug class
fixed in the scoring path. Our implementation embeds each MSM into the full
label space before comparison.

| Protein | fraction of (state, k) points where prediction lies within the 95 % CI | max abs deviation |
|---|---|---|
| 8H0R | 0.94 | 0.24 |
| 1VII | 0.75 | 0.22 |
| 1UBQ | 0.75 | 0.25 |
| 1CRN | 0.63 | 0.32 |

**Interpretation — important:** the test is passed in the weak sense (most
points fall inside the intervals), but the intervals are enormous (±0.3–0.5 in
probability) because each state has only tens of observed transitions. The CK
test therefore has almost **no statistical power** at this trajectory length;
it cannot currently distinguish a Markovian model from a non-Markovian one.
1CRN — the longest trajectory — has the *lowest* pass fraction, consistent
with tighter intervals exposing real deviations.

### 5c.2 Implied timescales
Relative change of the slowest implied timescale over the final lag step:
1VII **0.50**, 1UBQ 0.09, 1CRN 0.06, 8H0R 0.03. Only 8H0R/1CRN approach a
plateau; **1VII does not converge at all** (t₂ rises from 10.7 → 19.8 frames
across the tested lag range). Across all four systems the timescales continue
to drift upward with lag rather than flattening — the classic signature of
insufficient sampling, not of a well-chosen lag.

### 5c.3 Bootstrap uncertainty on π (block bootstrap, n=200, block=10)
Median 95 % CI width relative to π itself:
1VII **1.55×**, 1UBQ **1.53×**, 8H0R **1.11×**, 1CRN **0.56×**.

For the 100-frame systems the confidence interval on a state's stationary
probability is **wider than the value being estimated**. Since rarity = 1−π is
the pipeline's primary kinetic channel, per-state rarity values from
100-frame trajectories should be treated as **unresolved**; only the
rank-level conclusions (which frames are relatively rarer) survive. 1CRN is
roughly 3× better and is the only system approaching usable precision.
Bootstrap replicas also show states dropping in and out of the connected set
(`state_survival_fraction` in the JSON), confirming the connectivity problem
is a sampling artifact rather than genuine kinetic disconnection.

### 5c.4 VAMP-2 model selection — **new defect found**
`msm/select_lag_and_dim.py::compute_vamp2_score` is **numerically unstable on
real MD features**: its Cholesky-inverse whitening with `reg=1e-6` on
ill-conditioned covariances returned scores up to **2.5 × 10⁵**, where a valid
VAMP-2 score is bounded by the dimension (≤ 5). It behaves correctly on
well-conditioned random data, which is why the repo's mock-data tests never
caught it. All hyper-parameter selection based on this function is invalid.

We re-scored with deeptime's own cross-validated VAMP-2 (80/20 split):

| Protein | best (lag, dim) | score | pipeline default (lag 5, dim 3) |
|---|---|---|---|
| 1VII | (15, 5) | 5.00 — **saturated at dim** | 1.76 |
| 8H0R | (15, 5) | 5.00 — **saturated at dim** | 1.51 |
| 1UBQ | (15, 5) | 5.00 — **saturated at dim** | 1.75 |

Saturation at the dimension bound with only ~80 training frames indicates
**overfitting**, not a genuinely better model — so the grid cannot currently
justify hyper-parameters either. VAMP-2-based selection needs to be repeated
on longer trajectories before it can be reported.

### 5c.5 What Layer 2 establishes
The battery is now **implemented, executed, and figured on our own systems**
(no mock data) — that part is complete and reportable. Its scientific verdict,
however, is that **100-frame trajectories cannot support MSM validity claims**:
CK intervals too wide to be informative, timescales unconverged, π uncertain
to ±100 %, VAMP-2 saturating. The Layer-1 correctness results and the Layer-3
external correlations stand on their own; the *kinetic* claims require longer
trajectories (the 50 000-step protocol in `scripts/generate_md_trajectories.py`,
or a public long-trajectory dataset) before publication.

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
