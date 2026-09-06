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

## 0b. POST-AUDIT ROUND (external scientific review, Aug 2026)

Four independent audit reports were commissioned on a research-scientist
platform, including a fresh 5 ns explicit-solvent GROMACS/AMBER99SB-ILDN
simulation of ubiquitin. They corroborated the entire defect register (D-01 to
D-06, E-1 to E-3) with matching numbers, and surfaced two findings we had
missed. Both are now fixed. They also shipped a proposed replacement method
(KSD-RA) whose implementation had defects of its own.

### 0b.1 NEW DEFECT D-10 — equilibration frames were never discarded
The external ubiquitin run traced frames 1-3, which scored 99.7/100 (the
highest in the trajectory), to unrelaxed Cartesian coordinates falling out of
the deposited crystal structure. Our pipeline recorded from step 0 and cut
nothing, so simulation startup was being reported as rare-event biology.
`msm/preflight.py::detect_equilibration` now finds the RMSD plateau and
discards the relaxation phase (capped at 20% of the trajectory). Measured
discards: 1UBQ 7 frames, 1VII 17, 8H0R 20.

### 0b.2 NEW FINDING — every system we have run is UNSUITABLE for MSM anomaly detection
The audit argued that Crambin, locked by three disulfides, has no macrostate
switching and should never have been analysed with a multi-state MSM. We
implemented that as a general screen (`assess_suitability`) and applied it to
all our systems. **All of them fail**: 1VII, 8H0R and 1UBQ each have 100% of
frames within 2 A of a single medoid. They are single-basin ensembles, so the
MSM microstates partition thermal noise rather than conformational states.

This is the deepest explanation yet for the Layer-2 results. Disconnected
states, unresolved pi, powerless CK tests and unconverged timescales are not
independent problems; they are symptoms of running a multi-state kinetic model
on data containing one state. The screen now warns before scoring.

### 0b.3 E-1 RESOLVED — residue attribution is now orthogonal to RMSF
`scoring/residue_attribution.py` implements the audit's KSD-RA concept
(anomaly-conditioned contact flux + dihedral divergence) with four corrections
to their reference implementation, plus explicit RMSF orthogonalisation:

| System | Legacy score vs RMSF | New score vs RMSF |
|---|---|---|
| 1VII | +0.983 | **-0.012** |
| 8H0R | +0.940 | **+0.016** |
| 1UBQ | +0.944 | **+0.021** |

Rankings change completely: 1UBQ's legacy top-6 was the floppy C-terminal tail
(GLY76, GLY75, ARG74, LEU73...), the new top-6 is interior (ARG42, LEU15,
GLN62, LEU71...), with zero overlap. **Orthogonality is not validity** - that
the score is no longer RMSF does not establish it identifies functional
residues. OI-12 remains open.

### 0b.4 Corrections to the audit's own deliverables
- **KSD-RA reference code reintroduced defect D-02**: `argsort(argsort(x))` at
  line 114. Verified: four identical inputs return 0/25/50/75 instead of 37.5
  each. Fixed in our implementation with tie-aware ranks.
- **KSD-RA was memory-infeasible**: a dense (n_frames, n_res, n_res) contact
  array is 2.5 GB for 8H0R at ATLAS length and ~80 GB for 9UNN. Ours streams
  the group means, so memory is O(n_res^2) regardless of trajectory length.
- **KSD-RA dihedral indexing was misaligned**: mdtraj returns phi for residues
  2..N and psi for 1..N-1; pairing them column-wise associates phi(i+1) with
  psi(i). Ours resolves each angle's residue through the atom-index arrays.
- **KSD-RA is unvalidated**: no functional-site enrichment, no RMSF baseline,
  no significance test on the ensemble split, arbitrary 85th-percentile
  threshold. We adopted the idea, not the evidence, and say so in the module.

### 0b.5 CONTRADICTION RESOLVED — ubiquitin 52-60 is rigid, not flexible
Two audit reports disagreed. One called residues 52-60 "intrinsically flexible"
(RMSF 1.88-2.08 vs mean 1.3, sourced from the *thesis*); the other measured
them as hyper-rigid (0.57 +/- 0.07 A, 100% of frames in one cluster, from *new
MD*). We computed it independently on our own trajectory:

| Region | Our RMSF (A) |
|---|---|
| Whole protein mean | 0.60 |
| **Residues 52-60 (the candidate hotspot)** | **0.53 +/- 0.08** |
| beta-sheet core | 0.47 |
| Loop 1 (8-11) | 0.76 |
| C-terminal tail (72-76) | 1.49 |

Residues 52-60 sit at 1.13x the core and are 2.8x LESS mobile than the tail.
Our 0.53 +/- 0.08 A matches the external 0.57 +/- 0.07 A closely despite
entirely different solvent models, force fields and engines. **The rigidity
finding is confirmed and the flexibility claim is wrong**, which also means the
"general thermal flexibility" root cause stated in one report is incorrect; the
methodological-artifact explanation is the right one.

One open thread: our legacy RMSF-collinear score ranked the C-terminal TAIL
highest, not 52-60. So the thesis's 52-60 claim is unlikely to come from E-1
collinearity alone and more plausibly from D-01 active-set misindexing. We
cannot reproduce the original thesis run to confirm this.

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

---

# PART II — SCALE VALIDATION CAMPAIGN (ATLAS, 2026-08-29)

**Branch:** `atlas-validation-phase3-4`
**Data:** ATLAS v2023-03-09 release (1,938 systems), `analysis` archives —
1,000 frames × 3 independent replicates per protein, 100 ns each, protein-only
coordinates.
**Scope:** Phase 3 (frame-level detection at scale) and Phase 4 (residue-level
attribution against external functional labels), plus the trajectory and
suitability instrumentation both depend on.

This part supersedes §0b.2, which concluded that *every* system available was
unsuitable for MSM anomaly detection. That conclusion was correct about the
three systems then in the repo and wrong as a general statement; §7.2 replaces
it with a measured prevalence.

---

## 7. Phase 3 — frame-level validation across 25 proteins

### 7.1 Sample selection

Selection was made from the ATLAS release table **before any trajectory was
downloaded**, using only metadata ATLAS publishes (`validation/atlas_pick.py`).

A first attempt selected on `avg_RMSF` descending and produced 21 systems in
percentile 82–100 of the database — the disordered tail. That sample was
retained as a negative stratum but is not the Phase 3 sample. The reasoning:
an MSM requires discrete metastable states; a single basin has one, and a
diffusive chain never revisits anything, so it has none. Selecting on
fluctuation amplitude alone maximises the second failure mode.

The Phase 3 sample instead uses:

| criterion | value | rationale |
|---|---|---|
| `avg_RMSF` band | 0.9–3.5 Å | database median 1.33 Å; excludes frozen and disordered tails |
| length | 60–250 residues | keeps MSM estimation tractable |
| `coil%` | ≤ 55 | a chain with no secondary structure has no fold to be metastable in |
| redundancy | `non_redundant_protein` only | avoids near-duplicate folds |
| within-band ranking | excess `div_MM` after orthogonalisation vs `avg_RMSF` | see below |

`div_SE` and `div_MM` are **similarity** measures, not diversity measures. This
is not documented in the ATLAS API or on its download page; it was determined
empirically from the release table itself — Spearman ρ against `avg_RMSF` is
**−0.792** and **−0.732** respectively (p ≈ 0, n = 1938). A genuine diversity
measure must rise with fluctuation amplitude; a similarity measure must fall.

Because `div_MM` is 0.73 rank-correlated with `avg_RMSF`, selecting on it
directly would have re-selected amplitude under another name — the same
collinearity trap as erratum E-1. It was therefore regressed against `avg_RMSF`
in rank space and selection made on the **residual**: systems whose structural
diversity exceeds what their amplitude predicts. Residual-vs-amplitude
correlation after orthogonalisation: **ρ = +0.058** (from −0.732).

### 7.2 Suitability screen — the method has a measured domain of validity

`msm/preflight.py::assess_suitability` was extended with a basin census: leader
clustering of CA structures at a 2 Å cutoff, then a count of clusters holding
≥ 5 % of frames.

The screen was previously **one-sided** — it detected only ensembles that were
too tight. On the first ATLAS batch it returned 60/63 "suitable" at 15–44 Å
spread with 0.5 % medoid occupancy, i.e. one frame out of 200. Those are
diffusive chains, and an MSM has as little to find there as in a single basin.

Screen of all 135 trajectories (45 proteins, 3 replicates each):

| verdict | trajectories | share |
|---|---|---|
| suitable | 52 | 39 % |
| marginal | 22 | 16 % |
| unsuitable | 61 | 45 % |

**45 % of trajectories cannot support the analysis.** Aggregated to the protein
(a protein is usable when a majority of its replicates are): **25 of 45 usable**.

This is a product-relevant finding, not only a scientific one. A pipeline that
always returns a hotspot map will return a meaningless one on nearly half of
arbitrary inputs. The gate is now enforced in code and reports which failure
mode applies.

### 7.3 Replicates disagree — 100 ns is not converged

17 of 45 proteins received different verdicts across their three replicates.
`2hnu_A` is single-basin in R1 (top cluster 96.5 %) and eight-basin in R2.
`2p58_A` reads 97 %, 94 %, then suitable. Same protein, same force field, same
temperature; only the seed differs.

Consequence, applied throughout Part II: **replicates are averaged to the
protein before any statistic is computed.** Treating 75 trajectories as 75
samples would inflate the Wilcoxon test approximately threefold.

### 7.4 External cross-check of the geometry pipeline

Independent of the anomaly method, ATLAS publishes `avg_RMSF` per system. Our
RMSF — computed by our loader, our alignment and our fluctuation code from the
downloaded coordinates — was compared against it across 21 systems spanning
2.42–14.41 Å (`validation/phase3_diagnose_traj.py`):

| statistic | value |
|---|---|
| Pearson r | **0.892** (p = 5.5 × 10⁻⁸) |
| Spearman ρ | 0.822 |
| OLS slope / intercept | **0.966** / −0.231 (ideal 1.000 / 0.000) |
| median signed error | −7.8 % |
| median \|error\| | 10.6 % |
| within 10 % / 20 % | 9/21 · 14/21 |
| largest deviation | 4gip_A −40.5 % |

This is the first non-self-referential numerical check in the campaign: ATLAS
computed its value with its own code from its own simulations. The residual
scatter is consistent with sampling and convention differences — we scored one
replicate of the 1,000-frame `analysis` archive against what is presumably an
aggregate over three 10,000-frame runs, which would bias our values low, as
observed. It is **not** consistent with a coordinate-handling fault, which
inflates RMSF by hundreds of percent, not fourteen.

The check also settled a diagnostic question: max pairwise CA RMSD of 15–44 Å
in the first batch was **genuine disorder, not periodic-boundary breakage**.

### 7.5 Results — all four pre-registered criteria pass

Criteria were fixed in Phase 2 and were **not** adjusted. 25 independent
proteins, 75 trajectories, 5 scramble replicates per trajectory, block
permutation with randomised cut points, near mode (junctions placed between
structurally similar frames).

| detector | median AUROC | IQR | ≥ 0.75 |
|---|---|---|---|
| **surprise_only** | **0.816** | 0.774–0.840 | **80 %** |
| pipeline_fused | 0.666 | 0.633–0.705 | 4 % |
| frame_to_frame_rmsd | 0.608 | 0.567–0.647 | 0 % |
| rarity_only † | 0.606 | 0.550–0.623 | 0 % |
| feature_zscore ‡ | 0.595 | 0.547–0.635 | 0 % |
| isolation_forest ‡ | 0.592 | 0.562–0.625 | 0 % |
| local_outlier_factor ‡ | 0.589 | 0.514–0.637 | 0 % |
| density_only ‡ | 0.572 | 0.526–0.605 | 0 % |
| rmsd_from_mean ‡ | 0.487 | 0.446–0.546 | 0 % |

‡ time-blind control · † excluded from C3 by prior decision (Phase 2 finding
P2-3): π is estimated from transition counts and therefore inherits temporal
information, so rarity was never a valid time-blind control.

| criterion | requirement | result | |
|---|---|---|---|
| C1 | median surprise ≥ 0.75 | **0.816** | PASS |
| C2 | beats best time-blind on ≥ 80 % | **100 %** (25/25) | PASS |
| C3 | every time-blind median in 0.40–0.60 | max 0.595 | PASS |
| C4 | Wilcoxon p < 0.05 vs best blind | **p = 3.0 × 10⁻⁸** | PASS |
| C5 | null control at chance | 0.478–0.518 | PASS |

Per-protein surprise ranges 0.668 (`6kty_A`) to 0.895 (`1j5u_A`); 20 of 25
proteins clear the bar.

`best_blind` is the **per-protein maximum** across the five time-blind
detectors, a deliberately demanding baseline. Surprise exceeded it on every
protein without exception.

### 7.6 Null control

The identity-null keeps blocks in their original order — no real junctions —
and labels the same positions. All nine detectors fall in **0.478–0.518**,
confirming the labelled positions carry no intrinsic distinctiveness and the
construction is sound. C5 gates every other criterion; had it failed, nothing
above would be interpretable.

### 7.7 Confound check — the result is −log P, not imputation

`compute_kinetic_signals` assigns the maximum finite surprise to any transition
touching a state the MSM dropped from its active set (defect D-03; the previous
behaviour scored these zero, i.e. *least* anomalous). Splicing frames that were
never adjacent is precisely the operation most likely to create unobserved
transitions, so junction frames could in principle have received high scores by
**imputation rather than measurement**, and the run log shows states being
dropped frequently.

`validation/phase3_confound_check.py`, 10 proteins × 3 scramble replicates:

| measurement | result |
|---|---|
| cap rate at junction frames | **0.0 %** |
| cap rate elsewhere | 0.1 % |
| AUROC of cap indicator alone | **0.499** |
| surprise, all frames | 0.856 |
| surprise, capped frames excluded | **0.856** |

Near-mode splices join structurally similar frames, which fall in
well-connected microstates; essentially no junction transition takes the
imputation path. Excluding capped frames changes the AUROC in the fourth
decimal. **The claim rests on estimated transition probabilities.**

---

## 8. Phase 4 — residue attribution vs external functional sites: NEGATIVE

### 8.1 Why this test exists

Every claim in §7 is **frame-level**. The visualisation product paints
**residues**. Those are different claims, and the residue claim had no external
evidence: the original score correlated with RMSF at ρ 0.94–0.98 (erratum E-1),
the fix drove that to ≈ 0.00, but *not RMSF* is not *correct*.

### 8.2 Ground truth

A residue is functional if any atom lies within 4.5 Å of a ligand, metal ion,
nucleic acid, or partner protein chain **in the deposited crystal structure**.
Waters, cryoprotectants and buffer components (`HOH`, `GOL`, `SO4`, `PEG`, …)
are excluded as crystallisation artefacts. ATLAS simulates the isolated apo
chain, so the label cannot leak into the score — the non-circularity that
errata E-3 and E-4 lacked.

Validity preconditions, applied identically to both arms: ≥ 5 positives, ≥ 5
negatives, prevalence 5–80 %. These exclude cases where no AUROC is
interpretable (`3dso_A` had 3 labelled residues; `4v1g_B` had 77 of 86).

### 8.3 Result

n = 16 proteins (9 excluded: 4 deposited structures contain no ligand at all,
2 label-degenerate, 2 sequence-alignment failures, 1 prevalence).

| criterion | requirement | result | |
|---|---|---|---|
| F1 | median AUROC ≥ 0.65 | 0.556 | **FAIL** |
| F2 | beats RMSF on ≥ 70 % | 62.5 % | **FAIL** |
| F3 | shuffled null in 0.45–0.55 | 0.499 | PASS |
| F4 | Wilcoxon p < 0.05 vs RMSF | 0.900 | **FAIL** |

Attribution median **0.556** against an RMSF baseline of **0.536**. F3 passing
confirms the harness is sound, so the failure is a property of the method, not
of the test. Exploratory secondary analysis on the ligand/ion/nucleic label
alone (n = 10, not pre-registered): 0.565 vs 0.542, beating RMSF on 70 %.

Two proteins scored well — `3bpj_B` 0.748 and `4jgi_A` 0.719, both stronger on
the ligand-only label (0.798, 0.795) — but `5xct_B` (0.405 vs 0.711) and
`1j5u_A` (0.428 vs 0.750) lose badly to flexibility, and the median sits at
chance.

**Conclusion: the residue hotspot map does not identify functional sites and
does not outperform colouring by RMSF.** This result was predicted before the
defect-fix re-run and did not change after it — a deliberate guard against
rationalising a negative away.

### 8.4 What this does and does not mean

It does not invalidate §7. Frame-level detection and residue-level attribution
are separate claims tested separately, and only the second failed. It does mean
no product or publication claim of functional relevance for the residue map is
supported by this evidence.

---

## 9. Revised claim set

**Supported.**

1. Transition surprise detects kinetic discontinuities that time-blind
   detectors largely miss: 0.816 median across 25 proteins, beating the best of
   five blind baselines on 25/25, p = 3 × 10⁻⁸, with a clean null and no
   imputation artefact.
2. The geometry pipeline reproduces an independently published measurement
   (r = 0.892, slope 0.966 vs ATLAS `avg_RMSF`).
3. The method has a measurable domain of validity, enforced in code; 45 % of
   trajectories fall outside it.

**Not supported.**

4. That the residue hotspot map identifies functional sites (§8).
5. That the map adds information beyond RMSF at residue level (§8).
6. That the method detects naturally occurring rare biological events. Phase 1
   used seeded anomalies; Phase 2 and 3 use manufactured splices. Both are
   legitimate tests of sensitivity to kinetic discontinuity, but in both cases
   the anomaly was constructed by us. Multi-temperature data (mdCATH) remains
   the only designed route to a non-circular natural rare-event label.

**Qualifications on claim 1.**

- Time-blind detectors sit at 0.57–0.60, i.e. *slightly above* chance, not *at*
  chance. Four of five hug the top of the acceptance band. The defensible
  phrasing is "substantially outperforms the best time-blind detector", not
  "detects what geometry cannot see".
- Junctions are RMSD-invisible by construction, not invisible in the wider
  feature space. The pilot (n = 3) failed C3 with `feature_zscore` at 0.625;
  it passed at n = 25 with 0.595. The construction was **not** modified in
  response.
- The sample is ATLAS-derived, moderate-flexibility, 60–250 residues,
  non-redundant, single chains simulated apo. Generalisation beyond that
  envelope is untested.
- 1,000 frames per trajectory (`analysis` archive). The 10,000-frame `protein`
  archives would strengthen the MSM estimation underlying every number here.

---

## 10. Additional limitations found in Part II

6. The ATLAS `/parsable` endpoint returns one archive covering three datasets
   (ATLAS, chameleon, DPF). Parsing it without filtering yields identifiers
   that do not exist in the ATLAS namespace — the cause of an initial 20/20
   HTTP 404 rate.
7. `image_molecules` fails on ATLAS `analysis` archives (no bond records). It
   is not needed; §7.4 established the coordinates are sound.
8. Nine of 25 Phase 4 proteins could not be evaluated, four because the
   deposited entry contains no ligand. The ATLAS sample was not selected for
   ligand presence; re-fetching with `--with-ligand` would raise n for the same
   test under the same criteria.
9. `tests/test_chapter9_evaluation.py` fails to import (`No module named
   'experiments'`), breaking a bare `pytest tests/`.

---

## 11. Standards audit (2026-08-29) — amendments to Section 9

A literature audit was run against two bodies of standards that govern this
work: the MSM methodology canon, and the time-series anomaly-detection
benchmarking literature. Four amendments to Section 9 follow, plus three
controls added to the harness.

### 11.1 AMENDMENT — the claim is narrower than Section 9 states

Section 9 claim 1 says the kinetic channel "detects kinetic discontinuities".
Per **Timmer (2000), *Phys. Rev. E* 61, 1342**, rejecting a surrogate null
establishes that the data deviate from the null class — **not** that the
specific mechanism one had in mind is present. Applied here:

> Phase 3 establishes that transition surprise is **sensitive to temporal
> discontinuity**. It does **not** establish that it detects **conformational
> transitions**. Those are different sentences and only the first is earned by
> this design.

The second sentence requires held-out *real* transitions with labels from a
source independent of our pipeline (OI-18). Until that exists, the abstract and
any product copy must use the first form.

### 11.2 AMENDMENT — sampling is 13-27x below the published threshold

**Kozlowski & Grubmüller (2023), *JCTC* 19, 5516** (104,789 MSMs) measured
sampling "tipping points" below which uncertainty *increases* with more data and
timescales are *systematically* underestimated:

| system | residues | tipping point |
|---|---|---|
| Pin WW | 35 | < 1 us |
| HNF HD | 50 | 4-8 us |
| XPCB | 72 | 4-8 us |
| nNOS | 112 | 4-8 us |
| **this campaign** | **60-250** | **0.3 us** |

Two consequences, both stated rather than defended:

1. **Scope.** This standard governs *quantitative kinetic observables* — rates,
   stationary populations, MFPTs, free energies. This campaign reports none of
   them. Phase 3 measures whether a channel *ranks* junction frames above
   ordinary ones, on trajectories processed identically, with a null at chance.
   The detection claim survives; no pi-derived number may enter the abstract.
2. **Corroboration.** The replicate disagreement reported in Section 7.3 (17 of
   45 proteins changing verdict between replicates) is the documented
   below-tipping-point signature, not an artefact of our screen. The same paper
   finds a single long trajectory >3x more accurate than many short ones from a
   common starting structure, so the 3 x 100 ns design cannot be defended by
   appeal to MSM aggregation.

Named alternatives if quantitative kinetics is ever required: **qMSM / IGME**
(Cao et al., *J. Chem. Phys.* 153, 014105 (2020); 159, 134106 (2023)), which
reach comparable accuracy from an order of magnitude less data by carrying a
memory kernel, and **haMSMs** (Suárez et al., *JCTC* 17, 3119 (2021)) for path
observables.

### 11.3 AMENDMENT — VAMP-2 cannot carry model selection alone

**Arbon, Zhu & Mey (2024), *JCTC* 20, 977** analysed ~280,000 MSMs and concluded
variational scores are "only a guide to model selection"; enforcing reversibility
inside the score can make selection inconsistent, and lag time had negligible
influence. This retargets OI-3: "maximise VAMP-2" is no longer a sufficient
justification for hyper-parameters, and defect D-08 (unstable VAMP-2 scorer) is
consistent with the published picture rather than peculiar to us.

### 11.4 Two contributions the audit identified as unprecedented

- **Temporal-scramble validation has no precedent in MD.** No paper was found
  that validates an event detector by permuting trajectory blocks and scoring
  recovery of the artificial junctions against time-blind controls. What exists
  in MD is the weaker *null-model* control (Hess, *Phys. Rev. E* 62, 8438 (2000),
  PCA of random diffusion; Schultze & Grubmüller, *JCTC* 17, 5766 (2021), tICA of
  random walks). The near-exact precedent is outside MD: **Baldassano et al.
  (2017), *Neuron* 95, 709** — an "Interleaved Stories" stimulus switched at 32
  predetermined points, 20 recovered at p < 0.001 against duration-preserving
  permutation nulls. It should be cited as the methodological ancestor.
- **No published MSM applicability criterion exists.** There is no named
  suitability test, applicability domain, or go/no-go checklist in the
  literature; authoritative sources defer explicitly to expert judgment. The
  basin census of Section 7.2, with its measured 45% out-of-domain rate on a
  database sample, is a methodological contribution in its own right. The
  closest published precedent is Kozlowski & Grubmüller's warning flag that
  strong hyper-parameter dependence "is unphysical".

### 11.5 Controls added to the harness

Three controls the benchmarking literature now treats as mandatory were absent
and have been added. All three are reported for every channel and every run.

| control | why | source |
|---|---|---|
| `random_score` | Random scores can look excellent under common protocols; chance must be measured under OUR protocol. Gated as **C6** | Kim et al., *AAAI* 2022 |
| `abs_diff_oneliner` | The literal `abs(diff(x)) > b` on the feature matrix the pipeline consumes. One-liners solve 86% of the Yahoo benchmark | Wu & Keogh, *IEEE TKDE* 35, 2421 (2023) |
| **PR-AUC** | Junction frames are a small minority; ROC flatters imbalance. `metrics()` always returned AUPRC — Phase 3 discarded it | Liu & Paparrizos, *NeurIPS D&B* 2024 |

C6 is an **additional** gate. It can only make the test stricter, never looser,
so adding it after C1-C4 were fixed does not weaken the pre-registration.

### 11.6 Seam geometry — the design decision that was implicit and is now measured

The standard attack on splice-based validation is the **triviality trap**:
joining two arbitrary blocks creates a one-frame discontinuity in every
coordinate, which a one-line detector finds instantly, making "we beat geometric
detectors" unfalsifiable in the wrong direction. Near mode splices at
*geometrically matched* endpoints precisely to avoid this, but the campaign
asserted that rather than demonstrating it.

`run_system` now records, per replicate, the CA-RMSD across every junction and
across every ordinary consecutive step, and reports the ratio and the percentile
a junction occupies within the ordinary-step distribution. **If junctions sit far
out in the tail, the kinetic claim is not falsifiable on this design and must be
withdrawn for that system.**

This is a live risk, not a formality. On the 250-frame villin smoke test the
junction step is 3.05x an ordinary step and sits at the 100th percentile — and
there `abs_diff_oneliner` (0.683) beats `surprise_only` (0.639), exactly as the
critique predicts. That system is single-basin, unsuitable, and has only 9
blocks to match among, so it is a degenerate case; but it demonstrates the
diagnostic works and that the result is not guaranteed. **Phase 3 must be re-run
and the seam figures read before Section 7.5 is cited anywhere.**

---

## 12. RETRACTION — the Phase 3 headline claim does not survive its own control

**Date:** 2026-08-29, same day. **Run:** `validation/phase3_results_v2.json`,
25 proteins, 75 trajectories, identical to Section 7.5 except that the three
controls of Section 11.5 were active.

### 12.1 The result

| detector | AUROC | PR-AUC | >= 0.75 bar |
|---|---|---|---|
| **`abs_diff_oneliner`** | **0.839** [IQR 0.790-0.866] | **0.227** | **92%** |
| `surprise_only` | 0.814 [IQR 0.774-0.840] | 0.082 | 80% |
| `pipeline_fused` | 0.665 | 0.051 | 4% |
| `frame_to_frame_rmsd` | 0.608 | 0.041 | 0% |
| best time-blind (`feature_zscore`) | 0.595 | 0.035 | 0% |
| `random_score` (null) | 0.506 | 0.018 | 0% |

Head to head, per protein, replicates averaged:

- the one-line detector beats transition surprise on **17 of 25 proteins**
- Wilcoxon, one-liner > surprise: **p = 4.4 x 10^-3**
- on PR-AUC — the honest measure for a rare positive class — the one-liner is
  **2.8x** better (0.227 vs 0.082)

`abs_diff_oneliner` is four lines: standardise the feature matrix, take
`abs(diff(...))`, max over features, max over the lag window. It contains no
tICA, no clustering, no MSM, no stationary distribution and no transition
matrix.

### 12.2 Why C1-C6 still report "pass", and why that does not help

All six criteria pass. They are also **insufficient**, and this run is what
exposed it. C2 and C3 were written against the **time-blind** family
(`rmsd_from_mean`, `feature_zscore`, `isolation_forest`,
`local_outlier_factor`, `density_only`). The trivial **temporal** detector was
never in the comparison set, because until 2026-08-29 the harness did not
contain one. The criteria are satisfied and the claim is still dead.

This is precisely the failure mode Wu & Keogh (*IEEE TKDE* 35, 2421, 2023)
documented: a benchmark that omits the one-line baseline produces an illusion
of progress. We were inside that illusion for the whole campaign.

### 12.3 Root cause: the junctions are not geometrically invisible

| measurement | value |
|---|---|
| junction step, median CA-RMSD | **3.432 A** |
| ordinary consecutive step, median | 1.653 A |
| ratio | **2.19x** |
| percentile a junction occupies among ordinary steps | **100th** |

Near mode joins each block to the unused block whose first frame is closest to
the current block's last frame. With 9 blocks over 1,001 frames the matcher has
almost no choice: the "closest available" start is still twice an ordinary step
away. The construction was designed to make junctions geometrically ordinary and
**it does not achieve that on ATLAS-length trajectories**. Section 7.5's
description of the junctions as geometrically invisible is withdrawn.

### 12.4 What is retracted and what stands

**Retracted.** Section 9 claim 1, in its published form. Transition surprise does
**not** detect kinetic discontinuities that a trivial detector misses. On this
construction it is beaten by four lines of numpy.

**Still standing, and much weaker than the retracted claim.** Surprise (0.814)
substantially exceeds every **time-blind** detector (best 0.595), on 25/25
proteins, p = 3 x 10^-8, with a clean null. That says a temporal method beats
static methods on a temporal task, which is close to tautological and is not a
result worth a paper on its own.

**Unaffected.** Section 7.2 (domain of validity, 45% out-of-domain), Section 7.4
(external RMSF cross-check), Section 8 (Phase 4 negative), and Section 7.7 (the
confound check) are independent of this and stand as reported.

### 12.5 What must happen next, in order

1. **Do not cite Section 7.5.** It is superseded by this section.
2. **Rebuild the construction so junctions are genuinely invisible**, and
   re-register criteria BEFORE running. This is legitimate repair of a test that
   does not test what it claims, not result-shopping - provided the failure
   above is reported alongside whatever comes next. Three candidate fixes:
   (a) match block endpoints on the **full standardised feature vector** rather
   than CA-RMSD, since the one-liner reads features, not RMSD;
   (b) use **many more, shorter blocks** so the greedy matcher has real choice;
   (c) accept a junction only if its feature-space step falls **below the median
   ordinary step**, rejecting and re-cutting otherwise.
   Option (c) makes invisibility a hard constraint rather than a hope, and is
   the one to try first.
3. **Add `abs_diff_oneliner` to the C2/C3 comparison families** so no future run
   can pass while losing to a one-liner.
4. If, on a corrected construction, surprise still loses to the one-liner, the
   honest conclusion is that the MSM machinery is unnecessary for this task and
   the product should say so.

---

## 13. Phase 5 — channel dissociation, measured within-subject

**Run:** `validation/phase5_results_v3.json`, 12 proteins, 4 anomaly durations,
3 replicates each, relocated + null conditions.

### 13.1 Two corrections made during this phase, both declared

**The first null was unsatisfiable by construction.** v1 required every channel
to sit at chance in the null. It cannot: the source window is deliberately drawn
from the structurally distant quartile, so its frames are unusual whether or not
they were moved. `rmsd_from_mean` scored 0.837 in the null - correctly reading
frame identity. Two effects were confounded:

- **identity** - these frames are structurally unusual
- **relocation** - these frames are in the wrong temporal position

The null isolates identity. All criteria were therefore restated on the
**relocation effect = AUROC(relocated) - AUROC(null)**, per protein. This is
stricter than the original, not looser: a channel must now beat its own
identity-only score rather than beat chance.

**The random baseline was not random.** `random_score` was seeded
`default_rng(seed)` with `seed` fixed at 42, so every replicate and every
protein received the *identical* vector - one arbitrary fixed ranking. It duly
picked up positional structure (0.614 at D=1). It is now seeded from a CRC of
the coordinates, so it varies with the permutation and stays reproducible. **The
C6 value of 0.506 reported in Section 12.1 was computed with the broken version
and is being re-run.**

### 13.2 Result

Relocation effect by anomaly duration:

| channel | D=1 | D=4 | D=16 | D=64 |
|---|---|---|---|---|
| `abs_diff_oneliner` | **+0.274** | +0.256 | +0.071 | +0.019 |
| `surprise_only` | +0.149 | **+0.311** | **+0.102** | +0.021 |
| `pipeline_fused` | -0.005 | +0.094 | +0.013 | +0.004 |
| `rarity_only` | +0.023 | -0.017 | +0.028 | +0.018 |
| `rmsd_from_mean` | **+0.000** | **+0.000** | **+0.000** | **+0.000** |
| `feature_zscore` | **+0.000** | **+0.000** | **+0.000** | **+0.000** |
| `density_only` | -0.005 | -0.005 | -0.003 | -0.001 |
| `isolation_forest` | -0.003 | -0.014 | +0.001 | -0.012 |
| `frame_to_frame_rmsd` | -0.190 | -0.117 | +0.008 | -0.007 |
| `random_score` (null) | +0.036 | -0.128 | -0.057 | -0.013 |

| criterion | result | |
|---|---|---|
| G1 surprise effect falls with duration | rho = -0.415, p = 3.4e-03 | PASS |
| G2 surprise effect > 0 at D=1 on >= 70% | | PASS |
| G3 surprise effect exceeds every channel at D=1 | one-liner +0.274 vs surprise +0.149 | **FAIL** |
| G4 fusion never best at any duration | | PASS |
| G5 random at chance in both conditions | | PASS |

### 13.3 What this establishes

1. **Static detectors are blind to relocation, exactly.** `rmsd_from_mean` and
   `feature_zscore` score **+0.000** at every duration. Their high absolute
   AUROCs come entirely from frame identity. The temporal/static dissociation is
   clean and complete.
2. **P1-2 is explained, not a defect.** Surprise's response decays monotonically
   with dwell length (rho = -0.415, p = 3.4e-03) because `-log P` scores the
   MOVE, not the DWELL: once inside a state, its internal transitions are
   ordinary. P1-1 and P1-2 are the two ends of this axis and should be reported
   as a characterised operating envelope, not carried as open criticals.
3. **D-17 confirmed: fusion never wins.** `pipeline_fused` is below the best
   single channel at every duration. The product should ROUTE to a channel, not
   fuse. This also closes OI-9.

### 13.4 What this does NOT establish, and it is the same finding as Section 12

**G3 fails.** At D=1 the four-line `abs_diff_oneliner` has nearly twice the
relocation effect of transition surprise (+0.274 vs +0.149). Surprise leads only
at intermediate durations (D=4: +0.311 vs +0.256; D=16: +0.102 vs +0.071).

Two independent experiments, different constructions, now agree: **the MSM
machinery does not beat a trivial temporal detector.** Section 12 found it on
spliced junctions; Section 13 finds it on relocated excursions with the
identity confound removed. This is no longer attributable to one flawed test
design.

The remaining defensible statement is narrow and should be stated exactly:

> Transition surprise responds to temporal displacement in a way that static
> detectors do not at all (+0.000), and its response is duration-dependent in
> the manner `-log P` predicts. It does not outperform a four-line difference
> detector on the same task.

---

## 14. Phase 6 — early warning: the third independent design, and the clearest negative

**Run:** `validation/phase6_results.json`. **Unmodified** ATLAS trajectories, no
splicing, no relocation, real dynamics. 25 proteins attempted, 9 produced enough
events, 61 events total.

### 14.1 Design and the correction L5 caught

Events are basin changes (leader clustering, 2 A) persisting >= 20 frames. The
event label is geometric, which is deliberate and not circular: the question is
TIMING relative to geometry's own firing time, and a geometric detector cannot
fire before the geometry moves. Lookahead is matched - surprise,
`abs_diff_oneliner` and `frame_to_frame_rmsd` all read the same [t, t+lag]
window.

**A harness defect was caught by L5 mid-run and fixed.** v1 defined firing as
the first frame above the detector's own 95th percentile. That rule is base-rate
driven: a random score has 5% of frames above its own 95th percentile, so in a
50-frame window it crosses 93% of the time and the FIRST crossing is early by
construction. Measured null lead: 38 frames. The live run duly showed
`random_score` at 35-41 frames, ahead of both real detectors. Firing is now the
**peak** within the window, whose null distribution is uniform, so the expected
lead is exactly `max_lead/2` for every detector regardless of how often it
fires. The peak is always defined, which also removes a selection bias: v1 threw
away precisely the events where the one-liner was quiet.

### 14.2 Result

Median lead in frames, 61 events, paired against `random_score`:

| detector | median lead | vs random |
|---|---|---|
| `rarity_only` | 31.0 | earlier, p = 0.132 (n.s.) |
| `abs_diff_oneliner` | 28.0 | n.s., p = 0.435 |
| `surprise_only` | 26.0 | n.s., p = 0.487 |
| `density_only` | 26.0 | n.s., p = 0.348 |
| **`random_score`** | **25.0** | reference (uniform null = 25.0) |
| `rmsd_from_mean` | 18.0 | **LATER, p = 0.005** |
| `frame_to_frame_rmsd` | 16.0 | **LATER, p = 0.024** |

| criterion | result | |
|---|---|---|
| L1 median lead over one-liner > 0 | +1.0 frame | pass (meaningless, see below) |
| L2 leads on >= 60% of events | **50.8%** | **FAIL** |
| L3 Wilcoxon p < 0.05 | **p = 0.526** | **FAIL** |
| L4 surprise lead > random lead | 29.0 vs 28.0 | pass (meaningless) |
| L5 random at uniform expectation | 28.0 vs 25.0 | pass |

### 14.3 Reading it honestly

**L1 and L4 "pass" and mean nothing.** L1 is a one-frame median difference; L4 is
29 vs 28. L2 at 50.8% is a coin flip and L3 at p = 0.53 is the absence of an
effect. The correct summary is that surprise and the one-liner are
indistinguishable on timing, and both are indistinguishable from random.

**Surprise's peak carries no information about when a transition occurs.** Its
lead distribution is uniform over the pre-event window, exactly like a random
score (26.0 vs 25.0, p = 0.487).

**The only detectors that locate the event are the geometric ones - and they
locate it LATE, which is correct.** `rmsd_from_mean` (18.0, p = 0.005) and
`frame_to_frame_rmsd` (16.0, p = 0.024) peak significantly closer to the event
than chance. They fire when the structure moves, because that is what they
measure. There is no precursor signal for the kinetic channel to have caught.

**One thread survives, weakly.** `rarity_only` is the only channel trending
earlier than random (31.0 vs 25.0), at p = 0.132 - not significant, on 9
proteins. It is the one positive direction left anywhere in the campaign and it
is exactly the channel mdCATH would test properly.

### 14.4 Sample limitation, and what it says on its own

16 of 25 proteins produced fewer than three persistent basin changes; 8 produced
none at all. At 100 ns most of these proteins do not undergo persistent
conformational transitions, which is independent corroboration of the sampling
finding in Section 11.2 and of the 45% out-of-domain rate in Section 7.2. The
9-protein sample is thin and is reported as such.

### 14.5 Where the campaign now stands

Three independent designs, three different constructions, one conclusion:

| phase | construction | question | outcome |
|---|---|---|---|
| 3 | spliced junctions | detection | one-liner 0.839 vs surprise 0.814 |
| 5 | relocated excursions | detection, identity cancelled | one-liner +0.274 vs surprise +0.149 at D=1 |
| 6 | unmodified trajectories | timing | both indistinguishable from random |

**Transition surprise is not earning its place.** It reads temporal order -
Phase 5 showed static detectors at exactly +0.000 while surprise responds - but
that capability does not translate into detecting, ranking or anticipating real
conformational transitions better than four lines of numpy.

This is now a settled negative, not a pending question, and no further variation
on the same theme should be run without a new hypothesis behind it.

---

## 15. Phase 7 — the rarity channel, and a harder finding about the data

**Run:** `validation/phase7_results_v3.json`. 25 proteins attempted, 7 excluded
by the coverage gate, 18 analysed. Two earlier versions were discarded; both
failures are documented below because both were mine.

### 15.1 The question

Rarity's claim is not a detection claim. `1 - pi(s)` asserts that state s is
thermodynamically improbable, which is a CALIBRATION claim and is testable
without constructing any anomaly. ATLAS ships three independent replicates per
protein, so: discretise the pooled ensemble, fit the MSM on replicates 1-2, and
ask whether pi predicts the state frequencies of the held-out replicate 3.

The baseline that matters is `empirical_train` - the raw fraction of training
frames per state, with no MSM, no transition matrix, no eigenvector. This is the
rarity-channel analogue of `abs_diff_oneliner`.

### 15.2 Two harness failures, both caught before publication

**v1 was uninterpretable.** "Rare" was defined as the bottom decile of held-out
population, but a median 26 of 76 training states are never visited by the
held-out replicate, so the 10th percentile IS zero and the label became "every
unvisited state". Spearman ran on one enormous tie block and raw counting scored
rho = -0.508 on 24 of 24 proteins. A no-model baseline cannot genuinely
anti-predict held-out frequency that consistently. Discarded, not reported.

**v2 fixed the tie block and the negative correlation survived** - pi -0.515,
counting -0.548 on shared support. At that point the honest options were "the
harness is still broken" or "this is real", and there was no way to tell from
the numbers alone.

**The control that settled it (now permanent as R7).** Same clustering, same
counting, but the train/test split ignores which replicate a frame came from:

| protein | replicate split | random split |
|---|---|---|
| 1g2r_A | -0.548 | **+0.540** |
| 6kty_A | -0.356 | **+0.639** |
| 4zya_A | -0.576 | **+0.632** |
| all 18 (median) | -0.515 | **+0.620** |

A random split recovers a strong positive correlation, as it must. The harness
is sound. The negative correlation belongs to the replicates.

### 15.3 Result

| predictor | median rho | AUROC(rare) | missed mass |
|---|---|---|---|
| pi (MSM) | **-0.515** | 0.320 | **0.253** |
| count (no MSM) | -0.548 | 0.303 | 0.312 |
| uniform | 0.000 | 0.500 | 1.000 |
| shuffled pi (null) | 0.009 | 0.523 | 0.095 |
| **R7 random-split control** | **+0.620** | | |

| criterion | result | |
|---|---|---|
| R1 pi predicts held-out populations | rho = -0.515 | **FAIL** |
| R2 pi beats uniform on rare states | 0.320 vs 0.500 | **FAIL** |
| R3 pi beats counting on >= 60% of proteins | | PASS |
| R4 Wilcoxon pi vs counting | p < 0.05 | PASS |
| R5 shuffled pi at chance | rho 0.009, AUROC 0.523 | PASS |
| R7 random-split control clearly positive | +0.620 | PASS |
| R6 missed mass (reported, not gated) | pi 0.253 vs count 0.312 | pi better |

### 15.4 What this means, and it is not what R1's failure looks like

**R1 does not fail because pi is a bad estimator. It fails because there is no
stable equilibrium distribution to estimate.** Within states visited by BOTH the
training and held-out replicates, the states one replicate occupies most are the
states the others occupy least - rho = -0.515, against +0.620 when replicate
identity is removed. Three 100 ns runs of the same protein, same force field,
same temperature, do not sample a common distribution.

This is the sharpest statement of replicate non-convergence in the campaign,
sharper than the verdict-flipping of Section 7.3 and the coverage gaps of
Section 14.4, and it is measured directly rather than inferred. It is exactly
the below-tipping-point regime of Kozlowski & Grubmüller (Section 11.2), where
sampling uncertainty dominates and populations are systematically wrong.

**Consequences.**

1. `1 - pi` must not be presented to users as improbability on 100 ns data. Not
   because the arithmetic is wrong, but because the quantity it estimates is not
   determined by the data.
2. Any pi-derived number - stationary populations, free energies, MFPTs -
   inherits this and should not be reported at this sampling level. This is the
   concrete justification for the scope statement in Section 11.2.
3. The coverage gate excluded 7 of 25 proteins outright, including `4v1g_B`
   where 93% of the held-out replicate occupies states the training replicates
   never entered.

**The one thing pi does earn.** R3, R4 and R6 all favour pi over raw counting:
it correlates less badly, and it misses less held-out probability mass (0.253 vs
0.312). The MSM does add something over counting frames. But "less wrong than
counting, while both are wrong" is not a claim to build a product on.

### 15.5 Status of the rarity channel

The Phase 6 hint - rarity as the only channel trending earlier than random -
does not survive contact with a calibration test. Rarity joins surprise: the
machinery is not defective, the data cannot support what it claims. Testing it
properly needs sampling past the tipping point (OI-15, the 10,000-frame
archives) or a dataset that supplies the populations directly (OI-18, mdCATH).

---

## 16. OI-27 — the criteria are amended so the retraction cannot recur silently

### 16.1 The defect being closed

Section 12.2 recorded that C1–C6 all reported "pass" on a run whose central
claim was dead, because C2/C3/C4 raced `surprise_only` only against the
**time-blind** family. A trivial *temporal* detector — `abs_diff_oneliner`,
four lines, no model — was in the harness as of 2026-08-29 and was **reported**
in the distribution table, but it was not **gated on**. Reporting a baseline
that no criterion can fail against is not a control; it is a footnote.

This is the same class of defect as the one Wu & Keogh describe, one level up:
we fixed the missing baseline and left the missing *comparison*.

### 16.2 Amendment

| | Before | After (2026-08-30) |
|---|---|---|
| **C2** | surprise > best **time-blind** on ≥80% of proteins | surprise > best of (**time-blind ∪ trivial-temporal**) on ≥80% |
| **C3** | median of every time-blind baseline in 0.40–0.60 | **unchanged** |
| **C4** | Wilcoxon vs best time-blind | Wilcoxon vs that same widened best baseline |
| **C7** | — | **new**: surprise > best trivial temporal detector on **PR-AUC**, Wilcoxon p < 0.05 |
| **Phase 2 B2** | surprise > `frame_to_frame_rmsd` | surprise > best of `TRIVIAL_TEMPORAL`, on ROC-AUC **and** PR-AUC |

`TRIVIAL_TEMPORAL = {abs_diff_oneliner, frame_to_frame_rmsd}`.

**C3 is deliberately left alone.** It is the in-distribution sanity check, and
only *time-blind* detectors are supposed to sit at chance. A trivial temporal
detector firing on a splice is correct behaviour, not a failed control — moving
it into C3 would have made C3 unsatisfiable and disguised a real signal as a
harness fault.

**Why this does not weaken the pre-registration.** Every change is a maximum
taken over a **superset**, so the new threshold is ≥ the old one at every data
point; C7 is an additional gate. Neither can let through a result the original
criteria would have rejected. This is the same argument used when C6 was added
on 2026-08-29, and it is the only kind of post-hoc criterion change this
campaign permits: strictly-tightening, declared, and applied retroactively to
every stored run.

### 16.3 Retroactive re-scoring of the Phase 3 run

The tightened criteria were applied to the **unchanged** Phase 3 v3 data — same
25 proteins, same 75 trajectories, same scores. Only the comparison family moved.

| Criterion | Pre-OI-27 | Post-OI-27 |
|---|---|---|
| C1 median surprise AUROC | **PASS** 0.814 | **PASS** 0.814 |
| C2 win fraction | **PASS** 1.00 | **FAIL** 0.32 |
| C3 blind medians in band | **PASS** | **PASS** |
| C4 Wilcoxon p | **PASS** 3.0×10⁻⁸ | **FAIL** 0.996 |
| C6 random at chance | **PASS** 0.503 | **PASS** 0.503 |
| C7 PR-AUC vs trivial | — | **FAIL** 0.082 vs 0.249, p = 1.000 |
| **overall_pass** | **TRUE** | **FALSE** |

The criteria now reproduce the retraction on their own, from the same file, with
no human reading required. `validation/phase3_results_v3.json` carries both
blocks: the original verbatim under `acceptance_pre_OI27_2026-08-29`, the
re-scored one under `acceptance`.

The win fraction falling from 1.00 to 0.32 is the whole finding in one number.
Against detectors that cannot see time, the MSM wins on every protein. Against
a detector that can, it loses on two thirds of them.

### 16.4 Regression cover

Three tests added (`tests/test_production_scoring.py`, now 28 passing):

- `test_oi27_run_cannot_pass_while_losing_to_the_oneliner` — a synthetic run at
  exactly the retracted numbers (surprise 0.814, one-liner 0.839) must fail C2,
  C7 and `overall_pass`, **and** must show `legacy_blind_only_win_fraction ≥ 0.80`,
  pinning the fact that the old criteria would have waved it through.
- `test_oi27_genuine_win_still_passes` — a real win (0.88 vs 0.60) must still
  pass C1–C7, so the tightening cannot be mistaken for a blanket rejection.
- `test_oi27_phase2_b2_family_includes_the_oneliner` — asserts the one-liner is
  in `TRIVIAL_TEMPORAL` and that `TRIVIAL_TEMPORAL ∩ TIME_BLIND = ∅`, so no
  future edit can quietly move it into C3's family and neutralise the gate.

**OI-27 closed.** D-24 logged: *reported-but-ungated baselines*, a defect class,
not an instance — every future channel added to the harness must be assigned to
a family (`TIME_BLIND`, `TRIVIAL_TEMPORAL`, `NULL_CHANNELS`) and every family
must appear in at least one criterion.

---

## 17. OI-31 — the product claim, settled

### 17.1 Why this needed settling

Sections 12–15 record four negatives. None of them changed a single line of
user-facing copy. `README.md` still opened with "detect dynamic hotspot
residues"; `PIPELINE_OVERVIEW.md` still asserted that π "directly identifies
thermodynamically rare conformations" and that the residue map "often
identifies: hinge residues, allosteric nodes, domain linkers." A validation
campaign that produces retractions the product does not absorb is an
exercise, not a control.

### 17.2 The decision

**The detection claim is withdrawn.** SciML-MD is not a hotspot detector, an
anomaly detector, or an early-warning system, and no output of it may be
described as a detection at the sampling this pipeline runs on.

**The product is re-scoped to what validated:** an *applicability-domain and
diagnostic tool* for Markov state models of protein trajectories. The
capability that survived is also the one with no published equivalent — the
suitability screen that fails 45% of ATLAS trajectories (§11.4). It was built
as a preflight utility for the real product. It turns out to be the product.

### 17.3 `CLAIMS.md`

A claims register now sits at the repository root and is the single source of
truth. Every claim carries a status and a citation into this report:

- **SUPPORTED (S-1…S-5)** — the applicability screen; the externally verified
  RMSF profile (r = 0.892, slope 0.966, median |err| 10.6%, max 40.5%); channel
  non-redundancy (+0.000 vs +0.149 relocation effect); determinism; correct
  implementation of standard methods.
- **WITHDRAWN (W-1…W-9)** — nine specific sentences, each mapped to the file and
  line where it appears and to the section that killed it.
- **OPEN (O-1…O-3)** — the residue/function correspondence (OI-16), whether the
  kinetic channel works given adequate sampling (OI-15/OI-18), and whether the
  screen predicts downstream MSM quality.

The rule is: no citation, no claim. A new claim requires a row with a report
section reference *before* the copy ships.

### 17.4 The scope discipline that cuts both ways

O-2 is the row that matters most for honesty in the other direction. Every
negative in this report is conditional on **≤100 ns sampling**, which is 13–27×
below the published convergence threshold for proteins of this size (§11.2).
"The kinetic channel does not work" is an overclaim in exactly the way "the
kinetic channel detects anomalies" was. The register requires negatives to be
stated with their scope.

### 17.5 Copy amended

- `README.md` — subtitle rewritten; a scientific-status banner added above the
  fold; the `score_dynamic`, `component_rarity` and `component_transition_surprise`
  column descriptions annotated with their withdrawal rows; the residue-score
  section carries an explicit "do not rank, threshold, or publish these as
  detections" warning.
- `PIPELINE_OVERVIEW.md` — heading rewritten and a banner added enumerating
  every withdrawn interpretive claim in the document. The file is retained
  deliberately: its description of *what the code computes* is accurate and
  unchanged. Only the interpretation was ever wrong.

### 17.6 One consequence left open

The word "anomaly" asserts the withdrawn claim in a single word, and it is in
the repository name (`ensemble-anomaly-maps`) and in the module path
(`scoring/anomaly_v2.py`). Renaming is recommended in `CLAIMS.md` §4 but not
done here — it breaks import paths and the public clone URL, and it is a
decision about the project's identity rather than a correction. Logged as a
recommendation, not a defect.

**OI-31 closed.**

---

## 18. Phase 8 — the recut benchmark is not constructible, and that is the finding

### 18.1 What Phase 8 was going to be

Section 12 retracted the Phase 3 headline because the spliced junctions sat at
the **100th percentile** of ordinary frame-to-frame steps (3.43 Å against a
1.65 Å median). The diagnosis was that the benchmark handed `abs_diff_oneliner`
exactly the signal it reads, and that the pipeline's actual hypothesis — that an
MSM sees transitions which are *kinetically improbable but geometrically
unremarkable* — had therefore **never been tested**.

The fix was to be a reject-and-recut benchmark: accept only splices whose
junction step falls *inside* the ordinary step distribution, in both RMSD space
and the standardised feature space the one-liner reads, so that both trivial
detectors are at chance by construction and only a model of the dynamics can
flag the seam.

### 18.2 The precondition nobody had checked

That design has a precondition: **such frame pairs must exist.** Consecutive
ATLAS frames are 100 ps apart and barely move. If no two temporally distant
frames are ever as close as two consecutive frames, an invisible seam cannot be
built at all.

`validation/phase8_feasibility.py` measures this and nothing else. For every
trajectory it compares the ordinary consecutive-frame step distribution against
every candidate splice — all frame pairs separated by ≥100 frames (10 ns). A
pair is *admissible* at percentile P if its step is ≤ the Pth percentile of
ordinary steps in **both** spaces. The decision rule was fixed before running:
proceed only if a majority of trajectories are constructible at P50.

### 18.3 Result (25 ATLAS proteins, R1)

| Percentile | Fraction of trajectories constructible | Median admissible pairs |
|---|---|---|
| P10 | 0.00 | 0 |
| P25 | 0.00 | 0 |
| **P50** | **0.04** | **0** |
| P75 | 0.12 | 1 |
| P90 | 0.32 | 59 |
| P99 | 0.80 | 4,289 |

**Verdict: ABANDON.** One trajectory in 25 is constructible at P50.

The single decisive statistic: *where does the closest temporally distant frame
pair sit inside the ordinary step distribution?*

```
min 8.0    median 64.3    max 98.3    (percentile, n = 25)
only 7/25 trajectories have their closest distant pair below the ordinary median
```

In a typical ATLAS trajectory, **the two most similar frames that are 10 ns
apart are still further apart than a median pair of consecutive frames.**

### 18.4 F-2 — the finding, and it unifies the other four

The protein essentially never returns to where it has been. Over 100 ns there is
**no recurrence**. This is not a statement about our estimator; it is a measured
property of the trajectories, obtainable in one number without fitting anything.

That single fact explains every negative in this report:

- **No recurrence → no revisited states.** So π has nothing to converge to, and
  replicates anti-occupy rather than agree (§15, ρ = −0.515 against a +0.620
  control). Phase 7's finding is a corollary of Phase 8's.
- **No recurrence → no repeated transitions.** So the transition matrix is
  estimated almost entirely from singletons, and "surprise" cannot separate a
  rare transition from an unseen one. Phases 3, 5 and 6 follow.
- **No recurrence → no geometrically invisible seam exists.** So Phase 3's
  junctions *had* to be geometrically obvious, and `abs_diff_oneliner` *had* to
  win. §12 called this a design mistake. It was not: it was the only benchmark
  the data admits.

**Phase 3 did not fail on design. The experiment it was attempting cannot be run
on 100 ns trajectories.** Four scattered negatives collapse into one root cause.

### 18.5 What this does to the claims

This is *better* for the project than four independent failures, for three
reasons.

1. **The detection hypothesis is restored to OPEN.** It was never tested and,
   on this data, was never testable. "The kinetic channel does not detect
   anomalies" must not be written; the supported statement is "the kinetic
   channel cannot be tested at ≤100 ns, because the data contains no
   geometrically invisible transitions to test it on."
2. **It yields a new supported claim, and a cheap one.** Recurrence is a
   single-number, model-free sampling-adequacy diagnostic: compute the closest
   temporally distant frame pair, express it as a percentile of the ordinary
   step distribution, and you know whether *any* MSM quantity on that trajectory
   can mean anything — before fitting a model. This complements the basin census
   (S-1), which asks whether states exist; recurrence asks whether they are ever
   revisited. Both are needed and neither is published elsewhere as a gate.
3. **It makes OI-15 decisive rather than exploratory.** The 10,000-frame
   archives give 10× finer time resolution over the same 100 ns, which does *not*
   fix recurrence — recurrence is a function of trajectory *length*, not
   sampling rate. The correct target is therefore longer trajectories (mdCATH's
   high-temperature replicates, or the 4–8 μs regime Kozlowski & Grubmüller
   identify), not finer ones. **This changes the data plan.**

### 18.6 Caveats, stated

- `MIN_GAP` is fixed at 100 frames (10 ns). A shorter gap would admit more
  pairs, but a splice spanning <10 ns is not a meaningful temporal anomaly.
  Sensitivity to this choice is untested.
- Measured on R1 of each system only, at 1,001 frames (no subsampling applied).
- Admissibility is a *necessary* condition for the benchmark, not a sufficient
  one; systems passing at P50 were not further checked for usability.
- This measures recurrence in RMSD and in the tICA feature space. A trajectory
  could in principle recur in a coordinate neither space resolves.

**Phase 8 closed as ABANDON-with-finding. F-2 logged. No detector was scored.**

---

## 19. Phase 8b — should we download mdCATH? Measured, not assumed

### 19.1 The confound that would have faked a positive

mdCATH saves a frame every **1 ns**; ATLAS every **100 ps**. The Phase 8 probe
compares candidate splices against the *consecutive-frame* step distribution.
At a 10× coarser stride, consecutive frames are further apart geometrically, the
bar rises, and almost any temporally distant pair slips under it.

Run unmodified on mdCATH, the probe would have reported abundant recurrence that
is an artifact of the save interval — and on that basis we would have committed
to a 3.3 TB download. **This is the Phase 3 triviality trap arriving from the
opposite direction**, and it was caught before the download, not after.

`validation/phase8_recurrence.py` therefore reports three numbers and refuses to
collapse them:

- **A — native constructibility.** Reference = consecutive frames as delivered.
  Legitimately stride-dependent, because the detector is too.
- **B — matched constructibility.** The trajectory *re-delivered* at 1 ns, with
  candidates **and** reference both taken from the subsampled result. Comparable
  across datasets.
- **C — physical recurrence.** Full-resolution candidate pool against a 1 ns-lag
  reference. Reported for interpretation and **deliberately excluded** from the
  decision.

### 19.2 Two errors of mine, both caught by disagreement between measures

**(i) An inconsistent B.** The first version drew candidates from the *full*
1,001-frame pool but compared them to a *coarse* reference. That is optimistic
and incoherent: if the trajectory is delivered at 1 ns, only the 101 retained
frames can be spliced. The inconsistent version reported ATLAS at **0.84**
constructible; the self-consistent version reports **0.20**. Subsampling relaxes
the bar and destroys the candidate pool at the same time, and only the first
half of that was being counted.

**(ii) A gate that could not be satisfied.** The "spread over ≥8 distinct source
regions" check used a hard-coded 50-frame bucket. On a 101-frame subsample that
allows at most 2 regions, so the coarse-stride arm was failing *by construction*
regardless of the data. With the bucket scaled to trajectory length, ATLAS at
1 ns moves from 0.00 to 0.20.

Both were found because a quick stride sweep disagreed with the main probe. The
disagreement was the signal; neither error was visible from its own output.

### 19.3 Results — ATLAS, 25 proteins

| Measure | Constructible at P50 | Closest-pair percentile (median) |
|---|---|---|
| **A** native (100 ps delivery) | **0.04** | 64.3 |
| **B** matched (re-delivered at 1 ns) | **0.20** | 24.0 |
| **C** physical recurrence | — | **7.4** |

**§18 overstated its case, and C is the correction.** §18.4 said "the protein
essentially never returns to where it has been. Over 100 ns there is no
recurrence." That is too strong. The protein *does* revisit conformations: the
closest temporally distant pair sits at the **7.4th percentile** of a 1 ns
displacement. Physical recurrence is real.

The accurate statement is narrower and more interesting:

> Recurrence exists, but it cannot be exploited. To hide a splice you need a
> stride coarse enough that ordinary steps are large; at 100 ns total length,
> coarsening the stride destroys the candidate pool faster than it relaxes the
> bar. **The limit is trajectory length, not the physics.**

### 19.4 The scaling test that answers the download question

At a **fixed** 1 ns stride, varying only the number of frames — the single axis
mdCATH changes:

| Frames | Total ns | Fraction constructible at P50 | Median admissible pairs | Closest-pair pct |
|---|---|---|---|---|
| 26 | 26 | 0.00 | 0 | 52.0 |
| 51 | 51 | 0.04 | 4 | 36.0 |
| 76 | 76 | 0.20 | 6 | 26.7 |
| 101 | 101 | 0.20 | 26 | 24.0 |

Monotonic in every column. ATLAS runs out of trajectory, not out of recurrence.

**mdCATH sits at ~500 frames at the same 1 ns stride** — 5× the length, and
candidate pairs grow roughly as n², so ~25× the pool. On this trend it is likely
to clear the bar, and the 450 K replicates should add genuine recurrence on top.
"Likely" is not "measured", so the recommendation is a **staged** download:
5 domains (~1–2 GB, not 3.3 TB), run the probe, commit to bulk only on a pass.

### 19.5 Decision, pre-registered

Download mdCATH in bulk **only if it passes B at P50 on a majority of sampled
domains.** Passing A alone means the splice hides behind the save interval
rather than behind the physics — a weak benchmark, and not worth the bandwidth.

Corollary already established: **the 10,000-frame ATLAS archives (OI-15) will
not help.** They give 10× *finer* resolution over the *same* 100 ns, which moves
the wrong axis — it worsens A and leaves B unchanged. **OI-15 is withdrawn from
the plan.** That download would have been pure cost.

---

## 20. OI-34 — the register enforced in code, and the integration contract

### 20.1 What was wrong

`CLAIMS.md` (§17) declared π-derived quantities meaningless at ≤100 ns. The
pipeline kept emitting them. Anyone running it got a `component_rarity` column
the register says cannot be interpreted, with nothing in the output saying so.
A register the code does not enforce is a document, not a control — the same
criticism §17.1 made of the README, one level down.

### 20.2 Two diagnostics promoted into the pipeline

`msm/preflight.py` now runs, before any model is fitted:

- **`assess_recurrence`** (S-1b) — model-free. The closest pair of frames ≥10 ns
  apart, as a percentile of ordinary consecutive-frame steps. Reports its
  **stride** and its **atom selection**, because both change the answer
  materially (§20.4).
- **`assess_estimability`** — asked of the discrete trajectory the MSM is
  actually fitted to. Two conditions: within-trajectory singleton mass, and
  total sampling against the convergence threshold.

`msm/trust.py` turns the register into a machine-readable per-system contract
(`results/{SYSTEM}/trust.json`): each channel is `ok`, `descriptive_only` or
`withheld`, with the reason and the CLAIMS row. Withheld channels are written as
**NaN** and the fused score is recomputed from what survived. `INTEGRATION.md`
documents the contract for the visualiser.

Three properties worth stating:

- **No channel can be promoted above `descriptive_only`.** The ceiling encodes
  what validation supports in general; a good trajectory cannot earn a detection
  claim, because W-1 is not a per-trajectory question.
- **Missing diagnostics fail closed.** An absent check is never read as a pass.
- **A contract failure aborts the write.** The pipeline returns without emitting
  scores rather than emitting ungated ones.

### 20.3 The gate that would have given false assurance

The first version of `assess_estimability` keyed only on singleton mass — the
fraction of transition counts observed exactly once. It looked principled and
it was nearly useless.

On the first end-to-end run (1g2r_A, 987 frames, 20 states) it returned **4%
singleton mass, 20/20 states revisited, verdict `estimable`** — π green-lit on
exactly the ATLAS data where Phase 7 measured π *failing to generalise across
replicates* (ρ = −0.515 against a +0.620 control).

The two facts are entirely compatible, which is the lesson: with 20 coarse
states over ~1,000 frames, every state is revisited ~50 times *within* the run
while its population means nothing *across* runs. Cross-replicate generalisation
cannot be measured from a single trajectory, so no within-trajectory statistic
can stand in for it.

The gate now also requires total sampling against `PI_CONVERGENCE_NS = 4000`
(Kozlowski & Grubmüller's **lower** bound for 50–112 residue proteins). ATLAS at
100 ns is **40× short** and π is withheld — which is what §15 measured directly.
The consequence is that π is withheld on essentially all currently available
data, mdCATH's ~500 ns included. That is the finding, not a bug to soften.

### 20.4 Cα versus all atoms — the recurrence verdict is selection-dependent

The same trajectory, the same check, two answers:

| Atom selection | Closest distant pair | Ordinary step median | Percentile | Verdict |
|---|---|---|---|---|
| all atoms (§18) | 1.979 Å | 1.870 Å | **62.7** | non-recurrent |
| Cα only (preflight) | 0.921 Å | 1.673 Å | **0.6** | recurrent |

Side chains dominate all-atom RMSD and never repeat; the backbone fold does
recur. Both numbers are correct answers to different questions.

The pipeline uses **Cα**, deliberately: conformational *state* identity is a
backbone property and it is the space the MSM is meant to resolve. Phase 8 used
all atoms, which is why §18 read as strongly as it did.

**§18.4 is further narrowed by this.** "The protein never returns to where it
has been" is true of its side chains and false of its backbone. The surviving
statement — and it is the one that matters — is §19.3's: recurrence exists, but
100 ns is far too short for it to be exploitable. The estimability gate now
rests on sampling length, which is measured and selection-independent, rather
than on a recurrence percentile that moves by two orders of magnitude with an
atom mask.

**A recurrence percentile is meaningless without its stride and its atom
selection.** Both are now emitted with every value, and `INTEGRATION.md`
forbids displaying one without them.

### 20.5 Status

33 tests pass, including one that pins the exact false-assurance case in §20.3:
heavy within-run repetition plus 100 ns sampling must return `unresolved`.
End-to-end run verified — `rarity` and `transition_surprise` withheld as NaN,
`local_density` retained, contract written, headline naming *which* condition
failed. **OI-34 closed.**
