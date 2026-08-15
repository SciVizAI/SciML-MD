# SciML-MD Validation Campaign — Full Log

**Date:** 2026-08-15 · **Operator:** Claude (Cowork session) with Siya
**Repo:** `~/SciML-MD`, branch `validation-fixes` (created off `topology-contract-fix` @ 431132d)
**Companion documents:** `VALIDATION_REPORT.md` (findings + before/after metrics), result JSONs in this folder.

This log records *everything that was run*, in order, with the exact commands,
environments, and where each output landed — so every number in the report can
be regenerated.

---

## 0. Environment

Cloud workspace (validation executed here, results copied back to the repo):

| Component | Version |
|---|---|
| Python | 3.11.15 |
| deeptime | 0.4.5 |
| mdtraj | 1.11.1.post2 |
| scikit-learn | 1.8.0 |
| OpenMM | 8.5.2 (pip wheel) |
| PDBFixer | 1.12.0 (installed from source; pure-python copy) |
| pytest | latest at run date |

Laptop reference environment: conda env `mdcapstone` (Python 3.10/3.13
bytecode both present in repo `__pycache__`). Laptop artifacts used as
cross-machine reference: `artifacts/{1VII,8H0R,1CRN}`, `results/…`, `exports/…`.

---

## 1. Phase 0 — Orientation (read-only)

- Mapped repo layout, entry points (`batch_runner.py`, `run_all_proteins.py`),
  scoring path (`scoring/anomaly_v2.py`), exports (`tools/export_for_asvs.py`,
  `tools/export_unified.py`), tests, git state.
- Inspected the laptop run artifacts directly:
  - all three proteins: `dtraj.npy` uses labels 0–9 (10 KMeans clusters) while
    `pi.npy`/`P.npy` have **7 states** (1VII fresh rerun: 8) → state-space
    mismatch present in every stored run.
  - `exports/8H0R/anomaly_residue.json`: **every value = 0.25** (flat export).
  - `exports/*/rmsf_residue.json`, `tica_importance.json`: empty (`normalized: {}`).
- Decisions taken with Siya: baseline → fix → re-validate; rare-case ground
  truth operational on real proteins (no synthetic accuracy claims); fixes on
  new branch `validation-fixes`.

## 2. Phase 1 — Baseline (code as-is)

### 2.1 Reproduction run
```bash
# data/ = sample_data copies (1VII topology+traj, 8H0R canonical_topology+traj)
python3 run_all_proteins.py --data_dir data \
    --artifacts_dir baseline_artifacts --results_dir baseline_results \
    --lag_tica 5 --dim_tica 3 --n_clusters 10 --lag_msm 5 --k_neighbors 5 --window 3
```
Output: 2/2 succeeded; reproduced the laptop "Skipping state set [k]" warnings
(1VII: [2] → 8 active states; 8H0R: [2],[6] → 7 active states).
Mean scores 48.2 / 48.3 (cf. laptop 47.6 / 48.3).

### 2.2 Numerical validation
```bash
python3 validation/baseline_numerical.py     # → numerical_results_BASELINE.json
```
Key results — see `VALIDATION_REPORT.md` §2. Headlines: buggy-vs-correct
rarity Spearman −0.50…0.68; top-10 % overlap 0 % (1VII/8H0R rarity);
1CRN: 375/1001 frames read wrong π row, 53 in dropped states;
tie groups of 21–410 frames spread 20–42 score points;
residue-score vs RMSF Pearson r = 1.000 (1VII), 0.908 (8H0R, depressed only
by chain-key collisions: 182 residues → 91 entries);
tiny-MSM hand checks: formulas exact.

### 2.3 Edge cases (production path)
```bash
python3 validation/baseline_edge_cases.py    # → edge_results_BASELINE.json
```
21 cases; full behavior table in report §4. Notables: state-space mismatch in
*every* regime incl. 9-frame inputs; two-block disconnected trajectory loses
half its frames to silent defaults; constant features → deeptime ZeroRankError
(caught by pipeline, protein marked failed); `[1,1,2,2]` normalized to
`[0, 33, 67, 100]`.

### 2.4 Rare-case accuracy
```bash
python3 validation/baseline_rare_case.py     # → rare_case_results_BASELINE.json
```
Labels: bottom-quartile-π states ∪ disconnected states; transitions with
count ≤ 1 at lag. AUROC (fused): 0.83 / 0.86 / 0.84 (states),
0.70 / 0.79 / 0.65 (transitions). Rarity channel alone: 0.80 / 0.45 / 0.54.
Channel anti-correlation rarity↔surprise: −0.19 / −0.22 / −0.58.

### 2.5 Cross-machine reproducibility
Same inputs, same seed vs laptop artifacts: tICA per-component r = 0.9996 /
0.9979 / 0.9621 (1VII); KMeans ARI **0.425** (1VII), **0.510** (8H0R) —
clustering, not tICA, is the non-deterministic step.

## 3. Phase 2 — Fixes (branch `validation-fixes`)

| File | Change |
|---|---|
| `scoring/anomaly_v2.py` | `remap_dtraj_to_active_set()` (new); `compute_kinetic_signals` remaps labels, treats disconnected frames as maximally rare, caps disconnected-transition surprise at max finite surprise, NaN tail for last `lag` frames, internal lag clamp; `rank_normalize` tie-aware (scipy `rankdata`, average) and NaN-aware; `fuse_signals` NaN-aware (`nanmedian`/`nanmean`); negative labels guarded against wrap-around |
| `run_all_proteins.py` | mismatch WARNING with dropped-frame count; persists `state_symbols.npy`; `compute_residue_scores` rewritten (displacement-weighted anomaly participation + RMSF, chain-collision-safe keys); new `compute_rmsf_residue_json` → `residue_scores_rmsf.json` |
| `batch_runner.py` | writes `residue_scores_rmsf.json` via the new helper |
| `tools/export_for_asvs.py` | reads `score_dynamic` (÷100), positional residue-name mapping — kills the flat-0.25 export |

## 4. Phase 3 — Re-validation

```bash
python3 run_all_proteins.py --data_dir data \
    --artifacts_dir fixed_artifacts --results_dir fixed_results \
    --lag_tica 5 --dim_tica 3 --n_clusters 10 --lag_msm 5 --k_neighbors 5 --window 3
python3 validation/baseline_edge_cases.py    # → edge_results_POSTFIX.json
python3 validation/baseline_rare_case.py     # → rare_case_results_POSTFIX.json
python3 -m pytest tests/test_production_scoring.py -v   # 12/12 pass
python3 -m pytest tests/test_signals.py tests/test_pipeline_edge_cases.py \
    tests/test_integration.py tests/test_reproducibility.py  # 40 pass, 1 pre-existing fail*
```
\* `test_integration.py::test_tica_model_persistence` imports `run_msm_tica`
(never migrated from twin repo) — pre-existing, unrelated.

Post-fix AUROC (fused): rare states 0.947 / 0.958 / 0.955; rare transitions
0.811 / 0.837 / 0.999. Full before/after table in report §3.
Pipeline now logs e.g.
`[8H0R] MSM active set covers 7 of 10 cluster states; 11 frame(s) lie in disconnected states…`

## 5. Phase 4 — New targets: 9UNN, 9O6O

Goal: run the two remaining `batch_runner` default targets end-to-end with the
fixed code and apply the same validation battery.

- RCSB direct download is blocked from the cloud sandbox (proxy 403);
  structures fetched on the laptop via curl into `data/9UNN|9O6O/topology.pdb`
  and staged across.
- Commands and results: § below (appended as runs complete).

Target identities (RCSB, fetched 2026-08-15):
- **8H0R** — mouse βB1-crystallin Y202X cataract mutant, X-ray **1.20 Å**, 2 chains
- **9O6O** — human Siglec-10 + 2,6-sialyllactose, X-ray 2.70 Å, 3 chains
- **9UNN** — mouse NMDA receptor GluN1/N2A-S3 closed state, cryo-EM 3.29 Å, heterotetramer, ~23k atoms
- **1UBQ** — ubiquitin, X-ray 1.80 Å (added as extra X-ray benchmark; ran through fixed pipeline in cloud: 9/10 active states, 1 disconnected frame, mean 50.8)

MD status: cloud sandbox too slow for 9O6O/9UNN MD (2 cores, NoCutoff implicit
solvent O(N²)); trajectories being generated on the laptop (`batch_runner.py
--pdb_ids 9O6O --skip_download`; 9UNN with `--md_steps_per_frame 50`
recommended). Pipeline + validation for both will be appended here on arrival.

<!-- RESULTS_9UNN_9O6O -->

## 5b. Phase 5 — Layer 3: external validation (first results)

```bash
python3 validation/layer3_external.py     # → layer3_results.json
```
- **B-factors vs pipeline outputs** (raw RCSB B columns, CA atoms, Spearman):
  8H0R ρ = 0.595 (dynamic score) / 0.642 (RMSF), n=178 matched;
  1UBQ ρ = 0.573 / 0.656, n=76. External, non-circular evidence.
- **Baselines in same tICA space** (IsolationForest, LOF, kNN-distance
  ablation): pipeline fused wins rare-state AUROC on all four proteins
  (0.90–0.96 vs 0.49–0.93); on rare transitions geometric detectors drop to
  chance on 1UBQ (0.43–0.49) while the surprise channel scores 0.91 —
  kinetic anomalies are invisible to geometry-only methods.
- Full tables in `VALIDATION_REPORT.md` §5b; raw JSON in `layer3_results.json`.

Layer-2 status (for the record): CK / implied-timescales / VAMP-2 / bootstrap
code exists in `msm/` but has not yet been executed against our systems — this
is the next work package and is required for publication.

## 6. Deliverables inventory

| Path (repo) | What |
|---|---|
| `validation/VALIDATION_REPORT.md` | Findings, before/after metrics, limitations |
| `validation/CAMPAIGN_LOG.md` | This log |
| `validation/baseline_numerical.py` | Numerical validation script (re-runnable) |
| `validation/baseline_edge_cases.py` | Edge-case suite (re-runnable) |
| `validation/baseline_rare_case.py` | Rare-case AUROC/AUPRC evaluation (re-runnable) |
| `validation/numerical_results_BASELINE.json` | §2.2 raw numbers |
| `validation/edge_results_BASELINE.json` / `_POSTFIX.json` | §2.3 / §4 raw |
| `validation/rare_case_results_BASELINE.json` / `_POSTFIX.json` | §2.4 / §4 raw |
| `tests/test_production_scoring.py` | 12 permanent correctness tests (production path) |
| `artifacts/{ID}/state_symbols.npy` | Active-set mapping (new, when states pruned) |
| `results/{ID}/residue_scores_rmsf.json` | Per-residue RMSF in Å (new) |

## 7. Open items

- Commit the branch (user reviews diff first) and optionally push.
- `requirements.txt` cleanup; docs reference ~15 twin-repo-only files.
- `abc.py` shadows stdlib `abc` — delete/rename recommended.
- KMeans determinism across machines (report §4) if cross-machine artifact
  identity is ever required.
