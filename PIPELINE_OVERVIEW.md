# Pipeline Overview: Dynamic Hotspot Detection in MD Simulations

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Stage 1: Trajectory Parsing](#stage-1-trajectory-parsing)
4. [Stage 2: Feature Extraction](#stage-2-feature-extraction)
5. [Stage 3: tICA Dimensionality Reduction](#stage-3-tica-dimensionality-reduction)
6. [Stage 4: Clustering and MSM Construction](#stage-4-clustering-and-msm-construction)
7. [Stage 5: Anomaly Scoring](#stage-5-anomaly-scoring)
8. [Stage 6: Per-Frame and Per-Residue Metrics Export](#stage-6-per-frame-and-per-residue-metrics-export)
9. [I/O Specification](#io-specification)
10. [How to Run the Pipeline](#how-to-run-the-pipeline)
11. [Interpreting Results](#interpreting-results)
12. [Related Methods and Scientific Validation](#related-methods-and-scientific-validation)
13. [Limitations and Edge Cases](#limitations-and-edge-cases)

---

## Executive Summary

This pipeline detects **dynamic hotspots** in molecular dynamics (MD) simulations—protein residues that exhibit unusual, rare, or functionally important motions. It combines:

- **Time-lagged Independent Component Analysis (tICA)** for dimensionality reduction
- **Markov State Models (MSM)** for kinetic modeling
- **Multi-signal anomaly detection** using state rarity, transition surprise, and local density

**Key outputs:**
- Per-frame anomaly scores identifying unusual conformations
- Per-residue scores mapping anomalies to specific residues
- Three separate metric channels: dynamic anomaly, RMSF/stability, and tICA importance

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE OVERVIEW DIAGRAM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────────┐     ┌─────────────────────┐      │
│  │  topology.pdb │────▶│                  │     │                     │      │
│  └──────────────┘     │   MDTraj/        │     │   features.npy      │      │
│                       │   MDAnalysis     │────▶│   [T × F matrix]    │      │
│  ┌──────────────┐     │   Parsing        │     │                     │      │
│  │trajectory.xtc│────▶│                  │     └─────────┬───────────┘      │
│  └──────────────┘     └──────────────────┘               │                  │
│                                                          ▼                  │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                     STAGE 2: FEATURE EXTRACTION                     │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │  • Backbone dihedrals (φ, ψ) → sin/cos encoded                     │     │
│  │  • CA-CA distances (native contacts)                               │     │
│  │  • RMSD from reference                                             │     │
│  │  • Radius of gyration                                              │     │
│  │  • [Optional] Per-residue energies, pocket volumes                 │     │
│  └────────────────────────────────┬───────────────────────────────────┘     │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │              STAGE 3: tICA DIMENSIONALITY REDUCTION                 │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │  VAMP-2 Model Selection:                                           │     │
│  │    • Grid search over lag times: [5, 10, 15, 20, 30, 50]           │     │
│  │    • Grid search over dimensions: [2, 3, 4, 5, 6, 8, 10]           │     │
│  │    • Select parameters maximizing VAMP-2 score on validation set   │     │
│  │                                                                     │     │
│  │  Output: tica_coords.npy [T × d matrix] (d ≈ 3-8 dimensions)       │     │
│  └────────────────────────────────┬───────────────────────────────────┘     │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │             STAGE 4: CLUSTERING & MSM CONSTRUCTION                  │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │  1. K-Means clustering in tICA space → k=30 states (default)       │     │
│  │  2. Discrete trajectory: dtraj.npy [T × 1] state assignments       │     │
│  │  3. Count transitions at lag τ → Transition count matrix C         │     │
│  │  4. Maximum Likelihood Estimation → Transition matrix P            │     │
│  │  5. Stationary distribution π (left eigenvector with λ=1)          │     │
│  │  6. Bootstrap MSM for uncertainty quantification (100 iterations)  │     │
│  └────────────────────────────────┬───────────────────────────────────┘     │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    STAGE 5: ANOMALY SCORING                         │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │  Signal 1: State Rarity                                            │     │
│  │    rarity(t) = 1 - π[state(t)]                                     │     │
│  │                                                                     │     │
│  │  Signal 2: Transition Surprise                                     │     │
│  │    surprise(t) = -log(P[state(t) → state(t+τ)] + ε)               │     │
│  │                                                                     │     │
│  │  Signal 3: Local Density (k-NN in tICA space)                      │     │
│  │    density(t) = mean_distance_to_k_nearest_neighbors               │     │
│  │                                                                     │     │
│  │  Fusion: score(t) = median(normalize(signal_1, signal_2, signal_3))│     │
│  │  Smoothing: moving_median(score, window=5)                         │     │
│  └────────────────────────────────┬───────────────────────────────────┘     │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │            STAGE 6: METRIC EXPORT & VISUALIZATION                   │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │  Output Files:                                                      │     │
│  │    • frame_scores_dynamic.csv (per-frame anomaly timeseries)       │     │
│  │    • residue_scores_dynamic.json (per-residue anomaly)             │     │
│  │    • residue_scores_rmsf.json (flexibility/stability)              │     │
│  │    • residue_scores_tica_importance.json (slow-mode contribution)  │     │
│  │    • hotspots_unified.json (combined for viewer)                   │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Trajectory Parsing

### Purpose
Load and preprocess molecular dynamics trajectory data from standard formats.

### Inputs
| File | Format | Description |
|------|--------|-------------|
| `topology.pdb` | PDB | Static structure defining atom/residue definitions |
| `trajectory.xtc` | XTC/DCD/TRR | Time series of atomic coordinates |

### Implementation
The pipeline uses **MDTraj** (via `mdtraj.load()`) to parse trajectories:

```python
import mdtraj as md
traj = md.load(trajectory_path, top=topology_path, stride=stride)
```

### Key Parameters
- **stride**: Load every N-th frame (default: 1). Use higher values for initial exploration.

### Output
- MDTraj trajectory object containing:
  - `traj.n_frames`: Number of frames (T)
  - `traj.n_atoms`: Number of atoms
  - `traj.topology`: Residue/chain/atom definitions
  - `traj.xyz`: Atomic coordinates [T × n_atoms × 3]

---

## Stage 2: Feature Extraction

### Purpose
Convert raw atomic coordinates into a compact, informative feature representation suitable for machine learning.

### Features Computed

#### 1. Backbone Dihedral Angles (φ, ψ)
- **What**: Rotation angles around N-Cα and Cα-C bonds
- **Encoding**: sin/cos transformation (avoids periodicity issues)
- **Shape**: [T × (2 × n_residues)]

```python
_, phi = md.compute_phi(traj)  # [T × n_residues-1]
_, psi = md.compute_psi(traj)  # [T × n_residues-1]
features['phi_sin'] = np.sin(phi).mean(axis=1)
features['phi_cos'] = np.cos(phi).mean(axis=1)
```

#### 2. CA-CA Contacts
- **What**: Number of Cα-Cα pairs within 8 Å
- **Purpose**: Captures tertiary structure compactness

```python
distances = md.compute_distances(traj, ca_pairs)
contacts = (distances < 0.8).sum(axis=1)  # 0.8 nm = 8 Å
```

#### 3. RMSD from Reference
- **What**: Root mean square deviation from reference frame
- **Reference**: First frame (default) or specified structure

```python
rmsd = md.rmsd(traj, ref_frame)  # in nm
```

#### 4. Radius of Gyration
- **What**: Measure of molecular compactness
- **Formula**: Rg = √(Σ mᵢ|rᵢ - r_cm|² / Σ mᵢ)

```python
rg = md.compute_rg(traj)  # in nm
```

#### 5. Optional: Per-Residue Energies (Phase 2)
- **What**: Knowledge-based contact potentials (Miyazawa-Jernigan style)
- **Output**: `residue_energy.parquet`

#### 6. Optional: Pocket Dynamics (Phase 2)
- **What**: Binding cavity volume, mouth radius, SASA
- **Output**: `pockets.parquet`

### Output
- `features.npy`: Feature matrix [T × F] where F is number of features

---

## Stage 3: tICA Dimensionality Reduction

### Purpose
Identify the **slow collective motions** that are most relevant for protein function, reducing high-dimensional feature space to a few interpretable coordinates.

### Mathematical Formulation

Given feature vectors **x**_t at time t:

**1. Covariance Matrices**
```
C₀ = E[x_t ⊗ x_t]         (instantaneous covariance)
C_τ = E[x_t ⊗ x_{t+τ}]    (time-lagged covariance)
```

**2. Generalized Eigenvalue Problem**
```
C_τ v = λ C₀ v
```

**3. tICA Eigenvalues and Timescales**
```
τᵢ = -τ_lag / ln(λᵢ)      (implied timescale)
```

### VAMP-2 Model Selection (Phase 1)

The optimal lag time and dimensionality are selected by maximizing the **VAMP-2 score**:

```
VAMP-2 = Σᵢ σᵢ²
```

where σᵢ are singular values of the Koopman operator approximation.

**Grid Search Parameters:**
- Lag times: [5, 10, 15, 20, 30, 50] frames
- Dimensions: [2, 3, 4, 5, 6, 8, 10]
- Validation: 20% held-out data

**Implementation** (`msm/select_lag_and_dim.py`):
```python
from deeptime.decomposition import VAMP

vamp = VAMP(lagtime=lag, dim=dim).fit(X_train)
score = compute_vamp2_score(X_validation)  # Higher = better
```

### Output
- `tica_coords.npy`: Low-dimensional coordinates [T × d]
- `vamp2_best.json`: Selected parameters (lag, dim, score)
- `tica_model.npz`: Saved eigenvectors for residue importance

---

## Stage 4: Clustering and MSM Construction

### Purpose
Discretize continuous conformational space into metastable states and model kinetics as a Markov chain.

### Step 4.1: Clustering

**Algorithm**: K-Means in tICA-projected space

```python
from deeptime.clustering import KMeans

kmeans = KMeans(n_clusters=30, max_iter=100).fit(tica_coords)
dtraj = kmeans.transform(tica_coords)  # [T] integer labels
```

**Parameters:**
- `n_clusters`: 30 (default), typically 20-50
- Rule of thumb: 100-200 frames per state on average

### Step 4.2: Transition Counting

Count transitions at lag time τ:

```
C_ij = #{t : state(t) = i and state(t+τ) = j}
```

### Step 4.3: Transition Matrix Estimation

**Maximum Likelihood Estimation (MLE):**
```
P_ij = C_ij / Σⱼ C_ij
```

For reversible MSMs (detailed balance):
```
π_i P_ij = π_j P_ji
```

**Implementation** (`msm/bootstrap_msm.py`):
```python
from deeptime.markov.msm import MaximumLikelihoodMSM

msm = MaximumLikelihoodMSM(lagtime=30, reversible=True).fit(dtraj)
P = msm.transition_matrix       # [k × k]
pi = msm.stationary_distribution  # [k]
```

### Step 4.4: Stationary Distribution

The stationary distribution π is the left eigenvector of P with eigenvalue 1:

```
π P = π,   Σᵢ πᵢ = 1
```

**Physical Interpretation:**
- π[i] = long-term probability of being in state i
- Low π[i] → rare state (potentially important)

### Step 4.5: Bootstrap Uncertainty Quantification

To estimate confidence intervals on MSM parameters:

1. Resample trajectory (frame or block bootstrap)
2. Rebuild MSM on resampled data
3. Repeat 100 times
4. Compute percentile-based CIs

**Output:**
- `pi_ci.parquet`: Stationary distribution with 95% CIs
- `P_ci.npz`: Transition matrix with CIs
- `mfpt_ci.parquet`: Mean first passage times with CIs

---

## Stage 5: Anomaly Scoring

### Purpose
Identify conformations and transitions that deviate from typical behavior using multiple complementary signals.

### Signal 1: State Rarity

**Formula:**
```
rarity(t) = 1 - π[state(t)]
```

**Interpretation:**
- High values → visiting a rarely-populated state
- May indicate: transition intermediates, high-energy barriers, functional conformations

### Signal 2: Transition Surprise

**Formula:**
```
surprise(t) = -log(P[state(t) → state(t+τ)] + ε)
```
where ε = 10⁻¹² for numerical stability

**Interpretation:**
- High values → unexpected/forbidden transition
- May indicate: barrier crossing, large conformational change, artifacts

### Signal 3: Local Density

**Formula:**
```
density(t) = mean_distance_to_k_nearest_neighbors(tica_coords[t])
```

**Implementation:**
```python
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=20).fit(tica_coords)
distances, _ = nbrs.kneighbors(tica_coords)
local_density = distances.mean(axis=1)  # Higher = more isolated
```

**Interpretation:**
- High values → structurally isolated conformation
- May indicate: rare conformational region, outlier, artifact

### Signal Normalization

Each signal is normalized to [0, 1] using **rank-based normalization**:

```python
def rank_normalize(x):
    ranks = np.argsort(np.argsort(x))
    return ranks / (len(x) - 1)
```

**Why rank normalization?**
- Robust to outliers
- Distribution-free
- Preserves exact ordering

### Signal Fusion

**Default: Median fusion**
```python
fused_score = np.median([signal_1, signal_2, signal_3], axis=0)
```

**Why median?**
- Robust: requires multiple signals to agree
- Reduces false positives from noisy single signals

### Temporal Smoothing

**Moving median filter:**
```python
from scipy.ndimage import median_filter
smoothed = median_filter(fused_score, size=5, mode='nearest')
```

**Purpose:**
- Reduce frame-to-frame jitter
- Identify sustained (not transient) anomalies

### Implementation

Located in `scoring/signals.py` and `scoring/anomaly_v2.py`:

```python
from scoring.signals import compute_dynamic_anomaly_scores

signals = compute_dynamic_anomaly_scores(
    msm=msm,
    dtraj=dtraj,
    tica_coords=tica_coords,
    lag_msm=30,
    k_neighbors=20,
    normalize=True
)
# Returns: {'rarity': [...], 'transition_surprise': [...], 'local_density': [...]}
```

---

## Stage 6: Per-Frame and Per-Residue Metrics Export

### Purpose
Export metrics in formats suitable for visualization and analysis.

### Metric Channel 1: Dynamic Anomaly

**Per-frame:** Combined anomaly score from signal fusion
**Per-residue:** Aggregated by weighting frames where each residue contributes

### Metric Channel 2: RMSF/Stability

**Formula:**
```
RMSF_i = √(⟨(r_i - ⟨r_i⟩)²⟩)
```

**Implementation** (`scoring/signals.py`):
```python
from scoring.signals import compute_rmsf_scores

rmsf = compute_rmsf_scores(
    topology_path='topology.pdb',
    trajectory_path='trajectory.xtc',
    selection='name CA'
)
```

**Interpretation:**
- High RMSF → flexible/floppy region
- Low RMSF → rigid/stable region
- Typical values: <1 Å (rigid), 1-3 Å (moderate), >3 Å (flexible)

### Metric Channel 3: tICA Importance

**Formula:**
```
importance_residue = ||loadings_residue||₂
```

where loadings are the tICA eigenvector coefficients for features involving that residue.

**Interpretation:**
- High importance → residue drives slow collective motions
- Often identifies: hinge residues, allosteric nodes, domain linkers

### Output Files

| File | Format | Content |
|------|--------|---------|
| `frame_scores_dynamic.csv` | CSV | Per-frame scores with components |
| `residue_scores_dynamic.json` | JSON | Per-residue dynamic anomaly |
| `residue_scores_rmsf.json` | JSON | Per-residue RMSF normalized |
| `residue_scores_tica_importance.json` | JSON | Per-residue tICA importance |
| `hotspots_unified.json` | JSON | Combined output for viewer |
| `hotspots_residue.json` | JSON | Legacy format |

### Unified JSON Schema

```json
{
  "meta": {
    "n_frames": 200,
    "n_residues": 150,
    "metrics": ["dynamic_anomaly", "rmsf", "tica_importance"],
    "normalization": "percentile",
    "percentile_range": [0.05, 0.95]
  },
  "per_residue": {
    "dynamic_anomaly": {"0": 0.45, "1": 0.23, ...},
    "rmsf": {"0": 0.12, "1": 0.67, ...},
    "tica_importance": {"0": 0.89, "1": 0.15, ...}
  }
}
```

---

## I/O Specification

### Script Dependency Order

```
1. features/compute_md_features.py (or tools/extract_features.py)
   └── 2. tools/run_phase1.py
       ├── 2a. msm/select_lag_and_dim.py (VAMP-2 selection)
       └── 2b. msm/bootstrap_msm.py (Bootstrap MSM)
           └── 3. tools/run_msm_tica.py (Build final MSM)
               └── 4. tools/compute_all_metrics.py (Unified metrics)
                   └── 5. viewer/app.py (Visualization)
```

### Detailed I/O per Script

#### 1. Feature Extraction
**Script:** `features/compute_md_features.py`

| Input | Output |
|-------|--------|
| `topology.pdb` | `features.npy` [T × F] |
| `trajectory.xtc` | Feature names in header |

#### 2. VAMP-2 Model Selection
**Script:** `msm/select_lag_and_dim.py`

| Input | Output |
|-------|--------|
| `features.npy` | `reports/vamp2_grid.csv` |
| | `reports/vamp2_best.json` |

#### 3. Bootstrap MSM
**Script:** `msm/bootstrap_msm.py`

| Input | Output |
|-------|--------|
| `features.npy` | `models/msm_bootstrap/pi_ci.parquet` |
| `reports/vamp2_best.json` | `models/msm_bootstrap/P_ci.npz` |
| | `models/msm_bootstrap/mfpt_ci.parquet` |
| | `models/msm_bootstrap/bootstrap_metadata.json` |

#### 4. MSM Pipeline
**Script:** `tools/run_msm_tica.py`

| Input | Output |
|-------|--------|
| `features.npy` | `outputs/msm/dtraj.npy` |
| | `outputs/msm/tica_coords.npy` |
| | `outputs/msm/P.npy` |
| | `outputs/msm/pi.npy` |
| | `outputs/msm/tica_model.npz` |

#### 5. Unified Metrics Computation
**Script:** `tools/compute_all_metrics.py`

| Input | Output |
|-------|--------|
| `topology.pdb` | `outputs/metrics/frame_scores_dynamic.csv` |
| `trajectory.xtc` | `outputs/metrics/residue_scores_dynamic.json` |
| `outputs/msm/*` | `outputs/metrics/residue_scores_rmsf.json` |
| | `outputs/metrics/residue_scores_tica_importance.json` |
| | `outputs/metrics/hotspots_unified.json` |

---

## How to Run the Pipeline

### Prerequisites

```bash
# Install dependencies
pip install numpy scipy pandas scikit-learn
pip install deeptime mdtraj hmmlearn pyyaml

# Or use requirements files
pip install -r requirements_phase1.txt
pip install -r requirements_phase2.txt
pip install -r requirements_phase3.txt
```

### Full Pipeline Example

```bash
# Step 1: Prepare your data
# Place topology.pdb and trajectory.xtc in data/raw_trajectory/

# Step 2: Extract features (creates data/features.npy)
python features/compute_md_features.py \
    --topology data/raw_trajectory/topology.pdb \
    --trajectory data/raw_trajectory/trajectory.xtc \
    --output data/features.npy

# Step 3: Run Phase 1 - Model selection + Bootstrap
python tools/run_phase1.py \
    --features data/features.npy \
    --output outputs/phase1 \
    --config configs/pipeline.yaml

# This creates:
#   outputs/phase1/reports/vamp2_best.json
#   outputs/phase1/models/msm_bootstrap/*

# Step 4: Build MSM with selected parameters
# Read best parameters from vamp2_best.json
python tools/run_msm_tica.py \
    data/features.npy \
    outputs/msm \
    --lag_tica 10 \
    --lag_msm 30 \
    --n_clusters 30

# Step 5: Compute all metrics
python tools/compute_all_metrics.py \
    --topology data/raw_trajectory/topology.pdb \
    --trajectory data/raw_trajectory/trajectory.xtc \
    --msm_dir outputs/msm \
    --output_dir outputs/metrics \
    --normalization percentile \
    --low-percentile 0.05 \
    --high-percentile 0.95

# Step 6: Visualize (optional)
python viewer/app.py \
    --topology data/raw_trajectory/topology.pdb \
    --trajectory data/raw_trajectory/trajectory.xtc \
    --hotspots outputs/metrics/hotspots_unified.json
```

### Quick Start (Minimal Pipeline)

```bash
# For quick testing with existing features and MSM:
python tools/compute_all_metrics.py \
    --topology topology.pdb \
    --trajectory trajectory.xtc \
    --msm_dir outputs/msm \
    --output_dir outputs/metrics
```

### Robust Mode (for challenging trajectories)

```bash
# Use robust mode for short/noisy trajectories
python tools/compute_all_metrics.py \
    --topology topology.pdb \
    --trajectory trajectory.xtc \
    --msm_dir outputs/msm \
    --output_dir outputs/metrics \
    --robust
```

See [Limitations and Edge Cases](#limitations-and-edge-cases) for details.

---

## Interpreting Results

### Dynamic Anomaly Scores

| Score Range | Interpretation | Typical Examples |
|-------------|----------------|------------------|
| 0.0 - 0.25 | Normal, well-sampled | Core residues, stable secondary structure |
| 0.25 - 0.50 | Mild anomaly | Surface loops, moderate flexibility |
| 0.50 - 0.75 | Significant anomaly | Active site during catalysis, allosteric transitions |
| 0.75 - 1.0 | Strong anomaly | Rare conformations, barrier crossings |

### RMSF/Stability Scores

| Score | Raw RMSF | Interpretation |
|-------|----------|----------------|
| Low (<0.3) | <1 Å | Rigid core, secondary structure |
| Medium (0.3-0.7) | 1-3 Å | Moderately flexible, functional loops |
| High (>0.7) | >3 Å | Very flexible, termini, disordered regions |

### tICA Importance Scores

| Score | Interpretation | Typical Examples |
|-------|----------------|------------------|
| High (>0.7) | Drives slow collective motions | Hinge residues, allosteric nodes |
| Medium (0.3-0.7) | Moderately involved | Interface residues |
| Low (<0.3) | Fast local motions only | Surface, termini |

### Score Combination Matrix

| Dynamic Anomaly | RMSF | tICA Importance | Interpretation |
|----------------|------|-----------------|----------------|
| High | High | High | **Prime hotspot**: Flexible, drives slow modes, rare events |
| High | High | Low | Flexible region with local anomalies |
| High | Low | High | **Rigid hotspot**: Stable but critical for dynamics |
| High | Low | Low | Local rearrangement, possible artifact |
| Low | High | High | Constitutive driver of slow motions |
| Low | High | Low | Surface loop, not functionally important |
| Low | Low | High | Stable enabler of slow modes (hinge) |
| Low | Low | Low | Core residue, structurally passive |

---

## Related Methods and Scientific Validation

### Core Methods and References

#### Time-lagged Independent Component Analysis (tICA)

**What it captures:** Slow collective motions by maximizing autocorrelation at lag τ.

**Key references:**
1. Pérez-Hernández, G. et al. (2013). "Identification of slow molecular order parameters for Markov model construction." *J. Chem. Phys.* 139: 015102.
   - Original tICA application to molecular dynamics
   
2. Schwantes, C.R. & Pande, V.S. (2013). "Improvements in Markov State Model Construction." *J. Chem. Theory Comput.* 9(4): 2000-2009.
   - Comparison of tICA vs. PCA for MD analysis

3. Wu, H. & Noé, F. (2020). "Variational Approach for Learning Markov Processes from Time Series Data." *J. Nonlinear Sci.* 30: 23-66.
   - VAMP theory and score for model selection

**Why appropriate for hotspot detection:** Slow motions are often functionally relevant (domain movements, allosteric transitions, catalytic cycles). By focusing on slow modes, we filter out fast thermal noise.

#### Markov State Models (MSM)

**What they capture:** Kinetic information including state populations (π) and transition rates (P).

**Key references:**
1. Prinz, J.-H. et al. (2011). "Markov models of molecular kinetics: Generation and validation." *J. Chem. Phys.* 134: 174105.
   - Comprehensive MSM methodology and validation

2. Noé, F. et al. (2009). "Constructing the equilibrium ensemble of folding pathways from short off-equilibrium simulations." *Proc. Natl. Acad. Sci.* 106(45): 19011-19016.
   - MSM for rare event analysis

3. Trendelkamp-Schroer, B. et al. (2015). "Estimation and uncertainty of reversible Markov models." *J. Chem. Phys.* 143: 174101.
   - Bootstrap uncertainty quantification for MSMs

**Why appropriate for hotspot detection:** State rarity (π) directly identifies thermodynamically rare conformations. Transition surprise (P) identifies kinetically rare events.

#### Anomaly Detection via Outlier Analysis

**What we compute:** Multi-signal ensemble of kinetic, structural, and energetic outlier signals.

**Key references:**
1. Chandola, V. et al. (2009). "Anomaly detection: A survey." *ACM Comput. Surv.* 41(3): 1-58.
   - Comprehensive outlier detection theory

2. Aggarwal, C.C. (2017). *Outlier Analysis.* 2nd edition. Springer.
   - k-NN based density estimation for outliers

**Why appropriate for hotspot detection:** No single signal captures all types of anomalies. Kinetic signals miss energetically strained but accessible states. Structural signals miss kinetically forbidden transitions. Fusion provides comprehensive detection.

#### RMSF and B-factors

**What RMSF captures:** Average positional fluctuation of each residue around its mean position.

**Relationship to experiments:**
- RMSF ∝ √(B-factor / (8π²/3)) from X-ray crystallography
- Correlates with NMR order parameters S²

**Key references:**
1. Kuzmanic, A. & Zagrovic, B. (2010). "Determination of ensemble-average pairwise root mean-square deviation from experimental B-factors." *Biophys. J.* 98: 861-871.

#### Knowledge-Based Potentials (Phase 2)

**What they capture:** Interaction preferences from statistical analysis of known structures.

**Key references:**
1. Miyazawa, S. & Jernigan, R.L. (1996). "Residue-residue potentials..." *J. Mol. Biol.* 256: 623-644.
   - Original statistical potential derivation

#### Pocket Detection (Phase 2)

**What it captures:** Binding cavity dynamics including cryptic pocket opening.

**Key references:**
1. Schmidtke, P. et al. (2011). "MDpocket: open-source cavity detection and characterization on molecular dynamics trajectories." *Bioinformatics* 27: 3276-3285.

2. Beglov, D. et al. (2018). "Exploring the structural origins of cryptic sites on proteins." *Proc. Natl. Acad. Sci.* 115: E3416-E3425.
   - Cryptic pocket discovery

### Biological Validation

**Dynamic hotspots often correspond to:**
1. **Active sites** - Residues involved in catalysis
2. **Allosteric sites** - Regions that communicate with distant sites
3. **Druggable cryptic pockets** - Transient binding sites
4. **Mutation-sensitive residues** - Sites where mutations affect function

**Recommended validation:**
- Compare to known functional sites (UniProt, literature)
- Conservation analysis (MSA)
- Experimental dynamics (NMR, HDX-MS)
- Mutagenesis data

### Software Tools Used

1. **deeptime** (Hoffmann et al., 2021): Modern Python library for tICA/MSM
   - https://deeptime-ml.github.io/

2. **MDTraj** (McGibbon et al., 2015): MD trajectory analysis
   - https://mdtraj.org/

3. **scikit-learn**: k-NN, clustering, general ML
   - https://scikit-learn.org/

---

## Limitations and Edge Cases

### Minimum Requirements

| Parameter | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Trajectory length | 100 frames | 1000+ frames | Fewer frames → poor MSM statistics |
| Frames per state | 10 | 100-200 | Below this, state populations unreliable |
| tICA lag | 5 frames | 10-50 frames | Must be < trajectory length / 10 |
| MSM lag | 10 frames | 20-50 frames | Must satisfy Markov property |

### Known Failure Modes

#### 1. Very Short Trajectories (< 100 frames)
**Symptoms:**
- VAMP-2 scores return -inf
- MSM transition matrix has zero rows
- Anomaly scores all similar

**Mitigation:**
- Use `--robust` mode with fewer clusters
- Increase MSM lag time
- Skip MSM-based signals, use only density

#### 2. Low Variance Features
**Symptoms:**
- tICA fails with singular covariance matrix
- All frames assigned to same state
- No contrast in anomaly scores

**Mitigation:**
- Check for frozen/constrained atoms
- Verify trajectory is aligned
- Add regularization to covariance estimation

#### 3. Disconnected MSM States
**Symptoms:**
- Some states never transition to others
- Stationary distribution has zeros
- Division by zero in transition surprise

**Mitigation:**
- Use largest connected component
- Add small pseudo-counts to transition matrix
- Reduce number of clusters

#### 4. Extreme Outliers
**Symptoms:**
- 99% of scores clustered, 1% extreme
- Poor visualization (all blue with red tips)

**Mitigation:**
- Use percentile normalization with clipping
- Apply robust scaling
- Inspect outlier frames for artifacts

### Robust Mode

When using `--robust` flag, the pipeline automatically:

1. **Uses conservative parameters:**
   - Fewer tICA dimensions (max 5)
   - Fewer MSM clusters (max 20)
   - Longer lag times

2. **Adds numerical safeguards:**
   - Regularization of covariance matrices
   - Pseudo-counts for transition matrix
   - Clipping of extreme probabilities

3. **Graceful degradation:**
   - If MSM fails, uses only density-based signals
   - If tICA fails, uses PCA instead
   - Warnings logged but pipeline continues

### Edge Case Tests

Run the edge case test suite:
```bash
python -m pytest tests/test_pipeline_edge_cases.py -v
```

This tests:
- Short trajectories (10, 50, 100 frames)
- Constant features (zero variance)
- Single-state trajectories
- Highly fragmented MSM

---

## Appendix: Configuration File Reference

`configs/pipeline.yaml`:

```yaml
# Random seeds for reproducibility
seeds:
  global: 42
  kmeans: 42
  bootstrap: 123

# TICA parameters
tica:
  lag_candidates: [5, 10, 15, 20, 30, 50]
  dim_candidates: [2, 3, 4, 5, 6, 8, 10]
  default_lag: 10
  default_dim: 5

# MSM parameters
msm:
  lag: 30
  n_clusters: 30
  reversible: true

# Bootstrap parameters
bootstrap:
  n_iterations: 100
  method: 'frames'
  block_size: 10
  confidence_level: 0.95

# Anomaly scoring
scoring:
  normalize_method: 'rank'
  fusion_method: 'median'
  window_size: 5
```

---

*Document generated from codebase analysis. For questions or issues, see the GitHub repository.*
