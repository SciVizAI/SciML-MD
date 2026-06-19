

# Ensemble-Anomaly-Maps

**Dynamic Hotspot Detection in Molecular Dynamics Trajectories Using Machine Learning**

Ensemble-Anomaly-Maps is a computational biology pipeline that downloads protein structures from the RCSB PDB, generates short MD trajectories, and applies an unsupervised ML pipeline (tICA → KMeans → Markov State Model → anomaly scoring) to detect dynamic hotspot residues across an entire dataset of proteins — automatically, in a single command.

---

## Table of Contents

1. [What This Pipeline Does](#1-what-this-pipeline-does)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Quick Start — Full Pipeline in 3 Commands](#4-quick-start--full-pipeline-in-3-commands)
5. [Step-by-Step Guide](#5-step-by-step-guide)
   - [Step 1 — Download Protein Structures](#step-1--download-protein-structures)
   - [Step 2 — Generate MD Trajectories](#step-2--generate-md-trajectories)
   - [Step 3 — Run the ML Pipeline](#step-3--run-the-ml-pipeline)
6. [Using Your Own Data](#6-using-your-own-data)
7. [Output Files Explained](#7-output-files-explained)
8. [All Command-Line Options](#8-all-command-line-options)
9. [Repository Structure](#9-repository-structure)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What This Pipeline Does

```
RCSB PDB API
    │
    ▼
scripts/download_pdb_dataset.py
    │  Downloads ≥50 small proteins  (<150 residues, <3 Å resolution)
    │  Saves: data/{PDB_ID}/topology.pdb
    ▼
scripts/generate_md_trajectories.py
    │  For each protein: add H₂O → minimize → NVT MD (50k steps)
    │  Saves: data/{PDB_ID}/traj.xtc
    ▼
run_all_proteins.py
    │
    ├─ Feature extraction  (RMSD, Rg, contacts, φ/ψ dihedrals)
    ├─ tICA                (slow-motion dimensionality reduction)
    ├─ KMeans clustering   (conformational states)
    ├─ Markov State Model  (kinetic model)
    └─ Anomaly scoring     (rarity + transition surprise + local density)
         │
         ├─ artifacts/{PDB_ID}/tica_coords.npy
         ├─ artifacts/{PDB_ID}/dtraj.npy
         ├─ artifacts/{PDB_ID}/P.npy
         ├─ artifacts/{PDB_ID}/pi.npy
         ├─ results/{PDB_ID}/frame_scores_dynamic.csv
         └─ results/{PDB_ID}/residue_scores_dynamic.json
```

---

## 2. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.9 | 3.10 or 3.11 recommended |
| conda | any | Strongly recommended for OpenMM |
| Git | any | To clone the repo |
| Internet access | — | For RCSB downloads |
| RAM | ≥ 8 GB | 16 GB preferred for larger proteins |
| Disk | ≥ 5 GB free | For protein structures + trajectories |

> **Why conda?** OpenMM is most reliably installed via `conda-forge`. It can be installed with pip but may require a working C compiler.

---

## 3. Installation

### 3.1 Clone the repository

```bash
git clone https://github.com/siya7205/ensemble-anomaly-maps.git
cd ensemble-anomaly-maps
```

### 3.2 Create a Python environment

**Option A — conda (recommended)**

```bash
conda create -n anomaly-maps python=3.11 -y
conda activate anomaly-maps
```

**Option B — venv (if you prefer pip-only)**

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### 3.3 Install ML and MD dependencies

```bash
# Core ML libraries (tICA, MSM, clustering, anomaly scoring)
pip install -r requirements_phase1.txt

# MD trajectory analysis (mdtraj)
pip install -r requirements_phase2.txt

# Optional: Hidden Markov Model soft states
pip install -r requirements_phase3.txt
```

### 3.4 Install OpenMM for trajectory generation

OpenMM is needed only for **Step 2** (generating trajectories). Skip if you already have `.xtc`/`.dcd` files.

```bash
# Recommended via conda-forge:
conda install -c conda-forge openmm -y

# Alternative via pip (requires a C++ toolchain on some systems):
pip install openmm
```

### 3.5 Verify your installation

```bash
python -c "import deeptime; print('deeptime OK')"
python -c "import mdtraj; print('mdtraj OK')"
python -c "import openmm; print('openmm OK')"   # only if you installed it
```

---

## 4. Quick Start — Full Pipeline in 3 Commands

Run these three commands in order from the repository root. Each step builds on the previous one.

```bash
# 1. Download ≥50 small proteins from RCSB PDB
python scripts/download_pdb_dataset.py

# 2. Generate a short MD trajectory for each protein (needs OpenMM)
python scripts/generate_md_trajectories.py

# 3. Run the full ML pipeline on every protein
python run_all_proteins.py
```

That's it. Results land in `artifacts/` and `results/`.

---

## 5. Step-by-Step Guide

### Step 1 — Download Protein Structures

```bash
python scripts/download_pdb_dataset.py
```

**What it does:**
- Queries the RCSB PDB Search API with these filters:
  - Polymer entity type = Protein
  - Sequence length < 150 residues  
  - Single protein chain (no complexes)
  - Crystal resolution ≤ 3.0 Å
- Downloads PDB files and saves them as `data/{PDB_ID}/topology.pdb`
- Stops after downloading 50 proteins (configurable)

**What you'll see:**

```
============================================================
RCSB PDB Dataset Downloader
============================================================
[search] Querying RCSB for proteins: residues < 150, resolution ≤ 3.0 Å ...
[search] Found 3847 total matches, retrieved 200 PDB IDs.

[download] Attempting up to 200 proteins ...
  [ok]    1UBQ → data/1UBQ/topology.pdb
  [ok]    1L2Y → data/1L2Y/topology.pdb
  ...
  [progress] downloaded=10, skipped=0, failed=0
  ...
[done] Target of 50 downloads reached.

[summary] downloaded=50, skipped=0, failed=2
```

**Output structure after Step 1:**

```
data/
├── 1UBQ/
│   └── topology.pdb
├── 1L2Y/
│   └── topology.pdb
└── ... (50+ directories)
```

---

### Step 2 — Generate MD Trajectories

```bash
python scripts/generate_md_trajectories.py
```

**What it does** for each `data/{PDB_ID}/topology.pdb`:
1. Loads the PDB structure with OpenMM
2. Adds missing hydrogen atoms
3. Solvates the protein in a TIP3P explicit water box (1 nm padding)
4. Minimizes potential energy
5. Runs 50,000 NVT integration steps at 300 K (Langevin thermostat, 2 fs time step)
6. Saves the trajectory as `data/{PDB_ID}/traj.xtc` (via mdtraj) or `traj.dcd` (fallback)

> **Time estimate:** ~2–10 minutes per protein on a laptop CPU, depending on protein size. Run overnight for large batches.

**What you'll see:**

```
11:00:01 [INFO] Found 50 protein directories.
11:00:01 [INFO] [1UBQ] Loading structure: data/1UBQ/topology.pdb
11:00:02 [INFO] [1UBQ] Adding hydrogens and solvating...
11:00:05 [INFO] [1UBQ] System: 8247 atoms after solvation.
11:00:05 [INFO] [1UBQ] Minimizing energy...
11:00:08 [INFO] [1UBQ] Running 50000 steps...
...
11:08:23 [INFO] [1UBQ] Saved trajectory: data/1UBQ/traj.xtc
```

**Proteins that already have a trajectory are automatically skipped.**

---

### Step 3 — Run the ML Pipeline

```bash
python run_all_proteins.py
```

**What it does** for each protein directory that has both `topology.pdb` and a trajectory:

| Sub-step | Library | What happens |
|---|---|---|
| Feature extraction | mdtraj | Computes RMSD, Rg, native contacts, φ/ψ dihedrals per frame |
| tICA | deeptime | Projects features onto slow collective-motion coordinates |
| KMeans clustering | deeptime | Groups frames into 20 discrete conformational states |
| MSM | deeptime | Builds a reversible Markov State Model over the states |
| Anomaly scoring | scoring/anomaly_v2.py | Fuses rarity + transition surprise + local density signals |

**What you'll see:**

```
11:10:00 [INFO] ════════════════════════════════════════════════════
11:10:00 [INFO] Batch ML Pipeline — Ensemble Anomaly Maps
11:10:00 [INFO] ════════════════════════════════════════════════════
11:10:00 [INFO] Found 50 protein dataset(s) to process.

11:10:00 [INFO] ────────────────────────────────────────────────────
11:10:00 [INFO] [1/50] Processing: 1UBQ
11:10:00 [INFO] ────────────────────────────────────────────────────
11:10:00 [INFO] [1UBQ] ── Step 1/5: Feature extraction
11:10:01 [INFO] [1UBQ]   100 frames × 8 features
11:10:01 [INFO] [1UBQ] ── Step 2/5: tICA (lag=10, dim=5)
11:10:01 [INFO] [1UBQ] ── Step 3/5: Clustering (n_clusters=20)
11:10:02 [INFO] [1UBQ] ── Step 4/5: MSM (lag=10)
11:10:02 [INFO] [1UBQ] ── Step 5/5: Anomaly scoring
11:10:02 [INFO] [1UBQ] ✓ Pipeline complete. Mean score: 47.3

...

11:45:00 [INFO] ════════════════════════════════════════════════════
11:45:00 [INFO] Batch run complete: 49 succeeded, 1 failed.
11:45:00 [INFO] Artifacts → artifacts/
11:45:00 [INFO] Results   → results/
```

---

## 6. Using Your Own Data

If you already have an MD trajectory and don't need Steps 1–2, you can run the pipeline directly on your own data.

### Option A — Batch (multiple proteins in `data/`)

Place your data like this:

```
data/
├── MY_PROTEIN/
│   ├── topology.pdb      ← required
│   └── traj.xtc          ← required (or traj.dcd / trajectory.xtc / trajectory.dcd)
└── ANOTHER_PROTEIN/
    ├── topology.pdb
    └── trajectory.xtc
```

Then run:

```bash
python run_all_proteins.py
```

### Option B — Single trajectory (TAD-FM pipeline)

```bash
python run_tadfm.py \
    --top  path/to/topology.pdb \
    --traj path/to/trajectory.xtc \
    --out_dir outputs/my_run
```

**Outputs:**
- `outputs/my_run/segments.csv` — Segment-level anomaly scores
- `outputs/my_run/frame_scores.csv` — Per-frame anomaly scores

---

## 7. Output Files Explained

After a complete run of all three pipeline steps:

```
ensemble-anomaly-maps/
├── data/                          ← created by Steps 1 & 2
│   ├── 1UBQ/
│   │   ├── topology.pdb           (downloaded PDB structure)
│   │   └── traj.xtc               (generated MD trajectory)
│   └── ...
│
├── artifacts/                     ← created by Step 3
│   ├── 1UBQ/
│   │   ├── tica_coords.npy        tICA-projected coordinates   (T × dim)
│   │   ├── dtraj.npy              Discrete state trajectory    (T,)
│   │   ├── P.npy                  MSM transition matrix        (n_states × n_states)
│   │   └── pi.npy                 Stationary distribution      (n_states,)
│   └── ...
│
└── results/                       ← created by Step 3
    ├── 1UBQ/
    │   ├── frame_scores_dynamic.csv      Per-frame anomaly scores [0–100]
    │   └── residue_scores_dynamic.json   Per-residue anomaly scores
    └── ...
```

### frame_scores_dynamic.csv

```
frame,score_dynamic,component_rarity,component_transition_surprise,component_local_density
0,42.3,38.1,45.0,43.9
1,55.1,60.2,52.3,52.8
...
```

| Column | Description |
|---|---|
| `frame` | Frame index (0-based) |
| `score_dynamic` | Final fused anomaly score, range [0, 100] |
| `component_rarity` | Contribution from state rarity signal |
| `component_transition_surprise` | Contribution from transition surprise signal |
| `component_local_density` | Contribution from local density signal |

### residue_scores_dynamic.json

```json
{
  "MET1": 34.21,
  "GLN2": 41.85,
  "ILE3": 67.43,
  ...
}
```

Each key is a residue identifier. Values are anomaly scores in [0, 100] — higher means more anomalous / dynamic.

---

## 8. All Command-Line Options

### `scripts/download_pdb_dataset.py`

```
python scripts/download_pdb_dataset.py [OPTIONS]

Options:
  --data_dir DIR          Root data directory (default: data)
  --max_residues INT      Max residues per protein (default: 150)
  --max_resolution FLOAT  Max crystal resolution in Å (default: 3.0)
  --target INT            Stop after this many downloads (default: 50)
  --rows INT              Candidates requested from RCSB API (default: 200)
  --delay FLOAT           Pause between downloads in seconds (default: 0.5)
```

**Examples:**

```bash
# Download 100 proteins instead of 50
python scripts/download_pdb_dataset.py --target 100

# Be more selective: only very small, very high-resolution structures
python scripts/download_pdb_dataset.py --max_residues 80 --max_resolution 2.0

# Use a different output directory
python scripts/download_pdb_dataset.py --data_dir my_proteins
```

---

### `scripts/generate_md_trajectories.py`

```
python scripts/generate_md_trajectories.py [OPTIONS]

Options:
  --data_dir DIR          Root data directory (default: data)
  --steps INT             MD integration steps (default: 50000)
  --step_size FLOAT       Time step in picoseconds (default: 0.002)
  --temperature FLOAT     Temperature in Kelvin (default: 300.0)
  --padding FLOAT         Water box padding in nm (default: 1.0)
  --report_interval INT   Frames written every N steps (default: 500)
  --no_xtc                Keep DCD format even if mdtraj is available
  --overwrite             Re-generate trajectory even if one exists
```

**Examples:**

```bash
# Longer simulation (100k steps)
python scripts/generate_md_trajectories.py --steps 100000

# Higher temperature
python scripts/generate_md_trajectories.py --temperature 320

# Re-run everything from scratch (overwrite existing trajectories)
python scripts/generate_md_trajectories.py --overwrite
```

---

### `run_all_proteins.py`

```
python run_all_proteins.py [OPTIONS]

Options:
  --data_dir DIR          Root data directory (default: data)
  --artifacts_dir DIR     Numpy output directory (default: artifacts)
  --results_dir DIR       CSV/JSON output directory (default: results)
  --stride INT            Load every Nth frame (default: 1)
  --lag_tica INT          tICA lag time in frames (default: 10)
  --dim_tica INT          Number of tICA components (default: 5)
  --n_clusters INT        KMeans clusters (default: 20)
  --lag_msm INT           MSM lag time in frames (default: 10)
  --k_neighbors INT       k for local density signal (default: 10)
  --window INT            Anomaly score smoothing window (default: 5)
  --seed INT              Random seed (default: 42)
  --protein PDB_ID        Process only this protein (for debugging)
```

**Examples:**

```bash
# Run with faster/coarser settings (every 2nd frame, 10 clusters)
python run_all_proteins.py --stride 2 --n_clusters 10

# Run only one protein to test
python run_all_proteins.py --protein 1UBQ

# Use different output directories
python run_all_proteins.py --artifacts_dir my_artifacts --results_dir my_results

# More tICA components for complex proteins
python run_all_proteins.py --dim_tica 8 --lag_tica 20
```

---

## 9. Repository Structure

```
ensemble-anomaly-maps/
│
├── run_all_proteins.py          ← Batch ML pipeline runner (Step 3)
├── run_tadfm.py                 ← Single-protein TAD-FM pipeline
│
├── scripts/
│   ├── download_pdb_dataset.py  ← RCSB structure downloader (Step 1)
│   └── generate_md_trajectories.py  ← OpenMM trajectory generator (Step 2)
│
├── features/
│   └── compute_md_features.py   ← RMSD, Rg, contacts, φ/ψ dihedrals
│
├── segments/
│   └── segmenter.py             ← Trajectory segmentation algorithms
│
├── models/
│   └── autoencoder.py           ← Numpy autoencoder (no PyTorch needed)
│
├── msm/
│   ├── bootstrap_msm.py         ← Bootstrap uncertainty quantification
│   ├── select_lag_and_dim.py    ← VAMP-2 parameter selection
│   ├── soft_states.py           ← HMM soft state assignments
│   └── validation.py            ← Chapman-Kolmogorov tests
│
├── scoring/
│   ├── anomaly_v2.py            ← Multi-signal anomaly fusion
│   └── signals.py               ← RMSF, tICA importance, density
│
├── detect/
│   └── tadfm.py                 ← DBSCAN-based segment anomaly detection
│
├── analysis/
│   ├── local_ops.py             ← Per-residue Ramachandran, bonds, SASA
│   ├── deep_ops.py              ← PyTorch autoencoder (optional)
│   └── run_*.py                 ← Individual analysis scripts
│
├── configs/
│   └── pipeline.yaml            ← Default hyperparameters
│
├── data/                        ← Protein datasets (created at runtime)
├── artifacts/                   ← Numpy model outputs (created at runtime)
├── results/                     ← CSV/JSON scores (created at runtime)
│
├── requirements_phase1.txt      ← Core ML dependencies
├── requirements_phase2.txt      ← MD analysis (mdtraj)
└── requirements_phase3.txt      ← HMM soft states
```

---

## 10. Troubleshooting

### `ModuleNotFoundError: No module named 'deeptime'`

```bash
pip install deeptime>=0.4.0
```

### `ModuleNotFoundError: No module named 'mdtraj'`

```bash
pip install mdtraj
```

### `ModuleNotFoundError: No module named 'openmm'`

```bash
# Recommended (most reliable):
conda install -c conda-forge openmm -y

# Alternative:
pip install openmm
```

### RCSB download returns 0 results

The RCSB Search API is occasionally unavailable. Wait a few minutes and retry. Check your internet connection:

```bash
curl -s "https://search.rcsb.org/rcsbsearch/v2/query" -o /dev/null -w "%{http_code}\n"
# Should print 200 or 422 (both mean the server is reachable)
```

### `Too few frames` warning for a protein

The generated trajectory has fewer than 10 frames. This can happen if the simulation crashed. Re-run Step 2 with `--overwrite`:

```bash
python scripts/generate_md_trajectories.py --overwrite
```

### MSM building fails for a protein

The trajectory may be too short for the requested lag time. Try smaller lag values:

```bash
python run_all_proteins.py --lag_tica 5 --lag_msm 5 --n_clusters 10
```

### Running on a single protein to debug

```bash
# Step 3 only, for one protein:
python run_all_proteins.py --protein 1UBQ

# Or run the TAD-FM variant:
python run_tadfm.py \
    --top data/1UBQ/topology.pdb \
    --traj data/1UBQ/traj.xtc \
    --out_dir outputs/1UBQ_debug
```

### OpenMM is slow on CPU

OpenMM automatically uses a GPU (CUDA or OpenCL) if one is available. On CPU-only machines, reduce step count for faster testing:

```bash
python scripts/generate_md_trajectories.py --steps 5000 --report_interval 100
```

---

## Additional Documentation

| File | Contents |
|---|---|
| [QUICKSTART.md](QUICKSTART.md) | Single-trajectory quickstart (existing data) |
| [USAGE.md](USAGE.md) | Detailed parameter documentation |
| [PHASE1.md](PHASE1.md) | VAMP-2 model selection |
| [PHASE2.md](PHASE2.md) | Energetic and pocket features |
| [PHASE3.md](PHASE3.md) | Multi-signal scoring |
| [SCIENTIFIC_DOCUMENTATION.md](SCIENTIFIC_DOCUMENTATION.md) | Full scientific background |
| [PIPELINE_SUMMARY_FOR_BIOCHEMISTS.md](PIPELINE_SUMMARY_FOR_BIOCHEMISTS.md) | Non-CS overview |
