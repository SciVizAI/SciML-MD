# QUICKSTART: Running the Ensemble-Anomaly-Maps Pipeline

This guide provides the exact step-by-step terminal commands to run the dynamic hotspot detection pipeline.

---

## Prerequisites

### 1. Install Dependencies

```bash
# Install all required packages
pip install -r requirements_phase1.txt
pip install -r requirements_phase2.txt
pip install -r requirements_phase3.txt
```

### 2. Prepare Your Data

You need:
- **Topology file**: `topology.pdb` (protein structure)
- **Trajectory file**: `trajectory.xtc` (MD simulation)

Example data location in this repo:
- Topology: `data/raw_trajectory/align_topol.pdb`
- Trajectories: `data/raw_trajectory/trajectory_0.xtc` (and others)

---

## Pipeline Execution Sequence

### STEP 1: Generate Features from Trajectory

Extract geometric features from your MD trajectory:

```bash
python tools/extract_features.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory_0.xtc \
    --output data/features.npy
```

---

### STEP 2: Run tICA + MSM Pipeline

Perform dimensionality reduction (tICA) and build a Markov State Model (MSM):

```bash
python tools/run_msm_tica.py \
    --features data/features.npy \
    --out_dir outputs/msm \
    --lag_tica 10 \
    --lag_msm 30 \
    --n_clusters 30
```

**Outputs:**
- `outputs/msm/tica_coords.npy` - tICA-projected coordinates
- `outputs/msm/dtraj.npy` - Discrete state trajectory
- `outputs/msm/P.npy` - Transition probability matrix
- `outputs/msm/pi.npy` - Stationary distribution
- `outputs/msm/frame_scores.csv` - Per-frame anomaly scores

---

### STEP 3: Compute All Metrics (Recommended)

Compute dynamic anomaly, RMSF, and tICA importance scores:

```bash
python tools/compute_all_metrics.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory_0.xtc \
    --msm_dir outputs/msm \
    --output_dir outputs/metrics \
    --normalization percentile \
    --low-percentile 0.05 \
    --high-percentile 0.95
```

**Outputs:**
- `outputs/metrics/hotspots_unified.json` - All metrics for viewer
- `outputs/metrics/residue_scores_dynamic.json` - Dynamic anomaly scores
- `outputs/metrics/residue_scores_rmsf.json` - RMSF/flexibility scores
- `outputs/metrics/residue_scores_tica_importance.json` - Slow-mode importance
- `outputs/metrics/frame_scores_dynamic.csv` - Per-frame scores

---

### STEP 4: Launch the Visualization Server

Start the Flask web server to view results:

```bash
python app/app.py
```

Then open in browser: **http://localhost:5051**

---

## Complete Quick Run (All Steps)

Copy and paste this entire block to run the basic pipeline:

```bash
# Step 1: Create directories
mkdir -p outputs/msm outputs/metrics

# Step 2: Extract features
python tools/extract_features.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory_0.xtc \
    --output data/features.npy

# Step 3: Build MSM + TICA model
python tools/run_msm_tica.py \
    --features data/features.npy \
    --out_dir outputs/msm \
    --lag_tica 10 \
    --lag_msm 30 \
    --n_clusters 30

# Step 4: Compute unified metrics
python tools/compute_all_metrics.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory_0.xtc \
    --msm_dir outputs/msm \
    --output_dir outputs/metrics

# Step 5: View results
python app/app.py
```

---

## Enhanced Pipeline (Optional)

### Phase 1: Model Selection with VAMP-2

For optimal parameter selection:

```bash
python tools/run_phase1.py \
    --features data/features.npy \
    --output outputs/phase1 \
    --config configs/pipeline.yaml
```

### Phase 2: Energy & Pocket Features

Generate additional energetic and structural features:

```bash
# Energy features
python tools/generate_energy.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory_0.xtc

# Pocket/cavity features  
python tools/generate_pockets.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory_0.xtc
```

### Phase 3: Enhanced Scoring

Use multi-signal anomaly scoring:

```bash
python tools/score_v2.py \
    --features data/features.npy \
    --msm_dir outputs/msm \
    --energy data/derived/residue_energy.parquet \
    --pockets data/derived/pockets.parquet
```

---

## Troubleshooting

### "Module not found" errors
```bash
# Make sure you're in the repository root
cd /path/to/ensemble-anomaly-maps

# Install dependencies
pip install -r requirements_phase1.txt
pip install mdtraj deeptime pyemma scikit-learn pandas numpy
```

### "File not found" errors
```bash
# Check if your data files exist
ls -la data/raw_trajectory/

# Expected files:
#   align_topol.pdb
#   trajectory_0.xtc
```

### MSM convergence issues
```bash
# Try with fewer clusters or different lag times
python tools/run_msm_tica.py \
    --features data/features.npy \
    --out_dir outputs/msm \
    --lag_tica 5 \
    --lag_msm 20 \
    --n_clusters 15
```

---

## Integration with ASVS Visualization Tool

If you're using the external **ASVS viewer** (https://github.com/chiranjibsur/asvs/tree/siya-integration), follow these steps to export and visualize your ML pipeline outputs:

### STEP 5: Export for ASVS Viewer

After running the ML pipeline (Steps 1-3), export the outputs in ASVS-compatible format:

```bash
# Run the export script to generate ASVS-compatible JSON files
python tools/export_for_asvs.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory_0.xtc \
    --msm_dir outputs/msm \
    --metrics_dir outputs/metrics \
    --output_dir /path/to/asvs/viewer
```

**Outputs for ASVS viewer:**
- `hotspots_residue.json` - Per-residue, per-frame hotspot scores
- `anomaly_residue.json` - Per-residue, per-frame anomaly scores  
- `rmsf_residue.json` - Per-residue RMSF (flexibility) scores
- `tica_importance.json` - Per-residue tICA importance scores

### STEP 6: Launch ASVS Viewer

```bash
# Navigate to ASVS viewer directory
cd /path/to/asvs

# Copy your trajectory files to the viewer folder
cp /path/to/ensemble-anomaly-maps/data/raw_trajectory/align_topol.pdb viewer/topology.pdb
cp /path/to/ensemble-anomaly-maps/data/raw_trajectory/trajectory_0.xtc viewer/trajectory.xtc

# Start the ASVS viewer
python app.py
```

Then open in browser: **http://localhost:5000/viewer**

### Alternative: Use Environment Variables

```bash
# Set paths to your ML pipeline outputs
export ASVS_PDB="/path/to/ensemble-anomaly-maps/data/raw_trajectory/align_topol.pdb"
export ASVS_XTC="/path/to/ensemble-anomaly-maps/data/raw_trajectory/trajectory_0.xtc"
export ASVS_HOTSPOTS_RES="/path/to/ensemble-anomaly-maps/outputs/metrics/hotspots_residue.json"
export ASVS_RMSF="/path/to/ensemble-anomaly-maps/outputs/metrics/rmsf_residue.json"
export ASVS_TICA="/path/to/ensemble-anomaly-maps/outputs/metrics/tica_importance.json"
export ASVS_ANOMALY="/path/to/ensemble-anomaly-maps/outputs/metrics/anomaly_residue.json"

# Start ASVS viewer
cd /path/to/asvs
python app.py
```

### Complete Pipeline with ASVS Integration

```bash
# ====== ML PIPELINE (ensemble-anomaly-maps) ======
cd /path/to/ensemble-anomaly-maps

# Step 1: Create directories
mkdir -p outputs/msm outputs/metrics

# Step 2: Extract features
python tools/extract_features.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory_0.xtc \
    --output data/features.npy

# Step 3: Build MSM + TICA model
python tools/run_msm_tica.py \
    --features data/features.npy \
    --out_dir outputs/msm \
    --lag_tica 10 \
    --lag_msm 30 \
    --n_clusters 30

# Step 4: Compute unified metrics
python tools/compute_all_metrics.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory_0.xtc \
    --msm_dir outputs/msm \
    --output_dir outputs/metrics

# Step 5: Export for ASVS viewer
python tools/export_for_asvs.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory_0.xtc \
    --msm_dir outputs/msm \
    --metrics_dir outputs/metrics \
    --output_dir /path/to/asvs/viewer

# ====== ASVS VIEWER ======
cd /path/to/asvs

# Copy trajectory files
cp /path/to/ensemble-anomaly-maps/data/raw_trajectory/align_topol.pdb viewer/topology.pdb
cp /path/to/ensemble-anomaly-maps/data/raw_trajectory/trajectory_0.xtc viewer/trajectory.xtc

# Launch viewer
python app.py
# Open http://localhost:5000/viewer
```

### ASVS Output File Formats

The ASVS viewer expects these specific JSON formats:

**hotspots_residue.json / anomaly_residue.json** (per-frame, per-residue):
```json
{
  "0": {"0": 0.05, "1": 0.10, "2": 0.00, ...},
  "1": {"0": 0.03, "1": 0.12, "2": 0.01, ...},
  ...
}
```
- Keys are frame indices (as strings)
- Values are dictionaries mapping residue index → score [0-1]

**rmsf_residue.json / tica_importance.json** (static per-residue):
```json
{
  "min": 0.45,
  "max": 8.68,
  "normalized": {"0": 0.05, "1": 0.04, "2": 0.03, ...}
}
```
- `min`/`max`: Original value range
- `normalized`: Scores normalized to [0-1] range

---

## Evaluation Metrics and Presentation

After running your anomaly detection pipeline, you can compute evaluation metrics (AUROC, AUPRC with 95% CIs) and generate presentation-ready figures:

### Compute Evaluation Metrics

```bash
# Auto-detect predictions from outputs/ and compute metrics
python tools/compute_presentation_metrics.py

# Using specific predictions file
python tools/compute_presentation_metrics.py \
    --predictions outputs/frame_scores.csv \
    --out-dir outputs/summary \
    --bootstrap 2000 \
    --seed 42

# Quick test with sample data
python tools/compute_presentation_metrics.py \
    --predictions tests/sample_predictions.csv \
    --out-dir outputs/test_metrics \
    --bootstrap 500
```

**Required input format (CSV or Parquet):**
- `frame`: Frame index
- `y_true`: Ground truth label (0 = normal, 1 = anomaly)
- `y_score`: Anomaly score from model
- `run_id` (optional): Identifier for different runs/methods

**Outputs:**
- `outputs/summary/metrics_summary.csv` - All evaluation metrics
- `outputs/summary/predictions_roc.png` - ROC curve with AUROC
- `outputs/summary/predictions_pr.png` - Precision-Recall curve with AUPRC
- `outputs/summary/score_distributions.png` - Score distributions with thresholds
- `outputs/summary/metrics_summary_per_run.csv` - Per-run metrics (if run_id present)

### Update Presentation with Figures

The Beamer presentation `presentation_populated.tex` is configured to include the generated figures. After running the metrics script:

1. Compile the LaTeX presentation:
```bash
pdflatex presentation_populated.tex
```

2. The presentation will automatically include:
   - ROC curve with AUROC and 95% CI
   - Precision-Recall curve with AUPRC and 95% CI
   - Score distribution plots with top-k thresholds

### Interpretation Demo

See `notebooks/compute_metrics_demo.ipynb` for an interactive demonstration of the metrics computation and interpretation guidelines for chemists.

---

## Summary of Key Files

| File                                | Purpose                           |
|-------------------------------------|-----------------------------------|
| `tools/extract_features.py`         | Extract features from trajectory  |
| `tools/run_msm_tica.py`             | Build tICA + MSM models           |
| `tools/compute_all_metrics.py`      | Compute all hotspot metrics       |
| `tools/compute_presentation_metrics.py` | Evaluation metrics & plots    |
| `tools/export_for_asvs.py`          | Export outputs for ASVS viewer    |
| `tools/run_phase1.py`               | VAMP-2 model selection            |
| `tools/generate_energy.py`          | Energy features (Phase 2)         |
| `tools/generate_pockets.py`         | Pocket features (Phase 2)         |
| `tools/score_v2.py`                 | Enhanced scoring (Phase 3)        |
| `app/app.py`                        | Built-in visualization server     |
| `presentation_populated.tex`        | Beamer presentation template      |
| `notebooks/compute_metrics_demo.ipynb` | Metrics interpretation demo   |

---

## Next Steps

- See **[USAGE.md](USAGE.md)** for detailed parameter explanations
- See **[PIPELINE_SUMMARY_FOR_BIOCHEMISTS.md](PIPELINE_SUMMARY_FOR_BIOCHEMISTS.md)** for scientific background
- See **[PHASE1.md](PHASE1.md)**, **[PHASE2.md](PHASE2.md)**, **[PHASE3.md](PHASE3.md)** for advanced features
- See **ASVS Repo**: https://github.com/chiranjibsur/asvs/tree/siya-integration for visualization tool docs
