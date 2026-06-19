# tools/

**Export utilities for SciViz integration.**

`export_for_asvs.py` packages pipeline outputs into ASVS-compatible
JSON files that SciViz can consume directly for visualisation.

## Exported files

| File | Contents |
|---|---|
| `hotspots_residue.json` | Fused hotspot score per residue |
| `anomaly_residue.json` | Dynamic anomaly score per residue |
| `rmsf_residue.json` | RMSF per residue |
| `tica_importance.json` | tICA loading-based importance |

## Usage
```bash
python tools/export_for_asvs.py \
    --topology data/1CRN/topology.pdb \
    --trajectory data/1CRN/traj.xtc \
    --metrics_dir results/1CRN \
    --output_dir exports/1CRN
```

These exports are what SciViz loads to colour protein structures
by anomaly score in the 3D viewer.
