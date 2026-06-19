#!/usr/bin/env python3
# ================================================================ #
#  Script:      add_headers.py
#  Description: Automatically adds SciVizAI-style file headers and
#               function sub-headers to all Python files in SciML-MD
#  Author:      Siya Jethliya
#  Copyright (c) 2026 SciVizAI — All rights reserved.
#  Usage:       python add_headers.py --repo /path/to/SciML-MD
# ================================================================ #

import argparse
import ast
import os
import re
from pathlib import Path

# ---------------------------------------------------------------- #
# File descriptions — what each module does in one line
# ---------------------------------------------------------------- #
FILE_DESCRIPTIONS = {
    "run_all_proteins.py":                   "Batch ML pipeline orchestrator — runs all 5 steps for every protein in data/",
    "batch_runner.py":                       "End-to-end runner — download, MD generation, ML pipeline, ASVS export",
    "scripts/download_pdb_dataset.py":       "RCSB PDB Search API client — filters and downloads protein structures",
    "scripts/generate_md_trajectories.py":   "OpenMM MD trajectory generator — solvate, minimise, NVT simulation → .xtc",
    "features/compute_md_features.py":       "Physics feature extraction — RMSD, Rg, native contacts, φ/ψ dihedrals per frame",
    "features/__init__.py":                  "Features package initialiser",
    "scoring/anomaly_v2.py":                 "Multi-signal anomaly fusion — rarity + transition surprise + local density → [0-100]",
    "scoring/signals.py":                    "Individual signal computation — RMSF, tICA importance, local density",
    "scoring/__init__.py":                   "Scoring package initialiser",
    "msm/select_lag_and_dim.py":             "VAMP-2 hyperparameter grid search — selects optimal tICA lag and dimension",
    "msm/validation.py":                     "MSM validation — Chapman-Kolmogorov test, implied timescale analysis",
    "msm/bootstrap_msm.py":                  "Bootstrap uncertainty quantification for stationary distribution and transition matrix",
    "msm/soft_states.py":                    "HMM soft state assignments — probabilistic state membership per frame",
    "msm/input_validation.py":               "Input validation utilities for MSM pipeline inputs",
    "msm/reproducibility.py":                "Reproducibility utilities — deterministic seed management across pipeline steps",
    "msm/__init__.py":                       "MSM package initialiser",
    "tools/export_for_asvs.py":              "ASVS-compatible JSON exporter — packages residue scores for SciViz visualisation",
    "tests/run_all_validation.py":           "Runs all 13 validation tests and saves results",
    "tests/test_chapter9_evaluation.py":     "Chapter 9 evaluation tests — results and case study validation",
    "tests/test_dataset_validation.py":      "Dataset validation tests — trajectory and topology integrity checks",
    "tests/test_integration.py":             "Integration tests — end-to-end pipeline regression checks",
    "tests/test_pipeline_edge_cases.py":     "Edge case tests — short trajectories, zero-variance features, disconnected states",
    "tests/test_reproducibility.py":         "Reproducibility tests — deterministic output verification across runs",
    "tests/test_scientific_validation.py":   "Scientific validation tests — Chapman-Kolmogorov, implied timescales",
    "tests/test_signals.py":                 "Signal tests — rarity, transition surprise, local density unit tests",
    "tests/test_statistical_validation.py":  "Statistical validation tests — bootstrap confidence intervals and signal stability",
}

AUTHOR      = "Siya Jethliya"
COPYRIGHT   = "2026 SciVizAI — All rights reserved."
SEPARATOR_H = "# " + "=" * 62 + " #"
SEPARATOR_L = "# " + "-" * 62 + " #"

# ---------------------------------------------------------------- #
# Build the file-level header block
# ---------------------------------------------------------------- #
def make_file_header(rel_path: str, description: str) -> str:
    lines = [
        SEPARATOR_H,
        f"#  Module:      {rel_path}",
        f"#  Description: {description}",
        f"#  Author:      {AUTHOR}",
        f"#  Copyright (c) {COPYRIGHT}",
        SEPARATOR_H,
    ]
    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------- #
# Extract top-level function/class names and their line numbers
# ---------------------------------------------------------------- #
def get_definitions(source: str) -> list[tuple[int, str, str]]:
    """
    Returns list of (lineno, kind, name) for top-level defs.
    kind is 'def' or 'class'.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    defs = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.append((node.lineno, "def", node.name))
        elif isinstance(node, ast.ClassDef):
            defs.append((node.lineno, "class", node.name))
    return sorted(defs, key=lambda x: x[0])

# ---------------------------------------------------------------- #
# Insert function sub-headers above each top-level def/class
# ---------------------------------------------------------------- #
def insert_function_subheaders(source: str) -> str:
    lines = source.splitlines(keepends=True)
    defs = get_definitions(source)

    if not defs:
        return source

    # Work backwards so line numbers stay valid after insertions
    for lineno, kind, name in reversed(defs):
        idx = lineno - 1  # ast lineno is 1-based

        # Don't add if there's already a separator immediately above
        if idx > 0 and SEPARATOR_L[2:10] in lines[idx - 1]:
            continue

        label = f"{'Function' if kind == 'def' else 'Class'}: {name}"
        sub_header = f"\n{SEPARATOR_L}\n# {label}\n{SEPARATOR_L}\n"
        lines.insert(idx, sub_header)

    return "".join(lines)

# ---------------------------------------------------------------- #
# Strip any existing SciVizAI header at the top of the file
# ---------------------------------------------------------------- #
def strip_existing_header(source: str) -> str:
    if not source.startswith(SEPARATOR_H):
        return source
    # Find the closing separator line
    end = source.find(SEPARATOR_H, len(SEPARATOR_H))
    if end == -1:
        return source
    end += len(SEPARATOR_H)
    return source[end:].lstrip("\n")

# ---------------------------------------------------------------- #
# Process a single Python file
# ---------------------------------------------------------------- #
def process_file(filepath: Path, repo_root: Path) -> bool:
    rel = filepath.relative_to(repo_root).as_posix()
    description = FILE_DESCRIPTIONS.get(rel, "SciVizAI SciML-MD pipeline module")

    try:
        source = filepath.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  [skip] Cannot read {rel}: {e}")
        return False

    # Strip old header if present
    source = strip_existing_header(source)

    # Build new header
    header = make_file_header(rel, description)

    # Insert function sub-headers
    source_with_subs = insert_function_subheaders(source)

    # Final content
    new_content = header + "\n" + source_with_subs

    filepath.write_text(new_content, encoding="utf-8")
    print(f"  [ok]   {rel}")
    return True

# ---------------------------------------------------------------- #
# Folder README content per directory
# ---------------------------------------------------------------- #
FOLDER_READMES = {
    "scripts": """\
# scripts/

**Stage 1 & 2 of the SciML-MD pipeline.**

## Stage 1 — Download Protein Structures
`download_pdb_dataset.py` queries the RCSB PDB Search API and downloads
small proteins (< 150 residues, ≤ 3.0 Å resolution) as `topology.pdb` files.

```bash
python scripts/download_pdb_dataset.py --target 50
```

## Stage 2 — Generate MD Trajectories
`generate_md_trajectories.py` runs short OpenMM NVT simulations (50k steps,
300 K, Langevin thermostat) and saves trajectories as `traj.xtc`.
GPU is used automatically if CUDA or OpenCL is available.

```bash
python scripts/generate_md_trajectories.py --steps 50000
```

**Outputs:** `data/{PDB_ID}/topology.pdb`, `data/{PDB_ID}/traj.xtc`
""",

    "features": """\
# features/

**Stage 3 of the SciML-MD pipeline — Feature Extraction.**

`compute_md_features.py` loads an MD trajectory via MDTraj and computes
8 physics-based scalars per frame:

| Feature | Description |
|---|---|
| RMSD | Deviation from first frame (Å) |
| Rg | Radius of gyration |
| Contacts | Native Cα contacts within 8 Å |
| sin(φ), cos(φ) | Backbone phi dihedral encoding |
| sin(ψ), cos(ψ) | Backbone psi dihedral encoding |

**Output:** `T × 8` feature matrix (numpy array)

These features are hand-designed rather than learned — a deliberate
choice for interpretability and epistemic discipline (see thesis Chapter 4).
""",

    "msm": """\
# msm/

**Stages 4 & 5 of the SciML-MD pipeline — tICA and Markov State Model.**

## Files

| File | Purpose |
|---|---|
| `select_lag_and_dim.py` | VAMP-2 grid search to select optimal tICA lag and dimension |
| `validation.py` | Chapman-Kolmogorov test and implied timescale analysis |
| `bootstrap_msm.py` | Bootstrap uncertainty quantification for π and P |
| `soft_states.py` | HMM soft state assignments |
| `reproducibility.py` | Deterministic seed management |
| `input_validation.py` | Input integrity checks |

## What happens here

1. **tICA** — finds the slowest collective motions in the 8D feature space
2. **KMeans** — discretises the tICA space into 20 conformational states
3. **MSM** — estimates transition matrix P and stationary distribution π

**Outputs:** `tica_coords.npy`, `dtraj.npy`, `P.npy`, `pi.npy`

The MSM is validated with Chapman-Kolmogorov tests before anomaly scoring
proceeds. Invalid models are rejected rather than accepted by default.
""",

    "scoring": """\
# scoring/

**Stage 6 of the SciML-MD pipeline — Anomaly Scoring.**

This is the core of SciML-MD. Three independent signals are computed,
rank-normalised, fused by median, and smoothed to produce hotspot scores.

## Signals

| Signal | Formula | High score means |
|---|---|---|
| Rarity | `1 − π[s_t]` | Frame is in a rarely-visited state |
| Transition surprise | `−log P[s_t → s_{t+lag}]` | Unexpected state transition |
| Local density | `−mean k-NN distance in tICA space` | Frame is isolated in conformation space |

## Why median fusion?
Mean fusion lets one dominant signal skew the result. Median fusion
means all three signals must agree — more robust, less overfit.

## Residue score formula
```
residue_score = (0.5 × RMSF_norm[i] + 0.5 × mean_frame_score) × 100
```

**Outputs:** `frame_scores_dynamic.csv`, `residue_scores_dynamic.json`
""",

    "tools": """\
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
python tools/export_for_asvs.py \\
    --topology data/1CRN/topology.pdb \\
    --trajectory data/1CRN/traj.xtc \\
    --metrics_dir results/1CRN \\
    --output_dir exports/1CRN
```

These exports are what SciViz loads to colour protein structures
by anomaly score in the 3D viewer.
""",

    "tests": """\
# tests/

**Validation and testing suite for SciML-MD.**

## Test files

| File | What it validates |
|---|---|
| `test_integration.py` | End-to-end pipeline regression |
| `test_signals.py` | Rarity, transition surprise, local density unit tests |
| `test_reproducibility.py` | Deterministic output across runs |
| `test_scientific_validation.py` | Chapman-Kolmogorov, implied timescales |
| `test_statistical_validation.py` | Bootstrap confidence intervals |
| `test_dataset_validation.py` | Trajectory and topology integrity |
| `test_pipeline_edge_cases.py` | Short trajectories, disconnected states |
| `test_chapter9_evaluation.py` | Thesis Chapter 9 results validation |
| `run_all_validation.py` | Runs all tests, saves to TEST_RESULTS.md |

## Run all tests
```bash
python tests/run_all_validation.py
```

## Run individual test
```bash
pytest tests/test_signals.py -v
```
""",
}

# ---------------------------------------------------------------- #
# Write per-folder README files
# ---------------------------------------------------------------- #
def write_folder_readmes(repo_root: Path) -> None:
    for folder, content in FOLDER_READMES.items():
        readme_path = repo_root / folder / "README.md"
        readme_path.write_text(content, encoding="utf-8")
        print(f"  [readme] {folder}/README.md")

# ---------------------------------------------------------------- #
# Main entry point
# ---------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Add SciVizAI-style headers to all Python files in SciML-MD"
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Path to SciML-MD repo root (default: current directory)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be changed without writing files"
    )
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    if not repo_root.exists():
        print(f"Error: {repo_root} does not exist")
        return 1

    print(f"\nSciML-MD Header Injector")
    print(f"Repo: {repo_root}")
    print(f"{'DRY RUN — no files written' if args.dry_run else 'Writing files...'}")
    print("=" * 50)

    # Skip these directories entirely
    skip_dirs = {".git", "__pycache__", ".venv", "node_modules", "sample_data"}

    py_files = [
        p for p in repo_root.rglob("*.py")
        if not any(part in skip_dirs for part in p.parts)
        and p.name != "add_headers.py"  # don't process this script itself
    ]

    print(f"\nFound {len(py_files)} Python files\n")

    ok = 0
    for filepath in sorted(py_files):
        if args.dry_run:
            rel = filepath.relative_to(repo_root).as_posix()
            print(f"  [would process] {rel}")
        else:
            if process_file(filepath, repo_root):
                ok += 1

    print(f"\nWriting folder READMEs...")
    if not args.dry_run:
        write_folder_readmes(repo_root)

    print(f"\n{'=' * 50}")
    print(f"Done. {ok}/{len(py_files)} files processed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
