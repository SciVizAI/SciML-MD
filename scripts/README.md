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
