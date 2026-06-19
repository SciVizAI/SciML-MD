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
