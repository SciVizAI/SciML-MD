# INTEGRATION.md — the contract between the pipeline and the visualiser

The visualiser should never need to read `CLAIMS.md` or the validation report.
Everything it needs to decide what to render is in one file per system:

```
results/{SYSTEM}/trust.json
```

Read that file **first**, before touching any scores. It is authoritative.

---

## 1. Files the pipeline writes

| File | What it is |
|---|---|
| `trust.json` | **The contract.** Verdict, diagnostics, per-channel status, display rules |
| `frame_scores_dynamic.csv` | Per-frame scores. Withheld channels are `NaN` **by design** |
| `residue_scores_dynamic.json` | Per-residue scores, `[0, 100]` |
| `residue_scores_rmsf.json` | Per-residue RMSF (Å) — the externally verified quantity |
| `preflight.json` | Raw diagnostic output (equilibration, suitability, recurrence) |
| `artifacts/{SYSTEM}/*.npy` | tICA coords, discrete trajectory, transition matrix, π |

---

## 2. `trust.json`

```jsonc
{
  "schema": "sciml-md/trust@1",
  "claims_version": "2026-08-31",
  "system": "1g2r_A",
  "verdict": "suitable",              // suitable | marginal | unsuitable | unknown
  "recurrence_verdict": "recurrent",  // recurrent | weak | non_recurrent | unknown
  "estimability_verdict": "unresolved",
  "displayable": true,
  "headline": "This trajectory is too short for state populations to have converged...",
  "channels": {
    "rarity":              { "status": "withheld",         "reason": "...", "claims": "W-3" },
    "transition_surprise": { "status": "withheld",         "reason": "...", "claims": "W-2, W-4" },
    "local_density":       { "status": "descriptive_only", "reason": "...", "claims": "S-3" },
    "score_dynamic":       { "status": "descriptive_only", "channels_used": ["local_density"] }
  },
  "diagnostics": { "basin_census": {...}, "recurrence": {...}, "estimability": {...} },
  "display_rules": [ "..." ]
}
```

### The three statuses

| Status | What the UI must do |
|---|---|
| `ok` | Display freely. *(No channel currently reaches this.)* |
| `descriptive_only` | Display as a **track over time**. Never rank into a top-N list, never threshold into "hotspots", never colour on a severity scale. |
| `withheld` | **Do not display.** Values in the CSV are `NaN`. |

`descriptive_only` is the ceiling for every scoring channel, and no trajectory
can raise it — the ceiling reflects what validation supports in general, not
what this trajectory does. See `CLAIMS.md` W-1.

---

## 3. What the UI should show, in order

1. **The verdict and why.** `headline`, then the three diagnostics. This is the
   product's strongest claim (`CLAIMS.md` S-1, S-1b) and it belongs above the
   fold, not in a tooltip.
2. **Sampling context.** `diagnostics.recurrence.total_ns` and `.stride_ns`, and
   `diagnostics.estimability.shortfall_factor` — "100 ns, 40× below the
   published convergence threshold" is the single most useful thing a user can
   see. Without it every number below is over-read.
3. **The flexibility profile** (`residue_scores_rmsf.json`). Externally verified
   against ATLAS at r = 0.892, slope 0.966 — the only quantity here with an
   external check, so it can be shown without hedging. Quote the error too.
4. **Surviving channels only**, as time tracks, greyed-neutral.

## 4. Things the UI must not do

- Render a `withheld` channel, or fill its `NaN`s.
- Rank residues or frames by score, or produce a "top hotspots" list.
- Use a red/severity colour scale for any score.
- Use the word **"anomaly"**, or "detect", "hotspot", "early warning", "rare
  event". `CLAIMS.md` §4 has the replacement vocabulary: *conformational
  landscape*, *state occupancy*, *transition irregularity*.
- Show a recurrence percentile without its **stride** and **atom selection**.
  Both change the number materially — on `1g2r_A` the same check returns the
  0.6th percentile on Cα and the 62.7th on all atoms.

## 5. Reading it

```python
import json, pandas as pd

trust = json.load(open(f"results/{sid}/trust.json"))
if not trust["displayable"]:
    show_verdict_only(trust["headline"], trust["diagnostics"]); return

df = pd.read_csv(f"results/{sid}/frame_scores_dynamic.csv")
for name, ch in trust["channels"].items():
    if ch["status"] == "withheld":
        continue                       # NaN column; do not plot, do not fill
    plot_track(df[f"component_{name}"], label=name, caption=ch["reason"])
```

**Fail closed.** A missing or unparseable `trust.json` means show nothing but an
error. Never fall back to rendering raw scores — the contract exists precisely
because the raw scores are not self-describing.

## 6. Stability

`schema` is `sciml-md/trust@1`. Additive changes keep the version; anything that
changes the meaning of an existing field bumps it. `claims_version` tracks
`CLAIMS.md` — if it moves, re-read the register before shipping UI copy.
