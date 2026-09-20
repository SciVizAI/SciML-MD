# HANDOFF.md — start here

You are building the visualiser on top of this pipeline. This page tells you
where to start, which path is authoritative, and the one rule that matters.

---

## The short version

**Read `viewer/{SYSTEM}.json`. Render exactly what is in it. Render nothing that isn't.**

Three worked examples are committed at `viewer/` — you can build the entire
interface against them without running the pipeline or obtaining a single
trajectory.

---

## The one rule

This pipeline produces numbers that are **valid for some trajectories and
meaningless for others**, and which is which cannot be determined by looking at
the numbers. A trajectory too short for state populations to converge still
yields a perfectly well-formed column of state populations.

So the pipeline decides, per system, what may be shown — and the export
**omits** everything else. Not nulls, not flags, not greyed-out: absent.

If a channel is not in `frames.channels`, there is nothing to plot. The reason
is in `omitted[name]`, written for a human, and showing that sentence where the
chart would have been is the correct interface.

You never need to read `CLAIMS.md` or the validation report to build correctly.
The bundle is self-describing.

---

## Which path is authoritative

| Path | Use it? |
|---|---|
| `viewer/{SYS}.json` via `tools/export_for_viewer.py` | **yes — this one** |
| `results/{SYS}/trust.json` + CSVs | yes, if you'd rather read raw outputs (`INTEGRATION.md` §5) |
| `exports/{SYS}/hotspots_unified.json` via `tools/export_unified.py` | **no — deprecated** |

The third one is the trap, and it is an easy one to fall into: `PIPELINE_OVERVIEW.md`
labels it "combined for viewer", `exports/` contains built examples, and the
field names (`anomaly_score`, `hotspot`) read like exactly what a visualiser
wants. It carries no trust information, so it will happily hand you quantities
the pipeline has gated, under names asserting claims that were withdrawn. It now
refuses to run without an explicit override flag. **If you find yourself reading
`hotspots_unified.json`, you are on the wrong path.**

---

## The bundle

```jsonc
{
  "schema": "sciml-md/viewer@1",
  "system": "ATLAS_1g2r_A",

  "verdict": {
    "suitability": "suitable",      // suitable | marginal | unsuitable | unknown
    "recurrence":  "recurrent",     // recurrent | weak | non_recurrent | unknown
    "estimability":"unresolved",    // estimable | unresolved | unknown
    "displayable": true,
    "headline": "This trajectory is too short for state populations to have converged...",
    "reasons": ["..."]
  },

  "sampling": {
    "total_ns": 99, "stride_ns": 0.2, "shortfall_factor": 40,
    "caption": "99 ns at 0.2 ns per frame - 40x below the 4000 ns threshold..."
  },

  "diagnostics": { "basin_census": {...}, "recurrence": {...}, "estimability": {...} },

  "frames": {
    "index": [0, 1, 2, ...],
    "channels": {                    // ONLY what may be shown
      "local_density": { "status": "descriptive_only", "reason": "...", "values": [...] }
    }
  },

  "residues": {
    "ids": ["MET1", "GLN2", ...],
    "rmsf_ang": { "status": "ok", "values": [...] },     // the one unhedged quantity
    "scores":   { "status": "descriptive_only", "values": [...] }
  },

  "omitted": {                       // why each missing thing is missing
    "rarity": "100 ns of sampling is 40x below the 4000 ns convergence threshold..."
  },

  "display_rules": ["..."]
}
```

---

## Build it in this order

1. **The verdict panel.** `verdict.headline`, then `sampling.caption`, then the
   three diagnostics. This goes *above* any chart, not in a tooltip. It is the
   pipeline's strongest supported claim, and every number below it is over-read
   without it.
2. **The flexibility profile.** `residues.rmsf_ang` — externally verified
   (r = 0.892, slope 0.966), the only thing here you can show without a caveat.
3. **Surviving channels**, as time tracks, in neutral colour.
4. **The omissions**, rendered as explanatory text where the chart would be.

The three committed examples cover the shapes you'll hit:

| Example | Shape it exercises |
|---|---|
| `ATLAS_1g2r_A` | displayable, two channels shown, two omitted |
| `ATLAS_2hnu_A` | **unsuitable** — no model-derived data at all, RMSF only |
| `SAMPLE_1CRN` | unsuitable, plus recurrence `unknown` (too short to measure) |

Build against all three. The second and third are not edge cases — **45% of real
trajectories look like them.**

---

## Must not

- Render a channel that isn't in `frames.channels`, or fill an omission.
- Rank residues or frames into a "top N" or "hotspots" list.
- Threshold any score, or colour it on a red/severity scale.
- Use the words **anomaly, detect, hotspot, early warning, rare event**.
  Say *conformational landscape*, *state occupancy*, *transition irregularity*.
- Show a recurrence percentile without its stride and atom selection. Both are
  in `diagnostics.recurrence`; without them the number moves by 8× and 16×
  respectively and means nothing.

The naming matters more than it looks. Every one of those words asserts
something four experiments failed to support — and the interface is where a
withdrawn claim would reach a user.

---

## Fail closed

A missing or unparseable bundle means show an error, not raw scores. The
contract exists precisely because the raw scores are not self-describing.

---

## Running it yourself

```bash
pip install -r requirements.txt
python run_all_proteins.py --protein <SYSTEM>      # needs data/<SYSTEM>/{topology.pdb,traj.xtc}
python tools/export_for_viewer.py --all
pytest tests/test_production_scoring.py -q
```

---

## If you want the background

Not required to build, but it explains every rule above:

| File | What it is |
|---|---|
| [`PIPELINE.md`](PIPELINE.md) | what each stage computes and what it means |
| [`INTEGRATION.md`](INTEGRATION.md) | the raw-output contract, if you skip the bundle |
| [`CLAIMS.md`](CLAIMS.md) | what the software may and may not be said to do |
| [`validation/VALIDATION_REPORT.md`](validation/VALIDATION_REPORT.md) | the evidence, including a retraction |
| `validation/SciML-MD_Validation_Tracker.xlsx` | defects, tests, open items |

Questions about whether something may be displayed: `CLAIMS.md` is the single
source of truth, and the answer for anything not listed there is no.
