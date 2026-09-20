# PIPELINE.md — what SciML-MD computes, and what each number means

This is the technical description of the pipeline: every stage, what it
produces, and what that output is and is not evidence of.

It supersedes `PIPELINE_OVERVIEW.md`, which is accurate about mechanics but was
written while the project believed it was a hotspot detector, and carries
interpretive claims that have since been withdrawn.

- **What the software may claim:** [`CLAIMS.md`](CLAIMS.md)
- **How to consume its output:** [`INTEGRATION.md`](INTEGRATION.md)
- **Why the claims are what they are:** [`validation/VALIDATION_REPORT.md`](validation/VALIDATION_REPORT.md)

---

## 1. In one paragraph

SciML-MD takes a molecular dynamics trajectory and asks, **before fitting
anything**, whether that trajectory can support a Markov state model at all. It
answers with two model-free diagnostics. If the answer is no, it says so and
withholds the quantities that would be meaningless. If the answer is yes, it
builds the model and reports three descriptive channels plus an externally
verified flexibility profile. It is a triage and diagnostic tool, not a
discovery tool.

---

## 2. Stages

```
trajectory (.xtc/.dcd) + topology (.pdb)
   │
   ├─ 1. Equilibration trim      discard startup relaxation frames
   │
   ├─ 2. DIAGNOSTICS  ◀── run BEFORE any model is fitted
   │     ├─ basin census         do conformational states exist?
   │     └─ recurrence           are they ever revisited?
   │
   ├─ 3. Features                RMSD, Rg, contacts, phi/psi
   ├─ 4. tICA                    slow collective coordinates
   ├─ 5. KMeans                  microstates
   ├─ 6. MSM                     reversible MLE, active-set pruned
   │     └─ estimability         is there enough repeated evidence for P and pi?
   │
   ├─ 7. Channels                rarity · transition surprise · local density
   ├─ 8. TRUST CONTRACT  ◀── gates everything above
   │     └─ results/{SYS}/trust.json
   └─ 9. Export                  viewer/{SYS}.json
```

Stages 2 and 8 are what distinguish this pipeline from a standard MSM workflow.
Everything else is textbook.

---

## 3. The diagnostics

### 3.1 Basin census — do states exist?

Leader clustering at a 2 Å Cα-RMSD cutoff, then a count of clusters holding
≥5% of frames. Returns `suitable` / `marginal` / `unsuitable`.

This catches two opposite failures that both make an MSM meaningless:

- **One basin.** >95% of frames within the cutoff of a single medoid. There are
  no states to resolve; microstates will partition thermal noise.
- **Diffusive.** *No* cluster holds 5% of frames. The trajectory wanders without
  ever settling. An MSM has as little to find here as in a single basin, for the
  opposite reason.

**45% of ATLAS trajectories fail this.** Standard estimators do not check — they
return numbers regardless. (`CLAIMS.md` S-1)

### 3.2 Recurrence — are states revisited?

The closest pair of frames ≥10 ns apart, expressed as a percentile of the
ordinary consecutive-frame step distribution. Model-free: no clustering, no
tICA, no MSM.

An MSM estimates transition probabilities from *repeat visits*. States that
exist but are never revisited give you a transition matrix fitted to singletons.

**Two things must be reported with every percentile, or it is uninterpretable:**

| | Effect |
|---|---|
| **Frame stride** | a coarser save interval inflates the number ~8× |
| **Atom selection** | backbone vs all-atom moves it ~16× |

That second one is a finding in its own right. Measured across 25 ATLAS systems:

| Selection | Median percentile | Recurrent (≤25) |
|---|---|---|
| Cα | 4.0 | 88% |
| backbone | 4.2 | 88% |
| heavy atoms | 43.3 | 12% |
| all atoms | 64.3 | 8% |

**The backbone fold returns to visited states; the side chains never do.** The
pipeline takes its verdict from backbone, because conformational state identity
is a backbone property, and reports all-atom alongside — it answers a different
question (could a splice hide from a full-coordinate detector?). (`CLAIMS.md`
S-1b, report §22.1)

### 3.3 Estimability — is there enough repeated evidence?

Asked of the discrete trajectory the MSM is actually fitted to. Two conditions:

1. **Singleton mass** — the fraction of transition counts observed exactly once.
2. **Total sampling** against a convergence threshold (default 4 µs, Kozlowski &
   Grubmüller's lower bound for 50–112 residue proteins).

**Condition 1 alone is not sufficient and must never be used alone.** With 20
states over ~1,000 frames every state is revisited ~50 times and singleton mass
sits near 4% — comfortably passing on exactly the data where π was measured
*failing to generalise across replicates* (ρ = −0.515 against a +0.620 control).
Plenty of within-run repetition and no reproducible stationary distribution are
entirely compatible. Only condition 2 reflects that. (Report §15, §20.3)

---

## 4. The channels

| Channel | What it measures | Needs the MSM? |
|---|---|---|
| `rarity` | `1 − π(s)` — how unusual the current state's population is | yes |
| `transition_surprise` | `−log P(s_t → s_t+τ)` — how improbable this transition is | yes |
| `local_density` | kNN density in tICA space | no |
| `score_dynamic` | median fusion of whichever channels survived gating | — |

The three are **dissociable**, measured within-subject: under relocation the
static channels show an effect of exactly +0.000 while transition surprise shows
+0.149. Showing all three is therefore informative rather than three views of
one number. (`CLAIMS.md` S-3)

**None of them is a detector.** Four experimental designs failed to support the
detection claim, and a fifth showed the benchmark that would test it properly
cannot be built at 100 ns sampling. The claim is *untested*, not refuted — which
is a different status and is stated that way deliberately. (Report §12, §18)

---

## 5. The trust contract

`msm/trust.py` turns `CLAIMS.md` from a document into a control. Every scored
system gets `results/{SYS}/trust.json` declaring each channel:

| Status | Meaning |
|---|---|
| `ok` | display freely — **only `residue_rmsf` reaches this** |
| `descriptive_only` | plot as a track; never rank, threshold, or colour by severity |
| `withheld` | do not display; values are NaN, and absent from the viewer bundle |

Three properties worth knowing:

- **No channel can be promoted above its ceiling.** A perfect trajectory cannot
  earn a detection claim, because that is not a per-trajectory question.
- **Missing diagnostics fail closed.** An absent check is never read as a pass.
- **A contract failure aborts the write.** The pipeline returns without emitting
  scores rather than emitting ungated ones.

`residue_rmsf` is the single exception: externally verified against published
ATLAS values at r = 0.892, slope 0.966 (median |err| 10.6%, max 40.5%, n = 21).
It is the only quantity here with an external check and the only one that may be
shown without a caveat. (`CLAIMS.md` S-2)

---

## 6. Outputs

| Path | Contents |
|---|---|
| `results/{SYS}/trust.json` | **the contract** — read this first, always |
| `results/{SYS}/frame_scores_dynamic.csv` | per-frame scores; withheld channels are NaN by design |
| `results/{SYS}/residue_scores_rmsf.json` | per-residue RMSF (Å) — the verified quantity |
| `results/{SYS}/residue_scores_dynamic.json` | per-residue scores, `[0,100]` |
| `results/{SYS}/preflight.json` | raw diagnostic output |
| `artifacts/{SYS}/*.npy` | tICA coords, discrete trajectory, `P`, `π` |
| `viewer/{SYS}.json` | the governed bundle for the interface |

---

## 7. Vocabulary

The word **"anomaly"** must not appear in user-facing copy. It asserts the
withdrawn claim in a single word. It survives in the repository name
(`ensemble-anomaly-maps`) and one module path (`scoring/anomaly_v2.py`) because
renaming those breaks import paths and the public clone URL — a decision about
project identity rather than a correction.

Replacement vocabulary: *conformational landscape*, *state occupancy*,
*transition irregularity*. Descriptive nouns, no detection verb.

Claims take the form **"is sensitive to temporal discontinuity"**, never
"detects conformational transitions". Rejecting a surrogate null shows the
surrogate's assumptions fail, not that your alternative is true (Timmer 2000).

---

## 8. Known limits

- Every negative result is conditional on **≤100 ns** sampling, which is 13–27×
  below the published convergence threshold for proteins of this size. They must
  always be stated with that scope.
- π is withheld on essentially all currently available data, mdCATH's ~500 ns
  included. That is the finding, not a bug.
- Whether the residue map corresponds to functional sites is **open** — Phase 4
  was inconclusive because the labels failed a validity gate.
- The applicability screen is a principled gate with a stated criterion, not a
  classifier with a measured error rate. No ground-truth label of
  MSM-suitability exists to validate it against.
