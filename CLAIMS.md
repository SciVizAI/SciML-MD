# CLAIMS.md — what SciML-MD is allowed to say

**Status: settled 2026-08-30 (OI-31), revised 2026-08-31 after Phase 8 (§18).**
**This file is the single source of truth.**
If a claim is not listed here as `SUPPORTED`, no README, paper, slide, UI string,
docstring, or demo may assert it. Adding a claim means adding a row here first,
with the evidence, before the code or copy ships.

Evidence lives in `validation/VALIDATION_REPORT.md`; section numbers are cited
per row.

---

## 0. The settlement in one paragraph

SciML-MD was built and documented as a **dynamic hotspot detector**. Four
experimental designs failed to support that claim (§12, §13, §14, §15), and
Phase 8 (§18) then found the reason all four failed together: **at 100 ns these
trajectories are too SHORT for recurrence to be usable.** Refined in Phase 8b
(§19): the protein *does* revisit conformations — the closest temporally distant
pair sits at the 7.4th percentile of a 1 ns displacement — but the trajectories
are not long enough for that recurrence to be exploitable, either by an
estimator or by a benchmark. The limit is trajectory length, not the physics.

This changes the settlement in an important way. The detection claim is
**not refuted — it is untestable on this data**, and stating it as refuted
would be the same overclaim as the original, pointed backwards. What is
withdrawn is *asserting* detection without evidence. What is **supported**, and
is now the centre of the product, is the diagnostic layer: **SciML-MD tells you
whether a trajectory can support Markov-state conclusions at all — before you
draw them — and gives you an externally verified descriptive view when it can.**

tICA and the MSM are not demoted by this. They are the machinery the diagnosis
runs on, and the diagnosis is the part nobody else ships.

---

## 1. SUPPORTED — claims we can make

### S-1 · Applicability-domain screen  ⭐ the strongest claim we have

> "SciML-MD tells you, before you build a Markov state model, whether your
> trajectory can carry one."

`msm.preflight.assess_suitability` performs a basin census — leader clustering
at an RMSD cutoff, then the number of clusters occupying ≥5% of frames — and
returns `suitable` / `marginal` / `unsuitable`. On ATLAS, **45% of trajectories
fail it.**

- **Evidence:** §11.4; screen results in `validation/phase3_screen.json`.
- **Why it matters:** every downstream MSM quantity (π, P, timescales, CK test)
  is meaningless on a single-basin diffusive ensemble, and the estimators do not
  say so — they return numbers. We could not find a published tool that
  performs this check as a gate.
- **Permitted phrasing:** "applicability-domain screen", "MSM suitability
  triage", "tells you when *not* to trust an MSM on this trajectory".
- **Not permitted:** calling it validated *against a ground-truth label of
  MSM-suitability*, because no such label exists. It is a principled gate with a
  stated criterion, not a classifier with a measured error rate. Say so.

### S-1b · Recurrence diagnostic  ⭐ new, from Phase 8

> "SciML-MD measures, in one model-free number, whether a trajectory ever
> revisits a conformation — and therefore whether any MSM quantity computed on
> it can mean anything."

The closest temporally distant frame pair, expressed as a percentile of a
reference step distribution. Report it at the delivered stride **and** at a
fixed 1 ns physical lag — the two answer different questions and the stride-only
version is confounded by the save interval (§19.1).

On ATLAS: **64.3rd** percentile at the delivered 100 ps stride (1/25 systems
admit an invisible splice), **24.0th** re-delivered at 1 ns (5/25), and
**7.4th** against a 1 ns physical lag. Constructibility rises monotonically with
trajectory length at fixed stride (§19.4), so the binding constraint is length.

- **Evidence:** §18.3–18.4, §19.3–19.4; `validation/phase8_feasibility.py`,
  `validation/phase8_recurrence.py`.
- **Must be reported with its stride.** A single percentile with no stride
  attached is uninterpretable and can be inflated 8x by the save interval alone.
- **Why it matters:** S-1 (the basin census) asks whether states *exist*.
  This asks whether they are ever *revisited*. Both are required before an MSM
  means anything, and neither is published elsewhere as a gate. It requires no
  model fit.
- **Not permitted:** calling it validated against a ground-truth sampling-adequacy
  label. There isn't one. It is a measured property with a stated threshold.

### S-2 · Per-residue flexibility profile, externally verified

> "Per-residue flexibility (RMSF) agrees with independently published values."

Cross-checked against ATLAS's own published `avg_RMSF`: **r = 0.892, slope
0.966**, median absolute error 10.6%, max 40.5% (n = 21).

- **Evidence:** §11.4, `validation/phase3_diagnose_traj.py::cross_check`.
- **Permitted phrasing:** report the numbers, including the max error. This is
  the only externally-verified quantity in the pipeline.
- **Not permitted:** "within ~10%" — that was an overstatement, corrected before
  publication (only 9/21 fall within 10%). Quote r, slope, and median error.

### S-3 · The three channels are non-redundant

> "Rarity, transition surprise, and local density respond to different
> properties of the trajectory."

Phase 5 measured this within-subject: under relocation, static detectors show a
relocation effect of **exactly +0.000** while transition surprise shows **+0.149**.

- **Evidence:** §13.2–13.3.
- **Permitted phrasing:** the channels are dissociable, so showing all three is
  informative rather than three views of one number.
- **Not permitted:** that fusing them detects more anomalies. Dissociation is
  not detection (§13.4).

### S-4 · Determinism and reproducibility

Same input + same seed → identical states, scores, and figures. 28 tests,
`tests/test_production_scoring.py`.

### S-5 · tICA as a representation — never tested, never in doubt

**Correction to the 2026-08-30 settlement, which swept too broadly.** No
criterion in any phase tests tICA. tICA is a *representation* claim — these are
the slow collective coordinates — and it is supported by the standard
literature (Perez-Hernandez 2013) plus our own external check: the flexibility
profile built on these features matches published ATLAS values at r = 0.892.
**tICA may be presented as a selling point without qualification.**

Likewise, the MSM withdrawal is narrower than it first read. What failed is
(a) pi at <=100 ns and (b) transition surprise as a *timing* signal. The state
decomposition, the transition matrix, and the connectivity analysis are not
falsified; they are *undetermined* on non-recurrent data, which is a different
status and is what S-1/S-1b exist to detect.

### S-6 · Standard methods, correctly implemented

tICA → KMeans → reversible MLE MSM with active-set pruning is textbook and our
implementation passes numerical and edge-case suites
(`validation/baseline_numerical.py`, `baseline_edge_cases.py`). We may claim a
**correct implementation**; we may not claim that a correct implementation of
these methods yields the scientific conclusions in §2.

---

## 2. WITHDRAWN — claims that must be removed from all copy

| ID | Claim as currently written | Where it appears | Why it is dead |
|---|---|---|---|
| **W-1** | "detect dynamic hotspot residues" | `README.md:5`, `PIPELINE_OVERVIEW.md:1,22` | Unsupported, **not refuted** (revised after §18). Phase 3 showed a four-line `abs(diff)` detector beating the pipeline (0.839 vs 0.814, PR-AUC 0.227 vs 0.082, p = 0.0044) — but Phase 8 showed that benchmark was the *only one the data admits*, because no geometrically invisible transition exists at 100 ns. The claim may not be asserted; it may be pursued (**O-2**). |
| **W-2** | "per-frame anomaly scores identifying unusual conformations" | `PIPELINE_OVERVIEW.md:29`, `frame_scores_dynamic.csv` docs | Same. The score is computable and reproducible; it is not shown to identify anything a trivial baseline does not. |
| **W-3** | "State rarity (π) directly identifies thermodynamically rare conformations" | `PIPELINE_OVERVIEW.md:754` | Phase 7 (§15): ATLAS replicates **anti-occupy** each other's states, ρ = −0.515 against a random-split control of **+0.620**. At 100 ns there is no stable equilibrium distribution for π to estimate. π is not a thermodynamic quantity here. |
| **W-4** | "Transition surprise identifies kinetically rare events" | `PIPELINE_OVERVIEW.md:754` | Phase 6 (§14): surprise fires 26.0 frames ahead vs random's 25.0, p = 0.487 — no timing information **at this sampling**. §18 gives the mechanism: with no recurrence, the transition matrix is estimated almost entirely from singletons, so "rare" and "unseen" are indistinguishable. |
| **W-5** | "Often identifies: hinge residues, allosteric nodes, domain linkers" | `PIPELINE_OVERVIEW.md:488` | Never tested until Phase 4, which was **inconclusive** — labels failed the validity gate on most systems (OI-16 open). An untested claim stated as a habitual outcome. |
| **W-6** | The "Prime hotspot / Rigid hotspot" interpretation table | `PIPELINE_OVERVIEW.md:709–711` | Interpretive labels resting on W-1, W-3 and W-4. All three are withdrawn. |
| **W-7** | "Fusion provides comprehensive detection" | `PIPELINE_OVERVIEW.md:767` | Fusion was never compared against its own components on a task with a ground truth that the components failed. Untested. |
| **W-8** | Cryptic pocket discovery framing | `PIPELINE_OVERVIEW.md:788–800` | Cites the pocket-detection literature next to our output without ever running a pocket comparison. Guilt by adjacency. |
| **W-9** | "early warning" of conformational change | slides / demo copy | Phase 6, p = 0.487 (§14). |

**W-3 has a code consequence, not only a copy consequence** — see OI-34: every
π-derived quantity (`pi.npy`, `component_rarity`, anything reading
`msm.stationary_distribution`) must be gated behind the sampling check or
emitted with an explicit `unresolved` flag, so a user cannot read a number the
report says is meaningless.

---

## 3. OPEN — claims that are neither supported nor withdrawn

These may be **investigated**, and may appear in the report as open questions.
They may not appear in product copy in any form until they move to §1.

| ID | Claim | What would settle it |
|---|---|---|
| **O-1** | The residue-level map corresponds to functional sites | Phase 4 with valid labels on ≥10 systems (OI-16) |
| **O-2** | The kinetic channel detects transitions that are kinetically improbable but geometrically unremarkable — the class `abs(diff)` is structurally blind to | **The data plan changed after §18.** The 10,000-frame ATLAS archives (OI-15) give 10x finer resolution over the *same* 100 ns; recurrence is a function of trajectory **length**, not sampling rate, so they will not settle this. The correct target is longer trajectories — mdCATH high-temperature replicates (OI-18), or the 4-8 us regime of Kozlowski & Grubmuller. Then run the Phase 8 recurrence probe **first**: if it passes, the recut benchmark becomes constructible and the hypothesis is finally testable. **All current negatives are conditional on <=100 ns and must always be stated that way.** |
| **O-3** | The suitability screen predicts downstream MSM quality | Requires a quality label we do not have; would need CK-test outcomes as a proxy |

---

## 4. What the product is, then

**One line:** *SciML-MD is an applicability-domain and diagnostic tool for
Markov state models of protein trajectories.*

The visualization product should be built around what a user can act on:

1. **The verdict, first.** Suitable / marginal / unsuitable, with the basin
   census that produced it. This is the screen doing its job and it is the
   feature nothing else offers.
2. **The flexibility profile.** Externally verified, so it can be shown without
   hedging — with r, slope, and error stated.
3. **Three channel views, labelled descriptive.** Dissociable, therefore worth
   three panels; not detections, therefore never ranked, never thresholded into
   a "hotspot list", never coloured red.
4. **Sampling adequacy shown, not hidden.** 100 ns is 13–27× below the published
   convergence threshold for proteins this size (§11.2). A user who sees that
   number will interpret everything above it correctly. A user who doesn't, won't.

**The word "anomaly" should be retired from user-facing copy.** It asserts the
withdrawn claim in a single word, and it is in the repository name. Suggested
replacement vocabulary: *conformational landscape*, *state occupancy*,
*transition irregularity* — descriptive nouns, no detection verb.

### The honest pitch

> Most MD analysis tools will happily build you a Markov state model of a
> trajectory that cannot support one, and hand you an equilibrium distribution
> estimated from a simulation 20× too short to have reached equilibrium.
> SciML-MD checks first, tells you when the answer would be meaningless, and
> shows you what the trajectory does support.

That is a smaller product than "hotspot detector". It is also one we can defend
line by line, which the other never was.

---

## 5. Enforcement

- Any PR touching `README.md`, `PIPELINE_OVERVIEW.md`, `QUICKSTART.md`, UI
  strings, or abstract text must be checked against §1 and §2.
- A new claim requires a row in §1 **with a report section citation** before the
  copy ships. No citation, no claim.
- Negatives are stated with their scope: "at ≤100 ns sampling", not "in general".
  O-2 is genuinely open and overclaiming a negative is the same failure as
  overclaiming a positive, pointed the other way.

*Related: D-24 (reported-but-ungated baselines, §16), OI-34 (gate π in code),
OI-16 (O-1), OI-15/OI-18 (O-2).*
