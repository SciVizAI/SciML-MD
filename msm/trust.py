"""The trust contract - what a consumer of this pipeline is allowed to display.

WHY THIS MODULE EXISTS
----------------------
`CLAIMS.md` states which outputs of this pipeline mean something and which do
not. Until now that lived only in a Markdown file, so the pipeline happily
emitted `component_rarity` and `pi.npy` for trajectories where the validation
report says those numbers are meaningless. A register the code does not enforce
is a document, not a control (report section 17.1 made the same point about the
README).

This module turns the register into a machine-readable artifact. Every scored
system gets a `trust.json` next to its scores, declaring per channel whether it
is `ok`, `descriptive_only` or `withheld`, and why. The visualiser reads this
file and decides what to render; it should never need to know the validation
history, only the contract.

THE THREE STATUSES
  ok               display freely; the channel is supported (CLAIMS.md section 1)
  descriptive_only display, but never as a detection - no ranking into a
                   "hotspot list", no thresholding, no red. It describes the
                   trajectory; it does not identify anything (W-1, W-2)
  withheld         do not display. The diagnostics say this number cannot carry
                   meaning for THIS trajectory. Values are emitted as NaN

Nothing here is a judgement about the pipeline's quality. It is a judgement
about whether a given trajectory can support a given quantity, made per system,
from measurements taken before the model was fitted.
"""
from __future__ import annotations

CLAIMS_VERSION = "2026-08-31"

OK = "ok"
DESCRIPTIVE = "descriptive_only"
WITHHELD = "withheld"

# Channels this pipeline emits, and the strongest status each can EVER reach.
# A channel can be demoted per-trajectory by the diagnostics; it can never be
# promoted above its ceiling, because the ceiling is set by what validation
# supports in general rather than by this trajectory.
CEILING = {
    "local_density": DESCRIPTIVE,
    "transition_surprise": DESCRIPTIVE,
    "rarity": DESCRIPTIVE,
    "score_dynamic": DESCRIPTIVE,
    # Residue-level outputs. Added 2026-09-20: they were reaching consumers with
    # no machine-readable rule at all, covered only by prose in INTEGRATION.md.
    # residue_scores is DERIVED from the fused frame score, so it inherits every
    # gate applied upstream and can never be safer than its source.
    "residue_scores": DESCRIPTIVE,
    # The one exception in the whole pipeline. RMSF is externally verified
    # against published ATLAS values (r = 0.892, slope 0.966) - CLAIMS.md S-2 -
    # so it is the only channel that may be displayed without hedging.
    "residue_rmsf": OK,
}

CLAIM_REFS = {
    "local_density": "S-3 (dissociable channel); W-2 (not a detection)",
    "transition_surprise": "S-3, S-5; W-2, W-4 (carries no timing information)",
    "rarity": "W-3 (pi unresolved at <=100 ns); S-3",
    "score_dynamic": "W-2 (fused score is descriptive, not a detection)",
    "residue_scores": "W-1, W-5 (no validated correspondence to functional sites; O-1 open)",
    "residue_rmsf": "S-2 (externally verified: r = 0.892, slope 0.966, median |err| 10.6%, max 40.5%)",
}


def build_contract(system_id, suitability=None, recurrence=None,
                   estimability=None, n_frames=None):
    """Assemble the per-system trust contract.

    All three diagnostics are optional; a missing one is treated as UNKNOWN and
    is never read as a pass. Failing open would defeat the purpose.
    """
    diagnostics = {"basin_census": suitability, "recurrence": recurrence,
                   "estimability": estimability}

    suit_v = (suitability or {}).get("verdict", "unknown")
    rec_v = (recurrence or {}).get("verdict", "unknown")
    est_v = (estimability or {}).get("verdict", "unknown")

    channels, notes = {}, []

    # --- pi-derived quantities: the gate OI-34 exists for -------------------
    if est_v == "estimable" and rec_v in ("recurrent", "weak"):
        rarity_status, rarity_why = CEILING["rarity"], (
            "state occupancies rest on repeated visits in this trajectory")
    else:
        rarity_status = WITHHELD
        rarity_why = ((estimability or {}).get("reason")
                      or (recurrence or {}).get("reason")
                      or "estimability of pi was not established for this trajectory")
        notes.append("rarity withheld: " + rarity_why)
    channels["rarity"] = _chan(rarity_status, rarity_why, "rarity")

    # --- transition surprise: needs states that are revisited ---------------
    if est_v == "unresolved" or rec_v == "non_recurrent":
        s_status = WITHHELD
        s_why = ("transition probabilities are fitted to events that never "
                 "repeat in this trajectory, so 'surprise' cannot separate a "
                 "rare transition from an unseen one")
        notes.append("transition_surprise withheld: " + s_why)
    else:
        s_status = CEILING["transition_surprise"]
        s_why = ("descriptive only: dissociable from the static channels (S-3), "
                 "but never validated as a detector or as an early warning")
    channels["transition_surprise"] = _chan(s_status, s_why, "transition_surprise")

    # --- local density: geometric, needs no kinetic model -------------------
    channels["local_density"] = _chan(
        CEILING["local_density"],
        "geometric channel; independent of the MSM and unaffected by the "
        "recurrence and estimability findings",
        "local_density")

    # --- the fused score is only as good as what went into it ---------------
    live = [c for c, v in channels.items() if v["status"] != WITHHELD]  # frame channels only
    if not live:
        f_status, f_why = WITHHELD, "every input channel is withheld"
    else:
        f_status = CEILING["score_dynamic"]
        f_why = (f"fused from {len(live)} of 3 channels ({', '.join(sorted(live))}); "
                 "descriptive only - never rank, threshold or colour this as a "
                 "detection")
    channels["score_dynamic"] = _chan(f_status, f_why, "score_dynamic")
    channels["score_dynamic"]["channels_used"] = sorted(live)

    # --- residue level ------------------------------------------------------
    # Derived from the fused frame score: whatever gated that, gates this.
    channels["residue_scores"] = _chan(
        f_status if f_status == WITHHELD else CEILING["residue_scores"],
        ("derived from score_dynamic, so it inherits its gating. Never rank "
         "into a 'top hotspot' list: the correspondence between these scores "
         "and functional sites is an OPEN question (CLAIMS.md O-1), not a "
         "result"),
        "residue_scores")

    # RMSF is measured from the coordinates directly - no model, no gate.
    channels["residue_rmsf"] = _chan(
        CEILING["residue_rmsf"],
        ("externally verified against published ATLAS values; the only quantity "
         "here with an external check, and the only one displayable without a "
         "caveat. Quote r, slope and median error when reporting it"),
        "residue_rmsf")

    displayable = suit_v != "unsuitable" and f_status != WITHHELD

    return {
        "schema": "sciml-md/trust@1",
        "claims_version": CLAIMS_VERSION,
        "system": system_id,
        "n_frames": n_frames,
        "verdict": suit_v,
        "recurrence_verdict": rec_v,
        "estimability_verdict": est_v,
        "displayable": bool(displayable),
        "headline": _headline(suit_v, rec_v, est_v),
        "channels": channels,
        "diagnostics": diagnostics,
        "notes": notes,
        "display_rules": [
            "Show the verdict and the two diagnostics BEFORE any score.",
            "Never render a withheld channel. Its values are NaN by design.",
            "A descriptive_only channel may be plotted as a track over time, "
            "but must not be ranked into a top-N list, thresholded into "
            "'hotspots', or coloured on a severity scale.",
            "Always show sampling context: total ns and the frame stride. A "
            "recurrence percentile without its stride is uninterpretable.",
            "The word 'anomaly' must not appear in the interface (CLAIMS.md 4). "
            "Replacement vocabulary: conformational landscape, state occupancy, "
            "transition irregularity.",
            "residue_rmsf is the ONLY channel displayable without a caveat. "
            "Everything else is descriptive at best.",
        ],
    }


def _chan(status, why, name):
    return {"status": status, "reason": why, "claims": CLAIM_REFS.get(name, "")}


def _headline(suit, rec, est):
    """Say WHICH condition failed. "does not revisit conformations" and "is too
    short for populations to converge" are different diagnoses with different
    remedies, and a trajectory can recur strongly and still be far too short."""
    if suit == "unsuitable":
        return ("This trajectory cannot support a Markov state model. "
                "Scores are not shown.")
    if rec == "non_recurrent":
        return ("This trajectory never returns to a conformation it has already "
                "visited, so state populations and transition statistics are "
                "unresolved. Geometric channels only.")
    if est == "unresolved":
        return ("This trajectory is too short for state populations to have "
                "converged, so kinetic quantities are unresolved. Geometric "
                "channels only.")
    if suit == "marginal" or rec == "weak":
        return ("This trajectory marginally supports a Markov state model. "
                "Read every kinetic quantity with that caveat.")
    return ("This trajectory supports a Markov state model. All channels are "
            "descriptive, not detections.")


def gate_components(components, contract):
    """Blank out withheld channels. NaN, not deletion, so column layout is
    stable for consumers and the gap is visible rather than silent."""
    import numpy as np

    out, gated = {}, []
    for name, values in components.items():
        st = contract["channels"].get(name, {}).get("status")
        if st == WITHHELD:
            out[name] = np.full(len(np.asarray(values)), np.nan)
            gated.append(name)
        else:
            out[name] = values
    return out, gated
