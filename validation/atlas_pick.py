#!/usr/bin/env python3
"""
Choose the Phase 3 sample from the ATLAS release table. No downloads.

WHY
---
The first sample was selected on avg_RMSF descending. That is the wrong
variable. Two facts from the trajectory diagnosis:

  1. The coordinates are sound. Our independently recomputed RMSF matches the
     RMSF ATLAS publishes for the same systems to within ~10% across 21
     systems (1g2r_A 2.38 vs 2.42; 1j5u_A 6.04 vs 6.03; 2b1y_A 7.24 vs 7.31).
     So the 15-44 A spreads are real motion, not periodic-boundary breakage.

  2. They are real because the sample IS the disordered tail. 20 of 21 systems
     sit at ATLAS avg_RMSF 6-14 A against a database median of 1.33 A.

An MSM needs DISCRETE METASTABLE STATES. A single basin has none - that was the
old failure. A disordered chain has none either, for the opposite reason: it is
one broad diffusive blob with no barriers. High RMSF selects for the second
failure mode. What we actually want is multi-basin: moderate amplitude, high
structural DIVERSITY.

ATLAS ships div_SE and div_MM, which are diversity-like measures, but their
direction is not documented in the API or on the download page. This script
settles it empirically from the table itself: whichever sign the correlation
with avg_RMSF carries tells us whether high means diverse or means conserved.
Then it selects on that basis and prints the exact fetch command.

USAGE
    python validation/atlas_pick.py --info ~/atlas/atlas_info.tsv
    python validation/atlas_pick.py --info ~/atlas/atlas_info.tsv --have ~/atlas
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

NUMERIC = ["length", "avg_RMSF", "div_SE", "div_MM", "avg_gyration",
           "PDB_resolution", "alpha%", "beta%", "coil%"]


def load(path: Path):
    with path.open(newline="", encoding="utf-8", errors="ignore") as fh:
        rows = list(csv.reader(fh, delimiter="\t", quotechar='"'))
    header = [h.strip() for h in rows[0]]
    recs = []
    for r in rows[1:]:
        if not r:
            continue
        rec = {header[i]: (r[i] if i < len(r) else "") for i in range(len(header))}
        for c in NUMERIC:
            if c in rec:
                try:
                    rec[c] = float(str(rec[c]).strip())
                except (TypeError, ValueError):
                    rec[c] = np.nan
        recs.append(rec)
    return recs, header


def col(recs, name):
    return np.array([r.get(name, np.nan) for r in recs], dtype=float)


def truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes", "y")


def describe(recs):
    print(f"{len(recs)} systems\n")
    print(f"{'column':14s} {'min':>8s} {'q25':>8s} {'median':>8s} {'q75':>8s} "
          f"{'q90':>8s} {'max':>8s}")
    print("-" * 66)
    for c in ["length", "avg_RMSF", "div_SE", "div_MM", "avg_gyration"]:
        v = col(recs, c)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        q = np.percentile(v, [0, 25, 50, 75, 90, 100])
        print(f"{c:14s} " + " ".join(f"{x:8.2f}" for x in q))


def direction(recs):
    """Decide whether high div_* means diverse or conserved, from the sign of
    its rank correlation with avg_RMSF. A genuine diversity measure must rise
    with fluctuation amplitude; a conservation/similarity measure must fall."""
    rmsf = col(recs, "avg_RMSF")
    out = {}
    print("\nrank correlation with avg_RMSF (Spearman):")
    for c in ["div_SE", "div_MM", "avg_gyration", "length"]:
        v = col(recs, c)
        m = np.isfinite(v) & np.isfinite(rmsf)
        if m.sum() < 50:
            continue
        rho, p = spearmanr(v[m], rmsf[m])
        out[c] = float(rho)
        tag = ""
        if c.startswith("div"):
            tag = ("  -> HIGH = more diverse" if rho > 0.15 else
                   "  -> HIGH = more CONSERVED (similarity-like)" if rho < -0.15
                   else "  -> weakly related to amplitude (independent axis)")
        print(f"  {c:14s} rho={rho:+.3f}  p={p:.1e}  n={m.sum()}{tag}")
    return out


def pick(recs, args, rho):
    """Score systems for MSM suitability: enough motion to have more than one
    basin, not so much that there are no basins at all, and as much structural
    diversity as the release table can report at that amplitude."""
    rmsf = col(recs, "avg_RMSF")
    finite = np.isfinite(rmsf)

    keep = np.ones(len(recs), dtype=bool) & finite
    keep &= (rmsf >= args.min_rmsf) & (rmsf <= args.max_rmsf)
    ln = col(recs, "length")
    keep &= np.isfinite(ln) & (ln >= args.min_len) & (ln <= args.max_len)
    # A system with no secondary structure has no fold, and therefore no
    # basins to be metastable in. ATLAS reports coil% directly.
    coil = col(recs, "coil%")
    if args.max_coil is not None:
        keep &= ~(np.isfinite(coil) & (coil > args.max_coil))
    if args.non_redundant:
        keep &= np.array([truthy(r.get("non_redundant_protein", "")) for r in recs])
    if args.with_ligand:
        keep &= np.array([any(truthy(r.get(c, "")) for c in
                              ("contact_ligand", "contact_ion", "contact_nucleotide"))
                          for r in recs])
    if args.exclude:
        keep &= np.array([r.get("PDB", "") not in args.exclude for r in recs])

    idx = np.flatnonzero(keep)
    print(f"\n{len(idx)} candidate(s) in the band "
          f"avg_RMSF {args.min_rmsf}-{args.max_rmsf} A, "
          f"length {args.min_len}-{args.max_len}, coil<={args.max_coil}%"
          + (", non-redundant" if args.non_redundant else "")
          + (", with ligand/ion/nucleotide" if args.with_ligand else ""))
    if not len(idx):
        return []

    # diversity axis, oriented so that larger is always more diverse
    dcol = args.diversity
    d = col(recs, dcol)[idx]
    sign = 1.0 if rho.get(dcol, 0.0) >= 0 else -1.0
    if sign < 0:
        print(f"  ({dcol} correlates negatively with amplitude, so it is being "
              f"read as a similarity measure and inverted)")

    r = rmsf[idx]

    # ---- orthogonalise diversity against amplitude ------------------------
    # div_MM is ~0.73 rank-correlated with avg_RMSF, so selecting on raw
    # diversity mostly re-selects amplitude - the same collinearity trap that
    # made residue attribution track RMSF before the E-1 fix, and the same
    # remedy: regress in rank space and keep the residual.
    #
    # The residual is what we actually want. A system whose structural
    # diversity EXCEEDS what its fluctuation amplitude predicts is one whose
    # motion is distributed over discrete alternative structures rather than
    # spread diffusely - which is the definition of multi-basin, and the only
    # regime where an MSM has states to find.
    from scipy.stats import rankdata
    ok = np.isfinite(d) & np.isfinite(r)
    score = np.full(len(idx), -np.inf)
    if ok.sum() >= 10:
        rr = rankdata(r[ok])
        dd = rankdata(sign * d[ok])
        slope, intercept = np.polyfit(rr, dd, 1)
        score[ok] = dd - (slope * rr + intercept)
        resid_rho, _ = spearmanr(score[ok], r[ok])
        print(f"  orthogonalised {dcol} against avg_RMSF in rank space; "
              f"residual vs amplitude rho={resid_rho:+.3f} "
              f"(was {rho.get(dcol, float('nan')):+.3f})")
    else:
        score[ok] = sign * d[ok]
        print("  ! too few finite values to orthogonalise; using raw diversity")

    if args.score == "raw":
        score = np.where(np.isfinite(d), sign * d, -np.inf)
        print("  (--score raw: selecting on diversity WITHOUT removing amplitude)")

    # stratify across amplitude, then within each stratum take the system with
    # the most excess diversity. Stratifying keeps the sample spread over the
    # flexibility range instead of clustering at one end.
    order = np.argsort(r)
    n = min(args.n, len(idx))
    chosen = []
    for b in np.array_split(order, n):
        if len(b):
            chosen.append((idx[b[int(np.argmax(score[b]))]],
                           float(np.max(score[b]))))

    print(f"\n{'PDB':9s} {'len':>4s} {'avg_RMSF':>9s} {dcol:>8s} {'excess':>7s} "
          f"{'coil%':>6s} {'organism':<20s} name")
    print("-" * 108)
    for i, sc in chosen:
        r_ = recs[i]
        print(f"{r_['PDB']:9s} {r_['length']:4.0f} {r_['avg_RMSF']:9.2f} "
              f"{r_.get(dcol, float('nan')):8.3f} {sc:7.1f} "
              f"{r_.get('coil%', float('nan')):6.0f} "
              f"{str(r_.get('organism', ''))[:20]:<20s} "
              f"{str(r_.get('protein_name', ''))[:38]}")
    return [recs[i]["PDB"] for i, _ in chosen]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--info", default="~/atlas/atlas_info.tsv")
    ap.add_argument("--have", default=None,
                    help="Data root of systems already downloaded; they are "
                         "excluded from the picks and reported separately")
    ap.add_argument("--n", type=int, default=24, help="How many systems to pick")
    ap.add_argument("--min-rmsf", type=float, default=0.9)
    ap.add_argument("--max-rmsf", type=float, default=3.5)
    ap.add_argument("--min-len", type=int, default=60)
    ap.add_argument("--max-len", type=int, default=250)
    ap.add_argument("--diversity", default="div_MM", choices=["div_MM", "div_SE"])
    ap.add_argument("--score", default="residual", choices=["residual", "raw"],
                    help="residual (default) selects excess diversity after "
                         "regressing out avg_RMSF; raw selects diversity itself, "
                         "which is ~73%% collinear with amplitude")
    ap.add_argument("--max-coil", type=float, default=55.0,
                    help="Reject systems above this coil%% (default 55)")
    ap.add_argument("--non-redundant", action="store_true", default=True)
    ap.add_argument("--with-ligand", action="store_true")
    ap.add_argument("--out", default="validation/atlas_selection.json")
    args = ap.parse_args()

    path = Path(args.info).expanduser()
    if not path.exists():
        print(f"Not found: {path}\nRun phase3_fetch_atlas.py once - it saves the "
              f"release table next to the downloads.")
        return 1

    recs, _ = load(path)
    describe(recs)
    rho = direction(recs)

    args.exclude = set()
    if args.have:
        root = Path(args.have).expanduser()
        if root.is_dir():
            args.exclude = {p.name for p in root.iterdir() if p.is_dir()}
            have = [r for r in recs if r.get("PDB") in args.exclude]
            if have:
                v = np.array([r["avg_RMSF"] for r in have if np.isfinite(r["avg_RMSF"])])
                allv = col(recs, "avg_RMSF")
                allv = allv[np.isfinite(allv)]
                pct = [float((allv < x).mean() * 100) for x in v]
                print(f"\nalready downloaded: {len(have)} system(s), "
                      f"avg_RMSF {v.min():.2f}-{v.max():.2f} A "
                      f"= percentile {min(pct):.0f}-{max(pct):.0f} of ATLAS")
                print("  these remain a valid HIGH-flexibility stratum; the picks "
                      "below fill in the rest of the range")

    picks = pick(recs, args, rho)
    if picks:
        Path(args.out).write_text(json.dumps(
            {"picks": picks, "band": [args.min_rmsf, args.max_rmsf],
             "length": [args.min_len, args.max_len],
             "diversity_column": args.diversity,
             "spearman_vs_rmsf": rho}, indent=2))
        print(f"\nsaved -> {args.out}")
        print("\nFetch them:")
        print("  python validation/phase3_fetch_atlas.py --out ~/atlas --ids \\")
        for i in range(0, len(picks), 8):
            cont = " \\" if i + 8 < len(picks) else ""
            print("      " + " ".join(picks[i:i + 8]) + cont)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
