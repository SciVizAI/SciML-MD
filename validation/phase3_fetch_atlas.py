#!/usr/bin/env python3
"""
Fetch ATLAS trajectories and arrange them in the layout phase3_scale.py expects.

Standard library only - no extra dependencies. Run it on a machine with disk
space and network access.

    python validation/phase3_fetch_atlas.py --ids 1g2r_A 3p3o_A --out ~/atlas
    python validation/phase3_fetch_atlas.py --n 20 --select rmsf --out ~/atlas
    python validation/phase3_fetch_atlas.py --dump-list          # diagnostics only

ENDPOINTS (https://www.dsimb.inserm.fr/ATLAS/api/docs)
    /parsable                     zip of the release info/PDB list files
    /ATLAS/metadata/{pdb_chain}   small JSON  <- used here to validate an id
    /ATLAS/analysis/{pdb_chain}   1000 frames, protein only   <- default, small
    /ATLAS/protein/{pdb_chain}   10000 frames, protein only   <- full length
    /ATLAS/total/{pdb_chain}     10000 frames, full system (with solvent)

WHY THE OLD LIST CODE 404'd
    /parsable returns ONE zip covering THREE datasets - ATLAS, chameleon and
    DPF - as six files:
        2024_11_18_ATLAS_info.tsv      2024_11_18_ATLAS_pdb.txt
        2022_06_13_chameleon_info.tsv  2022_06_13_chameleon_pdb.txt
        2022_06_13_DPF_info.tsv        2022_06_13_DPF_pdb.txt
    The previous parser walked every .tsv/.txt in archive order and took the
    first column of every line, so the ids it handed back were chameleon/DPF
    systems. Those are real PDB chains but they do not exist under the
    /ATLAS/... namespace, hence 20 consecutive HTTP 404s while an explicitly
    passed ATLAS id (1g2r_A) downloaded fine.

    This version selects the ATLAS member files only, newest release first,
    parses the info TSV by HEADER NAME rather than by position, and validates
    every candidate id against /ATLAS/metadata/{id} before spending a download
    on it. Ids that fail validation are skipped and the next candidate is
    taken, so `--n 20` means twenty systems that actually arrived.

SCREENING
    The ATLAS info TSV carries per-system MD summary statistics (avg_RMSF,
    div_SE, div_MM, avg_gyration) plus length, redundancy flags and ligand /
    ion / nucleotide contact flags. `--select` ranks candidates on one of those
    columns before downloading, which is the cheap way to bias the sample
    towards systems with real conformational spread instead of the single-basin
    ensembles that made every system screen UNSUITABLE so far.

    The TSV is always saved to <out>/atlas_info.tsv - the functional-enrichment
    validation needs the contact_* columns from it.
"""
import argparse
import csv
import io
import json
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://www.dsimb.inserm.fr/ATLAS/api"
TIMEOUT = 600
ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}_[A-Za-z0-9]{1,4}$")

# columns of the ATLAS info TSV that we rank on; direction is "bigger is more
# interesting" unless --ascending is passed.
RANK_COLUMNS = {
    "rmsf": "avg_RMSF",
    "div_mm": "div_MM",
    "div_se": "div_SE",
    "gyration": "avg_gyration",
    "length": "length",
}


# --------------------------------------------------------------------------- io

def http_get(url, timeout=TIMEOUT, progress=False, label=""):
    """Download a URL. With progress=True, stream in chunks and report as it goes
    so a multi-hundred-MB archive does not look like a hung process."""
    req = urllib.request.Request(url, headers={"User-Agent": "SciML-MD/phase3"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if not progress:
            return r.read()
        total = int(r.headers.get("Content-Length") or 0)
        chunks, got, last = [], 0, 0.0
        t0 = time.time()
        while True:
            chunk = r.read(1 << 20)          # 1 MB
            if not chunk:
                break
            chunks.append(chunk)
            got += len(chunk)
            now = time.time()
            if now - last > 0.5:
                last = now
                mb = got / 1e6
                rate = mb / max(now - t0, 1e-6)
                if total:
                    pct = 100.0 * got / total
                    bar_n = int(pct / 4)
                    bar = "#" * bar_n + "." * (25 - bar_n)
                    sys.stdout.write(f"\r    {label} [{bar}] {pct:5.1f}%  "
                                     f"{mb:7.1f}/{total/1e6:.1f} MB  {rate:5.1f} MB/s")
                else:
                    sys.stdout.write(f"\r    {label} {mb:7.1f} MB  {rate:5.1f} MB/s")
                sys.stdout.flush()
        sys.stdout.write("\r" + " " * 78 + "\r")
        sys.stdout.flush()
        return b"".join(chunks)


def fetch_parsable():
    """Return the /parsable payload as a ZipFile. Raises on failure."""
    last = None
    for path in ("/parsable", "/ATLAS/parsable"):
        try:
            raw = http_get(BASE + path, timeout=180)
        except Exception as e:                       # noqa: BLE001
            last = e
            continue
        try:
            return zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile as e:
            last = e
    raise RuntimeError(f"could not retrieve {BASE}/parsable as a zip ({last})")


# ------------------------------------------------------------------ list parsing

def _atlas_members(namelist):
    """Split the archive member names into ATLAS-only .txt and .tsv lists,
    newest release first. chameleon/DPF members are excluded - they are the
    reason the old parser produced ids that 404."""
    def is_atlas(name):
        stem = Path(name).name.lower()
        return ("atlas" in stem
                and "chameleon" not in stem
                and "dpf" not in stem)

    atlas = [n for n in namelist if is_atlas(n)]
    txt = sorted((n for n in atlas if n.lower().endswith(".txt")), reverse=True)
    tsv = sorted((n for n in atlas if n.lower().endswith(".tsv")), reverse=True)
    return txt, tsv


def _read_tsv(zf, member):
    """Return (header, rows) from a TSV member, handling the quoted fields the
    ATLAS info file uses."""
    text = zf.read(member).decode("utf-8", "ignore")
    reader = csv.reader(io.StringIO(text), delimiter="\t", quotechar='"')
    rows = [r for r in reader if r]
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _pdb_column(header):
    """Index of the column holding the pdbid_chain key, by name."""
    for i, h in enumerate(header):
        if h.strip().lower() in ("pdb", "pdb_chain", "pdbid_chain", "id"):
            return i
    return 0


def _to_float(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes", "y")


def load_atlas_table(zf):
    """Return (records, header, source_member).

    records is a list of dicts keyed by the info-TSV header, each guaranteed to
    carry a valid 'PDB' id. Falls back to the plain PDB list if no info TSV is
    present (in which case every record has only the id)."""
    txt_members, tsv_members = _atlas_members(zf.namelist())

    for member in tsv_members:
        header, rows = _read_tsv(zf, member)
        if not header:
            continue
        pcol = _pdb_column(header)
        recs = []
        for r in rows:
            if pcol >= len(r):
                continue
            pid = r[pcol].strip()
            if not ID_RE.match(pid):
                continue
            recs.append({header[i].strip(): (r[i] if i < len(r) else "")
                         for i in range(len(header))} | {"PDB": pid})
        if recs:
            return recs, header, member

    for member in txt_members:
        ids = []
        for line in zf.read(member).decode("utf-8", "ignore").splitlines():
            tok = line.split("\t")[0].strip()
            if ID_RE.match(tok):
                ids.append(tok)
        if ids:
            return [{"PDB": i} for i in ids], ["PDB"], member

    raise RuntimeError(
        "no ATLAS member found inside /parsable; archive contained: "
        + ", ".join(sorted(zf.namelist())[:12]))


# --------------------------------------------------------------------- selection

def apply_filters(recs, args):
    out, dropped = [], {}

    def drop(reason):
        dropped[reason] = dropped.get(reason, 0) + 1

    for r in recs:
        n = _to_float(r.get("length"))
        if args.min_len is not None and n is not None and n < args.min_len:
            drop(f"length < {args.min_len}"); continue
        if args.max_len is not None and n is not None and n > args.max_len:
            drop(f"length > {args.max_len}"); continue
        if args.non_redundant and "non_redundant_protein" in r \
                and not _truthy(r["non_redundant_protein"]):
            drop("redundant"); continue
        f = _to_float(r.get("avg_RMSF"))
        if args.min_rmsf is not None and f is not None and f < args.min_rmsf:
            drop(f"avg_RMSF < {args.min_rmsf}"); continue
        if args.max_rmsf is not None and f is not None and f > args.max_rmsf:
            drop(f"avg_RMSF > {args.max_rmsf}"); continue
        if args.with_ligand and not any(
                _truthy(r.get(c, "")) for c in
                ("contact_ligand", "contact_ion", "contact_nucleotide")):
            drop("no ligand/ion/nucleotide contact"); continue
        out.append(r)
    return out, dropped


def stratify(recs, column, n_strata):
    """Spread the selection evenly across the range of `column` instead of
    taking one end of it.

    Taking the top of avg_RMSF pulls the disordered tail of ATLAS (up to 14.4 A
    against a database median of 1.33 A), which is no more representative than
    the single-basin systems it was meant to escape. A method validated only on
    the extreme does not generalise in either direction. This orders candidates
    so the first n_strata picks are spaced across the whole distribution, with
    the remainder appended as fallback if any of them fail to download."""
    keyed = [(r, _to_float(r.get(column))) for r in recs]
    have = sorted((k for k in keyed if k[1] is not None), key=lambda kv: kv[1])
    if not have:
        return recs
    n = max(1, min(n_strata, len(have)))
    picks, used = [], set()
    for j in range(n):
        idx = int(round(j * (len(have) - 1) / max(n - 1, 1)))
        while idx in used and idx < len(have) - 1:
            idx += 1
        used.add(idx)
        picks.append(have[idx])
    rest = [have[i] for i in range(len(have)) if i not in used]
    lo, hi = picks[0][1], picks[-1][1]
    print(f"  stratified on {column}: {n} strata spanning {lo:.2f} .. {hi:.2f} "
          f"(pool median {have[len(have)//2][1]:.2f}, n={len(have)})")
    return [r for r, _ in picks] + [r for r, _ in rest] + \
           [r for r, v in keyed if v is None]


def rank(recs, args):
    if args.select == "stratified":
        return stratify(recs, RANK_COLUMNS["rmsf"], args.n or len(recs))
    if args.select == "first":
        return recs
    if args.select == "random":
        rng = random.Random(args.seed)
        shuffled = list(recs)
        rng.shuffle(shuffled)
        return shuffled

    col = RANK_COLUMNS[args.select]
    keyed = [(r, _to_float(r.get(col))) for r in recs]
    have = [k for k in keyed if k[1] is not None]
    if not have:
        print(f"  ! column '{col}' is absent or non-numeric in this release; "
              f"falling back to list order")
        return recs
    have.sort(key=lambda kv: kv[1], reverse=not args.ascending)
    lo, hi = have[-1][1], have[0][1]
    print(f"  ranking on {col} ({'ascending' if args.ascending else 'descending'}); "
          f"selected range {min(lo, hi):.3g} .. {max(lo, hi):.3g} "
          f"over {len(have)} systems")
    return [r for r, _ in have] + [r for r, v in keyed if v is None]


# ------------------------------------------------------------------- downloading

def validate_id(pdb_chain):
    """Cheap existence check against the metadata endpoint. Returns
    (ok, detail)."""
    try:
        raw = http_get(f"{BASE}/ATLAS/metadata/{pdb_chain}", timeout=60)
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:                            # noqa: BLE001
        return False, str(e)[:60]
    try:
        json.loads(raw.decode("utf-8", "ignore"))
    except Exception:                                 # noqa: BLE001
        pass                                          # non-JSON but it exists
    return True, "ok"


def fetch_one(pdb_chain, kind, out_root, force=False):
    dest = out_root / pdb_chain
    if dest.exists() and not force:
        pdbs = list(dest.glob("*.pdb"))
        xtcs = list(dest.glob("*.xtc")) + list(dest.glob("*.dcd"))
        if pdbs and xtcs:
            return "cached", sum(f.stat().st_size for f in dest.rglob("*")) / 1e6

    url = f"{BASE}/ATLAS/{kind}/{pdb_chain}"
    raw = http_get(url, progress=True, label=pdb_chain)
    if len(raw) < 1000:
        raise RuntimeError(f"suspiciously small response ({len(raw)} bytes)")

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        zf.extractall(dest)

    # flatten any nested directory so the files sit directly in <dest>/
    for sub in [p for p in dest.iterdir() if p.is_dir()]:
        for f in sub.rglob("*"):
            if f.is_file():
                target = dest / f.name
                if not target.exists():
                    shutil.move(str(f), str(target))
        shutil.rmtree(sub, ignore_errors=True)

    pdbs = list(dest.glob("*.pdb"))
    xtcs = list(dest.glob("*.xtc")) + list(dest.glob("*.dcd"))
    if not pdbs or not xtcs:
        raise RuntimeError(f"archive lacked a .pdb/.xtc pair; got "
                           f"{[f.name for f in dest.iterdir()][:8]}")
    return "downloaded", sum(f.stat().st_size for f in dest.rglob("*")) / 1e6


# -------------------------------------------------------------------------- main

def dump_list():
    """Print what /parsable actually contains. Use this whenever ids start
    failing again - it shows the member names and the info-TSV header."""
    zf = fetch_parsable()
    print("members of /parsable:")
    for n in sorted(zf.namelist()):
        print(f"  {n:44s} {zf.getinfo(n).file_size/1e3:8.1f} kB")
    txt, tsv = _atlas_members(zf.namelist())
    print(f"\nATLAS .txt members (newest first): {txt}")
    print(f"ATLAS .tsv members (newest first): {tsv}")
    recs, header, member = load_atlas_table(zf)
    print(f"\nusing {member}: {len(recs)} systems")
    print(f"columns ({len(header)}): {', '.join(header)}")
    print("\nfirst 5 ids:", ", ".join(r["PDB"] for r in recs[:5]))
    for key, col in RANK_COLUMNS.items():
        vals = [v for v in (_to_float(r.get(col)) for r in recs) if v is not None]
        if vals:
            vals.sort()
            print(f"  {key:9s} ({col:13s}) n={len(vals):5d}  "
                  f"min={vals[0]:8.3f}  median={vals[len(vals)//2]:8.3f}  "
                  f"max={vals[-1]:8.3f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ids", nargs="+", help="Explicit systems, e.g. 1g2r_A 3p3o_A")
    g.add_argument("--n", type=int,
                   help="Download this many systems that actually arrive "
                        "(candidates that fail validation are skipped, not counted)")
    g.add_argument("--dump-list", action="store_true",
                   help="Print what /parsable contains and exit. Diagnostics.")
    ap.add_argument("--out", default="~/atlas", help="Destination root directory")
    ap.add_argument("--kind", choices=["analysis", "protein", "total"],
                    default="analysis",
                    help="analysis=1000 frames (default, small); protein=10000 frames")
    ap.add_argument("--select", default="stratified",
                    choices=["stratified", "first", "random", *RANK_COLUMNS],
                    help="How to order candidates before downloading. Default "
                         "'stratified' spreads the sample evenly across the ATLAS "
                         "avg_RMSF distribution. 'rmsf' takes the most flexible "
                         "first, which lands in the disordered tail - useful for "
                         "probing, wrong for a representative validation.")
    ap.add_argument("--min-rmsf", type=float, default=None,
                    help="Minimum ATLAS avg_RMSF in Angstrom (database median 1.33)")
    ap.add_argument("--max-rmsf", type=float, default=None,
                    help="Maximum ATLAS avg_RMSF in Angstrom (database max 14.41; "
                         "above ~4 A the systems are effectively disordered)")
    ap.add_argument("--ascending", action="store_true",
                    help="Invert --select ordering")
    ap.add_argument("--min-len", type=int, default=None, help="Minimum residue count")
    ap.add_argument("--max-len", type=int, default=250,
                    help="Maximum residue count (default 250; keeps the MSM cheap)")
    ap.add_argument("--non-redundant", action="store_true",
                    help="Keep only systems flagged non_redundant_protein")
    ap.add_argument("--with-ligand", action="store_true",
                    help="Keep only systems with a ligand/ion/nucleotide contact "
                         "(needed for the functional-enrichment validation)")
    ap.add_argument("--seed", type=int, default=0, help="Seed for --select random")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="Cap candidates tried with --n (0 = no cap)")
    ap.add_argument("--force", action="store_true", help="Re-download cached systems")
    args = ap.parse_args()

    if args.dump_list:
        return dump_list()

    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    validate = True
    if args.ids:
        candidates = list(args.ids)
        target = len(candidates)
        validate = False                    # explicit ids: let the download speak
    else:
        print("querying ATLAS for the system list ...")
        zf = fetch_parsable()
        recs, header, member = load_atlas_table(zf)
        print(f"  {member}: {len(recs)} ATLAS systems")

        info_path = out_root / "atlas_info.tsv"
        if member.lower().endswith(".tsv"):
            info_path.write_bytes(zf.read(member))
            print(f"  saved release table -> {info_path}")

        kept, dropped = apply_filters(recs, args)
        if dropped:
            print("  filters: " + "; ".join(f"{v} dropped for {k}"
                                            for k, v in dropped.items()))
        print(f"  {len(kept)} candidate(s) after filters")
        if not kept:
            print("no candidates survive the filters - relax --max-len / --with-ligand")
            return 1
        candidates = [r["PDB"] for r in rank(kept, args)]
        target = args.n

    if args.max_attempts:
        candidates = candidates[:args.max_attempts]

    print(f"\ndestination: {out_root}")
    print(f"endpoint   : {BASE}/ATLAS/{args.kind}/<id>")
    print(f"target     : {target} system(s)")
    print("note: archives are large; each system may take several minutes.\n")

    ok, failed, skipped, total_mb = [], {}, {}, 0.0
    for pid in candidates:
        if len(ok) >= target:
            break
        if validate:
            good, detail = validate_id(pid)
            if not good:
                skipped[pid] = detail
                continue
        try:
            status, mb = fetch_one(pid, args.kind, out_root, args.force)
            total_mb += mb
            ok.append(pid)
            print(f"[{len(ok)}/{target}] {pid:12s} {status:10s} {mb:8.1f} MB",
                  flush=True)
        except urllib.error.HTTPError as e:
            failed[pid] = f"HTTP {e.code}"
            print(f"[--/{target}] {pid:12s} FAILED  HTTP {e.code}", flush=True)
        except Exception as e:                        # noqa: BLE001
            failed[pid] = str(e)[:120]
            print(f"[--/{target}] {pid:12s} FAILED  {str(e)[:80]}", flush=True)

    print(f"\n{len(ok)} system(s) ready in {out_root}  ({total_mb:.0f} MB total)")
    if skipped:
        print(f"{len(skipped)} candidate(s) skipped at validation "
              f"(first few: {json.dumps(dict(list(skipped.items())[:5]))})")
    if failed:
        print(f"{len(failed)} failed: {json.dumps(failed, indent=2)[:800]}")
    if len(ok) < target:
        print(f"! only {len(ok)}/{target} arrived. Run --dump-list to inspect "
              f"the release table, or widen --max-len.")
    if ok:
        manifest = out_root / "fetch_manifest.json"
        manifest.write_text(json.dumps(
            {"kind": args.kind, "select": args.select, "ascending": args.ascending,
             "filters": {"min_len": args.min_len, "max_len": args.max_len,
                         "non_redundant": args.non_redundant,
                         "with_ligand": args.with_ligand},
             "downloaded": ok, "skipped": skipped, "failed": failed},
            indent=2))
        print(f"manifest -> {manifest}")
        print("\nNext:")
        print(f"  python validation/phase3_scale.py --data_root {args.out} "
              f"--screen-only")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
