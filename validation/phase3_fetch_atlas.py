#!/usr/bin/env python3
"""
Fetch ATLAS trajectories and arrange them in the layout phase3_scale.py expects.

Standard library only - no extra dependencies. Run it on a machine with disk
space and network access:

    python validation/phase3_fetch_atlas.py --ids 1g2r_A 3p3o_A 1tag_A --out ~/atlas
    python validation/phase3_fetch_atlas.py --n 10 --out ~/atlas
    python validation/phase3_fetch_atlas.py --n 50 --kind protein --out ~/atlas

ENDPOINTS (from https://www.dsimb.inserm.fr/ATLAS/api/docs)
    /ATLAS/analysis/{pdb_chain}   1000 frames, protein only   <- default, small
    /ATLAS/protein/{pdb_chain}   10000 frames, protein only   <- full length
    /ATLAS/total/{pdb_chain}     10000 frames, full system (with solvent)

Use `analysis` for a pilot: 1000 frames x 3 replicates per protein is already
12x the sampling of the villin run Phase 2 used, at a fraction of the download.
Move to `protein` for the full sweep.

Each system unpacks to <out>/<pdb_chain>/ containing a .pdb topology and one or
more .xtc trajectories, which is exactly what phase3_scale.py discovers.
"""
import argparse
import io
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://www.dsimb.inserm.fr/ATLAS/api"
TIMEOUT = 600


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


def list_systems(limit):
    """Try the /parsable endpoint for the official PDB list."""
    for path in ("/parsable", "/ATLAS/parsable"):
        try:
            raw = http_get(BASE + path, timeout=120)
        except Exception:
            continue
        # the payload is a zip of TSV/TXT in at least one release
        ids = []
        try:
            zf = zipfile.ZipFile(io.BytesIO(raw))
            for name in zf.namelist():
                if not name.lower().endswith((".txt", ".tsv")):
                    continue
                for line in zf.read(name).decode("utf-8", "ignore").splitlines():
                    tok = line.split("\t")[0].strip()
                    if len(tok) >= 6 and "_" in tok and not tok.lower().startswith("pdb"):
                        ids.append(tok)
        except zipfile.BadZipFile:
            for line in raw.decode("utf-8", "ignore").splitlines():
                tok = line.split("\t")[0].strip()
                if len(tok) >= 6 and "_" in tok:
                    ids.append(tok)
        seen, uniq = set(), []
        for i in ids:
            if i not in seen:
                seen.add(i); uniq.append(i)
        if uniq:
            return uniq[:limit]
    return []


def fetch_one(pdb_chain, kind, out_root, force=False):
    dest = out_root / pdb_chain
    if dest.exists() and not force:
        pdbs = list(dest.glob("*.pdb"))
        xtcs = list(dest.glob("*.xtc")) + list(dest.glob("*.dcd"))
        if pdbs and xtcs:
            return "cached", sum(f.stat().st_size for f in dest.rglob("*")) / 1e6

    url = f"{BASE}/ATLAS/{kind}/{pdb_chain}"
    print(f"    downloading {pdb_chain} ({kind}) ...", flush=True)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ids", nargs="+", help="Explicit systems, e.g. 1g2r_A 3p3o_A")
    g.add_argument("--n", type=int, help="Take the first N systems from the ATLAS list")
    ap.add_argument("--out", default="~/atlas", help="Destination root directory")
    ap.add_argument("--kind", choices=["analysis", "protein", "total"],
                    default="analysis",
                    help="analysis=1000 frames (default, small); protein=10000 frames")
    ap.add_argument("--force", action="store_true", help="Re-download cached systems")
    args = ap.parse_args()

    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.ids:
        ids = args.ids
    else:
        print("querying ATLAS for the system list ...")
        ids = list_systems(args.n)
        if not ids:
            print("Could not retrieve the system list automatically.\n"
                  "Pass systems explicitly instead, e.g.:\n"
                  "  --ids 1g2r_A 3p3o_A 1tag_A\n"
                  "Browse candidates at https://www.dsimb.inserm.fr/ATLAS")
            return 1
        print(f"  got {len(ids)} system id(s)")

    print(f"destination: {out_root}")
    print(f"endpoint   : {BASE}/ATLAS/{args.kind}/<id>")
    print("note: archives are large; each system may take several minutes.\n")
    ok, failed, total_mb = [], {}, 0.0
    for i, pid in enumerate(ids, 1):
        try:
            status, mb = fetch_one(pid, args.kind, out_root, args.force)
            total_mb += mb
            ok.append(pid)
            print(f"[{i}/{len(ids)}] {pid:12s} {status:10s} {mb:8.1f} MB", flush=True)
        except urllib.error.HTTPError as e:
            failed[pid] = f"HTTP {e.code}"
            print(f"[{i}/{len(ids)}] {pid:12s} FAILED  HTTP {e.code}", flush=True)
        except Exception as e:
            failed[pid] = str(e)[:120]
            print(f"[{i}/{len(ids)}] {pid:12s} FAILED  {str(e)[:80]}", flush=True)

    print(f"\n{len(ok)} system(s) ready in {out_root}  ({total_mb:.0f} MB total)")
    if failed:
        print(f"{len(failed)} failed: {json.dumps(failed, indent=2)[:800]}")
    if ok:
        print("\nNext:")
        print(f"  python validation/phase3_scale.py --data_root {args.out} "
              f"--limit 3 --max_frames 1500")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
