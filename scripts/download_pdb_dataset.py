#!/usr/bin/env python3
"""
Download small protein structures from the RCSB Protein Data Bank.

Filter criteria:
- Polymer entity type: protein
- Number of residues < 150
- Single chain structures only
- Resolution better than 3.0 Å

Downloads at least 50 proteins, saving each as:
    data/{PDB_ID}/topology.pdb
"""

import os
import time
import argparse
import requests
from pathlib import Path


RCSB_SEARCH_API = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def build_search_query(max_residues=150, max_resolution=3.0, rows=200):
    """
    Build RCSB search query for small, single-chain protein structures.

    Args:
        max_residues: Upper bound on sequence length (exclusive).
        max_resolution: Maximum resolution in Ångströms (inclusive).
        rows: Number of results to request from the API.

    Returns:
        query: dict suitable for the RCSB search endpoint.
    """
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "less",
                        "value": max_residues,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": max_resolution,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "equals",
                        "value": 1,
                    },
                },
            ],
        },
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
        "return_type": "entry",
    }


def search_pdb(max_residues=150, max_resolution=3.0, rows=200):
    """
    Query RCSB PDB search API and return a list of matching PDB IDs.

    Args:
        max_residues: Maximum number of residues.
        max_resolution: Maximum resolution in Å.
        rows: Number of results requested.

    Returns:
        pdb_ids: List of PDB ID strings.
    """
    query = build_search_query(max_residues, max_resolution, rows)

    print(
        f"[search] Querying RCSB for proteins: "
        f"residues < {max_residues}, resolution ≤ {max_resolution} Å ..."
    )

    resp = requests.post(
        RCSB_SEARCH_API,
        headers={"Content-Type": "application/json"},
        json=query,
        timeout=60,
    )
    resp.raise_for_status()

    data = resp.json()
    pdb_ids = [hit["identifier"] for hit in data.get("result_set", [])]
    total = data.get("total_count", 0)

    print(f"[search] Found {total} total matches, retrieved {len(pdb_ids)} PDB IDs.")
    return pdb_ids


def download_pdb(pdb_id, data_dir, retries=3, delay=1.0):
    """
    Download a single PDB structure and save it as topology.pdb.

    Args:
        pdb_id: Four-character RCSB PDB ID (e.g., "1UBQ").
        data_dir: Root data directory.  File is written to
                  data_dir/{pdb_id}/topology.pdb.
        retries: Number of download attempts before giving up.
        delay: Seconds to wait between retries.

    Returns:
        "downloaded"  – file freshly downloaded.
        "exists"      – file was already present; skipped.
        "failed"      – all attempts failed.
    """
    out_path = Path(data_dir) / pdb_id / "topology.pdb"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"  [skip]  {pdb_id}: already exists at {out_path}")
        return "exists"

    url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id)

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                out_path.write_text(resp.text)
                print(f"  [ok]    {pdb_id} → {out_path}")
                return "downloaded"
            elif resp.status_code == 404:
                print(f"  [skip]  {pdb_id}: HTTP 404 — not found.")
                return "failed"
            else:
                print(
                    f"  [warn]  {pdb_id}: HTTP {resp.status_code} "
                    f"(attempt {attempt}/{retries})"
                )
        except requests.RequestException as exc:
            print(f"  [warn]  {pdb_id}: {exc} (attempt {attempt}/{retries})")

        if attempt < retries:
            time.sleep(delay)

    print(f"  [fail]  {pdb_id}: all {retries} download attempts failed.")
    return "failed"


def main():
    parser = argparse.ArgumentParser(
        description="Download small protein structures from RCSB PDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Root data directory (proteins saved to data/{PDB_ID}/topology.pdb)",
    )
    parser.add_argument(
        "--max_residues",
        type=int,
        default=150,
        help="Maximum number of residues (exclusive upper bound)",
    )
    parser.add_argument(
        "--max_resolution",
        type=float,
        default=3.0,
        help="Maximum resolution in Å (inclusive)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=50,
        help="Minimum number of proteins to download",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=200,
        help="Number of candidates to request from RCSB (should be > target)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Pause between successive downloads (seconds)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RCSB PDB Dataset Downloader")
    print("=" * 60)

    # Search
    pdb_ids = search_pdb(
        max_residues=args.max_residues,
        max_resolution=args.max_resolution,
        rows=args.rows,
    )

    if not pdb_ids:
        print("[error] No proteins returned. Check filters or network connection.")
        return 1

    # Download
    n_downloaded = 0
    n_failed = 0
    n_skipped = 0

    print(f"\n[download] Attempting up to {len(pdb_ids)} proteins ...")

    for pdb_id in pdb_ids:
        if n_downloaded >= args.target:
            print(f"\n[done] Target of {args.target} downloads reached.")
            break

        result = download_pdb(pdb_id, data_dir, delay=args.delay)

        if result == "downloaded":
            n_downloaded += 1
        elif result == "exists":
            n_skipped += 1
        else:
            n_failed += 1

        if (n_downloaded + n_skipped) % 10 == 0 and (n_downloaded + n_skipped) > 0:
            print(
                f"  [progress] downloaded={n_downloaded}, "
                f"skipped={n_skipped}, failed={n_failed}"
            )

        time.sleep(args.delay)

    print("\n" + "=" * 60)
    print(f"[summary] downloaded={n_downloaded}, skipped={n_skipped}, failed={n_failed}")
    print(f"[summary] Data directory: {data_dir.resolve()}")
    print("=" * 60)

    if n_downloaded + n_skipped < args.target:
        print(
            f"[warn] Only {n_downloaded + n_skipped} proteins available "
            f"(target was {args.target})."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
