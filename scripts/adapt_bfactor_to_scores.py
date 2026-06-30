#!/usr/bin/env python
"""Convert a directory of scored PDBs (e.g. PeSTo output, which writes the interface
probability into the B-factor column) into the per-residue scores CSV that
scripts/compare_external.py consumes.

Each scored PDB is matched back to a group_id via the pdb_index.csv produced by
scripts/export_test_pdbs.py (matched on filename stem, allowing a method-added
suffix such as '_i0'). Per-residue score = max (default) or mean B-factor over the
residue's atoms.

Usage
-----
python scripts/adapt_bfactor_to_scores.py \
    --scored-dir eval/pesto_out --index eval/test_pdbs/pdb_index.csv \
    --out eval/pesto/pesto_scores.csv --agg max
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_bfactors(pdb_path: Path):
    """Yield (chain, res_seq, insertion, res_name, bfactor) per ATOM line."""
    for ln in pdb_path.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        try:
            chain = ln[21].strip()
            res_seq = int(ln[22:26])
            icode = ln[26].strip()
            res_name = ln[17:20].strip()
            bfac = float(ln[60:66])
        except (ValueError, IndexError):
            continue
        yield chain, res_seq, icode, res_name, bfac


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-dir", required=True, help="dir of scored PDBs (B-factor = score)")
    ap.add_argument("--index", required=True, help="pdb_index.csv from export_test_pdbs.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--agg", choices=["max", "mean"], default="max")
    args = ap.parse_args()

    with open(args.index, newline="") as f:
        index = list(csv.DictReader(f))
    # map filename stem of OUR exported pdb -> group_id
    stem_to_gid = {Path(r["pdb_path"]).stem: r["group_id"] for r in index}

    scored = {p.stem: p for p in Path(args.scored_dir).glob("*.pdb")}

    rows = []
    matched_files = 0
    for stem, gid in stem_to_gid.items():
        # exact stem, else a scored file that startswith our stem (suffix tolerant)
        path = scored.get(stem)
        if path is None:
            cand = [p for s, p in scored.items() if s.startswith(stem)]
            path = cand[0] if cand else None
        if path is None:
            continue
        matched_files += 1
        per_res: dict[tuple, list] = {}
        for chain, res_seq, icode, res_name, bfac in parse_bfactors(path):
            per_res.setdefault((chain, res_seq, icode), []).append(bfac)
        for (chain, res_seq, icode), vals in per_res.items():
            score = max(vals) if args.agg == "max" else sum(vals) / len(vals)
            rows.append({"group_id": gid, "res_chain": chain, "res_seq": res_seq, "insertion": icode, "score": score})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group_id", "res_chain", "res_seq", "insertion", "score"])
        w.writeheader()
        w.writerows(rows)
    print(f"matched {matched_files}/{len(stem_to_gid)} chains; wrote {len(rows)} residue scores -> {args.out}")


if __name__ == "__main__":
    main()
