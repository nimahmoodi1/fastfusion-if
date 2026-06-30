#!/usr/bin/env python
"""Export the test chains we evaluate on to single-chain PDB files, so PeSTo /
ScanNet (which take PDBs) can be run on exactly the same residues. Pair this with
scripts/compare_external.py.

It reads the per-residue prediction CSV from scripts/evaluate.py to learn which
(source_path, chain_id) pairs were actually scored, re-parses each source structure,
keeps that chain's atoms, and writes one PDB per chain plus an index CSV mapping
group_id -> pdb path.

Usage
-----
python scripts/export_test_pdbs.py \
    --predictions eval/full_v2_esm_reg/test_per_residue_predictions.csv \
    --config configs/full_v2_cached.json \
    --out-dir eval/test_pdbs
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def write_pdb(atoms, path: Path) -> int:
    """Write AtomRecords (with .coord, .element, .atom_name, .chain_id, .res_seq,
    .insertion, .res_name) as a minimal but standards-compliant PDB. Returns count."""
    lines = []
    serial = 0
    for a in atoms:
        serial += 1
        name = str(a.atom_name)
        elem = str(a.element).strip().upper()[:2]
        # PDB atom-name field is cols 13-16; 1-char elements are padded with a
        # leading space (so "CA" the atom sits in 14-15), matching Biopython.
        if len(name) >= 4:
            name_field = name[:4]
        elif len(elem) == 1:
            name_field = f" {name:<3}"
        else:
            name_field = f"{name:<4}"
        x, y, z = (float(c) for c in list(a.coord)[:3])
        icode = (str(a.insertion) or " ")[:1] or " "
        resn = str(a.res_name)[:3]
        chain = (str(a.chain_id) or "A")[:1]
        resseq = int(a.res_seq)
        lines.append(
            f"ATOM  {serial:>5} {name_field}{'':1}{resn:>3} {chain:1}{resseq:>4}{icode:1}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {elem:>2}"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return serial


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="per-residue CSV from scripts/evaluate.py")
    ap.add_argument("--config", required=True, help="a config json (for parse settings)")
    ap.add_argument("--out-dir", default="eval/test_pdbs")
    args = ap.parse_args()

    import json as _json

    from fastfusion_if.config import ExperimentConfig
    from fastfusion_if.data.dataset import parse_any_atoms

    cfg = ExperimentConfig.from_dict(_json.load(open(args.config)))

    with open(args.predictions, newline="") as f:
        rows = list(csv.DictReader(f))
    # unique chains that were evaluated, preserving source path
    wanted: dict[str, tuple[str, str]] = {}
    for r in rows:
        gid = r["group_id"]
        wanted.setdefault(gid, (r["source_path"], r["chain_id"]))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    done = failed = 0
    for gid, (src, chain) in sorted(wanted.items()):
        try:
            atoms = [a for a in parse_any_atoms(src, cfg.data) if str(a.chain_id) == str(chain)]
            if not atoms:
                failed += 1
                continue
            safe = gid.replace("/", "_").replace(":", "__")
            pdb_path = out / f"{safe}.pdb"
            write_pdb(atoms, pdb_path)
            index.append({"group_id": gid, "pdb_path": str(pdb_path), "chain_id": chain, "n_atoms": len(atoms)})
            done += 1
            if done % 200 == 0:
                print(f"exported {done} chains...")
        except Exception as e:  # noqa: BLE001
            failed += 1
            if failed <= 5:
                print(f"[warn] {gid}: {e}")
    with open(out / "pdb_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group_id", "pdb_path", "chain_id", "n_atoms"])
        w.writeheader()
        w.writerows(index)
    print(json.dumps({"exported": done, "failed": failed, "index": str(out / 'pdb_index.csv')}, indent=2))


if __name__ == "__main__":
    main()
