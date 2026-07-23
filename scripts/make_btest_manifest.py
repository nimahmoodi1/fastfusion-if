#!/usr/bin/env python
"""Derive the Btest_31-6 (Btest_25) evaluation manifest from Test_60.

Btest_31-6 is not shipped as its own .pkl file. It is defined as the 25 *bound*
chains inside Test_60 whose monomeric (unbound) counterparts make up UBtest_31-6.
The pairing is given by ``Dataset/bound_unbound_mapping31-6.txt`` in the
AGAT-PPIS repository.

This script reads that mapping, filters ``bench_test60.csv`` down to the bound
members, and writes ``bench_btest.csv``. No new structures, labels or features
are needed -- Btest_25 is a strict subset of Test_60, which is already parsed,
labelled and cached.

The script verifies the result against the published statistics
(25 chains / 5,864 residues / 739 interacting) and refuses to write a manifest
that does not match, so a silent mismatch cannot slip into a results table.

Usage
-----
    python scripts/make_btest_manifest.py \
        --mapping ~/AGAT-PPIS/Dataset/bound_unbound_mapping31-6.txt \
        --manifest-dir manifests/benchmark

If the mapping file is missing locally, pass --download to fetch it from the
AGAT-PPIS repository instead.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

MAPPING_URL = (
    "https://raw.githubusercontent.com/AILBC/AGAT-PPIS/master/"
    "Dataset/bound_unbound_mapping31-6.txt"
)

# Published Btest_31-6 statistics (GTE-PPIS, Brief. Bioinform. 26(3) bbaf290, Table 1;
# identical figures appear in the AGAT-PPIS and MEG-PPIS papers).
EXPECTED_CHAINS = 25
EXPECTED_RESIDUES = 5864
EXPECTED_POSITIVES = 739


def read_mapping(path: Path) -> list[tuple[str, str]]:
    """Parse the bound/unbound mapping file -> [(bound_id, unbound_id), ...]."""
    pairs: list[tuple[str, str]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        # skip the header row ("bound unbound position")
        if parts[0].lower() == "bound":
            continue
        if len(parts) < 2:
            continue
        pairs.append((parts[0], parts[1]))
    return pairs


def download_mapping(dest: Path) -> Path:
    from urllib.request import urlopen

    print(f"downloading {MAPPING_URL}")
    with urlopen(MAPPING_URL, timeout=60) as r:  # noqa: S310
        data = r.read().decode("utf-8")
    dest.write_text(data)
    print(f"saved to {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", default="manifests/benchmark",
                    help="directory holding bench_test60.csv (default: manifests/benchmark)")
    ap.add_argument("--mapping", default=None,
                    help="path to bound_unbound_mapping31-6.txt "
                         "(default: <manifest-dir>/bound_unbound_mapping31-6.txt)")
    ap.add_argument("--download", action="store_true",
                    help="fetch the mapping file from the AGAT-PPIS repository if absent")
    ap.add_argument("--out", default=None,
                    help="output manifest (default: <manifest-dir>/bench_btest.csv)")
    ap.add_argument("--labels", default=None,
                    help="bench_labels.json, used to verify the positive count "
                         "(default: <manifest-dir>/bench_labels.json)")
    ap.add_argument("--force", action="store_true",
                    help="write the manifest even if the published statistics do not match")
    args = ap.parse_args()

    mdir = Path(args.manifest_dir)
    test60 = mdir / "bench_test60.csv"
    if not test60.exists():
        sys.exit(f"ERROR: {test60} not found. Run scripts/prepare_benchmark.py first.")

    mapping_path = Path(args.mapping) if args.mapping else mdir / "bound_unbound_mapping31-6.txt"
    if not mapping_path.exists():
        if args.download:
            mapping_path = download_mapping(mdir / "bound_unbound_mapping31-6.txt")
        else:
            sys.exit(
                f"ERROR: mapping file not found at {mapping_path}\n"
                f"  It ships with AGAT-PPIS as Dataset/bound_unbound_mapping31-6.txt.\n"
                f"  Point --mapping at it, or re-run with --download."
            )

    pairs = read_mapping(mapping_path)
    bound_ids = [b for b, _ in pairs]
    print(f"mapping: {len(pairs)} bound/unbound pairs from {mapping_path.name}")

    # ---- filter Test_60 down to the bound members ---------------------------
    with test60.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_id = {Path(r["path"]).stem: r for r in rows}

    missing = [b for b in bound_ids if b not in by_id]
    if missing:
        sys.exit(
            f"ERROR: {len(missing)} bound chains are not in bench_test60.csv: {missing}\n"
            f"  Btest_31-6 must be a subset of Test_60; this indicates a mismatched "
            f"benchmark version."
        )

    # keep Test_60's original ordering for reproducibility
    selected = [r for r in rows if Path(r["path"]).stem in set(bound_ids)]

    # ---- verify against the published statistics ----------------------------
    ok = True
    print(f"\n  chains   : {len(selected)}  (published: {EXPECTED_CHAINS})")
    if len(selected) != EXPECTED_CHAINS:
        ok = False

    labels_path = Path(args.labels) if args.labels else mdir / "bench_labels.json"
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())
        ids = {Path(r["path"]).stem for r in selected}
        n_res = sum(len(labels[i]) for i in ids if i in labels)
        n_pos = sum(sum(labels[i].values()) for i in ids if i in labels)
        print(f"  residues : {n_res}  (published: {EXPECTED_RESIDUES})")
        print(f"  positives: {n_pos}  (published: {EXPECTED_POSITIVES})")
        if n_res != EXPECTED_RESIDUES or n_pos != EXPECTED_POSITIVES:
            ok = False
    else:
        print(f"  [skip] {labels_path} not found -- residue counts not verified")

    if not ok:
        msg = ("\nWARNING: the derived set does not match the published Btest_31-6 "
               "statistics.\n  Do NOT put these numbers in a table next to published "
               "baselines without\n  understanding why they differ.")
        if not args.force:
            sys.exit(msg + "\n  Re-run with --force to write the manifest anyway.")
        print(msg + "\n  --force given; writing anyway.")
    else:
        print("\n  OK: matches the published Btest_31-6 statistics exactly.")

    out = Path(args.out) if args.out else mdir / "bench_btest.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "split"])
        w.writeheader()
        for r in selected:
            w.writerow({"path": r["path"], "split": "test"})
    print(f"\nwrote {out}  ({len(selected)} chains, split=test)")
    print("\nEvaluate it with the existing cache -- no rebuild needed:")
    print("  python scripts/evaluate_ensemble.py \\")
    print("    --checkpoints runs/bench_evo_pp/best.pt runs/bench_evo_pp_s1/best.pt "
          "runs/bench_evo_pp_s2/best.pt \\")
    print(f"    --manifest {out} --split test --threshold 0.63 \\")
    print("    --cache-dir cache/bench_evo --out-dir eval/bench_evo_ens_btest_tuned")


if __name__ == "__main__":
    main()
