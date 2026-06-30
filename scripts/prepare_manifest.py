#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from fastfusion_if.config import ExperimentConfig
from fastfusion_if.data.dataset import parse_any_atoms
from fastfusion_if.data.sequences import ChainSequenceRecord, sequence_records_from_atoms, write_fasta
from fastfusion_if.data.splits import (
    build_cluster_aware_manifest_rows,
    cluster_with_cdhit,
    cluster_with_internal_identity,
    cluster_with_mmseqs,
    manifest_leakage_report,
    write_manifest_and_report,
    write_random_manifest,
)
from fastfusion_if.utils import find_structure_files


def extract_chain_sequences(files: list[str], cfg: ExperimentConfig, min_chain_length: int) -> list[ChainSequenceRecord]:
    records: list[ChainSequenceRecord] = []
    for i, path in enumerate(tqdm(files, desc="extracting chain sequences")):
        try:
            atoms = parse_any_atoms(path, cfg.data)
            records.extend(sequence_records_from_atoms(atoms, file_path=path, file_index=i, min_length=min_chain_length))
        except Exception as exc:
            print(f"WARNING: failed to parse {path}: {exc}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a leakage-safe manifest by clustering protein chains and assigning whole connected components to splits."
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing PDB/mmCIF/DIPS/DB5 pickle complex files")
    parser.add_argument("--out", required=True, help="Output manifest CSV")
    parser.add_argument("--config", default=None, help="Optional JSON config; only data parsing options are used")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.80)
    parser.add_argument("--val", type=float, default=0.10)
    parser.add_argument("--identity", type=float, default=0.30, help="Sequence identity threshold, e.g. 0.30 for 30%%")
    parser.add_argument("--coverage", type=float, default=0.80, help="MMseqs2 coverage threshold")
    parser.add_argument("--cov-mode", type=int, default=1, help="MMseqs2 coverage mode")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--min-chain-length", type=int, default=20)
    parser.add_argument("--work-dir", default=None, help="Directory for FASTA and cluster files; defaults to <out>.work")
    parser.add_argument(
        "--cluster-method",
        choices=["mmseqs", "cdhit", "identity", "random_debug"],
        default="mmseqs",
        help="Use mmseqs/cdhit for reportable experiments. identity/random_debug are only for tiny debugging.",
    )
    parser.add_argument("--file-glob", default=None, help="Override file glob, e.g. '*.dill' or '*.pdb'")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_json(args.config) if args.config else ExperimentConfig()
    file_glob = args.file_glob or cfg.data.file_glob
    files = [str(p) for p in find_structure_files(args.data_dir, file_glob)]
    if not files:
        raise FileNotFoundError(f"No structure files found under {args.data_dir} with glob {file_glob!r}")

    if args.cluster_method == "random_debug":
        print("WARNING: creating a random DEBUG manifest. Do not report results from this split.")
        write_random_manifest(files, args.out, seed=args.seed, train=args.train, val=args.val)
        rows = []
        import csv
        with Path(args.out).open() as f:
            rows = list(csv.DictReader(f))
        Path(args.out).with_suffix(".report.json").write_text(json.dumps(manifest_leakage_report(rows), indent=2))
        print(f"Wrote {args.out} with {len(files)} files")
        return

    work_dir = Path(args.work_dir) if args.work_dir else Path(args.out).with_suffix(".work")
    work_dir.mkdir(parents=True, exist_ok=True)
    records = extract_chain_sequences(files, cfg, min_chain_length=args.min_chain_length)
    if not records:
        raise RuntimeError("No protein chain sequences could be extracted. Check parser/data format.")
    fasta_path = work_dir / "chains.fasta"
    write_fasta(records, fasta_path)

    if args.cluster_method == "mmseqs":
        seq_to_cluster = cluster_with_mmseqs(
            fasta_path,
            work_dir / "mmseqs",
            min_seq_id=args.identity,
            coverage=args.coverage,
            cov_mode=args.cov_mode,
            threads=args.threads,
        )
    elif args.cluster_method == "cdhit":
        seq_to_cluster = cluster_with_cdhit(
            fasta_path,
            work_dir / "cdhit",
            min_seq_id=args.identity,
            threads=args.threads,
        )
    else:
        print("WARNING: using simple internal identity clustering. Use MMseqs2/CD-HIT for paper numbers.")
        seq_to_cluster = cluster_with_internal_identity(fasta_path, min_seq_id=args.identity)

    rows, report = build_cluster_aware_manifest_rows(
        files=files,
        records=records,
        seq_to_cluster=seq_to_cluster,
        seed=args.seed,
        train=args.train,
        val=args.val,
    )
    write_manifest_and_report(rows, args.out)
    Path(args.out).with_suffix(".cluster_report.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote manifest: {args.out}")
    print(json.dumps(report, indent=2))
    if report.get("n_chain_cluster_leaks", 0) != 0:
        raise RuntimeError("Manifest has chain-cluster leakage across splits. Do not use it until fixed.")


if __name__ == "__main__":
    main()
