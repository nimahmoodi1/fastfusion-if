#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv

from fastfusion_if.config import ExperimentConfig
from fastfusion_if.utils import find_structure_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a manifest that marks every file as test for external benchmark evaluation.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--file-glob", default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_json(args.config) if args.config else ExperimentConfig()
    files = [str(p) for p in find_structure_files(args.data_dir, args.file_glob or cfg.data.file_glob)]
    if not files:
        raise FileNotFoundError(f"No supported structure files found under {args.data_dir}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "split", "cluster_id"])
        writer.writeheader()
        for i, path in enumerate(files):
            writer.writerow({"path": path, "split": "test", "cluster_id": f"external_{i:06d}"})
    print(f"Wrote {args.out} with {len(files)} test files")


if __name__ == "__main__":
    main()
