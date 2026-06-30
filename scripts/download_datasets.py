#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path


def download_url(url: str, out_path: Path, overwrite: bool = False) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        print(f"Exists, skipping: {out_path}")
        return out_path
    print(f"Downloading {url}\n  -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    return out_path


def zenodo_files(record_id: str) -> list[dict]:
    url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("files", [])


def download_zenodo(record_id: str, out_dir: Path, pattern: str | None = None, overwrite: bool = False) -> list[Path]:
    paths: list[Path] = []
    for item in zenodo_files(record_id):
        key = item.get("key") or item.get("filename") or "file"
        if pattern and pattern not in key:
            continue
        link = item.get("links", {}).get("self") or item.get("links", {}).get("download")
        if not link:
            continue
        paths.append(download_url(link, out_dir / key, overwrite=overwrite))
    if not paths:
        raise RuntimeError(f"No Zenodo files matched record={record_id!r}, pattern={pattern!r}")
    return paths


def maybe_extract_tar(path: Path, out_dir: Path) -> None:
    if path.suffixes[-2:] in [[".tar", ".gz"], [".tar", ".xz"]] or path.suffix == ".tar":
        print(f"Extracting {path} -> {out_dir}")
        with tarfile.open(path) as tar:
            tar.extractall(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download recommended FastFusion-IF datasets/benchmarks.")
    parser.add_argument("--dataset", choices=["dips-plus", "db5-plus"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument(
        "--zenodo-pattern",
        default=None,
        help="Optional filename substring for DIPS-Plus; useful for final_raw_dips tarballs.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "dips-plus":
        # Supplementary DIPS-Plus record containing final_raw_dips and FoldSeek splits.
        # Record 8140981 is from the DIPS-Plus supplementary data page.
        paths = download_zenodo("8140981", out_dir / "DIPS-Plus", pattern=args.zenodo_pattern, overwrite=args.overwrite)
    else:
        # Prepared DB5 archive linked from the DIPS-Plus repository.
        url = "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/H93ZKK/BXXQCG"
        paths = [download_url(url, out_dir / "DB5.tar.gz", overwrite=args.overwrite)]

    if args.extract:
        for path in paths:
            maybe_extract_tar(path, out_dir)
    print("Downloaded files:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
