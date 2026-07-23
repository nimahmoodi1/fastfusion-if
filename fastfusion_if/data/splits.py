from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
from dataclasses import asdict
from typing import Iterable

from .sequences import ChainSequenceRecord, read_fasta, write_fasta


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1

    def find(self, x: str) -> str:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def read_manifest(path: str | Path) -> dict[str, list[str]]:
    """Read a CSV manifest with at least columns: path, split."""
    splits: dict[str, list[str]] = defaultdict(list)
    with Path(path).open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "path" not in reader.fieldnames or "split" not in reader.fieldnames:
            raise ValueError("Manifest must contain columns: path, split")
        for row in reader:
            splits[row["split"]].append(row["path"])
    return dict(splits)


def read_manifest_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_random_manifest(
    files: list[str],
    out_csv: str | Path,
    seed: int = 42,
    train: float = 0.8,
    val: float = 0.1,
) -> None:
    """Debug-only random split. Never use this for reported experiments."""
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * train)
    n_val = int(n * val)
    rows = []
    for i, path in enumerate(files):
        split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
        rows.append({"path": path, "split": split, "cluster_id": f"debug_random_{i}"})
    _write_rows(rows, out_csv)


def _write_rows(rows: list[dict[str, str]], out_csv: str | Path) -> None:
    fieldnames = ["path", "split", "cluster_id", "n_chains", "chain_clusters", "component_size"]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_csv).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _check_executable(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise FileNotFoundError(
            f"Could not find '{name}' on PATH. Install it first or choose another --cluster-method."
        )
    return exe


def cluster_with_mmseqs(
    fasta_path: str | Path,
    out_dir: str | Path,
    min_seq_id: float = 0.30,
    coverage: float = 0.80,
    cov_mode: int = 1,
    threads: int = 8,
    mmseqs_bin: str = "mmseqs",
) -> dict[str, str]:
    """Cluster sequences with MMseqs2 easy-cluster and return sequence_id -> cluster_id."""
    _check_executable(mmseqs_bin)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "mmseqs_cluster"
    tmp = out_dir / "mmseqs_tmp"
    cmd = [
        mmseqs_bin,
        "easy-cluster",
        str(fasta_path),
        str(prefix),
        str(tmp),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--threads",
        str(threads),
    ]
    subprocess.run(cmd, check=True)
    tsv_candidates = sorted(out_dir.glob("*cluster.tsv")) + sorted(out_dir.glob("*_cluster.tsv"))
    if not tsv_candidates:
        raise FileNotFoundError(f"MMseqs2 finished but no cluster TSV was found under {out_dir}")
    return parse_mmseqs_cluster_tsv(tsv_candidates[0])


def parse_mmseqs_cluster_tsv(path: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            rep, member, *_ = line.rstrip("\n").split("\t")
            mapping[member] = rep
            mapping.setdefault(rep, rep)
    return mapping


def cluster_with_cdhit(
    fasta_path: str | Path,
    out_dir: str | Path,
    min_seq_id: float = 0.30,
    threads: int = 8,
    cdhit_bin: str = "cd-hit",
) -> dict[str, str]:
    """Cluster sequences with CD-HIT and return sequence_id -> cluster_id."""
    _check_executable(cdhit_bin)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / "cdhit_cluster.fasta"
    # CD-HIT word size constraints: c=0.3 normally requires n=2.
    word_size = 2 if min_seq_id < 0.4 else 3 if min_seq_id < 0.5 else 4 if min_seq_id < 0.7 else 5
    cmd = [
        cdhit_bin,
        "-i",
        str(fasta_path),
        "-o",
        str(out_prefix),
        "-c",
        str(min_seq_id),
        "-n",
        str(word_size),
        "-d",
        "0",
        "-M",
        "0",
        "-T",
        str(threads),
    ]
    subprocess.run(cmd, check=True)
    return parse_cdhit_clusters(str(out_prefix) + ".clstr")


def parse_cdhit_clusters(path: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    current_cluster = None
    rep_id = None
    cluster_members: list[str] = []
    pattern = re.compile(r">([^\.\s]+)")

    def flush() -> None:
        if current_cluster is None or not cluster_members:
            return
        cluster_name = rep_id or f"cdhit_{current_cluster}"
        for member in cluster_members:
            mapping[member] = cluster_name

    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                flush()
                current_cluster = line.split()[-1]
                rep_id = None
                cluster_members = []
                continue
            match = pattern.search(line)
            if match:
                seq_id = match.group(1)
                cluster_members.append(seq_id)
                if line.endswith("*"):
                    rep_id = seq_id
    flush()
    return mapping


def _global_identity(a: str, b: str) -> float:
    """Simple identity for fallback clustering. Uses padded positional identity, not MMseqs/CD-HIT."""
    if not a or not b:
        return 0.0
    m = min(len(a), len(b))
    matches = sum(aa == bb for aa, bb in zip(a[:m], b[:m]))
    return matches / max(len(a), len(b), 1)


def cluster_with_internal_identity(fasta_path: str | Path, min_seq_id: float = 0.30) -> dict[str, str]:
    """Pure-Python fallback for tiny smoke tests. Not recommended for publication splits."""
    seqs = read_fasta(fasta_path)
    reps: list[str] = []
    mapping: dict[str, str] = {}
    for seq_id, seq in sorted(seqs.items(), key=lambda kv: len(kv[1]), reverse=True):
        assigned = None
        for rep in reps:
            if _global_identity(seq, seqs[rep]) >= min_seq_id:
                assigned = rep
                break
        if assigned is None:
            reps.append(seq_id)
            assigned = seq_id
        mapping[seq_id] = assigned
    return mapping


def build_cluster_aware_manifest_rows(
    files: list[str],
    records: list[ChainSequenceRecord],
    seq_to_cluster: dict[str, str],
    seed: int = 42,
    train: float = 0.8,
    val: float = 0.1,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Assign connected components of complexes and sequence clusters to splits.

    A complex can contain multiple chains. If any chain from complex X clusters with any chain from
    complex Y, both complexes are placed in the same connected component and therefore same split.
    """
    if not math.isclose(train + val, min(train + val, 1.0)) or train <= 0 or val < 0 or train + val >= 1.0:
        raise ValueError("Expected train > 0, val >= 0, and train + val < 1")

    file_to_id = {str(path): f"file::{i}" for i, path in enumerate(files)}
    seq_records_by_file: dict[str, list[ChainSequenceRecord]] = defaultdict(list)
    for rec in records:
        seq_records_by_file[str(rec.file_path)].append(rec)

    uf = UnionFind()
    for path in files:
        uf.add(file_to_id[str(path)])
    for rec in records:
        file_node = file_to_id[str(rec.file_path)]
        cluster_node = f"cluster::{seq_to_cluster.get(rec.sequence_id, rec.sequence_id)}"
        uf.union(file_node, cluster_node)

    comp_to_files: dict[str, list[str]] = defaultdict(list)
    for path in files:
        comp = uf.find(file_to_id[str(path)])
        comp_to_files[comp].append(str(path))

    components = list(comp_to_files.items())
    rng = random.Random(seed)
    rng.shuffle(components)
    components.sort(key=lambda kv: len(kv[1]), reverse=True)

    total_files = len(files)
    targets = {"train": train * total_files, "val": val * total_files, "test": (1.0 - train - val) * total_files}
    counts = {"train": 0, "val": 0, "test": 0}
    split_by_comp: dict[str, str] = {}
    for comp, comp_files in components:
        # Greedy load balancing toward target split sizes.
        deficits = {split: targets[split] - counts[split] for split in counts}
        split = max(deficits, key=deficits.get)
        split_by_comp[comp] = split
        counts[split] += len(comp_files)

    rows: list[dict[str, str]] = []
    for comp, comp_files in components:
        split = split_by_comp[comp]
        comp_cluster_id = "component_" + hashlib.sha1(comp.encode("utf-8")).hexdigest()[:12]
        for path in sorted(comp_files):
            recs = seq_records_by_file.get(path, [])
            clusters = sorted({seq_to_cluster.get(rec.sequence_id, rec.sequence_id) for rec in recs})
            rows.append(
                {
                    "path": path,
                    "split": split,
                    "cluster_id": comp_cluster_id,
                    "n_chains": str(len(recs)),
                    "chain_clusters": ";".join(clusters),
                    "component_size": str(len(comp_files)),
                }
            )

    report = manifest_leakage_report(rows)
    report.update(
        {
            "n_files": len(files),
            "n_chain_sequences": len(records),
            "n_sequence_clusters": len(set(seq_to_cluster.values())),
            "n_components": len(components),
            "split_file_counts": counts,
        }
    )
    return rows, report


def write_cluster_manifest(
    file_to_cluster: dict[str, str],
    out_csv: str | Path,
    seed: int = 42,
    train: float = 0.8,
    val: float = 0.1,
) -> None:
    """Assign whole precomputed clusters to splits to avoid homology leakage."""
    cluster_to_files: dict[str, list[str]] = defaultdict(list)
    for path, cluster in file_to_cluster.items():
        cluster_to_files[str(cluster)].append(path)
    clusters = list(cluster_to_files)
    rng = random.Random(seed)
    rng.shuffle(clusters)
    n = len(clusters)
    n_train = int(n * train)
    n_val = int(n * val)
    split_by_cluster = {}
    for i, c in enumerate(clusters):
        split_by_cluster[c] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    rows = []
    for c, c_files in cluster_to_files.items():
        for path in c_files:
            rows.append({"path": path, "split": split_by_cluster[c], "cluster_id": c})
    _write_rows(rows, out_csv)


def write_manifest_and_report(rows: list[dict[str, str]], out_csv: str | Path) -> None:
    _write_rows(rows, out_csv)
    report = manifest_leakage_report(rows)
    report_path = Path(out_csv).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))


def manifest_leakage_report(rows: list[dict[str, str]]) -> dict[str, object]:
    split_counts: dict[str, int] = defaultdict(int)
    cluster_to_splits: dict[str, set[str]] = defaultdict(set)
    chain_cluster_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = row.get("split", "")
        split_counts[split] += 1
        cid = row.get("cluster_id", "")
        if cid:
            cluster_to_splits[cid].add(split)
        for chain_cluster in str(row.get("chain_clusters", "")).split(";"):
            if chain_cluster:
                chain_cluster_to_splits[chain_cluster].add(split)

    cluster_leaks = {c: sorted(s) for c, s in cluster_to_splits.items() if len(s) > 1}
    chain_cluster_leaks = {c: sorted(s) for c, s in chain_cluster_to_splits.items() if len(s) > 1}
    return {
        "split_counts": dict(split_counts),
        "n_rows": len(rows),
        "n_cluster_leaks": len(cluster_leaks),
        "n_chain_cluster_leaks": len(chain_cluster_leaks),
        "cluster_leaks": cluster_leaks,
        "chain_cluster_leaks": chain_cluster_leaks,
    }
