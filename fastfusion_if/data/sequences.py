from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .structures import AtomRecord

# Standard and common modified amino-acid mapping. Unknown/non-protein residues become X.
AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
    # Frequent PDB modifications mapped to their parent residue.
    "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y", "CSO": "C",
    "HYP": "P", "MLY": "K", "KCX": "K", "CME": "C", "CSD": "C",
    "CYX": "C", "HSD": "H", "HSE": "H", "HSP": "H", "ASX": "B",
    "GLX": "Z", "UNK": "X",
}


@dataclass(frozen=True)
class ChainSequenceRecord:
    """One protein chain sequence extracted from a complex file."""

    sequence_id: str
    file_path: str
    chain_id: str
    sequence: str
    n_residues: int


def _residue_order_for_chain(atoms: Iterable[AtomRecord]) -> OrderedDict[tuple[int, str], str]:
    residues: OrderedDict[tuple[int, str], str] = OrderedDict()
    for atom in atoms:
        key = (int(atom.res_seq), str(atom.insertion or ""))
        if key not in residues:
            residues[key] = str(atom.res_name).upper()
    return residues


def chain_sequence(chain_atoms: Iterable[AtomRecord], min_length: int = 1) -> str:
    """Return a one-letter amino-acid sequence for atoms from one chain."""
    residues = _residue_order_for_chain(chain_atoms)
    seq = "".join(AA3_TO_AA1.get(res_name, "X") for res_name in residues.values())
    if len(seq) < min_length:
        return ""
    return seq


def sequence_records_from_atoms(
    atoms: Iterable[AtomRecord],
    file_path: str | Path,
    file_index: int,
    min_length: int = 20,
) -> list[ChainSequenceRecord]:
    """Extract one sequence record per chain from atom records.

    The sequence IDs are deterministic and safe for FASTA/MMseqs/CD-HIT parsing:
    ``f{file_index}|{chain_id}``.
    """
    chains: dict[str, list[AtomRecord]] = {}
    for atom in atoms:
        chains.setdefault(str(atom.chain_id), []).append(atom)

    records: list[ChainSequenceRecord] = []
    for chain_id in sorted(chains):
        seq = chain_sequence(chains[chain_id], min_length=min_length)
        # Exclude chains that are mostly unknown/non-protein tokens.
        known = sum(aa not in {"X", "B", "Z"} for aa in seq)
        if not seq or known / max(len(seq), 1) < 0.70:
            continue
        records.append(
            ChainSequenceRecord(
                sequence_id=f"f{file_index}|{chain_id}",
                file_path=str(file_path),
                chain_id=chain_id,
                sequence=seq,
                n_residues=len(seq),
            )
        )
    return records


def write_fasta(records: Iterable[ChainSequenceRecord], path: str | Path) -> None:
    """Write chain sequence records to FASTA."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(f">{rec.sequence_id}\n")
            seq = rec.sequence
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")


def read_fasta(path: str | Path) -> dict[str, str]:
    """Small FASTA reader used by the pure-Python identity fallback."""
    seqs: dict[str, list[str]] = {}
    current: str | None = None
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                seqs[current] = []
            elif current is not None:
                seqs[current].append(line)
    return {k: "".join(v) for k, v in seqs.items()}
