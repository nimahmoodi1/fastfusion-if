from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser

from .features import normalize_element
from .structures import AtomRecord


def parse_structure_atoms(path: str | Path, drop_hydrogens: bool = True) -> list[AtomRecord]:
    """Parse PDB/mmCIF atoms from the first model, keeping standard residues and common hetero atoms."""
    path = Path(path)
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))

    records: list[AtomRecord] = []
    model = next(structure.get_models())
    for chain in model:
        for res in chain:
            # Skip water. Keep standard residues and useful ligands/ions only when user wants them later.
            if res.get_resname().upper() in {"HOH", "WAT", "DOD"}:
                continue
            for atom in res:
                atom_name = atom.get_name().strip()
                element = normalize_element(getattr(atom, "element", ""), atom_name)
                if drop_hydrogens and element == "H":
                    continue
                if atom.is_disordered():
                    atom = atom.selected_child
                coord = np.asarray(atom.get_coord(), dtype=np.float32)
                records.append(
                    AtomRecord(
                        coord=coord,
                        element=element,
                        atom_name=atom_name,
                        chain_id=str(chain.id),
                        res_seq=int(res.id[1]),
                        insertion=str(res.id[2]).strip(),
                        res_name=str(res.get_resname()).upper(),
                    )
                )
    return records


def atoms_by_chain(atoms: Iterable[AtomRecord]) -> dict[str, list[AtomRecord]]:
    chains: dict[str, list[AtomRecord]] = {}
    for atom in atoms:
        chains.setdefault(atom.chain_id, []).append(atom)
    return chains
