from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AtomRecord:
    coord: np.ndarray
    element: str
    atom_name: str
    chain_id: str
    res_seq: int
    insertion: str
    res_name: str
    idr_annotation: float = 0.0
    idr_propensity: float = 0.0
    residue_index: int = 0

    @property
    def residue_key(self) -> tuple[str, int, str]:
        return (str(self.chain_id), int(self.res_seq), str(self.insertion or ""))


@dataclass
class ChainExample:
    """A processed single-chain training/inference example."""

    atom_pos: np.ndarray
    atom_elem: np.ndarray
    atom2res: np.ndarray
    residue_keys: list[tuple[str, int, str]]
    residue_names: list[str]
    labels: Optional[np.ndarray]

    residue_pos: np.ndarray
    residue_edge_index: np.ndarray
    residue_features: np.ndarray

    surface_pos: np.ndarray
    surface_features: np.ndarray
    surface2res: np.ndarray

    atom_edge_index: np.ndarray
    surface_edge_index: np.ndarray
    atom_query_surface_key: np.ndarray
    surface_query_atom_key: np.ndarray

    source_path: str = ""
    chain_id: str = ""

    # Optional precomputed protein-language-model residue embeddings [n_res, D].
    residue_plm: Optional[np.ndarray] = None

    @property
    def n_residues(self) -> int:
        return len(self.residue_keys)

    @property
    def n_atoms(self) -> int:
        return int(self.atom_pos.shape[0])

    @property
    def n_surface_points(self) -> int:
        return int(self.surface_pos.shape[0])
