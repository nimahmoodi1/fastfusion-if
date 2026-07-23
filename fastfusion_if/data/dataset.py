from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import DataConfig
from ..constants import UNKNOWN_ELEMENT_INDEX
from ..utils import find_structure_files
from .features import element_to_index
from .graphs import build_all_graphs, build_residue_graph
from .labels import labels_against_all_other_chains, residue_table
from .pdb_parser import atoms_by_chain, parse_structure_atoms
from .residue_features import build_residue_features
from .pickle_loader import parse_pickle_atoms
from .structures import AtomRecord, ChainExample
from .surface import sample_surface_points, surface_scalar_features


def parse_any_atoms(path: str | Path, cfg: DataConfig) -> list[AtomRecord]:
    path = Path(path)
    if path.suffix.lower() in {".pkl", ".pickle", ".dill", ".p"}:
        return parse_pickle_atoms(path, drop_hydrogens=cfg.drop_hydrogens)
    return parse_structure_atoms(path, drop_hydrogens=cfg.drop_hydrogens)


def make_chain_example(
    chain_atoms: list[AtomRecord],
    all_atoms: list[AtomRecord],
    cfg: DataConfig,
    source_path: str = "",
    with_labels: bool = True,
    seed: int = 0,
    augment: bool = False,
    label_override: dict[str, int] | None = None,
    feature_override: dict[str, list[float]] | None = None,
) -> ChainExample:
    if not chain_atoms:
        raise ValueError("Cannot build example from an empty chain")
    raw_atom_pos = np.stack([a.coord for a in chain_atoms]).astype(np.float32)

    if cfg.center_coordinates:
        center = raw_atom_pos.mean(axis=0, keepdims=True)
        atom_pos = raw_atom_pos - center
        centered_atoms = [
            AtomRecord(atom_pos[i], a.element, a.atom_name, a.chain_id, a.res_seq, a.insertion, a.res_name, a.idr_annotation, a.idr_propensity, a.residue_index)
            for i, a in enumerate(chain_atoms)
        ]
    else:
        atom_pos = raw_atom_pos
        centered_atoms = chain_atoms

    atom2res, residue_keys, residue_names = residue_table(centered_atoms)
    atom_elem = np.asarray([element_to_index(a.element) for a in centered_atoms], dtype=np.int64)
    atom_elem = np.where(atom_elem < 0, UNKNOWN_ELEMENT_INDEX, atom_elem)

    # Residue center = mean coordinate of atoms belonging to that residue.
    # This graph is used only after atom/surface fusion, so it is lightweight.
    residue_pos = np.zeros((len(residue_keys), 3), dtype=np.float32)
    residue_count = np.zeros((len(residue_keys), 1), dtype=np.float32)
    np.add.at(residue_pos, atom2res, atom_pos)
    np.add.at(residue_count, atom2res, 1.0)
    residue_pos = residue_pos / np.maximum(residue_count, 1.0)
    residue_edge_index = build_residue_graph(residue_pos, len(residue_keys), cfg)
    residue_features = build_residue_features(residue_names, residue_keys, atom2res, centered_atoms, residue_edge_index)

    # Benchmark mode: replace the lightweight handcrafted residue features with an
    # externally-supplied per-residue feature vector (e.g. PSSM|HMM|DSSP profiles),
    # aligned by residue key exactly like label_override. Residues with no entry
    # get a zero vector. This feeds the same evolutionary/structural feature set
    # the competing methods use through our existing residue_features channel.
    if feature_override is not None:
        dim = 0
        for v in feature_override.values():
            dim = len(v)
            break
        if dim > 0:
            residue_features = np.array(
                [feature_override.get(f"{k[0]}:{k[1]}:{k[2]}", [0.0] * dim) for k in residue_keys],
                dtype=np.float32,
            )
        labels = np.array(
            [float(label_override.get(f"{k[0]}:{k[1]}:{k[2]}", 0)) for k in residue_keys],
            dtype=np.float32,
        )
    elif with_labels:
        labels = labels_against_all_other_chains(chain_atoms, all_atoms, cfg.label_cutoff)
    else:
        labels = None

    surface_pos, normals, surface2atom = sample_surface_points(centered_atoms, cfg, seed=seed)
    surface_features, surface2res = surface_scalar_features(centered_atoms, surface_pos, normals, surface2atom, cfg)
    if augment and cfg.random_rotation:
        rng = np.random.default_rng()  # fresh entropy -> true per-epoch augmentation
        q = rng.normal(size=4).astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        w, x, y, z = q
        rot = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float32)
        atom_pos = atom_pos @ rot.T
        surface_pos = surface_pos @ rot.T
        residue_pos = residue_pos @ rot.T
    if augment and cfg.coordinate_jitter_std > 0:
        rng = np.random.default_rng()
        atom_pos = atom_pos + rng.normal(0.0, cfg.coordinate_jitter_std, size=atom_pos.shape).astype(np.float32)
        surface_pos = surface_pos + rng.normal(0.0, cfg.coordinate_jitter_std, size=surface_pos.shape).astype(np.float32)

    graphs = build_all_graphs(atom_pos, surface_pos, cfg)

    return ChainExample(
        atom_pos=atom_pos,
        atom_elem=atom_elem,
        atom2res=atom2res,
        residue_keys=residue_keys,
        residue_names=residue_names,
        labels=labels,
        residue_pos=residue_pos,
        residue_edge_index=residue_edge_index,
        residue_features=residue_features,
        surface_pos=surface_pos,
        surface_features=surface_features,
        surface2res=surface2res,
        source_path=str(source_path),
        chain_id=chain_atoms[0].chain_id,
        **graphs,
    )


class ProteinInterfaceDataset(Dataset):
    """On-the-fly dataset for complexes stored as PDB/mmCIF or DIPS-like pickles.

    Each structure file can yield multiple single-chain examples. For a complex with chains A/B/C, the
    model sees A alone with labels computed against B+C, then B alone against A+C, etc.
    """

    def __init__(
        self,
        root_or_files: str | Path | Iterable[str | Path],
        cfg: DataConfig,
        with_labels: bool = True,
        max_chains_per_file: Optional[int] = None,
        augment: bool = False,
    ) -> None:
        self.cfg = cfg
        self.with_labels = with_labels
        self.max_chains_per_file = max_chains_per_file
        self.augment = augment
        if isinstance(root_or_files, (str, Path)) and Path(root_or_files).is_dir():
            files = find_structure_files(root_or_files, cfg.file_glob)
        else:
            files = [Path(p) for p in root_or_files] if not isinstance(root_or_files, (str, Path)) else [Path(root_or_files)]
        if cfg.max_files is not None:
            files = files[: cfg.max_files]
        self.files = list(files)
        if not self.files:
            raise FileNotFoundError("No structure/pickle files were found for ProteinInterfaceDataset")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> list[ChainExample]:
        path = self.files[index]
        try:
            atoms = parse_any_atoms(path, self.cfg)
            chains = atoms_by_chain(atoms)
            chain_items = sorted(chains.items(), key=lambda kv: len(kv[1]), reverse=True)
            if self.max_chains_per_file is not None:
                chain_items = chain_items[: self.max_chains_per_file]
            examples = []
            for chain_id, chain_atoms in chain_items:
                if len(chain_atoms) < 5:
                    continue
                if self.with_labels and len(chains) < 2:
                    continue
                examples.append(
                    make_chain_example(
                        chain_atoms=chain_atoms,
                        all_atoms=atoms,
                        cfg=self.cfg,
                        source_path=str(path),
                        with_labels=self.with_labels,
                        seed=index,
                        augment=self.augment,
                    )
                )
            return examples
        except Exception as exc:
            if self.cfg.skip_errors:
                return []
            raise RuntimeError(f"Failed processing {path}: {exc}") from exc
