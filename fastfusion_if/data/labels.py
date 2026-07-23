from __future__ import annotations

from collections import OrderedDict, defaultdict
from itertools import product
from typing import Iterable

import numpy as np

from .structures import AtomRecord


def residue_table(atoms: Iterable[AtomRecord]) -> tuple[np.ndarray, list[tuple[str, int, str]], list[str]]:
    keys: list[tuple[str, int, str]] = []
    names: list[str] = []
    key_to_idx: OrderedDict[tuple[str, int, str], int] = OrderedDict()
    atom2res: list[int] = []

    for atom in atoms:
        key = atom.residue_key
        if key not in key_to_idx:
            key_to_idx[key] = len(keys)
            keys.append(key)
            names.append(atom.res_name)
        atom2res.append(key_to_idx[key])

    return np.asarray(atom2res, dtype=np.int64), keys, names


def _coords_from_atoms(atoms: list[AtomRecord]) -> np.ndarray:
    if not atoms:
        return np.zeros((0, 3), dtype=np.float32)

    xyz = np.asarray([a.coord for a in atoms], dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float32)

    return np.ascontiguousarray(xyz, dtype=np.float32)


def _atoms_within_cutoff_grid(xyz_a: np.ndarray, xyz_b: np.ndarray, cutoff: float) -> np.ndarray:
    """Return mask for atoms in xyz_a that are within cutoff of any atom in xyz_b.

    This avoids scipy.spatial.cKDTree, which caused native-code segmentation
    faults during long DIPS-Plus training. The grid uses cell size = cutoff, so
    any true neighbor must be in the same or one of the 26 neighboring cells.
    """
    n_a = int(xyz_a.shape[0])
    n_b = int(xyz_b.shape[0])
    hit = np.zeros((n_a,), dtype=bool)

    if n_a == 0 or n_b == 0:
        return hit

    if cutoff <= 0:
        return hit

    cutoff2 = float(cutoff) * float(cutoff)
    cell_size = float(cutoff)

    # Build partner atom grid.
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    b_cells = np.floor(xyz_b / cell_size).astype(np.int64)

    for j, key_arr in enumerate(b_cells):
        cells[(int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))].append(j)

    # Convert lists to arrays once.
    cell_arrays: dict[tuple[int, int, int], np.ndarray] = {
        key: np.asarray(ids, dtype=np.int64) for key, ids in cells.items()
    }

    neighbor_offsets = list(product((-1, 0, 1), repeat=3))

    a_cells = np.floor(xyz_a / cell_size).astype(np.int64)

    for i in range(n_a):
        p = xyz_a[i]
        base = a_cells[i]

        for off in neighbor_offsets:
            key = (
                int(base[0] + off[0]),
                int(base[1] + off[1]),
                int(base[2] + off[2]),
            )
            ids = cell_arrays.get(key)
            if ids is None or ids.size == 0:
                continue

            pts = xyz_b[ids]
            diff = pts - p[None, :]
            d2 = np.sum(diff * diff, axis=1)

            if bool(np.any(d2 <= cutoff2)):
                hit[i] = True
                break

    return hit


def interface_labels_for_chain(chain_atoms: list[AtomRecord], partner_atoms: list[AtomRecord], cutoff: float) -> np.ndarray:
    atom2res, keys, _ = residue_table(chain_atoms)
    y = np.zeros((len(keys),), dtype=np.float32)

    if not chain_atoms or not partner_atoms or len(keys) == 0:
        return y

    xyz_a_all = _coords_from_atoms(chain_atoms)
    xyz_b_all = _coords_from_atoms(partner_atoms)

    if xyz_a_all.shape[0] == 0 or xyz_b_all.shape[0] == 0:
        return y

    # Drop non-finite coordinates safely.
    finite_a = np.isfinite(xyz_a_all).all(axis=1)
    finite_b = np.isfinite(xyz_b_all).all(axis=1)

    if not bool(finite_a.any()) or not bool(finite_b.any()):
        return y

    xyz_a = np.ascontiguousarray(xyz_a_all[finite_a], dtype=np.float32)
    xyz_b = np.ascontiguousarray(xyz_b_all[finite_b], dtype=np.float32)
    atom2res_finite = atom2res[finite_a]

    atom_hit = _atoms_within_cutoff_grid(xyz_a, xyz_b, cutoff=cutoff)

    if bool(atom_hit.any()):
        y[np.unique(atom2res_finite[atom_hit])] = 1.0

    return y


def labels_against_all_other_chains(chain_atoms: list[AtomRecord], all_atoms: list[AtomRecord], cutoff: float) -> np.ndarray:
    chain_id = chain_atoms[0].chain_id if chain_atoms else ""
    partner = [a for a in all_atoms if a.chain_id != chain_id]
    return interface_labels_for_chain(chain_atoms, partner, cutoff=cutoff)
