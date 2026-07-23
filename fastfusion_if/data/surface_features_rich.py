from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..config import DataConfig
from .features import hydropathy, residue_charge
from .labels import residue_table
from .structures import AtomRecord


# ---------------------------------------------------------------------------
# Richer surface point features for FastFusion-IF.
#
# Your current surface_scalar_features() emits only 4 scalars (hydropathy,
# a Coulomb proxy, normal-variance curvature, and a distance-based exposure
# proxy). For a paper whose headline is "molecular surface representation",
# that descriptor is thin compared with MaSIF/dMaSIF (shape index, curvature,
# electrostatics, H-bond potential, hydropathy). This module adds genuinely
# informative but still cheap features, all computed from coordinates +
# residue/atom identity (no external PB solver, no MSMS mesh):
#
#   0  hydropathy            (owner residue, Kyte-Doolittle, normalised)
#   1  electrostatics proxy  (truncated Coulomb over nearby residue charges)
#   2  normal-variance curv. (your original curvature proxy)
#   3  exposure proxy        (your original normalised distance-from-owner)
#   4  burial / atom density (log atoms within burial_radius; concave=buried)
#   5  shape planarity       (PCA of local surface patch: (l2 - l3)/(l1+eps))
#   6  shape curvedness      (PCA: l3 / (l1 + l2 + l3 + eps), 0=flat 1=spherical)
#   7  donor propensity      (owner atom is N-type, i.e. likely H-bond donor)
#   8  acceptor propensity   (owner atom is O-type, i.e. likely H-bond acceptor)
#   9  aromatic context      (owner residue is aromatic: F/W/Y/H)
#
# -> RICH_SURFACE_FEATURE_DIM = 10 (+3 if cfg.use_surface_normals_as_features).
#
# Wiring (minimal):
#   1) add to DataConfig in fastfusion_if/config.py:
#         surface_feature_set: str = "basic"   # "basic" or "rich"
#         burial_radius: float = 10.0
#         shape_k_neighbors: int = 16
#   2) in fastfusion_if/data/surface.py import this function and dispatch:
#         from .surface_features_rich import surface_scalar_features_rich
#         ...
#         if getattr(cfg, "surface_feature_set", "basic") == "rich":
#             return surface_scalar_features_rich(atoms, surface_pos, normals,
#                                                 surface2atom, cfg)
#   3) regenerate the cache (the input dim changes, so this is a NEW experiment;
#      old v2 checkpoints expect 4-dim surface features and will not load).
# ---------------------------------------------------------------------------

RICH_SURFACE_FEATURE_DIM = 10

# N-type / O-type atom-name prefixes used as cheap H-bond donor/acceptor flags.
_DONOR_PREFIX = ("N",)
_ACCEPTOR_PREFIX = ("O",)
_AROMATIC = {"PHE", "TRP", "TYR", "HIS"}


def _owner_atom_chem(atoms: list[AtomRecord], surface2atom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    donor = np.zeros((surface2atom.shape[0],), dtype=np.float32)
    acceptor = np.zeros((surface2atom.shape[0],), dtype=np.float32)
    for i, owner in enumerate(surface2atom):
        elem = str(atoms[int(owner)].element).upper()
        if elem.startswith(_DONOR_PREFIX):
            donor[i] = 1.0
        elif elem.startswith(_ACCEPTOR_PREFIX):
            acceptor[i] = 1.0
    return donor, acceptor


def surface_scalar_features_rich(
    atoms: list[AtomRecord],
    surface_pos: np.ndarray,
    normals: np.ndarray,
    surface2atom: np.ndarray,
    cfg: DataConfig,
) -> tuple[np.ndarray, np.ndarray]:
    atom2res, _, _ = residue_table(atoms)
    if surface_pos.shape[0] == 0:
        n_feat = RICH_SURFACE_FEATURE_DIM + (3 if cfg.use_surface_normals_as_features else 0)
        return np.zeros((0, n_feat), np.float32), np.zeros((0,), np.int64)

    coords = np.stack([a.coord for a in atoms]).astype(np.float32)
    res_charges = np.asarray([residue_charge(a.res_name) for a in atoms], dtype=np.float32)
    atom_tree = cKDTree(coords)
    surf_tree = cKDTree(surface_pos)

    hydro = np.asarray(
        [hydropathy(atoms[int(o)].res_name) for o in surface2atom], dtype=np.float32
    )
    aromatic = np.asarray(
        [1.0 if str(atoms[int(o)].res_name).upper() in _AROMATIC else 0.0 for o in surface2atom],
        dtype=np.float32,
    )
    surface2res = atom2res[surface2atom]

    # Electrostatics proxy (truncated Coulomb over nearby residue charges).
    neigh = atom_tree.query_ball_point(surface_pos, r=8.0)
    electro = np.zeros((surface_pos.shape[0],), dtype=np.float32)
    burial = np.zeros((surface_pos.shape[0],), dtype=np.float32)
    burial_r = float(getattr(cfg, "burial_radius", 10.0))
    burial_counts = atom_tree.query_ball_point(surface_pos, r=burial_r, return_length=True)
    for i, ids in enumerate(neigh):
        if ids:
            d = np.linalg.norm(coords[ids] - surface_pos[i], axis=1)
            electro[i] = float(np.sum(res_charges[ids] / np.maximum(d, 1.0)))
    electro = np.tanh(electro).astype(np.float32)
    burial = (np.log1p(burial_counts.astype(np.float32)) / np.log1p(64.0)).astype(np.float32)
    burial = np.clip(burial, 0.0, 1.0)

    # Curvature (your original normal-variance proxy) + PCA shape descriptors.
    k = int(getattr(cfg, "shape_k_neighbors", 16))
    curvature = np.zeros((surface_pos.shape[0],), dtype=np.float32)
    planarity = np.zeros((surface_pos.shape[0],), dtype=np.float32)
    curvedness = np.zeros((surface_pos.shape[0],), dtype=np.float32)
    kq = min(k, surface_pos.shape[0])
    _, knn_idx = surf_tree.query(surface_pos, k=kq)
    if knn_idx.ndim == 1:
        knn_idx = knn_idx[:, None]
    for i in range(surface_pos.shape[0]):
        ids = knn_idx[i]
        if normals.shape[0] == surface_pos.shape[0] and len(ids) > 1:
            mean_n = normals[ids].mean(axis=0)
            curvature[i] = float(np.mean(np.sum((normals[ids] - mean_n) ** 2, axis=1)))
        if len(ids) >= 3:
            pts = surface_pos[ids]
            pts = pts - pts.mean(axis=0, keepdims=True)
            cov = (pts.T @ pts) / float(len(ids))
            ev = np.linalg.eigvalsh(cov)  # ascending
            l3, l2, l1 = float(ev[0]), float(ev[1]), float(ev[2])
            s = l1 + l2 + l3 + 1e-6
            planarity[i] = (l2 - l3) / (l1 + 1e-6)
            curvedness[i] = l3 / s
    curvature = np.clip(curvature, 0.0, 1.0).astype(np.float32)
    planarity = np.clip(planarity, 0.0, 1.0).astype(np.float32)
    curvedness = np.clip(curvedness, 0.0, 1.0).astype(np.float32)

    # Exposure proxy (your original): normalised distance from owner atom.
    owner_coords = coords[surface2atom]
    exposure = np.linalg.norm(surface_pos - owner_coords, axis=1).astype(np.float32)
    exposure = (exposure - exposure.mean()) / (exposure.std() + 1e-6)
    exposure = np.tanh(exposure).astype(np.float32)

    donor, acceptor = _owner_atom_chem(atoms, surface2atom)

    feats = [
        hydro[:, None],
        electro[:, None],
        curvature[:, None],
        exposure[:, None],
        burial[:, None],
        planarity[:, None],
        curvedness[:, None],
        donor[:, None],
        acceptor[:, None],
        aromatic[:, None],
    ]
    if cfg.use_surface_normals_as_features:
        feats.insert(0, normals.astype(np.float32))
    return np.concatenate(feats, axis=1).astype(np.float32), surface2res.astype(np.int64)
