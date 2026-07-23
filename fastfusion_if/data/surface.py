from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from ..config import DataConfig
from .features import hydropathy, residue_charge, vdw_radius
from .structures import AtomRecord
from .labels import residue_table


def fibonacci_sphere(n: int) -> np.ndarray:
    """Approximately uniform directions on a unit sphere."""
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    i = np.arange(n, dtype=np.float32)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - (i / max(1, n - 1)) * 2.0
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * i
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.stack([x, y, z], axis=1).astype(np.float32)


def farthest_point_subsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """Deterministic farthest-point subsampling indices. Falls back to all points if small."""
    n = points.shape[0]
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = np.empty((max_points,), dtype=np.int64)
    selected[0] = int(rng.integers(0, n))
    dist2 = np.full((n,), np.inf, dtype=np.float32)
    for i in range(1, max_points):
        last = points[selected[i - 1]]
        dist2 = np.minimum(dist2, np.sum((points - last) ** 2, axis=1))
        selected[i] = int(np.argmax(dist2))
    return selected


def sample_surface_points(atoms: list[AtomRecord], cfg: DataConfig, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a mesh-free solvent-accessible surface point cloud.

    The algorithm samples candidates on expanded atom spheres and prunes candidates that are inside another
    expanded atom. It is much faster than a triangulated MSMS-style mesh and is adequate for a clean first
    implementation of FastFusion-IF. Production experiments can swap this with a dMaSIF-style differentiable
    sampler without changing the model API.

    Returns:
        surface_pos: [S, 3]
        normals: [S, 3]
        surface2atom: [S]
    """
    if not atoms:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0,), np.int64)

    coords = np.stack([a.coord for a in atoms]).astype(np.float32)
    radii = np.asarray([vdw_radius(a.element) + cfg.probe_radius for a in atoms], dtype=np.float32)
    tree = cKDTree(coords)
    dirs = fibonacci_sphere(cfg.n_surface_dirs)

    pts: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    owners: list[int] = []
    max_r = float(radii.max()) if len(radii) else cfg.probe_radius + 2.0
    for atom_i, atom in enumerate(atoms):
        cand = atom.coord[None, :] + radii[atom_i] * dirs
        near_lists = tree.query_ball_point(cand, r=max_r + 0.5)
        for j, near in enumerate(near_lists):
            keep = True
            p = cand[j]
            for nb in near:
                if nb == atom_i:
                    continue
                # If the candidate lies well inside another expanded atom, it is buried.
                if np.linalg.norm(p - coords[nb]) < radii[nb] - 0.05:
                    keep = False
                    break
            if keep:
                pts.append(p.astype(np.float32))
                normals.append(dirs[j].astype(np.float32))
                owners.append(atom_i)

    if not pts:
        # Degenerate fallback: use C-alpha/atom positions as pseudo-surface points.
        pts = [a.coord.astype(np.float32) for a in atoms]
        normals = [np.array([1.0, 0.0, 0.0], dtype=np.float32) for _ in atoms]
        owners = list(range(len(atoms)))

    surface_pos = np.stack(pts).astype(np.float32)
    normal_arr = np.stack(normals).astype(np.float32)
    owner_arr = np.asarray(owners, dtype=np.int64)

    if surface_pos.shape[0] > cfg.max_surface_points:
        ids = farthest_point_subsample(surface_pos, cfg.max_surface_points, seed=seed)
        surface_pos = surface_pos[ids]
        normal_arr = normal_arr[ids]
        owner_arr = owner_arr[ids]

    return surface_pos, normal_arr, owner_arr


def surface_scalar_features(
    atoms: list[AtomRecord],
    surface_pos: np.ndarray,
    normals: np.ndarray,
    surface2atom: np.ndarray,
    cfg: DataConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-surface-point features and assign surface points to residues."""
    # Optional richer descriptor (burial, PCA shape, H-bond donor/acceptor, ...).
    if getattr(cfg, "surface_feature_set", "basic") == "rich":
        from .surface_features_rich import surface_scalar_features_rich

        return surface_scalar_features_rich(atoms, surface_pos, normals, surface2atom, cfg)

    atom2res, _, residue_names = residue_table(atoms)
    if surface_pos.shape[0] == 0:
        n_feat = 4 + (3 if cfg.use_surface_normals_as_features else 0)
        return np.zeros((0, n_feat), np.float32), np.zeros((0,), np.int64)

    coords = np.stack([a.coord for a in atoms]).astype(np.float32)
    res_charges = np.asarray([residue_charge(a.res_name) for a in atoms], dtype=np.float32)
    tree = cKDTree(coords)

    hydro = np.asarray([hydropathy(atoms[int(owner)].res_name) for owner in surface2atom], dtype=np.float32)
    surface2res = atom2res[surface2atom]

    # Local truncated Coulomb-like electrostatics proxy.
    neigh = tree.query_ball_point(surface_pos, r=8.0)
    electro = np.zeros((surface_pos.shape[0],), dtype=np.float32)
    for i, ids in enumerate(neigh):
        if not ids:
            continue
        d = np.linalg.norm(coords[ids] - surface_pos[i], axis=1)
        q = res_charges[ids]
        electro[i] = np.sum(q / np.maximum(d, 1.0))
    electro = np.tanh(electro).astype(np.float32)

    # Curvature proxy: variance of neighbor normals around each point.
    surf_tree = cKDTree(surface_pos)
    neigh_s = surf_tree.query_ball_point(surface_pos, r=4.0)
    curvature = np.zeros((surface_pos.shape[0],), dtype=np.float32)
    for i, ids in enumerate(neigh_s):
        if len(ids) > 1:
            mean_n = normals[ids].mean(axis=0)
            curvature[i] = float(np.mean(np.sum((normals[ids] - mean_n) ** 2, axis=1)))
    curvature = np.clip(curvature, 0.0, 1.0).astype(np.float32)

    # Exposure proxy: candidates kept after pruning tend to be exposed; use distance from owner atom / radius scale.
    owner_coords = coords[surface2atom]
    exposure = np.linalg.norm(surface_pos - owner_coords, axis=1).astype(np.float32)
    exposure = (exposure - exposure.mean()) / (exposure.std() + 1e-6)
    exposure = np.tanh(exposure).astype(np.float32)

    feats = [hydro[:, None], electro[:, None], curvature[:, None], exposure[:, None]]
    if cfg.use_surface_normals_as_features:
        feats.insert(0, normals.astype(np.float32))
    return np.concatenate(feats, axis=1).astype(np.float32), surface2res.astype(np.int64)
