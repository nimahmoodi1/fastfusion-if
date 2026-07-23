from __future__ import annotations

import numpy as np

from ..config import DataConfig
from ..geometry import radius_edges_numpy, radius_graph_numpy


def build_all_graphs(atom_pos: np.ndarray, surface_pos: np.ndarray, cfg: DataConfig) -> dict[str, np.ndarray]:
    atom_edge = radius_graph_numpy(atom_pos, cfg.atom_edge_radius, cfg.max_atom_neighbors)
    surface_edge = radius_graph_numpy(surface_pos, cfg.surface_edge_radius, cfg.max_surface_neighbors)

    # Query atoms attend to key surface points; query surface points attend to key atoms.
    atom_query_surface_key = radius_edges_numpy(
        query_xyz=atom_pos,
        key_xyz=surface_pos,
        radius=cfg.cross_edge_radius,
        max_neighbors=cfg.max_cross_neighbors,
    )
    surface_query_atom_key = radius_edges_numpy(
        query_xyz=surface_pos,
        key_xyz=atom_pos,
        radius=cfg.cross_edge_radius,
        max_neighbors=cfg.max_cross_neighbors,
    )
    return {
        "atom_edge_index": atom_edge,
        "surface_edge_index": surface_edge,
        "atom_query_surface_key": atom_query_surface_key,
        "surface_query_atom_key": surface_query_atom_key,
    }


def build_residue_graph(residue_pos: np.ndarray, n_residues: int, cfg: DataConfig) -> np.ndarray:
    """Build a residue graph with spatial edges plus sequence-neighbor edges.

    Edge format is [2, E] = (source, target), compatible with message passing.
    Spatial edges connect nearby residue centers. Sequence edges keep local
    backbone continuity even when spatial radius misses neighboring residues.
    """
    if n_residues <= 0:
        return np.zeros((2, 0), dtype=np.int64)

    spatial = radius_graph_numpy(
        residue_pos,
        cfg.residue_edge_radius,
        cfg.max_residue_neighbors,
    )

    if n_residues > 1:
        src = np.arange(n_residues - 1, dtype=np.int64)
        dst = src + 1
        seq = np.concatenate([
            np.stack([src, dst], axis=0),
            np.stack([dst, src], axis=0),
        ], axis=1)
    else:
        seq = np.zeros((2, 0), dtype=np.int64)

    if spatial.size == 0:
        edge = seq
    elif seq.size == 0:
        edge = spatial
    else:
        edge = np.concatenate([spatial, seq], axis=1)

    if edge.size == 0:
        return edge.astype(np.int64)

    edge_t = np.unique(edge.T, axis=0)
    return edge_t.T.astype(np.int64)
