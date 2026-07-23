from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .structures import ChainExample
from .residue_features import RESIDUE_FEATURE_DIM


def _cat_edges(edge_list: list[np.ndarray], q_offsets: list[int], k_offsets: list[int]) -> torch.Tensor:
    parts = []
    for edge, qo, ko in zip(edge_list, q_offsets, k_offsets):
        if edge.size == 0:
            continue
        e = edge.copy()
        e[0] += qo
        e[1] += ko
        parts.append(torch.as_tensor(e, dtype=torch.long))
    if not parts:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.cat(parts, dim=1)


def collate_chain_examples(batch: list[list[ChainExample]]) -> dict[str, Any] | None:
    examples: list[ChainExample] = []
    for item in batch:
        examples.extend(item)
    if not examples:
        return None

    atom_offsets = []
    surface_offsets = []
    residue_offsets = []
    a_off = s_off = r_off = 0
    for ex in examples:
        atom_offsets.append(a_off)
        surface_offsets.append(s_off)
        residue_offsets.append(r_off)
        a_off += ex.n_atoms
        s_off += ex.n_surface_points
        r_off += ex.n_residues

    atom_pos = torch.as_tensor(np.concatenate([ex.atom_pos for ex in examples], axis=0), dtype=torch.float32)
    atom_elem = torch.as_tensor(np.concatenate([ex.atom_elem for ex in examples], axis=0), dtype=torch.long)
    atom2res = torch.as_tensor(
        np.concatenate([ex.atom2res + ro for ex, ro in zip(examples, residue_offsets)], axis=0), dtype=torch.long
    )
    residue_pos = torch.as_tensor(
        np.concatenate([ex.residue_pos for ex in examples], axis=0),
        dtype=torch.float32,
    )

    if sum(ex.n_surface_points for ex in examples) > 0:
        surface_pos = torch.as_tensor(np.concatenate([ex.surface_pos for ex in examples], axis=0), dtype=torch.float32)
        surface_features = torch.as_tensor(np.concatenate([ex.surface_features for ex in examples], axis=0), dtype=torch.float32)
        surface2res = torch.as_tensor(
            np.concatenate([ex.surface2res + ro for ex, ro in zip(examples, residue_offsets)], axis=0), dtype=torch.long
        )
    else:
        feat_dim = examples[0].surface_features.shape[1] if examples[0].surface_features.ndim == 2 else 4
        surface_pos = torch.zeros((0, 3), dtype=torch.float32)
        surface_features = torch.zeros((0, feat_dim), dtype=torch.float32)
        surface2res = torch.zeros((0,), dtype=torch.long)

    atom_edge_index = _cat_edges([ex.atom_edge_index for ex in examples], atom_offsets, atom_offsets)
    surface_edge_index = _cat_edges([ex.surface_edge_index for ex in examples], surface_offsets, surface_offsets)
    residue_edge_index = _cat_edges([ex.residue_edge_index for ex in examples], residue_offsets, residue_offsets)
    atom_query_surface_key = _cat_edges([ex.atom_query_surface_key for ex in examples], atom_offsets, surface_offsets)
    surface_query_atom_key = _cat_edges([ex.surface_query_atom_key for ex in examples], surface_offsets, atom_offsets)

    if all(ex.labels is not None for ex in examples):
        y = torch.as_tensor(np.concatenate([ex.labels for ex in examples], axis=0), dtype=torch.float32)
    else:
        y = None

    res_batch = []
    atom_batch = []
    surface_batch = []
    residue_feature_arrays = []
    for ex in examples:
        rf = getattr(ex, "residue_features", None)
        if rf is None:
            rf = np.zeros((ex.n_residues, RESIDUE_FEATURE_DIM), dtype=np.float32)
        residue_feature_arrays.append(rf)

    residue_features = torch.as_tensor(
        np.concatenate(residue_feature_arrays, axis=0),
        dtype=torch.float32,
    )

    # Optional protein-language-model residue embeddings. Only emitted when every
    # example in the batch carries them (i.e. the cache was built with --plm-model).
    residue_plm = None
    if all(getattr(ex, "residue_plm", None) is not None for ex in examples):
        plm_arrays = [np.asarray(ex.residue_plm, dtype=np.float32) for ex in examples]
        if all(a.ndim == 2 and a.shape[0] == ex.n_residues for a, ex in zip(plm_arrays, examples)):
            residue_plm = torch.as_tensor(np.concatenate(plm_arrays, axis=0), dtype=torch.float32)

    metadata = []
    for batch_i, ex in enumerate(examples):
        res_batch.append(torch.full((ex.n_residues,), batch_i, dtype=torch.long))
        atom_batch.append(torch.full((ex.n_atoms,), batch_i, dtype=torch.long))
        surface_batch.append(torch.full((ex.n_surface_points,), batch_i, dtype=torch.long))
        metadata.append(
            {
                "source_path": ex.source_path,
                "chain_id": ex.chain_id,
                "residue_keys": ex.residue_keys,
                "residue_names": ex.residue_names,
            }
        )

    return {
        "atom_pos": atom_pos,
        "atom_elem": atom_elem,
        "atom2res": atom2res,
        "residue_pos": residue_pos,
        "residue_features": residue_features,
        "residue_plm": residue_plm,
        "surface_pos": surface_pos,
        "surface_features": surface_features,
        "surface2res": surface2res,
        "atom_edge_index": atom_edge_index,
        "surface_edge_index": surface_edge_index,
        "residue_edge_index": residue_edge_index,
        "atom_query_surface_key": atom_query_surface_key,
        "surface_query_atom_key": surface_query_atom_key,
        "y": y,
        "n_residues": r_off,
        "res_batch": torch.cat(res_batch, dim=0),
        "atom_batch": torch.cat(atom_batch, dim=0),
        "surface_batch": torch.cat(surface_batch, dim=0) if surface_batch else torch.zeros((0,), dtype=torch.long),
        "metadata": metadata,
    }
