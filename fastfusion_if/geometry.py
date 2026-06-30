from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree


def radius_edges_numpy(query_xyz: np.ndarray, key_xyz: np.ndarray, radius: float, max_neighbors: int | None = None,
                       exclude_self: bool = False) -> np.ndarray:
    """Return edges [2, E] with rows (query_index, key_index)."""
    if len(query_xyz) == 0 or len(key_xyz) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    tree = cKDTree(key_xyz)
    neigh = tree.query_ball_point(query_xyz, r=radius)
    q_list: list[int] = []
    k_list: list[int] = []
    same_array = query_xyz.shape == key_xyz.shape and np.may_share_memory(query_xyz, key_xyz)
    for q, ids in enumerate(neigh):
        if exclude_self and same_array:
            ids = [k for k in ids if k != q]
        if max_neighbors is not None and len(ids) > max_neighbors:
            d2 = np.sum((key_xyz[ids] - query_xyz[q]) ** 2, axis=1)
            order = np.argsort(d2)[:max_neighbors]
            ids = [ids[int(i)] for i in order]
        for k in ids:
            q_list.append(q)
            k_list.append(int(k))
    if not q_list:
        return np.zeros((2, 0), dtype=np.int64)
    return np.stack([np.asarray(q_list, dtype=np.int64), np.asarray(k_list, dtype=np.int64)], axis=0)


def radius_graph_numpy(xyz: np.ndarray, radius: float, max_neighbors: int | None = None) -> np.ndarray:
    """Return source->target graph edges [2, E] for message passing."""
    qk = radius_edges_numpy(xyz, xyz, radius, max_neighbors=max_neighbors, exclude_self=True)
    # qk is target(query), source(key). Convert to source, target.
    return np.stack([qk[1], qk[0]], axis=0) if qk.size else qk


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size, src.size(-1)), device=src.device, dtype=src.dtype)
    if src.numel() > 0:
        out.index_add_(0, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = scatter_sum(src, index, dim_size)
    count = torch.zeros((dim_size, 1), device=src.device, dtype=src.dtype)
    if src.numel() > 0:
        count.index_add_(0, index, torch.ones((src.size(0), 1), device=src.device, dtype=src.dtype))
    return out / count.clamp_min(1.0)


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Softmax over entries sharing the same segment index.

    Compute the reduction in float32 even under CUDA AMP, then cast back.
    This avoids fp16/fp32 mismatches in index_add_ during mixed-precision training.
    """
    if scores.numel() == 0:
        return scores

    original_dtype = scores.dtype

    if scores.dim() == 1:
        scores = scores[:, None]
        squeeze = True
    else:
        squeeze = False

    scores_f = scores.float()

    max_per = torch.full(
        (dim_size, scores_f.size(1)),
        -torch.inf,
        device=scores.device,
        dtype=torch.float32,
    )

    if hasattr(max_per, "scatter_reduce_"):
        expanded = index[:, None].expand(-1, scores_f.size(1))
        max_per.scatter_reduce_(0, expanded, scores_f, reduce="amax", include_self=True)
    else:
        for i in range(dim_size):
            mask = index == i
            if mask.any():
                max_per[i] = scores_f[mask].max(dim=0).values

    exp = torch.exp(scores_f - max_per[index])
    denom = torch.zeros_like(max_per)
    denom.index_add_(0, index, exp)

    out = exp / denom[index].clamp_min(1e-8)
    out = out.to(original_dtype)

    return out.squeeze(1) if squeeze else out
