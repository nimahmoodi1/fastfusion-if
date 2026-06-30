from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry import scatter_sum, segment_softmax


class MLP(nn.Module):
    def __init__(self, dims: list[int], dropout: float = 0.0, activation: type[nn.Module] = nn.SiLU):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalMultiheadAttention(nn.Module):
    """Local edge-indexed multi-head attention.

    Edges use [2, E] = (query_index, key_index). No quadratic attention matrix is created.
    """

    def __init__(self, query_dim: int, key_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if out_dim % n_heads != 0:
            raise ValueError("out_dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(query_dim, out_dim)
        self.k_proj = nn.Linear(key_dim, out_dim)
        self.v_proj = nn.Linear(key_dim, out_dim)
        self.dist_bias = MLP([16, out_dim, n_heads], dropout=dropout)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        centers = torch.linspace(0.0, 12.0, 16)
        self.register_buffer("rbf_centers", centers)
        self.rbf_gamma = 0.5

    def _rbf(self, dist: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.rbf_gamma * (dist[:, None] - self.rbf_centers[None, :]) ** 2)

    def forward(
        self,
        query_h: torch.Tensor,
        key_h: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        edge_qk: torch.Tensor,
    ) -> torch.Tensor:
        n_query = query_h.size(0)
        if edge_qk.numel() == 0 or n_query == 0:
            return torch.zeros((n_query, self.n_heads * self.head_dim), device=query_h.device, dtype=query_h.dtype)

        q_idx, k_idx = edge_qk
        q = self.q_proj(query_h).view(-1, self.n_heads, self.head_dim)
        k = self.k_proj(key_h).view(-1, self.n_heads, self.head_dim)
        v = self.v_proj(key_h).view(-1, self.n_heads, self.head_dim)

        rel = query_pos[q_idx] - key_pos[k_idx]
        dist = torch.linalg.norm(rel, dim=-1)
        bias = self.dist_bias(self._rbf(dist))
        scores = (q[q_idx] * k[k_idx]).sum(dim=-1) * self.scale + bias
        attn = segment_softmax(scores, q_idx, n_query)
        attn = self.dropout(attn)
        msg = attn[..., None] * v[k_idx]
        msg = msg.reshape(msg.size(0), -1)
        out = scatter_sum(msg, q_idx, n_query)
        return self.out_proj(out)


class LocalTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = LocalMultiheadAttention(dim, dim, dim, n_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = MLP([dim, dim * 4, dim], dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_qk: torch.Tensor) -> torch.Tensor:
        ctx = self.attn(h, h, pos, pos, edge_qk)
        h = self.norm1(h + self.dropout(ctx))
        h = self.norm2(h + self.dropout(self.ff(h)))
        return h
