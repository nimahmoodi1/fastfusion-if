from __future__ import annotations

import torch
import torch.nn as nn

from .layers import LocalTransformerBlock, MLP


class SurfaceEncoder(nn.Module):
    """Local point-cloud transformer over mesh-free surface points."""

    def __init__(self, in_dim: int, dim: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.input = MLP([in_dim, dim, dim], dropout=dropout)
        self.layers = nn.ModuleList([LocalTransformerBlock(dim, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)])

    def forward(self, surface_features: torch.Tensor, surface_pos: torch.Tensor, surface_edge_index: torch.Tensor) -> torch.Tensor:
        if surface_features.size(0) == 0:
            return torch.zeros((0, self.input.net[-1].out_features), device=surface_features.device, dtype=surface_features.dtype)
        h = self.input(surface_features)
        # LocalTransformerBlock expects query-key edges, while surface_edge_index is source-target.
        if surface_edge_index.numel() > 0:
            edge_qk = torch.stack([surface_edge_index[1], surface_edge_index[0]], dim=0)
        else:
            edge_qk = surface_edge_index
        for layer in self.layers:
            h = layer(h, surface_pos, edge_qk)
        return h
