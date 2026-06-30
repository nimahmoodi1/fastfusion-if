from __future__ import annotations

import torch
import torch.nn as nn

from .layers import LocalTransformerBlock


class ResidueContextEncoder(nn.Module):
    """Local transformer over residue-level tokens.

    It refines pooled atom/surface residue embeddings using spatial residue
    neighborhoods and sequence-neighbor edges. This encourages contiguous
    interface patches without hard post-processing.
    """

    def __init__(self, dim: int, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LocalTransformerBlock(dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, h: torch.Tensor, residue_pos: torch.Tensor, residue_edge_index: torch.Tensor) -> torch.Tensor:
        if h.numel() == 0 or len(self.layers) == 0:
            return h

        if residue_edge_index.numel() > 0:
            # LocalTransformerBlock expects [query, key], while residue_edge_index is [source, target].
            edge_qk = torch.stack([residue_edge_index[1], residue_edge_index[0]], dim=0)
        else:
            edge_qk = residue_edge_index

        for layer in self.layers:
            h = layer(h, residue_pos, edge_qk)

        return h
