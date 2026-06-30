from __future__ import annotations

import torch
import torch.nn as nn

from ..constants import ELEMENTS, UNKNOWN_ELEMENT_INDEX
from ..geometry import scatter_mean, scatter_sum
from .layers import MLP


class EGNNLayer(nn.Module):
    """A practical E(n)-equivariant message passing layer for atomic coordinates."""

    def __init__(self, dim: int, dropout: float = 0.1, update_coords: bool = True):
        super().__init__()
        self.update_coords = update_coords
        self.edge_mlp = MLP([dim * 2 + 1, dim, dim], dropout=dropout)
        self.node_mlp = MLP([dim * 2, dim, dim], dropout=dropout)
        self.coord_mlp = MLP([dim, dim, 1], dropout=dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return h, x
        src, dst = edge_index
        diff = x[src] - x[dst]
        d2 = (diff ** 2).sum(dim=-1, keepdim=True)
        msg = self.edge_mlp(torch.cat([h[src], h[dst], d2], dim=-1))
        agg = scatter_sum(msg, dst, h.size(0))
        deg = torch.zeros((h.size(0), 1), device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones((msg.size(0), 1), device=h.device, dtype=h.dtype))
        agg = agg / deg.clamp_min(1.0)
        h = self.norm(h + self.node_mlp(torch.cat([h, agg], dim=-1)))

        if self.update_coords:
            scale = torch.tanh(self.coord_mlp(msg))
            dx = diff * scale
            dx_agg = scatter_sum(dx, dst, x.size(0)) / deg.clamp_min(1.0)
            x = x + 0.1 * dx_agg
        return h, x


class AtomEncoder(nn.Module):
    def __init__(self, dim: int, n_layers: int, dropout: float, update_coords: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(len(ELEMENTS) + 1, dim)
        self.layers = nn.ModuleList([EGNNLayer(dim, dropout=dropout, update_coords=update_coords) for _ in range(n_layers)])

    def forward(self, atom_elem: torch.Tensor, atom_pos: torch.Tensor, atom_edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.embedding(atom_elem.clamp(min=0, max=len(ELEMENTS)))
        x = atom_pos
        for layer in self.layers:
            h, x = layer(h, x, atom_edge_index)
        return h, x
