from __future__ import annotations

import torch
import torch.nn as nn

from .layers import LocalMultiheadAttention, MLP


class CrossModalFusionLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.atom_from_surface = LocalMultiheadAttention(dim, dim, dim, n_heads=n_heads, dropout=dropout)
        self.surface_from_atom = LocalMultiheadAttention(dim, dim, dim, n_heads=n_heads, dropout=dropout)
        self.atom_norm1 = nn.LayerNorm(dim)
        self.surface_norm1 = nn.LayerNorm(dim)
        self.atom_ff = MLP([dim, dim * 4, dim], dropout=dropout)
        self.surface_ff = MLP([dim, dim * 4, dim], dropout=dropout)
        self.atom_norm2 = nn.LayerNorm(dim)
        self.surface_norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        atom_h: torch.Tensor,
        surface_h: torch.Tensor,
        atom_pos: torch.Tensor,
        surface_pos: torch.Tensor,
        atom_query_surface_key: torch.Tensor,
        surface_query_atom_key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atom_ctx = self.atom_from_surface(atom_h, surface_h, atom_pos, surface_pos, atom_query_surface_key)
        surf_ctx = self.surface_from_atom(surface_h, atom_h, surface_pos, atom_pos, surface_query_atom_key)
        atom_h = self.atom_norm1(atom_h + self.dropout(atom_ctx))
        surface_h = self.surface_norm1(surface_h + self.dropout(surf_ctx))
        atom_h = self.atom_norm2(atom_h + self.dropout(self.atom_ff(atom_h)))
        surface_h = self.surface_norm2(surface_h + self.dropout(self.surface_ff(surface_h)))
        return atom_h, surface_h


class CrossModalFusion(nn.Module):
    def __init__(self, atom_dim: int, surface_dim: int, fusion_dim: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.atom_proj = nn.Linear(atom_dim, fusion_dim)
        self.surface_proj = nn.Linear(surface_dim, fusion_dim)
        self.layers = nn.ModuleList([CrossModalFusionLayer(fusion_dim, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)])

    def forward(
        self,
        atom_h: torch.Tensor,
        surface_h: torch.Tensor,
        atom_pos: torch.Tensor,
        surface_pos: torch.Tensor,
        atom_query_surface_key: torch.Tensor,
        surface_query_atom_key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atom_h = self.atom_proj(atom_h)
        surface_h = self.surface_proj(surface_h)
        for layer in self.layers:
            atom_h, surface_h = layer(atom_h, surface_h, atom_pos, surface_pos, atom_query_surface_key, surface_query_atom_key)
        return atom_h, surface_h
