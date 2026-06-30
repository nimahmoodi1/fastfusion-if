from __future__ import annotations

import torch
import torch.nn as nn

from ..geometry import scatter_mean, scatter_sum, segment_softmax


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
        if h.numel() == 0:
            return torch.zeros((dim_size, self.score.in_features), device=h.device, dtype=h.dtype)
        scores = self.score(h).squeeze(-1)
        weights = segment_softmax(scores, index, dim_size).unsqueeze(-1)
        return scatter_sum(h * weights, index, dim_size)


class ResidueFusionDecoder(nn.Module):
    def __init__(self, dim: int, pooling: str = "attention", dropout: float = 0.1):
        super().__init__()
        if pooling not in {"mean", "attention"}:
            raise ValueError("pooling must be 'mean' or 'attention'")
        self.pooling = pooling
        self.atom_pool = AttentionPool(dim) if pooling == "attention" else None
        self.surface_pool = AttentionPool(dim) if pooling == "attention" else None
        self.gate = nn.Sequential(nn.Linear(dim * 4, dim), nn.SiLU(), nn.Linear(dim, dim), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def _pool(self, h: torch.Tensor, index: torch.Tensor, dim_size: int, which: str) -> torch.Tensor:
        if self.pooling == "mean":
            return scatter_mean(h, index, dim_size) if h.numel() > 0 else torch.zeros((dim_size, self.head[1].in_features), device=index.device)
        pool = self.atom_pool if which == "atom" else self.surface_pool
        return pool(h, index, dim_size)

    def fuse_residues(
        self,
        atom_h: torch.Tensor,
        surface_h: torch.Tensor,
        atom2res: torch.Tensor,
        surface2res: torch.Tensor,
        n_residues: int,
    ) -> torch.Tensor:
        atom_res = self._pool(atom_h, atom2res, n_residues, "atom")
        if surface_h.numel() == 0 or surface2res.numel() == 0:
            surface_res = torch.zeros_like(atom_res)
        else:
            surface_res = self._pool(surface_h, surface2res, n_residues, "surface")

        joined = torch.cat(
            [atom_res, surface_res, torch.abs(atom_res - surface_res), atom_res * surface_res],
            dim=-1,
        )
        gate = self.gate(joined)
        return gate * atom_res + (1.0 - gate) * surface_res

    def classify(self, residue_h: torch.Tensor) -> torch.Tensor:
        return self.head(residue_h).squeeze(-1)

    def forward(self, atom_h: torch.Tensor, surface_h: torch.Tensor, atom2res: torch.Tensor, surface2res: torch.Tensor,
                n_residues: int) -> torch.Tensor:
        fused = self.fuse_residues(atom_h, surface_h, atom2res, surface2res, n_residues)
        return self.classify(fused)
