from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .atom_encoder import AtomEncoder
from .fusion import CrossModalFusion
from .pooling import ResidueFusionDecoder
from .residue_context import ResidueContextEncoder
from .surface_encoder import SurfaceEncoder


class FastFusionIF(nn.Module):
    """FastFusion-IF: atom encoder + mesh-free surface encoder + local cross-modal fusion."""

    def __init__(self, cfg: ModelConfig, surface_feature_dim: int, residue_feature_dim: int = 0, plm_dim: int = 0):
        super().__init__()
        self.cfg = cfg
        self.use_surface = bool(getattr(cfg, "use_surface", True))
        self.atom_encoder = AtomEncoder(
            dim=cfg.atom_dim,
            n_layers=cfg.n_atom_layers,
            dropout=cfg.dropout,
            update_coords=cfg.use_coordinate_updates,
        )
        self.surface_encoder = SurfaceEncoder(
            in_dim=surface_feature_dim,
            dim=cfg.surface_dim,
            n_layers=cfg.n_surface_layers,
            n_heads=cfg.n_attention_heads,
            dropout=cfg.dropout,
        )
        self.fusion = CrossModalFusion(
            atom_dim=cfg.atom_dim,
            surface_dim=cfg.surface_dim,
            fusion_dim=cfg.fusion_dim,
            n_layers=cfg.n_fusion_layers,
            n_heads=cfg.n_attention_heads,
            dropout=cfg.dropout,
        )
        self.decoder = ResidueFusionDecoder(dim=cfg.fusion_dim, pooling=cfg.residue_pooling, dropout=cfg.dropout)
        self.residue_feature_dim = int(residue_feature_dim or 0)
        self.residue_feature_scale = float(getattr(cfg, "residue_feature_scale", 0.25))
        self.residue_feature_mlp = (
            nn.Sequential(
                nn.LayerNorm(self.residue_feature_dim),
                nn.Linear(self.residue_feature_dim, cfg.fusion_dim),
                nn.SiLU(),
                nn.Dropout(cfg.residue_feature_dropout),
                nn.Linear(cfg.fusion_dim, cfg.fusion_dim),
            )
            if cfg.use_residue_features and self.residue_feature_dim > 0
            else None
        )
        if self.residue_feature_mlp is not None:
            # Small NON-zero init so the evolutionary/structural feature pathway is
            # active from step 1. A zero-init final layer starts as a no-op and
            # (like the PLM path before its fix) can sit on a dead saddle; a small
            # init keeps it well-scaled yet immediately trainable.
            nn.init.normal_(self.residue_feature_mlp[-1].weight, std=0.02)
            nn.init.zeros_(self.residue_feature_mlp[-1].bias)
        self.plm_dim = int(plm_dim or getattr(cfg, "plm_dim", 0) or 0)
        self.plm_inject = str(getattr(cfg, "plm_inject", "concat")).lower()
        self._plm_on = bool(getattr(cfg, "use_plm_features", False)) and self.plm_dim > 0
        # Project ESM-2 embeddings to a fusion-dim "PLM token". LayerNorm on the
        # raw embeddings puts ESM on the same scale as the geometric features so
        # one cannot swamp or vanish relative to the other.
        self.plm_proj = (
            nn.Sequential(
                nn.LayerNorm(self.plm_dim),
                nn.Linear(self.plm_dim, cfg.fusion_dim),
                nn.SiLU(),
                nn.Dropout(getattr(cfg, "plm_dropout", 0.10)),
            )
            if self._plm_on
            else None
        )
        # concat mode: mix [residue_h | plm_h] -> fusion_dim with a LayerNorm for
        # scale control, then a plain Linear (DEFAULT init, immediately active).
        self.plm_combine_norm = nn.LayerNorm(2 * cfg.fusion_dim) if (self._plm_on and self.plm_inject == "concat") else None
        self.plm_combine = (
            nn.Linear(2 * cfg.fusion_dim, cfg.fusion_dim)
            if (self._plm_on and self.plm_inject == "concat")
            else None
        )
        # add mode: residue_h + gate * plm_h, with a small but NON-zero learnable
        # gate so the pathway is active from step 1 (a zero gate is a dead saddle).
        self.plm_gate = (
            nn.Parameter(torch.tensor(0.1))
            if (self._plm_on and self.plm_inject != "concat")
            else None
        )
        self.residue_context = (
            ResidueContextEncoder(
                dim=cfg.fusion_dim,
                n_layers=cfg.n_residue_layers,
                n_heads=cfg.n_attention_heads,
                dropout=cfg.dropout,
            )
            if cfg.use_residue_context and cfg.n_residue_layers > 0
            else None
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        atom_h, atom_pos = self.atom_encoder(batch["atom_elem"], batch["atom_pos"], batch["atom_edge_index"])
        if self.use_surface:
            surface_h = self.surface_encoder(batch["surface_features"], batch["surface_pos"], batch["surface_edge_index"])
            atom_h, surface_h = self.fusion(
                atom_h=atom_h,
                surface_h=surface_h,
                atom_pos=atom_pos,
                surface_pos=batch["surface_pos"],
                atom_query_surface_key=batch["atom_query_surface_key"],
                surface_query_atom_key=batch["surface_query_atom_key"],
            )
        else:
            # Surface-off ablation: no surface encoder, no cross-modal fusion. Project
            # atom features to fusion_dim with the same atom_proj so dims/decoder are
            # unchanged, and hand the decoder an empty surface tensor (its existing
            # empty-surface path zeros the surface contribution).
            atom_h = self.fusion.atom_proj(atom_h)
            surface_h = atom_h.new_zeros((0, atom_h.shape[-1]))
        residue_h = self.decoder.fuse_residues(
            atom_h,
            surface_h,
            batch["atom2res"],
            batch["surface2res"],
            int(batch["n_residues"]),
        )

        if self.residue_feature_mlp is not None and "residue_features" in batch:
            feature_h = self.residue_feature_mlp(batch["residue_features"].to(dtype=residue_h.dtype))
            residue_h = residue_h + self.residue_feature_scale * feature_h

        if self.plm_proj is not None and batch.get("residue_plm", None) is not None:
            plm_h = self.plm_proj(batch["residue_plm"].to(dtype=residue_h.dtype))
            if self.plm_combine is not None:  # concat mode (immediately active)
                mixed = torch.cat([residue_h, plm_h], dim=-1)
                residue_h = self.plm_combine(self.plm_combine_norm(mixed))
            else:  # gated additive mode
                residue_h = residue_h + self.plm_gate * plm_h

        if self.residue_context is not None:
            residue_h = self.residue_context(
                residue_h,
                batch["residue_pos"],
                batch["residue_edge_index"],
            )

        logits = self.decoder.classify(residue_h)
        return logits
