#!/usr/bin/env python
"""Fast smoke test for the FastFusion-IF upgrades (run this BEFORE a real run).

It builds a couple of synthetic ChainExamples, runs the real collate, constructs
the model in three modes (baseline / +rich-surface-dim / +PLM), and does a
forward + loss(incl. focal+tversky) + backward step on each. It needs torch but
NOT the dataset, so it isolates the model/loss/collate wiring.

    python scripts/smoke_test_upgrades.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from fastfusion_if.config import ExperimentConfig
from fastfusion_if.data.collate import collate_chain_examples
from fastfusion_if.data.structures import ChainExample
from fastfusion_if.losses import interface_loss
from fastfusion_if.models import FastFusionIF


def synth_example(n_res=20, atoms_per_res=6, surf_dim=4, plm_dim=0, seed=0) -> ChainExample:
    rng = np.random.default_rng(seed)
    n_atoms = n_res * atoms_per_res
    atom_pos = rng.normal(0, 10, size=(n_atoms, 3)).astype(np.float32)
    atom_elem = rng.integers(0, 5, size=n_atoms).astype(np.int64)
    atom2res = np.repeat(np.arange(n_res), atoms_per_res).astype(np.int64)
    residue_pos = rng.normal(0, 10, size=(n_res, 3)).astype(np.float32)
    # simple residue chain graph
    src = np.arange(n_res - 1); dst = src + 1
    residue_edge_index = np.concatenate(
        [np.stack([src, dst]), np.stack([dst, src])], axis=1
    ).astype(np.int64)
    n_surf = 120
    surface_pos = rng.normal(0, 10, size=(n_surf, 3)).astype(np.float32)
    surface_features = rng.normal(size=(n_surf, surf_dim)).astype(np.float32)
    surface2res = rng.integers(0, n_res, size=n_surf).astype(np.int64)
    # a few cross / self edges
    ae = np.stack([rng.integers(0, n_atoms, 200), rng.integers(0, n_atoms, 200)]).astype(np.int64)
    se = np.stack([rng.integers(0, n_surf, 200), rng.integers(0, n_surf, 200)]).astype(np.int64)
    aqsk = np.stack([rng.integers(0, n_atoms, 200), rng.integers(0, n_surf, 200)]).astype(np.int64)
    sqak = np.stack([rng.integers(0, n_surf, 200), rng.integers(0, n_atoms, 200)]).astype(np.int64)
    labels = (rng.random(n_res) < 0.2).astype(np.float32)
    plm = rng.normal(size=(n_res, plm_dim)).astype(np.float32) if plm_dim > 0 else None
    return ChainExample(
        atom_pos=atom_pos, atom_elem=atom_elem, atom2res=atom2res,
        residue_keys=[("A", i + 1, "") for i in range(n_res)],
        residue_names=["ALA"] * n_res, labels=labels,
        residue_pos=residue_pos, residue_edge_index=residue_edge_index,
        residue_features=np.zeros((n_res, 36), dtype=np.float32),
        surface_pos=surface_pos, surface_features=surface_features, surface2res=surface2res,
        atom_edge_index=ae, surface_edge_index=se,
        atom_query_surface_key=aqsk, surface_query_atom_key=sqak,
        source_path="synthetic", chain_id="A", residue_plm=plm,
    )


def run_case(name, cfg, surf_dim, plm_dim):
    batch = collate_chain_examples([[synth_example(surf_dim=surf_dim, plm_dim=plm_dim, seed=1)],
                                    [synth_example(surf_dim=surf_dim, plm_dim=plm_dim, seed=2)]])
    model = FastFusionIF(cfg.model, surface_feature_dim=surf_dim, residue_feature_dim=0, plm_dim=plm_dim)
    n_params = sum(p.numel() for p in model.parameters())
    logits = model(batch)
    y = batch["y"]
    loss = interface_loss(
        logits, y,
        positive_weight=5.0, dice_weight=0.2,
        focal_weight=0.5, focal_gamma=2.0, focal_alpha=0.25,
        tversky_weight=0.5, tversky_alpha=0.7, tversky_beta=0.3, tversky_gamma=1.0,
    )
    loss.backward()
    grad_ok = all(
        (p.grad is None) or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert logits.shape[0] == y.shape[0], "logit/label length mismatch"
    assert torch.isfinite(loss), "non-finite loss"
    print(f"[OK] {name:28s} params={n_params/1e6:.3f}M  logits={tuple(logits.shape)}  "
          f"loss={float(loss):.4f}  grads_finite={grad_ok}")


def main():
    torch.manual_seed(0)
    base = ExperimentConfig()
    base.model.use_residue_context = True
    base.model.n_residue_layers = 2

    run_case("baseline (basic surf=4)", base, surf_dim=4, plm_dim=0)
    run_case("rich surface (surf=10)", base, surf_dim=10, plm_dim=0)

    esm = ExperimentConfig()
    esm.model.use_residue_context = True
    esm.model.use_plm_features = True
    esm.model.plm_dropout = 0.1
    esm.model.plm_inject = "concat"
    run_case("+PLM concat (esm 1280-d)", esm, surf_dim=4, plm_dim=1280)
    run_case("+PLM concat (esm 640-d)", esm, surf_dim=10, plm_dim=640)

    esm_add = ExperimentConfig()
    esm_add.model.use_residue_context = True
    esm_add.model.use_plm_features = True
    esm_add.model.plm_inject = "add"
    run_case("+PLM add (esm 1280-d)", esm_add, surf_dim=4, plm_dim=1280)

    print("\nSMOKE TEST PASSED — model/loss/collate/PLM wiring is consistent.")


if __name__ == "__main__":
    main()
