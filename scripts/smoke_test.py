#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
torch.set_num_threads(1)

from fastfusion_if.config import DataConfig, ModelConfig
from fastfusion_if.data.collate import collate_chain_examples
from fastfusion_if.data.dataset import make_chain_example
from fastfusion_if.data.structures import AtomRecord
from fastfusion_if.models import FastFusionIF


def synthetic_atoms():
    atoms = []
    for i in range(8):
        atoms.append(AtomRecord(coord=torch.tensor([i * 1.5, 0.0, 0.0]).numpy(), element="C", atom_name="CA", chain_id="A", res_seq=i + 1, insertion="", res_name="ALA"))
    for i in range(8):
        atoms.append(AtomRecord(coord=torch.tensor([i * 1.5, 4.0, 0.0]).numpy(), element="C", atom_name="CA", chain_id="B", res_seq=i + 1, insertion="", res_name="LYS"))
    return atoms


def main():
    data_cfg = DataConfig(n_surface_dirs=4, max_surface_points=64, random_rotation=False)
    atoms = synthetic_atoms()
    chain_a = [a for a in atoms if a.chain_id == "A"]
    ex = make_chain_example(chain_a, atoms, data_cfg, with_labels=True)
    batch = collate_chain_examples([[ex]])
    model = FastFusionIF(ModelConfig(atom_dim=16, surface_dim=16, fusion_dim=16, n_atom_layers=1, n_surface_layers=1, n_fusion_layers=1, n_attention_heads=4), surface_feature_dim=batch["surface_features"].shape[-1])
    logits = model(batch)
    assert logits.shape[0] == ex.n_residues
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["y"])
    loss.backward()
    print("Smoke test OK", {"n_res": ex.n_residues, "n_atoms": ex.n_atoms, "n_surface": ex.n_surface_points, "loss": float(loss.detach())})


if __name__ == "__main__":
    main()
