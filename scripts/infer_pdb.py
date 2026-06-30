#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd
import torch
from Bio.PDB import PDBIO, PDBParser

from fastfusion_if.config import DataConfig, ModelConfig
from fastfusion_if.data.collate import collate_chain_examples
from fastfusion_if.data.dataset import make_chain_example, parse_any_atoms
from fastfusion_if.data.pdb_parser import atoms_by_chain
from fastfusion_if.models import FastFusionIF


def write_bfactor_pdb(input_pdb: str, output_pdb: str, chain_id: str, scores: dict[tuple[str, int, str], float]) -> None:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pred", input_pdb)
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                key = (chain.id, int(res.id[1]), str(res.id[2]).strip())
                value = float(scores.get(key, 0.0)) * 100.0
                for atom in res:
                    atom.set_bfactor(value)
        break
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict interface probabilities for one PDB chain")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--out-prefix", default="prediction")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw_cfg = ckpt.get("cfg", {})
    data_cfg = DataConfig(**raw_cfg.get("data", {}))
    model_cfg = ModelConfig(**raw_cfg.get("model", {}))
    surface_feature_dim = int(ckpt.get("surface_feature_dim", 4))

    atoms = parse_any_atoms(args.pdb, data_cfg)
    chains = atoms_by_chain(atoms)
    if args.chain not in chains:
        raise ValueError(f"Chain {args.chain!r} not found. Available chains: {sorted(chains)}")
    ex = make_chain_example(chains[args.chain], atoms, data_cfg, source_path=args.pdb, with_labels=False)
    batch = collate_chain_examples([[ex]])

    model = FastFusionIF(model_cfg, surface_feature_dim=surface_feature_dim, residue_feature_dim=int(ckpt.get("residue_feature_dim", 0)), plm_dim=int(ckpt.get("plm_dim", 0)))
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()

    rows = []
    score_map = {}
    for key, resn, prob in zip(ex.residue_keys, ex.residue_names, probs):
        score_map[key] = float(prob)
        rows.append({"chain": key[0], "res_seq": key[1], "insertion": key[2], "res_name": resn, "prob_interface": float(prob)})
    df = pd.DataFrame(rows)
    out_csv = f"{args.out_prefix}_{Path(args.pdb).stem}_{args.chain}.csv"
    df.to_csv(out_csv, index=False)
    print("Wrote", out_csv)

    out_pdb = f"{args.out_prefix}_{Path(args.pdb).stem}_{args.chain}_bfactor.pdb"
    write_bfactor_pdb(args.pdb, out_pdb, args.chain, score_map)
    print("Wrote", out_pdb)


if __name__ == "__main__":
    main()
