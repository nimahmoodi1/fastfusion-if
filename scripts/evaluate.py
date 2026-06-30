#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import torch
from torch.utils.data import DataLoader

from fastfusion_if.config import ExperimentConfig
from fastfusion_if.data.collate import collate_chain_examples
from fastfusion_if.data.dataset import ProteinInterfaceDataset
from fastfusion_if.data.splits import read_manifest
from fastfusion_if.evaluation import collect_predictions, evaluate_predictions, write_csv
from fastfusion_if.metrics import best_f1_threshold
from fastfusion_if.models import FastFusionIF
from fastfusion_if.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a FastFusion-IF checkpoint on a manifest split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default="eval_fastfusion_if")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold. If omitted and split=val, chosen by best F1; otherwise 0.5.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default=None, help="Evaluate using a precomputed example cache (required if the checkpoint uses PLM features).")
    parser.add_argument("--tta", type=int, default=1, help="Test-time augmentation: average probabilities over N random rotations (1 = off). 8 is a good default for final numbers.")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.from_dict(ckpt["cfg"])
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.cache_dir is not None:
        cfg.data.cache_dir = args.cache_dir

    splits = read_manifest(args.manifest)
    files = splits.get(args.split, [])
    if not files:
        raise RuntimeError(f"No files for split={args.split!r} in {args.manifest}")

    if getattr(cfg.data, "cache_dir", None):
        from fastfusion_if.data.cached_dataset import CachedInterfaceDataset

        ds = CachedInterfaceDataset.from_manifest_split(files, cfg.data.cache_dir, cfg.data, augment=False)
    else:
        ds = ProteinInterfaceDataset(files, cfg.data, with_labels=True, augment=False)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers, collate_fn=collate_chain_examples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastFusionIF(
        cfg.model,
        surface_feature_dim=int(ckpt["surface_feature_dim"]),
        residue_feature_dim=int(ckpt.get("residue_feature_dim", 0)),
        plm_dim=int(ckpt.get("plm_dim", 0)),
    ).to(device)
    model.load_state_dict(ckpt["model"])

    from fastfusion_if.evaluation import collect_predictions_tta

    if args.tta and args.tta > 1:
        y, p, group_ids, residue_rows = collect_predictions_tta(model, loader, device, n_aug=args.tta)
    else:
        y, p, group_ids, residue_rows = collect_predictions(model, loader, device)
    threshold = args.threshold
    if threshold is None:
        threshold = best_f1_threshold(y, p)[0] if args.split == "val" and len(y) else float(ckpt.get("threshold", 0.5))
    evaluated = evaluate_predictions(y, p, group_ids, threshold=threshold)
    metrics = {"split": args.split, "threshold": threshold, "tta": int(args.tta), "global": evaluated["global"], "per_group_summary": evaluated["per_group_summary"]}

    out_dir = ensure_dir(args.out_dir)
    (out_dir / f"{args.split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    write_csv(out_dir / f"{args.split}_per_residue_predictions.csv", residue_rows)
    write_csv(out_dir / f"{args.split}_per_protein_metrics.csv", evaluated["per_group_rows"])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
