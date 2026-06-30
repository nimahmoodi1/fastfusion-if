#!/usr/bin/env python
"""Evaluate an ENSEMBLE of FastFusion-IF checkpoints on one manifest split.

Per-residue probabilities are averaged across all checkpoints (optionally with
per-model test-time augmentation), then the standard metrics are computed once
on the averaged probabilities. Averaging independently-trained models (different
seeds, or different feature variants such as geometry-only + ESM-2) is the
standard way to gain the last fraction of PR-AUC / MCC for the final reported
numbers — it almost always beats any single member.

All checkpoints must be evaluated on the SAME split of the SAME manifest, and
their residue ordering must match (it does, because the cache / dataset is
deterministic for a given manifest). Checkpoints may differ in architecture and
features (e.g. one geometry-only model and one ESM-2 model); each is run with
its own config, surface-feature dim and plm_dim as stored in its checkpoint.

Example:
    python scripts/evaluate_ensemble.py \
        --checkpoints runs/full_v2_esm/best.pt runs/full_v2_rich/best.pt runs/full_v2_cached/best.pt \
        --manifest manifests/dips_plus_mmseqs30_full.csv --split test \
        --cache-dir cache/dips_plus_v2_esm --tta 8 --out-dir eval/ensemble_full
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from fastfusion_if.config import ExperimentConfig
from fastfusion_if.data.collate import collate_chain_examples
from fastfusion_if.data.dataset import ProteinInterfaceDataset
from fastfusion_if.data.splits import read_manifest
from fastfusion_if.evaluation import (
    collect_predictions,
    collect_predictions_tta,
    evaluate_predictions,
    write_csv,
)
from fastfusion_if.metrics import best_f1_threshold
from fastfusion_if.models import FastFusionIF
from fastfusion_if.utils import ensure_dir


def _build_loader(files, cfg):
    if getattr(cfg.data, "cache_dir", None):
        from fastfusion_if.data.cached_dataset import CachedInterfaceDataset

        ds = CachedInterfaceDataset.from_manifest_split(files, cfg.data.cache_dir, cfg.data, augment=False)
    else:
        ds = ProteinInterfaceDataset(files, cfg.data, with_labels=True, augment=False)
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_chain_examples,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an ensemble of FastFusion-IF checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Two or more .pt checkpoints.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default="eval/ensemble")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache dir to use for any PLM checkpoints (and geometry models too if cached).")
    parser.add_argument("--tta", type=int, default=1, help="Per-model test-time augmentation rotations (1 = off).")
    parser.add_argument("--weights", type=float, nargs="+", default=None, help="Optional per-checkpoint weights (same count/order as --checkpoints).")
    args = parser.parse_args()

    if len(args.checkpoints) < 2:
        print("[warn] ensemble with a single checkpoint is just a normal eval.")
    weights = args.weights if args.weights else [1.0] * len(args.checkpoints)
    if len(weights) != len(args.checkpoints):
        raise SystemExit("--weights must have the same number of entries as --checkpoints")
    wsum = float(sum(weights))

    splits = read_manifest(args.manifest)
    files = splits.get(args.split, [])
    if not files:
        raise RuntimeError(f"No files for split={args.split!r} in {args.manifest}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p_accum = None
    y_ref = None
    group_ref = None
    rows_ref = None
    members = []

    for ckpt_path, w in zip(args.checkpoints, weights):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ExperimentConfig.from_dict(ckpt["cfg"])
        if args.batch_size is not None:
            cfg.train.batch_size = args.batch_size
        if args.num_workers is not None:
            cfg.train.num_workers = args.num_workers
        # A PLM checkpoint needs a PLM-baked cache; honour --cache-dir if given.
        if args.cache_dir is not None:
            cfg.data.cache_dir = args.cache_dir

        loader = _build_loader(files, cfg)
        model = FastFusionIF(
            cfg.model,
            surface_feature_dim=int(ckpt["surface_feature_dim"]),
            residue_feature_dim=int(ckpt.get("residue_feature_dim", 0)),
            plm_dim=int(ckpt.get("plm_dim", 0)),
        ).to(device)
        model.load_state_dict(ckpt["model"])

        if args.tta and args.tta > 1:
            y, p, group_ids, residue_rows = collect_predictions_tta(model, loader, device, n_aug=args.tta)
        else:
            y, p, group_ids, residue_rows = collect_predictions(model, loader, device)

        if p_accum is None:
            p_accum = w * p
            y_ref, group_ref, rows_ref = y, group_ids, residue_rows
        else:
            if p.shape != p_accum.shape or not np.array_equal(y, y_ref) or group_ids != group_ref:
                raise RuntimeError(
                    f"Checkpoint {ckpt_path} produced a different residue ordering/labels than the first "
                    f"checkpoint. All ensemble members must use the same manifest split (and matching cache)."
                )
            p_accum = p_accum + w * p
        members.append({"checkpoint": str(ckpt_path), "weight": float(w)})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    p_mean = p_accum / wsum

    threshold = args.threshold
    if threshold is None:
        threshold = best_f1_threshold(y_ref, p_mean)[0] if args.split == "val" and len(y_ref) else 0.5
    evaluated = evaluate_predictions(y_ref, p_mean, group_ref, threshold=threshold)

    # Rewrite per-residue probabilities to the ensembled values.
    for row, prob in zip(rows_ref, p_mean):
        row["probability"] = float(prob)

    metrics = {
        "split": args.split,
        "threshold": threshold,
        "tta": int(args.tta),
        "members": members,
        "global": evaluated["global"],
        "per_group_summary": evaluated["per_group_summary"],
    }
    out_dir = ensure_dir(args.out_dir)
    (out_dir / f"{args.split}_ensemble_metrics.json").write_text(json.dumps(metrics, indent=2))
    write_csv(out_dir / f"{args.split}_ensemble_per_residue_predictions.csv", rows_ref)
    write_csv(out_dir / f"{args.split}_ensemble_per_protein_metrics.csv", evaluated["per_group_rows"])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
