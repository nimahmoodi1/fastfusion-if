#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import gc
import faulthandler
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fastfusion_if.config import ExperimentConfig
from fastfusion_if.data.collate import collate_chain_examples
from fastfusion_if.data.dataset import ProteinInterfaceDataset
from fastfusion_if.data.cached_dataset import CachedInterfaceDataset
from fastfusion_if.data.splits import read_manifest
from fastfusion_if.evaluation import collect_predictions, evaluate_predictions, move_to_device, write_csv
from fastfusion_if.losses import interface_loss
from fastfusion_if.metrics import best_f1_threshold, binary_metrics
from fastfusion_if.models import FastFusionIF
from fastfusion_if.utils import ensure_dir, seed_everything, cosine_warmup_lr


def estimate_pos_weight(loader: DataLoader, max_batches: int = 128) -> float:
    pos = 0.0
    total = 0.0
    for i, batch in enumerate(loader):
        if batch is None or batch["y"] is None:
            continue
        y = batch["y"].float()
        pos += float(y.sum())
        total += float(y.numel())
        if i + 1 >= max_batches:
            break
    neg = max(0.0, total - pos)
    return 1.0 if pos <= 0 else float(neg / pos)


def run_epoch(model, loader, optimizer, device, cfg, train: bool, scaler=None) -> tuple[dict, np.ndarray, np.ndarray]:
    model.train(train)
    losses = []
    all_logits = []
    all_y = []
    pbar = tqdm(loader, leave=False, desc="train" if train else "eval")
    for batch in pbar:
        if batch is None or batch["y"] is None:
            continue
        batch = move_to_device(batch, device)
        y = batch["y"].float()

        with torch.autocast(device_type=device.type, enabled=(cfg.train.amp and device.type == "cuda")):
            logits = model(batch)
            loss = interface_loss(
                logits,
                y,
                positive_weight=cfg.train.positive_weight,
                dice_weight=cfg.train.dice_weight,
                focal_weight=cfg.train.focal_weight,
                focal_gamma=cfg.train.focal_gamma,
                focal_alpha=cfg.train.focal_alpha,
                tversky_weight=cfg.train.tversky_weight,
                tversky_alpha=cfg.train.tversky_alpha,
                tversky_beta=cfg.train.tversky_beta,
                tversky_gamma=cfg.train.tversky_gamma,
            )

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()

        losses.append(float(loss.detach().cpu()))

        # For full-size training, storing all train logits/labels for the whole epoch
        # can grow RAM. We only need full predictions for validation/test.
        if not train:
            all_logits.append(logits.detach().cpu())
            all_y.append(y.detach().cpu())

    mean_loss = float(np.mean(losses)) if losses else float("nan")

    if train:
        return {"loss": mean_loss}, np.array([]), np.array([])

    if not all_logits:
        return {"loss": float("nan")}, np.array([]), np.array([])

    logits = torch.cat(all_logits).numpy()
    y = torch.cat(all_y).numpy().astype(int)
    probs = torch.sigmoid(torch.from_numpy(logits).float()).numpy()
    metrics = binary_metrics(y, probs)
    metrics["loss"] = mean_loss
    return metrics, y, probs


def _make_dataset(cfg: ExperimentConfig, files: list[str], augment: bool):
    """Use the on-disk cache when cfg.data.cache_dir is set, else build on the fly."""
    if getattr(cfg.data, "cache_dir", None):
        return CachedInterfaceDataset.from_manifest_split(files, cfg.data.cache_dir, cfg.data, augment=augment)
    return ProteinInterfaceDataset(files, cfg.data, with_labels=True, augment=augment)


def make_loaders(cfg: ExperimentConfig, train_files: list[str], val_files: list[str], test_files: list[str] | None = None):
    train_ds = _make_dataset(cfg, train_files, augment=True)
    val_ds = _make_dataset(cfg, val_files, augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, collate_fn=collate_chain_examples)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers, collate_fn=collate_chain_examples)
    test_loader = None
    if test_files:
        test_ds = _make_dataset(cfg, test_files, augment=False)
        test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers, collate_fn=collate_chain_examples)
    return train_loader, val_loader, test_loader


def first_non_empty_batch(loader: DataLoader) -> dict:
    for batch in loader:
        if batch is not None:
            return batch
    raise RuntimeError("No valid training examples found")


def main() -> None:
    faulthandler.enable(all_threads=True)
    gc.disable()  # avoid mid-batch garbage-collection crashes during heavy data loading
    parser = argparse.ArgumentParser(description="Train FastFusion-IF on a leakage-safe manifest")
    parser.add_argument("--manifest", type=str, required=True, help="CSV with path,split,cluster_id columns. Use scripts/prepare_manifest.py.")
    parser.add_argument("--config", type=str, default=None, help="JSON config file")
    parser.add_argument("--out-dir", type=str, default="runs/fastfusion_if")
    parser.add_argument("--allow-missing-test", action="store_true", help="Allow training when manifest has no test split")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from a checkpoint, usually last.pt")
    parser.add_argument("--auto-resume", action="store_true", help="Automatically resume from out-dir/last.pt if it exists")
    parser.add_argument("--cache-dir", type=str, default=None, help="Use a precomputed example cache directory (overrides config).")
    parser.add_argument("--num-workers", type=int, default=None, help="Override config num_workers (safe to raise when using --cache-dir).")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--seed", type=int, default=None, help="Override config train.seed (for training ensemble members with different seeds).")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_json(args.config) if args.config else ExperimentConfig()
    if args.cache_dir is not None:
        cfg.data.cache_dir = args.cache_dir
    if args.num_workers is not None:
        cfg.train.num_workers = int(args.num_workers)
    if args.epochs is not None:
        cfg.train.epochs = int(args.epochs)
    if args.seed is not None:
        cfg.train.seed = int(args.seed)
    seed_everything(cfg.train.seed)
    out_dir = ensure_dir(args.out_dir)
    cfg.save_json(out_dir / "config.json")
    shutil.copyfile(args.manifest, out_dir / "manifest.csv")

    splits = read_manifest(args.manifest)
    train_files = splits.get("train", [])
    val_files = splits.get("val", [])
    test_files = splits.get("test", [])
    if not train_files or not val_files:
        raise RuntimeError("Manifest must contain non-empty train and val splits.")
    if not test_files and not args.allow_missing_test:
        raise RuntimeError("Manifest has no test split. Add a test split or pass --allow-missing-test for development only.")

    print(f"Files: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    train_loader, val_loader, test_loader = make_loaders(cfg, train_files, val_files, test_files)

    first_batch = first_non_empty_batch(train_loader)
    surface_feature_dim = int(first_batch["surface_features"].shape[-1])
    residue_feature_dim = int(first_batch["residue_features"].shape[-1])
    plm_dim = int(first_batch["residue_plm"].shape[-1]) if first_batch.get("residue_plm", None) is not None else 0
    if cfg.model.use_plm_features and plm_dim == 0:
        raise RuntimeError(
            "model.use_plm_features=true but the data has no residue_plm. "
            "Rebuild the cache with PLM embeddings, e.g.:\n"
            "  python scripts/precompute_cache.py --manifest <m> --config <c> "
            "--cache-dir <cache> --plm-model esm2_t33_650M_UR50D"
        )
    if plm_dim > 0 and not cfg.model.use_plm_features:
        print("Note: residue_plm present in data but model.use_plm_features=false; PLM features will be ignored.")

    if cfg.train.positive_weight is None:
        cfg.train.positive_weight = estimate_pos_weight(train_loader)
        print(f"Estimated positive weight: {cfg.train.positive_weight:.3f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastFusionIF(cfg.model, surface_feature_dim=surface_feature_dim, residue_feature_dim=residue_feature_dim, plm_dim=plm_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.train.amp and device.type == "cuda"))

    best_score = -float("inf")
    best_threshold = 0.5
    start_epoch = 0
    history = []

    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
    elif args.auto_resume and (out_dir / "last.pt").exists():
        resume_path = out_dir / "last.pt"

    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        if "scaler" in ckpt and ckpt["scaler"] is not None and scaler is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as exc:
                print(f"Warning: could not load scaler state: {exc}")

        best_score = float(ckpt.get("best_score", best_score))
        best_threshold = float(ckpt.get("best_threshold", best_threshold))
        start_epoch = int(ckpt.get("epoch", 0))
        history = ckpt.get("history", history)

        print(
            f"Resume state: start_epoch={start_epoch}, "
            f"best_score={best_score:.6f}, best_threshold={best_threshold:.3f}"
        )

    for epoch in range(start_epoch, cfg.train.epochs):
        lr = cosine_warmup_lr(epoch, cfg.train.epochs, cfg.train.warmup_epochs, cfg.train.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        train_metrics, _, _ = run_epoch(model, train_loader, optimizer, device, cfg, train=True, scaler=scaler)
        val_metrics, val_y, val_probs = run_epoch(model, val_loader, optimizer, device, cfg, train=False, scaler=None)
        threshold, best_f1 = best_f1_threshold(val_y, val_probs) if len(val_y) else (0.5, float("nan"))
        val_metrics_at_threshold = binary_metrics(val_y, val_probs, threshold=threshold) if len(val_y) else {}
        val_metrics["best_f1_threshold"] = threshold
        val_metrics["best_f1"] = best_f1
        val_metrics["at_best_f1_threshold"] = val_metrics_at_threshold

        row = {"epoch": epoch + 1, "lr": lr, "train": train_metrics, "val": val_metrics}
        history.append(row)

        # Auto-save history after every epoch so interrupted runs still keep logs.
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        print(json.dumps(row, indent=2))

        score = val_metrics.get(cfg.train.checkpoint_metric, float("nan"))
        if np.isfinite(score) and score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.to_dict(),
                    "surface_feature_dim": surface_feature_dim,
                    "residue_feature_dim": residue_feature_dim,
                    "plm_dim": plm_dim,
                    "threshold": best_threshold,
                    "best_val_metric": cfg.train.checkpoint_metric,
                    "best_val_score": best_score,
                },
                out_dir / "best.pt",
            )
            print(f"Saved best checkpoint with {cfg.train.checkpoint_metric}={best_score:.4f}, threshold={best_threshold:.3f}")

        last_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
            "cfg": cfg.to_dict(),
            "epoch": epoch + 1,
            "best_score": best_score,
            "best_threshold": best_threshold,
            "history": history,
            "surface_feature_dim": surface_feature_dim,
            "residue_feature_dim": residue_feature_dim,
            "plm_dim": plm_dim,
        }
        torch.save(last_payload, out_dir / "last.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    if test_loader is not None and (out_dir / "best.pt").exists():
        print("Evaluating best checkpoint on test split...")
        ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        threshold = float(ckpt.get("threshold", best_threshold))
        y, p, group_ids, residue_rows = collect_predictions(model, test_loader, device)
        evaluated = evaluate_predictions(y, p, group_ids, threshold=threshold)
        test_metrics = {"threshold": threshold, "global": evaluated["global"], "per_group_summary": evaluated["per_group_summary"]}
        (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
        write_csv(out_dir / "test_per_residue_predictions.csv", residue_rows)
        write_csv(out_dir / "test_per_protein_metrics.csv", evaluated["per_group_rows"])
        print(json.dumps({"test": test_metrics}, indent=2))

    print("Finished. Best checkpoint:", out_dir / "best.pt")


if __name__ == "__main__":
    main()
