from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import binary_metrics, per_group_binary_metrics


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _rotate_batch_coords(batch_dev: dict, R: torch.Tensor) -> dict:
    """Return a shallow copy of the batch with all 3D coordinate tensors rotated.

    Only positions are rotated. The scalar surface descriptors (hydropathy,
    curvature, burial, H-bond counts, ...) are rotation-invariant, so they are
    left untouched. NOTE: this assumes surface *normals* are not used as input
    features (the default); if you enable normals-as-features, rotate them too.
    """
    out = dict(batch_dev)
    for key in ("atom_pos", "surface_pos", "residue_pos"):
        t = batch_dev.get(key, None)
        if torch.is_tensor(t) and t.numel() and t.shape[-1] == 3:
            out[key] = t @ R.T
    return out


def collect_predictions_tta(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_aug: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    """Like collect_predictions but averages probabilities over n_aug random
    rotations (the first pass is always the identity / un-rotated structure).

    The network is approximately rotation-invariant (relative-distance attention
    + coordinate updates), so averaging over orientations reduces prediction
    variance and typically improves PR-AUC/MCC and calibration by a small margin
    at no training cost. Use this for the final reported numbers.
    """
    if n_aug <= 1:
        return collect_predictions(model, loader, device)
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    ys: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    group_ids: list[str] = []
    residue_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluate(tta={n_aug})", leave=False):
            if batch is None or batch["y"] is None:
                continue
            batch_dev = move_to_device(batch, device)
            acc = None
            for a in range(n_aug):
                if a == 0:
                    bd = batch_dev  # identity pass
                else:
                    q = torch.randn(4, generator=gen)
                    q = q / q.norm().clamp_min(1e-8)
                    w, x, y_, z = q
                    R = torch.stack([
                        torch.stack([1 - 2 * (y_*y_ + z*z), 2 * (x*y_ - z*w), 2 * (x*z + y_*w)]),
                        torch.stack([2 * (x*y_ + z*w), 1 - 2 * (x*x + z*z), 2 * (y_*z - x*w)]),
                        torch.stack([2 * (x*z - y_*w), 2 * (y_*z + x*w), 1 - 2 * (x*x + y_*y_)]),
                    ]).to(device=device, dtype=batch_dev["atom_pos"].dtype)
                    bd = _rotate_batch_coords(batch_dev, R)
                p = torch.sigmoid(model(bd).detach().cpu().float())
                acc = p if acc is None else acc + p
            p = (acc / float(n_aug)).numpy()
            y = batch["y"].detach().cpu().numpy().astype(int)
            ys.append(y)
            probs.append(p)
            offset = 0
            for meta in batch["metadata"]:
                n = len(meta["residue_keys"])
                gid = f"{Path(meta['source_path']).name}:{meta['chain_id']}"
                for local_i, (res_key, res_name) in enumerate(zip(meta["residue_keys"], meta["residue_names"])):
                    idx = offset + local_i
                    group_ids.append(gid)
                    residue_rows.append(
                        {
                            "group_id": gid,
                            "source_path": meta["source_path"],
                            "chain_id": meta["chain_id"],
                            "res_chain": res_key[0],
                            "res_seq": res_key[1],
                            "insertion": res_key[2],
                            "res_name": res_name,
                            "label": int(y[idx]),
                            "probability": float(p[idx]),
                        }
                    )
                offset += n
    if not ys:
        return np.array([]), np.array([]), [], []
    return np.concatenate(ys), np.concatenate(probs), group_ids, residue_rows


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    """Collect residue-level labels/probabilities plus protein grouping metadata."""
    model.eval()
    ys: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    group_ids: list[str] = []
    residue_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate", leave=False):
            if batch is None or batch["y"] is None:
                continue
            batch_dev = move_to_device(batch, device)
            logits = model(batch_dev).detach().cpu().float()
            p = torch.sigmoid(logits).numpy()
            y = batch["y"].detach().cpu().numpy().astype(int)
            ys.append(y)
            probs.append(p)

            offset = 0
            for meta in batch["metadata"]:
                n = len(meta["residue_keys"])
                gid = f"{Path(meta['source_path']).name}:{meta['chain_id']}"
                for local_i, (res_key, res_name) in enumerate(zip(meta["residue_keys"], meta["residue_names"])):
                    idx = offset + local_i
                    group_ids.append(gid)
                    residue_rows.append(
                        {
                            "group_id": gid,
                            "source_path": meta["source_path"],
                            "chain_id": meta["chain_id"],
                            "res_chain": res_key[0],
                            "res_seq": res_key[1],
                            "insertion": res_key[2],
                            "res_name": res_name,
                            "label": int(y[idx]),
                            "probability": float(p[idx]),
                        }
                    )
                offset += n
    if not ys:
        return np.array([]), np.array([]), [], []
    return np.concatenate(ys), np.concatenate(probs), group_ids, residue_rows


def evaluate_predictions(y: np.ndarray, p: np.ndarray, group_ids: list[str], threshold: float) -> dict[str, Any]:
    if len(y) == 0:
        return {"global": {}, "per_group_summary": {}, "per_group_rows": []}
    global_metrics = binary_metrics(y, p, threshold=threshold)
    group_rows, group_summary = per_group_binary_metrics(y, p, group_ids, threshold=threshold)
    return {"global": global_metrics, "per_group_summary": group_summary, "per_group_rows": group_rows}


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
