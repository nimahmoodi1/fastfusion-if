from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def random_rotation_matrix(device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate a random 3D rotation matrix."""
    q = torch.randn(4, device=device, dtype=dtype)
    q = q / q.norm().clamp_min(1e-8)
    w, x, y, z = q
    return torch.stack([
        torch.stack([1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)]),
        torch.stack([2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)]),
        torch.stack([2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]),
    ])


def cosine_warmup_lr(epoch: int, max_epochs: int, warmup_epochs: int, base_lr: float) -> float:
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
    t = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def find_structure_files(root: str | Path, file_glob: str = "**/*") -> list[Path]:
    root = Path(root)
    files = [p for p in root.glob(file_glob) if p.is_file()]
    allowed = {".pdb", ".ent", ".cif", ".mmcif", ".pkl", ".pickle", ".dill", ".p"}
    return sorted([p for p in files if p.suffix.lower() in allowed])
