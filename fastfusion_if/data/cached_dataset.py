from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from torch.utils.data import Dataset

from ..config import DataConfig
from .structures import ChainExample


# ---------------------------------------------------------------------------
# CachedInterfaceDataset
#
# Loads ChainExample objects that were precomputed once by
# scripts/precompute_cache.py, instead of re-running surface sampling, graph
# construction and label generation on every __getitem__ of every epoch.
#
# Why this matters for FastFusion-IF:
#   * Your on-the-fly ProteinInterfaceDataset recomputes the cKDTree-based
#     surface point cloud (Python loops over query_ball_point), all radius
#     graphs, residue graphs and labels every single epoch, single-threaded
#     (num_workers=0 was forced for stability). That is the main reason full
#     training only reached 12 epochs and why the loader segfaulted during GC.
#   * With a cache, __getitem__ is just pickle.load + a cheap rigid rotation,
#     so epochs are far faster, num_workers can be raised safely, and the heavy
#     native code is no longer in the training hot loop.
#
# Augmentation note (this also fixes a latent bug):
#   Your current augmentation seeds the RNG with the file index, so each file
#   receives the SAME rotation every epoch -> effectively a fixed per-sample
#   rotation, not stochastic augmentation. Here we draw a fresh rotation each
#   __getitem__, giving real rotational augmentation across epochs. Because all
#   graphs are radius-based and rotation is rigid, the cached edges remain valid
#   after rotation, so we do not need to rebuild any graph.
# ---------------------------------------------------------------------------


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4).astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def cache_path_for(cache_dir: str | Path, source_path: str) -> Path:
    """Deterministic cache filename for a source structure file."""
    import hashlib

    h = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:16]
    stem = Path(source_path).name.replace(".", "_")
    return Path(cache_dir) / f"{stem}__{h}.pkl"


class CachedInterfaceDataset(Dataset):
    def __init__(
        self,
        cache_files: Iterable[str | Path],
        cfg: DataConfig,
        augment: bool = False,
    ) -> None:
        self.cfg = cfg
        self.augment = augment
        self.cache_files: List[Path] = [Path(p) for p in cache_files]
        self.cache_files = [p for p in self.cache_files if p.exists()]
        if not self.cache_files:
            raise FileNotFoundError(
                "No cache files found. Run scripts/precompute_cache.py first."
            )

    @classmethod
    def from_manifest_split(
        cls,
        source_paths: Iterable[str],
        cache_dir: str | Path,
        cfg: DataConfig,
        augment: bool = False,
    ) -> "CachedInterfaceDataset":
        files = [cache_path_for(cache_dir, p) for p in source_paths]
        return cls(files, cfg, augment=augment)

    def __len__(self) -> int:
        return len(self.cache_files)

    def __getitem__(self, index: int) -> List[ChainExample]:
        with self.cache_files[index].open("rb") as f:
            examples: List[ChainExample] = pickle.load(f)

        if not self.augment or not examples:
            return examples

        rng = np.random.default_rng()  # fresh entropy -> true per-epoch augmentation
        rot = _random_rotation_matrix(rng) if self.cfg.random_rotation else None
        jitter = float(self.cfg.coordinate_jitter_std)

        for ex in examples:
            if rot is not None:
                ex.atom_pos = (ex.atom_pos @ rot.T).astype(np.float32)
                ex.residue_pos = (ex.residue_pos @ rot.T).astype(np.float32)
                if ex.surface_pos.size:
                    ex.surface_pos = (ex.surface_pos @ rot.T).astype(np.float32)
            if jitter > 0:
                ex.atom_pos = (
                    ex.atom_pos + rng.normal(0.0, jitter, size=ex.atom_pos.shape)
                ).astype(np.float32)
                if ex.surface_pos.size:
                    ex.surface_pos = (
                        ex.surface_pos + rng.normal(0.0, jitter, size=ex.surface_pos.shape)
                    ).astype(np.float32)
        return examples
