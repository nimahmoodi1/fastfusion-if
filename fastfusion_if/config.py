from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass
class DataConfig:
    """Dataset and preprocessing parameters."""

    label_cutoff: float = 5.0
    atom_edge_radius: float = 4.8
    surface_edge_radius: float = 6.0
    cross_edge_radius: float = 6.0
    max_atom_neighbors: int = 32
    max_surface_neighbors: int = 32
    max_cross_neighbors: int = 48
    residue_edge_radius: float = 12.0
    max_residue_neighbors: int = 32

    # Mesh-free surface sampling. n_surface_dirs is per atom before pruning.
    probe_radius: float = 1.4
    n_surface_dirs: int = 24
    max_surface_points: int = 4096
    min_surface_points: int = 64
    use_surface_normals_as_features: bool = False

    # Robustness and speed.
    drop_hydrogens: bool = True
    center_coordinates: bool = True
    random_rotation: bool = True
    coordinate_jitter_std: float = 0.02

    # Dataset loading.
    file_glob: str = "**/*"
    max_files: Optional[int] = None
    skip_errors: bool = True

    # Surface feature set: "basic" (4 scalars, original) or "rich" (10 scalars).
    surface_feature_set: str = "basic"
    burial_radius: float = 10.0
    shape_k_neighbors: int = 16

    # Optional on-disk example cache (set by precompute_cache.py). When this is a
    # path, training/eval can use CachedInterfaceDataset instead of recomputing
    # surfaces/graphs/labels on the fly every epoch.
    cache_dir: Optional[str] = None


@dataclass
class ModelConfig:
    """FastFusion-IF network parameters."""

    atom_dim: int = 128
    surface_dim: int = 128
    fusion_dim: int = 128
    use_surface: bool = True  # ablation: False removes the surface point-cloud branch entirely
    n_atom_layers: int = 6
    n_surface_layers: int = 4
    n_fusion_layers: int = 2
    n_attention_heads: int = 4
    dropout: float = 0.10
    use_coordinate_updates: bool = True
    residue_pooling: str = "attention"  # "mean" or "attention"
    use_residue_context: bool = False  # keep False by default for v1 checkpoint compatibility
    n_residue_layers: int = 2
    use_residue_features: bool = False
    residue_feature_dropout: float = 0.10
    residue_feature_scale: float = 0.25
    # Protein language model (e.g. ESM-2) residue embeddings. Precomputed and
    # cached; injected into the residue token before the residue-context
    # transformer. plm_dim is set at runtime from the data.
    #   plm_inject="concat": concatenate projected PLM features with the residue
    #       token and mix with a linear (recommended; strongest in practice).
    #   plm_inject="add": residual project-and-add.
    # In both modes the PLM contribution is ZERO-INITIALISED, so at the start of
    # training the network behaves exactly like the no-PLM model and then learns
    # to incorporate the embeddings — this makes the ESM-2 upgrade strictly safe
    # (it can never degrade the geometric backbone at initialisation).
    use_plm_features: bool = False
    plm_dim: int = 0
    plm_dropout: float = 0.10
    plm_inject: str = "concat"  # "concat" or "add"


@dataclass
class TrainConfig:
    """Training parameters."""

    seed: int = 42
    batch_size: int = 1
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    positive_weight: Optional[float] = None
    dice_weight: float = 0.20
    amp: bool = True
    num_workers: int = 0
    eval_every: int = 1
    checkpoint_metric: str = "pr_auc"
    # Optional extra loss terms (default 0.0 -> identical to weighted BCE + Dice).
    focal_weight: float = 0.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    tversky_weight: float = 0.0
    tversky_alpha: float = 0.7
    tversky_beta: float = 0.3
    tversky_gamma: float = 1.0


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ExperimentConfig":
        """Build config from a nested dictionary stored in JSON/checkpoints."""
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        raw = json.loads(Path(path).read_text())
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
