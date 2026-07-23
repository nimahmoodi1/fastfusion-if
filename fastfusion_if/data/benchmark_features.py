"""Load the precomputed per-residue features that ship with the AGAT-PPIS /
GraphPPIS benchmark (``AGAT-PPIS/Feature/<set>/<protein_id>.npy``).

These are exactly the evolutionary / structural profiles the published methods
are built around:

  * ``pssm``   : PSI-BLAST position-specific scoring matrix      (L, 20)
  * ``hmm``    : HHblits HMM profile                              (L, ~30)
  * ``dssp``   : DSSP secondary structure + solvent accessibility (L, ~9-14)
  * ``resAF``  : residue atom features used by AGAT-PPIS          (L, ~7)

FastFusion-IF computes its own geometry, surface point cloud and ESM-2
embeddings; this module only *loads* the benchmark's precomputed per-residue
profiles so we can feed the same feature set the competing methods use, fused
through our existing ``residue_features`` channel. Each file is a self-describing
``.npy`` (the width is read from the array), so this code does not assume any
particular feature dimensionality.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Per-residue feature sets to load and concatenate (in this order).
# ``psepos`` (pseudo-coordinates) and ``distance_map_*`` are geometry, not
# per-residue scalar features, so they are intentionally excluded.
DEFAULT_FEATURE_SETS = ("pssm", "hmm", "dssp", "resAF")


def load_raw_features(
    feature_dir: str | Path,
    protein_id: str,
    feature_sets=DEFAULT_FEATURE_SETS,
    row_mismatch_tol: int = 8,
):
    """Load and horizontally concatenate the per-residue feature matrices for one
    protein.

    Returns ``(feat, widths)`` where ``feat`` is an ``(L, D)`` float32 array and
    ``widths`` is a list of ``(set_name, width)``; or ``(None, reason)`` when a
    required set is missing or the row counts disagree by more than
    ``row_mismatch_tol``. When sets disagree by a small amount (a common terminal
    off-by-one in the benchmark's preprocessing, e.g. resAF having one extra
    residue), all sets are truncated to the shortest common length so the protein
    is still usable; the final per-residue alignment onto our structure is done
    later by align_rows().
    """
    feature_dir = Path(feature_dir)
    raw: list[tuple[str, np.ndarray]] = []
    for name in feature_sets:
        fp = feature_dir / name / f"{protein_id}.npy"
        if not fp.exists():
            return None, f"missing {name}/{protein_id}.npy"
        try:
            arr = np.load(fp, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            return None, f"unreadable {name}/{protein_id}.npy: {type(exc).__name__}"
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            return None, f"{name}/{protein_id}.npy has ndim={arr.ndim}"
        raw.append((name, arr))

    rows = [a.shape[0] for _, a in raw]
    lo, hi = min(rows), max(rows)
    if hi - lo > row_mismatch_tol:
        worst = max(raw, key=lambda na: abs(na[1].shape[0] - lo))
        return None, f"{worst[0]} rows {worst[1].shape[0]} != {lo}"
    if hi != lo:
        raw = [(name, a[:lo]) for name, a in raw]  # terminal truncation to common length

    widths = [(name, int(a.shape[1])) for name, a in raw]
    feat = np.concatenate([a for _, a in raw], axis=1).astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat, widths


def feature_set_layout(widths) -> str:
    """Human-readable summary like ``pssm:20 + hmm:30 + dssp:14 + resAF:7 = 71``."""
    parts = [f"{name}:{w}" for name, w in widths]
    total = sum(w for _, w in widths)
    return " + ".join(parts) + f" = {total}"
