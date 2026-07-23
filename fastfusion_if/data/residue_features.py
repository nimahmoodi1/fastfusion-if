from __future__ import annotations

import numpy as np

from .structures import AtomRecord

AA3 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA3)}
UNK_AA_IDX = len(AA3)

HYDROPATHY = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5, "MET": 1.9,
    "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8, "TRP": -0.9, "TYR": -1.3,
    "PRO": -1.6, "HIS": -3.2, "GLU": -3.5, "GLN": -3.5, "ASP": -3.5, "ASN": -3.5,
    "LYS": -3.9, "ARG": -4.5,
}

POSITIVE = {"LYS", "ARG", "HIS"}
NEGATIVE = {"ASP", "GLU"}
POLAR = {"SER", "THR", "ASN", "GLN", "CYS", "TYR", "TRP", "HIS"}
AROMATIC = {"PHE", "TRP", "TYR", "HIS"}
SULFUR = {"CYS", "MET"}
HYDROPHOBIC = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO"}
SMALL = {"GLY", "ALA", "SER", "CYS", "THR"}

# 21 amino-acid identity + 15 scalar/biochemical/context features
RESIDUE_FEATURE_DIM = 36


def residue_feature_names() -> list[str]:
    return (
        [f"aa_{aa}" for aa in AA3] + ["aa_UNK"] +
        [
            "hydropathy_norm",
            "is_positive",
            "is_negative",
            "is_neutral",
            "is_polar",
            "is_aromatic",
            "is_sulfur",
            "is_hydrophobic",
            "is_small",
            "is_glycine",
            "is_proline",
            "relative_position",
            "local_residue_density",
            "idr_annotation",
            "idr_propensity",
        ]
    )


def build_residue_features(
    residue_names: list[str],
    residue_keys: list[tuple[str, int, str]],
    atom2res: np.ndarray,
    atoms: list[AtomRecord],
    residue_edge_index: np.ndarray,
) -> np.ndarray:
    """Build lightweight leakage-safe residue features for FastFusion-IF-v3-lite."""
    n = len(residue_names)
    feats = np.zeros((n, RESIDUE_FEATURE_DIM), dtype=np.float32)

    if n == 0:
        return feats

    idr_ann = np.zeros((n,), dtype=np.float32)
    idr_prop = np.zeros((n,), dtype=np.float32)
    seen = np.zeros((n,), dtype=bool)

    for atom_i, res_i in enumerate(atom2res):
        r = int(res_i)
        if 0 <= r < n and not seen[r] and atom_i < len(atoms):
            idr_ann[r] = float(getattr(atoms[atom_i], "idr_annotation", 0.0) or 0.0)
            idr_prop[r] = float(getattr(atoms[atom_i], "idr_propensity", 0.0) or 0.0)
            seen[r] = True

    degree = np.zeros((n,), dtype=np.float32)
    if residue_edge_index is not None and residue_edge_index.size > 0:
        src = residue_edge_index[0].astype(np.int64, copy=False)
        valid = (src >= 0) & (src < n)
        np.add.at(degree, src[valid], 1.0)

    density = np.log1p(degree) / np.log1p(max(float(degree.max()), 1.0))
    denom = max(n - 1, 1)

    for i, name in enumerate(residue_names):
        aa = str(name).upper()
        idx = AA_TO_IDX.get(aa, UNK_AA_IDX)
        feats[i, idx] = 1.0

        offset = 21
        h = HYDROPATHY.get(aa, 0.0) / 4.5

        feats[i, offset + 0] = float(np.clip(h, -1.0, 1.0))
        feats[i, offset + 1] = 1.0 if aa in POSITIVE else 0.0
        feats[i, offset + 2] = 1.0 if aa in NEGATIVE else 0.0
        feats[i, offset + 3] = 0.0 if (aa in POSITIVE or aa in NEGATIVE) else 1.0
        feats[i, offset + 4] = 1.0 if aa in POLAR else 0.0
        feats[i, offset + 5] = 1.0 if aa in AROMATIC else 0.0
        feats[i, offset + 6] = 1.0 if aa in SULFUR else 0.0
        feats[i, offset + 7] = 1.0 if aa in HYDROPHOBIC else 0.0
        feats[i, offset + 8] = 1.0 if aa in SMALL else 0.0
        feats[i, offset + 9] = 1.0 if aa == "GLY" else 0.0
        feats[i, offset + 10] = 1.0 if aa == "PRO" else 0.0
        feats[i, offset + 11] = float(i / denom)
        feats[i, offset + 12] = float(density[i])
        feats[i, offset + 13] = float(idr_ann[i])
        feats[i, offset + 14] = float(np.clip(idr_prop[i], 0.0, 1.0))

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
