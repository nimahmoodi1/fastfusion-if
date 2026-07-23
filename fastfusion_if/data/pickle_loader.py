from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import dill
except Exception:  # pragma: no cover
    dill = None

from .features import normalize_element
from .structures import AtomRecord


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default


def _safe_res_seq(x) -> int:
    try:
        return int(x)
    except Exception:
        digits = "".join(ch for ch in str(x) if ch.isdigit() or ch == "-")
        return int(digits) if digits else 0


def _residue_idr_maps(df: pd.DataFrame, annotations=None, propensities=None) -> tuple[dict, dict, dict]:
    """Map DataFrame residue keys to IDR annotation/propensity/sequence position."""
    annotations = annotations or []
    propensities = propensities or []

    cols = list(df.columns)
    chain_col = _pick_column(cols, ["chain_id", "chain", "chainID", "chainid", "chain_label", "chain_name"])
    resi_col = _pick_column(cols, ["residue_number", "res_id", "residue_id", "resSeq", "resi", "resnum", "res_num", "residue_num", "residue"])
    icode_col = _pick_column(cols, ["insertion", "iCode", "icode", "insertion_code", "ins_code"])

    ann_map: dict[tuple[str, int, str], float] = {}
    prop_map: dict[tuple[str, int, str], float] = {}
    idx_map: dict[tuple[str, int, str], int] = {}

    if not resi_col:
        return ann_map, prop_map, idx_map

    seen = []
    seen_set = set()

    for row in df.itertuples(index=False):
        d = row._asdict()
        res_seq = _safe_res_seq(d[resi_col])
        chain_id = str(d[chain_col]) if chain_col else "A"
        insertion = str(d.get(icode_col, "")).strip() if icode_col else ""
        key = (chain_id, res_seq, insertion)

        if key not in seen_set:
            seen_set.add(key)
            seen.append(key)

    for i, key in enumerate(seen):
        idx_map[key] = i
        ann_map[key] = _safe_float(annotations[i], 0.0) if i < len(annotations) else 0.0
        prop_map[key] = _safe_float(propensities[i], 0.0) if i < len(propensities) else 0.0

    return ann_map, prop_map, idx_map



def _load_pickle(path: str | Path) -> Any:
    path = Path(path)
    if dill is not None:
        try:
            with path.open("rb") as f:
                return dill.load(f)
        except Exception:
            pass
    return pd.read_pickle(path)


def _collect_dataframes(obj: Any, out: list[pd.DataFrame] | None = None) -> list[pd.DataFrame]:
    """Collect atom DataFrames from DIPS/DB5-style nested pickles.

    DIPS/DB5 pair files are often tuples/lists/dicts containing one DataFrame per partner.
    Older code that extracted only the first DataFrame silently converted a complex into a
    single chain and destroyed interface labels. This function intentionally collects all
    plausible atom DataFrames and leaves chain assignment to ``dataframe_to_atoms``.
    """
    if out is None:
        out = []
    if isinstance(obj, pd.DataFrame):
        out.append(obj)
        return out
    if isinstance(obj, dict):
        # Prefer common keys first, then scan remaining values.
        preferred = ["df", "atoms", "atom_df", "dataframe", "data", "complex", "structure", "graph", "ligand", "receptor"]
        seen = set()
        for key in preferred:
            if key in obj:
                seen.add(id(obj[key]))
                _collect_dataframes(obj[key], out)
        for value in obj.values():
            if id(value) not in seen:
                _collect_dataframes(value, out)
        return out
    if isinstance(obj, (tuple, list)):
        for value in obj:
            _collect_dataframes(value, out)
    return out


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    exact = {c: c for c in columns}
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in exact:
            return exact[cand]
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def dataframe_to_atoms(df: pd.DataFrame, drop_hydrogens: bool = True, default_chain_id: str = "A", idr_annotation_map: dict | None = None, idr_propensity_map: dict | None = None, residue_index_map: dict | None = None) -> list[AtomRecord]:
    cols = list(df.columns)
    x = _pick_column(cols, ["x", "coord_x", "atom_x", "x_coord", "coords_x", "pos_x"])
    y = _pick_column(cols, ["y", "coord_y", "atom_y", "y_coord", "coords_y", "pos_y"])
    z = _pick_column(cols, ["z", "coord_z", "atom_z", "z_coord", "coords_z", "pos_z"])
    chain = _pick_column(cols, ["chain_id", "chain", "chainID", "chainid", "chain_label", "chain_name"])
    resi = _pick_column(cols, ["residue_number", "res_id", "residue_id", "resSeq", "resi", "resnum", "res_num", "residue_num", "residue"])
    icode = _pick_column(cols, ["insertion", "iCode", "icode", "insertion_code", "ins_code"])
    resn = _pick_column(cols, ["residue_name", "resname", "res_name", "res", "residue_type"])
    element = _pick_column(cols, ["element", "elem", "atom_element", "atom_type", "symbol", "element_symbol", "type"])
    atom_name = _pick_column(cols, ["atom_name", "name", "atomname", "atom", "atom_type"])
    idr_annotation_map = idr_annotation_map or {}
    idr_propensity_map = idr_propensity_map or {}
    residue_index_map = residue_index_map or {}

    # Some pair pickles store coordinates in a single ndarray-like column.
    coords = _pick_column(cols, ["coords", "coord", "coordinate", "coordinates", "pos", "position"])
    if coords and not all([x, y, z]):
        pass
    elif not all([x, y, z]):
        raise ValueError(f"Missing coordinate columns. Available columns: {cols}")
    if not resi:
        raise ValueError(f"Missing residue id column. Available columns: {cols}")

    atoms: list[AtomRecord] = []
    for row in df.itertuples(index=False):
        d = row._asdict()
        name = str(d.get(atom_name, "")) if atom_name else ""
        elem = normalize_element(d.get(element, None) if element else None, name)
        if drop_hydrogens and elem == "H":
            continue
        res_seq = _safe_res_seq(d[resi])
        if coords and not all([x, y, z]):
            xyz = np.asarray(d[coords], dtype=np.float32).reshape(-1)[:3]
        else:
            xyz = np.asarray([float(d[x]), float(d[y]), float(d[z])], dtype=np.float32)
        chain_id = str(d[chain]) if chain else default_chain_id
        insertion = str(d.get(icode, "")).strip() if icode else ""
        res_key = (chain_id, res_seq, insertion)

        atoms.append(
            AtomRecord(
                coord=xyz,
                element=elem,
                atom_name=name,
                chain_id=chain_id,
                res_seq=res_seq,
                insertion=insertion,
                res_name=str(d.get(resn, "UNK")).upper() if resn else "UNK",
                idr_annotation=float(idr_annotation_map.get(res_key, 0.0)),
                idr_propensity=float(idr_propensity_map.get(res_key, 0.0)),
                residue_index=int(residue_index_map.get(res_key, 0)),
            )
        )
    return atoms


def parse_pickle_atoms(path: str | Path, drop_hydrogens: bool = True) -> list[AtomRecord]:
    obj = _load_pickle(path)
    dataframes = _collect_dataframes(obj)
    if not dataframes:
        raise ValueError(f"Could not find a pandas DataFrame inside object of type {type(obj)!r}")

    meta = {}
    try:
        maybe_meta = obj[7]
        if isinstance(maybe_meta, dict):
            meta = maybe_meta
    except Exception:
        meta = {}

    atoms: list[AtomRecord] = []
    for i, df in enumerate(dataframes):
        # Avoid duplicate DataFrames from nested dicts by checking object identity is not enough after recursion,
        # so we accept duplicates only when they produce distinct coordinates/chains in practice.
        default_chain = chr(ord("A") + min(i, 25))

        if i == 0:
            anns = meta.get("l_b_idr_annotations", [])
            props = meta.get("l_b_idr_propensities", [])
        elif i == 1:
            anns = meta.get("r_b_idr_annotations", [])
            props = meta.get("r_b_idr_propensities", [])
        else:
            anns = []
            props = []

        ann_map, prop_map, idx_map = _residue_idr_maps(df, anns, props)

        try:
            atoms.extend(
                dataframe_to_atoms(
                    df,
                    drop_hydrogens=drop_hydrogens,
                    default_chain_id=default_chain,
                    idr_annotation_map=ann_map,
                    idr_propensity_map=prop_map,
                    residue_index_map=idx_map,
                )
            )
        except Exception:
            # Skip non-atom feature tables that may be stored next to atom tables.
            continue
    if not atoms:
        raise ValueError("No atom records could be parsed from collected DataFrames")
    return atoms
