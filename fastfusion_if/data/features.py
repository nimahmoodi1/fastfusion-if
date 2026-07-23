from __future__ import annotations

from typing import Optional

from ..constants import ELEMENT_TO_INDEX, UNKNOWN_ELEMENT_INDEX, HYDROPATHY, RESIDUE_CHARGE, VDW_RADII, DEFAULT_VDW_RADIUS


def normalize_element(raw_element: Optional[str], atom_name: Optional[str] = None) -> str:
    if raw_element is None or str(raw_element).strip() == "":
        raw_element = atom_name or ""
    e = str(raw_element).strip().upper()
    if e in ELEMENT_TO_INDEX or e in VDW_RADII:
        return e
    # PDB atom names are often aligned: " CA " can mean carbon alpha, not calcium.
    # Prefer the first alphabetic character for ordinary protein atom names.
    letters = "".join(ch for ch in str(atom_name or e).strip().upper() if ch.isalpha())
    if len(letters) >= 2 and letters[:2] in ELEMENT_TO_INDEX and letters[:2] not in {"CA"}:
        return letters[:2]
    if letters:
        return letters[0]
    return "UNK"


def element_to_index(element: str) -> int:
    return ELEMENT_TO_INDEX.get(element.upper(), UNKNOWN_ELEMENT_INDEX)


def vdw_radius(element: str) -> float:
    return VDW_RADII.get(element.upper(), DEFAULT_VDW_RADIUS)


def hydropathy(res_name: str) -> float:
    return HYDROPATHY.get(str(res_name).upper(), 0.0) / 4.5


def residue_charge(res_name: str) -> float:
    return RESIDUE_CHARGE.get(str(res_name).upper(), 0.0)
