from __future__ import annotations

# Common heavy elements found in proteins, cofactors, ions, and PDB structures.
ELEMENTS = [
    "C", "N", "O", "S", "P", "F", "CL", "BR", "I", "SE",
    "ZN", "MG", "CA", "FE", "CU", "MN", "NA", "K", "CO", "NI",
]
ELEMENT_TO_INDEX = {e: i for i, e in enumerate(ELEMENTS)}
UNKNOWN_ELEMENT_INDEX = len(ELEMENTS)

# Bondi-like / common VDW radii in Angstrom. Good enough for mesh-free SAS proxy.
VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "SE": 1.90,
    "ZN": 1.39, "MG": 1.73, "CA": 1.94, "FE": 1.56, "CU": 1.40,
    "MN": 1.61, "NA": 2.27, "K": 2.75, "CO": 1.52, "NI": 1.63,
}
DEFAULT_VDW_RADIUS = 1.70

# Kyte-Doolittle hydropathy values. Used as a cheap surface chemistry feature.
HYDROPATHY = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
    "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
    "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
    "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5,
}

# Approximate residue side-chain formal charge around neutral pH.
RESIDUE_CHARGE = {
    "ASP": -1.0, "GLU": -1.0, "LYS": 1.0, "ARG": 1.0, "HIS": 0.1,
}
