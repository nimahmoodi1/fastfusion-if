from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

from .sequences import AA3_TO_AA1

# ---------------------------------------------------------------------------
# Protein language model (ESM-2) residue embeddings for FastFusion-IF.
#
# Design choices:
#   * Lazy import: torch/esm are only imported when you actually build embeddings,
#     so importing fastfusion_if never requires the `fair-esm` package.
#   * Per-sequence disk cache keyed by (model_name, sequence) hash, so identical
#     chains across the dataset are embedded once. Embeddings are stored as fp16
#     .npy files to save disk.
#   * embed_residue_names() builds the 1-letter sequence directly from a chain's
#     residue_names list, guaranteeing row-for-row alignment with the residues in
#     a ChainExample (no reliance on atom ordering).
#
# Recommended models for an 8 GB GPU (inference only, fits comfortably):
#   esm2_t30_150M_UR50D  -> 640-d   (fast, light)
#   esm2_t33_650M_UR50D  -> 1280-d  (stronger, still fits)
# ---------------------------------------------------------------------------

_PLM_DIMS = {
    "esm2_t6_8M_UR50D": 320,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t36_3B_UR50D": 2560,
}


def plm_dim_for(model_name: str) -> int:
    if model_name not in _PLM_DIMS:
        raise ValueError(f"Unknown ESM-2 model {model_name!r}. Known: {sorted(_PLM_DIMS)}")
    return _PLM_DIMS[model_name]


def residue_names_to_sequence(residue_names: list[str]) -> str:
    return "".join(AA3_TO_AA1.get(str(n).upper(), "X") for n in residue_names)


def _seq_hash(model_name: str, seq: str) -> str:
    return hashlib.sha1(f"{model_name}::{seq}".encode("utf-8")).hexdigest()


class ESM2Extractor:
    """Loads an ESM-2 model once and produces per-residue embeddings.

    Parameters
    ----------
    model_name : str
        One of the keys in _PLM_DIMS.
    cache_dir : str | Path | None
        If given, per-sequence embeddings are cached here as .npy (fp16).
    device : str
        "cuda" or "cpu". Embedding is inference-only.
    max_len : int
        Sequences longer than this are processed in overlapping windows and
        stitched, so very long chains do not exceed GPU memory.
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        cache_dir: Optional[str | Path] = None,
        device: str = "cuda",
        max_len: int = 1022,
    ) -> None:
        self.model_name = model_name
        self.dim = plm_dim_for(model_name)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_len = int(max_len)
        self._model = None
        self._bc = None
        self._repr_layer = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import esm  # lazy
        import torch

        loader = getattr(esm.pretrained, self.model_name)
        model, alphabet = loader()
        model = model.eval()
        if self.device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
        else:
            self.device = "cpu"
        self._model = model
        self._bc = alphabet.get_batch_converter()
        self._repr_layer = model.num_layers

    def _embed_raw(self, seq: str) -> np.ndarray:
        import torch

        self._ensure_model()
        # Replace any character ESM does not know with X to be safe.
        clean = "".join(c if c.isalpha() else "X" for c in seq.upper())
        if len(clean) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        if len(clean) <= self.max_len:
            windows = [(0, len(clean))]
        else:
            # Overlapping windows, averaged on the overlap region.
            step = self.max_len // 2
            windows = []
            start = 0
            while start < len(clean):
                end = min(start + self.max_len, len(clean))
                windows.append((start, end))
                if end == len(clean):
                    break
                start += step

        acc = np.zeros((len(clean), self.dim), dtype=np.float32)
        cov = np.zeros((len(clean),), dtype=np.float32)
        with torch.no_grad():
            for (s, e) in windows:
                sub = clean[s:e]
                _, _, toks = self._bc([("x", sub)])
                if self.device == "cuda":
                    toks = toks.cuda()
                out = self._model(toks, repr_layers=[self._repr_layer])
                rep = out["representations"][self._repr_layer][0, 1 : len(sub) + 1]
                acc[s:e] += rep.float().cpu().numpy()
                cov[s:e] += 1.0
        acc /= np.maximum(cov[:, None], 1.0)
        return acc.astype(np.float32)

    def embed_sequence(self, seq: str) -> np.ndarray:
        if self.cache_dir is not None and len(seq) > 0:
            key = _seq_hash(self.model_name, seq)
            path = self.cache_dir / f"{key}.npy"
            if path.exists():
                return np.load(path).astype(np.float32)
            emb = self._embed_raw(seq)
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, emb.astype(np.float16))
            tmp.replace(path)
            return emb
        return self._embed_raw(seq)

    def embed_residue_names(self, residue_names: list[str]) -> np.ndarray:
        seq = residue_names_to_sequence(residue_names)
        emb = self.embed_sequence(seq)
        # Guarantee exact alignment with the residue list.
        if emb.shape[0] != len(residue_names):
            fixed = np.zeros((len(residue_names), self.dim), dtype=np.float32)
            m = min(emb.shape[0], len(residue_names))
            fixed[:m] = emb[:m]
            emb = fixed
        return emb
