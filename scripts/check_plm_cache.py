#!/usr/bin/env python
"""Sanity-check ESM-2 embeddings baked into a precomputed cache.

A flat ESM result usually means one of three things: the embeddings are missing,
they are degenerate (a per-amino-acid lookup with no context), or they are
misaligned with the residues. This script inspects the actual cached arrays and
reports on all three, with no training involved.

Usage:
    python scripts/check_plm_cache.py --cache-dir cache/dips_plus_v2_esm --n 200
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from fastfusion_if.data.sequences import AA3_TO_AA1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--n", type=int, default=200, help="How many cached files to scan.")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    files = [p for p in sorted(cache_dir.glob("*.pkl"))][: args.n]
    if not files:
        raise SystemExit(f"No .pkl files in {cache_dir}")

    n_examples = 0
    n_with_plm = 0
    dims = set()
    all_norms = []
    aligned_ok = 0
    finite_ok = 0
    # contextual test: collect, per amino-acid letter, embeddings from many
    # positions; if ESM is contextual the SAME letter has DIFFERENT vectors.
    by_letter: dict[str, list[np.ndarray]] = {}
    seq_preview = None

    for fp in files:
        try:
            with fp.open("rb") as f:
                examples = pickle.load(f)
        except Exception:
            continue
        for ex in examples:
            n_examples += 1
            plm = getattr(ex, "residue_plm", None)
            names = list(getattr(ex, "residue_names", []))
            if plm is None:
                continue
            n_with_plm += 1
            plm = np.asarray(plm)
            dims.add(plm.shape[-1])
            if plm.shape[0] == len(names):
                aligned_ok += 1
            if np.all(np.isfinite(plm)):
                finite_ok += 1
            norms = np.linalg.norm(plm, axis=-1)
            all_norms.append(norms)
            if seq_preview is None and names:
                seq_preview = "".join(AA3_TO_AA1.get(str(n).upper(), "X") for n in names)[:80]
            for i, nm in enumerate(names):
                letter = AA3_TO_AA1.get(str(nm).upper(), "X")
                if len(by_letter.get(letter, [])) < 50 and i < plm.shape[0]:
                    by_letter.setdefault(letter, []).append(plm[i])

    print(f"scanned files            : {len(files)}")
    print(f"examples                 : {n_examples}")
    print(f"examples with residue_plm: {n_with_plm}  ({100*n_with_plm/max(1,n_examples):.0f}%)")
    print(f"embedding dim(s)         : {sorted(dims)}")
    print(f"row-count aligned        : {aligned_ok}/{n_with_plm}")
    print(f"all-finite               : {finite_ok}/{n_with_plm}")
    if all_norms:
        cat = np.concatenate(all_norms)
        print(f"per-residue L2 norm      : mean={cat.mean():.3f} std={cat.std():.3f} min={cat.min():.3f} max={cat.max():.3f}")
    if seq_preview:
        print(f"reconstructed seq (head) : {seq_preview}")

    # Contextuality: within-letter cosine spread. ~1.0 everywhere => degenerate
    # (context-free lookup). Clearly <1.0 => real contextual ESM.
    print("\ncontextuality check (same amino acid, different positions):")
    worst = 1.0
    for letter, vecs in sorted(by_letter.items()):
        if len(vecs) < 5:
            continue
        V = np.stack(vecs)
        Vn = V / np.clip(np.linalg.norm(V, axis=-1, keepdims=True), 1e-8, None)
        # mean pairwise cosine via the mean direction
        mean_dir = Vn.mean(0)
        mean_cos = float(np.clip((Vn @ mean_dir) / max(np.linalg.norm(mean_dir), 1e-8), -1, 1).mean())
        worst = min(worst, mean_cos)
        print(f"  {letter}: n={len(vecs):3d}  mean cosine to centroid = {mean_cos:.3f}")
    verdict = "REAL CONTEXTUAL ESM (good)" if worst < 0.98 else "WARNING: embeddings look context-free / degenerate"
    print(f"\nverdict: {verdict}")


if __name__ == "__main__":
    main()
