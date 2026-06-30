#!/usr/bin/env python
"""Precompute and cache processed ChainExamples for FastFusion-IF.

This runs your EXISTING make_chain_example() pipeline once per structure and
pickles the resulting list[ChainExample] to disk, so training/eval can load
tensors instead of recomputing the surface point cloud, graphs and labels every
epoch. It reuses your tested code paths, so it does not change any model logic.

Usage (run once per manifest; safe to interrupt and re-run, it skips done files):

    python scripts/precompute_cache.py \
        --manifest manifests/dips_plus_mmseqs30_large.csv \
        --config   configs/large_v2_resctx.json \
        --cache-dir cache/dips_plus_v2 \
        --splits train,val,test

Tips:
  * Run several shards in parallel terminals with --shard i/N to use more cores
    (each process handles files where (file_index % N == i)).
  * The cache is keyed by source path hash, so the SAME cache directory can be
    shared by the debug/large/full manifests (they reference the same files).
  * Augmentation is intentionally NOT baked into the cache; CachedInterfaceDataset
    applies rotation/jitter at load time.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastfusion_if.config import ExperimentConfig
from fastfusion_if.data.cached_dataset import cache_path_for
from fastfusion_if.data.dataset import make_chain_example, parse_any_atoms
from fastfusion_if.data.pdb_parser import atoms_by_chain
from fastfusion_if.data.splits import read_manifest_rows


def _apply_feature_override(ex, override, evo_dim=None) -> None:
    """Set an example's residue_features to externally-supplied per-residue vectors
    (e.g. PSSM|HMM|DSSP), aligned by residue key. When this benchmark uses evo
    features but a protein has none (override is empty), fall back to a zero vector
    of width evo_dim so the residue_features channel stays a uniform width across
    the whole cache (otherwise collate would crash on mixed widths)."""
    import numpy as np

    if override:
        dim = 0
        for v in override.values():
            dim = len(v)
            break
        if dim <= 0:
            return
        ex.residue_features = np.array(
            [override.get(f"{k[0]}:{k[1]}:{k[2]}", [0.0] * dim) for k in ex.residue_keys],
            dtype=np.float32,
        )
    elif evo_dim:
        ex.residue_features = np.zeros((len(ex.residue_keys), int(evo_dim)), dtype=np.float32)


def process_one(path: str, cfg: ExperimentConfig, with_labels: bool, plm_extractor=None, labels_map=None, evo_map=None, evo_dim=None):
    atoms = parse_any_atoms(path, cfg.data)
    chains = atoms_by_chain(atoms)
    chain_items = sorted(chains.items(), key=lambda kv: len(kv[1]), reverse=True)
    # Benchmark mode: a per-protein residue->label override keyed by the file stem.
    override = labels_map.get(Path(path).stem) if labels_map is not None else None
    feat_override = evo_map.get(Path(path).stem) if evo_map is not None else None
    examples = []
    for chain_id, chain_atoms in chain_items:
        if len(chain_atoms) < 5:
            continue
        # The <2-chain skip exists only because partner labels need a partner; when
        # an explicit label override is supplied (single-chain benchmark PDBs) we keep it.
        if with_labels and override is None and len(chains) < 2:
            continue
        ex = make_chain_example(
            chain_atoms=chain_atoms,
            all_atoms=atoms,
            cfg=cfg.data,
            source_path=str(path),
            with_labels=with_labels,
            seed=0,
            augment=False,  # augmentation is applied at load time
            label_override=override,
        )
        if evo_map is not None:
            # Uniform-width evo channel: present -> profiles, missing -> zeros(evo_dim).
            _apply_feature_override(ex, feat_override, evo_dim)
        if plm_extractor is not None:
            # Built directly from residue_names -> guaranteed aligned with residues.
            ex.residue_plm = plm_extractor.embed_residue_names(ex.residue_names)
        examples.append(ex)
    return examples


def process_from_cache(src_file: Path, plm_extractor, evo_override=None, evo_dim=None, has_evo=False) -> list:
    """Load already-cached examples (geometry/surface/labels done) and only attach
    PLM embeddings and/or external residue features. Avoids recomputing the
    expensive geometry when you already have a cache for the same files."""
    with src_file.open("rb") as f:
        examples = pickle.load(f)
    for ex in examples:
        if plm_extractor is not None and getattr(ex, "residue_plm", None) is None:
            ex.residue_plm = plm_extractor.embed_residue_names(ex.residue_names)
        if has_evo:
            _apply_feature_override(ex, evo_override, evo_dim)
    return examples


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute FastFusion-IF example cache")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--with-labels", type=int, default=1)
    ap.add_argument("--shard", default="0/1", help="i/N to process a subset of files")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--plm-model", default=None, help="If set (e.g. esm2_t33_650M_UR50D), bake ESM-2 residue embeddings into the cache.")
    ap.add_argument("--plm-cache-dir", default=None, help="Per-sequence ESM embedding cache (defaults to <cache-dir>/_plm).")
    ap.add_argument("--plm-device", default="cuda")
    ap.add_argument("--from-cache", default=None, help="Reuse geometry from an existing cache dir and only add PLM embeddings (much faster than rebuilding; requires --plm-model).")
    ap.add_argument("--labels-file", default=None, help="Benchmark mode: JSON mapping protein_id -> {residue_key: 0/1} from prepare_benchmark.py. Uses these labels instead of the 5A partner rule, and keeps single-chain structures.")
    ap.add_argument("--evo-file", default=None, help="Benchmark mode: bench_evo.npz from prepare_benchmark.py (per-protein standardized PSSM|HMM|DSSP profiles). Stored into the residue_features channel.")
    ap.add_argument("--evo-keys", default=None, help="bench_evo_keys.json (defaults to <evo-file dir>/bench_evo_keys.json).")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_json(args.config) if args.config else ExperimentConfig()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    wanted = {s.strip() for s in args.splits.split(",") if s.strip()}
    i_shard, n_shard = (int(x) for x in args.shard.split("/"))

    plm_extractor = None
    if args.plm_model:
        from fastfusion_if.data.plm import ESM2Extractor

        plm_cache = args.plm_cache_dir or str(cache_dir / "_plm")
        plm_extractor = ESM2Extractor(model_name=args.plm_model, cache_dir=plm_cache, device=args.plm_device)
        print(f"PLM embeddings ON: {args.plm_model} (dim={plm_extractor.dim}), seq-cache={plm_cache}")

    # Benchmark evolutionary/structural features (PSSM|HMM|DSSP) -> residue_features.
    evo_map = None
    evo_dim = 0
    if args.evo_file:
        import numpy as np

        evo_keys_path = args.evo_keys or str(Path(args.evo_file).with_name("bench_evo_keys.json"))
        with open(evo_keys_path) as f:
            evo_keys = json.load(f)
        npz = np.load(args.evo_file, allow_pickle=False)
        evo_dim = int(npz["__mean__"].shape[0]) if "__mean__" in npz.files else 0
        evo_map = {}
        for pid in evo_keys:
            if pid not in npz.files:
                continue
            arr = npz[pid]
            keys = evo_keys[pid]
            n = min(len(keys), int(arr.shape[0]))
            evo_map[pid] = {keys[i]: arr[i].tolist() for i in range(n)}
            if evo_dim == 0 and arr.ndim == 2:
                evo_dim = int(arr.shape[1])
        print(f"Benchmark evo features ON: {len(evo_map)} proteins (dim={evo_dim}) from {args.evo_file} "
              f"-> residue_features channel (proteins without features get a zero vector of width {evo_dim}).")

    from_cache_dir = None
    if args.from_cache:
        if plm_extractor is None and evo_map is None:
            raise SystemExit("--from-cache only makes sense together with --plm-model and/or --evo-file (it adds those to existing geometry).")
        from_cache_dir = Path(args.from_cache)
        if not from_cache_dir.exists():
            raise SystemExit(f"--from-cache dir does not exist: {from_cache_dir}")
        added = "+".join([x for x in ["PLM" if plm_extractor is not None else "", "evo" if evo_map is not None else ""] if x])
        print(f"Reusing geometry from {from_cache_dir} (will only add {added}; falls back to full build for any missing file).")

    rows = read_manifest_rows(args.manifest)
    rows = [r for r in rows if r.get("split", "") in wanted]
    if args.limit:
        rows = rows[: args.limit]

    labels_map = None
    if args.labels_file:
        with open(args.labels_file) as f:
            labels_map = json.load(f)
        print(f"Benchmark labels ON: {len(labels_map)} proteins from {args.labels_file} (single-chain PDBs kept).")

    done = skipped = failed = 0
    t0 = time.time()
    for idx, row in enumerate(rows):
        if idx % n_shard != i_shard:
            continue
        path = row["path"]
        out = cache_path_for(cache_dir, path)
        if out.exists():
            # Skip only when nothing new is being attached. When building a fresh
            # cache dir (the evo case), out won't exist so we always build.
            if plm_extractor is None and evo_map is None:
                skipped += 1
                continue
            if evo_map is None and plm_extractor is not None:
                # Only PLM requested: skip when the cached example already has it.
                try:
                    with out.open("rb") as f:
                        cached = pickle.load(f)
                    if cached and getattr(cached[0], "residue_plm", None) is not None:
                        skipped += 1
                        continue
                except Exception:
                    pass  # rebuild on any read problem
        try:
            src = cache_path_for(from_cache_dir, path) if from_cache_dir is not None else None
            evo_override = evo_map.get(Path(path).stem) if evo_map is not None else None
            if src is not None and src.exists():
                examples = process_from_cache(src, plm_extractor, evo_override=evo_override,
                                              evo_dim=evo_dim, has_evo=evo_map is not None)  # reuse geometry
            else:
                examples = process_one(path, cfg, with_labels=bool(args.with_labels), plm_extractor=plm_extractor,
                                       labels_map=labels_map, evo_map=evo_map, evo_dim=evo_dim)
            tmp = out.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(out)  # atomic, so interrupted writes never corrupt the cache
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {path}: {exc}")
        if (done + skipped) % 200 == 0:
            rate = done / max(1e-6, time.time() - t0)
            print(
                f"shard {i_shard}/{n_shard}  done={done} skipped={skipped} "
                f"failed={failed}  {rate:.1f} files/s"
            )

    print(f"FINISHED shard {i_shard}/{n_shard}: done={done} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
