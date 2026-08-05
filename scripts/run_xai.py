#!/usr/bin/env python
"""XAI analysis for FastFusion-IF.

Subcommands
-----------
  keys        print the real contents of one collated batch (run this first)
  smoke       instrument one protein, check shapes and identities
  attribute   integrated gradients over a manifest split -> per-residue table
  ablate      modality interventions -> per-residue delta table
  faithful    deletion curves against a random baseline
  analyse     reliance / error statistics and article-ready tables (no GPU)

All stages write incrementally to ``--out-dir`` and skip proteins that already
have output, so a long run can be resumed after an interruption. Proteins that
raise are recorded in ``failed.csv`` with the traceback and the run continues.

Example
-------
    python scripts/run_xai.py attribute \\
        --checkpoint runs/bench_evo_pp/best.pt \\
        --manifest manifests/benchmark/bench_test315.csv --split test \\
        --cache-dir cache/bench_evo --config configs/xai_default.yaml \\
        --out-dir results/xai
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# Must be set before any CUDA context is created. Without it,
# torch.use_deterministic_algorithms() cannot make cuBLAS GEMMs deterministic
# and emits a warning on every matmul. torch is imported lazily below, so
# setting it here is early enough.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ------------------------------------------------------------------ utilities
def load_config(path: str | None) -> dict:
    defaults = {
        "seed": 0,
        "ig_steps": 64,
        "ig_target": "logit",
        "ig_scope": "total",
        "ig_baseline": None,   # None -> zero for scope=total, mean for scope=self
        "threshold": 0.63,
        "deletion_fractions": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        "interventions": [
            {"modality": "surface", "mode": "zero"},
            {"modality": "evolutionary", "mode": "zero"},
            {"modality": "evolutionary", "mode": "shuffle"},
            {"modality": "residue_context", "mode": "zero"},
        ],
        "bootstrap_n": 10000,
        "device": "auto",
    }
    if path:
        import yaml

        with open(path) as f:
            defaults.update(yaml.safe_load(f) or {})
    return defaults


def resolve_device(spec: str):
    import torch

    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_model(ckpt_path: str, device):
    """Rebuild the model from the checkpoint's own stored config.

    Mirrors ``scripts/evaluate.py`` exactly, including
    ``ExperimentConfig.from_dict`` rather than ``ModelConfig(**...)``: the stored
    dict carries nested data/train sections that the model construction path
    depends on. Reading the architecture from the checkpoint rather than from a
    config file on disk means an XAI run cannot silently analyse a
    differently-shaped model than the one that produced the reported metrics.

    Returns ``(model, cfg, ckpt)`` -- ``cfg`` is needed to build the dataset.
    """
    import torch

    from fastfusion_if.config import ExperimentConfig
    from fastfusion_if.models import FastFusionIF

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.from_dict(ck["cfg"])
    model = FastFusionIF(
        cfg.model,
        surface_feature_dim=int(ck["surface_feature_dim"]),
        residue_feature_dim=int(ck.get("residue_feature_dim", 0)),
        plm_dim=int(ck.get("plm_dim", 0)),
    )
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, cfg, ck


# Field names that differ between collate versions. Resolved at runtime and
# reported by `run_xai.py keys`, rather than guessed -- guessing a class name is
# what broke the first version of this script.
_LABEL_KEYS = ("labels", "label", "y", "residue_labels", "targets")
_GROUP_KEYS = ("group_id", "group_ids", "chain_id", "name", "source_path")


def resolve_key(batch: dict, candidates) -> str | None:
    for k in candidates:
        if k in batch and batch[k] is not None:
            return k
    return None


def diagnose_cache(files, cache_dir: str, pdb_root: str | None = None) -> str:
    """Explain a cache miss precisely instead of leaving 'No cache files found'.

    ``cache_path_for`` hashes the *source path string*, so a manifest holding
    ``AGAT-PPIS/Dataset/pdb/X.pdb`` and a cache built from
    ``/home/you/AGAT-PPIS/Dataset/pdb/X.pdb`` produce different filenames for the
    same structure. That is exactly what sanitising the committed manifests to
    relative paths does. This function detects it and names the fix.
    """
    from fastfusion_if.data.cached_dataset import cache_path_for

    cdir = Path(cache_dir)
    on_disk = {p.name for p in cdir.glob("*.pkl")} if cdir.is_dir() else set()
    wanted = [cache_path_for(cache_dir, f) for f in files]
    n_hit = sum(1 for w in wanted if w.name in on_disk)
    if n_hit:
        return ""

    lines = [
        "",
        "No cache file matched any manifest entry.",
        f"  manifest entries : {len(files)}",
        f"  .pkl in cache    : {len(on_disk)}",
        f"  cache dir        : {cdir}",
    ]
    if not cdir.is_dir():
        lines.append("  -> the cache directory does not exist. Check --cache-dir.")
        return "\n".join(lines)
    if not on_disk:
        lines.append("  -> the cache directory is empty. Run scripts/precompute_cache.py.")
        return "\n".join(lines)

    ex = str(files[0])
    lines += ["", f"  example manifest path : {ex}",
              f"  looked for            : {wanted[0].name}"]
    # Same structure, different hash => the path *string* differs, not the data.
    stem = Path(ex).name.replace(".", "_")
    same_stem = [n for n in on_disk if n.startswith(stem + "__")]
    if same_stem:
        lines += [
            f"  found on disk         : {same_stem[0]}",
            "",
            "  Same structure, different hash: the cache was built from a different",
            "  path STRING than the manifest now holds. cache_path_for() hashes the",
            "  full source path, so relative and absolute forms never match.",
            "",
            "  Fix, in order of preference:",
            "    1. point --manifest at the ORIGINAL manifests, whose paths match",
            "       the cache:  --manifest $PROJ/manifests/benchmark/<name>.csv",
            "    2. or pass --pdb-root to re-absolutise the relative entries, e.g.",
            "       --pdb-root ~/  (so AGAT-PPIS/... becomes ~/AGAT-PPIS/...)",
        ]
    else:
        lines += ["", "  No file with a matching structure name either; this cache was",
                  "  probably built for a different split or dataset."]
    return "\n".join(lines)


def iter_proteins(manifest: str, split: str, cache_dir: str, cfg,
                  batch_size: int = 1, pdb_root: str | None = None):
    """Yield one collated chain at a time, exactly as ``scripts/evaluate.py`` does.

    ``batch_size`` is fixed at 1. Attribution is per protein, and
    ``collate_chain_examples`` concatenates chains into one flat residue index,
    so a larger batch would silently mix proteins together in every downstream
    table without raising.

    ``pdb_root`` prefixes relative manifest paths. The committed manifests hold
    dataset-relative paths, but the feature cache is keyed by a hash of the full
    path string, so the two only line up if the same form is used for both. Pass
    ``--pdb-root`` when using the committed manifests, or point ``--manifest`` at
    the original absolute-path ones.

    ``collate_chain_examples`` may return ``None`` for an unusable example; those
    are skipped and counted rather than crashing the run.
    """
    from torch.utils.data import DataLoader

    from fastfusion_if.data.cached_dataset import CachedInterfaceDataset
    from fastfusion_if.data.collate import collate_chain_examples
    from fastfusion_if.data.dataset import ProteinInterfaceDataset
    from fastfusion_if.data.splits import read_manifest

    splits = read_manifest(manifest)
    files = splits.get(split, [])
    if not files:
        raise RuntimeError(f"No files for split={split!r} in {manifest}")

    if pdb_root:
        root = Path(pdb_root).expanduser()
        files = [str(f) if Path(f).is_absolute() else str(root / f) for f in files]

    if cache_dir:
        msg = diagnose_cache(files, cache_dir, pdb_root)
        if msg:
            raise FileNotFoundError(msg)
        ds = CachedInterfaceDataset.from_manifest_split(files, cache_dir, cfg.data, augment=False)
    else:
        ds = ProteinInterfaceDataset(files, cfg.data, with_labels=True, augment=False)

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_chain_examples,
    )
    n_skipped = 0
    for i, batch in enumerate(loader):
        if batch is None:
            n_skipped += 1
            continue
        if resolve_key(batch, _GROUP_KEYS) is None and i < len(files):
            batch = dict(batch)
            batch["group_id"] = Path(files[i]).stem
        yield batch
    if n_skipped:
        print(f"  [note] {n_skipped} example(s) skipped: collate returned None")


def batch_group_id(batch: dict, fallback: str = "unknown") -> str:
    k = resolve_key(batch, _GROUP_KEYS)
    if k is None:
        return fallback
    v = batch[k]
    if isinstance(v, (list, tuple)):
        v = v[0] if v else fallback
    return str(v)


# -------------------------------------------------------------------- commands
def cmd_smoke(args) -> int:
    """One protein, no writes: verify the instrumentation is sound."""
    import torch

    from fastfusion_if.xai import InstrumentedModel, integrated_gradients, modality_availability

    cfg = load_config(args.config)
    if getattr(args, "scope", None):
        cfg["ig_scope"] = args.scope
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    model, cfg_exp, ck = load_model(args.checkpoint, device)
    print(f"checkpoint      : {args.checkpoint}")
    print(f"device          : {device}")
    print(f"parameters      : {sum(p.numel() for p in model.parameters()):,}")
    print(f"modalities      : {modality_availability(model)}")

    batch = next(iter_proteins(args.manifest, args.split, args.cache_dir, cfg_exp,
                          pdb_root=getattr(args, "pdb_root", None)))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    with torch.no_grad():
        base = model(batch).clone()
    inst = InstrumentedModel(model)
    with inst.capture() as cap:
        with torch.no_grad():
            during = model(batch)
    assert torch.equal(base, during), "FAIL: hooks changed the forward pass"
    print("baseline identical with hooks installed : PASS")

    recon = cap.gate * cap.atom_res + (1 - cap.gate) * cap.surface_res
    ok = torch.allclose(recon, cap.z_geom, atol=1e-5)
    print(f"gate decomposition identity            : {'PASS' if ok else 'FAIL'}")
    print(f"gate shape / range                     : {tuple(cap.gate.shape)}  "
          f"[{cap.gate.min():.3f}, {cap.gate.max():.3f}]  mean {cap.gate.mean():.3f}")

    # The model is not bitwise reproducible: CUDA scatter reductions use
    # atomicAdd. Measure that spread so the checks below are read against it.
    with torch.no_grad():
        noise = float((model(batch) - model(batch)).abs().max())
    print(f"model run-to-run noise (atomicAdd)     : {noise:.3e}")

    a = integrated_gradients(model, batch, n_steps=64, target=cfg["ig_target"],
                             scope=cfg.get("ig_scope", "total"))
    ident = torch.allclose(a.atom + a.surface, a.geom, atol=1e-4)
    print(f"atom + surface == geom                 : {'PASS' if ident else 'FAIL'}")
    print(f"IG completeness, global (64 steps)     : "
          f"{a.relative_global_error:.4f}  {'PASS' if a.relative_global_error < 0.02 else 'FAIL'}")
    finite = all(torch.isfinite(t).all() for t in a.as_dict().values())
    print(f"all attributions finite                : {'PASS' if finite else 'FAIL'}")

    b = integrated_gradients(model, batch, n_steps=64, target=cfg["ig_target"],
                             scope="self", baseline="mean")
    rel = float((b.convergence_delta.abs() / (b.logit.abs().mean() + 1e-6)).mean())
    print(f"IG completeness, per-residue (self/mean): {rel:.4f}  "
          f"{'PASS' if rel < 0.05 else 'FAIL'}")

    z = integrated_gradients(model, batch, n_steps=64, target=cfg["ig_target"],
                             scope="self", baseline="zero")
    relz = float((z.convergence_delta.abs() / (z.logit.abs().mean() + 1e-6)).mean())
    print(f"  [diagnostic] same with a zero baseline: {relz:.4f}  "
          f"(expected to be poor: the head starts with a scale-invariant LayerNorm)")
    return 0


def cmd_keys(args) -> int:
    """Report the actual contents of one collated batch.

    Run this first on a new checkpoint or after any change to the collate
    function. It prints every key, its dtype and shape, and which of them the
    XAI code resolved as the label and group-id fields, so nothing downstream
    has to be guessed.
    """
    import torch

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    model, cfg_exp, ck = load_model(args.checkpoint, device)
    batch = next(iter_proteins(args.manifest, args.split, args.cache_dir, cfg_exp,
                          pdb_root=getattr(args, "pdb_root", None)))

    print(f"checkpoint : {args.checkpoint}")
    print(f"n keys     : {len(batch)}\n")
    print(f"{'key':32s} {'type':10s} {'dtype':12s} shape")
    print("-" * 78)
    for k in sorted(batch):
        v = batch[k]
        if torch.is_tensor(v):
            print(f"{k:32s} {'Tensor':10s} {str(v.dtype).replace('torch.',''):12s} {tuple(v.shape)}")
        elif isinstance(v, (list, tuple)):
            print(f"{k:32s} {type(v).__name__:10s} {'-':12s} len={len(v)}  first={v[0] if v else None!r}")
        else:
            print(f"{k:32s} {type(v).__name__:10s} {'-':12s} {v!r}")

    lk = resolve_key(batch, _LABEL_KEYS)
    gk = resolve_key(batch, _GROUP_KEYS)
    print("\nresolved by the XAI code:")
    print(f"  label key    : {lk!r}" + ("" if lk else "   <-- NOT FOUND: add it to _LABEL_KEYS"))
    print(f"  group-id key : {gk!r}" + ("" if gk else "   <-- NOT FOUND: add it to _GROUP_KEYS"))

    required = ["atom_elem", "atom_pos", "atom_edge_index", "atom2res",
                "surface2res", "n_residues", "residue_pos", "residue_edge_index"]
    missing = [k for k in required if k not in batch]
    print(f"\nrequired model inputs present : {'YES' if not missing else 'NO -> ' + str(missing)}")
    n_res = int(batch["n_residues"]) if "n_residues" in batch else -1
    print(f"n_residues in this batch      : {n_res}")
    if lk and torch.is_tensor(batch[lk]):
        n_lab = batch[lk].numel()
        ok = n_lab == n_res
        print(f"label count matches n_residues: {'YES' if ok else f'NO ({n_lab} vs {n_res})'}")
        if not ok:
            print("  -> batch_size must be 1 and one dataset item must be one chain;")
            print("     if this fails, the dataset yields multiple ChainExamples per item.")
    return 0


def cmd_attribute(args) -> int:
    import torch

    from fastfusion_if.xai import InstrumentedModel, integrated_gradients

    cfg = load_config(args.config)
    if getattr(args, "scope", None):
        cfg["ig_scope"] = args.scope
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    model, cfg_exp, _ = load_model(args.checkpoint, device)

    out = Path(args.out_dir)
    (out / "residue_attributions").mkdir(parents=True, exist_ok=True)
    (out / "gates").mkdir(parents=True, exist_ok=True)
    failed, quality = [], []

    for batch in iter_proteins(args.manifest, args.split, args.cache_dir, cfg_exp,
                          pdb_root=getattr(args, "pdb_root", None)):
        gid = batch_group_id(batch)
        safe = gid.replace("/", "_")
        dest = out / "residue_attributions" / f"{safe}.csv"
        gates_dest = out / "gates" / f"{safe}.npz"
        # Both artefacts must exist to count as done. Checking only the CSV
        # would permanently skip proteins whose CSV was written before a later
        # step failed.
        if dest.exists() and gates_dest.exists() and not args.overwrite:
            continue
        try:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            attr = integrated_gradients(model, b, n_steps=cfg["ig_steps"],
                                        target=cfg["ig_target"],
                                        scope=cfg.get("ig_scope", "total"),
                                        baseline=cfg.get("ig_baseline"))
            with InstrumentedModel(model).capture() as cap:
                with torch.no_grad():
                    model(b)
            df = pd.DataFrame({k: v.numpy() for k, v in attr.as_dict().items()})
            df.insert(0, "group_id", gid)
            df.insert(1, "res_index", np.arange(len(df)))
            lk = resolve_key(batch, _LABEL_KEYS)
            if lk is not None:
                lv = batch[lk]
                df["label"] = (lv.detach().cpu().numpy() if torch.is_tensor(lv)
                               else np.asarray(lv)).reshape(-1)[: len(df)]
            for k in ("res_chain", "res_seq", "insertion", "res_name"):
                if k in batch and batch[k] is not None:
                    v = batch[k]
                    v = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
                    if len(v) == len(df):
                        df[k] = v
            df["ig_global_rel_error"] = attr.relative_global_error
            df["ig_scope"] = attr.scope
            df["ig_baseline"] = attr.baseline
            df.to_csv(dest, index=False)
            quality.append({"group_id": gid, "n_residues": len(df),
                            "ig_global_rel_error": attr.relative_global_error,
                            "ig_scope": attr.scope})
            np.savez_compressed(gates_dest,
                                gate=cap.gate.numpy(), atom_res=cap.atom_res.numpy(),
                                surface_res=cap.surface_res.numpy())
        except Exception as e:  # keep going; record what broke
            failed.append({"group_id": gid, "error": repr(e), "traceback": traceback.format_exc()})
            print(f"  [fail] {gid}: {e}")
            # An OOM leaves the caching allocator fragmented; without this a
            # single oversized chain can make every later protein fail too.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if quality:
        q = pd.DataFrame(quality)
        q.to_csv(out / "attribution_quality.csv", index=False)
        bad = q[q.ig_global_rel_error > 0.02]
        print(f"IG completeness: median {q.ig_global_rel_error.median():.4f}, "
              f"{len(bad)}/{len(q)} proteins above the 0.02 threshold")
        if len(bad):
            print(f"  -> raise ig_steps in {args.config} and re-run those "
                  f"(listed in attribution_quality.csv)")
    fail_path = out / "failed.csv"
    if failed:
        pd.DataFrame(failed).to_csv(fail_path, index=False)
        print(f"{len(failed)} proteins failed; see {fail_path}")
    elif fail_path.exists():
        fail_path.unlink()  # stale file from an earlier run would mislead
        print("all proteins succeeded (removed the stale failed.csv)")
    print(f"wrote per-residue attributions to {out / 'residue_attributions'}")
    return 0


def cmd_ablate(args) -> int:
    import torch

    from fastfusion_if.xai import run_intervention

    cfg = load_config(args.config)
    if getattr(args, "scope", None):
        cfg["ig_scope"] = args.scope
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    model, cfg_exp, _ = load_model(args.checkpoint, device)
    out = Path(args.out_dir) / "modality_ablation"
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device="cpu").manual_seed(cfg["seed"])

    rows = []
    for batch in iter_proteins(args.manifest, args.split, args.cache_dir, cfg_exp,
                          pdb_root=getattr(args, "pdb_root", None)):
        gid = batch_group_id(batch)
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        for spec in cfg["interventions"]:
            if spec["modality"] == "plm" and getattr(model, "plm_proj", None) is None:
                continue  # cache carries residue_plm but this checkpoint ignores it
            try:
                r = run_intervention(model, b, spec["modality"], spec.get("mode", "zero"), gen)
                rows.append({
                    "group_id": gid, "intervention": r.name,
                    "mean_abs_delta_prob": r.mean_abs_delta_prob,
                    "mean_delta_logit": float(np.mean(r.delta_logit)),
                    "max_abs_delta_logit": float(np.max(np.abs(r.delta_logit))),
                    "n_residues": len(r.delta_logit),
                })
            except KeyError as e:
                print(f"  [skip] {gid} {spec}: {e}")
    pd.DataFrame(rows).to_csv(out / "intervention_summary.csv", index=False)
    print(f"wrote {out / 'intervention_summary.csv'}")
    return 0


def cmd_faithful(args) -> int:
    """Deletion curves against a random baseline.

    A single deletion curve proves nothing -- some proteins collapse under any
    perturbation. The statistic is the area *between* the attribution-ordered
    curve and the random-order curve, averaged over repeats. Zero or negative
    means the attribution carries no more information than chance.
    """
    import torch

    from fastfusion_if.xai import deletion_curve, faithfulness_gap

    cfg = load_config(args.config)
    if getattr(args, "scope", None):
        cfg["ig_scope"] = args.scope
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    model, cfg_exp, _ = load_model(args.checkpoint, device)

    out = Path(args.out_dir) / "faithfulness"
    out.mkdir(parents=True, exist_ok=True)
    attr_dir = Path(args.out_dir) / "residue_attributions"
    fracs = tuple(cfg["deletion_fractions"])
    n_rep = int(cfg.get("n_random_repeats", 5))
    score_col = args.score_col

    rows, curves = [], []
    for batch in iter_proteins(args.manifest, args.split, args.cache_dir, cfg_exp,
                          pdb_root=getattr(args, "pdb_root", None)):
        gid = batch_group_id(batch)
        f = attr_dir / f"{gid.replace('/', '_')}.csv"
        if not f.exists():
            continue
        scores = pd.read_csv(f)[score_col].to_numpy(dtype=float)
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        lk = resolve_key(b, _LABEL_KEYS)
        if lk is None:
            print(f"  [skip] {gid}: no label key in batch"); continue
        b = dict(b); b["labels"] = b[lk]
        if "n_residues" not in b:
            print(f"  [skip] {gid}: batch has no 'n_residues'; run `keys` to inspect")
            continue
        if len(scores) != int(b["n_residues"]):
            print(f"  [skip] {gid}: {len(scores)} scores vs {int(b['n_residues'])} residues")
            continue
        try:
            desc = deletion_curve(model, b, scores, fractions=fracs, order="descending")
            gaps = []
            for r in range(n_rep):
                g = torch.Generator().manual_seed(cfg["seed"] + r)
                rnd = deletion_curve(model, b, scores, fractions=fracs, order="random",
                                     generator=g)
                gaps.append(faithfulness_gap(desc, rnd))
                curves.append({"group_id": gid, "order": f"random{r}",
                               **{f"auprc@{x:.2f}": y for x, y in
                                  zip(rnd["fraction"], rnd["auprc"])}})
            curves.append({"group_id": gid, "order": "descending",
                           **{f"auprc@{x:.2f}": y for x, y in
                              zip(desc["fraction"], desc["auprc"])}})
            rows.append({"group_id": gid, "score_col": score_col,
                         "faithfulness_gap_mean": float(np.nanmean(gaps)),
                         "faithfulness_gap_std": float(np.nanstd(gaps)),
                         "n_repeats": n_rep, "n_residues": len(scores)})
        except Exception as e:
            print(f"  [fail] {gid}: {e}")

    if not rows:
        print("no proteins processed; run `attribute` first")
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(out / f"faithfulness_{score_col}.csv", index=False)
    pd.DataFrame(curves).to_csv(out / f"deletion_curves_{score_col}.csv", index=False)
    g = df["faithfulness_gap_mean"]
    print(f"\nfaithfulness gap ({score_col}), {len(df)} proteins")
    print(f"  mean   : {g.mean():+.4f}")
    print(f"  median : {g.median():+.4f}")
    print(f"  > 0    : {int((g > 0).sum())}/{len(g)} proteins")
    print(f"\nwrote {out}/faithfulness_{score_col}.csv")
    return 0


def cmd_analyse(args) -> int:
    """Statistics from the attribution tables. Runs without torch."""
    from fastfusion_if.xai import (aggregate_by_protein, bootstrap_ci, confusion_class,
                                   error_analysis, holm_bonferroni, modality_reliance,
                                   paired_test, select_cases)

    cfg = load_config(args.config)
    out = Path(args.out_dir)
    files = sorted((out / "residue_attributions").glob("*.csv"))
    if not files:
        print(f"no attribution files in {out / 'residue_attributions'}; run `attribute` first")
        return 1
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = modality_reliance(df)
    if "label" in df.columns:
        df["confusion"] = confusion_class(df["label"], df["prob"], cfg["threshold"])
        error_analysis(df).to_csv(out / "error_analysis" / "by_confusion.csv", index=False)
    per_protein = aggregate_by_protein(df)
    per_protein.to_csv(out / "tables" / "per_protein_reliance.csv", index=False)

    rows = []
    for m in [c for c in per_protein.columns if c.startswith("reliance_")]:
        ci = bootstrap_ci(per_protein[m].values, n_boot=cfg["bootstrap_n"], seed=cfg["seed"])
        rows.append({"modality": m.replace("reliance_", ""), "mean": ci.mean,
                     "ci_lo": ci.lo, "ci_hi": ci.hi, "n_proteins": ci.n})
    pd.DataFrame(rows).to_csv(out / "tables" / "global_modality_reliance.csv", index=False)
    df.to_csv(out / "tables" / "per_residue_reliance.csv.gz", index=False, compression="gzip")
    print(f"wrote reliance tables to {out / 'tables'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, needs_model=True):
        if needs_model:
            sp.add_argument("--checkpoint", required=True)
            sp.add_argument("--manifest", required=True)
            sp.add_argument("--split", default="test")
            sp.add_argument("--cache-dir", required=True)
        if needs_model:
            sp.add_argument("--scope", default=None, choices=["total", "self"],
                            help="attribution scope; 'total' (default) gives global "
                                 "completeness, 'self' gives per-residue completeness "
                                 "by bypassing the residue-context encoder")
            sp.add_argument("--pdb-root", default=None,
                            help="prefix for relative manifest paths; needed when using the "
                                 "committed manifests, whose paths were sanitised to be relative")
        sp.add_argument("--config", default="configs/xai_default.yaml")
        sp.add_argument("--out-dir", default="results/xai")
        sp.add_argument("--overwrite", action="store_true")

    for name, fn, needs in [("keys", cmd_keys, True), ("smoke", cmd_smoke, True),
                            ("attribute", cmd_attribute, True),
                            ("ablate", cmd_ablate, True), ("faithful", cmd_faithful, True),
                            ("analyse", cmd_analyse, False)]:
        sp = sub.add_parser(name, help=fn.__doc__)
        common(sp, needs)
        if name == "faithful":
            sp.add_argument("--score-col", default="attr_evolutionary",
                            help="attribution column to rank residues by")
        sp.set_defaults(func=fn)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
