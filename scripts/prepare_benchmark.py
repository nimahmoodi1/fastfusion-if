#!/usr/bin/env python
"""Convert the AGAT-PPIS / GraphPPIS benchmark into FastFusion-IF manifests + a
label-override JSON, so we can train on Train_335 and test on Test_60 / Test_315-28 /
UBtest_31-6 and drop the numbers straight into those papers' tables.

The benchmark ships:
  Dataset/<Split>.pkl  : pickled {protein_id: [sequence, labels]} (labels = 0/1 per residue)
  Dataset/pdb/<id>.pdb : single-chain structure for each protein_id

We parse each PDB with our own parser, align the benchmark's per-residue labels onto
our residues (exact when lengths match, otherwise via longest-matching-blocks so a few
missing/extra residues don't corrupt the mapping), and emit:
  bench_labels.json    : {protein_id: {"chain:resseq:icode": 0/1}}  (for precompute_cache --labels-file)
  bench_all.csv        : every unique protein (split=all) -> build the cache in one pass
  bench_train.csv      : Train_335 split into train/val + Test_60 as test (for train.py)
  bench_test60.csv / bench_test315.csv / bench_ubtest.csv : split=test (for evaluate.py)
  bench_alignment_report.csv : per-protein match ratio (inspect anything < 1.0)

Usage
-----
python scripts/prepare_benchmark.py \
    --dataset-dir ~/Nima/AGAT-PPIS/Dataset \
    --config configs/full_v2_cached.json \
    --out-dir manifests/benchmark --val-frac 0.1
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# filename -> logical role
SPLIT_FILES = {
    "Train_335.pkl": "train_pool",
    "Test_60.pkl": "test60",
    "Test_315-28.pkl": "test315",
    "Test_315.pkl": "test315_full",
    "UBtest_31-6.pkl": "ubtest",
}


def load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def align_rows(residue_keys, our_seq: str, pkl_seq: str, rows):
    """Align a (L_pkl, D) per-residue feature matrix onto OUR residues, returning a
    (L_our, D) array (rows for unmatched residues are NaN) plus the match ratio.

    Uses the SAME longest-matching-blocks alignment as align_labels so the feature
    rows and the labels stay consistent residue-for-residue.
    """
    import numpy as _np

    n_our = len(residue_keys)
    d = int(rows.shape[1])
    out = _np.full((n_our, d), _np.nan, dtype=_np.float32)
    if len(our_seq) == len(pkl_seq) == n_our == int(rows.shape[0]):
        return rows.astype(_np.float32).copy(), 1.0
    matched = 0
    sm = difflib.SequenceMatcher(a=our_seq, b=pkl_seq, autojunk=False)
    for block in sm.get_matching_blocks():
        for t in range(block.size):
            ri = block.a + t          # our residue index
            li = block.b + t          # pkl row index
            if ri < n_our and li < int(rows.shape[0]):
                out[ri] = rows[li]
                matched += 1
    denom = max(1, min(n_our, int(rows.shape[0])))
    return out, matched / denom


def align_labels(residue_keys, our_seq: str, pkl_seq: str, pkl_labels: list[int]):
    """Return ({res_key_str: label}, match_ratio). Exact zip when lengths match;
    otherwise map labels through longest matching blocks (unmatched residues -> 0)."""
    def kstr(k):
        return f"{k[0]}:{k[1]}:{k[2]}"

    out = {kstr(k): 0 for k in residue_keys}
    if len(our_seq) == len(pkl_seq) == len(residue_keys):
        for k, lab in zip(residue_keys, pkl_labels):
            out[kstr(k)] = int(lab)
        return out, 1.0

    matched = 0
    sm = difflib.SequenceMatcher(a=our_seq, b=pkl_seq, autojunk=False)
    for block in sm.get_matching_blocks():
        for t in range(block.size):
            ri = block.a + t          # index into our residues
            li = block.b + t          # index into pkl labels
            if ri < len(residue_keys) and li < len(pkl_labels):
                out[kstr(residue_keys[ri])] = int(pkl_labels[li])
                matched += 1
    denom = max(1, min(len(residue_keys), len(pkl_seq)))
    return out, matched / denom


def write_manifest(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "split"])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="AGAT-PPIS/Dataset")
    ap.add_argument("--config", required=True, help="config json for parser settings")
    ap.add_argument("--out-dir", default="manifests/benchmark")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--protocol",
        choices=["paper", "holdout"],
        default="paper",
        help=(
            "paper (default): train=Train_335 (all 335), val=Test_60, test=Test_315-28 -- "
            "the GraphPPIS/AGAT-PPIS convention, so numbers drop straight into their tables. "
            "holdout: split Train_335 into train/val by --val-frac and use Test_60 as the in-loop test."
        ),
    )
    ap.add_argument(
        "--feature-dir",
        default=None,
        help=(
            "Optional AGAT-PPIS/Feature directory. If set, the precomputed per-residue "
            "profiles (pssm/hmm/dssp/resAF) are loaded, key-aligned onto our residues, "
            "z-scored with TRAIN-only stats, and written to bench_evo.npz (+bench_evo_keys.json) "
            "for precompute_cache --evo-file. These are the same features the published methods use."
        ),
    )
    ap.add_argument("--feature-sets", default="pssm,hmm,dssp,resAF",
                    help="Comma-separated feature subdirs under --feature-dir to concatenate.")
    args = ap.parse_args()

    from fastfusion_if.config import ExperimentConfig
    from fastfusion_if.data.dataset import parse_any_atoms
    from fastfusion_if.data.labels import residue_table
    from fastfusion_if.data.sequences import AA3_TO_AA1

    cfg = ExperimentConfig.from_dict(json.load(open(args.config)))
    ds_dir = Path(args.dataset_dir).expanduser()
    pdb_dir = ds_dir / "pdb"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    feature_dir = Path(args.feature_dir).expanduser() if args.feature_dir else None
    feature_sets = tuple(s.strip() for s in args.feature_sets.split(",") if s.strip())
    evo_rows: dict[str, "np.ndarray"] = {}     # pid -> (L_our, D) aligned, NaN where unmatched
    evo_keys: dict[str, list[str]] = {}        # pid -> [residue_key_str] in our order
    evo_train_pids: list[str] = []             # train_pool pids, for standardization stats
    evo_widths_ref = None
    evo_missing: list[tuple[str, str]] = []
    if feature_dir is not None:
        from fastfusion_if.data.benchmark_features import load_raw_features, feature_set_layout

    labels_all: dict[str, dict] = {}
    members: dict[str, list[str]] = {}     # role -> [protein_id]
    report = []

    for fname, role in SPLIT_FILES.items():
        fpath = ds_dir / fname
        if not fpath.exists():
            print(f"[skip] {fname} not found")
            continue
        data = load_pkl(fpath)
        ids = []
        for pid, val in data.items():
            pdb = pdb_dir / f"{pid}.pdb"
            if not pdb.exists():
                report.append({"protein_id": pid, "role": role, "status": "no_pdb", "match_ratio": 0.0, "n_res": 0})
                continue
            try:
                pkl_seq = str(val[0])
                pkl_labels = [int(x) for x in val[1]]
                atoms = parse_any_atoms(str(pdb), cfg.data)
                _, residue_keys, residue_names = residue_table(atoms)
                our_seq = "".join(AA3_TO_AA1.get(n, "X") for n in residue_names)
                per_res, ratio = align_labels(residue_keys, our_seq, pkl_seq, pkl_labels)
                labels_all[pid] = per_res
                ids.append(pid)
                report.append({"protein_id": pid, "role": role, "status": "ok",
                               "match_ratio": round(ratio, 4), "n_res": len(residue_keys)})
                if feature_dir is not None and pid not in evo_rows:
                    raw, info = load_raw_features(feature_dir, pid, feature_sets)
                    if raw is None:
                        evo_missing.append((pid, info))
                    else:
                        aligned, _ = align_rows(residue_keys, our_seq, pkl_seq, raw)
                        evo_rows[pid] = aligned
                        evo_keys[pid] = [f"{k[0]}:{k[1]}:{k[2]}" for k in residue_keys]
                        if evo_widths_ref is None:
                            evo_widths_ref = info
                        if role == "train_pool":
                            evo_train_pids.append(pid)
            except Exception as e:  # noqa: BLE001
                report.append({"protein_id": pid, "role": role, "status": f"err:{type(e).__name__}", "match_ratio": 0.0, "n_res": 0})
        members[role] = ids
        print(f"{fname:<18} role={role:<12} proteins_ok={len(ids)}")

    def manifest_rows(ids, split):
        return [{"path": str((pdb_dir / f"{pid}.pdb").resolve()), "split": split} for pid in ids]

    pool = list(members.get("train_pool", []))
    test60_ids = list(members.get("test60", []))
    test315_ids = list(members.get("test315", []))
    ubtest_ids = list(members.get("ubtest", []))

    if args.protocol == "paper":
        # GraphPPIS / AGAT-PPIS convention: train on ALL of Train_335, select the best
        # epoch on Test_60, report the held-out Test_315-28 (and UBtest). bench_train.csv
        # carries all three splits so train.py auto-evaluates Test_315-28 at the end.
        train_ids = pool
        val_ids = test60_ids
        in_loop_test_ids = test315_ids
    else:
        # holdout: carve a val slice out of Train_335, use Test_60 as the in-loop test.
        random.Random(args.seed).shuffle(pool)
        n_val = int(round(len(pool) * args.val_frac))
        val_ids, train_ids = pool[:n_val], pool[n_val:]
        in_loop_test_ids = test60_ids

    train_rows = manifest_rows(train_ids, "train")
    val_rows = manifest_rows(val_ids, "val")
    write_manifest(out / "bench_train.csv", train_rows + val_rows + manifest_rows(in_loop_test_ids, "test"))

    # standalone eval manifests (always split=test) for evaluate.py
    test60_rows = manifest_rows(test60_ids, "test")
    test315_rows = manifest_rows(test315_ids, "test")
    ubtest_rows = manifest_rows(ubtest_ids, "test")
    if test315_rows:
        write_manifest(out / "bench_test315.csv", test315_rows)
    if ubtest_rows:
        write_manifest(out / "bench_ubtest.csv", ubtest_rows)
    if test60_rows:
        write_manifest(out / "bench_test60.csv", test60_rows)

    # one manifest with everything (split=all) for a single cache build
    all_ids = sorted(set(sum((members.get(r, []) for r in members), [])))
    write_manifest(out / "bench_all.csv", manifest_rows(all_ids, "all"))

    (out / "bench_labels.json").write_text(json.dumps(labels_all))
    with open(out / "bench_alignment_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["protein_id", "role", "status", "match_ratio", "n_res"])
        w.writeheader()
        w.writerows(report)

    # ---- evolutionary/structural feature export (optional) -------------------
    if feature_dir is not None and evo_rows:
        dims = {int(a.shape[1]) for a in evo_rows.values()}
        if len(dims) != 1:
            print(f"  [evo SKIPPED] inconsistent feature widths across proteins: {sorted(dims)}. "
                  f"Try a fixed subset, e.g. --feature-sets pssm,hmm,dssp")
        else:
            # Per-column standardization stats from TRAIN proteins only (no leakage).
            train_stack = [evo_rows[p] for p in evo_train_pids if p in evo_rows]
            if not train_stack:
                train_stack = list(evo_rows.values())
            train_all = np.concatenate(train_stack, axis=0)            # (sum L_train, D)
            mean = np.nanmean(train_all, axis=0)
            std = np.nanstd(train_all, axis=0)
            mean = np.nan_to_num(mean, nan=0.0).astype(np.float32)
            std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)

            npz_payload: dict[str, "np.ndarray"] = {}
            for pid, arr in evo_rows.items():
                a = arr.astype(np.float32)
                a = np.where(np.isnan(a), mean[None, :], a)            # unmatched residues -> column mean
                a = (a - mean[None, :]) / std[None, :]                 # z-score
                a = np.clip(a, -5.0, 5.0).astype(np.float32)           # tame outliers
                npz_payload[pid] = a
            npz_payload["__mean__"] = mean
            npz_payload["__std__"] = std
            np.savez_compressed(out / "bench_evo.npz", **npz_payload)
            (out / "bench_evo_keys.json").write_text(json.dumps(evo_keys))
            layout = feature_set_layout(evo_widths_ref) if evo_widths_ref else "?"
            print(f"  evo features: {len(evo_rows)} proteins, dim={dims.pop()}, layout [{layout}]"
                  f"{' (' + str(len(evo_missing)) + ' missing)' if evo_missing else ''}")
            if evo_missing:
                print("    [check] no features for:", ", ".join(p for p, _ in evo_missing[:8]),
                      "..." if len(evo_missing) > 8 else "")
                print("            example reason:", evo_missing[0][1])

    ok = [r for r in report if r["status"] == "ok"]
    low = [r for r in ok if r["match_ratio"] < 0.99]
    print("\nwrote:", out, f"(protocol={args.protocol})")
    print(f"  proteins: {len(ok)} ok  |  train={len(train_ids)} val={len(val_ids)} "
          f"test60={len(test60_rows)} test315={len(test315_rows)} ubtest={len(ubtest_rows)}")
    if args.protocol == "paper":
        print(f"  bench_train.csv -> train={len(train_ids)} (Train_335), val={len(val_ids)} (Test_60), "
              f"test={len(test315_rows)} (Test_315-28, auto-evaluated at end of training)")
    else:
        print(f"  bench_train.csv -> train={len(train_ids)} val={len(val_ids)} test={len(test60_rows)} (Test_60)")
    print(f"  label alignment: {len(ok)-len(low)} exact/near-exact, {len(low)} below 0.99 "
          f"(see bench_alignment_report.csv)")
    if low:
        print("  [check] low-alignment proteins:", ", ".join(r["protein_id"] for r in low[:10]),
              "..." if len(low) > 10 else "")


if __name__ == "__main__":
    main()
