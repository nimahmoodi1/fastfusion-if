#!/usr/bin/env python
"""Compare an external predictor (PeSTo, ScanNet, ...) against FastFusion-IF on the
SAME residues, SAME labels, SAME metric definitions — the only reviewer-proof way to
put our DIPS-Plus numbers next to another method's.

Inputs
------
--ours   : the per-residue CSV written by scripts/evaluate.py
           (columns: group_id, source_path, chain_id, res_chain, res_seq,
            insertion, res_name, label, probability)
--external : a CSV of the other method's per-residue interface scores with columns
           group_id, res_chain, res_seq, insertion, score
           (use scripts/adapt_*_scores.py or your own adapter to produce it)

The two are inner-joined on (group_id, res_chain, res_seq, insertion). Metrics for
BOTH methods are then computed on exactly that intersection using the project's own
evaluate_predictions(), so the comparison is apples-to-apples. Coverage (fraction of
our residues that the external method also scored) is reported — a fair comparison
needs coverage near 1.0.

Usage
-----
python scripts/compare_external.py \
    --ours eval/full_v2_esm_reg/test_per_residue_predictions.csv \
    --external eval/pesto/pesto_scores.csv \
    --name PeSTo --out-dir eval/compare_pesto
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from fastfusion_if.evaluation import evaluate_predictions
from fastfusion_if.metrics import best_f1_threshold
from fastfusion_if.utils import ensure_dir


def _key(row: dict) -> tuple:
    # Normalise the join key so "A"/"a", "12"/12, ""/"?" line up across files.
    return (
        str(row["group_id"]).strip(),
        str(row["res_chain"]).strip(),
        str(int(float(row["res_seq"]))).strip(),
        str(row.get("insertion", "") or "").strip(),
    )


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _find_col(fieldnames, *candidates):
    low = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="per-residue CSV from scripts/evaluate.py")
    ap.add_argument("--external", required=True, help="external per-residue scores CSV")
    ap.add_argument("--name", default="external", help="display name of the external method")
    ap.add_argument("--out-dir", default="eval/compare_external")
    args = ap.parse_args()

    ours = _read_csv(args.ours)
    ext = _read_csv(args.external)
    if not ours or not ext:
        raise SystemExit("empty input CSV(s)")

    score_col = _find_col(ext[0].keys(), "score", "probability", "prob", "pred", "bfactor", "b_factor")
    if score_col is None:
        raise SystemExit(f"external CSV needs a score column; got {list(ext[0].keys())}")

    ext_by_key: dict[tuple, float] = {}
    for r in ext:
        try:
            ext_by_key[_key(r)] = float(r[score_col])
        except (KeyError, ValueError):
            continue

    y, p_ext, p_ours, groups = [], [], [], []
    matched = 0
    for r in ours:
        k = _key(r)
        if k in ext_by_key:
            matched += 1
            y.append(int(float(r["label"])))
            p_ours.append(float(r["probability"]))
            p_ext.append(ext_by_key[k])
            groups.append(str(r["group_id"]))
    coverage = matched / max(1, len(ours))
    if matched == 0:
        raise SystemExit("no residues matched between the two files — check the join keys (group_id / chain / res_seq / insertion formatting)")

    y = np.asarray(y); p_ext = np.asarray(p_ext); p_ours = np.asarray(p_ours)

    def _eval(scores):
        thr = best_f1_threshold(y, scores)[0]
        ev = evaluate_predictions(y, scores, groups, threshold=thr)
        return thr, ev["global"], ev["per_group_summary"]

    thr_o, g_o, pg_o = _eval(p_ours)
    thr_e, g_e, pg_e = _eval(p_ext)

    report = {
        "coverage": coverage,
        "residues_compared": int(matched),
        "residues_ours_total": len(ours),
        "proteins_compared": len(set(groups)),
        "FastFusion_IF": {"threshold": thr_o, "global": g_o, "per_group_summary": pg_o},
        args.name: {"threshold": thr_e, "global": g_e, "per_group_summary": pg_e},
    }
    out = ensure_dir(args.out_dir)
    (out / f"compare_{args.name}.json").write_text(json.dumps(report, indent=2))

    def _row(tag, g, pg):
        return (f"{tag:<16} PR-AUC={g['pr_auc']:.4f}  ROC-AUC={g['roc_auc']:.4f}  "
                f"F1={g['f1']:.4f}  MCC={g['mcc']:.4f}  "
                f"medPP-PRAUC={pg['median_per_group_pr_auc']:.4f}")
    print(f"\nSame-split comparison on {matched} residues / {len(set(groups))} proteins "
          f"(coverage {coverage:.1%} of our test residues)\n")
    print(_row("FastFusion-IF", g_o, pg_o))
    print(_row(args.name, g_e, pg_e))
    if coverage < 0.9:
        print(f"\n[warning] coverage is {coverage:.1%}; below ~90% the comparison may be biased "
              "(the external method skipped residues). Investigate the unmatched keys before reporting.")


if __name__ == "__main__":
    main()
