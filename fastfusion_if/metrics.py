from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (equal-width bins). Lower is better."""
    y_true = y_true.astype(float)
    probs = probs.astype(float)
    if y_true.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = float(y_true.size)
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        conf = float(np.mean(probs[m]))
        acc = float(np.mean(y_true[m]))
        ece += (np.sum(m) / n) * abs(acc - conf)
    return float(ece)


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = y_true.astype(int)
    probs = probs.astype(float)
    pred = (probs >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, probs))
        out["pr_auc"] = float(average_precision_score(y_true, probs))
        out["ece"] = expected_calibration_error(y_true, probs)
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
        out["ece"] = float("nan")
    return out


def best_f1_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    if len(y_true) == 0:
        return 0.5, float("nan")
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y_true.astype(int), (probs >= t).astype(int), zero_division=0) for t in thresholds]
    i = int(np.argmax(f1s))
    return float(thresholds[i]), float(f1s[i])


def per_group_binary_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    group_ids: list[str] | np.ndarray,
    threshold: float = 0.5,
) -> tuple[list[dict[str, float | str | int]], dict[str, float]]:
    """Compute metrics for each protein/chain plus robust aggregate summaries."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, gid in enumerate(group_ids):
        groups[str(gid)].append(i)

    rows: list[dict[str, float | str | int]] = []
    for gid, indices in sorted(groups.items()):
        idx = np.asarray(indices, dtype=int)
        yy = y_true[idx]
        pp = probs[idx]
        m = binary_metrics(yy, pp, threshold=threshold)
        row: dict[str, float | str | int] = {
            "group_id": gid,
            "n_residues": int(len(idx)),
            "n_positive": int(np.sum(yy)),
        }
        row.update(m)
        rows.append(row)

    summary: dict[str, float] = {}
    for metric in ["roc_auc", "pr_auc", "precision", "recall", "f1", "mcc"]:
        vals = np.asarray([float(r[metric]) for r in rows if np.isfinite(float(r[metric]))], dtype=float)
        summary[f"mean_per_group_{metric}"] = float(np.mean(vals)) if vals.size else float("nan")
        summary[f"median_per_group_{metric}"] = float(np.median(vals)) if vals.size else float("nan")
    summary["n_groups"] = float(len(rows))
    return rows, summary
