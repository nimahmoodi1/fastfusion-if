"""Aggregation, statistics and case selection for the XAI results.

Everything here is pure numpy/pandas so it runs without torch, which means the
statistical layer can be tested and exercised independently of the GPU stage.

Statistical unit
----------------
Residues within a protein are not independent: they share a structure, a
profile, and a fold. Protein-level statistics are therefore the default
throughout. Residue-level numbers are reported only as distributions, never as
the basis of a significance test, and every table carries both the protein and
the residue count so a reader can see which unit a number rests on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-8


# ------------------------------------------------------------------ reliance
def modality_reliance(df: pd.DataFrame, cols: dict[str, str] | None = None) -> pd.DataFrame:
    """Normalised reliance per residue from signed attributions.

    Signed attributions cannot be ratioed directly: a modality that pushes the
    logit *down* is being used, and a naive ratio of signed values can exceed 1
    or go negative and stops being a share. Magnitudes are used for the share,
    and the sign is preserved separately so that "used to suppress" is
    distinguishable from "used to support".

        share_m = |A_m| / (sum_m' |A_m'| + eps)

    Adds ``reliance_<m>`` for each modality, ``srs`` (the surface share, i.e. the
    Surface Reliance Score), and ``attr_total_abs``.
    """
    cols = cols or {
        "atom": "attr_atom",
        "surface": "attr_surface",
        "evolutionary": "attr_evolutionary",
        "plm": "attr_plm",
    }
    present = {m: c for m, c in cols.items() if c in df.columns}
    if not present:
        raise ValueError("no attribution columns found in dataframe")

    mags = {m: df[c].abs().to_numpy(dtype=float) for m, c in present.items()}
    total = np.sum(list(mags.values()), axis=0) + EPS

    out = df.copy()
    for m, v in mags.items():
        out[f"reliance_{m}"] = v / total
        out[f"sign_{m}"] = np.sign(df[present[m]].to_numpy(dtype=float))
    out["attr_total_abs"] = total - EPS
    out["srs"] = out["reliance_surface"] if "reliance_surface" in out else np.nan
    return out


def confusion_class(labels: np.ndarray, probs: np.ndarray, threshold: float) -> np.ndarray:
    """TP / FP / FN / TN per residue."""
    y = np.asarray(labels).astype(int)
    p = (np.asarray(probs, dtype=float) >= threshold).astype(int)
    out = np.full(len(y), "TN", dtype=object)
    out[(y == 1) & (p == 1)] = "TP"
    out[(y == 0) & (p == 1)] = "FP"
    out[(y == 1) & (p == 0)] = "FN"
    return out


# ---------------------------------------------------------------- aggregation
def aggregate_by_protein(df: pd.DataFrame, group_col: str = "group_id") -> pd.DataFrame:
    """Protein-level means of every reliance/attribution/gate column."""
    value_cols = [
        c
        for c in df.columns
        if c.startswith(("reliance_", "attr_", "gate", "srs", "prob", "logit"))
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    g = df.groupby(group_col)
    out = g[value_cols].mean()
    out["n_residues"] = g.size()
    if "label" in df.columns:
        out["n_interface"] = g["label"].sum()
        out["interface_fraction"] = out["n_interface"] / out["n_residues"]
    return out.reset_index()


# ---------------------------------------------------------------- statistics
@dataclass
class BootstrapCI:
    mean: float
    lo: float
    hi: float
    n: int

    def as_tuple(self) -> tuple[float, float, float]:
        return self.mean, self.lo, self.hi

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.lo:.4f}, {self.hi:.4f}] (n={self.n})"


def bootstrap_ci(
    x: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> BootstrapCI:
    """Percentile bootstrap CI over proteins."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return BootstrapCI(np.nan, np.nan, np.nan, 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return BootstrapCI(
        float(x.mean()),
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
        len(x),
    )


def paired_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Two-sided paired Wilcoxon plus a paired effect size.

    Wilcoxon rather than a t-test because per-protein attribution shares are
    bounded in [0, 1] and skewed. The effect size is the matched-pairs rank
    biserial correlation, which is the effect size that belongs with Wilcoxon.
    """
    from scipy.stats import wilcoxon

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    d = a - b
    nz = d[d != 0]
    if len(nz) < 3:
        return {"n": len(a), "mean_diff": float(d.mean()) if len(d) else np.nan,
                "statistic": np.nan, "p": np.nan, "effect_size": np.nan}
    stat, p = wilcoxon(a, b, alternative="two-sided")
    n = len(nz)
    total = n * (n + 1) / 2
    rbc = 1.0 - 2.0 * stat / total  # matched-pairs rank biserial correlation
    return {
        "n": int(len(a)),
        "mean_diff": float(d.mean()),
        "median_diff": float(np.median(d)),
        "statistic": float(stat),
        "p": float(p),
        "effect_size": float(rbc),
    }


def holm_bonferroni(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values.

    Holm rather than Bonferroni because it is uniformly more powerful at the
    same family-wise error rate, and the comparisons here (one per modality)
    are few and correlated.
    """
    items = [(k, v) for k, v in pvals.items() if v is not None and not np.isnan(v)]
    if not items:
        return {k: np.nan for k in pvals}
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)  # enforce monotonicity
        out[k] = running
    for k in pvals:
        out.setdefault(k, np.nan)
    return out


# -------------------------------------------------------------- case studies
def select_cases(
    per_protein: pd.DataFrame,
    per_residue: pd.DataFrame,
    metric: str = "pr_auc",
    group_col: str = "group_id",
) -> pd.DataFrame:
    """Pick representative cases by rule, so the choice is not cherry-picked.

    Every selection is a deterministic function of the data: extremes and the
    median of a named metric, and the residues with the most extreme reliance or
    the largest attribution-versus-gate disagreement. The rule used is recorded
    in the ``rule`` column so a reader can verify the choice.
    """
    rows = []
    d = per_protein.dropna(subset=[metric]).sort_values(metric)
    if len(d):
        for rule, r in [
            ("worst_protein", d.iloc[0]),
            ("median_protein", d.iloc[len(d) // 2]),
            ("best_protein", d.iloc[-1]),
        ]:
            rows.append({"rule": rule, group_col: r[group_col], metric: r[metric]})

    if "srs" in per_protein.columns:
        s = per_protein.dropna(subset=["srs"]).sort_values("srs")
        if len(s):
            rows.append({"rule": "most_surface_reliant", group_col: s.iloc[-1][group_col],
                         "srs": s.iloc[-1]["srs"]})
            rows.append({"rule": "least_surface_reliant", group_col: s.iloc[0][group_col],
                         "srs": s.iloc[0]["srs"]})

    if {"confusion", "attr_total_abs"} <= set(per_residue.columns):
        for cls in ("FP", "FN"):
            sub = per_residue[per_residue["confusion"] == cls]
            if len(sub):
                r = sub.loc[sub["attr_total_abs"].idxmax()]
                rows.append({"rule": f"most_confident_{cls}", group_col: r[group_col],
                             "res_seq": r.get("res_seq"), "prob": r.get("prob")})

    if {"gate_mean", "reliance_atom"} <= set(per_residue.columns):
        dis = (per_residue["gate_mean"] - per_residue["reliance_atom"]).abs()
        r = per_residue.loc[dis.idxmax()]
        rows.append({"rule": "max_gate_attribution_disagreement", group_col: r[group_col],
                     "res_seq": r.get("res_seq"), "gate_mean": r.get("gate_mean"),
                     "reliance_atom": r.get("reliance_atom")})
        r2 = per_residue.loc[dis.idxmin()]
        rows.append({"rule": "min_gate_attribution_disagreement", group_col: r2[group_col],
                     "res_seq": r2.get("res_seq"), "gate_mean": r2.get("gate_mean"),
                     "reliance_atom": r2.get("reliance_atom")})

    return pd.DataFrame(rows)


def error_analysis(per_residue: pd.DataFrame) -> pd.DataFrame:
    """Compare reliance and explanation magnitude across TP/TN/FP/FN."""
    if "confusion" not in per_residue.columns:
        raise ValueError("per_residue needs a 'confusion' column; call confusion_class first")
    cols = [c for c in per_residue.columns
            if c.startswith(("reliance_", "attr_", "gate", "srs", "prob"))
            and pd.api.types.is_numeric_dtype(per_residue[c])]
    g = per_residue.groupby("confusion")
    out = g[cols].agg(["mean", "median", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_residues"] = g.size()
    if "group_id" in per_residue.columns:
        out["n_proteins"] = g["group_id"].nunique()
    return out.reset_index()
