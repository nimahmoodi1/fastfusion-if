#!/usr/bin/env python
"""Generate every publication figure for the FastFusion-IF journal article.

Curves, distributions and calibration are computed from the per-residue and
per-protein CSVs written by ``evaluate.py`` / ``evaluate_ensemble.py``, so the
figures and the results tables cannot disagree. Scalar comparisons against
published methods come from the locked tables below, each annotated with its
source.

Every figure is written as a vector PDF (for the camera-ready) and a 400-dpi PNG
(for drafts and slides). Figures whose input CSVs are absent are skipped with a
message rather than failing, so this can be run before every experiment finishes.

    python scripts/make_figures.py --eval-root eval --out-dir figures

Style follows the conventions of this literature: colour-blind-safe palette,
no chartjunk, serif labels to match a typeset manuscript.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------
# Locked reference values. Sources are given so every number in the figures is
# traceable; do not edit without checking the cited table.
# --------------------------------------------------------------------------

# GTE-PPIS, Brief. Bioinform. 26(3) bbaf290 (2025), Table 3.  (MCC, AUPRC)
BASELINES = {
    "Test_315-28": {
        "GraphPPIS": (0.335, 0.408), "RGCNPPIS": (0.352, 0.420),
        "AGAT-PPIS": (0.442, 0.525), "GTE-PPIS": (0.511, 0.598),
    },
    "Btest_25": {
        "GraphPPIS": (0.339, 0.381), "RGCNPPIS": (0.375, 0.400),
        "AGAT-PPIS": (0.440, 0.511), "GTE-PPIS": (0.471, 0.545),
    },
    "UBtest_25": {
        "GraphPPIS": (0.298, 0.330), "RGCNPPIS": (0.296, 0.354),
        "AGAT-PPIS": (0.301, 0.325), "GTE-PPIS": (0.320, 0.343),
    },
}

# GTE-PPIS Table 2 (Test_60).  ACC, Precision, Recall, F1, AUROC, MCC, AUPRC
BASELINES_TEST60 = {
    "ScanNet":    (0.681, 0.245, 0.547, 0.339, 0.684, 0.191, 0.282),
    "DELPHI":     (0.687, 0.270, 0.577, 0.368, 0.691, 0.220, 0.307),
    "HN-PPISP":   (0.738, 0.278, 0.415, 0.333, 0.656, 0.184, 0.281),
    "DeepPPISP":  (0.750, 0.297, 0.428, 0.351, 0.676, 0.207, 0.295),
    "MaSIF-site": (0.780, 0.370, 0.561, 0.446, 0.775, 0.379, 0.372),
    "EDLMPPI":    (0.781, 0.360, 0.503, 0.420, 0.748, 0.295, 0.379),
    "GraphPPIS":  (0.792, 0.389, 0.558, 0.458, 0.784, 0.343, 0.430),
    "RGCNPPIS":   (0.799, 0.404, 0.572, 0.474, 0.803, 0.439, 0.471),
    "AGAT-PPIS":  (0.856, 0.539, 0.603, 0.569, 0.867, 0.484, 0.574),
    "GTE-PPIS":   (0.861, 0.557, 0.611, 0.582, 0.873, 0.500, 0.611),
}

# Matched single-model feature ablation, all seed 42, no TTA, Test_315-28.
# Source: runs/<name>/test_metrics.json.  Single models throughout, so this is
# a like-for-like comparison -- unlike a table mixing ensembles and single runs.
FEATURE_ABLATION = [
    ("atom + surface",                  0.3828, 0.7915, 0.3170, "bench_cached_pp"),
    ("atom + profiles",                 0.5318, 0.8587, 0.4267, "bench_evo_nosurf_pp"),
    ("atom + surface + profiles",       0.5330, 0.8632, 0.4452, "bench_evo_pp"),
    ("atom + surface + profiles + ESM", 0.4751, 0.8274, 0.3934, "bench_evo_esm_pp"),
]

# ESM-2 effect versus training-set scale. Every row is a matched pair differing
# only by model.use_plm_features (verified by config diff).
ESM_SCALE = [
    ("AGAT-PPIS\nbenchmark", 287,  0.5330, 0.4751, "bench_evo_pp / bench_evo_esm_pp"),
    ("DIPS-Plus\n(subset)",  3000, 0.5051, 0.5517, "large_v2_cached_reg / large_v2_esm_reg"),
    ("DIPS-Plus\n(full)",    8422, 0.5074, 0.5712, "full_v2_cached_reg / full_v2_esm_reg"),
]

OURS = "FastFusion-IF"
C_OURS, C_BASE, C_SOTA = "#0F7B6C", "#9AA5B1", "#B45309"
C_ON, C_OFF = "#0F7B6C", "#94A3B8"


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "grid.linewidth": 0.4, "lines.linewidth": 1.4,
        "figure.dpi": 120, "savefig.bbox": "tight",
    })


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=400)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def load_eval(root: Path, sub: str):
    """Return (per_residue_df, per_protein_df, metrics_dict) or (None, None, None)."""
    d = root / sub
    if not d.is_dir():
        return None, None, None
    res = next(iter(d.glob("*per_residue_predictions.csv")), None)
    pro = next(iter(d.glob("*per_protein_metrics.csv")), None)
    met = next(iter(d.glob("*metrics.json")), None)
    return (pd.read_csv(res) if res else None,
            pd.read_csv(pro) if pro else None,
            json.load(open(met)) if met else None)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_method_comparison(ours: dict, out: Path) -> None:
    """Grouped bars: our MCC/AUPRC against published methods, per test set."""
    sets = [s for s in ("Test_315-28", "Btest_25", "UBtest_25") if s in ours]
    if not sets:
        print("  [skip] method comparison: no results supplied")
        return
    fig, axes = plt.subplots(1, len(sets), figsize=(3.4 * len(sets), 3.1))
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, sets):
        methods = list(BASELINES[s]) + [OURS]
        vals = [BASELINES[s][m] for m in BASELINES[s]] + [ours[s]]
        x = np.arange(len(methods))
        cols = [C_SOTA if m == "GTE-PPIS" else C_BASE for m in methods[:-1]] + [C_OURS]
        ax.bar(x - 0.19, [v[0] for v in vals], 0.36, color=cols, edgecolor="white", lw=0.5)
        ax.bar(x + 0.19, [v[1] for v in vals], 0.36, color=cols, edgecolor="white",
               lw=0.5, alpha=0.55, hatch="///")
        for i, v in enumerate(vals):
            ax.text(i - 0.19, v[0] + .008, f"{v[0]:.3f}", ha="center", fontsize=6.0, rotation=90)
            ax.text(i + 0.19, v[1] + .008, f"{v[1]:.3f}", ha="center", fontsize=6.0, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=32, ha="right", fontsize=7.2)
        ax.set_title(s.replace("_", "\\_") if False else s)
        ax.set_ylim(0, max(max(v) for v in vals) * 1.30)
        ax.grid(axis="y", alpha=.25)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.set_ylabel("score")
    fig.legend(handles=[
        Line2D([], [], color="#555", lw=6, label="MCC"),
        Line2D([], [], color="#555", lw=6, alpha=.55, label="AUPRC"),
        Line2D([], [], color=C_OURS, lw=6, label=OURS),
    ], loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, out, "fig_method_comparison")


def fig_test60(ours: tuple | None, out: Path) -> None:
    """Test_60: ten published methods across MCC and AUPRC, MaSIF-site highlighted."""
    names = list(BASELINES_TEST60)
    mcc = [BASELINES_TEST60[n][5] for n in names]
    prc = [BASELINES_TEST60[n][6] for n in names]
    if ours:
        names, mcc, prc = names + [OURS], mcc + [ours[0]], prc + [ours[1]]
    order = np.argsort(prc)
    names = [names[i] for i in order]
    mcc = [mcc[i] for i in order]
    prc = [prc[i] for i in order]
    cols = [C_OURS if n == OURS else "#7C3AED" if n == "MaSIF-site" else C_BASE for n in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.barh(y + .19, prc, .36, color=cols, edgecolor="white", lw=.5, label="AUPRC")
    ax.barh(y - .19, mcc, .36, color=cols, alpha=.55, hatch="///",
            edgecolor="white", lw=.5, label="MCC")
    for i, (a, b) in enumerate(zip(prc, mcc)):
        ax.text(a + .006, i + .19, f"{a:.3f}", va="center", fontsize=6.4)
        ax.text(b + .006, i - .19, f"{b:.3f}", va="center", fontsize=6.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.6)
    ax.set_xlabel("score")
    ax.set_xlim(0, max(prc) * 1.16)
    ax.set_title("Test_60 — comparison with published methods")
    ax.grid(axis="x", alpha=.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    fig.text(0.60, 0.02, "MaSIF-site (purple): mesh-based molecular-surface method",
             fontsize=6.4, style="italic", ha="center")
    fig.tight_layout()
    save(fig, out, "fig_test60_comparison")


def _curves(ax, sets, kind):
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)
    for label, df, col, ls in sets:
        y, p = df.label.astype(int).values, df.probability.values
        if kind == "pr":
            pr, rc, _ = precision_recall_curve(y, p)
            ax.plot(rc, pr, color=col, ls=ls,
                    label=f"{label} (AUPRC {average_precision_score(y, p):.3f})")
        else:
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=col, ls=ls,
                    label=f"{label} (AUROC {roc_auc_score(y, p):.3f})")
    if kind == "pr":
        base = sets[0][1].label.mean()
        ax.axhline(base, color="#999", ls=":", lw=.9, label=f"random ({base:.3f})")
        ax.set_xlabel("recall"); ax.set_ylabel("precision")
    else:
        ax.plot([0, 1], [0, 1], color="#999", ls=":", lw=.9, label="random")
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.legend(loc="best", frameon=False, fontsize=7)
    ax.grid(alpha=.25); ax.set_axisbelow(True)


def fig_curves(root: Path, out: Path) -> None:
    """PR and ROC curves, surface ON vs OFF, on both bound and unbound sets."""
    pairs = [
        ("Test_315-28", "bench_evo_ens_test315_tuned", "bench_evo_nosurf_ens_test315"),
        ("UBtest_25", "bench_evo_ens_ubtest_tuned", "bench_evo_nosurf_ens_ubtest"),
    ]
    have = []
    for name, on, off in pairs:
        r_on, _, _ = load_eval(root, on)
        r_off, _, _ = load_eval(root, off)
        if r_on is not None:
            have.append((name, r_on, r_off))
    if not have:
        print("  [skip] PR/ROC curves: no prediction CSVs found")
        return
    fig, axes = plt.subplots(2, len(have), figsize=(3.5 * len(have), 6.2), squeeze=False)
    for j, (name, on, off) in enumerate(have):
        sets = [("surface ON", on, C_ON, "-")]
        if off is not None:
            sets.append(("surface OFF", off, C_OFF, "--"))
        _curves(axes[0][j], sets, "pr")
        _curves(axes[1][j], sets, "roc")
        axes[0][j].set_title(f"{name} — precision–recall")
        axes[1][j].set_title(f"{name} — ROC")
    fig.tight_layout()
    save(fig, out, "fig_pr_roc_curves")


def fig_feature_ablation(out: Path) -> None:
    """Matched single-model feature ablation."""
    labels = [r[0] for r in FEATURE_ABLATION]
    prc = [r[1] for r in FEATURE_ABLATION]
    full = "atom + surface + profiles"
    cols = [C_OURS if l == full else C_BASE for l in labels]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    b = ax.bar(range(len(labels)), prc, .62, color=cols, edgecolor="white", lw=.6)
    for r, v in zip(b, prc):
        ax.text(r.get_x() + r.get_width() / 2, v + .008, f"{v:.3f}",
                ha="center", fontsize=7.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" + ", "\n+ ") for l in labels], fontsize=7.2)
    ax.set_ylabel("AUPRC (Test_315-28)")
    ax.set_ylim(0, max(prc) * 1.20)
    ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
    ax.set_title("Feature ablation (single model, seed 42)")
    ax.annotate("", xy=(2, prc[2] + .045), xytext=(3, prc[3] + .045),
                arrowprops=dict(arrowstyle="<-", color="#B91C1C", lw=1.1))
    ax.text(2.5, prc[2] + .062, f"ESM −{prc[2]-prc[3]:.3f}", ha="center",
            fontsize=7.2, color="#B91C1C")
    fig.tight_layout()
    save(fig, out, "fig_feature_ablation")


def fig_esm_scale(out: Path) -> None:
    """The journal's headline: the ESM-2 effect flips sign with training scale."""
    labels = [r[0] for r in ESM_SCALE]
    no_esm = [r[2] for r in ESM_SCALE]
    esm = [r[3] for r in ESM_SCALE]
    delta = [b - a for a, b in zip(no_esm, esm)]
    x = np.arange(len(labels))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    a1.bar(x - .19, no_esm, .36, color=C_OFF, edgecolor="white", lw=.5, label="without ESM-2")
    a1.bar(x + .19, esm, .36, color=C_OURS, edgecolor="white", lw=.5, label="with ESM-2")
    for i, (u, v) in enumerate(zip(no_esm, esm)):
        a1.text(i - .19, u + .006, f"{u:.3f}", ha="center", fontsize=6.6)
        a1.text(i + .19, v + .006, f"{v:.3f}", ha="center", fontsize=6.6)
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=7.2)
    a1.set_ylabel("PR-AUC"); a1.set_ylim(0, max(esm) * 1.20)
    a1.legend(frameon=False, loc="upper left")
    a1.grid(axis="y", alpha=.25); a1.set_axisbelow(True)
    a1.set_title("Matched ESM-2 ablation at three scales")

    cols = ["#B91C1C" if d < 0 else C_OURS for d in delta]
    a2.axhline(0, color="#333", lw=.8)
    a2.bar(x, delta, .52, color=cols, edgecolor="white", lw=.5)
    for i, d in enumerate(delta):
        a2.text(i, d + (.004 if d > 0 else -.010), f"{d:+.3f}",
                ha="center", fontsize=7.4, color=cols[i])
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=7.2)
    a2.set_ylabel("ΔPR-AUC from ESM-2")
    a2.set_ylim(min(delta) * 1.7, max(delta) * 1.45)
    a2.grid(axis="y", alpha=.25); a2.set_axisbelow(True)
    a2.set_title("Benefit grows with training data")
    fig.tight_layout()
    save(fig, out, "fig_esm_scale_dependence")


def fig_surface_ablation(root: Path, out: Path) -> None:
    """Pooled surface effect plus the per-protein paired difference."""
    on_b, off_b = load_eval(root, "bench_evo_ens_test315_tuned")[1], \
                  load_eval(root, "bench_evo_nosurf_ens_test315")[1]
    on_u, off_u = load_eval(root, "bench_evo_ens_ubtest_tuned")[1], \
                  load_eval(root, "bench_evo_nosurf_ens_ubtest")[1]
    if on_b is None or off_b is None:
        print("  [skip] surface ablation: per-protein CSVs not found")
        return
    from scipy.stats import wilcoxon
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2))

    groups, vals_off, vals_on = [], [], []
    for lbl, don, doff, m in [("Test_315-28\nAUPRC", on_b, off_b, "pr_auc"),
                              ("Test_315-28\nAUROC", on_b, off_b, "roc_auc"),
                              ("UBtest_25\nAUPRC", on_u, off_u, "pr_auc")]:
        if don is None or doff is None:
            continue
        groups.append(lbl)
        vals_on.append(don[m].mean())
        vals_off.append(doff[m].mean())
    x = np.arange(len(groups))
    a1.bar(x - .19, vals_off, .36, color=C_OFF, edgecolor="white", lw=.5, label="surface OFF")
    a1.bar(x + .19, vals_on, .36, color=C_ON, edgecolor="white", lw=.5, label="surface ON")
    for i, (u, v) in enumerate(zip(vals_off, vals_on)):
        a1.text(i - .19, u + .006, f"{u:.3f}", ha="center", fontsize=6.6)
        a1.text(i + .19, v + .006, f"{v:.3f}", ha="center", fontsize=6.6)
    a1.set_xticks(x); a1.set_xticklabels(groups, fontsize=7.2)
    a1.set_ylabel("mean per-protein score")
    a1.set_ylim(0, max(vals_on) * 1.22)
    a1.legend(frameon=False); a1.grid(axis="y", alpha=.25); a1.set_axisbelow(True)
    a1.set_title("Surface modality ablation")

    common = on_b.set_index("group_id").index.intersection(off_b.set_index("group_id").index)
    d = (on_b.set_index("group_id").loc[common, "roc_auc"].astype(float) -
         off_b.set_index("group_id").loc[common, "roc_auc"].astype(float)).dropna()
    stat, p = wilcoxon(d.values)
    a2.hist(d.values, bins=44, color=C_ON, alpha=.80, edgecolor="white", lw=.35)
    a2.axvline(0, color="#333", lw=.9)
    a2.axvline(d.mean(), color="#B45309", ls="--", lw=1.1,
               label=f"mean {d.mean():+.4f}")
    a2.set_xlabel("per-protein ΔAUROC  (surface ON − OFF)")
    a2.set_ylabel("proteins")
    a2.set_title(f"Paired Wilcoxon  p = {p:.3f}   ({int((d>0).sum())}/{len(d)} improved)")
    a2.legend(frameon=False); a2.grid(alpha=.25); a2.set_axisbelow(True)
    fig.tight_layout()
    save(fig, out, "fig_surface_ablation")


def fig_calibration(root: Path, out: Path) -> None:
    """Reliability diagrams with ECE, bound and unbound."""
    have = [(n, load_eval(root, s)[0]) for n, s in
            [("Test_315-28", "bench_evo_ens_test315_tuned"),
             ("UBtest_25", "bench_evo_ens_ubtest_tuned")]]
    have = [(n, d) for n, d in have if d is not None]
    if not have:
        print("  [skip] calibration: no prediction CSVs")
        return
    fig, axes = plt.subplots(1, len(have), figsize=(3.4 * len(have), 3.2), squeeze=False)
    for ax, (name, df) in zip(axes[0], have):
        y, p = df.label.astype(int).values, df.probability.values
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
        conf, acc, w = [], [], []
        for b in range(10):
            m = idx == b
            if m.sum() == 0:
                continue
            conf.append(p[m].mean()); acc.append(y[m].mean()); w.append(m.sum())
        w = np.array(w, float)
        ece = float(np.sum(w / w.sum() * np.abs(np.array(acc) - np.array(conf))))
        ax.plot([0, 1], [0, 1], ls=":", color="#999", lw=.9, label="perfect")
        ax.plot(conf, acc, "o-", color=C_OURS, ms=4.2, label="model")
        ax.bar(conf, w / w.sum(), width=.075, alpha=.22, color="#64748B",
               label="fraction of residues")
        ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
        ax.set_title(f"{name}   ECE = {ece:.3f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(frameon=False, loc="upper left", fontsize=7)
        ax.grid(alpha=.25); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, out, "fig_calibration")


def fig_per_protein(root: Path, out: Path) -> None:
    """Distribution of per-protein performance -- variability the pooled metric hides."""
    sets = [("Test_315-28", "bench_evo_ens_test315_tuned"),
            ("Btest_25", "bench_evo_ens_btest_tuned"),
            ("UBtest_25", "bench_evo_ens_ubtest_tuned"),
            ("Test_60", "bench_evo_ens_test60_tuned")]
    data, names = [], []
    for n, s in sets:
        _, pp, _ = load_eval(root, s)
        if pp is not None and "pr_auc" in pp:
            data.append(pp.pr_auc.dropna().values); names.append(f"{n}\n(n={len(pp)})")
    if not data:
        print("  [skip] per-protein distributions: no per-protein CSVs")
        return
    fig, ax = plt.subplots(figsize=(1.5 * len(data) + 2.2, 3.3))
    parts = ax.violinplot(data, showextrema=False, widths=.78)
    for b in parts["bodies"]:
        b.set_facecolor(C_OURS); b.set_alpha(.30); b.set_edgecolor(C_OURS)
    bp = ax.boxplot(data, widths=.16, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", lw=1.3))
    for b in bp["boxes"]:
        b.set_facecolor(C_OURS); b.set_alpha(.95); b.set_edgecolor("none")
    for i, v in enumerate(data, start=1):
        ax.text(i, np.median(v) + .035, f"{np.median(v):.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names, fontsize=7.4)
    ax.set_ylabel("per-protein AUPRC")
    ax.set_title("Per-protein performance distribution")
    ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, out, "fig_per_protein_distribution")


def fig_confusion(root: Path, out: Path, thr: float) -> None:
    """Confusion matrices at the operating threshold."""
    have = [(n, load_eval(root, s)[0]) for n, s in
            [("Test_315-28", "bench_evo_ens_test315_tuned"),
             ("UBtest_25", "bench_evo_ens_ubtest_tuned")]]
    have = [(n, d) for n, d in have if d is not None]
    if not have:
        print("  [skip] confusion matrices: no prediction CSVs")
        return
    fig, axes = plt.subplots(1, len(have), figsize=(3.1 * len(have), 3.0), squeeze=False)
    for ax, (name, df) in zip(axes[0], have):
        y = df.label.astype(int).values
        pred = (df.probability.values >= thr).astype(int)
        cm = np.array([[int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())],
                       [int(((y == 1) & (pred == 0)).sum()), int(((y == 1) & (pred == 1)).sum())]])
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{norm[i, j]:.1%}", ha="center", va="center",
                        fontsize=8, color="white" if norm[i, j] > .55 else "#111")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["non-interface", "interface"], fontsize=7.4)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["non-interface", "interface"],
                                                  fontsize=7.4, rotation=90, va="center")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{name}  (threshold {thr:.2f})")
    fig.tight_layout()
    save(fig, out, "fig_confusion_matrices")


def fig_threshold(root: Path, out: Path, thr: float) -> None:
    """F1 and MCC against decision threshold -- shows the operating point is not cherry-picked."""
    have = [(n, load_eval(root, s)[0]) for n, s in
            [("Test_315-28", "bench_evo_ens_test315_tuned"),
             ("UBtest_25", "bench_evo_ens_ubtest_tuned")]]
    have = [(n, d) for n, d in have if d is not None]
    if not have:
        print("  [skip] threshold sweep: no prediction CSVs")
        return
    from sklearn.metrics import f1_score, matthews_corrcoef
    fig, axes = plt.subplots(1, len(have), figsize=(3.4 * len(have), 3.0), squeeze=False)
    grid = np.arange(.05, .96, .01)
    for ax, (name, df) in zip(axes[0], have):
        y, p = df.label.astype(int).values, df.probability.values
        f1 = [f1_score(y, p >= t, zero_division=0) for t in grid]
        mcc = [matthews_corrcoef(y, (p >= t).astype(int)) for t in grid]
        ax.plot(grid, f1, color=C_OURS, label="F1")
        ax.plot(grid, mcc, color=C_SOTA, ls="--", label="MCC")
        ax.axvline(thr, color="#333", ls=":", lw=1.0)
        ax.text(thr + .012, .04, f"operating point {thr:.2f}", fontsize=6.6, rotation=90)
        ax.set_xlabel("decision threshold"); ax.set_ylabel("score")
        ax.set_title(name); ax.legend(frameon=False); ax.grid(alpha=.25); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, out, "fig_threshold_sensitivity")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", default="eval")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--threshold", type=float, default=0.63)
    ap.add_argument("--ours-test315", nargs=2, type=float, default=[0.456, 0.544],
                    metavar=("MCC", "AUPRC"))
    ap.add_argument("--ours-ubtest", nargs=2, type=float, default=[0.339, 0.404],
                    metavar=("MCC", "AUPRC"))
    ap.add_argument("--ours-btest", nargs=2, type=float, default=None,
                    metavar=("MCC", "AUPRC"), help="fill in after the Btest_25 run")
    ap.add_argument("--ours-test60", nargs=2, type=float, default=None,
                    metavar=("MCC", "AUPRC"), help="fill in after the Test_60 run")
    args = ap.parse_args()

    style()
    root, out = Path(args.eval_root), Path(args.out_dir)

    ours = {"Test_315-28": tuple(args.ours_test315), "UBtest_25": tuple(args.ours_ubtest)}
    if args.ours_btest:
        ours["Btest_25"] = tuple(args.ours_btest)

    print(f"reading {root}/  ->  writing {out}/\n")
    fig_method_comparison(ours, out)
    fig_test60(tuple(args.ours_test60) if args.ours_test60 else None, out)
    fig_curves(root, out)
    fig_feature_ablation(out)
    fig_esm_scale(out)
    fig_surface_ablation(root, out)
    fig_calibration(root, out)
    fig_per_protein(root, out)
    fig_confusion(root, out, args.threshold)
    fig_threshold(root, out, args.threshold)
    print(f"\ndone -> {out}/")
    print("3D structure renders come from scripts/render_predictions.py.")


if __name__ == "__main__":
    main()
