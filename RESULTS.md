# FastFusion-IF — Final Results (paper-ready)

**Model.** FastFusion-IF (~2.86M params): atom EGNN + mesh-free solvent-excluded
**surface point-cloud encoder** + cross-modal fusion + attention residue pooling +
residue-context graph transformer + ESM-2 injection + decoder. Residue-level PPI
interface prediction from single-chain, partner-blind input.

**Headlines.**
1. **AGAT-PPIS benchmark, Test_315-28:** AUPRC **0.544**, MCC **0.456** — beats GraphPPIS, RGCNPPIS, and AGAT-PPIS; below only GTE-PPIS.
2. **AGAT-PPIS benchmark, UBtest_25 (unbound):** AUPRC **0.404** — **best of all methods, including SOTA GTE-PPIS** (0.343).
3. **DIPS-Plus at scale** (33,690 training proteins, leakage-safe splits): PR-AUC **0.571** with ESM, clean **+0.064** ESM ablation.

The FastFusion-IF benchmark model = atom + surface + evolutionary profiles (PSSM/HMM/DSSP/resAF,
the same 61-dim feature set AGAT-PPIS/GTE-PPIS use); ESM is excluded on the benchmark because it
overfits at 335 training proteins. Reported numbers are a **3-seed ensemble with TTA×8**, threshold
tuned on Test_60 (validation). Single-model AUPRC on Test_315-28 averages 0.534 — still above AGAT-PPIS.

---

## 1. Head-to-head — Test_315-28 (287 proteins, held-out test)

Published values from GTE-PPIS (Briefings in Bioinformatics 2025, bbaf290), Table 3.
That table reports MCC and AUPRC (the standard threshold-robust + imbalance-robust metrics).

| Method | MCC | AUPRC |
|---|---|---|
| GraphPPIS | 0.335 | 0.408 |
| RGCNPPIS | 0.352 | 0.420 |
| AGAT-PPIS | 0.442 | 0.525 |
| **FastFusion-IF (ours)** | **0.456** | **0.544** |
| GTE-PPIS (SOTA) | 0.511 | 0.598 |

FastFusion-IF full metrics on Test_315-28 (3-seed ensemble, TTA×8, val-tuned threshold 0.63):
AUPRC 0.544, AUROC 0.869, F1 0.536, MCC 0.456, Precision 0.433, Recall 0.702.

## 2. Head-to-head — UBtest_25 (25 unbound proteins, held-out test)

Published values from GTE-PPIS (bbaf290), Table 3.

| Method | MCC | AUPRC |
|---|---|---|
| GraphPPIS | 0.298 | 0.330 |
| RGCNPPIS | 0.296 | 0.354 |
| AGAT-PPIS | 0.301 | 0.325 |
| GTE-PPIS | 0.320 | 0.343 |
| **FastFusion-IF (ours)** | **0.339** | **0.404** |

**FastFusion-IF has the best AUPRC of all methods on the unbound set** (+0.050 over the previous
best, RGCNPPIS 0.354; +0.061 over GTE-PPIS). FastFusion-IF full metrics on UBtest (3-seed ensemble,
TTA×8): AUPRC 0.404, AUROC 0.798, F1 0.418, MCC 0.339, Precision 0.419, Recall 0.418 (val-tuned threshold 0.63).

> Note for the manuscript: numbers for the four published methods are as compiled in GTE-PPIS Table 3
> under one evaluation protocol. For the final paper, cross-check each against its original paper
> (GraphPPIS, RGCNPPIS, AGAT-PPIS) and cite accordingly. Disclose the 3-seed ensemble + TTA×8.


## 2b. FastFusion-IF full metrics (both held-out test sets)

3-seed ensemble, TTA×8, threshold tuned on Test_60 (= 0.63).

| Metric | Test_315-28 | UBtest_25 |
|---|---|---|
| Accuracy | 0.827 | 0.860 |
| Precision | 0.433 | 0.419 |
| Recall (sensitivity) | 0.702 | 0.418 |
| Specificity | 0.848 | 0.921 |
| F1 | 0.536 | 0.418 |
| AUROC | 0.869 | 0.798 |
| MCC | 0.456 | 0.339 |
| AUPRC | 0.544 | 0.404 |
| ECE (calibration) | 0.174 | 0.098 |

Confusion matrix at threshold 0.63 — Test_315-28: TP 6016, FP 7886, TN 43924, FN 2550; UBtest_25: TP 297, FP 412, TN 4794, FN 414.
Accuracy is exact (computed from the per-residue predictions). The confusion-matrix totals match the benchmark dataset
statistics exactly (Test_315-28: 60 376 residues, 8566 positive; UBtest_25: 5917 residues, 711 positive), confirming
evaluation on identical data to the published methods. Loss is not reported — not a held-out comparison metric and
unreported by any baseline; ECE covers calibration.

---

## 3. Feature ablation — Test_315-28 (global AUPRC, 3-seed ensembles)

| Model | AUPRC | ROC-AUC | MCC† | note |
|---|---|---|---|---|
| atom geometry + surface (baseline) | 0.395 | 0.801 | 0.324 | geometry only |
| atom + surface + ESM (no evo) | 0.380 | — | — | ESM alone hurts at 335 proteins |
| atom + evo (no surface) | 0.531 | 0.858 | 0.419 | evo adds **+0.136** AUPRC |
| **atom + surface + evo (FastFusion-IF, full)** | **0.544** | **0.869** | **0.456** | surface adds **+0.013** AUPRC |
| atom + surface + evo + ESM | 0.475 | 0.827 | 0.393 | ESM overfits, **−0.069** |

Per-seed AUPRC: full = 0.533 / 0.530 / 0.539; no-surface = 0.532 / 0.522 / 0.523.
†MCC at val-tuned threshold (full 0.63, no-surface 0.57); others at their own tuned threshold.

**Scale-dependent ESM finding.** ESM helps at 33,690 proteins (+0.064, DIPS-Plus) but hurts at 335
(−0.069 added to evo here; train loss → 0.37 while val loss → 2.74). Compact 61-dim evolutionary
profiles generalize in both regimes; the 1280-dim PLM embedding overfits the small benchmark.

---

## 4. Surface modality ablation (surface ON vs OFF, evo fixed, ESM off)

| Test set | metric | surface-OFF | surface-ON | Δ |
|---|---|---|---|---|
| Test_315-28 (bound) | ensemble AUPRC | 0.531 | 0.544 | +0.013 |
| Test_315-28 (bound) | ensemble ROC | 0.858 | 0.869 | +0.010 |
| UBtest_25 (unbound) | ensemble AUPRC | 0.348 | 0.404 | **+0.056** |

**Per-protein significance (paired Wilcoxon, two-sided):**

| Test | Δ | proteins improved | p | significant |
|---|---|---|---|---|
| Bound, per-protein ROC | +0.0051 | 169/287 | **0.017** | yes |
| Bound, per-protein AUPRC | +0.0084 | 151/287 | 0.251 | no |
| Unbound, per-protein AUPRC | +0.0224 | 14/25 | 0.411 | no (n=25 underpowered) |

**How to state it.** The surface significantly improves per-residue **ranking** on bound complexes
(per-protein ROC-AUC, p = 0.017); the per-protein AUPRC gain is positive but not individually
significant (significant only pooled, +0.013); the **unbound** effect is the largest (+0.056 pooled)
and drives the best-in-class UBtest result. Do not claim a significant AUPRC gain from the surface —
claim the ROC result and the unbound performance. Mechanism: PSSM/HMM/DSSP are conformation-invariant,
so the surface supplies conformation-aware signal exactly where sequence priors fail (unbound).

---

## 5. DIPS-Plus at scale (internal ablation; leakage-safe MMseqs2 30%/80% splits)

33,690 training proteins, single-chain partner-blind input, partner-specific labels. Test metrics.

| Model | PR-AUC | ROC-AUC | MCC |
|---|---|---|---|
| Full baseline (matched reg) | 0.507 | — | — |
| **Full + ESM-2** | **0.571** | **0.882** | **0.496** |

ESM contribution at scale: **+0.064 PR-AUC**. This is an internal ablation (no external baseline on the
identical leakage-safe split, which is itself part of the contribution — a rigorous large-scale eval).
Reg recipe: plm_dropout 0.35, weight_decay 0.03, lr 1.5e-4, dropout 0.15, 30 epochs.

---

## 6. Reproducibility notes

- Benchmark: train = Train_335, val = Test_60, test = Test_315-28 (287); UBtest_25 (unbound, 25). Same splits as AGAT-PPIS / GTE-PPIS.
- Features: PSSM 20 + HMM 20 + DSSP 14 + resAF 7 = 61-dim evolutionary (identical to AGAT-PPIS) + mesh-free surface point cloud + atom EGNN.
- Cache `cache/bench_evo`: geometry + ESM-2 650M (1280-d) + 61-dim evolutionary, uniform width.
- Training: 40 epochs, config `bench_evo_reg.json` (surface-off = `bench_evo_nosurf_reg.json`, identical except `use_surface=False`).
- Ensembles: 3 seeds (0/1/2), per-residue probability averaging, TTA×8, threshold tuned on Test_60.
- Exact accuracy: derive from the per-residue prediction CSVs if a reviewer requests ACC (script available).
