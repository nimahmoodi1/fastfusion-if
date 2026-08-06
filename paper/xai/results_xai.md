# Results — explainability

**Status.** All sections below now report measured results from the completed
analysis. Attribution was computed for all 397 chains across the four test
sets, with no failed proteins. Prediction-error and case-selection summaries
were computed from the released per-residue predictions.

---

## 1. Prediction-error structure  — EXECUTED

Computed from the released per-residue predictions (three-seed ensemble,
threshold 0.63), 66,293 residues over 312 proteins.

| Dataset | Class | Residues | Proteins | % of set | Mean prob | Median prob |
|---|---|---|---|---|---|---|
| Test_315-28 | TP | 6,016 | 277 | 9.96 | 0.884 | 0.921 |
| Test_315-28 | FP | 7,886 | 283 | 13.06 | 0.814 | 0.816 |
| Test_315-28 | FN | 2,550 | 267 | 4.22 | 0.335 | 0.347 |
| Test_315-28 | TN | 43,924 | 287 | 72.75 | 0.147 | 0.067 |
| UBtest_25 | TP | 297 | 21 | 5.02 | 0.858 | 0.882 |
| UBtest_25 | FP | 412 | 24 | 6.96 | 0.797 | 0.783 |
| UBtest_25 | FN | 414 | 25 | 7.00 | 0.224 | 0.174 |
| UBtest_25 | TN | 4,794 | 25 | 81.02 | 0.107 | 0.038 |

Two observations that the attribution analysis is designed to explain.

**False positives are confident, not borderline.** On the bound set they average
0.814 against 0.884 for true positives — a gap of 0.07 on a scale where the
decision boundary sits at 0.63. The model is not hedging on its false positives;
it is as sure of them as of its correct calls. An error mode of this shape is
not fixed by moving the threshold, and it is the single most useful thing for
attribution to characterise.

**Unbound errors are symmetric where bound errors are not.** On Test_315-28 false
positives outnumber false negatives 3.1:1. On UBtest_25 the ratio is 1.0:1
(412 vs 414), and the false negatives are confidently negative (mean 0.224). The
model does not merely become less accurate on unbound structures; the character
of its failure changes. Whether this coincides with a shift in modality reliance
is the central question of Section 3.

## 2. Global modality reliance — EXECUTED

Attribution was computed for all 397 chains: 60 from Test 60, 287 from
Test 315-28, 25 from Btest 25 and 25 from UBtest 25. Median integrated-gradients
completeness residuals were 0.0007, 0.0006, 0.0005 and 0.0006, respectively.
No protein exceeded the prespecified 0.02 residual threshold. A median residual
of 0.0006 means that the attributions account for 99.94% of the logit mass.

Mean per-residue reliance was averaged over proteins. The three modality shares
sum to one by construction.

| Test set | Atomic | Surface | Evolutionary | Proteins |
|---|---:|---:|---:|---:|
| Test 60 | 0.427 [0.418, 0.435] | 0.358 [0.345, 0.372] | 0.215 [0.208, 0.223] | 60 |
| Test 315-28 | 0.423 [0.419, 0.427] | 0.353 [0.347, 0.359] | 0.224 [0.221, 0.228] | 287 |
| Btest 25 | 0.432 [0.419, 0.444] | 0.352 [0.332, 0.372] | 0.216 [0.206, 0.227] | 25 |
| UBtest 25 | 0.432 [0.417, 0.446] | 0.356 [0.332, 0.381] | 0.212 [0.198, 0.227] | 25 |

Reliance is therefore nearly uniform across the four test sets: approximately
0.42–0.43 atomic, 0.35–0.36 surface and 0.21–0.22 evolutionary. The bootstrap
confidence intervals overlap across all sets.

## 3. Bound versus unbound — EXECUTED

The 25 proteins available in both bound and unbound conformations were compared
as matched pairs.

| Quantity | Bound | Unbound | Difference | Wilcoxon p | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| Atomic reliance | 0.4316 | 0.4319 | +0.0003 | 0.895 | 1.000 |
| Surface reliance | 0.3525 | 0.3560 | +0.0035 | 0.381 | 1.000 |
| Evolutionary reliance | 0.2160 | 0.2122 | -0.0038 | 0.339 | 1.000 |
| Gate value | 0.5155 | 0.5154 | -0.00003 | 0.711 | — |

The predicted increase in surface reliance on unbound structures was not
observed. Surface reliance changed by only +0.0035, with p = 0.381 before and
p = 1.000 after Holm correction. Atomic and evolutionary reliance were likewise
unchanged.

This null result does not contradict the ablation result. Ablation compares two
differently trained models and asks how much performance is lost when the
surface branch is absent. Attribution interrogates one trained model and asks
how much its output depends on each pathway. The model does not weight the
surface more heavily on unbound structures; instead, the ablation result
indicates that the surface information is harder for the remaining inputs to
replace there.

## 4. Gate versus attribution — EXECUTED

The learned gate was nearly constant and did not track attribution-derived
modality reliance.

| Quantity | Mean | Standard deviation | Coefficient of variation |
|---|---:|---:|---:|
| Gate value | 0.5154 | 0.0061 | 0.0117 |
| Atomic reliance from attribution | 0.4351 | 0.2098 | 0.4822 |

The gate varied approximately 41 times less than atomic attribution reliance.
Their Spearman correlation was only rho = +0.275, corresponding to about 7% of
rank variance. The gate remained near 0.515 across residue classes, test sets
and bound and unbound conformations. Its mean therefore cannot be interpreted
as a modality-reliance fraction.

Attribution nevertheless separated prediction classes on Test 315-28:

| Class | Atomic | Surface | Evolutionary | Mean absolute attribution |
|---|---:|---:|---:|---:|
| TN | 0.457 | 0.321 | 0.222 | 17.6 |
| TP | 0.372 | 0.367 | 0.261 | 35.9 |
| FP | 0.388 | 0.385 | 0.227 | 31.4 |
| FN | 0.378 | 0.412 | 0.210 | 23.7 |

Correct rejections relied more strongly on atomic geometry than predictions the
model treated as interface-like: 0.449 versus 0.376, paired over 281 proteins,
p = 9.2 x 10^-47.

Missed interface residues relied more strongly on the surface than correctly
identified interface residues: 0.399 versus 0.377, difference +0.022,
p = 0.0075 over 184 paired proteins. False-negative surface reliance exceeded
true-positive reliance in 106 of those proteins.

Explanation magnitude followed prediction confidence:

TP 35.9 > FP 31.4 > FN 23.7 > TN 17.6.

The similarity between true-positive and false-positive magnitudes agrees with
the prediction-error analysis: false positives are confident rather than
borderline.

## 5. Faithfulness — EXECUTED

Deletion curves compared residues removed in descending attribution order with
five random-order baselines.

| Set | Mean faithfulness gap | Median gap | Proteins with positive gap |
|---|---:|---:|---:|
| Test 315-28 | +0.0780 | +0.0665 | 230 / 287 |
| UBtest 25 | +0.0293 | +0.0050 | 15 / 25 |

On Test 315-28, removing the highest-attributed 5% of residues reduced AUPRC
from 0.585 to 0.510, whereas removing 5% at random reduced it only to 0.582.
At 10% removal, attribution ordering reduced AUPRC to 0.475, compared with
0.577 under random ordering. The attribution-ordered and random curves meet at
both 0% and 100% removal, as required by the deletion procedure.

The unbound result was weaker: its mean gap was +0.0293, its median was only
+0.0050, and only 15 of 25 proteins had a positive gap. The bound attributions
are therefore strongly faithful. The unbound attributions are faithful on
average, but not reliably for every individual protein, and the 25-protein
sample is too small to establish stronger per-protein conclusions.

## 6. Case studies — EXECUTED

Cases were selected deterministically by per-protein AUPRC rather than chosen
by hand.

| Dataset | Rule | Chain | Per-protein AUPRC |
|---|---|---|---:|
| Test 315-28 | Best | 7BU5B | 0.986 |
| Test 315-28 | Median | 6UX8A | 0.641 |
| Test 315-28 | Worst | 5VXMA | 0.053 |
| UBtest 25 | Best | 1shuX | 0.953 |
| UBtest 25 | Median | 2v77A | 0.355 |
| UBtest 25 | Worst | 1glfO | 0.036 |

The matched-conformation example compares bound 1R8S:E with unbound 1R8M:E.
Both structures have 34 true positives and 6 false negatives, while false
positives decrease from 21 in the bound structure to 10 in the unbound
structure. The predicted interface remains localised on the correct face
despite the conformational change.

Across the Test 315-28 performance range:

- 7BU5B, AUPRC 0.986, recovers the interface almost exactly, with 25 true
  positives and no false positives.
- 6UX8A, AUPRC 0.641, is representative of median performance: the predicted
  patch is split between the correct face and an adjacent face.
- 5VXMA, AUPRC 0.053, fails by placing a confident patch on the wrong face.

The manuscript does not make additional qualitative claims about 1shuX, 2v77A
or 1glfO beyond their deterministic selection and per-protein AUPRC values.

Per-protein AUPRC with bootstrap intervals over proteins was 0.595
[0.565, 0.625] for Test 315-28 (n = 287) and 0.442 [0.349, 0.539] for
UBtest 25 (n = 25). The wider UBtest interval reflects its smaller sample size.


## 7. XAI sample summary — EXECUTED

| Dataset | Proteins | Residues | Interface | Interface % | Threshold |
|---|---|---|---|---|---|
| Test_315-28 | 287 | 60,376 | 8,566 | 14.19 | 0.63 |
| UBtest_25 | 25 | 5,917 | 711 | 12.02 | 0.63 |
