# Results — explainability

**Status.** Sections marked EXECUTED contain real numbers produced in this
session. Sections marked PENDING contain the analysis plan and the exact command
that produces them; they contain no numbers, because the model could not be run
here (no torch, no GPU, and the feature caches are not in this environment).
Do not fill them by hand — run the command.

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

## 2. Global modality reliance — PENDING

    python scripts/run_xai.py attribute --checkpoint runs/bench_evo_pp/best.pt \
      --manifest manifests/benchmark/bench_test315.csv --split test \
      --cache-dir cache/bench_evo --out-dir results/xai
    python scripts/run_xai.py analyse --out-dir results/xai

Produces `tables/global_modality_reliance.csv`: mean reliance per modality with
bootstrap CIs over 287 proteins.

## 3. Bound versus unbound — PENDING

Repeat Section 2 for `bench_btest.csv` and `bench_ubtest.csv`, then run the
paired test over the 25 matched pairs in
`results/xai/bound_unbound/matched_pairs.csv` (already generated).

The hypothesis is directional but is **not** to be assumed: the paper's existing
ablation shows the surface branch contributing four times more on unbound than
bound structures (pooled AUPRC +0.056 vs +0.013), so if attribution is faithful,
SRS should be higher on unbound. A null or reversed result is a real finding
about the attribution method and must be reported as such.

## 4. Gate versus attribution — PENDING

The prediction here is specific and falsifiable: because the gate operates on
post-fusion representations, `mean(g_k)` and `reliance_atom` should correlate
only weakly. Strong agreement would mean cross-modal fusion mixes less than the
architecture implies; weak agreement confirms that gate values must not be
reported as modality reliance. Either way the answer belongs in the paper.

## 5. Faithfulness — PENDING

Deletion curves against a random baseline, 5 repeats. The statistic is the area
between the curves; a value at or below zero would mean the attribution carries
no more information than chance, and would invalidate Sections 2–4.

## 6. Case studies — EXECUTED (selection), PENDING (explanations)

Selected by deterministic rule on per-protein AUPRC, so the choice is auditable:

| Dataset | Rule | Chain | per-protein AUPRC |
|---|---|---|---|
| Test_315-28 | best | 7BU5B | 0.986 |
| Test_315-28 | median | 6UX8A | 0.641 |
| Test_315-28 | worst | 5VXMA | 0.053 |
| UBtest_25 | best | 1shuX | 0.953 |
| UBtest_25 | median | 2v77A | 0.355 |
| UBtest_25 | worst | 1glfO | 0.036 |

Per-protein AUPRC, bootstrap over proteins: Test_315-28 **0.595 [0.565, 0.625]**
(n = 287); UBtest_25 **0.442 [0.349, 0.539]** (n = 25). The unbound interval is
three times wider, which is the 25-protein sample size showing through and is
why per-protein significance testing on that set is underpowered.

## 7. XAI sample summary — EXECUTED

| Dataset | Proteins | Residues | Interface | Interface % | Threshold |
|---|---|---|---|---|---|
| Test_315-28 | 287 | 60,376 | 8,566 | 14.19 | 0.63 |
| UBtest_25 | 25 | 5,917 | 711 | 12.02 | 0.63 |
