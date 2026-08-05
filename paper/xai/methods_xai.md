# Methods — explainability

*Draft for the journal article. Numbers marked `[PENDING]` require the GPU run;
every other statement is verified against the released source code.*

## Choice of method

FastFusion-IF builds its residue representation as a sum of terms injected at a
single point in a single space:

    z_k = g_k ⊙ a_k + (1 − g_k) ⊙ s_k
    r_k = z_k + λ · φ_r(f_k)                    (λ = 0.5)
    p_k = σ(head(residue_context(r_k)))

This structure determines which explanation methods are appropriate, and rules
out several that are common in the literature.

**Attribution is computed at the injection points, not at the raw inputs.**
Integrated Gradients requires a baseline that means "absent". For a
61-dimensional evolutionary profile the zero vector is a defensible baseline;
for atomic coordinates and a surface point cloud it is not — zeroing coordinates
does not produce a protein with no geometry, it produces a degenerate one at the
origin. Because the geometric, evolutionary and language-model terms are
additive at a common point with a common dimensionality, gradients taken there
are on one scale and can be compared and ratioed. This is the condition a
modality-reliance score requires, and it is why the analysis is performed in the
latent space rather than the input space.

**The target is the logit, never the probability.** A residue predicted at 0.98
has a sigmoid gradient near zero, so probability-space attribution would report
that the model's most confident predictions are caused by nothing.

**The gate is measured but not trusted.** The learned gate `g_k ∈ R^96` is the
most immediately interpretable object in the network: a per-channel convex
combination weight between the atom-pooled and surface-pooled representations.
It is tempting to read `mean(g_k)` as "fraction of reliance on atoms". That
reading is wrong, and the reason is structural. `a_k` and `s_k` are pooled from
the *outputs* of the bidirectional cross-modal fusion module, so surface
information has already entered `a_k` and atomic information has already entered
`s_k`. The gate mixes two entangled representations; it reports which pathway is
weighted, not where the information came from. We therefore report the gate,
report attribution, report intervention, and report their agreement, rather than
treating any one of them as the explanation.

**Methods not used, and why.** SHAP over input features is not applicable: the
inputs are a variable-size atom graph and point cloud, so there is no fixed
feature vector to form coalitions over, and the KernelSHAP approximation would
require a reference distribution that does not exist for protein geometry.
GNNExplainer optimises an edge mask over a single graph, which would explain the
residue-context transformer while leaving the atomic and surface branches — the
components the paper is about — untouched. LIME requires local perturbations
that stay on the data manifold; perturbing atomic coordinates does not.

## Attribution

For each term `t ∈ {z_geom, evo, plm}`, Layer Integrated Gradients with a zero
baseline and 64 midpoint Riemann steps:

    A_t,k = (t_k − 0) · ∫₀¹ ∂ logit_k / ∂ t_k |_(α·t) dα

Only the tail of the network (residue-context encoder and head) is replayed
across the α path, so the cost is `O(n_steps × tail)` rather than
`O(n_steps × full forward)`.

Completeness (`Σ_t A_t,k ≈ logit_k − logit_k(0)`) is recorded per residue as
`ig_convergence_delta` and reported as a fraction of the logit range.

The geometric term splits **exactly** by the gate, with no further
approximation, because `z_geom` is linear in `a` and `s` given `g`:

    A_atom,k = Σ_c ∇_c · g_kc · a_kc
    A_surf,k = Σ_c ∇_c · (1 − g_kc) · s_kc
    A_atom,k + A_surf,k = A_geom,k

This identity is asserted in the test suite rather than assumed.

## Modality reliance

Signed attributions cannot be ratioed directly: a modality that pushes the logit
*down* is being used, and a signed ratio can exceed 1 or go negative and stops
being a share. Reliance uses magnitudes, with the sign retained separately:

    reliance_m,k = |A_m,k| / (Σ_m' |A_m',k| + ε)

The Surface Reliance Score is the surface share, `SRS_k = reliance_surface,k`.
The shares sum to 1 by construction and lie in [0, 1].

## Interventions

Attribution says which term the gradient flows through; intervention says what
happens when a term is removed. Four interventions are run: surface zeroing
(`use_surface = False`, which the architecture already supports), evolutionary
zeroing, evolutionary shuffling across residues (preserving the marginal
distribution, so that "this feature matters" is separable from "this feature's
*value* matters"), and removal of the residue-context encoder.

Surface zeroing on a surface-trained checkpoint is an out-of-distribution
intervention: the gate receives `[a; 0; |a|; 0]`, an input it never saw in
training. It measures the effect of removing the surface at test time, which is
not the same as the surface's contribution. The separately trained surface-off
checkpoints (`runs/bench_evo_nosurf_pp*`) give the honest estimate, because the
rest of the model was free to compensate. Both are reported; where they
disagree, the retrained comparison is the one to believe.

## Faithfulness

Deletion curves: the highest-attributed residues have their evolutionary
features progressively zeroed, and AUPRC is tracked. A single curve is not
evidence — some proteins degrade quickly under any perturbation — so each is run
against a random-order baseline (5 repeats) and the **area between the curves**
is the faithfulness statistic. Positive means the attribution identifies
residues that matter more than chance.

## Statistics

Residues within a protein share a structure, a profile and a fold, and are not
independent. Proteins are therefore the statistical unit throughout. Confidence
intervals are percentile bootstrap over proteins (10,000 resamples). Bound
versus unbound uses the 25 matched pairs from the official
`bound_unbound_mapping31-6.txt`, tested with a two-sided paired Wilcoxon
signed-rank test and reported with the matched-pairs rank biserial correlation
as effect size. Multiple comparisons across modalities are corrected with
Holm–Bonferroni. Every table reports both the protein and residue count.

## Reproducibility

Seeds are fixed and `torch.use_deterministic_algorithms` is enabled. The
architecture is rebuilt from the checkpoint's own stored config rather than from
a config file on disk, so an XAI run cannot analyse a differently-shaped model
than the one that produced the reported metrics. The full configuration is
copied into `results/xai/report/run_metadata.json`. Instrumentation is by
forward hook only; `models/model.py` is unmodified and predictions with XAI
disabled are bit-identical to the baseline, which the test suite asserts.
