# Limitations — explainability

**Post-hoc attribution describes the model, not the biology.** Every number here
is a statement about how FastFusion-IF computes its output. A residue with high
surface reliance is one whose prediction the surface pathway drives; it is not
thereby a residue whose binding is driven by surface complementarity. The model
may rely on a feature for reasons that have no biological content — a dataset
artefact, a spurious correlation, or a shortcut. Nothing in an attribution map
licenses a causal claim about molecular recognition, and we make none.

**The gate does not measure information provenance.** `a_k` and `s_k` are pooled
from the outputs of bidirectional cross-modal fusion, so each already carries
information from the other stream. The gate is a mixing weight over two entangled
representations. We report it because it is what the model computes, and we
validate it against attribution and intervention rather than presenting it as an
explanation.

**Attribution and pathway reliance are not the same thing.** For the same reason,
`A_atom` and `A_surf` measure reliance on the atom- and surface-*pathways*, not
on atomic and surface *information*. Only the intervention results speak to
information, and only the retrained surface-off comparison does so cleanly.

**Test-time surface zeroing is out of distribution.** Setting `use_surface =
False` on a surface-trained checkpoint gives the decoder a gate input it never
saw. It measures what removing the surface does at test time, which overstates
the surface contribution relative to a model that was trained without it.

**Integrated Gradients depends on its baseline.** The zero vector is meaningful
at the injection points but is still a choice, and attributions are defined
relative to it. The completeness residual is reported so that the reader can see
how much of the logit the attributions actually account for.

**The unbound set has 25 proteins.** Per-protein tests on it are underpowered;
the bootstrap interval for per-protein AUPRC is 0.349–0.539, three times wider
than the bound interval. Bound/unbound differences in reliance should be read as
suggestive unless the effect is large.

**Explanations are for one architecture and one training run.** Reliance is a
property of the trained weights. A model retrained with a different seed may
distribute reliance differently while making similar predictions, which is why
the three-seed ensemble is analysed member by member rather than as a single
averaged object.
