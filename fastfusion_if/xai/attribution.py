"""Modality attribution for FastFusion-IF.

Method
------
The residue representation entering the classifier head is a sum of terms
injected at one point in one space::

    z_k = g_k * a_k + (1 - g_k) * s_k
    r_k = z_k + evo_k [+ plm_k]

Attribution is computed for each term at that point with **Layer Integrated
Gradients**: the term is scaled by alpha in [0, 1] while the others are held at
their observed values, and the gradient of the target with respect to the term
is integrated over alpha.

The baseline is the zero vector *for that term*, which is meaningful here in a
way it is not at the input: zeroing ``evo_k`` is exactly the intervention
"remove the evolutionary contribution", and the model's own surface-off ablation
zeroes ``surface_res`` in the same way. Because all terms share a space and are
summed, the resulting attributions are on one scale and may be compared and
ratioed. This is the precondition a modality-reliance score requires.

The target is the **logit**, never the probability. A residue predicted at 0.98
has a near-zero sigmoid gradient, so probability-space attribution would report
that confident predictions are caused by nothing.

Two attribution scopes, and why the distinction matters
------------------------------------------------------
``residue_context`` is a graph transformer over the residue neighbourhood, so
``logit_j`` depends on ``r_k`` for every ``k`` in ``j``'s neighbourhood, not just
on ``r_j``. That makes "the attribution of residue k" ambiguous, and the two
sensible readings have different completeness properties.

``scope="total"`` (default)
    Backward target is ``sum_j logit_j``, so ``A_t,k`` is the total influence of
    residue *k*'s term *t* on the whole chain's summed logit. Completeness holds
    **globally**::

        sum_k sum_t A_t,k  =  sum_j logit_j - sum_j logit_j(baseline)

    It does **not** hold per residue, because residue k's term also moves its
    neighbours' logits. One backward pass per integration step.

``scope="self"``
    ``residue_context`` is bypassed during the replay, so ``logit_k`` depends
    only on ``r_k`` and per-residue completeness holds exactly. This explains a
    slightly different function -- the model without its context encoder -- and
    is the right choice for a single-residue case study where the question is
    "what drove *this* residue".

``convergence_delta_global`` is the number to check; ``convergence_delta`` is a
per-residue diagnostic that is only expected to be near zero under
``scope="self"``.

Why ``scope="self"`` cannot use a zero baseline
-----------------------------------------------
The classifier head begins with ``LayerNorm``, and LayerNorm is **scale
invariant**: ``LN(a*x) == LN(x)`` for any ``a > 0``, because subtracting the mean
and dividing by the standard deviation both cancel the scale. Under
``scope="self"`` the head is the entire tail, so ``classify(a*r)`` is constant
along the whole ray from the origin. All of the change from ``F(0)`` to ``F(r)``
is concentrated in a discontinuity at ``a = 0``, which no Riemann sum can see,
and the completeness residual stays large no matter how many steps are used.
(In practice it is large but not total, because the ``eps`` inside LayerNorm
breaks exact scale invariance near zero.)

The fix is a baseline that does not lie on that ray. ``baseline="mean"`` uses the
chain-average of each term, so the path runs from "the average residue of this
protein" to "this residue" -- which is also the more natural question for a
single-residue case study. ``scope="self"`` therefore defaults to
``baseline="mean"``.

Under ``scope="total"`` the residue-context encoder runs first. Its residual
connections (``h + Attn(h)`` before the norm) break the scale invariance, so the
zero baseline is well behaved there and remains the default.

Splitting geometry into atom and surface
----------------------------------------
``z_geom = g * a + (1 - g) * s`` is linear in ``a`` and ``s`` given ``g``, so the
geometric attribution splits **exactly**, with no further approximation::

    A_atom,k = sum_c  grad_kc * g_kc * a_kc
    A_surf,k = sum_c  grad_kc * (1 - g_kc) * s_kc
    A_atom,k + A_surf,k = A_geom,k                       (identity, asserted in tests)

Interpretation limit
--------------------
``a`` and ``s`` are outputs of bidirectional ``CrossModalFusion``, so each already
carries information from the other stream. ``A_atom`` and ``A_surf`` measure
reliance on the atom- and surface-*pathways*, not on atomic and surface
*information*. Reliance on information is estimated separately, by intervention
(``interventions.py``), and the two are compared in the reported results.

Reproducibility
---------------
Scatter reductions on CUDA use ``atomicAdd``, which is not deterministic: two
identical forward passes of this model differ in the last bits of float32. That
is a property of the model, not of this code, and it means attributions are
reproducible to about single precision rather than bitwise. The test suite
measures the model's own run-to-run spread and asserts against that, rather than
demanding exact equality.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .hooks import InstrumentedModel

EPS = 1e-8


@dataclass
class ResidueAttribution:
    """Per-residue scalar attribution for each modality term."""

    atom: torch.Tensor           # (R,)
    surface: torch.Tensor        # (R,)
    evolutionary: torch.Tensor   # (R,)
    plm: torch.Tensor | None     # (R,) or None
    geom: torch.Tensor           # (R,) = atom + surface, kept for the identity check
    logit: torch.Tensor          # (R,)
    prob: torch.Tensor           # (R,)
    gate_mean: torch.Tensor      # (R,) mean over channels of g_k
    convergence_delta: torch.Tensor    # (R,) per-residue residual; see module docstring
    convergence_delta_global: float    # the residual that is expected to vanish
    scope: str = "total"
    baseline: str = "zero"

    def as_dict(self) -> dict[str, torch.Tensor]:
        d = {
            "attr_atom": self.atom,
            "attr_surface": self.surface,
            "attr_evolutionary": self.evolutionary,
            "attr_geom": self.geom,
            "logit": self.logit,
            "prob": self.prob,
            "gate_mean": self.gate_mean,
            "ig_convergence_delta": self.convergence_delta,
        }
        if self.plm is not None:
            d["attr_plm"] = self.plm
        return d

    @property
    def relative_global_error(self) -> float:
        """Completeness residual as a fraction of the total logit mass."""
        denom = float(self.logit.abs().sum()) + 1e-6
        return abs(self.convergence_delta_global) / denom


def _riemann_alphas(n_steps: int, device, dtype) -> torch.Tensor:
    """Midpoint rule: lower variance than left/right endpoints at equal cost."""
    i = torch.arange(n_steps, device=device, dtype=dtype)
    return (i + 0.5) / n_steps


def _classify(model, r: torch.Tensor) -> torch.Tensor:
    """Run the decoder readout, tolerating either API name."""
    dec = model.decoder
    fn = getattr(dec, "classify", None)
    if callable(fn):
        out = fn(r)
    else:
        out = dec.head(r)
    return out.squeeze(-1) if out.dim() > 1 else out


def integrated_gradients(
    model,
    batch: dict,
    n_steps: int = 64,
    target: str = "logit",
    scope: str = "total",
    baseline: str | None = None,
) -> ResidueAttribution:
    """Layer-IG at the residue injection points.

    Parameters
    ----------
    model : FastFusionIF, in eval mode.
    batch : one protein, as produced by the project collate function.
    n_steps : Riemann steps. 64 keeps the global completeness residual well
        under 1% on the benchmark proteins.
    target : ``"logit"`` (default and recommended) or ``"prob"``.
    scope : ``"total"`` or ``"self"``; see the module docstring.
    baseline : ``"zero"`` or ``"mean"``. ``None`` selects the appropriate default
        for the scope -- ``"zero"`` for ``total``, ``"mean"`` for ``self``. See
        the module docstring for why ``self`` cannot use a zero baseline.
    """
    if target not in {"logit", "prob"}:
        raise ValueError(f"target must be 'logit' or 'prob', got {target!r}")
    if scope not in {"total", "self"}:
        raise ValueError(f"scope must be 'total' or 'self', got {scope!r}")
    if baseline is None:
        baseline = "zero" if scope == "total" else "mean"
    if baseline not in {"zero", "mean"}:
        raise ValueError(f"baseline must be 'zero' or 'mean', got {baseline!r}")

    model.eval()
    inst = InstrumentedModel(model)

    # Pass 1: observe the terms at their real values.
    # to_cpu=False: these tensors are replayed through the network tail below,
    # so they must stay on the model's device.
    with torch.no_grad(), inst.capture(to_cpu=False) as obs:
        model(batch)

    z_geom = obs.z_geom
    device, dtype = z_geom.device, z_geom.dtype
    R, D = z_geom.shape

    evo = obs.evo_term
    if evo is None:
        evo = torch.zeros(R, D, dtype=dtype, device=device)
    plm = obs.plm_term
    has_plm = plm is not None

    terms = {"geom": z_geom, "evo": evo}
    if has_plm:
        terms["plm"] = plm

    use_ctx = scope == "total" and getattr(model, "residue_context", None) is not None

    def tail(r: torch.Tensor) -> torch.Tensor:
        if use_ctx:
            r = model.residue_context(r, batch["residue_pos"], batch["residue_edge_index"])
        return _classify(model, r)

    # Baselines, one per term. "mean" uses the chain-average of that term, which
    # keeps the IG path off the scale-invariant ray through the origin.
    if baseline == "zero":
        base = {k: torch.zeros_like(v) for k, v in terms.items()}
    else:
        base = {k: v.mean(0, keepdim=True).expand_as(v).contiguous() for k, v in terms.items()}
    delta = {k: terms[k] - base[k] for k in terms}

    grads = {k: torch.zeros_like(v) for k, v in terms.items()}
    for a in _riemann_alphas(n_steps, device, dtype):
        scaled = {k: (base[k] + delta[k] * a).detach().requires_grad_(True) for k in terms}
        r = scaled["geom"] + scaled["evo"]
        if has_plm:
            r = r + scaled["plm"]
        out = tail(r)
        if target == "prob":
            out = torch.sigmoid(out)
        g = torch.autograd.grad(out.sum(), list(scaled.values()))
        for k, gi in zip(scaled.keys(), g):
            grads[k] += gi.detach() / n_steps

    attr_evo = (grads["evo"] * delta["evo"]).sum(-1)
    attr_plm = (grads["plm"] * delta["plm"]).sum(-1) if has_plm else None

    # Exact split of the geometric term by the gate. z = g*a + (1-g)*s is linear
    # in a and s given g, so the same linearity carries the baseline offset.
    gate, a_res, s_res = obs.gate, obs.atom_res, obs.surface_res
    if baseline == "zero":
        d_a, d_s = a_res, s_res
    else:
        d_a = a_res - a_res.mean(0, keepdim=True)
        d_s = s_res - s_res.mean(0, keepdim=True)
    attr_atom = (grads["geom"] * gate * d_a).sum(-1)
    attr_surf = (grads["geom"] * (1.0 - gate) * d_s).sum(-1)
    attr_geom = (grads["geom"] * delta["geom"]).sum(-1)

    with torch.no_grad():
        r_full = z_geom + evo + (plm if has_plm else 0.0)
        logit_full = tail(r_full)
        r_base = base["geom"] + base["evo"] + (base["plm"] if has_plm else 0.0)
        logit_base = tail(r_base)
        per_residue = attr_geom + attr_evo + (attr_plm if has_plm else 0.0)
        delta_res = per_residue - (logit_full - logit_base)
        # Completeness is a statement about the backward target, which is the
        # SUM of logits. Only the summed residual is expected to vanish under
        # scope="total"; under scope="self" the per-residue residual vanishes too.
        delta_global = float(delta_res.sum())

    return ResidueAttribution(
        atom=attr_atom.detach().cpu(),
        surface=attr_surf.detach().cpu(),
        evolutionary=attr_evo.detach().cpu(),
        plm=attr_plm.detach().cpu() if has_plm else None,
        geom=attr_geom.detach().cpu(),
        logit=logit_full.detach().cpu(),
        prob=torch.sigmoid(logit_full).detach().cpu(),
        gate_mean=gate.mean(-1).detach().cpu(),
        convergence_delta=delta_res.detach().cpu(),
        convergence_delta_global=delta_global,
        scope=scope,
        baseline=baseline,
    )


def _project(attr_residue, index, weights):
    """Shared body of the atom/surface projections, device-safe.

    ``attr_residue`` arrives on CPU (attributions are detached there) while the
    index tensors live on the model's device. Indexing across devices raises, so
    everything is brought onto the attribution's device first.
    """
    dev = attr_residue.device
    index = index.to(dev).long()
    if weights is None:
        counts = torch.bincount(index, minlength=int(attr_residue.shape[0])).clamp_min(1)
        return attr_residue[index] / counts[index].to(attr_residue.dtype)
    return attr_residue[index] * weights.to(dev).to(attr_residue.dtype)


def project_to_atoms(attr_residue, atom2res, atom_pool_weights):
    """Distribute a per-residue score over that residue's atoms.

    Uses the attention-pool weights the model itself computed, so the split
    reflects which atoms the model actually attended to. Falls back to a uniform
    split when the checkpoint uses mean pooling, in which case the total is
    conserved exactly.
    """
    return _project(attr_residue, atom2res, atom_pool_weights)


def project_to_surface(attr_residue, surface2res, surface_pool_weights):
    """Same as :func:`project_to_atoms`, for surface points."""
    return _project(attr_residue, surface2res, surface_pool_weights)
