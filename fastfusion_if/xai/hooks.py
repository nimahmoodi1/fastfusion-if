"""Non-invasive instrumentation of FastFusion-IF for explainability.

Design constraint: ``fastfusion_if/models/model.py`` is not modified. All
intermediates are captured with forward hooks on existing submodules, so
checkpoints stay byte-compatible and predictions with XAI disabled are
bit-identical to the baseline (verified by ``tests/test_xai.py``).

What gets captured, and why these points
----------------------------------------
``FastFusionIF.forward`` builds the residue representation as a sum of terms
that are injected at the *same* point, with the *same* dimensionality::

    z_k       = g_k * a_k + (1 - g_k) * s_k        # decoder.fuse_residues
    r_k       = z_k + lambda * phi_r(f_k)          # residue_feature_mlp, lambda = 0.5
    r_k       = r_k + plm_gate * plm_h             # optional, "add" mode
    logits    = head(residue_context(r_k))

Because the geometric, evolutionary and language-model terms are *additive at a
common point in a common space*, gradients taken there are directly comparable
across modalities. That is the precondition a modality-reliance ratio needs, and
it is why attribution is computed at these latent injection points rather than at
the raw inputs, where "zero" has no meaning for coordinates or a graph.

One caveat is structural and must be carried into any interpretation. ``a_k`` and
``s_k`` are pooled from the *outputs* of ``CrossModalFusion``, which is
bidirectional. Surface information has therefore already entered ``a_k``, and
atomic information has already entered ``s_k``. The gate is a mixing weight over
two entangled representations, not a measure of information provenance. See
``attribution.py`` for how this is handled and ``interventions.py`` for the
independent check.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class Intermediates:
    """Tensors captured during one instrumented forward pass.

    Every tensor is indexed by residue except ``atom_pool_weights`` and
    ``surface_pool_weights``, which are indexed by atom and surface point.
    """

    gate: torch.Tensor | None = None            # (R, D) per-channel mixing weight
    atom_res: torch.Tensor | None = None        # (R, D) atom-pathway residue repr
    surface_res: torch.Tensor | None = None     # (R, D) surface-pathway residue repr
    z_geom: torch.Tensor | None = None          # (R, D) gate output
    evo_term: torch.Tensor | None = None        # (R, D) lambda * phi_r(f)
    plm_term: torch.Tensor | None = None        # (R, D) plm contribution, if any
    residue_h_pre_ctx: torch.Tensor | None = None   # (R, D) before residue context
    residue_h_post_ctx: torch.Tensor | None = None  # (R, D) after residue context
    logits: torch.Tensor | None = None          # (R,)
    atom_pool_weights: torch.Tensor | None = None     # (A,) segment-softmax weights
    surface_pool_weights: torch.Tensor | None = None  # (S,)
    meta: dict[str, Any] = field(default_factory=dict)

    def detach_cpu_(self) -> "Intermediates":
        """Detach every tensor onto the CPU **in place**, and return self.

        In place matters. ``capture()`` yields this object to the caller, so
        rebinding the manager's own attribute to a fresh copy would leave the
        caller holding the original CUDA tensors -- which then raise
        "can't convert cuda:0 device type tensor to numpy" the moment the caller
        calls ``.numpy()``. Mutating the yielded object is the only way the
        caller sees the result.
        """
        for k, v in list(self.__dict__.items()):
            if k == "meta":
                continue
            if torch.is_tensor(v):
                setattr(self, k, v.detach().to("cpu"))
        return self

    def detach_cpu(self) -> "Intermediates":
        """Backwards-compatible alias; now mutates in place and returns self."""
        return self.detach_cpu_()


class InstrumentedModel:
    """Wraps a ``FastFusionIF`` instance and records intermediates via hooks.

    Usage::

        inst = InstrumentedModel(model)
        with inst.capture() as cap:
            logits = model(batch)
        gate = cap.gate            # (R, D)

    The wrapper installs hooks only inside the ``capture`` context, so a model
    used outside it behaves exactly as before.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._cap: Intermediates | None = None

    # ---------------------------------------------------------------- helpers
    @property
    def _decoder(self) -> nn.Module:
        return self.model.decoder

    def _has_evo(self) -> bool:
        return getattr(self.model, "residue_feature_mlp", None) is not None

    def _has_plm(self) -> bool:
        return getattr(self.model, "plm_proj", None) is not None

    # ------------------------------------------------------------------ hooks
    def _hook_gate(self, _mod, inp, out) -> None:
        # decoder.gate is nn.Sequential(...Sigmoid()); `out` is the gate itself.
        self._cap.gate = out
        joined = inp[0]
        d = out.shape[-1]
        # joined = [atom_res | surface_res | |a-s| | a*s]
        self._cap.atom_res = joined[..., :d]
        self._cap.surface_res = joined[..., d : 2 * d]
        self._cap.z_geom = out * joined[..., :d] + (1.0 - out) * joined[..., d : 2 * d]

    def _hook_evo(self, _mod, _inp, out) -> None:
        scale = float(getattr(self.model, "residue_feature_scale", 0.25))
        self._cap.evo_term = scale * out

    def _hook_plm(self, _mod, _inp, out) -> None:
        # "add" mode scales the projection by a learned scalar gate; "concat"
        # mode feeds it through plm_combine instead, in which case the additive
        # decomposition does not apply and the PLM term is left unset rather
        # than reported wrongly.
        if getattr(self.model, "plm_combine", None) is not None:
            self._cap.meta["plm_mode"] = "concat"
            self._cap.plm_term = None
            return
        gate = getattr(self.model, "plm_gate", None)
        self._cap.meta["plm_mode"] = "add"
        self._cap.plm_term = out * gate if gate is not None else out

    def _hook_head(self, _mod, inp, out) -> None:
        self._cap.residue_h_post_ctx = inp[0]
        self._cap.logits = out.squeeze(-1) if out.dim() > 1 else out

    def _hook_pool(self, which: str):
        def fn(mod, inp, _out) -> None:
            # AttentionPool.forward(h, index, dim_size); recompute the weights it used.
            try:
                from ..models.pooling import segment_softmax  # type: ignore
            except ImportError:  # pooling helper renamed: skip weights, not fatal
                return

            h, index = inp[0], inp[1]
            if h.numel() == 0:
                return
            scores = mod.score(h).squeeze(-1)
            w = segment_softmax(scores, index, int(inp[2]))
            setattr(self._cap, f"{which}_pool_weights", w)

        return fn

    # ---------------------------------------------------------------- context
    @contextmanager
    def capture(self, to_cpu: bool = True):
        """Install hooks, yield the :class:`Intermediates` they fill.

        Parameters
        ----------
        to_cpu
            Move every captured tensor to the CPU when the context exits.
            ``True`` (default) is what a caller wants if it will call
            ``.numpy()`` or write the values to disk.

            Pass ``False`` when the captured tensors are about to be fed back
            through the model -- :func:`~fastfusion_if.xai.attribution.integrated_gradients`
            replays the network tail with them, so they must stay on the model's
            device. Getting this wrong raises either "can't convert cuda:0
            device type tensor to numpy" (moved when it should not have been) or
            "Expected all tensors to be on the same device" (not moved when it
            should have been); the test suite pins both directions.
        """
        self._cap = Intermediates()
        dec = self._decoder
        self._handles = [
            dec.gate.register_forward_hook(self._hook_gate),
            dec.head.register_forward_hook(self._hook_head),
        ]
        if getattr(dec, "atom_pool", None) is not None:
            self._handles.append(dec.atom_pool.register_forward_hook(self._hook_pool("atom")))
        if getattr(dec, "surface_pool", None) is not None:
            self._handles.append(
                dec.surface_pool.register_forward_hook(self._hook_pool("surface"))
            )
        if self._has_evo():
            self._handles.append(
                self.model.residue_feature_mlp.register_forward_hook(self._hook_evo)
            )
        if self._has_plm():
            self._handles.append(self.model.plm_proj.register_forward_hook(self._hook_plm))
        try:
            yield self._cap
        finally:
            for h in self._handles:
                h.remove()
            self._handles = []
            if to_cpu and self._cap is not None:
                # In place: the caller holds a reference to this exact object,
                # so rebinding self._cap would leave them with device tensors.
                self._cap.detach_cpu_()


def modality_availability(model: nn.Module) -> dict[str, bool]:
    """Which modality pathways this checkpoint actually has.

    Read from the loaded module, not from a config file, so a checkpoint whose
    stored config disagrees with its weights is reported correctly.
    """
    return {
        "atom": True,
        "surface": bool(getattr(model, "use_surface", True)),
        "evolutionary": getattr(model, "residue_feature_mlp", None) is not None,
        "plm": getattr(model, "plm_proj", None) is not None,
        "residue_context": getattr(model, "residue_context", None) is not None,
    }
