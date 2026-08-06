"""Interventions and faithfulness tests.

Attribution says which term the gradient flows through. Intervention says what
happens to the prediction when a term is actually removed. They answer different
questions and can disagree, so both are computed and their agreement is reported
rather than assumed.

Two kinds of surface removal
----------------------------
``surface_zero`` sets ``use_surface = False`` at inference on a checkpoint that
was *trained with* surface. The decoder then sees ``[a; 0; |a|; 0]``, a gate
input it never saw during training, so this is an out-of-distribution
intervention. It measures "what happens if the surface is taken away at test
time", which is not the same as "what the surface contributes".

``surface_retrained`` compares against the separately trained surface-off
checkpoints (``runs/bench_evo_nosurf_pp*``). This is the honest estimate of the
surface contribution, because the rest of the model was free to compensate.

Both are reported. Where they disagree, the retrained comparison is the one to
believe, and the report says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

MODALITIES = ("surface", "evolutionary", "plm", "residue_context")


@dataclass
class InterventionResult:
    name: str
    prob_before: np.ndarray
    prob_after: np.ndarray
    delta_logit: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def mean_abs_delta_prob(self) -> float:
        return float(np.abs(self.prob_after - self.prob_before).mean())


def _clone_batch(batch: dict) -> dict:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def run_intervention(
    model,
    batch: dict,
    modality: str,
    mode: str = "zero",
    generator: torch.Generator | None = None,
) -> InterventionResult:
    """Ablate one modality and measure the change in prediction.

    Parameters
    ----------
    modality : one of :data:`MODALITIES`.
    mode : ``"zero"`` replaces the input with zeros; ``"shuffle"`` permutes it
        across residues, which preserves the marginal distribution and so
        separates "this feature matters" from "this feature's *value* matters";
        ``"mean"`` replaces it with the batch mean.
    """
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}; expected one of {MODALITIES}")

    model.eval()
    base_logits = model(batch).detach()
    b = _clone_batch(batch)

    if modality == "surface":
        prev = model.use_surface
        model.use_surface = False
        try:
            new_logits = model(b).detach()
        finally:
            model.use_surface = prev

    elif modality == "residue_context":
        prev = model.residue_context
        model.residue_context = None
        try:
            new_logits = model(b).detach()
        finally:
            model.residue_context = prev

    else:  # evolutionary / plm -> act on the input tensor
        key = "residue_features" if modality == "evolutionary" else "residue_plm"
        if key not in b or b[key] is None:
            raise KeyError(f"batch has no {key!r}; this checkpoint has no {modality} pathway")
        x = b[key]
        if mode == "zero":
            b[key] = torch.zeros_like(x)
        elif mode == "mean":
            b[key] = x.mean(0, keepdim=True).expand_as(x).clone()
        elif mode == "shuffle":
            # Draw on CPU and move: a torch.Generator is bound to one device, and
            # a CPU generator with device="cuda" raises. Generating on CPU also
            # keeps the permutation identical across CPU and GPU runs for a given
            # seed, which is what makes the shuffle intervention reproducible.
            perm = torch.randperm(x.shape[0], generator=generator).to(x.device)
            b[key] = x[perm].clone()
        else:
            raise ValueError(f"unknown mode {mode!r}")
        new_logits = model(b).detach()

    return InterventionResult(
        name=f"{modality}:{mode}",
        prob_before=torch.sigmoid(base_logits).cpu().numpy(),
        prob_after=torch.sigmoid(new_logits).cpu().numpy(),
        delta_logit=(new_logits - base_logits).cpu().numpy(),
        meta={"modality": modality, "mode": mode},
    )


# --------------------------------------------------------------------- deletion
def deletion_curve(
    model,
    batch: dict,
    scores: np.ndarray,
    key: str = "residue_features",
    fractions: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
    order: str = "descending",
    generator: torch.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Progressively neutralise the highest-attributed residues and track AUPRC.

    A faithful explanation should degrade the prediction faster when the
    highest-scored residues are removed first than when random residues are
    removed. The gap between the ``descending`` and ``random`` curves is the
    faithfulness signal; a single curve in isolation says nothing, because some
    proteins simply degrade quickly under any perturbation.
    """
    from sklearn.metrics import average_precision_score

    model.eval()
    lab = None
    for k in ("labels", "label", "y", "residue_labels", "targets"):
        if k in batch and batch[k] is not None:
            lab = batch[k]
            break
    if lab is None:
        raise KeyError("no label tensor in batch; tried labels/label/y/residue_labels/targets")
    y = (lab.detach().cpu().numpy() if torch.is_tensor(lab) else np.asarray(lab)).astype(int).reshape(-1)
    n = len(scores)

    if order == "descending":
        rank = np.argsort(-scores)
    elif order == "ascending":
        rank = np.argsort(scores)
    elif order == "random":
        g = np.random.default_rng(0 if generator is None else int(generator.initial_seed()))
        rank = g.permutation(n)
    else:
        raise ValueError(f"unknown order {order!r}")

    out_frac, out_auprc, out_meanprob = [], [], []
    with torch.no_grad():
        for f in fractions:
            k = int(round(f * n))
            b = _clone_batch(batch)
            if k > 0 and key in b and b[key] is not None:
                b[key][torch.as_tensor(rank[:k], device=b[key].device)] = 0.0
            p = torch.sigmoid(model(b)).cpu().numpy()
            out_frac.append(f)
            out_meanprob.append(float(p.mean()))
            out_auprc.append(
                float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
            )
    return {
        "fraction": np.asarray(out_frac),
        "auprc": np.asarray(out_auprc),
        "mean_prob": np.asarray(out_meanprob),
        "order": order,
    }


def faithfulness_gap(desc: dict, rand: dict) -> float:
    """Area between the random and descending deletion curves.

    Positive means removing high-attribution residues hurts more than removing
    random ones, i.e. the attribution is faithful. Zero or negative means the
    explanation carries no more information than chance.
    """
    d = np.asarray(desc["auprc"], dtype=float)
    r = np.asarray(rand["auprc"], dtype=float)
    x = np.asarray(desc["fraction"], dtype=float)
    m = ~(np.isnan(d) | np.isnan(r))
    if m.sum() < 2:
        return float("nan")
    return float(np.trapz(r[m] - d[m], x[m]))
