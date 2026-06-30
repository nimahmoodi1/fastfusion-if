from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Drop-in replacement for fastfusion_if/losses.py
#
# Backward compatible: with the default arguments (focal_weight=0.0,
# tversky_weight=0.0) interface_loss() is byte-for-byte equivalent to your
# current weighted-BCE + Dice loss, so existing configs keep their behaviour.
#
# New optional terms:
#   - focal_weight > 0   -> adds alpha-balanced focal BCE (Lin et al. 2017)
#   - tversky_weight > 0 -> adds a Tversky / focal-Tversky region loss, which
#     is often stronger than Dice for heavy class imbalance because alpha/beta
#     let you weight false negatives more than false positives.
#
# To use focal loss (your planned Phase 5):
#   1) add the fields below to TrainConfig in fastfusion_if/config.py:
#         focal_weight: float = 0.0
#         focal_gamma: float = 2.0
#         focal_alpha: float = 0.25
#         tversky_weight: float = 0.0
#         tversky_alpha: float = 0.7   # weight on false negatives
#         tversky_beta: float = 0.3    # weight on false positives
#         tversky_gamma: float = 1.0   # 1.0 = Tversky, >1 = focal-Tversky
#   2) in scripts/train.py change the loss call inside run_epoch() to:
#         loss = interface_loss(
#             logits, y,
#             positive_weight=cfg.train.positive_weight,
#             dice_weight=cfg.train.dice_weight,
#             focal_weight=cfg.train.focal_weight,
#             focal_gamma=cfg.train.focal_gamma,
#             focal_alpha=cfg.train.focal_alpha,
#             tversky_weight=cfg.train.tversky_weight,
#             tversky_alpha=cfg.train.tversky_alpha,
#             tversky_beta=cfg.train.tversky_beta,
#             tversky_gamma=cfg.train.tversky_gamma,
#         )
# ---------------------------------------------------------------------------


def soft_dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum() + eps
    den = probs.sum() + targets.sum() + eps
    return 1.0 - num / den


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Numerically stable alpha-balanced focal BCE.

    Computed in float32 internally so it is safe under CUDA AMP/autocast.
    """
    logits_f = logits.float()
    targets_f = targets.float()
    p = torch.sigmoid(logits_f)
    ce = F.binary_cross_entropy_with_logits(logits_f, targets_f, reduction="none")
    p_t = p * targets_f + (1.0 - p) * (1.0 - targets_f)
    focal_term = (1.0 - p_t).clamp_min(eps) ** gamma
    if alpha is not None and alpha >= 0.0:
        alpha_t = alpha * targets_f + (1.0 - alpha) * (1.0 - targets_f)
        loss = alpha_t * focal_term * ce
    else:
        loss = focal_term * ce
    return loss.mean().to(logits.dtype)


def tversky_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """(Focal-)Tversky loss. alpha weights false negatives, beta false positives.

    gamma == 1.0 -> standard Tversky. gamma > 1.0 -> focal-Tversky (focuses on
    hard, low-overlap examples). For interface prediction you usually want
    alpha > beta because recall on the rare positive (interface) class matters.
    """
    logits_f = logits.float()
    targets_f = targets.float()
    p = torch.sigmoid(logits_f)
    tp = (p * targets_f).sum()
    fn = ((1.0 - p) * targets_f).sum()
    fp = (p * (1.0 - targets_f)).sum()
    tversky = (tp + eps) / (tp + alpha * fn + beta * fp + eps)
    loss = (1.0 - tversky) ** gamma
    return loss.to(logits.dtype)


def interface_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: float | None = None,
    dice_weight: float = 0.2,
    focal_weight: float = 0.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    tversky_weight: float = 0.0,
    tversky_alpha: float = 0.7,
    tversky_beta: float = 0.3,
    tversky_gamma: float = 1.0,
) -> torch.Tensor:
    pos_weight = (
        None
        if positive_weight is None
        else torch.tensor(float(positive_weight), device=logits.device, dtype=logits.dtype)
    )
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    loss = bce
    if dice_weight > 0:
        loss = loss + dice_weight * soft_dice_loss_with_logits(logits, targets)
    if focal_weight > 0:
        loss = loss + focal_weight * focal_bce_with_logits(
            logits, targets, alpha=focal_alpha, gamma=focal_gamma
        )
    if tversky_weight > 0:
        loss = loss + tversky_weight * tversky_loss_with_logits(
            logits, targets, alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma
        )
    return loss
