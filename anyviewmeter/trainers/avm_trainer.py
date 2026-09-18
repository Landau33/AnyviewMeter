"""Losses and the training step.

The interesting term is ``pose_consistency``: it penalises disagreement between the
progress curves two cameras produce for the SAME trajectory, which is the direct
training signal for viewpoint robustness.

It is also the term most able to produce a fraudulent result, so it is worth being
explicit.  The cheapest way to satisfy it is to stop looking at the image and emit
a fixed function of frame index -- which is *exactly* the position prior phase A
measured in Robometer, the checkpoint we start from.  A consistency loss alone will
happily drive S_view to zero that way.  That is why:

  * the loss keeps ``progress`` and ``success`` terms at full weight, so ignoring
    pixels is penalised;
  * ``pose_dropout`` forces the model to remain usable without pose;
  * the E4 gate (evals/diagnostics.py) is the acceptance criterion, not the loss
    value, and it disqualifies an S_view win bought with H_time or P_obj.

Do not raise ``pose_consistency_weight`` without watching H_time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossTerms:
    total: torch.Tensor
    parts: Dict[str, float] = field(default_factory=dict)


def progress_loss(pred: torch.Tensor, target: torch.Tensor,
                  discrete: bool = False, n_bins: int = 10) -> torch.Tensor:
    """L2 on continuous progress, or cross-entropy on a binned distribution."""
    if not discrete:
        return F.mse_loss(pred.float(), target.float())
    b = torch.clamp((target * n_bins).long(), 0, n_bins - 1)
    return F.cross_entropy(pred.reshape(-1, n_bins).float(), b.reshape(-1))


def success_loss(logits: torch.Tensor, target: torch.Tensor,
                 pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.float(), target.float(),
                                              pos_weight=pos_weight)


def pose_consistency_loss(progress: torch.Tensor, groups: Dict[int, List[int]],
                          ) -> torch.Tensor:
    """Mean squared disagreement between views of the same trajectory.

    Only defined for groups with >= 2 views; groups of one contribute nothing
    rather than silently pulling toward zero.
    """
    terms = []
    for rows in groups.values():
        if len(rows) < 2:
            continue
        p = progress[rows]                        # (V, T)
        centre = p.mean(dim=0, keepdim=True)
        terms.append(((p - centre) ** 2).mean())
    if not terms:
        return progress.new_zeros(())
    return torch.stack(terms).mean()


def pose_probe_loss(pred: torch.Tensor, pose_vec: torch.Tensor) -> torch.Tensor:
    """Auxiliary probe: can the frame features recover the viewpoint descriptor?

    Diagnostic only.  If this stays high the injection never reached the features
    the heads read, which means a tier comparison would be measuring nothing.
    """
    return F.mse_loss(pred.float(), pose_vec.float())


def compute_losses(output, batch, cfg, groups: Optional[Dict[int, List[int]]] = None,
                   discrete: bool = False, n_bins: int = 10) -> LossTerms:
    parts: Dict[str, float] = {}
    total = output.progress_logits.new_zeros(())

    lp = progress_loss(output.progress_logits, batch.target_progress, discrete, n_bins)
    total = total + cfg.progress_weight * lp
    parts["progress"] = float(lp.detach())

    if cfg.success_weight > 0 and output.success_logits is not None:
        ls = success_loss(output.success_logits, batch.success_label)
        total = total + cfg.success_weight * ls
        parts["success"] = float(ls.detach())

    if cfg.pose_consistency_weight > 0 and groups:
        lc = pose_consistency_loss(output.progress_logits, groups)
        total = total + cfg.pose_consistency_weight * lc
        parts["pose_consistency"] = float(lc.detach())

    if output.pose_pred is not None:
        # frame-mean pose target: the probe reads a per-frame feature
        pv = batch.pose_vec.reshape(output.pose_pred.shape[0], -1, batch.pose_vec.shape[-1])
        lprobe = pose_probe_loss(output.pose_pred, pv)
        parts["pose_probe"] = float(lprobe.detach())   # reported, not optimised

    parts["total"] = float(total.detach())
    return LossTerms(total=total, parts=parts)


def build_optimizer(model, training_cfg):
    """Two parameter groups: the pose adapter gets its own (higher) learning rate.

    The adapter starts at zero by construction and has to travel much further than
    pretrained weights, so sharing one LR either crawls on the adapter or wrecks
    the backbone.
    """
    pose_ids = {id(p) for p in model.pose_parameters()}
    pose_params = [p for p in model.parameters() if id(p) in pose_ids and p.requires_grad]
    other = [p for p in model.parameters() if id(p) not in pose_ids and p.requires_grad]

    groups = [{"params": other, "lr": training_cfg.lr}]
    if pose_params:
        groups.append({"params": pose_params,
                       "lr": training_cfg.pose_lr or training_cfg.lr})
    return torch.optim.AdamW(groups, weight_decay=training_cfg.weight_decay)


def clip_grads(model: nn.Module, max_norm: float) -> float:
    return float(torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm))
