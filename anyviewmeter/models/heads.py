"""Prediction heads.  Same shapes as Robometer's so checkpoints stay comparable."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _mlp_head(hidden_dim: int, out_dim: int, dropout: float,
              sigmoid: bool) -> nn.Sequential:
    layers = [
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.LayerNorm(hidden_dim // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim // 2, out_dim),
    ]
    if sigmoid:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class PredictionHeadsMixin(nn.Module):
    """Progress / success / preference heads.

    ``progress_loss_type='discrete'`` switches the progress head to a C51-style
    distribution over ``progress_discrete_bins`` and drops the sigmoid, matching
    Robometer.
    """

    def __init__(self, *args, hidden_dim: Optional[int] = None,
                 model_config: Optional[object] = None, dropout: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if hidden_dim is None:
            self.use_discrete_progress = False
            return

        loss_type = getattr(model_config, "progress_loss_type", "l2") if model_config else "l2"
        self.use_discrete_progress = str(loss_type).lower() == "discrete"
        if self.use_discrete_progress:
            out_dim = int(getattr(model_config, "progress_discrete_bins", 10))
            use_sigmoid = False
        else:
            out_dim, use_sigmoid = 1, True

        self.progress_head = _mlp_head(hidden_dim, out_dim, dropout, use_sigmoid)
        self.success_head = _mlp_head(hidden_dim, 1, dropout, False)
        self.preference_head = _mlp_head(hidden_dim, 1, dropout, False)


class PoseConsistencyHead(nn.Module):
    """Optional auxiliary: predict the viewpoint descriptor back from the features.

    A probe, not a component -- if the frame features cannot recover the camera
    pose, the injection did not reach the representation the heads read from, and
    the tier comparison would be measuring nothing.  Reported as a diagnostic
    alongside the E4 gate rather than used to make claims on its own.
    """

    def __init__(self, hidden_dim: int, pose_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, pose_dim),
        )

    def forward(self, frame_feats: torch.Tensor) -> torch.Tensor:
        return self.net(frame_feats)
