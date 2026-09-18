"""Predict the camera from the image, trained ONLY by the progress loss.

WHAT THIS IS FOR.  P1 needs extrinsics at inference; without them it scores BELOW the
no-camera baseline (az_far tau +0.625 against R1's +0.677), because it was trained with
--pose-dropout 0 and has never seen a missing camera.  A supervised pose regressor was
tried first and fails where it matters: trained on in-cone cameras it reaches 6-9 deg
median error in-cone but 41 deg at az_edge and 82 deg at az_far, which is past the point
where a wrong camera is worse than none.  That failure is extrapolation, not capacity.

WHAT THIS CHANGES.  Nothing supervises the pose here.  The head is trained by the task
loss alone, so it is free to learn whatever camera-shaped code helps predict progress.

  THE RISK, STATED UP FRONT: nothing forces that code to be the TRUE pose.  If it is not,
  this stops being camera conditioning and becomes image-conditioned extra capacity --
  exactly the null hypothesis the perturbation ladder was built to exclude, and the ladder
  cannot be run on a learned code because there is no ground truth to perturb.  So the
  experiment is only interpretable together with a probe of the learned code against the
  true extrinsics.  A win with an uninformative code is a negative result, not a win.

WHAT IS ASSUMED KNOWN.  The workspace centre.  Every camera in this rig looks at it, so
the free parameters are (az, el, radius, fov) rather than a full 6-DoF pose -- a property
of the rig, not of the method, and it makes the "no calibration" claim weaker than it
would be for a camera pointing anywhere.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PoseHead(nn.Module):
    def __init__(self, el_range=(0.05, 1.05), r_range=(0.42, 0.95),
                 fov_range=(0.60, 1.15), width: int = 48):
        super().__init__()
        self.el_range, self.r_range, self.fov_range = el_range, r_range, fov_range

        def blk(i, o, s=2):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.SiLU())
        w = width
        self.body = nn.Sequential(blk(3, w), blk(w, w * 2), blk(w * 2, w * 2),
                                  blk(w * 2, w * 3), blk(w * 3, w * 3),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten())
        # azimuth comes out as (sin, cos) so the +-pi wrap is not a cliff in the loss
        self.out = nn.Linear(w * 3, 5)

    def forward(self, frames: torch.Tensor):
        """frames ``(T, 3, H, W)`` in [0, 1] -> az, el, radius, fov, each ``(T,)``."""
        h = self.out(self.body(frames))
        az = torch.atan2(h[:, 0], h[:, 1])
        def span(x, lo, hi):
            return lo + (hi - lo) * torch.sigmoid(x)
        return (az, span(h[:, 2], *self.el_range), span(h[:, 3], *self.r_range),
                span(h[:, 4], *self.fov_range))
