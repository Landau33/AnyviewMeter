"""Camera parameters and the one convention this whole project depends on.

CONVENTION (OpenCV, matching ManiSkill's ``intrinsic_cv`` / ``extrinsic_cv``):

  * ``extrinsic`` is the 3x4 world-to-camera matrix ``[R | t]``, so a world point
    ``p_w`` maps to camera coordinates as ``p_c = R @ p_w + t``.
  * ``+x`` right, ``+y`` down, ``+z`` forward (into the scene).  A point in front of
    the camera has ``p_c[2] > 0``.
  * ``intrinsic`` is the 3x3 ``K`` in pixels, with the principal point in pixel
    coordinates where pixel centres sit at ``(j + 0.5, i + 0.5)``.
  * The camera centre in world coordinates is ``c = -R^T t``.

That last identity is the cheapest available check that a dataset's extrinsics
mean what we think they mean, so :func:`CameraParams.center` is validated against
an independently recorded eye position in the tests.  Getting this wrong silently
produces plausible-looking Plucker maps that encode the wrong geometry, which is
exactly the kind of bug that survives to the results table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence]


def _as_tensor(x: ArrayLike, dtype=torch.float32, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device) if device is not None else x.to(dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


@dataclass
class CameraParams:
    """Intrinsics + world-to-camera extrinsics for a batch of cameras.

    Shapes are ``(..., 3, 3)`` for ``intrinsic`` and ``(..., 3, 4)`` for
    ``extrinsic``; the leading dimensions are free so a whole clip of per-frame
    cameras can be carried in one object.
    """

    intrinsic: torch.Tensor          # (..., 3, 3)
    extrinsic: torch.Tensor          # (..., 3, 4)  world -> camera [R|t]
    height: int
    width: int

    def __post_init__(self):
        self.intrinsic = _as_tensor(self.intrinsic)
        self.extrinsic = _as_tensor(self.extrinsic)
        if self.intrinsic.shape[-2:] != (3, 3):
            raise ValueError(f"intrinsic must be (...,3,3), got {tuple(self.intrinsic.shape)}")
        if self.extrinsic.shape[-2:] != (3, 4):
            raise ValueError(f"extrinsic must be (...,3,4), got {tuple(self.extrinsic.shape)}")
        if self.intrinsic.shape[:-2] != self.extrinsic.shape[:-2]:
            raise ValueError("intrinsic and extrinsic must share leading dimensions: "
                             f"{tuple(self.intrinsic.shape[:-2])} vs {tuple(self.extrinsic.shape[:-2])}")

    # ------------------------------------------------------------------ pieces
    @property
    def R(self) -> torch.Tensor:
        return self.extrinsic[..., :3, :3]

    @property
    def t(self) -> torch.Tensor:
        return self.extrinsic[..., :3, 3]

    @property
    def batch_shape(self):
        return tuple(self.extrinsic.shape[:-2])

    def center(self) -> torch.Tensor:
        """Camera centre in world coordinates, ``c = -R^T t``.  Shape ``(..., 3)``."""
        return -torch.einsum("...ji,...j->...i", self.R, self.t)

    def cam_to_world_rot(self) -> torch.Tensor:
        """``R^T``: rotates a direction from camera frame into world frame."""
        return self.R.transpose(-1, -2)

    def forward_axis(self) -> torch.Tensor:
        """Optical axis (camera +z) expressed in world coordinates.  ``(..., 3)``."""
        return self.cam_to_world_rot()[..., :, 2]

    def fov(self) -> torch.Tensor:
        """Horizontal and vertical field of view in radians.  ``(..., 2)``."""
        fx = self.intrinsic[..., 0, 0]
        fy = self.intrinsic[..., 1, 1]
        return torch.stack([2 * torch.atan(0.5 * self.width / fx),
                            2 * torch.atan(0.5 * self.height / fy)], dim=-1)

    # ------------------------------------------------------------- conversions
    def to(self, *args, **kwargs) -> "CameraParams":
        return CameraParams(self.intrinsic.to(*args, **kwargs),
                            self.extrinsic.to(*args, **kwargs), self.height, self.width)

    def reshape_batch(self, *shape) -> "CameraParams":
        return CameraParams(self.intrinsic.reshape(*shape, 3, 3),
                            self.extrinsic.reshape(*shape, 3, 4), self.height, self.width)

    def scaled(self, height: int, width: int) -> "CameraParams":
        """Intrinsics rescaled to a different image resolution.

        Resizing an image rescales focal length and principal point by the same
        factor; forgetting this is the standard way Plucker maps end up subtly
        wrong after a resize.
        """
        sx, sy = width / self.width, height / self.height
        K = self.intrinsic.clone()
        K[..., 0, 0] *= sx
        K[..., 0, 2] *= sx
        K[..., 1, 1] *= sy
        K[..., 1, 2] *= sy
        return CameraParams(K, self.extrinsic, height, width)

    # -------------------------------------------------------------- projection
    def project(self, points_world: torch.Tensor) -> torch.Tensor:
        """World points ``(..., N, 3)`` -> pixel coordinates ``(..., N, 2)``.

        Points behind the camera come back as NaN rather than silently wrapping
        around through the negative-depth division.
        """
        p_cam = torch.einsum("...ij,...nj->...ni", self.R, points_world) + self.t[..., None, :]
        z = p_cam[..., 2:3]
        uv = torch.einsum("...ij,...nj->...ni", self.intrinsic, p_cam)
        uv = uv[..., :2] / uv[..., 2:3]
        return torch.where(z > 1e-8, uv, torch.full_like(uv, float("nan")))

    @classmethod
    def from_dict(cls, d: dict, height: Optional[int] = None,
                  width: Optional[int] = None) -> "CameraParams":
        """Build from the ``cam_params`` dict stored alongside our rendered clips."""
        h = height or d.get("height") or d.get("res")
        w = width or d.get("width") or d.get("res")
        if h is None or w is None:
            raise ValueError("height/width must be given or present in the dict")
        return cls(d["intrinsic_cv"], d["extrinsic_cv"], int(h), int(w))


def look_at_extrinsic(eye: ArrayLike, target: ArrayLike,
                      up: ArrayLike = (0.0, 0.0, 1.0)) -> torch.Tensor:
    """Build an OpenCV world-to-camera ``[R|t]`` from an eye/target pair.

    Used by the tests and by synthetic camera sweeps; real data carries its own
    extrinsics and should never route through this.
    """
    eye = _as_tensor(eye).reshape(3)
    target = _as_tensor(target).reshape(3)
    up = _as_tensor(up).reshape(3)

    z = target - eye
    z = z / z.norm().clamp_min(1e-12)                       # forward (camera +z)
    x = torch.cross(z, up, dim=-1)
    if x.norm() < 1e-6:                                     # up parallel to view dir
        alt = torch.tensor([1.0, 0.0, 0.0], dtype=z.dtype)
        x = torch.cross(z, alt, dim=-1)
    x = x / x.norm().clamp_min(1e-12)                       # right (camera +x)
    y = torch.cross(z, x, dim=-1)                           # down  (camera +y)

    R = torch.stack([x, y, z], dim=0)                       # rows are camera axes
    t = -R @ eye
    return torch.cat([R, t[:, None]], dim=1)


def pinhole_intrinsic(fov_y: float, height: int, width: int) -> torch.Tensor:
    """Square-pixel pinhole ``K`` from a vertical field of view in radians."""
    f = 0.5 * height / np.tan(0.5 * float(fov_y))
    return torch.tensor([[f, 0.0, width / 2.0],
                         [0.0, f, height / 2.0],
                         [0.0, 0.0, 1.0]], dtype=torch.float32)
