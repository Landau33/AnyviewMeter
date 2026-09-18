"""Plucker ray maps: the pose representation this project conditions on.

For the ray through pixel ``(u, v)`` we build

    d = normalize(R^T K^-1 [u, v, 1]^T)      direction, world frame
    c = -R^T t                               camera centre, world frame
    m = c x d                                moment
    r = (d, m)                               6 channels, image-aligned

Why Plucker and not raw extrinsics.  A 6-DoF pose vector is one token's worth of
information for the whole frame, and a large VLM is free to ignore it.  A Plucker
map is *per pixel*: it says which world ray each patch is looking along, so it is
spatially aligned with the patch tokens and can be added to or attended from them.
This is the representation used by CameraCtrl / Cosmos / VD3D and it is what makes
"which viewpoint is this" a local, learnable signal rather than a global side note.

Two properties worth keeping in mind, both asserted in ``tests/test_plucker.py``:

  * ``m`` is invariant to where along the ray you put ``c``: for any point ``p`` on
    the ray, ``p x d == c x d``.  So the map encodes the ray as a geometric object,
    not the particular camera centre.
  * ``d . m == 0`` always, since ``m`` is a cross product with ``d``.  A violated
    orthogonality check means the direction and moment fell out of sync, usually a
    normalisation applied to one but not the other.

TOKEN ALIGNMENT.  Qwen3-VL uses patch 16 with a 2x2 spatial merge, so one visual
token covers 32x32 pixels.  :func:`plucker_map` therefore takes an explicit output
grid and samples the ray at each *token's* centre, rather than building a
full-resolution map and pooling it.  Pooling would average directions across a
token footprint, which is wrong near the image edge where directions fan out.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .camera import CameraParams


def pixel_grid(height: int, width: int, out_h: int, out_w: int,
               device=None, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pixel coordinates of the centre of each cell in an ``out_h x out_w`` grid.

    With ``out_h == height`` this is the usual ``(j + 0.5, i + 0.5)`` pixel centre.
    With a coarser grid it is the centre of the pixel block that a visual token
    covers, which is what we want for token-aligned conditioning.
    """
    sy, sx = height / out_h, width / out_w
    ys = (torch.arange(out_h, device=device, dtype=dtype) + 0.5) * sy
    xs = (torch.arange(out_w, device=device, dtype=dtype) + 0.5) * sx
    v, u = torch.meshgrid(ys, xs, indexing="ij")
    return u, v


def ray_directions(cam: CameraParams, out_h: int, out_w: int,
                   normalize: bool = True) -> torch.Tensor:
    """World-frame ray directions, ``(..., 3, out_h, out_w)``."""
    dev, dt = cam.intrinsic.device, cam.intrinsic.dtype
    u, v = pixel_grid(cam.height, cam.width, out_h, out_w, device=dev, dtype=dt)
    ones = torch.ones_like(u)
    px = torch.stack([u, v, ones], dim=0).reshape(3, -1)              # (3, HW)

    K_inv = torch.linalg.inv(cam.intrinsic)                           # (..., 3, 3)
    d_cam = K_inv @ px                                                # (..., 3, HW)
    d_world = cam.cam_to_world_rot() @ d_cam                          # (..., 3, HW)
    if normalize:
        d_world = d_world / d_world.norm(dim=-2, keepdim=True).clamp_min(1e-12)
    return d_world.reshape(*cam.batch_shape, 3, out_h, out_w)


def plucker_map(cam: CameraParams, out_h: int, out_w: int,
                normalize: bool = True) -> torch.Tensor:
    """Plucker ray map ``(..., 6, out_h, out_w)`` ordered as ``(d, m)``.

    Args:
        cam: camera parameters; leading batch dims are preserved.
        out_h, out_w: the token grid to align to (e.g. 8x8 for a 256px frame
            through a patch-16 / merge-2 vision tower).
        normalize: unit-length directions.  Keep this on -- with unnormalised
            directions the magnitude of ``d`` (and hence of ``m``) scales with
            focal length, so the conditioning signal would change if the same
            camera were re-rendered at a different resolution.
    """
    d = ray_directions(cam, out_h, out_w, normalize=normalize)        # (..., 3, H, W)
    c = cam.center()                                                  # (..., 3)
    c = c[..., :, None, None].expand_as(d)
    m = torch.cross(c, d, dim=-3)
    return torch.cat([d, m], dim=-3)


def plucker_map_from_arrays(intrinsic, extrinsic, height: int, width: int,
                            out_h: int, out_w: int) -> torch.Tensor:
    """Convenience wrapper for callers holding raw arrays."""
    return plucker_map(CameraParams(intrinsic, extrinsic, height, width), out_h, out_w)


def pose_vector(cam: CameraParams, workspace_centre: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
    """Compact global pose descriptor for the ``<cam>`` register token, ``(..., 16)``.

    Plucker maps carry per-patch geometry but deliberately no global summary, and
    the token that routes "which viewpoint is this" wants one.  Layout:

        0:3    camera centre (world)
        3:6    optical axis (world)
        6:9    camera "up" axis, i.e. -y_cam (world)
        9:11   horizontal / vertical FOV (radians)
        11:14  unit vector from camera towards the workspace centre
        14     distance to the workspace centre
        15     cosine between the optical axis and that direction (framing error)

    The last three entries need a workspace anchor; without one they are zeroed
    and the descriptor is purely egocentric.  They matter because phase A found
    viewpoint quality is largely "is the manipulated object big and unoccluded",
    which is a statement about the camera *relative to the workspace*, not about
    the camera alone.
    """
    c = cam.center()
    fwd = cam.forward_axis()
    up = -cam.cam_to_world_rot()[..., :, 1]                 # camera +y is down
    fov = cam.fov()

    if workspace_centre is None:
        rel = torch.zeros_like(c)
        dist = torch.zeros_like(c[..., :1])
        cos = torch.zeros_like(c[..., :1])
    else:
        w = torch.as_tensor(workspace_centre, dtype=c.dtype, device=c.device)
        w = w.expand_as(c)
        delta = w - c
        dist = delta.norm(dim=-1, keepdim=True)
        rel = delta / dist.clamp_min(1e-12)
        cos = (rel * fwd).sum(-1, keepdim=True)

    return torch.cat([c, fwd, up, fov, rel, dist, cos], dim=-1)


POSE_VECTOR_DIM = 16


def check_plucker(r: torch.Tensor, atol: float = 1e-4) -> dict:
    """Geometric self-checks on a Plucker map.  Returns the measured residuals.

    ``ortho`` should be ~0 (``d . m == 0`` by construction) and ``unit`` should be
    ~1 for normalised directions.  Both are cheap enough to assert in tests and in
    the data pipeline's first batch.
    """
    d, m = r[..., :3, :, :], r[..., 3:, :, :]
    ortho = (d * m).sum(-3).abs().max()
    unit = d.norm(dim=-3)
    return {
        "ortho_max": float(ortho),
        "dir_norm_min": float(unit.min()),
        "dir_norm_max": float(unit.max()),
        "ok": bool(ortho < atol and (unit - 1).abs().max() < 1e-3),
    }


# ===========================================================================
# Moment scale: the NGI feature
# ===========================================================================
# WHY THIS EXISTS.  ``m = c x d`` with ``||d|| == 1`` has a direct reading:
#
#     ||m||  =  perpendicular distance from the WORLD ORIGIN to the ray line
#
# so the moment's magnitude is not a property of the ray's orientation at all,
# it is a property of how far the ray passes from wherever we happened to put
# the origin.  ``plucker_map`` normalises ``d`` and then leaves ``m`` raw, which
# means the conditioning signal carries an unnormalised distance in three of its
# six channels while the other three are unit vectors.
#
# For a video-diffusion model trained across SfM / SLAM / metric datasets that is
# fatal, and it is what SCoPE's Normalize-Gate-Inject is built to fix.  Our
# situation is milder but not harmless: every camera in ``pickcube_avm`` sits at
# roughly the same radius, so ``||m||`` barely moves and the raw channels look
# fine -- until the training cone is widened, at which point the camera centres
# spread and ``||m||`` spreads with them.  Since widening the cone is exactly the
# change that made pose conditioning work (see the P1 notes), the distribution of
# this quantity is worth measuring rather than assuming; ``moment_stats`` and
# ``scripts/moment_scale_report.py`` do that.
#
# The decomposition splits the two roles instead of mixing them:
#
#     m_hat = m / max(||m||, eps)      unit -- WHERE the ray is, scale-free
#     s     = log max(||m||, eps)      scalar -- HOW FAR, in log units
#
# ``m_hat`` is invariant to a global rescaling of the scene and ``s`` absorbs it
# additively, which is what makes a scale perturbation a single additive nudge on
# one channel (see :func:`jitter_scale`) rather than a nonlinear mess.

#: Channel count of the NGI feature ``(d, m_hat, s)``.
NGI_FEATURE_DIM = 7


def normalize_moment(r: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``(..., 6, H, W)`` Plucker map -> ``(..., 7, H, W)`` as ``(d, m_hat, s)``.

    ``d`` passes through untouched (it is already unit), ``m`` is split into a unit
    direction and a log magnitude.  The result is the feature every NGI-style
    injector consumes; :func:`denormalize_moment` inverts it exactly.
    """
    if r.shape[-3] != 6:
        raise ValueError(f"expected 6 Plucker channels, got {r.shape[-3]}")
    d, m = r[..., :3, :, :], r[..., 3:, :, :]
    norm = m.norm(dim=-3, keepdim=True).clamp_min(eps)
    return torch.cat([d, m / norm, norm.log()], dim=-3)


def denormalize_moment(f: torch.Tensor) -> torch.Tensor:
    """``(..., 7, H, W)`` -> ``(..., 6, H, W)``.  Inverse of :func:`normalize_moment`.

    Kept because it is the only way to state the round-trip as a test, and because a
    scale-perturbed feature has to be turned back into a Plucker map to be checked
    against the geometry it now claims.
    """
    if f.shape[-3] != NGI_FEATURE_DIM:
        raise ValueError(f"expected {NGI_FEATURE_DIM} NGI channels, got {f.shape[-3]}")
    d, m_hat, s = f[..., :3, :, :], f[..., 3:6, :, :], f[..., 6:, :, :]
    return torch.cat([d, m_hat * s.exp()], dim=-3)


def jitter_scale(f: torch.Tensor, delta) -> torch.Tensor:
    """Add ``delta`` to the log-magnitude channel of an NGI feature.

    ONE DELTA PER CAMERA, not per token: this represents a rescaling of the whole
    scene, and applying an independent nudge per patch would describe a set of rays
    that no camera can produce.  ``delta`` may be a scalar or shaped to broadcast
    against ``f[..., 6:, :, :]``.

    READ THE WARNING IN ``jitter_scale``'s CALLERS.  SCoPE uses this as an honest
    augmentation because its pose sources are up-to-scale: a SLAM trajectory scaled
    by k describes the same video, so perturbing s leaves the pose-image pair
    consistent.  OUR poses are metric and come from the renderer, so a jittered s
    describes a camera that would NOT have produced these pixels.  Here it is a
    regulariser, not an augmentation -- it says "do not read absolute distance off
    this channel", which is a defensible thing to want given that a model trained
    inside a narrow camera cone extrapolates worse the longer it trains.  It is off
    by default and any run using it has to clear the same causal check as any other
    arm.
    """
    d_mh, s = f[..., :6, :, :], f[..., 6:, :, :]
    return torch.cat([d_mh, s + delta], dim=-3)


def reciprocal_product(r_i: torch.Tensor, r_j: torch.Tensor) -> torch.Tensor:
    """Plucker reciprocal product ``d_i . m_j + d_j . m_i`` for ``(..., 6)`` rays.

    Zero exactly when the two lines are coplanar -- i.e. they meet, or are parallel.
    Two rays sharing a camera centre ALWAYS meet, so this vanishes identically inside
    a single view; ``tests/test_ngi.py`` pins that, because it is the reason a
    ray-space attention bias is only non-trivial once several cameras share one
    sequence.
    """
    d_i, m_i = r_i[..., :3], r_i[..., 3:6]
    d_j, m_j = r_j[..., :3], r_j[..., 3:6]
    return (d_i * m_j).sum(-1) + (d_j * m_i).sum(-1)


def moment_stats(r: torch.Tensor, eps: float = 1e-6) -> dict:
    """``||m||`` and ``s = log||m||`` summary for one or many Plucker maps.

    The quantity to compare across camera groups: if a test group's ``s`` range does
    not overlap the training range, the pose channels are being asked to extrapolate
    in magnitude, not just in orientation, and a conditioning failure out there says
    nothing about the geometry itself.
    """
    m = r[..., 3:6, :, :]
    norm = m.norm(dim=-3).clamp_min(eps).flatten()
    s = norm.log()
    return {
        "m_min": float(norm.min()), "m_max": float(norm.max()),
        "m_mean": float(norm.mean()), "m_median": float(norm.median()),
        "s_min": float(s.min()), "s_max": float(s.max()),
        "s_mean": float(s.mean()), "s_std": float(s.std(unbiased=False)),
        "n": int(norm.numel()),
    }


# ------------------------------------------------- differentiable camera synthesis
def look_at_camera(az, el, radius, fov, centre, height: int, width: int) -> CameraParams:
    """Build CameraParams from (az, el, radius, fov) with gradients intact.

    Everything downstream of CameraParams -- ray_directions, plucker_map, pose_vector --
    is already pure torch, so making THIS step differentiable is what lets a pose head be
    trained by the task loss instead of by pose supervision.

    OpenCV convention, matching the stored ``extrinsic_cv``: camera x right, y DOWN,
    z forward, extrinsic is world -> camera.  Getting the handedness wrong here produces
    a Plucker map that is plausible, self-consistent and mirrored, which no shape check
    would catch -- ``tests`` compares this against the renderer's own extrinsic instead.

    Args:
        az, el, radius, fov: ``(B,)`` tensors.  az/el in radians, fov vertical in radians.
        centre: ``(3,)`` world point the camera looks at.
    """
    ce, se, ca, sa = torch.cos(el), torch.sin(el), torch.cos(az), torch.sin(az)
    offset = torch.stack([ce * ca, ce * sa, se], dim=-1) * radius[..., None]
    eye = centre[None, :] + offset                                   # (B, 3)

    fwd = centre[None, :] - eye
    fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    up_w = torch.tensor([0.0, 0.0, 1.0], device=eye.device, dtype=eye.dtype)
    right = torch.cross(fwd, up_w.expand_as(fwd), dim=-1)
    right = right / right.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    down = torch.cross(fwd, right, dim=-1)                           # y points DOWN

    R = torch.stack([right, down, fwd], dim=-2)                      # (B, 3, 3) world->cam
    t = -torch.einsum("bij,bj->bi", R, eye)
    extrinsic = torch.cat([R, t[..., None]], dim=-1)                 # (B, 3, 4)

    f = 0.5 * height / torch.tan(0.5 * fov).clamp_min(1e-6)
    z = torch.zeros_like(f)
    o = torch.ones_like(f)
    cx = torch.full_like(f, width / 2.0)
    cy = torch.full_like(f, height / 2.0)
    K = torch.stack([torch.stack([f, z, cx], -1),
                     torch.stack([z, f, cy], -1),
                     torch.stack([z, z, o], -1)], dim=-2)            # (B, 3, 3)
    return CameraParams(K, extrinsic, height, width)
