"""Correctness tests for the Plucker geometry.

These are the tests that matter most in the project: every downstream result is
conditioned on these maps, and a wrong-but-plausible Plucker map produces
wrong-but-plausible numbers rather than a crash.

Run:  python -m pytest AnyviewMeter/tests/test_plucker.py -v
  or: python AnyviewMeter/tests/test_plucker.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anyviewmeter.geometry.camera import (CameraParams, look_at_extrinsic,  # noqa: E402
                                          pinhole_intrinsic)
from anyviewmeter.geometry.plucker import (POSE_VECTOR_DIM, check_plucker,  # noqa: E402
                                           plucker_map, pose_vector, ray_directions)

# A camera taken verbatim from outputs/clips/StackCube/success/traj_0__frontal_good.npz,
# together with the eye position ManiSkill independently recorded for it.  This is
# the ground truth that pins our extrinsic convention to the data we actually have.
REAL_K = [[309.0193, 0.0, 128.0], [0.0, 309.0193, 128.0], [0.0, 0.0, 1.0]]
REAL_EXT = [[0.0, 1.0, 0.0, 0.02],
            [0.669, 0.0, -0.7433, 0.023],
            [-0.7433, 0.0, -0.669, 0.5723]]
REAL_EYE = [0.41, -0.02, 0.4]
REAL_TARGET = [0.01, -0.02, 0.04]
RES = 256


def _real_cam():
    return CameraParams(REAL_K, REAL_EXT, RES, RES)


def test_center_matches_recorded_eye():
    """c = -R^T t must reproduce the eye position the renderer recorded."""
    c = _real_cam().center().numpy()
    assert np.allclose(c, REAL_EYE, atol=2e-3), f"centre {c} != eye {REAL_EYE}"


def test_optical_axis_points_at_target():
    """The camera was built with look_at, so its +z must point at the target."""
    cam = _real_cam()
    axis = cam.forward_axis().numpy()
    want = np.asarray(REAL_TARGET) - np.asarray(REAL_EYE)
    want = want / np.linalg.norm(want)
    assert np.allclose(axis, want, atol=2e-3), f"axis {axis} != {want}"


def test_principal_ray_equals_optical_axis():
    """The ray through the principal point is the optical axis."""
    cam = _real_cam()
    d = ray_directions(cam, RES, RES)                  # (3, H, W)
    centre_ray = d[:, RES // 2, RES // 2].numpy()
    axis = cam.forward_axis().numpy()
    assert np.allclose(centre_ray, axis, atol=5e-3)


def test_moment_is_orthogonal_and_directions_unit():
    r = plucker_map(_real_cam(), 8, 8)
    assert r.shape == (6, 8, 8)
    stats = check_plucker(r)
    assert stats["ok"], stats


def test_moment_invariant_to_point_on_ray():
    """m = c x d must not depend on WHERE on the ray the reference point sits.

    This is the property that makes the map a description of the ray rather than
    of the camera centre, and it is the one that breaks first if directions get
    normalised inconsistently.
    """
    cam = _real_cam()
    d = ray_directions(cam, 8, 8)                       # (3,8,8)
    c = cam.center()[:, None, None].expand_as(d)
    m0 = torch.cross(c, d, dim=0)
    for s in (0.5, 2.0, 17.0):
        p = c + s * d                                   # another point on the ray
        m = torch.cross(p, d, dim=0)
        assert torch.allclose(m, m0, atol=1e-4), f"moment changed at s={s}"


def test_rays_reproject_to_their_own_pixels():
    """Walk along each ray and project back; we must land on the pixel we started from."""
    cam = _real_cam()
    out = 16
    d = ray_directions(cam, out, out)                   # (3,out,out)
    c = cam.center()
    pts = (c[:, None, None] + 3.0 * d).reshape(3, -1).T  # (N,3) 3 m along each ray
    uv = cam.project(pts).numpy()

    # ray_directions returns (3, out, out) in row-major (v, u) order, so flattening
    # gives u cycling fastest and v repeating -- match that when building the target.
    sy = sx = RES / out
    want_u = (np.arange(out) + 0.5) * sx
    want_v = (np.arange(out) + 0.5) * sy
    want = np.stack([np.tile(want_u, out), np.repeat(want_v, out)], axis=1)
    assert np.allclose(uv, want, atol=1e-2), f"max err {np.abs(uv - want).max():.4f}"


def test_intrinsic_rescale_keeps_geometry():
    """Rescaling K to a new resolution must leave the ray field unchanged."""
    cam = _real_cam()
    small = cam.scaled(128, 128)
    a = plucker_map(cam, 8, 8)
    b = plucker_map(small, 8, 8)
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max().item()


def test_distinct_viewpoints_give_distinct_maps():
    """Two different cameras on the same scene must not produce the same map."""
    K = pinhole_intrinsic(np.deg2rad(45.0), RES, RES)
    a = CameraParams(K, look_at_extrinsic([0.4, 0.0, 0.4], [0, 0, 0]), RES, RES)
    b = CameraParams(K, look_at_extrinsic([0.0, 0.55, 0.05], [0, 0, 0]), RES, RES)
    ra, rb = plucker_map(a, 8, 8), plucker_map(b, 8, 8)
    assert (ra - rb).abs().max() > 0.5, "different viewpoints produced near-identical maps"


def test_batched_matches_looped():
    """Leading batch dims must give the same answer as looping one camera at a time."""
    K = pinhole_intrinsic(np.deg2rad(45.0), RES, RES)
    eyes = [[0.4, 0.0, 0.4], [0.0, 0.55, 0.05], [0.0, 0.02, 0.66]]
    exts = torch.stack([look_at_extrinsic(e, [0, 0, 0]) for e in eyes])
    Ks = K[None].expand(3, 3, 3)
    batched = plucker_map(CameraParams(Ks, exts, RES, RES), 8, 8)
    assert batched.shape == (3, 6, 8, 8)
    for i, e in enumerate(eyes):
        one = plucker_map(CameraParams(K, look_at_extrinsic(e, [0, 0, 0]), RES, RES), 8, 8)
        assert torch.allclose(batched[i], one, atol=1e-5)


def test_pose_vector_shape_and_framing():
    """The <cam> descriptor should report near-perfect framing for a look_at camera."""
    cam = _real_cam()
    v = pose_vector(cam, workspace_centre=torch.tensor(REAL_TARGET))
    assert v.shape == (POSE_VECTOR_DIM,)
    assert np.allclose(v[:3].numpy(), REAL_EYE, atol=2e-3)      # centre
    assert abs(float(v[14]) - np.linalg.norm(np.asarray(REAL_TARGET) - np.asarray(REAL_EYE))) < 2e-3
    assert float(v[15]) > 0.999, "camera aimed at the workspace should have cos ~ 1"


def test_pose_vector_detects_misframing():
    """A camera pointed away from the workspace must show it in the framing cosine."""
    K = pinhole_intrinsic(np.deg2rad(45.0), RES, RES)
    away = CameraParams(K, look_at_extrinsic([0.4, 0, 0.4], [2.0, 2.0, 0.4]), RES, RES)
    v = pose_vector(away, workspace_centre=torch.tensor([0.0, 0.0, 0.0]))
    assert float(v[15]) < 0.9, "misframed camera should not report cos ~ 1"


def _run_all():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"[ok ] {name}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[ERR ] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
