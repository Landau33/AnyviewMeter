"""Tests for the four pieces borrowed from SCoPE (arXiv 2606.27345).

Each test below exists because the corresponding claim is one we would otherwise
be taking on faith from a paper written about a different model on a different
task:

  moment scale     that (d, m_hat, s) is a lossless re-encoding, and that s really
                   is the log distance from the world origin to the ray
  scale gate       that an input-conditioned gate is constant across tokens at
                   init (so it does not disturb the geometric initialisation) and
                   that every one of its parameters still receives gradient
  geometric init   that term (D) of the attention score really is proportional to
                   the Plucker reciprocal product at initialisation, RMSNorm and
                   gate included -- the argument for that is three paragraphs of
                   prose in ngi.py and it is cheaper to measure it
  the degeneracy   that the reciprocal product VANISHES between two rays of the
                   same camera.  This is the single most important number in this
                   file: it says term (D) contributes exactly nothing to a
                   single-view sequence, which is what our clips currently are.

Run:  python AnyviewMeter/tests/test_ngi.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anyviewmeter.geometry.camera import (CameraParams, look_at_extrinsic,  # noqa: E402
                                          pinhole_intrinsic)
from anyviewmeter.geometry.plucker import (NGI_FEATURE_DIM, denormalize_moment,  # noqa: E402
                                           jitter_scale, moment_stats,
                                           normalize_moment, plucker_map,
                                           reciprocal_product)
from anyviewmeter.models.injection import RayPE, ScaleGate, build_injector  # noqa: E402

RES, GRID = 256, 8


def _cam(eye, target=(0.0, 0.0, 0.05), fov=0.9):
    K = pinhole_intrinsic(fov_y=fov, height=RES, width=RES)
    return CameraParams(K, look_at_extrinsic(eye, target), RES, RES)


def _map(eye, **kw):
    return plucker_map(_cam(eye, **kw), GRID, GRID).reshape(1, 6, GRID, GRID)


# ------------------------------------------------------------------ moment scale
def test_ngi_roundtrip_is_lossless():
    r = _map([0.5, 0.0, 0.4])
    back = denormalize_moment(normalize_moment(r))
    err = (back - r).abs().max()
    assert err < 1e-5, f"round trip lost {err}"


def test_ngi_feature_layout():
    f = normalize_moment(_map([0.5, 0.0, 0.4]))
    assert f.shape[1] == NGI_FEATURE_DIM
    d, m_hat = f[:, :3], f[:, 3:6]
    assert (d.norm(dim=1) - 1).abs().max() < 1e-5, "d should stay unit"
    assert (m_hat.norm(dim=1) - 1).abs().max() < 1e-5, "m_hat should be unit"
    assert (d * m_hat).sum(1).abs().max() < 1e-5, "normalising m must preserve d.m == 0"


def test_s_is_log_distance_from_origin_to_the_ray():
    """||m|| with unit d is the perpendicular distance from the origin to the line.

    This is the reading that makes the channel interpretable at all: it is not a
    property of where the camera is looking, it is a property of where we put the
    world origin relative to the ray.
    """
    cam = _cam([0.6, 0.1, 0.45])
    r = plucker_map(cam, GRID, GRID)
    d = r[:3].reshape(3, -1).T                       # (N, 3)
    c = cam.center()
    # distance from the origin to the line through c with direction d
    proj = (c[None, :] * d).sum(-1, keepdim=True) * d
    want = (c[None, :] - proj).norm(dim=-1)
    got = normalize_moment(r.reshape(1, 6, GRID, GRID))[0, 6].reshape(-1).exp()
    assert torch.allclose(got, want, atol=1e-4), \
        f"max err {(got - want).abs().max():.2e}"


def test_scene_rescaling_moves_only_s():
    """Scale the whole scene by k: d and m_hat are unchanged, s shifts by log k.

    The property the decomposition exists for.  It is also what makes
    :func:`jitter_scale` a single additive nudge instead of a nonlinear edit.
    """
    k = 2.5
    a = normalize_moment(_map([0.5, 0.1, 0.4], target=(0.0, 0.0, 0.05)))
    b = normalize_moment(_map([0.5 * k, 0.1 * k, 0.4 * k], target=(0.0, 0.0, 0.05 * k)))
    assert (a[:, :6] - b[:, :6]).abs().max() < 1e-4, "orientation channels moved"
    ds = (b[:, 6] - a[:, 6])
    assert (ds - np.log(k)).abs().max() < 1e-4, f"s shift {ds.mean()} != log {k}"


def test_jitter_scale_touches_only_s():
    f = normalize_moment(_map([0.5, 0.0, 0.4]))
    g = jitter_scale(f, 0.7)
    assert (g[:, :6] - f[:, :6]).abs().max() == 0.0
    assert torch.allclose(g[:, 6], f[:, 6] + 0.7)


def test_moment_stats_reports_a_usable_range():
    st = moment_stats(_map([0.5, 0.0, 0.4]))
    assert st["m_min"] > 0 and st["m_max"] >= st["m_min"]
    assert abs(st["s_mean"] - np.log(st["m_median"])) < 1.0
    assert st["n"] == GRID * GRID


# ------------------------------------------------------------- the degeneracy
def test_reciprocal_product_vanishes_within_one_camera():
    """THE constraint on tier D in our setting.

    Any two rays from the same centre intersect (at the centre), so their
    reciprocal product is identically zero.  Our clips are single-camera
    sequences, so the pure-geometry term of a ray-space attention bias is a
    constant 0 across the entire sequence: only the content<->geometry terms can
    do any work.  Making (D) non-trivial needs several cameras packed into one
    attention sequence, which the data supports (25 cameras per trajectory) but
    the collator does not do yet.
    """
    r = _map([0.55, 0.05, 0.42])[0].reshape(6, -1).T          # (N, 6)
    a, b = r[:, None, :], r[None, :, :]
    rp = reciprocal_product(a.expand(-1, r.shape[0], -1), b.expand(r.shape[0], -1, -1))
    assert rp.abs().max() < 1e-4, \
        f"same-camera reciprocal product should vanish, got {rp.abs().max():.3e}"


def test_reciprocal_product_is_nonzero_across_cameras():
    """...and does carry signal once two viewpoints share a sequence."""
    r1 = _map([0.55, 0.05, 0.42])[0].reshape(6, -1).T
    r2 = _map([-0.1, 0.6, 0.35])[0].reshape(6, -1).T
    rp = reciprocal_product(r1[:, None, :].expand(-1, r2.shape[0], -1),
                            r2[None, :, :].expand(r1.shape[0], -1, -1))
    assert rp.abs().max() > 1e-2, "cross-camera geometry term is dead"
    # and it is near zero for the pairs that do meet -- the rays through the
    # shared workspace point.  A term that were never zero would not be measuring
    # coplanarity at all.
    assert rp.abs().min() < 1e-2 * rp.abs().max()


# ----------------------------------------------------------------- scale gate
def test_scale_gate_is_uniform_and_constant_at_init():
    """Both properties matter, and for different reasons.

    Constant across TOKENS keeps the gate from reweighting one ray against another;
    uniform across CHANNELS keeps ``<g*x, g*y>`` proportional to ``<x, y>``, which
    is the whole geometric-initialisation claim.  The first version of ScaleGate
    had the second property wrong and term (D) came out 40% off.
    """
    g = ScaleGate(16)
    out = g(torch.linspace(-3, 3, 32).reshape(32, 1))
    token_span = float(out.max(0).values.sub(out.min(0).values).max())
    channel_span = float(out.max(1).values.sub(out.min(1).values).max())
    assert token_span < 1e-6, f"gate varies by {token_span} across s at init"
    assert channel_span < 1e-6, f"gate varies by {channel_span} across channels at init"
    assert 0.7 < float(out.mean()) < 0.99, "gate should start open, not shut"


def test_scale_gate_output_layer_is_alive_and_the_rest_revives():
    """The deadlock check, in the form the ControlNet arrangement actually allows.

    ``lin2.weight`` is zero at init, so everything upstream of it is momentarily
    dead -- that is fine and expected.  What must NOT be true is that ``lin2``
    itself is dead too, because then nothing could ever move.  One step is enough
    to bring the whole module back.
    """
    g = ScaleGate(16)
    opt = torch.optim.SGD(g.parameters(), lr=0.1)
    g(torch.randn(8, 1)).sum().backward()
    assert float(g.lin2.weight.grad.abs().max()) > 0, "the output layer is dead at init"
    opt.step()
    g.zero_grad()
    g(torch.randn(8, 1)).sum().backward()
    dead = [n for n, p in g.named_parameters()
            if p.grad is None or float(p.grad.abs().max()) == 0.0]
    assert not dead, f"still no gradient reaching {dead} after one step"


def test_scale_gate_becomes_conditional_once_trained():
    """It is constant at init by construction; it must not be STUCK constant."""
    g = ScaleGate(8)
    opt = torch.optim.SGD(g.parameters(), lr=0.5)
    s = torch.tensor([[-2.0], [2.0]])
    target = torch.tensor([[0.05], [0.95]]).expand(2, 8)
    for _ in range(300):
        opt.zero_grad()
        ((g(s) - target) ** 2).mean().backward()
        opt.step()
    out = g(s)
    assert float(out[1].mean() - out[0].mean()) > 0.3, \
        f"gate could not learn to depend on s: {out.mean(-1).tolist()}"


# ------------------------------------------------------------- geometric init
def _term_d(pe, feat):
    """Term (D) of the score matrix for one RayPE and a set of NGI features."""
    q, k = pe(feat)
    return q @ k.transpose(-1, -2)


def _pairwise(rays):
    n = rays.shape[0]
    return reciprocal_product(rays[:, None, :].expand(-1, n, -1),
                              rays[None, :, :].expand(n, -1, -1))


def test_geometric_init_reproduces_the_reciprocal_product():
    """Term (D) at init must be the reciprocal product up to POSITIVE per-token scales.

    RMSNorm divides each token by its own norm, so exact proportionality with one
    constant is not the claim under the default ``moment_mode="raw"`` (where token
    norms differ).  What must hold is the part that carries meaning: the same zero
    set and the same sign everywhere -- i.e. term (D) still measures coplanarity.
    Dividing out the two norms recovers exact proportionality, and that is checked
    too, because it is what pins the init to the reciprocal product rather than to
    some other bilinear form with the same sign pattern.
    """
    pe = RayPE(head_dim=64, alpha_init=1.0)
    r = torch.cat([_map([0.55, 0.05, 0.42]), _map([-0.1, 0.6, 0.35])], dim=0)
    feat = RayPE.features(r).reshape(1, -1, NGI_FEATURE_DIM)
    got = _term_d(pe, feat)[0]

    f = feat[0]
    rays = torch.cat([f[:, :3], f[:, 3:6] * f[:, 6:].exp()], dim=-1)     # (d, m)
    want = _pairwise(rays)
    live = want.abs() > 1e-4
    assert live.any(), "no non-degenerate pairs to compare"
    assert bool((got[live].sign() == want[live].sign()).all()), \
        "term (D) does not share the reciprocal product's sign"

    scale = rays.norm(dim=-1)
    ratio = (got * scale[:, None] * scale[None, :])[live] / want[live]
    spread = float(ratio.std() / ratio.mean().abs())
    assert spread < 1e-4, \
        f"term (D) is not the reciprocal product up to token norms (spread {spread:.2e})"
    assert float(ratio.mean()) > 0


def test_raw_moment_keeps_the_coplanarity_zero_and_unit_does_not():
    """The reason ``moment_mode`` defaults away from SCoPE's normalised moment.

    Two rays of one camera meet, so the reciprocal product is zero -- with the RAW
    moment.  With ``m_hat`` the term becomes ``d_i.m_j/||m_j|| + d_j.m_i/||m_i||``,
    which is not zero unless the magnitudes match, so the coplanarity reading is
    gone.  Both numbers are asserted so the trade-off is a recorded measurement
    rather than a paragraph of prose.
    """
    feat = RayPE.features(_map([0.55, 0.05, 0.42]))[0]
    d, m_hat, s = feat[:, :3], feat[:, 3:6], feat[:, 6:]
    off = ~torch.eye(d.shape[0], dtype=torch.bool)
    raw = _pairwise(torch.cat([d, m_hat * s.exp()], -1))
    unit = _pairwise(torch.cat([d, m_hat], -1))
    assert float(raw[off].abs().max()) < 1e-5, "raw moment lost the coplanarity zero"
    assert float(unit[off].abs().max()) > 1e-2, \
        "the unit-moment term is expected to break coplanarity; if it stopped doing " \
        "so, this camera no longer spans a range of ||m|| and the test is vacuous"

    # and the two modes really do reach the attention differently
    for mode, hi in (("raw", 1e-4), ("unit", None)):
        pe = RayPE(head_dim=64, alpha_init=1.0, moment_mode=mode)
        td = _term_d(pe, feat[None])[0]
        worst = float(td[off].abs().max())
        if hi is not None:
            assert worst < hi, f"{mode} term (D) should vanish within one view, got {worst:.2e}"
        else:
            assert worst > 1e-3, f"{mode} term (D) should NOT vanish, got {worst:.2e}"


def test_geometric_init_is_the_flip_not_a_coincidence():
    """Without the q/k flip the same weights give a DIFFERENT bilinear form.

    Guards the one line that is easy to 'clean up' into a bug: if both sides read
    (d, m_hat), term (D) becomes d_i.d_j + m_i.m_j, which is a similarity, not a
    coplanarity, and is never zero for a ray against itself.
    """
    pe = RayPE(head_dim=64, alpha_init=1.0, use_gate=False)
    feat = RayPE.features(_map([0.55, 0.05, 0.42])).reshape(1, -1, NGI_FEATURE_DIM)
    q, k = pe(feat)
    diag_flipped = (q * k).sum(-1)
    unflipped = pe.norm_k(pe.E_k(torch.cat([feat[..., :3], feat[..., 3:6],
                                            feat[..., 6:]], -1))) * pe.alpha
    diag_plain = (q * unflipped).sum(-1)
    assert diag_flipped.abs().max() < 1e-4, "a ray meets itself; flipped diagonal must be 0"
    assert diag_plain.abs().min() > 1e-2, "unflipped diagonal should NOT vanish"


def test_alpha_zero_is_a_bit_exact_identity():
    pe = RayPE(head_dim=64)                       # alpha_init = 0.0 by default
    feat = RayPE.features(_map([0.5, 0.0, 0.4])).reshape(1, -1, NGI_FEATURE_DIM)
    q, k = pe(feat)
    assert float(q.detach().abs().max()) == 0.0 and float(k.detach().abs().max()) == 0.0


def test_alpha_lifts_off_while_E_holds_the_geometry():
    """The way out of the tier-C bind: alpha carries the zero, E carries the answer.

    At alpha=0 the branch is silent, but d(out)/d(alpha) != 0 so alpha moves on the
    first step -- unlike tier C, where zeroing the gate also zeroed the gradient of
    everything behind it.
    """
    pe = RayPE(head_dim=64)
    feat = RayPE.features(_map([0.5, 0.0, 0.4])).reshape(1, -1, NGI_FEATURE_DIM)
    q, k = pe(feat)
    (q.pow(2).sum() + k.pow(2).sum()).backward()
    # a quadratic in alpha has zero gradient at 0, so probe the linear route instead
    pe.zero_grad()
    q, k = pe(feat)
    (q.sum() + k.sum()).backward()
    assert float(pe.alpha.grad.abs().max()) > 0, "alpha is dead at init"
    assert float(pe.E_q.weight.grad.abs().max()) == 0.0, \
        "E should be still at alpha=0 -- if it moves, it moves off the geometry"


# ------------------------------------------------------------------- tier D
def test_tier_d_builds_and_reports_alphas():
    inj = build_injector("qk_pe", hidden_dim=64, pose_dim=16, head_dim=32,
                         layers=[1, 3], patch_add=False)
    assert inj.uses_qk_pe and not inj.uses_cross_attention
    assert sorted(inj.alpha_values()) == [1, 3]
    assert all(v == 0.0 for v in inj.alpha_values().values())
    feat = RayPE.features(_map([0.5, 0.0, 0.4])).reshape(1, -1, NGI_FEATURE_DIM)
    assert inj.qk_pe(2, feat) is None, "layer 2 is not injected and must return None"
    assert inj.qk_pe(1, feat)[0].shape[-1] == 32
    assert "alpha" in " ".join(inj.gate_telemetry())


def test_tier_d_head_dim_below_six_is_refused():
    try:
        RayPE(head_dim=4)
    except ValueError as e:
        assert "head_dim" in str(e)
    else:
        raise AssertionError("head_dim 4 cannot hold (d, m_hat) and must raise")


def test_tier_c_scale_gate_mode_is_identity_at_init_and_alive():
    """Item 1 applied where we actually measured the failure."""
    inj = build_injector("cross_attn", hidden_dim=32, pose_dim=16, control_dim=16,
                         n_heads=2, bottleneck=16, layers=[0], gate_mode="scale")
    pl = _map([0.5, 0.0, 0.4])
    pv = torch.zeros(1, 16)
    ctrl = inj.control_features(pl, pv)
    h = torch.randn(1, 5, 32)
    out = inj.apply_to_layer(0, h, ctrl)
    assert (out - h).abs().max() < 1e-6, "scale-gated block must start as the identity"
    out.sum().backward()
    blk = inj.blocks["0"]
    # ``proj`` carries the zero, so the block upstream of it is silent for one step
    # -- the ControlNet arrangement.  What matters is that proj itself can move.
    assert float(blk.proj.weight.grad.abs().max()) > 0, \
        "the output projection is dead, so the block can never climb out"
    assert blk.gate is None, "scale mode must not keep a free scalar switch"
    assert blk.scale_gate is not None


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
