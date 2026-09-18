"""Tests for the three pose-injection tiers.

The properties asserted here are the ones the tiers' comparability depends on:

  * every tier is the IDENTITY on the backbone at initialisation (zero-init), so a
    difference between tiers after training is attributable to the pathway and not
    to a different starting point;
  * pose actually reaches the output once weights are non-zero -- a pathway that
    cannot change the output no matter the weights is a silent no-op, which is the
    exact failure mode VD3D reported and the one we are trying to detect;
  * different cameras produce different conditioning (a tier that maps every
    viewpoint to the same features would look "robust" for the wrong reason).

Run:  python AnyviewMeter/tests/test_injection.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anyviewmeter.geometry.camera import CameraParams, look_at_extrinsic, pinhole_intrinsic  # noqa: E402
from anyviewmeter.geometry.plucker import POSE_VECTOR_DIM, plucker_map, pose_vector  # noqa: E402
from anyviewmeter.models.injection import available_injectors, build_injector  # noqa: E402

D = 64            # backbone hidden dim (small, for speed)
G = 8             # token grid
RES = 256
TIERS = ["patch_add", "cam_token", "cross_attn"]


def _cams(eyes):
    K = pinhole_intrinsic(np.deg2rad(45.0), RES, RES)
    ext = torch.stack([look_at_extrinsic(e, [0.0, 0.0, 0.0]) for e in eyes])
    Ks = K[None].expand(len(eyes), 3, 3)
    return CameraParams(Ks, ext, RES, RES)


def _inputs(n=2):
    cam = _cams([[0.4, 0.0, 0.4], [0.0, 0.55, 0.05]][:n])
    pl = plucker_map(cam, G, G)                                # (n,6,G,G)
    pv = pose_vector(cam, workspace_centre=torch.zeros(3))     # (n,16)
    patches = torch.randn(n, G * G, D)
    return patches, pl, pv


def _build(name, **kw):
    return build_injector(name, hidden_dim=D, pose_dim=POSE_VECTOR_DIM,
                          n_backbone_layers=8, control_dim=32, n_heads=4,
                          n_control_layers=1, **kw)


def test_registry_has_all_three():
    assert set(TIERS).issubset(set(available_injectors())), available_injectors()


def test_all_tiers_are_identity_at_init():
    """Zero-init means the backbone is untouched before any learning."""
    patches, pl, pv = _inputs()
    for name in TIERS:
        inj = _build(name).eval()
        out = inj.inject_patches(patches, pl)
        assert torch.allclose(out, patches, atol=1e-6), \
            f"{name} perturbed patch tokens at init (max {(out - patches).abs().max():.2e})"


def test_cross_attn_gate_starts_closed_and_is_identity():
    patches, pl, pv = _inputs()
    inj = _build("cross_attn").eval()
    assert all(abs(g) < 1e-8 for g in inj.gate_values().values()), inj.gate_values()
    ctrl = inj.control_features(pl, pv)
    for layer in inj.layers:
        out = inj.apply_to_layer(layer, patches, ctrl)
        assert torch.allclose(out, patches, atol=1e-6), f"layer {layer} not identity at init"


def test_pose_reaches_patches_once_weights_are_nonzero():
    """The patch pathway must be able to move the output -- not a dead branch."""
    patches, pl, pv = _inputs()
    for name in TIERS:
        inj = _build(name)
        with torch.no_grad():
            inj.encoder.out.weight.normal_(std=0.05)
        out = inj.inject_patches(patches, pl)
        assert (out - patches).abs().max() > 1e-4, f"{name} patch pathway is a no-op"


def test_zero_init_does_not_kill_the_gradient():
    """A block that is silent at init must still be able to START learning.

    Zero-initialising BOTH the gate and the output projection makes the block an
    identity that can never move: d/dW = tanh(g) = 0 and d/dg = W(x) = 0.  This test
    exists because that bug shipped once and was invisible to the identity tests --
    the model trained, the loss moved (via other params), and the gates stayed at
    exactly 0.0 forever.
    """
    patches, pl, pv = _inputs()
    inj = _build("cross_attn")
    ctrl = inj.control_features(pl, pv)
    layer = inj.layers[0]
    out = inj.apply_to_layer(layer, patches, ctrl)
    assert torch.allclose(out, patches, atol=1e-6), "must still be identity at init"
    out.sum().backward()
    blk = inj.blocks[str(layer)]
    assert blk.gate.grad is not None and blk.gate.grad.abs().item() > 0, \
        "gate has zero gradient at init -- the cross-attention pathway is dead"


def test_cross_attn_gate_opens_and_changes_output():
    patches, pl, pv = _inputs()
    inj = _build("cross_attn")
    with torch.no_grad():
        for b in inj.blocks.values():
            b.gate.fill_(1.0)
            b.proj.weight.normal_(std=0.05)
    ctrl = inj.control_features(pl, pv)
    layer = inj.layers[0]
    out = inj.apply_to_layer(layer, patches, ctrl)
    assert (out - patches).abs().max() > 1e-4, "cross-attention pathway is a no-op when open"


def test_cam_token_only_for_tiers_b_and_c():
    _, _, pv = _inputs()
    assert _build("patch_add").cam_token_embedding(pv) is None
    for name in ("cam_token", "cross_attn"):
        inj = _build(name)
        emb = inj.cam_token_embedding(pv)
        assert emb is not None and emb.shape == (pv.shape[0], D), name
        assert emb.abs().max() > 0, f"{name} <cam> token is all zeros -- indistinguishable from padding"


def test_capability_flags_match_tier():
    flags = {n: (_build(n).emits_cam_token, _build(n).uses_cross_attention) for n in TIERS}
    assert flags["patch_add"] == (False, False), flags
    assert flags["cam_token"] == (True, False), flags
    assert flags["cross_attn"] == (True, True), flags


def test_different_viewpoints_give_different_conditioning():
    """A tier that collapses all viewpoints would fake robustness."""
    patches, pl, pv = _inputs(n=2)
    for name in TIERS:
        inj = _build(name)
        with torch.no_grad():
            inj.encoder.out.weight.normal_(std=0.05)
        out = inj.inject_patches(patches, pl)
        delta = (out - patches)
        assert (delta[0] - delta[1]).abs().max() > 1e-4, \
            f"{name} produced identical conditioning for two different cameras"

    for name in ("cam_token", "cross_attn"):
        emb = _build(name).cam_token_embedding(pv)
        assert (emb[0] - emb[1]).abs().max() > 1e-4, f"{name} <cam> token ignores the camera"


def test_grid_mismatch_is_a_loud_error():
    """A Plucker map built on the wrong grid must fail, not broadcast silently."""
    patches, pl, _ = _inputs()
    inj = _build("patch_add")
    bad = torch.randn(patches.shape[0], 6, G * 2, G * 2)
    try:
        inj.inject_patches(patches, bad)
    except ValueError as e:
        assert "token" in str(e).lower()
        return
    raise AssertionError("mismatched Plucker grid did not raise")


def test_gradients_flow_to_pose_pathway():
    patches, pl, pv = _inputs()
    for name in TIERS:
        inj = _build(name)
        with torch.no_grad():
            inj.encoder.out.weight.normal_(std=0.05)
        out = inj.inject_patches(patches, pl).sum()
        out.backward()
        g = inj.encoder.out.weight.grad
        assert g is not None and g.abs().max() > 0, f"{name}: no gradient into the Plucker encoder"


def test_parameter_budget_is_small_relative_to_backbone():
    """Injection should be an adapter, not a second model."""
    for name in TIERS:
        n = _build(name).trainable_parameter_count()
        assert n < 5e6, f"{name} has {n/1e6:.1f}M params at D={D}; too heavy for an adapter"


def _run_all():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            torch.manual_seed(0)
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
