"""End-to-end model tests on a tiny stub backbone (no GPU, no weights download).

The stub has the same surface AnyviewMeter relies on -- an embedding table and a
list of decoder layers -- so the wiring under test is the real wiring: patch
injection, <cam> token scattering, and the tier-C forward hooks.

Run:  python AnyviewMeter/tests/test_model.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anyviewmeter.configs.experiment_configs import ModelConfig, PoseConfig  # noqa: E402
from anyviewmeter.geometry.camera import CameraParams, look_at_extrinsic, pinhole_intrinsic  # noqa: E402
from anyviewmeter.models.avm import AnyviewMeter, token_grid_for  # noqa: E402

D, G, RES, NLAYERS = 64, 8, 256, 8
VOCAB, CAM_TOKEN_ID = 100, 99


class StubLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x, *_a, **_k):
        return (self.lin(x),)


class StubBackbone(nn.Module):
    """Minimal stand-in with the attribute layout AnyviewMeter looks for."""

    def __init__(self, d=D, n=NLAYERS):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d)
        self.layers = nn.ModuleList([StubLayer(d) for _ in range(n)])

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds):
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h)[0]
        return h


def _cams(eyes):
    K = pinhole_intrinsic(np.deg2rad(45.0), RES, RES)
    ext = torch.stack([look_at_extrinsic(e, [0.0, 0.0, 0.0]) for e in eyes])
    return CameraParams(K[None].expand(len(eyes), 3, 3), ext, RES, RES)


def _model(injector="cross_attn", enabled=True):
    cfg = ModelConfig(pose=PoseConfig(enabled=enabled, injector=injector,
                                      control_dim=32, n_heads=4, n_control_layers=1,
                                      layer_fraction=0.5, pose_dropout=0.0))
    return AnyviewMeter(StubBackbone(), hidden_dim=D, model_config=cfg,
                        n_backbone_layers=NLAYERS, token_grid=G)


def test_token_grid_matches_qwen3vl():
    """256px through patch-16 / merge-2 must be an 8x8 token grid."""
    assert token_grid_for(256, 16, 2) == 8
    assert token_grid_for(448, 16, 2) == 14
    try:
        token_grid_for(250, 16, 2)
    except ValueError:
        return
    raise AssertionError("indivisible image size should raise")


def test_build_pose_inputs_shapes():
    m = _model()
    cams = _cams([[0.4, 0, 0.4], [0, 0.55, 0.05]])
    pl, pv = m.build_pose_inputs([cams], workspace_centre=torch.zeros(3))
    assert pl.shape == (2, 6, G, G), pl.shape
    assert pv.shape == (2, 16), pv.shape


def test_patch_injection_is_identity_at_init_for_every_tier():
    cams = _cams([[0.4, 0, 0.4]])
    for tier in ("patch_add", "cam_token", "cross_attn"):
        m = _model(tier).eval()
        pl, _ = m.build_pose_inputs([cams], torch.zeros(3))
        patches = torch.randn(1, G * G, D)
        out = m.inject_into_patches(patches, pl)
        assert torch.allclose(out, patches, atol=1e-6), tier


def test_pose_disabled_is_a_clean_control():
    """pose.enabled=False must walk the same path with no pose machinery."""
    m = _model(enabled=False)
    assert m.injector is None and m.pose_probe is None
    patches = torch.randn(1, G * G, D)
    assert torch.equal(m.inject_into_patches(patches, torch.zeros(1, 6, G, G)), patches)
    assert m.cam_token_embeddings(torch.zeros(1, 16)) is None
    assert m.attach_cross_attention() == 0
    assert "pose disabled" in m.describe()


def test_cam_token_scatter_overwrites_only_reserved_slots():
    m = _model("cam_token")
    cams = _cams([[0.4, 0, 0.4], [0, 0.55, 0.05]])
    _, pv = m.build_pose_inputs([cams], torch.zeros(3))
    emb = m.cam_token_embeddings(pv)

    ids = torch.tensor([[1, CAM_TOKEN_ID, 2, 3, CAM_TOKEN_ID, 4]])
    base = torch.randn(1, 6, D)
    out = m.scatter_cam_tokens(base, ids, CAM_TOKEN_ID, emb)

    keep = [0, 2, 3, 5]
    assert torch.allclose(out[0, keep], base[0, keep]), "non-<cam> positions were modified"
    assert torch.allclose(out[0, 1], emb[0].to(out.dtype))
    assert torch.allclose(out[0, 4], emb[1].to(out.dtype))


def test_cam_token_count_mismatch_raises():
    m = _model("cam_token")
    cams = _cams([[0.4, 0, 0.4]])
    _, pv = m.build_pose_inputs([cams], torch.zeros(3))
    emb = m.cam_token_embeddings(pv)                    # 1 embedding
    ids = torch.tensor([[CAM_TOKEN_ID, CAM_TOKEN_ID]])  # 2 slots
    try:
        m.scatter_cam_tokens(torch.randn(1, 2, D), ids, CAM_TOKEN_ID, emb)
    except ValueError as e:
        assert "disagree" in str(e)
        return
    raise AssertionError("count mismatch did not raise")


def test_cross_attention_hooks_attach_and_are_identity_at_init():
    m = _model("cross_attn").eval()
    n = m.attach_cross_attention()
    assert n == len(m.injector.layers) > 0

    cams = _cams([[0.4, 0, 0.4]])
    pl, pv = m.build_pose_inputs([cams], torch.zeros(3))
    x = torch.randn(1, G * G, D)

    m.set_control(None, None)
    ref = m.backbone(x)
    m.set_control(pl, pv)
    got = m.backbone(x)
    assert torch.allclose(ref, got, atol=1e-6), "zero-gated hooks changed the output at init"
    m.detach_cross_attention()


def test_cross_attention_changes_output_once_gates_open():
    m = _model("cross_attn").eval()
    m.attach_cross_attention()
    cams = _cams([[0.4, 0, 0.4]])
    pl, pv = m.build_pose_inputs([cams], torch.zeros(3))
    x = torch.randn(1, G * G, D)

    m.set_control(None, None)
    ref = m.backbone(x)
    with torch.no_grad():
        for b in m.injector.blocks.values():
            b.gate.fill_(1.0)
            b.proj.weight.normal_(std=0.05)
    m.set_control(pl, pv)
    got = m.backbone(x)
    assert (got - ref).abs().max() > 1e-4, "opened gates did not affect the backbone"
    m.detach_cross_attention()


def test_detach_restores_backbone():
    m = _model("cross_attn").eval()
    m.attach_cross_attention()
    with torch.no_grad():
        for b in m.injector.blocks.values():
            b.gate.fill_(1.0)
            b.proj.weight.normal_(std=0.05)
    cams = _cams([[0.4, 0, 0.4]])
    pl, pv = m.build_pose_inputs([cams], torch.zeros(3))
    m.set_control(pl, pv)
    x = torch.randn(1, G * G, D)
    hooked = m.backbone(x)
    m.detach_cross_attention()
    clean = m.backbone(x)
    assert not torch.allclose(hooked, clean, atol=1e-6)


def test_heads_shapes_and_backward():
    m = _model("cross_attn")
    feats = torch.randn(2, 5, D, requires_grad=True)
    out = m.apply_heads(feats)
    assert out.progress_logits.shape == (2, 5)
    assert out.success_logits.shape == (2, 5)
    assert out.pose_pred.shape == (2, 5, 16)
    assert any(k.startswith("gate/") for k in out.diagnostics), out.diagnostics
    out.progress_logits.sum().backward()
    assert feats.grad is not None


def test_discrete_progress_head_shape():
    cfg = ModelConfig(progress_loss_type="discrete", progress_discrete_bins=7,
                      pose=PoseConfig(enabled=False))
    m = AnyviewMeter(StubBackbone(), hidden_dim=D, model_config=cfg,
                     n_backbone_layers=NLAYERS, token_grid=G)
    out = m.apply_heads(torch.randn(2, 5, D))
    assert out.progress_logits.shape == (2, 5, 7), out.progress_logits.shape


def test_pose_dropout_blanks_whole_samples_in_train_mode():
    cfg = ModelConfig(pose=PoseConfig(enabled=True, injector="patch_add", pose_dropout=1.0))
    m = AnyviewMeter(StubBackbone(), hidden_dim=D, model_config=cfg,
                     n_backbone_layers=NLAYERS, token_grid=G).train()
    pl = torch.randn(4, 6, G, G)
    pv = torch.randn(4, 16)
    pl2, pv2 = m.maybe_drop_pose(pl, pv)
    assert pl2.abs().max() == 0 and pv2.abs().max() == 0
    m.eval()
    pl3, pv3 = m.maybe_drop_pose(pl, pv)
    assert torch.equal(pl3, pl), "dropout must not fire in eval"


def test_parameter_groups_are_disjoint():
    m = _model("cross_attn")
    pose = {id(p) for p in m.pose_parameters()}
    heads = {id(p) for p in m.head_parameters()}
    assert pose and heads and not (pose & heads)


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
