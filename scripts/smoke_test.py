"""End-to-end smoke run over REAL clips, all three tiers, no backbone download.

What this proves, in order:
  1. the configs parse and the tiers instantiate;
  2. real rendered clips load and their stored extrinsics pass the geometric checks;
  3. Plucker maps land on the token grid the vision tower will produce;
  4. every tier is the identity on the backbone at init, and stops being one once
     its weights move -- i.e. none of them is a silent no-op;
  5. losses and the E4 gate run on the resulting tensors.

Run:  python AnyviewMeter/scripts/smoke_test.py
      python AnyviewMeter/scripts/smoke_test.py --clips /path/to/clips
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anyviewmeter.data.collators.avm_collator import AVMCollator  # noqa: E402
from anyviewmeter.data.datasets.multicam import MultiCamClipDataset  # noqa: E402
from anyviewmeter.geometry.plucker import check_plucker  # noqa: E402
from anyviewmeter.models.avm import AnyviewMeter, token_grid_for  # noqa: E402
from anyviewmeter.trainers.avm_trainer import compute_losses  # noqa: E402
from anyviewmeter.utils.config_utils import load_config  # noqa: E402

TIERS = ["tier_a_patch_add.yaml", "tier_b_cam_token.yaml", "tier_c_cross_attn.yaml",
         "pose_blind_control.yaml"]
D, NLAYERS = 128, 8


class _StubLayer(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)

    def forward(self, x, *_a, **_k):
        return (self.lin(x),)


class _StubBackbone(torch.nn.Module):
    """Stands in for Qwen3-VL so the smoke test needs no weights."""

    def __init__(self, d=D, n=NLAYERS):
        super().__init__()
        self.layers = torch.nn.ModuleList([_StubLayer(d) for _ in range(n)])

    def forward(self, x):
        for l in self.layers:
            x = l(x)[0]
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="/home/yuang/ws_jepa/multicam_ws/outputs/clips")
    ap.add_argument("--task", default="StackCube")
    ap.add_argument("--views", type=int, default=3)
    a = ap.parse_args()

    ok = True
    grid = token_grid_for(256, 16, 2)
    print(f"token grid for 256px @ patch16/merge2 = {grid}x{grid}\n")

    # ---- 1. data + geometry on real clips
    print("=" * 72)
    print("1. real clips + geometry")
    ds = MultiCamClipDataset(a.clips, tasks=[a.task], kinds=["success", "failure"],
                             max_trajectories=2, views_per_sample=a.views, split="val")
    print(f"   {ds.describe()}")
    batch = AVMCollator(token_grid=grid)([ds[0], ds[1]])
    print(f"   frames  {tuple(batch.frames.shape)}")
    print(f"   plucker {tuple(batch.plucker.shape)}   pose {tuple(batch.pose_vec.shape)}")
    stats = check_plucker(batch.plucker)
    print(f"   geometry check: ok={stats['ok']}  |d.m|max={stats['ortho_max']:.2e}  "
          f"|d| in [{stats['dir_norm_min']:.4f}, {stats['dir_norm_max']:.4f}]")
    ok &= stats["ok"]
    if batch.plucker.shape[-1] != grid:
        print(f"   !! Plucker grid {batch.plucker.shape[-1]} != token grid {grid}")
        ok = False

    # ---- 2. each tier
    n_frames = batch.frames.shape[1]
    for cfg_name in TIERS:
        print("=" * 72)
        cfg = load_config(cfg_name)
        model = AnyviewMeter(_StubBackbone(), hidden_dim=D, model_config=cfg.model,
                             n_backbone_layers=NLAYERS, token_grid=grid)
        print(f"2. {cfg_name}\n   {model.describe()}")

        patches = torch.randn(batch.plucker.shape[0], grid * grid, D)
        pl = batch.plucker
        pv = batch.pose_vec

        # identity at init
        out0 = model.inject_into_patches(patches, pl)
        ident = torch.allclose(out0, patches, atol=1e-6)
        print(f"   identity at init: {ident}")
        ok &= ident

        # not a dead branch
        if model.pose_enabled:
            with torch.no_grad():
                model.injector.encoder.out.weight.normal_(std=0.05)
            moved = (model.inject_into_patches(patches, pl) - patches).abs().max()
            print(f"   patch pathway moves output once trained: {moved > 1e-4} "
                  f"(max delta {moved:.4f})")
            ok &= bool(moved > 1e-4)

            emb = model.cam_token_embeddings(pv)
            print(f"   <cam> token: {'none (tier A)' if emb is None else tuple(emb.shape)}")

            n_hooks = model.attach_cross_attention()
            print(f"   cross-attn hooks: {n_hooks}"
                  + (f" at layers {model.injector.layers}" if n_hooks else ""))
            if n_hooks:
                model.set_control(pl[:1], pv[:1])
                x = torch.randn(1, grid * grid, D)
                before = model.backbone(x)
                with torch.no_grad():
                    for b in model.injector.blocks.values():
                        b.gate.fill_(1.0)
                        b.proj.weight.normal_(std=0.05)
                after = model.backbone(x)
                delta = (after - before).abs().max()
                print(f"   opened gates change the backbone: {delta > 1e-4} "
                      f"(max delta {delta:.4f})")
                ok &= bool(delta > 1e-4)
            model.detach_cross_attention()

        # ---- 3. heads + losses
        feats = torch.randn(batch.num_views, n_frames, D)
        out = model.apply_heads(feats)
        losses = compute_losses(out, batch, cfg.loss, groups=batch.groups())
        print(f"   progress {tuple(out.progress_logits.shape)}  "
              f"losses {({k: round(v, 4) for k, v in losses.parts.items()})}")
        losses.total.backward()
        print(f"   backward ok; pose params = "
              f"{sum(p.numel() for p in model.pose_parameters())/1e3:.1f}k")

    print("=" * 72)
    print("SMOKE TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
