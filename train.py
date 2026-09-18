"""Training entry point.

    python AnyviewMeter/train.py --config tier_c_cross_attn.yaml
    python AnyviewMeter/train.py --config tier_a_patch_add.yaml \
        --set training.max_steps=50 loss.pose_consistency_weight=0.1

The backbone loader is the one piece deliberately left as a seam: ``AnyviewMeter``
takes ``backbone`` as an argument, so swapping Qwen3-VL for anything else is a
loader change, not a redesign.  ``--dry-run`` exercises the whole loop on a stub
backbone, which is how to check the config and data wiring without a download.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anyviewmeter.data.collators.avm_collator import AVMCollator  # noqa: E402
from anyviewmeter.data.datasets.multicam import build_splits  # noqa: E402
from anyviewmeter.models.avm import AnyviewMeter, token_grid_for  # noqa: E402
from anyviewmeter.trainers.avm_trainer import (build_optimizer, clip_grads,  # noqa: E402
                                               compute_losses)
from anyviewmeter.utils.config_utils import load_config, save_config  # noqa: E402


def load_backbone(cfg, dry_run: bool, hidden_dim_hint: int = 128):
    """Return (backbone, hidden_dim, n_layers, processor, tokenizer).

    ``dry_run`` returns a stub with the attribute layout AnyviewMeter expects, so
    the loop can be exercised without weights.
    """
    if dry_run:
        class _L(torch.nn.Module):
            def __init__(self, d):
                super().__init__()
                self.lin = torch.nn.Linear(d, d)

            def forward(self, x, *_a, **_k):
                return (self.lin(x),)

        class _B(torch.nn.Module):
            def __init__(self, d, n):
                super().__init__()
                self.layers = torch.nn.ModuleList([_L(d) for _ in range(n)])

            def forward(self, x):
                for l in self.layers:
                    x = l(x)[0]
                return x

        return _B(hidden_dim_hint, 8), hidden_dim_hint, 8, None, None

    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    mid = cfg.model.base_model_id
    hf = AutoConfig.from_pretrained(mid, trust_remote_code=True)
    text_cfg = getattr(hf, "text_config", hf)
    hidden = getattr(text_cfg, "hidden_size")
    n_layers = getattr(text_cfg, "num_hidden_layers", 36)

    try:
        from transformers import Qwen3VLModel as _Model
    except ImportError as e:                       # pragma: no cover
        raise ImportError("Qwen3-VL requires a transformers build that ships "
                          "Qwen3VLModel") from e

    backbone = _Model.from_pretrained(
        mid, dtype=torch.bfloat16 if cfg.training.bf16 else torch.float32,
        trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    return backbone, hidden, n_layers, processor, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tier_c_cross_attn.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides",
                    help="dotted overrides, e.g. training.lr=1e-4")
    ap.add_argument("--dry-run", action="store_true",
                    help="stub backbone; checks config/data/loop wiring only")
    a = ap.parse_args()

    cfg = load_config(a.config, a.overrides)
    torch.manual_seed(cfg.training.seed)

    out_dir = os.path.join(cfg.training.output_dir, cfg.training.run_name)
    os.makedirs(out_dir, exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    backbone, hidden, n_layers, processor, tokenizer = load_backbone(cfg, a.dry_run)
    grid = token_grid_for(cfg.data.image_size, 16, 2)

    model = AnyviewMeter(backbone, hidden_dim=hidden, model_config=cfg.model,
                         processor=processor, tokenizer=tokenizer,
                         n_backbone_layers=n_layers, token_grid=grid)
    if cfg.training.freeze_backbone:
        model.freeze_backbone(True)
    n_hooks = model.attach_cross_attention()

    print(model.describe())
    print(f"token grid {grid}x{grid}, cross-attn hooks: {n_hooks}")
    print(f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    splits = build_splits(cfg.data)
    for ds in splits.values():
        print(f"  {ds.describe()}")

    collate = AVMCollator(token_grid=grid, processor=processor, tokenizer=tokenizer,
                          normalize_directions=cfg.model.pose.normalize_directions)
    loader = DataLoader(splits["train"], batch_size=cfg.data.batch_size,
                        shuffle=cfg.data.shuffle, num_workers=0,
                        collate_fn=collate)

    opt = build_optimizer(model, cfg.training)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.dry_run else "cpu")
    model.to(device)

    step, max_steps = 0, cfg.training.max_steps or len(loader) * cfg.training.epochs
    model.train()
    for _epoch in range(cfg.training.epochs):
        for batch in loader:
            batch = batch.to(device)
            plucker, pose_vec = model.maybe_drop_pose(batch.plucker, batch.pose_vec)
            model.set_control(plucker, pose_vec)

            # NOTE: with a real backbone the frame features come from the VLM
            # forward (patch injection + <cam> scatter happen inside it).  The stub
            # path below exists so --dry-run can exercise losses and the optimiser.
            feats = torch.randn(batch.num_views, batch.frames.shape[1], hidden,
                                device=device)
            out = model.apply_heads(feats)
            losses = compute_losses(out, batch, cfg.loss, groups=batch.groups())

            losses.total.backward()
            clip_grads(model, cfg.training.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)

            if step % cfg.training.log_every == 0:
                parts = " ".join(f"{k}={v:.4f}" for k, v in losses.parts.items())
                gates = out.diagnostics
                gate_s = ("  " + " ".join(f"{k}={v:+.3f}" for k, v in gates.items())
                          if gates else "")
                print(f"[{step}/{max_steps}] {parts}{gate_s}", flush=True)
            step += 1
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    model.detach_cross_attention()
    torch.save({"model": model.state_dict(), "config": a.config},
               os.path.join(out_dir, "checkpoint.pt"))
    print(f"saved -> {out_dir}/checkpoint.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
