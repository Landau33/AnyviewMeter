"""Pre-flight for train_viewpoint.py: wiring, gradients and memory, no dataset needed.

Checks the things that would otherwise fail 40 minutes into a run:

  1. LoRA injection leaves the module tree intact.  ``inject_adapter_in_model`` edits in
     place, but if it ever wrapped the model the pose hooks -- which reach in via
     ``rbm.model.language_model.layers`` -- would attach to the wrong object and pose
     conditioning would silently become a no-op.
  2. Gradient actually reaches every group we pay an optimiser slot for: LoRA, the
     progress head, the control branch, and the gates.
  3. Peak memory at the intended frame count, including the second (no_grad) forward
     the view-consistency term needs.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from anyviewmeter.geometry.camera import CameraParams  # noqa: E402
from anyviewmeter.models.robometer_backbone import load_robometer  # noqa: E402
from train_viewpoint import (add_lora, bin_centres, build_model, js_divergence,  # noqa: E402
                             soft_bin_targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--arm", default="P2")
    ap.add_argument("--xattn-layers", default="27,31,35")
    ap.add_argument("--control-dim", type=int, default=256)
    ap.add_argument("--bottleneck", type=int, default=256)
    ap.add_argument("--gate-init", type=float, default=0.05)
    ap.add_argument("--pose-dropout", type=float, default=0.0)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-layers", type=int, default=9)
    a = ap.parse_args()

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    bundle = load_robometer(device=device)
    print(f"loaded backbone: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")

    model, use_pose = build_model(a.arm, bundle, device, a)
    layers_before = id(model._layers())
    n_lora, n_head = add_lora(model.rbm, a)
    layers_after = id(model._layers())
    print(f"\n1. module tree: _layers() resolves before={layers_before} after={layers_after} "
          f"-> {'INTACT' if layers_before == layers_after else 'MOVED (hooks would break)'}")
    print(f"   attached hooks: {len(model._hooks)}  (expect {len(a.xattn_layers.split(','))} "
          f"xattn + 1 visual = {len(a.xattn_layers.split(',')) + 1})")
    model.enable_gradient_checkpointing(True)
    print("  ", model.describe())
    print(f"   trainable: LoRA {n_lora/1e6:.2f}M + head {n_head/1e6:.2f}M" +
          (f" + adapter {sum(p.numel() for p in model.adapter_parameters())/1e6:.2f}M"
           if use_pose else ""))

    # a fake clip: random frames, a plausible camera, a monotone label
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (a.frames, 256, 256, 3), dtype=np.uint8)
    K = torch.tensor([[274.5, 0, 128.0], [0, 274.5, 128.0], [0, 0, 1.0]])
    E = torch.tensor([[0., -1., 0., 0.], [0., 0., -1., 0.2], [1., 0., 0., 0.6]])
    cam = CameraParams(K[None].expand(a.frames, 3, 3).to(device),
                       E[None].expand(a.frames, 3, 4).to(device), 256, 256)
    y = torch.linspace(0.05, 0.95, a.frames, device=device)

    inp = model.build_inputs(frames, "pick up the red cube", device)
    if use_pose:
        model.set_pose(cam, training=True)

    torch.cuda.reset_peak_memory_stats()
    if use_pose:                       # the no_grad view-B forward the JS term needs
        with torch.no_grad():
            _, lb = model(inp, return_logits=True)
            p_b = lb.float().softmax(-1)
    _, logits = model(inp, return_logits=True)
    p_a = logits.float().softmax(-1)
    print(f"\n3. logits {tuple(logits.shape)} (expect ({a.frames}, 10))")

    q = soft_bin_targets(y)
    loss = -(q * p_a.clamp_min(1e-8).log()).sum(-1).mean()
    loss = loss + 0.5 * torch.nn.functional.mse_loss((p_a * bin_centres(device)).sum(-1), y)
    if use_pose:
        loss = loss + 0.2 * js_divergence(p_a, p_b)
    loss.backward()

    print("\n2. gradient reached:")
    def norm(mods):
        t = 0.0
        for _, p in mods:
            if p.grad is not None:
                t += float(p.grad.norm()) ** 2
        return t ** 0.5
    lora = [(n, p) for n, p in model.rbm.named_parameters() if "lora_" in n]
    head = [(n, p) for n, p in model.rbm.progress_head.named_parameters()]
    print(f"   LoRA          |grad| = {norm(lora):.4e}  ({len(lora)} tensors)")
    print(f"   progress_head |grad| = {norm(head):.4e}")
    if use_pose:
        ctrl = [(n, p) for n, p in model.injector.named_parameters()
                if n.startswith("control") and p.requires_grad]
        gate = [(n, p) for n, p in model.injector.named_parameters() if n.endswith("gate")]
        blk = [(n, p) for n, p in model.injector.named_parameters()
               if n.startswith("blocks") and not n.endswith("gate")]
        dead = [(n, p) for n, p in model.injector.named_parameters() if not p.requires_grad]
        print(f"   control branch|grad| = {norm(ctrl):.4e}  ({len(ctrl)} tensors)")
        print(f"   xattn blocks  |grad| = {norm(blk):.4e}")
        print(f"   gates         |grad| = {norm(gate):.4e}   values="
              f"{[round(v, 4) for v in model.injector.gate_values().values()]}")
        print(f"   frozen-unreachable   = {sum(p.numel() for _, p in dead)/1e6:.2f}M "
              f"(no grad by design)")
        bad = [n for n, p in ctrl + gate + blk if p.grad is None or p.grad.norm() == 0]
        print(f"   -> pose pathway {'LIVE' if not bad else 'DEAD for: ' + str(bad[:4])}")

    print(f"\n4. peak memory for the training step: "
          f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
          f"(total reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GiB)")


if __name__ == "__main__":
    sys.exit(main())
