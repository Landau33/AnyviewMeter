"""Finetune Robometer-4B's pose adapter and compare against the frozen baseline.

The comparison this script is built to make honest:

  * backbone FROZEN, only the ~22M pose adapter trains, so any difference is
    attributable to the pose pathway rather than to general finetuning;
  * baseline and conditioned model share the same collator, the same token
    sequence and the same clips -- ``clear_pose()`` is literally the baseline
    forward through the same object;
  * evaluation splits cameras into TRAIN and HELD-OUT.  Scoring on poses that
    appeared in training measures interpolation, not viewpoint transfer;
  * the E4 gate decides.  An S_view improvement bought with H_time or P_obj is
    reported as DISQUALIFIED, not as a win.

Usage:
    python scripts/finetune_robometer.py --steps 300
    python scripts/finetune_robometer.py --eval-only --ckpt outputs/avm/adapter.pt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, "/home/yuang/ws_jepa/multicam_ws/scripts")

from anyviewmeter.evals.diagnostics import (format_report, gate_verdict,  # noqa: E402
                                            h_time, p_obj, progress_tau, s_view)
from anyviewmeter.geometry.camera import CameraParams  # noqa: E402
from anyviewmeter.models.robometer_backbone import (PoseConditionedRobometer,  # noqa: E402
                                                    load_robometer)
from probe_lib import load_clip, perturb  # noqa: E402

CLIPS = "/home/yuang/ws_jepa/multicam_ws/outputs/clips_avm"
TASK = "PegInsertionSide"

# Cameras the adapter never sees during training.  'occluded' and 'side' are the
# interesting ones: phase A measured that on this task 'side' hides the peg in 27%
# of trajectories, so it is a genuine degradation, not just an unseen angle.
HOLDOUT_CAMS = ["side", "sweep08", "sweep09", "sweep10", "sweep11"]


def clip_path(kind: str, traj: str, cam: str) -> str:
    return f"{CLIPS}/{TASK}/{kind}/{traj}__{cam}.npz"


def available(kind="success") -> Dict[str, List[str]]:
    """-> {traj: [cam, ...]}"""
    out: Dict[str, List[str]] = {}
    d = f"{CLIPS}/{TASK}/{kind}"
    for f in sorted(os.listdir(d)):
        if not f.endswith(".npz"):
            continue
        traj, cam = f[:-4].split("__", 1)
        out.setdefault(traj, []).append(cam)
    return out


def cam_params(meta, n_frames: int) -> CameraParams:
    cp = meta["cam_params"]
    K = torch.tensor(cp["intrinsic_cv"], dtype=torch.float32)[None].expand(n_frames, 3, 3)
    E = torch.tensor(cp["extrinsic_cv"], dtype=torch.float32)[None].expand(n_frames, 3, 4)
    return CameraParams(K, E, int(meta.get("res", 256)), int(meta.get("res", 256)))


def subsample(frames, n):
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return frames[idx]


def score_clip(model, kind, traj, cam, n_frames, device, use_pose: bool,
               perturb_kind: Optional[str] = None):
    frames, meta = load_clip(clip_path(kind, traj, cam))
    frames = subsample(frames, n_frames)
    if perturb_kind:
        frames = perturb(frames, perturb_kind,
                         np.random.default_rng(abs(hash((traj, cam, perturb_kind))) % 2**32))
    inp = model.build_inputs(frames, meta["prompt"], device)
    if use_pose:
        model.set_pose(cam_params(meta, len(frames)))
    else:
        model.clear_pose()
    with torch.no_grad():
        prog = model(inp)
    return prog.float().cpu().numpy(), meta


# ------------------------------------------------------------------- training
def train(model, trajs, train_cams, device, steps, n_frames, lr, log_every, seed=0):
    rng = random.Random(seed)
    params = model.adapter_parameters()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    model.rbm.eval()                       # frozen backbone stays in eval
    model.injector.train()

    t0, hist = time.time(), []
    for step in range(steps):
        traj = rng.choice(trajs)
        cam = rng.choice(train_cams)
        kind = "success" if rng.random() < 0.5 else "failure"

        frames, meta = load_clip(clip_path(kind, traj, cam))
        frames = subsample(frames, n_frames)
        inp = model.build_inputs(frames, meta["prompt"], device)
        model.set_pose(cam_params(meta, len(frames)), training=True)

        prog = model(inp)
        # Same targets as the phase-A probe: a linear ramp for a successful demo and
        # a flat zero for the object-frozen failure, where no task progress happens.
        target = (torch.linspace(0, 1, len(prog), device=prog.device)
                  if kind == "success" else torch.zeros(len(prog), device=prog.device))
        loss = torch.nn.functional.mse_loss(prog, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        hist.append(float(loss.detach()))

        if step % log_every == 0 or step == steps - 1:
            gates = model.injector.gate_values()
            gstr = " ".join(f"{v:+.2f}" for v in gates.values())
            print(f"[{step:4d}/{steps}] loss={np.mean(hist[-log_every:]):.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} gates=[{gstr}] "
                  f"{(time.time()-t0)/(step+1):.2f}s/step", flush=True)
    return hist


# ----------------------------------------------------------------- evaluation
def evaluate(model, trajs, cams, device, n_frames, use_pose: bool, label: str):
    """Per-camera tau on success/failure plus the temporal battery."""
    per_cam_succ: Dict[str, Dict[str, float]] = {c: {} for c in cams}
    succ: Dict[str, float] = {}
    fail: Dict[str, float] = {}
    valid: Dict[str, bool] = {}
    temporal: Dict[str, List[float]] = {"stuck": [], "reversed": [], "shuffled": []}

    ref_cam = "frontal_good"
    for i, traj in enumerate(trajs):
        for cam in cams:
            if not os.path.exists(clip_path("success", traj, cam)):
                continue
            p, _ = score_clip(model, "success", traj, cam, n_frames, device, use_pose)
            per_cam_succ[cam][traj] = progress_tau(p)
            if cam == ref_cam:
                succ[traj] = per_cam_succ[cam][traj]
                pf, _ = score_clip(model, "failure", traj, cam, n_frames, device, use_pose)
                fail[traj] = progress_tau(pf)
                fs, _ = load_clip(clip_path("success", traj, cam))
                ff, _ = load_clip(clip_path("failure", traj, cam))
                valid[traj] = not np.array_equal(fs, ff)
        for k in temporal:
            p, _ = score_clip(model, "success", traj, ref_cam, n_frames, device,
                              use_pose, perturb_kind=k)
            temporal[k].append(progress_tau(p))
        print(f"    {label}: {i+1}/{len(trajs)} trajectories", flush=True)

    return {
        "S_view": s_view({c: v for c, v in per_cam_succ.items() if v}),
        "H_time": h_time(temporal),
        "P_obj": p_obj(succ, fail, valid),
        "per_cam": {c: float(np.mean(list(v.values()))) for c, v in per_cam_succ.items() if v},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--frames", type=int, default=4,
                    help="frames per clip.  Measured on a 12GB 4070S next to a 4B bf16\n"
                         "backbone: 4 frames peaks at 10.9GB, 6 at 11.6GB (the cap).")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-frames", type=int, default=0,
                    help="frames per clip at EVAL time (0 = same as --frames).  "
                         "Evaluation runs under no_grad and peaks at 9.1GB, so it is "
                         "not bound by the training memory limit -- and it must not "
                         "be: at 4 frames Kendall tau takes only 7 distinct values, "
                         "which quantises P_obj to exactly 0 and makes S_view unable "
                         "to separate cameras.  Phase A used 32.")
    ap.add_argument("--eval-trajs", type=int, default=8)
    ap.add_argument("--train-trajs", type=int, default=40)
    ap.add_argument("--injector", default="cross_attn")
    ap.add_argument("--out", default="/home/yuang/ws_jepa/multicam_ws/outputs/avm")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--patch-inject", action="store_true",
                    help="also inject Plucker at the vision tower.  Forces gradients "
                         "back through all 36 layers; does not fit in 12GB.")
    ap.add_argument("--xattn-layers", default="",
                    help="explicit cross-attn layer indices, e.g. 27,31,35")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda")

    avail = available()
    all_trajs = sorted(avail, key=lambda t: int(t.split("_")[1]))
    all_cams = sorted(avail[all_trajs[0]])
    train_cams = [c for c in all_cams if c not in HOLDOUT_CAMS]
    held_cams = [c for c in all_cams if c in HOLDOUT_CAMS]
    train_trajs = all_trajs[:a.train_trajs]
    eval_trajs = all_trajs[a.train_trajs:a.train_trajs + a.eval_trajs]

    print(f"task={TASK}  trajectories: {len(train_trajs)} train / {len(eval_trajs)} eval")
    print(f"cameras: {len(train_cams)} train {train_cams}")
    print(f"         {len(held_cams)} HELD OUT {held_cams}")
    ef = a.eval_frames or a.frames
    print(f"frames/clip: train={a.frames}  eval={ef}\n")

    bundle = load_robometer(device=device)
    layers = [int(x) for x in a.xattn_layers.split(",") if x.strip()] or None
    model = PoseConditionedRobometer(bundle, injector_name=a.injector,
                                     control_dim=256, bottleneck=256,
                                     layer_fraction=0.25, layers=layers,
                                     patch_inject=a.patch_inject,
                                     pose_dropout=0.1)
    model.injector.to(device)
    model.attach()
    ok = model.enable_gradient_checkpointing(True)
    print(model.describe(), flush=True)
    print(f"gradient checkpointing: {ok}", flush=True)

    results = {}

    # ---- baseline: identical object, pose pathway switched off
    print("\n=== BASELINE Robometer-4B (no pose) ===", flush=True)
    for name, cams in (("train_cams", train_cams), ("holdout_cams", held_cams)):
        results[f"baseline/{name}"] = evaluate(model, eval_trajs, cams, device,
                                               ef, use_pose=False, label=f"base/{name}")

    if a.ckpt and os.path.exists(a.ckpt):
        model.injector.load_state_dict(torch.load(a.ckpt, map_location=device))
        print(f"loaded adapter from {a.ckpt}")
    elif not a.eval_only:
        print(f"\n=== TRAIN pose adapter ({a.steps} steps) ===", flush=True)
        hist = train(model, train_trajs, train_cams, device, a.steps, a.frames,
                     a.lr, a.log_every)
        torch.save(model.injector.state_dict(), f"{a.out}/adapter.pt")
        json.dump(hist, open(f"{a.out}/loss_history.json", "w"))
        print(f"saved adapter -> {a.out}/adapter.pt")

    print("\n=== AnyviewMeter (pose-conditioned) ===", flush=True)
    model.injector.eval()
    for name, cams in (("train_cams", train_cams), ("holdout_cams", held_cams)):
        results[f"avm/{name}"] = evaluate(model, eval_trajs, cams, device,
                                          ef, use_pose=True, label=f"avm/{name}")

    results["gates"] = model.injector.gate_values() if model.injector.uses_cross_attention else {}
    json.dump(results, open(f"{a.out}/comparison.json", "w"), indent=2, default=float)

    # ------------------------------------------------------------------ report
    L = ["", "=" * 78, f"AnyviewMeter vs Robometer-4B  ({TASK}, {len(eval_trajs)} held-out trajectories)",
         "=" * 78, ""]
    for split in ("train_cams", "holdout_cams"):
        b, v = results[f"baseline/{split}"], results[f"avm/{split}"]
        L += [f"--- {split} ---",
              format_report("Robometer-4B (frozen)", b),
              format_report("AnyviewMeter (pose)", v, gate_verdict(v, b)), ""]
    if results["gates"]:
        L += ["cross-attention gates: " +
              " ".join(f"L{k}={v:+.3f}" for k, v in sorted(results["gates"].items())),
              "(a gate near zero means the backbone declined the pose signal)", ""]
    txt = "\n".join(L)
    open(f"{a.out}/comparison.txt", "w").write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
