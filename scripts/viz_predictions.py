"""Plot what R1 actually predicts, and render the clips it predicted on.

Aggregate metrics say R1 loses to a frame counter and degrades off-viewpoint; they do
not say WHAT the curves look like.  Three things are worth seeing directly:

  * where the prediction tracks the state-derived label and where it does not,
  * whether it reproduces the one shape a frame counter cannot -- the collapse when a
    grasped cube is dropped -- or whether it just ramps,
  * how far apart its answers are for the SAME physical state seen from 21 cameras,
    which is the S_view number as a picture.

Failure trajectories carry no training target under the masking policy, but the
state-derived curve still exists in the clip's own metadata, so they can be plotted --
and they are the interesting ones.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from PIL import Image, ImageDraw                                     # noqa: E402

from anyviewmeter.models.robometer_backbone import (PoseConditionedRobometer,  # noqa: E402
                                                    load_robometer)
from train_viewpoint import add_lora, bin_centres, forward_probs, load_state  # noqa: E402

# dataviz reference palette, slots 1-3, validated all-pairs light mode
TRUE, PRED, BASE = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"


def frame_counter_prior(root: str, n: int) -> np.ndarray:
    """Mean label per timestep over labelled TRAIN trajectories -- the pixel-blind line."""
    idx = json.load(open(f"{root}/index.json"))["index"]
    seen, acc, cnt = set(), np.zeros(n), 0
    for r in idx:
        if r["split"] != "train" or r["kind"].startswith("failure"):
            continue
        if r["traj"] in seen:
            continue
        seen.add(r["traj"])
        z = np.load(f"{root}/{r['path']}", allow_pickle=False)
        acc += np.asarray(json.loads(str(z["meta"]))["progress"])
        cnt += 1
    return acc / max(cnt, 1)


def predict(model, path, device, use_pose=False, zero_pose=False):
    """-> frames, label, prediction, meta.

    ``zero_pose`` keeps the pose pathway attached but hands it an all-zero camera, which
    is the ablation that showed the branch contributes nothing: if the curve does not
    move, the gate is shut whatever the geometry says.
    """
    z = np.load(path, allow_pickle=False)
    frames, meta = z["frames"], json.loads(str(z["meta"]))
    if zero_pose:
        cp = dict(meta["cam_params"])
        cp["extrinsic_cv"] = np.zeros((3, 4)).tolist()
        meta = dict(meta, cam_params=cp)
    p = forward_probs(model, frames, meta, device, use_pose and not zero_pose, False,
                      grad=False)
    yhat = (p * bin_centres(device)).sum(-1).float().cpu().numpy()
    return frames, np.asarray(meta["progress"], dtype=float), yhat, meta


def style(ax, title, ylab=True):
    ax.set_title(title, fontsize=9, color=INK, pad=6, loc="left")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("frame", fontsize=8, color=MUTED)
    if ylab:
        ax.set_ylabel("progress", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED, length=3)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def video(frames, y, yhat, out, fps=6, scale=2):
    """The clip with a live true-vs-predicted readout burned in."""
    import imageio.v2 as imageio
    h = frames.shape[1] * scale
    bar_h = 54
    w = frames.shape[2] * scale
    writer = imageio.get_writer(out, fps=fps, macro_block_size=1)
    for i, f in enumerate(frames):
        im = Image.fromarray(f).resize((w, h), Image.NEAREST)
        canvas = Image.new("RGB", (w, h + bar_h), (252, 252, 251))
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        for k, (lab, v, col) in enumerate((("true", y[i], TRUE), ("R1", yhat[i], PRED))):
            top = h + 8 + k * 22
            d.text((6, top + 2), f"{lab} {v:.2f}", fill=INK)
            x0 = 74
            d.rectangle([x0, top + 3, w - 8, top + 13], outline=GRID)
            d.rectangle([x0, top + 3, x0 + int((w - 8 - x0) * float(np.clip(v, 0, 1))),
                         top + 13], fill=col)
        writer.append_data(np.asarray(canvas))
    writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/yuang/ws_jepa/multicam_ws/outputs/pickcube_avm")
    ap.add_argument("--arm", default="R1", choices=["R1", "P1"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--gate-init", type=float, default=0.05,
                    help="must match what the checkpoint was trained with; the value is "
                         "overwritten by the loaded gate anyway, but a mismatch here "
                         "would silently change an un-checkpointed block")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    root = "/home/yuang/ws_jepa/multicam_ws/outputs"
    a.ckpt = a.ckpt or f"{root}/viewpoint/{a.arm}/best.pt"
    a.out = a.out or f"{root}/{a.arm.lower()}_viz"
    use_pose = a.arm == "P1"
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda")

    idx = json.load(open(f"{a.data}/index.json"))["index"]
    rows = [r for r in idx if r["split"] == a.split]
    # resolve through the index, not by pattern: the source tree is split/ while the
    # upload staging is clips/, and only the index knows which layout it is describing
    at = {(r["traj"], r["kind"], r["cam"]): f"{a.data}/{r['path']}" for r in rows}
    prior = frame_counter_prior(a.data, 32)

    bundle = load_robometer(device=device)
    model = PoseConditionedRobometer(bundle, injector_name="cross_attn", layers=[27, 31, 35],
                                     patch_inject=False, pose_dropout=0.0,
                                     gate_init=a.gate_init)
    model.injector.to(device)
    if use_pose:
        model.attach()
    add_lora(model.rbm, argparse.Namespace(lora_r=8, lora_layers=9))
    load_state(model, torch.load(a.ckpt, map_location=device), use_pose)
    model.rbm.eval()
    if use_pose:
        model.injector.eval()
        g = model.injector.gate_values()
        print("gates: " + " ".join(f"L{k}={v:+.4f}" for k, v in sorted(g.items())))
    print(f"loaded {a.ckpt}")

    # one trajectory of each kind, seen from the easiest and a held-out viewpoint
    kinds = ["success", "recovery", "failure_dropped", "failure_missed"]
    cams = ["canonical", "oodpose01"]
    picks = []
    for k in kinds:
        t = sorted({r["traj"] for r in rows if r["kind"] == k})
        if t:
            picks.append((t[0], k))

    fig, axes = plt.subplots(len(picks), len(cams), figsize=(9.5, 2.5 * len(picks)),
                             squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for i, (traj, kind) in enumerate(picks):
        for j, cam in enumerate(cams):
            ax = axes[i][j]
            ax.set_facecolor("#fcfcfb")
            frames, y, yhat, _ = predict(model, at[(traj, kind, cam)], device, use_pose)
            x = np.arange(len(y))
            ax.plot(x, prior, color=BASE, lw=2, ls=(0, (4, 3)), zorder=2)
            ax.plot(x, y, color=TRUE, lw=2, zorder=3)
            ax.plot(x, yhat, color=PRED, lw=2, zorder=4)
            style(ax, f"{kind}  ·  {cam}  ·  MAE {np.abs(yhat - y).mean():.3f}", ylab=(j == 0))
            if i == 0 and j == 0:                       # direct labels, not a legend box
                ax.text(x[-1], y[-1], " true", color=TRUE, fontsize=8, va="center")
                ax.text(x[-1], yhat[-1], " R1", color=PRED, fontsize=8, va="center")
                ax.text(x[8], prior[8] - 0.10, "frame counter", color=BASE, fontsize=8)
            if j == 0:
                video(frames, y, yhat, f"{a.out}/{kind}__{cam}.mp4")
                video(*predict(model, at[(traj, kind, cams[1])], device, use_pose)[:3],
                      f"{a.out}/{kind}__{cams[1]}.mp4")
    fig.suptitle(f"{a.arm} predicted progress vs state-derived label", fontsize=11,
                 color=INK, x=0.005, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(f"{a.out}/curves.png", dpi=150, facecolor="#fcfcfb")
    print(f"wrote {a.out}/curves.png")

    # viewpoint spread: same states, all 21 cameras, one colour -- the spread IS the point
    fig2, axes2 = plt.subplots(1, len(picks), figsize=(4.0 * len(picks), 3.1), squeeze=False)
    fig2.patch.set_facecolor("#fcfcfb")
    for i, (traj, kind) in enumerate(picks):
        ax = axes2[0][i]
        ax.set_facecolor("#fcfcfb")
        preds, y = [], None
        for r in [r for r in rows if r["traj"] == traj and r["kind"] == kind]:
            _, y, yhat, _ = predict(model, f"{a.data}/{r['path']}", device, use_pose)
            preds.append(yhat)
        P = np.stack(preds)
        x = np.arange(P.shape[1])
        ax.fill_between(x, P.min(0), P.max(0), color=PRED, alpha=0.22, lw=0, zorder=2)
        for row in P:
            ax.plot(x, row, color=PRED, lw=0.8, alpha=0.5, zorder=3)
        ax.plot(x, y, color=TRUE, lw=2, zorder=4)
        style(ax, f"{kind}  ·  {P.shape[0]} cameras  ·  spread {P.std(0).mean():.3f}",
              ylab=(i == 0))
        if i == 0:
            ax.text(x[-1], y[-1], " true", color=TRUE, fontsize=8, va="center")
            ax.text(x[-1], P.max(0)[-1], " R1 per camera", color=PRED, fontsize=8, va="center")
    fig2.suptitle("Same physical states, 21 viewpoints", fontsize=11, color=INK,
                  x=0.005, ha="left", y=0.99)
    fig2.tight_layout(rect=(0, 0, 1, 0.94))
    fig2.savefig(f"{a.out}/viewpoint_spread.png", dpi=150, facecolor="#fcfcfb")
    print(f"wrote {a.out}/viewpoint_spread.png")

    if use_pose:
        # The gate ended near zero, so the branch should contribute nothing.  Handing the
        # same checkpoint an all-zero camera tests that behaviourally: two curves lying on
        # top of each other means the pose pathway is inert no matter what the geometry is.
        fig3, axes3 = plt.subplots(1, len(picks), figsize=(4.0 * len(picks), 3.1),
                                   squeeze=False)
        fig3.patch.set_facecolor("#fcfcfb")
        worst = 0.0
        for i, (traj, kind) in enumerate(picks):
            ax = axes3[0][i]
            ax.set_facecolor("#fcfcfb")
            path = at[(traj, kind, "canonical")]
            _, y, with_pose, _ = predict(model, path, device, True)
            _, _, no_pose, _ = predict(model, path, device, True, zero_pose=True)
            x = np.arange(len(y))
            d = float(np.abs(with_pose - no_pose).max())
            worst = max(worst, d)
            ax.plot(x, y, color=TRUE, lw=2, zorder=3)
            ax.plot(x, with_pose, color=PRED, lw=2, zorder=4)
            ax.plot(x, no_pose, color=BASE, lw=2, ls=(0, (4, 3)), zorder=5)
            style(ax, f"{kind}  ·  max |Δ| = {d:.4f}", ylab=(i == 0))
            if i == 0:
                ax.text(x[-1], y[-1], " true", color=TRUE, fontsize=8, va="center")
                ax.text(x[2], with_pose[2] + 0.10, "with pose", color=PRED, fontsize=8)
                ax.text(x[2], with_pose[2] - 0.16, "pose zeroed", color=BASE, fontsize=8)
        fig3.suptitle(f"Does the pose pathway change anything?  worst |Δ| over these "
                      f"clips: {worst:.4f}", fontsize=11, color=INK, x=0.005, ha="left",
                      y=0.99)
        fig3.tight_layout(rect=(0, 0, 1, 0.94))
        fig3.savefig(f"{a.out}/pose_ablation.png", dpi=150, facecolor="#fcfcfb")
        print(f"wrote {a.out}/pose_ablation.png")
    print("videos:", ", ".join(sorted(f for f in os.listdir(a.out) if f.endswith(".mp4"))))


if __name__ == "__main__":
    main()
