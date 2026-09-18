"""Is "blacking out the image makes the model better" an artefact of an invisible cube?

THE OBSERVATION.  On pickcube_avm both the frozen backbone (R0) and the LoRA finetune
(R1) score a LOWER MAE and a HIGHER tau on all-black frames than on the real render.
Two readings, opposite fixes:

  (a) the image is genuinely unreadable on much of this test set -- the cube covers
      0.5% of the frame on average and under 0.1% on a quarter of the clips, so the
      model has nothing to see and a black frame simply lets its time prior through
      undisturbed; or
  (b) the pixels reach the model wrong (preprocessing, frame/token alignment), in which
      case the image is actively misleading everywhere and visibility is irrelevant.

They separate cleanly: under (a) the black-vs-real gap must SHRINK as the cube gets
bigger and vanish on the clips where it is plainly visible.  Under (b) the gap stays
flat across visibility -- a broken pixel path does not care how big the cube is.

``d_pred`` is the third column that matters: the mean |p(real) - p(black)| per frame.
It is not a score, it is an influence measure -- how much the image moves the output at
all.  A model whose d_pred is near zero is not being hurt by the image, it is ignoring
it, and then neither (a) nor (b) is the story.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from anyviewmeter.models.robometer_backbone import (PoseConditionedRobometer,  # noqa: E402
                                                    load_robometer)
from train_viewpoint import (ClipStore, bin_centres, forward_probs,  # noqa: E402
                             kendall_tau, load_state)

# Edges in PERCENT of frame area covered by the cube.  The first bin is "the object is
# effectively not in the picture": 0.1% of 256x256 is 65 pixels, and one Qwen3-VL token
# after the 2x2 merge covers 32x32 = 1024 of them.
EDGES = [0.0, 0.1, 0.3, 0.6, 100.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R0", choices=["R0", "R1"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--data", default="/home/yuang/ws_jepa/multicam_ws/outputs/pickcube_avm")
    ap.add_argument("--split", default="test")
    ap.add_argument("--trajs", type=int, default=15)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    device = torch.device("cuda")

    idx = json.load(open(f"{a.data}/index.json"))["index"]
    frac = {(r["traj"], r["kind"], r["cam"]): r["mean_cube_frac"] * 100.0 for r in idx}

    store = ClipStore(a.data, a.split, "mask")
    keys = store.keys[:a.trajs] if a.trajs else store.keys

    bundle = load_robometer(device=device)
    model = PoseConditionedRobometer(bundle, injector_name="cross_attn",
                                     layers=[27, 31, 35], patch_inject=False,
                                     pose_dropout=0.0, gate_init=0.0)
    model.injector.to(device)
    if a.ckpt:
        # the LoRA modules have to exist before their weights can be loaded into them
        import argparse as _ap
        from train_viewpoint import add_lora
        add_lora(model.rbm, _ap.Namespace(lora_r=8, lora_layers=9))
        load_state(model, torch.load(a.ckpt, map_location=device), False)
        print(f"loaded {a.ckpt}")
    model.rbm.eval()

    rows = []
    for i, key in enumerate(keys):
        traj, kind = key.split("__", 1)
        for cam in store.cams_in(key):
            frames, meta = store.load(key, cam)
            y = meta["progress"]
            if y is None:                     # failures carry no progress target
                continue
            y = np.asarray(y, dtype=np.float64)
            pr = forward_probs(model, frames, meta, device, False, False, grad=False)
            pb = forward_probs(model, np.zeros_like(frames), meta, device, False, False,
                               grad=False)
            c = bin_centres(device)
            yr = (pr * c).sum(-1).float().cpu().numpy()
            yb = (pb * c).sum(-1).float().cpu().numpy()
            rows.append(dict(traj=traj, kind=kind, cam=cam,
                             group=store.cam_group[cam],
                             frac=frac[(traj, kind, cam)],
                             mae_real=float(np.abs(yr - y).mean()),
                             mae_black=float(np.abs(yb - y).mean()),
                             tau_real=kendall_tau(y, yr), tau_black=kendall_tau(y, yb),
                             d_pred=float(np.abs(yr - yb).mean())))
        print(f"  {i+1}/{len(keys)}", end="\r", flush=True)

    def table(title, groups):
        print(f"\n{title}\n" + "-" * 86)
        print(f"{'bin':>16}{'n':>5}{'MAE real':>10}{'MAE black':>11}{'tau real':>10}"
              f"{'tau black':>11}{'d_pred':>9}")
        for name, rs in groups:
            if not rs:
                continue
            m = lambda k: float(np.mean([r[k] for r in rs]))  # noqa: E731
            print(f"{name:>16}{len(rs):5d}{m('mae_real'):10.4f}{m('mae_black'):11.4f}"
                  f"{m('tau_real'):+10.3f}{m('tau_black'):+11.3f}{m('d_pred'):9.4f}")

    bins = []
    for lo, hi in zip(EDGES, EDGES[1:]):
        rs = [r for r in rows if lo <= r["frac"] < hi]
        bins.append((f"{lo:.1f}-{hi:.1f}%" if hi < 100 else f">{lo:.1f}%", rs))
    table("by cube visibility (% of frame area the cube covers)", bins)

    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    table("by camera group", sorted(by_group.items()))

    # A frame counter scores tau +0.97 on `success` and +0.51 on `recovery`, so the two
    # kinds ask completely different questions of the model and averaging them hides
    # the only part of the set where reading the image can pay.
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    table("by trajectory kind", sorted(by_kind.items()))
    table("all", [("all", rows)])

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
