"""Score a checkpoint on the graded occlusion ladder, occluded vs clear.

Read as PAIRS.  Each rung has two cameras at the same elevation, the same |azimuth|, the
same radius and FOV -- mirrored across y so that one has the arm in the line of sight and
the other does not.  Apparent cube size is matched within a pair (measured: 0.247 vs
0.251 percent of frame on the mild rung, 0.102 vs 0.096 on the severe one), so the
within-pair difference is attributable to occlusion rather than to the target shrinking.

The absolute numbers across rungs are NOT a clean comparison -- higher rungs are further
round and further down, so they differ in more than occlusion.  The occluded-minus-clear
delta is the quantity to read.
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
from train_viewpoint import (add_lora, bin_centres, forward_probs,  # noqa: E402
                             kendall_tau, load_state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R1", choices=["R1", "P1"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--data", default="/home/yuang/ws_jepa/multicam_ws/outputs/occlusion_probe")
    ap.add_argument("--out", default="")
    ap.add_argument("--kinds", default="all", choices=["all", "labelled"],
                    help="'labelled' drops the failure trajectories, matching the "
                         "fail-policy=mask convention the main eval scores under")
    a = ap.parse_args()
    root = "/home/yuang/ws_jepa/multicam_ws/outputs"
    a.ckpt = a.ckpt or f"{root}/viewpoint/{a.arm}/best.pt"
    a.out = a.out or f"{a.data}/scores_{a.arm}.json"
    use_pose = a.arm == "P1"
    device = torch.device("cuda")

    idx = json.load(open(f"{a.data}/index.json"))
    rows = idx["index"]
    if a.kinds == "labelled":
        rows = [r for r in rows if not r["kind"].startswith("failure")]

    bundle = load_robometer(device=device)
    model = PoseConditionedRobometer(bundle, injector_name="cross_attn", layers=[27, 31, 35],
                                     patch_inject=False, pose_dropout=0.0, gate_init=0.05)
    model.injector.to(device)
    if use_pose:
        model.attach()
    add_lora(model.rbm, argparse.Namespace(lora_r=8, lora_layers=9))
    load_state(model, torch.load(a.ckpt, map_location=device), use_pose)
    model.rbm.eval()
    if use_pose:
        model.injector.eval()
    print(f"loaded {a.ckpt}  ({len(rows)} clips)")

    per = defaultdict(lambda: dict(ae=[], tau=[], blind=[], cube=[], preds={}))
    for n, r in enumerate(rows):
        z = np.load(f"{a.data}/{r['path']}", allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        y = np.asarray(meta["progress"], dtype=float)
        p = forward_probs(model, z["frames"], meta, device, use_pose, False, grad=False)
        yhat = (p * bin_centres(device)).sum(-1).float().cpu().numpy()
        d = per[r["cam"]]
        d["ae"].append(float(np.abs(yhat - y).mean()))
        d["tau"].append(kendall_tau(y, yhat))
        d["blind"].append(r["blind_frac"])
        d["cube"].append(r["mean_cube_frac"])
        d["preds"][(r["traj"], r["kind"])] = yhat
        if n % 40 == 0:
            print(f"  {n}/{len(rows)}", end="\r", flush=True)

    bat = {c["name"]: c for c in idx["battery"]}
    print(f"\n{'camera':20s}{'az':>5}{'el':>4}{'blind%':>8}{'cube%':>7}"
          f"{'MAE':>8}{'tau':>8}{'n':>5}")
    print("-" * 63)
    out = {}
    for name in [c["name"] for c in idx["battery"]]:
        d = per.get(name)
        if not d:
            continue
        out[name] = dict(mae=float(np.mean(d["ae"])), tau=float(np.mean(d["tau"])),
                         blind=float(np.mean(d["blind"])), cube=float(np.mean(d["cube"])),
                         n=len(d["ae"]))
        b = bat[name]
        print(f"{name:20s}{b['az']:5.0f}{b['el']:4.0f}{out[name]['blind'] * 100:8.1f}"
              f"{out[name]['cube'] * 100:7.3f}{out[name]['mae']:8.4f}"
              f"{out[name]['tau']:+8.3f}{out[name]['n']:5d}")

    print(f"\n{'rung':12s}{'Δblind%':>9}{'MAE clear':>11}{'MAE occl':>10}{'ΔMAE':>9}"
          f"{'tau clear':>11}{'tau occl':>10}{'Δtau':>9}")
    print("-" * 81)
    for label, *_ in idx["ladder"]:
        c, o = out.get(f"{label}_clear"), out.get(f"{label}_occluded")
        if not (c and o):
            continue
        print(f"{label:12s}{(o['blind'] - c['blind']) * 100:9.1f}{c['mae']:11.4f}"
              f"{o['mae']:10.4f}{o['mae'] - c['mae']:+9.4f}{c['tau']:+11.3f}"
              f"{o['tau']:+10.3f}{o['tau'] - c['tau']:+9.3f}")

    # per-clip correlation: does the error track how much of the cube was actually hidden?
    blind = np.concatenate([d["blind"] for d in per.values()])
    ae = np.concatenate([d["ae"] for d in per.values()])
    tau = np.concatenate([d["tau"] for d in per.values()])
    print(f"\nper-clip correlation with the measured blind fraction ({len(ae)} clips):"
          f"\n  MAE  r = {np.corrcoef(blind, ae)[0, 1]:+.3f}"
          f"\n  tau  r = {np.corrcoef(blind, tau)[0, 1]:+.3f}")

    json.dump(dict(per_camera=out, arm=a.arm, ckpt=a.ckpt, kinds=a.kinds),
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
