"""Score a checkpoint per camera group AND per trajectory kind, against the shortcut.

WHY.  On this data a predictor that ignores pixels entirely and returns the mean label
per frame index scores tau 0.82-0.86 -- higher than most arms we have trained.  Any
headline tau is therefore partly a statement about how well the model tracks TIME, and
the arms are only separable on the trajectories where time is a bad predictor.

  success            the arm goes for the cube and gets it.  Progress and time are
                     nearly the same signal, so a frame counter is near-perfect and
                     nothing here distinguishes two models' perception.
  recovery           forward, back, forward again.  Time keeps rising while progress
                     dips, so the frame counter is beaten by construction.
  failure_dropped    progress climbs and then collapses when the cube is released.
  failure_missed     the gripper closes on nothing; progress rises then falls back.

The last three are the interpretable ones.  This script reports every arm against the
frame counter ON THE SAME SUBSET, so "better than the shortcut" is a claim you can read
off the table rather than infer.

Failures carry no training target under fail_policy=mask but their state-derived curve
exists in the clip metadata, so they can be SCORED -- that is the whole reason they were
rendered.
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

from anyviewmeter.models.robometer_backbone import load_robometer      # noqa: E402
from train_viewpoint import (ClipStore, add_lora, bin_centres, build_model,  # noqa: E402
                             forward_probs, kendall_tau, load_state)
from viz_curve_compare import true_curve                               # noqa: E402


def frame_counter_by_kind(store: ClipStore, root: str) -> dict:
    """Mean TRAIN label per frame index, then scored per kind on the test split.

    One prior for all kinds -- that is the point.  A frame counter cannot know which
    kind it is looking at, so a kind whose curve disagrees with the average shape is
    exactly where it fails and where a perception result becomes readable.
    """
    idx = json.load(open(f"{root}/index.json"))
    seen, curves = set(), []
    for r in idx["index"]:
        if r["split"] != "train" or r["kind"].startswith("failure") or r["traj"] in seen:
            continue
        seen.add(r["traj"])
        m = json.loads(str(np.load(f"{root}/{r['path']}", allow_pickle=False)["meta"]))
        curves.append(np.asarray(m["progress"], dtype=float))
    prior = np.mean(curves, axis=0)

    out = defaultdict(lambda: dict(ae=[], tau=[]))
    for key in store.keys:
        cam = store.cams_in(key)[0]          # kind-level, so one camera is enough
        y = true_curve(store, key, cam)
        d = out[store.kind[key]]
        d["ae"].append(np.abs(prior[:len(y)] - y))
        d["tau"].append(kendall_tau(y, prior[:len(y)]))
    return {k: dict(mae=float(np.mean(np.concatenate(v["ae"]))),
                    tau=float(np.mean(v["tau"])), n=len(v["tau"]))
            for k, v in out.items()}


@torch.no_grad()
def score(arm: str, ckpt: str, store: ClipStore, device, a) -> dict:
    args = argparse.Namespace(
        xattn_layers=a.xattn_layers, control_dim=256, bottleneck=256, pose_dropout=0.0,
        gate_init=a.gate_init, gate_open=None, pose_null=False,
        inject=a.inject if arm == "P1" else "xattn", lora_r=8, lora_layers=9)
    bundle = load_robometer(device=device)
    model, use_pose = build_model(arm, bundle, device, args)
    add_lora(model.rbm, args)
    load_state(model, torch.load(ckpt, map_location=device), use_pose)
    model.rbm.eval()
    if use_pose:
        model.injector.eval()

    per = defaultdict(lambda: dict(ae=[], tau=[]))
    keys = store.keys[:a.trajs] if a.trajs else store.keys
    todo = [(k, c) for k in keys for c in store.cams_in(k)
            if store.cam_group[c] in a.groups.split(",")]
    for n, (key, cam) in enumerate(todo):
        frames, meta = store.load(key, cam)
        p = forward_probs(model, frames, meta, device, use_pose, False, grad=False)
        yhat = (p * bin_centres(device)).sum(-1).float().cpu().numpy()
        y = true_curve(store, key, cam)
        kind, group = store.kind[key], store.cam_group[cam]
        for bucket in (kind, f"{group}|{kind}"):
            per[bucket]["ae"].append(np.abs(yhat - y))
            per[bucket]["tau"].append(kendall_tau(y, yhat))
        if n % 40 == 0:
            print(f"    {n}/{len(todo)}", end="\r", flush=True)
    print()
    del model, bundle
    torch.cuda.empty_cache()
    return {k: dict(mae=float(np.mean(np.concatenate(v["ae"]))),
                    tau=float(np.mean(v["tau"])), n=len(v["tau"]))
            for k, v in per.items()}


def main():
    root = "/home/yuang/ws_jepa/multicam_ws"
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{root}/outputs/pickcube_avm_wide")
    ap.add_argument("--arms", default=f"R1:R1:{root}/outputs/gate_open/R1_wide/best.pt,"
                                      f"P1:P1:{root}/outputs/gate_open/P1_patch_wide/best.pt",
                    help="name:arm:ckpt triples, comma separated")
    ap.add_argument("--inject", default="patch")
    ap.add_argument("--groups", default="az_seen,az_edge,az_far")
    ap.add_argument("--trajs", type=int, default=0, help="0 = all")
    ap.add_argument("--xattn-layers", default="27,31,35")
    ap.add_argument("--gate-init", type=float, default=0.05)
    ap.add_argument("--out", default=f"{root}/outputs/by_kind.json")
    a = ap.parse_args()
    device = torch.device("cuda")

    store = ClipStore(a.data, "test")
    fc = frame_counter_by_kind(store, a.data)
    results = {"frame_counter": fc}

    for spec in a.arms.split(","):
        name, arm, ckpt = spec.split(":", 2)
        print(f"### {name}")
        results[name] = score(arm, ckpt, store, device, a)

    names = [s.split(":")[0] for s in a.arms.split(",")]
    kinds = sorted(fc)
    print(f"\n{'kind':<18}{'n':>4}{'帧计数 tau':>12}" +
          "".join(f"{n + ' tau':>14}" for n in names))
    print("-" * (34 + 14 * len(names)))
    for k in kinds:
        row = f"{k:<18}{fc[k]['n']:>4}{fc[k]['tau']:>+12.3f}"
        for n in names:
            row += f"{results[n].get(k, {}).get('tau', float('nan')):>+14.3f}"
        print(row)

    print(f"\n按相机组 x 轨迹类型的 tau（只列时间捷径弱的类型）")
    hard = [k for k in kinds if k != "success"]
    for g in a.groups.split(","):
        print(f"\n  {g}")
        for k in hard:
            b = f"{g}|{k}"
            row = f"    {k:<18}" + "".join(
                f"{results[n].get(b, {}).get('tau', float('nan')):>+14.3f}" for n in names)
            print(row)

    json.dump(results, open(a.out, "w"), indent=2, default=float)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
