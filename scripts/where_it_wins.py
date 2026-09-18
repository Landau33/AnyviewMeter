"""Where does the conditioned arm actually beat the baselines -- not by how much on average.

A per-group table says P1 wins everywhere and hides WHY.  Four questions instead, each
aimed at a specific claim rather than at a mean:

  BY KIND        success / recovery / failure_*.  On `success` a time ramp is nearly
                 perfect, so the arms cannot separate there; the interpretable kinds are
                 the ones where progress and elapsed time disagree.

  FOLLOWS THE DIP  the sharpest of the four.  On `recovery` the label goes up, DOWN, up.
                 Correlating the per-frame label DELTA with the predicted delta asks
                 whether the model sees the regression: a model riding the clock always
                 predicts "up" and scores ~0 here no matter how good its MAE looks.

  BY PHASE       MAE over the early / middle / late third of each clip.  "Commits at the
                 end" is a claim about the late third specifically.

  TERMINAL       how close the last frames of a SUCCESSFUL clip get to 1.0.  A hedging
                 model stops around 0.6 and buys MAE with it; that shows up here and
                 nowhere else.
"""
from __future__ import annotations
import argparse, json, os, sys, collections
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import train_viewpoint as T


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(HERE))
    ap.add_argument("--data", default=f"{root}/outputs/pickcube_avm_wide600")
    ap.add_argument("--arms", default=f"R0:R0:,R1:R1:{root}/outputs/gate_open/R1_pick600_R1/best.pt,"
                                     f"P1:P1:{root}/outputs/gate_open/P1_pick600_patchA/best.pt")
    ap.add_argument("--groups", default="az_seen,az_far")
    ap.add_argument("--trajs", type=int, default=70)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--out", default=f"{root}/outputs/where_it_wins2.json")
    a = ap.parse_args()
    a.workspace_centre = json.load(open(f"{a.data}/index.json"))["battery"][0]["target"]

    dev = torch.device("cuda")
    store = T.ClipStore(a.data, "test", "mask")
    groups = a.groups.split(",")
    keys = store.keys[:a.trajs] if a.trajs else store.keys
    base = dict(inject="patch", xattn_layers="27,31,35", control_dim=256, bottleneck=256,
                gate_init=0.05, gate_open=None, pose_dropout=0.0, pose_null=False,
                pose_source="true", encoder_init="zero",
                workspace_centre=a.workspace_centre, lora_r=8, lora_layers=9)
    rows = []
    for spec in a.arms.split(","):
        name, arm, ckpt = spec.split(":", 2)
        aa = argparse.Namespace(arm=arm, **base)
        m, use_pose = T.build_model(arm, T.load_robometer(device=dev), dev, aa)
        if arm != "R0":
            T.add_lora(m.rbm, aa)
        m.pose_head = None; m.pose_random = False; m.workspace_centre = a.workspace_centre
        if ckpt:
            T.load_state(m, torch.load(ckpt, map_location=dev), use_pose)
        m.rbm.eval()
        n = 0
        for key in keys:
            for cam in store.cams_in(key):
                if store.cam_group[cam] not in groups:
                    continue
                frames, meta = store.load(key, cam)
                if meta["progress"] is None:
                    # A failed episode carries NO training target under fail_policy=mask,
                    # but the renderer did store its state-derived curve -- that is the
                    # whole reason failures were rendered.  Read it back from the file so
                    # the kinds where progress and elapsed time disagree most can be
                    # scored at all; masking is a TRAINING policy, not an eval one.
                    raw = json.loads(str(np.load(store.groups[key][cam],
                                                 allow_pickle=False)["meta"]))
                    if raw.get("progress") is None:
                        continue
                    meta = dict(meta, progress=raw["progress"])
                y = np.asarray(meta["progress"], float)
                sel = np.linspace(0, len(y) - 1, a.frames).astype(int)
                p = T.forward_probs(m, frames[sel], meta, dev, use_pose, False, grad=False)
                yh = (p * T.bin_centres(dev)).sum(-1).float().cpu().numpy()
                rows.append(dict(arm=name, kind=store.kind[key], group=store.cam_group[cam],
                                 traj=key, cam=cam, y=y[sel].tolist(), yhat=yh.tolist()))
                n += 1
        print(f"  {name}: {n} clips", flush=True)
        del m; torch.cuda.empty_cache()
    json.dump(rows, open(a.out, "w"))
    print(f"wrote {a.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
