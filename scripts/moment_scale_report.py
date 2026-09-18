"""Is the moment magnitude of the test cameras inside the training range?

THE QUESTION.  ``||m|| = ||c x d||`` with unit ``d`` is the perpendicular distance
from the world origin to the ray, so three of the six Plucker channels carry a raw
distance while the other three are unit vectors.  Nothing normalises it in
``plucker_map``.  That is harmless as long as every camera sits at roughly the same
radius -- and it stops being harmless exactly when the training cone is widened,
which is the change that made pose conditioning work in the first place.

So before crediting or blaming the geometry for anything measured outside the
training cone, check whether the pose channels were being asked to extrapolate in
MAGNITUDE rather than in orientation.  This script answers that and nothing else.
It needs no GPU, no checkpoint and no model.

    python AnyviewMeter/scripts/moment_scale_report.py outputs/pickcube_avm
    python AnyviewMeter/scripts/moment_scale_report.py outputs/pickcube_avm_wide

READING IT.  ``cover`` is the fraction of a group's per-token ``s = log||m||``
values that fall inside the [min, max] range the TRAIN split spans.  Well below 1.0
would mean the pose channels are being asked to extrapolate in magnitude, not just
in orientation, and a conditioning failure out there would say nothing about the
geometry itself.

The two ``sd`` columns are what stop the total spread from being misread.  ``in``
is the average spread of ``s`` WITHIN one frame -- rays through different pixels
pass at different distances from the origin, which is a property of perspective and
has nothing to do with which camera it is.  ``btw`` is the spread of per-clip means,
i.e. how much the CAMERA moves ``s``.  Only ``btw`` can carry viewpoint information,
and only ``btw`` can go out of distribution when the cone is widened.

MEASURED 2026-08-24 on both datasets: cover is 0.99-1.00 everywhere, and ``btw`` is
about a quarter of ``in`` (0.13-0.14 against 0.50).  So the moment magnitude is NOT
what breaks camera conditioning outside the training cone -- most of the x60 raw
spread is within-frame perspective, identical for every camera.  That is a negative
and it is the point of running this before building on the hypothesis; the NGI
decomposition earns its place by making ``s`` an explicit channel a gate can read,
not by fixing a distribution shift that turns out not to exist here.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anyviewmeter.geometry.camera import CameraParams          # noqa: E402
from anyviewmeter.geometry.plucker import plucker_map  # noqa: E402


def clip_s_values(path: str, grid: int) -> np.ndarray:
    """Per-token ``log||m||`` for one clip.  Reads metadata only, not the frames."""
    meta = json.loads(str(np.load(path, allow_pickle=False)["meta"]))
    cp = meta["cam_params"]
    res = int(meta["res"])
    cam = CameraParams(torch.tensor(cp["intrinsic_cv"], dtype=torch.float32),
                       torch.tensor(cp["extrinsic_cv"], dtype=torch.float32), res, res)
    r = plucker_map(cam, grid, grid).reshape(1, 6, grid, grid)
    return r[0, 3:6].norm(dim=0).clamp_min(1e-6).log().flatten().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="dataset root holding index.json")
    ap.add_argument("--grid", type=int, default=8, help="visual-token grid side")
    ap.add_argument("--per-group", type=int, default=60,
                    help="clips sampled per group; the cameras are per-trajectory, "
                         "so one clip per group would not span the group")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    idx = json.load(open(f"{a.data}/index.json"))["index"]
    by_group = defaultdict(list)
    for r in idx:
        by_group[(r["split"], r["group"])].append(f"{a.data}/{r['path']}")

    rng = random.Random(a.seed)
    vals, per_clip = {}, {}
    for key, paths in sorted(by_group.items()):
        pick = rng.sample(paths, min(a.per_group, len(paths)))
        clips = [clip_s_values(p, a.grid) for p in pick]
        vals[key] = np.concatenate(clips)
        # within-frame vs between-camera.  Reported separately because the raw
        # spread is dominated by the former, which no camera can change.
        per_clip[key] = (float(np.mean([c.std() for c in clips])),
                         float(np.std([c.mean() for c in clips])))

    train_key = next((k for k in vals if k[0] == "train"), None)
    if train_key is None:
        raise SystemExit(f"{a.data} has no train split to compare against")
    lo, hi = float(vals[train_key].min()), float(vals[train_key].max())

    print(f"\n{a.data}   grid {a.grid}x{a.grid}   "
          f"train s range [{lo:+.3f}, {hi:+.3f}]  "
          f"(||m|| {np.exp(lo):.3f} .. {np.exp(hi):.3f})\n")
    print(f"{'split/group':<22} {'n':>7}  {'s_mean':>7} {'s_min':>7} {'s_max':>7}  "
          f"{'sd in':>6} {'sd btw':>7}  {'cover':>6}")
    print("-" * 80)
    for key, v in sorted(vals.items()):
        cover = float(((v >= lo) & (v <= hi)).mean())
        sd_in, sd_btw = per_clip[key]
        flag = "" if cover > 0.98 else ("   <-- outside train range" if cover < 0.9 else "   <--")
        print(f"{key[0] + '/' + key[1]:<22} {v.size:>7}  {v.mean():>+7.3f} "
              f"{v.min():>+7.3f} {v.max():>+7.3f}  {sd_in:>6.3f} {sd_btw:>7.3f}  "
              f"{cover:>6.3f}{flag}")

    all_v = np.concatenate(list(vals.values()))
    sd_in = float(np.mean([p[0] for p in per_clip.values()]))
    sd_btw = float(np.mean([p[1] for p in per_clip.values()]))
    print(f"\n||m|| across every group: {np.exp(all_v).min():.3f} .. "
          f"{np.exp(all_v).max():.3f}  "
          f"(x{np.exp(all_v).max() / max(np.exp(all_v).min(), 1e-9):.1f} spread)")
    print(f"of which: within-frame sd {sd_in:.3f}, between-camera sd {sd_btw:.3f} "
          f"({sd_btw / max(sd_in, 1e-9):.1%} of the within-frame spread)")
    if sd_btw < 0.35 * sd_in and min(
            float(((v >= lo) & (v <= hi)).mean()) for v in vals.values()) > 0.95:
        print(f"\nVERDICT: moment magnitude is NOT a distribution-shift problem in this\n"
              f"dataset.  The camera moves ||m|| by {sd_btw / sd_in:.0%} of what perspective\n"
              f"does inside a single frame, and every group sits inside the training\n"
              f"range.  Do not attribute out-of-cone conditioning failures to this\n"
              f"channel; the NGI decomposition earns its place by making s explicit,\n"
              f"not by repairing a shift that is not there.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
