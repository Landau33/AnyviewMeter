"""Dump every camera view of one trajectory to PNG so the battery can be eyeballed.

The numbers already say six of the twenty-one test cameras never see the cube, but a
number does not show you WHY -- whether the camera points at empty table, sits inside
the robot, or watches the arm swallow the cube halfway through the episode.  So each
frame is annotated with what the geometry says should be there:

  green crosshair   where the cube's world position projects to through the clip's own
                    stored ``extrinsic_cv`` / analytic ``K``.  A crosshair sitting on
                    bare table with no cube under it means the cube is occluded, not
                    out of frame -- the two look identical in a visibility number and
                    call for different fixes.
  red crosshair     the same, when the projection falls outside the image; drawn
                    clamped to the border, so the direction still reads.

``cube%`` / ``arm%`` in each caption are the renderer's own per-frame segmentation
fractions, not re-derived here -- if a caption disagrees with the picture, the bug is
in the renderer's bookkeeping and that is worth knowing on its own.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

PAD = 16          # caption strip under each tile


def load(root: str, rel: str):
    z = np.load(f"{root}/{rel}", allow_pickle=False)
    return z["frames"], json.loads(str(z["meta"]))


def project(meta, fi: int):
    K = np.asarray(meta["cam_params"]["intrinsic_cv"], dtype=float)
    E = np.asarray(meta["cam_params"]["extrinsic_cv"], dtype=float)
    X = np.asarray(list(meta["states"][fi]["cube"][:3]) + [1.0], dtype=float)
    Xc = E @ X
    if Xc[2] <= 1e-6:                       # behind the camera
        return None, False
    uv = K @ Xc
    return uv[:2] / uv[2], True


def tile(frame, meta, fi, res):
    img = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(img)
    uv, front = project(meta, fi)
    if uv is not None:
        u, v = float(uv[0]), float(uv[1])
        inside = 0 <= u < res and 0 <= v < res
        col = (0, 255, 0) if inside else (255, 40, 40)
        u, v = min(max(u, 2), res - 3), min(max(v, 2), res - 3)
        d.line([(u - 10, v), (u + 10, v)], fill=col, width=1)
        d.line([(u, v - 10), (u, v + 10)], fill=col, width=1)
    return img


def sheet(frames, meta, steps, res, caption):
    w = res * len(steps)
    out = Image.new("RGB", (w, res + PAD), (18, 18, 18))
    for j, fi in enumerate(steps):
        out.paste(tile(frames[fi], meta, fi, res), (j * res, 0))
    ImageDraw.Draw(out).text((4, res + 3), caption, fill=(235, 235, 235))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/yuang/ws_jepa/multicam_ws/outputs/pickcube_avm")
    ap.add_argument("--out", default="/home/yuang/ws_jepa/multicam_ws/outputs/camera_views")
    ap.add_argument("--split", default="test")
    ap.add_argument("--traj", default="", help="default: first success trajectory in the split")
    ap.add_argument("--frames", type=int, default=6)
    a = ap.parse_args()

    idx = json.load(open(f"{a.data}/index.json"))
    battery = {b["name"]: b for b in idx["battery"]}
    rows = [r for r in idx["index"] if r["split"] == a.split and r["kind"] == "success"]
    traj = a.traj or sorted({r["traj"] for r in rows})[0]
    rows = sorted([r for r in rows if r["traj"] == traj], key=lambda r: r["cam"])
    if not rows:
        raise SystemExit(f"no success clips for {traj} in split {a.split}")
    os.makedirs(a.out, exist_ok=True)

    f0, m0 = load(a.data, rows[0]["path"])
    n, res = len(f0), f0.shape[1]
    steps = list(np.linspace(0, n - 1, a.frames).astype(int))
    print(f"{traj} ({a.split}): {len(rows)} cameras x {n} frames @ {res}px  "
          f"-> frames {steps}")

    strips, dead = [], []
    for r in rows:
        frames, meta = load(a.data, r["path"])
        b = battery.get(r["cam"], {})
        cube = float(np.mean(meta["cube_frac"])) * 100
        arm = float(np.mean(meta["arm_frac"])) * 100
        std = float(frames.reshape(len(frames), -1).std(1).mean())
        flag = "  <<< CUBE NEVER VISIBLE" if cube < 0.1 else ""
        cap = (f"{r['cam']}  [{r['group']}]  el={b.get('el', 0):.0f} az={b.get('az', 0):.0f} "
               f"r={b.get('radius', 0):.2f} fov={np.rad2deg(b.get('fov', 0)):.0f}deg   "
               f"cube={cube:.3f}%  arm={arm:.1f}%  pxstd={std:.1f}{flag}")
        s = sheet(frames, meta, steps, res, cap)
        s.save(f"{a.out}/{r['group']}__{r['cam']}.png")
        strips.append(s)
        print(f"  {cap}")
        if cube < 0.1:
            dead.append(r["cam"])

    contact = Image.new("RGB", (strips[0].width, sum(s.height for s in strips)),
                        (18, 18, 18))
    y = 0
    for s in strips:
        contact.paste(s, (0, y))
        y += s.height
    contact.save(f"{a.out}/ALL_{a.split}_{traj}.png")

    print(f"\n{len(dead)}/{len(rows)} cameras never show the cube: {', '.join(dead)}")
    print(f"wrote {len(rows)} per-camera PNGs + ALL_{a.split}_{traj}.png -> {a.out}")


if __name__ == "__main__":
    main()
