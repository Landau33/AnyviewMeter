"""Two arms' progress curves side by side, as the viewpoint walks out of the trained cone.

WHY THIS PICTURE.  The tables say the conditioned arm wins tau on every camera group
while sometimes LOSING MAE, which reads as a contradiction until you see the curves.
What is actually happening is that the unconditioned arm retreats toward the frame-
counter prior when the viewpoint gets hard -- and because the labels are concentrated
in the middle, hugging that prior scores a low MAE while destroying the ordering.  The
grey dashed line on every panel is that prior, so "it collapsed onto the baseline" is
something you can see rather than infer from a spread statistic.

TWO FIGURES:

  curves.png   rows = trajectory kind, columns = az_seen / az_edge / az_far.  One
               camera per panel, so a single physical trajectory is followed as the
               camera walks 0 -> 60 degrees outside the trained cone.
  spread.png   one trajectory, all four cameras of a group at once, drawn as a band
               between the min and max prediction.  This is S_view as a picture: a
               narrow band means the model gives the same answer from every viewpoint,
               which is the property the whole experiment is about.

Both arms are run one at a time and their predictions cached, because two 4B backbones
do not fit on a 12 GB card at once.
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

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

# Without this every CJK label renders as a row of tofu boxes and the figure is useless
# to read.  "Noto Sans CJK JP" is the family matplotlib has indexed here; it carries the
# same Han glyphs as the SC face.  The minus sign has no CJK glyph, hence unicode_minus.
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback",
                                          "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from anyviewmeter.models.robometer_backbone import load_robometer     # noqa: E402
from train_viewpoint import (ClipStore, add_lora, bin_centres, build_model,  # noqa: E402
                             forward_probs, kendall_tau, load_state)

# dataviz reference palette, categorical slots 1-3, validated light mode:
# CVD worst adjacent dE 9.2 (deutan), normal-vision 27.6.  Slot 3 carries a contrast
# WARN against the surface, which is why every series is direct-labelled.
TRUE, ARM_A, ARM_B = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdcd8", "#fcfcfb"


def true_curve(store, key: str, cam: str) -> np.ndarray:
    """The state-derived label, INCLUDING for failures.

    ``fail_policy=mask`` hands back ``progress=None`` for a failed episode because it
    carries no training target -- but the curve still exists in the clip's own metadata,
    and a dropped cube is the one shape a frame counter cannot produce, so it is exactly
    what this figure is for.  Read it from the npz rather than from the masked meta,
    where ``np.asarray(None)`` silently plots nothing.
    """
    m = json.loads(str(np.load(store.groups[key][cam], allow_pickle=False)["meta"]))
    return np.asarray(m["progress"], dtype=float)


def frame_counter_prior(root: str, n_frames: int) -> np.ndarray:
    """Mean label per timestep over labelled TRAIN trajectories -- the pixel-blind line.

    Not a straight ramp: the state-derived labels bunch up on the approach-and-grasp
    plateau, so the prior a model collapses onto has that shape too.  Drawing the true
    prior rather than a diagonal is the difference between "it ignored the image" being
    visible and being merely plausible.
    """
    idx = json.load(open(f"{root}/index.json"))
    seen, curves = set(), []
    for r in idx["index"]:
        if r["split"] != "train" or r["kind"].startswith("failure") or r["traj"] in seen:
            continue
        seen.add(r["traj"])
        m = json.loads(str(np.load(f"{root}/{r['path']}", allow_pickle=False)["meta"]))
        y = np.asarray(m["progress"], dtype=float)
        if len(y) == n_frames:
            curves.append(y)
    return np.mean(curves, axis=0) if curves else np.linspace(0, 1, n_frames)


def predict(arm: str, ckpt: str, jobs, data: str, device, a) -> dict:
    """Run one arm over `jobs` = [(key, cam), ...] and return {(key, cam): curve}.

    One arm per call and the bundle dropped on the way out: two 4B backbones do not
    coexist on a 12 GB card, so the caller must not hold onto the model.
    """
    args = argparse.Namespace(
        xattn_layers=a.xattn_layers, control_dim=256, bottleneck=256,
        pose_dropout=0.0, gate_init=a.gate_init, gate_open=None, pose_null=False,
        inject=("patch" if arm == "P1" else "xattn"), lora_r=8, lora_layers=9)
    bundle = load_robometer(device=device)
    model, use_pose = build_model(arm, bundle, device, args)
    add_lora(model.rbm, args)
    load_state(model, torch.load(ckpt, map_location=device), use_pose)
    model.rbm.eval()
    if use_pose:
        model.injector.eval()
    print(f"  loaded {arm} <- {ckpt}", flush=True)

    store = ClipStore(data, "test")
    out = {}
    for i, (key, cam) in enumerate(jobs):
        frames, meta = store.load(key, cam)
        p = forward_probs(model, frames, meta, device, use_pose, False, grad=False)
        out[(key, cam)] = (p * bin_centres(device)).sum(-1).float().cpu().numpy()
        print(f"    [{i + 1}/{len(jobs)}] {key} {cam}", end="\r", flush=True)
    print()
    del model, bundle
    torch.cuda.empty_cache()
    return out


def _panel(ax, prior, show_x, show_y):
    ax.set_facecolor(SURFACE)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    # labelbottom/labelleft and NOT set_xticklabels([]): on shared axes the latter
    # replaces the shared formatter, which blanks the numbers on every panel including
    # the ones that were supposed to keep them
    ax.tick_params(colors=MUTED, labelsize=8, length=3,
                   labelbottom=show_x, labelleft=show_y)
    # the pixel-blind line, drawn first and recessive: it is a reference, not a series
    ax.plot(prior, color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)


def figure_curves(store, preds_a, preds_b, prior, rows, cols, names, out):
    """rows = [(key, label)], cols = [(group, cam, label)]."""
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(4.0 * len(cols), 2.7 * len(rows)),
                             sharex=True, sharey=True, facecolor=SURFACE)
    axes = np.atleast_2d(axes)
    for r, (key, row_label) in enumerate(rows):
        for c, (_group, cam, col_label) in enumerate(cols):
            ax = axes[r, c]
            _panel(ax, prior, r == len(rows) - 1, c == 0)
            y = true_curve(store, key, cam)
            ax.plot(y, color=TRUE, linewidth=2.4, zorder=4)
            ax.plot(preds_a[(key, cam)], color=ARM_A, linewidth=2.0, zorder=3)
            ax.plot(preds_b[(key, cam)], color=ARM_B, linewidth=2.0, zorder=3)
            if r == 0:
                ax.set_title(col_label, color=INK, fontsize=10, pad=8)
            if c == 0:
                ax.set_ylabel(row_label, color=INK, fontsize=9.5)
            # Direct labels once, on the LAST column: a number on every point is noise,
            # and slot 3's contrast WARN is what obliges labels rather than colour alone.
            # Last column and not first because the text is drawn outside the axes, and
            # only the right edge has margin reserved for it -- on panel 0 it would land
            # on top of its neighbour.
            if r == 0 and c == len(cols) - 1:
                ends = [(y[-1], TRUE, "真实标签"),
                        (preds_a[(key, cam)][-1], ARM_A, names[0]),
                        (preds_b[(key, cam)][-1], ARM_B, names[1]),
                        (prior[-1], MUTED, "帧计数先验")]
                # nudge apart in draw order: four curves can end within a hair of each
                # other, and overlapping labels are worse than no labels
                ends.sort()
                placed = []
                for yv, col, txt in ends:
                    # A single clamp, not a `while`.  The obvious loop form
                    #   while yv - placed[-1] < gap: yv = placed[-1] + gap
                    # reassigns the SAME value every pass, and float rounding can leave
                    # the difference a hair under gap forever -- an infinite loop that
                    # spins at 100% CPU with no output, which is exactly what it did.
                    if placed and yv < placed[-1] + 0.075:
                        yv = placed[-1] + 0.075
                    placed.append(yv)
                    ax.annotate(txt, xy=(len(y) - 1, yv), xytext=(6, 0),
                                textcoords="offset points", color=col, fontsize=8,
                                va="center", fontweight="medium", annotation_clip=False)
    fig.supxlabel("帧序号", color=MUTED, fontsize=9)
    fig.supylabel("进度", color=MUTED, fontsize=9)
    fig.suptitle("同一条轨迹，相机逐步走出训练锥", color=INK, fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 0.88, 0.97))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def figure_spread(store, preds_a, preds_b, prior, key, cols, names, out):
    """One trajectory; per group, each arm's four cameras as a min-max band."""
    fig, axes = plt.subplots(2, len(cols), figsize=(4.0 * len(cols), 5.4),
                             sharex=True, sharey=True, facecolor=SURFACE)
    axes = np.atleast_2d(axes)
    y = true_curve(store, key, cols[0][1][0])
    for r, (preds, colour, name) in enumerate(((preds_a, ARM_A, names[0]),
                                               (preds_b, ARM_B, names[1]))):
        for c, (_group, cams, col_label) in enumerate(cols):
            ax = axes[r, c]
            _panel(ax, prior, r == 1, c == 0)
            M = np.stack([preds[(key, cam)] for cam in cams])
            ax.fill_between(np.arange(M.shape[1]), M.min(0), M.max(0),
                            color=colour, alpha=0.22, linewidth=0, zorder=2)
            ax.plot(M.mean(0), color=colour, linewidth=2.0, zorder=3)
            ax.plot(y, color=TRUE, linewidth=2.0, zorder=4)
            spread = float(M.std(axis=0).mean())
            ax.annotate(f"S_view {spread:.3f}", xy=(0.03, 0.93), xycoords="axes fraction",
                        color=MUTED, fontsize=8.5, va="top")
            if r == 0:
                ax.set_title(col_label, color=INK, fontsize=10, pad=8)
            if c == 0:
                ax.set_ylabel(name, color=colour, fontsize=10, fontweight="medium")
    fig.supxlabel("帧序号", color=MUTED, fontsize=9)
    fig.suptitle("同一物理状态，四台相机的预测带（带越窄＝视角越一致）",
                 color=INK, fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def figure_scatter(store, preds_a, preds_b, keys, cols, names, out):
    """Predicted vs true over many trajectories -- where the tau gap is actually visible.

    The per-clip curves are the honest picture of what a model does, but two curves that
    differ by 0.26 in Kendall tau can look similar side by side: tau is about ORDER, and
    order is hard to see in a line that mostly goes up.  Here it is the shape of the
    cloud.  A model that ranks well hugs the diagonal; one that has retreated toward a
    constant flattens into a horizontal band no matter how good its MAE looks.
    """
    fig, axes = plt.subplots(2, len(cols), figsize=(3.5 * len(cols), 7.0),
                             sharex=True, sharey=True, facecolor=SURFACE)
    axes = np.atleast_2d(axes)
    for r, (preds, colour, name) in enumerate(((preds_a, ARM_A, names[0]),
                                               (preds_b, ARM_B, names[1]))):
        for c, (_group, cams, col_label) in enumerate(cols):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=8, length=3,
                           labelbottom=r == 1, labelleft=c == 0)
            xs, ys, taus = [], [], []
            for key in keys:
                for cam in cams:
                    if (key, cam) not in preds:
                        continue
                    t, pr = true_curve(store, key, cam), preds[(key, cam)]
                    xs.append(t)
                    ys.append(pr)
                    taus.append(kendall_tau(t, pr))
            X, Y = np.concatenate(xs), np.concatenate(ys)
            ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2,
                    linestyle=(0, (4, 3)), zorder=1)
            ax.scatter(X, Y, s=7, color=colour, alpha=0.18, linewidths=0, zorder=2)
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.03, 1.03)
            ax.annotate(f"tau {np.mean(taus):+.3f}", xy=(0.04, 0.95),
                        xycoords="axes fraction", color=INK, fontsize=9.5,
                        va="top", fontweight="medium")
            if r == 0:
                ax.set_title(col_label, color=INK, fontsize=10, pad=8)
            if c == 0:
                ax.set_ylabel(name, color=colour, fontsize=10, fontweight="medium")
    fig.supxlabel("真实进度", color=MUTED, fontsize=9)
    fig.suptitle(f"预测 vs 真实（{len(keys)} 条轨迹 × 每组 4 台相机，虚线为理想对角）",
                 color=INK, fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    root = "/home/yuang/ws_jepa/multicam_ws"
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{root}/outputs/pickcube_avm_wide")
    ap.add_argument("--ckpt-a", default=f"{root}/outputs/gate_open/R1_wide/best.pt")
    ap.add_argument("--ckpt-b", default=f"{root}/outputs/gate_open/P1_patch_wide/best.pt")
    ap.add_argument("--arm-a", default="R1", choices=["R1", "P1"])
    ap.add_argument("--arm-b", default="P1", choices=["R1", "P1"])
    ap.add_argument("--name-a", default="R1（无位姿）")
    ap.add_argument("--name-b", default="P1 tier A（位姿）")
    ap.add_argument("--groups", default="az_seen,az_edge,az_far")
    ap.add_argument("--kinds", default="success,recovery,failure_dropped")
    ap.add_argument("--xattn-layers", default="27,31,35")
    ap.add_argument("--gate-init", type=float, default=0.05)
    ap.add_argument("--reuse-preds", action="store_true",
                    help="plot from a cached predictions.npz instead of re-running both "
                         "backbones.  Refuses a cache that does not cover this run.")
    ap.add_argument("--scatter-trajs", type=int, default=12,
                    help="trajectories behind the predicted-vs-true clouds.  Each one "
                         "costs 4 cameras x 3 groups x 2 arms of forwards.")
    ap.add_argument("--out", default=f"{root}/outputs/curve_compare")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda")

    store = ClipStore(a.data, "test")
    groups = a.groups.split(",")
    by_group = defaultdict(list)
    for cam, g in store.cam_group.items():
        by_group[g].append(cam)
    for g in groups:
        by_group[g].sort()
        if not by_group[g]:
            raise SystemExit(f"group {g!r} not in this battery (has {sorted(by_group)})")

    # one trajectory per kind, and the SAME trajectory across all columns: the columns
    # are meant to isolate the camera, so anything else must be held fixed
    rows = []
    for kind in a.kinds.split(","):
        keys = [k for k in store.keys if store.kind[k] == kind]
        if keys:
            rows.append((keys[0], {"success": "成功", "recovery": "recovery（回退重抓）",
                                   "failure_dropped": "failure（掉落）"}.get(kind, kind)))
    if not rows:
        raise SystemExit(f"no trajectories of kinds {a.kinds} in {a.data}")

    cols = [(g, by_group[g][0],
             f"{g}  ({by_group[g][0]})") for g in groups]
    spread_cols = [(g, by_group[g], g) for g in groups]
    spread_key = rows[0][0]

    scatter_keys = store.keys[:a.scatter_trajs]
    jobs = sorted({(k, cam) for k, _ in rows for _, cam, _ in cols}
                  | {(spread_key, cam) for _, cams, _ in spread_cols for cam in cams}
                  | {(k, cam) for k in scatter_keys
                     for _, cams, _ in spread_cols for cam in cams})
    print(f"{len(jobs)} clips x 2 arms")

    cache = f"{a.out}/predictions.npz"
    if a.reuse_preds and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        preds_a, preds_b = {}, {}
        for name in z.files:
            arm, key, cam = name.split("|")
            (preds_a if arm == a.arm_a else preds_b)[(key, cam)] = z[name]
        missing = [j for j in jobs if j not in preds_a or j not in preds_b]
        if missing:
            raise SystemExit(f"{cache} is missing {len(missing)} of the {len(jobs)} "
                             f"clips this run needs, e.g. {missing[:2]} -- drop "
                             f"--reuse-preds rather than plotting a partial cache")
        print(f"reused {cache}")
    else:
        preds_a = predict(a.arm_a, a.ckpt_a, jobs, a.data, device, a)
        preds_b = predict(a.arm_b, a.ckpt_b, jobs, a.data, device, a)
        # Written BEFORE the figures: twelve minutes of forwards should not be lost to a
        # plotting bug, and iterating on a figure should not re-run two 4B backbones.
        np.savez(cache, **{f"{arm}|{k}|{cam}": v
                           for arm, P in ((a.arm_a, preds_a), (a.arm_b, preds_b))
                           for (k, cam), v in P.items()})
        print(f"wrote {cache}")

    n = len(store.load(rows[0][0], cols[0][1])[0])
    prior = frame_counter_prior(a.data, n)

    names = (a.name_a, a.name_b)
    figure_curves(store, preds_a, preds_b, prior, rows, cols, names,
                  f"{a.out}/curves.png")
    figure_spread(store, preds_a, preds_b, prior, spread_key, spread_cols, names,
                  f"{a.out}/spread.png")
    figure_scatter(store, preds_a, preds_b, scatter_keys, spread_cols, names,
                   f"{a.out}/scatter.png")


if __name__ == "__main__":
    main()
