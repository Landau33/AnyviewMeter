"""Reproduce the phase-A Robometer-4B numbers, through the code path our experiments use.

WHY THIS EXISTS.  Across four trained arms, every model scored far below a predictor
that ignores pixels entirely, and blacking out the image made all of them BETTER.  That
has two possible causes and they call for opposite fixes:

  (a) the backbone genuinely does not do perception on this task at this budget, or
  (b) something in OUR pipeline -- the wrapper, the collator path, the new clips, the
      state-derived labels -- broke a model that otherwise works.

Phase A measured Robometer-4B on PickCube with uniform 32-frame full-episode clips and
got S_view = +0.619 (best=side, worst=topdown), P_obj = +0.448, stuck |tau| = +0.646,
reversed = -0.396.  Those clips still exist.  Re-scoring them through
``PoseConditionedRobometer`` (pose pathway switched off, which is byte-identical to
plain Robometer) separates the two: if the numbers come back, the code path is fine and
the new dataset is what changed; if they do not, the wrapper is the bug and every
comparison built on it is void.

tau here is against FRAME INDEX, which is Robometer's own progress convention
(``absolute_first_frame``: progress = (i - start) / (N - start - 1)).  That is the
target it was trained on, so this is the most favourable possible reading of it.
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

from anyviewmeter.models.robometer_backbone import (PoseConditionedRobometer,  # noqa: E402
                                                    load_robometer)

# phase-A reference, from outputs/G1_signoff.md
REF = dict(S_view=0.619, best="side", worst="topdown", P_obj=0.448,
           stuck=0.646, reversed=-0.396, shuffled=0.208)
CAMS = ["frontal_good", "side", "topdown", "occluded", "far"]


def load_clip(path):
    z = np.load(path, allow_pickle=False)
    return z["frames"], json.loads(str(z["meta"]))


def kendall_tau(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 2:
        return 0.0
    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    return float(np.triu(dx * dy, 1).sum() / (0.5 * n * (n - 1)))


def is_abstention(p) -> bool:
    """Phase-A rule: a constant score is a DECLINED answer, never tau = 0."""
    return len(np.unique(np.round(np.asarray(p), 6))) <= 1


def perturb(frames, kind, rng):
    if kind == "stuck":
        return np.repeat(frames[:1], len(frames), axis=0)
    if kind == "reversed":
        return frames[::-1].copy()
    if kind == "shuffled":
        return frames[rng.permutation(len(frames))]
    return frames


def boot_ci(v, n=10000, seed=20260728):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_ci_paired(a, b, n=10000, seed=20260728):
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) < 2:
        return (float(d.mean()) if len(d) else float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="/var/tmp/avm/data/clips_phaseA/PickCube")
    ap.add_argument("--n-traj", type=int, default=30)
    ap.add_argument("--out", default="/var/tmp/avm/runs/reproduce")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda")

    trajs = sorted({f.split("__")[0] for f in os.listdir(f"{a.clips}/success")
                    if f.endswith(".npz")}, key=lambda t: int(t.split("_")[1]))[:a.n_traj]
    print(f"reproducing on {len(trajs)} PickCube trajectories x {len(CAMS)} cameras\n")

    bundle = load_robometer(device=device)
    # identical object to the experiments; clear_pose() is the no-conditioning path
    model = PoseConditionedRobometer(bundle, injector_name="cross_attn",
                                     layers=[27, 31, 35], patch_inject=False,
                                     pose_dropout=0.0, gate_init=0.0)
    model.injector.to(device)
    model.rbm.eval()

    def score(path, kind=None, rng=None):
        frames, meta = load_clip(path)
        if kind:
            frames = perturb(frames, kind, rng)
        inp = model.build_inputs(frames, meta["prompt"], device)
        model.clear_pose()
        with torch.no_grad():
            p = model(inp)
        return p.float().cpu().numpy()

    per_cam = defaultdict(dict)
    abst = defaultdict(list)
    succ, fail = {}, {}
    temporal = defaultdict(list)
    temporal_abst = defaultdict(list)

    for i, t in enumerate(trajs):
        for cam in CAMS:
            p = f"{a.clips}/success/{t}__{cam}.npz"
            if not os.path.exists(p):
                continue
            pr = score(p)
            abst[cam].append(is_abstention(pr))
            if not is_abstention(pr):
                per_cam[cam][t] = kendall_tau(np.arange(len(pr)), pr)
        # P_obj at the phase-A reference camera
        ps, pf = f"{a.clips}/success/{t}__frontal_good.npz", f"{a.clips}/failure/{t}__frontal_good.npz"
        if os.path.exists(ps) and os.path.exists(pf):
            fs, _ = load_clip(ps)
            ff, _ = load_clip(pf)
            if not np.array_equal(fs, ff):       # object never rendered -> undefined
                a_ = score(ps)
                b_ = score(pf)
                if not is_abstention(a_):
                    succ[t] = kendall_tau(np.arange(len(a_)), a_)
                if not is_abstention(b_):
                    fail[t] = kendall_tau(np.arange(len(b_)), b_)
        for k in ("stuck", "reversed", "shuffled"):
            pr = score(ps, k, np.random.default_rng(abs(hash((t, k))) % 2**32))
            temporal_abst[k].append(is_abstention(pr))
            if not is_abstention(pr):
                temporal[k].append(kendall_tau(np.arange(len(pr)), pr))
        print(f"  {i+1}/{len(trajs)}", end="\r", flush=True)

    print("\n" + "=" * 78)
    print("Robometer-4B zero-shot, PickCube, phase-A clips, through the AVM code path")
    print("=" * 78)
    print(f"\n{'camera':<14}{'mean tau':>10}{'95% CI':>22}{'abstain':>9}{'n':>5}")
    means = {}
    for c in CAMS:
        if not per_cam[c]:
            print(f"{c:<14}{'ALL ABSTAINED':>10}")
            continue
        v = list(per_cam[c].values())
        m, lo, hi = boot_ci(v)
        means[c] = m
        print(f"{c:<14}{m:10.3f}   [{lo:+.3f}, {hi:+.3f}]{np.mean(abst[c])*100:8.0f}%{len(v):5d}")

    if len(means) >= 2:
        best = max(means, key=lambda k: means[k])
        worst = min(means, key=lambda k: means[k])
        shared = sorted(set(per_cam[best]) & set(per_cam[worst]))
        m, lo, hi = boot_ci_paired([per_cam[best][t] for t in shared],
                                   [per_cam[worst][t] for t in shared])
        print(f"\nS_view  {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  best={best} worst={worst} n={len(shared)}")
        print(f"  phase-A reference: {REF['S_view']:+.3f}  best={REF['best']} worst={REF['worst']}")

    shared = sorted(set(succ) & set(fail))
    if len(shared) >= 2:
        m, lo, hi = boot_ci_paired([succ[t] for t in shared], [fail[t] for t in shared])
        print(f"\nP_obj   {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  n={len(shared)}")
        print(f"  phase-A reference: {REF['P_obj']:+.3f}")

    print(f"\n{'perturbation':<14}{'mean tau':>10}{'95% CI':>22}{'abstain':>9}  phase-A")
    for k in ("stuck", "reversed", "shuffled"):
        if temporal[k]:
            m, lo, hi = boot_ci(temporal[k])
            print(f"{k:<14}{m:10.3f}   [{lo:+.3f}, {hi:+.3f}]"
                  f"{np.mean(temporal_abst[k])*100:8.0f}%   {REF[k]:+.3f}")
        else:
            print(f"{k:<14}{'ALL ABSTAINED':>10}{'':>22}{100:8.0f}%   {REF[k]:+.3f}")

    json.dump(dict(per_cam={c: per_cam[c] for c in per_cam}, succ=succ, fail=fail,
                   temporal={k: temporal[k] for k in temporal},
                   abstention={c: float(np.mean(abst[c])) for c in abst}),
              open(f"{a.out}/reproduce.json", "w"), indent=2, default=float)
    print(f"\nwrote {a.out}/reproduce.json")


if __name__ == "__main__":
    main()
